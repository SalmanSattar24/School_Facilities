from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import rasterio
from jsonschema import Draft202012Validator
from rasterio.warp import transform as transform_coordinates

from .campus import detail_extent_plan
from .schema import read_csv
from .vlm import (
    InteractionCreator,
    RequestLedger,
    VLMConfigurationError,
    VLMError,
    VLMResponseError,
    _atomic_json,
    _gemini_response_schema,
    _read_object,
    _request_fingerprint,
    _response_value,
    _sanitized_provider_diagnostics,
    _serializable,
    _sha256,
    _validate_image,
)


@dataclass(frozen=True)
class BoundaryVLMInput:
    school_id: str
    school_name: str
    context_path: Path
    context_geotiff_path: Path
    source: str
    capture_vintage: str
    metres_per_pixel: float
    requested_coordinate: tuple[float, float]
    public_match_name: str | None
    public_match_source: str | None
    public_match_coordinate: tuple[float, float] | None
    resolver_reason: str


def load_boundary_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = root.resolve()
    config = _read_object(root / "config" / "boundary_vlm.json")
    schema = _read_object(root / config["generation"]["response_schema_path"])
    prompt = (root / config["generation"]["prompt_path"]).read_text(encoding="utf-8")
    return config, schema, prompt


def load_boundary_input(root: Path, school_id: str) -> BoundaryVLMInput:
    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", school_id):
        raise VLMConfigurationError(f"unsafe school_id: {school_id!r}")
    schools = [row for row in read_csv(root / "schools_sample.csv") if row["school_id"] == school_id]
    if len(schools) != 1:
        raise VLMConfigurationError(f"school_id not found exactly once: {school_id}")
    school = schools[0]
    resolution = _read_object(root / "data" / "campus_resolutions" / f"{school_id}.json")
    if resolution.get("school_id") != school_id:
        raise VLMConfigurationError("campus resolution school_id mismatch")
    if resolution.get("requires_human_review") is not True:
        raise VLMConfigurationError("boundary fallback is only for unresolved review cases")
    if resolution.get("input_pair_frozen"):
        raise VLMConfigurationError("boundary fallback cannot replace frozen campus inputs")

    config, _, _ = load_boundary_bundle(root)
    image_dir = root / "data" / "imagery" / school_id
    context_path = image_dir / "context.jpg"
    context_geotiff = image_dir / "context.tif"
    _validate_image(
        context_path,
        root / "data" / "imagery",
        list(config["inputs"]["image_dimensions"]),
    )
    if not context_geotiff.is_file():
        raise VLMConfigurationError("boundary fallback requires the context GeoTIFF")
    sidecar = _read_object(image_dir / "context.json")
    if sidecar.get("school_id") != school_id or sidecar.get("product") != "context":
        raise VLMConfigurationError("context provenance does not match the school/product")
    if float(sidecar.get("target_coverage_fraction", 0)) < 0.995:
        raise VLMConfigurationError("context imagery coverage is below 99.5 percent")
    requested = resolution.get("requested_ccd_coordinate")
    resolved = resolution.get("resolved_center")
    try:
        requested_coordinate = (float(requested["latitude"]), float(requested["longitude"]))
        public_match_coordinate = (
            (float(resolved["latitude"]), float(resolved["longitude"]))
            if resolution.get("matched_name")
            else None
        )
    except (KeyError, TypeError, ValueError) as error:
        raise VLMConfigurationError(f"invalid campus resolver coordinates: {error}") from error
    return BoundaryVLMInput(
        school_id=school_id,
        school_name=school["school_name"],
        context_path=context_path,
        context_geotiff_path=context_geotiff,
        source=str(sidecar.get("source", "")),
        capture_vintage=str(sidecar.get("capture_datetime_or_vintage", "")),
        metres_per_pixel=float(sidecar.get("output_resolution_m", 0)),
        requested_coordinate=requested_coordinate,
        public_match_name=(
            str(resolution["matched_name"])
            if isinstance(resolution.get("matched_name"), str)
            else None
        ),
        public_match_source=(
            str(resolution["source_element"])
            if isinstance(resolution.get("source_element"), str)
            else None
        ),
        public_match_coordinate=public_match_coordinate,
        resolver_reason=str(resolution.get("reason", "")),
    )


def build_boundary_request(root: Path, school: BoundaryVLMInput) -> dict[str, Any]:
    config, schema, prompt = load_boundary_bundle(root)
    if not school.school_name.strip() or not school.source.strip() or not school.capture_vintage.strip():
        raise VLMConfigurationError("boundary request requires school and imagery provenance")
    if school.metres_per_pixel <= 0:
        raise VLMConfigurationError("context metres_per_pixel must be positive")
    metadata = {
        "school_id": school.school_id,
        "school_name": school.school_name,
        "requested_ccd_coordinate": {
            "latitude": school.requested_coordinate[0],
            "longitude": school.requested_coordinate[1],
        },
        "public_match": {
            "name": school.public_match_name,
            "source": school.public_match_source,
            "coordinate": (
                {
                    "latitude": school.public_match_coordinate[0],
                    "longitude": school.public_match_coordinate[1],
                }
                if school.public_match_coordinate
                else None
            ),
            "resolver_reason": school.resolver_reason,
        },
        "image": {
            "role": "context",
            "source": school.source,
            "capture_vintage": school.capture_vintage,
            "metres_per_pixel": school.metres_per_pixel,
            "coordinate_system": config["inputs"]["coordinate_system"],
        },
    }
    generation = config["generation"]
    return {
        "model": config["model"],
        "input": [
            {
                "type": "text",
                "text": "Propose the campus boundary for this school. Metadata:\n"
                + json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            },
            {
                "type": "image",
                "data": school.context_path.resolve(),
                "mime_type": "image/jpeg",
                "resolution": config["inputs"]["media_resolution"],
            },
        ],
        "system_instruction": prompt,
        "generation_config": {
            "thinking_level": generation["thinking_level"],
            "max_output_tokens": generation["max_output_tokens"],
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": _gemini_response_schema(schema),
        },
        "service_tier": "standard",
        "store": False,
        "stream": False,
    }


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    for index in range(len(polygon) - 1):
        x1, y1 = polygon[index]
        x2, y2 = polygon[index + 1]
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orientation(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> int:
        value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(value) < 1e-12:
            return 0
        return 1 if value > 0 else 2

    def on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def normalize_boundary_vocabulary(
    parsed: Any,
) -> tuple[Any, list[dict[str, str]]]:
    """Normalize a small whitelist of meaning-equivalent provider terms."""
    if not isinstance(parsed, dict):
        return parsed, []
    normalized = json.loads(json.dumps(parsed))
    changes: list[dict[str, str]] = []

    def replace(container: Any, key: str, mapping: dict[str, str], path: str) -> None:
        if not isinstance(container, dict):
            return
        current = container.get(key)
        replacement = mapping.get(current) if isinstance(current, str) else None
        if replacement is None or replacement == current:
            return
        container[key] = replacement
        changes.append({"path": path, "from": current, "to": replacement})

    replace(
        normalized,
        "schema_version",
        {"1.0": "1.0.0", "2.0": "1.0.0"},
        "schema_version",
    )
    replace(
        normalized,
        "campus_visibility",
        {"full": "complete", "fully_visible": "complete"},
        "campus_visibility",
    )
    region_types = {
        "building_complex": "school_building",
        "school_complex": "school_building",
        "school_campus": "other",
        "campus": "other",
        "buildings_and_parking": "school_building",
        "recreation_field": "athletic_ground",
        "athletic_field": "athletic_ground",
        "athletic_grounds": "athletic_ground",
        "agriculture": "other",
        "agricultural": "other",
        "residential_and_roads": "other",
        "road_residential": "other",
        "forest": "other",
        "woodland": "other",
        "residential_or_natural": "other",
        "public_road": "road",
        "golf_course": "other",
        "school_grounds": "other",
    }
    for collection in (
        "included_regions",
        "excluded_adjacent_regions",
        "shared_or_ambiguous_regions",
    ):
        regions = normalized.get(collection)
        if not isinstance(regions, list):
            continue
        for index, region in enumerate(regions):
            replace(
                region,
                "region_type",
                region_types,
                f"{collection}.{index}.region_type",
            )
    cue_type_mapping = {
        "roads": "road",
        "sidewalks": "sidewalk",
        "fences": "fence_or_wall",
        "walls": "fence_or_wall",
        "tree lines": "tree_line",
        "tree_lines": "tree_line",
        "treelines": "tree_line",
        "parking edges": "parking_edge",
        "parking_edges": "parking_edge",
        "property transitions": "land_cover_change",
        "property_transitions": "land_cover_change",
        "property_transition": "land_cover_change",
        "land cover changes": "land_cover_change",
        "building edges": "building_edge",
    }
    cues = normalized.get("boundary_cues")
    if isinstance(cues, dict):
        for side, cue in cues.items():
            replace(
                cue,
                "visibility",
                {
                    "visible": "clear",
                    "not_visible": "unknown",
                    "partial": "weak",
                },
                f"boundary_cues.{side}.visibility",
            )
            if not isinstance(cue, dict) or not isinstance(cue.get("cue_types"), list):
                continue
            for index, value in enumerate(cue["cue_types"]):
                replacement = cue_type_mapping.get(value) if isinstance(value, str) else None
                if replacement is not None and replacement != value:
                    cue["cue_types"][index] = replacement
                    changes.append(
                        {
                            "path": f"boundary_cues.{side}.cue_types.{index}",
                            "from": value,
                            "to": replacement,
                        }
                    )
    polygon = normalized.get("boundary_polygon_normalized")
    if (
        isinstance(polygon, list)
        and len(polygon) >= 3
        and isinstance(polygon[0], dict)
        and polygon[0] != polygon[-1]
    ):
        polygon.append(json.loads(json.dumps(polygon[0])))
        changes.append(
            {
                "path": "boundary_polygon_normalized",
                "from": "open vertex sequence",
                "to": "closed by repeating the first vertex",
            }
        )
    return normalized, changes


def validate_boundary_output(
    parsed: dict[str, Any],
    schema: dict[str, Any],
    school_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    errors = sorted(Draft202012Validator(schema).iter_errors(parsed), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.path) or "<root>"
        raise VLMResponseError(f"boundary output failed schema validation at {location}: {first.message}")
    if parsed.get("school_id") != school_id:
        raise VLMResponseError("boundary output school_id mismatch")

    visibility = parsed["campus_visibility"]
    raw_polygon = parsed["boundary_polygon_normalized"]
    if visibility == "not_locatable":
        if raw_polygon:
            raise VLMResponseError("not_locatable output must use an empty boundary polygon")
        return {
            "candidate_quality_gate_passed": False,
            "guarded_review_required": True,
            "guard_reasons": ["model could not locate the sampled school"],
            "area_fraction": 0.0,
        }

    guards = config["geometry_guards"]
    points = [(float(point["x"]), float(point["y"])) for point in raw_polygon]
    if len(points) < 4 or points[0] != points[-1]:
        raise VLMResponseError("boundary polygon must be explicitly closed")
    unique = set(points[:-1])
    if len(unique) < int(guards["minimum_unique_vertices"]):
        raise VLMResponseError("boundary polygon has too few unique vertices")
    if len(unique) > int(guards["maximum_vertices"]):
        raise VLMResponseError("boundary polygon has too many unique vertices")
    for first in range(len(points) - 1):
        for second in range(first + 1, len(points) - 1):
            if abs(first - second) <= 1 or (first == 0 and second == len(points) - 2):
                continue
            if _segments_intersect(
                points[first], points[first + 1], points[second], points[second + 1]
            ):
                raise VLMResponseError("boundary polygon self-intersects")
    area = abs(
        sum(
            points[index][0] * points[index + 1][1]
            - points[index + 1][0] * points[index][1]
            for index in range(len(points) - 1)
        )
    ) / 2
    if not (float(guards["minimum_area_fraction"]) <= area <= float(guards["maximum_area_fraction"])):
        raise VLMResponseError(f"boundary polygon area fraction {area:.4f} is outside guard limits")
    anchor = (float(parsed["school_anchor"]["x"]), float(parsed["school_anchor"]["y"]))
    if guards["require_school_anchor_inside_polygon"] and not _point_in_polygon(anchor, points):
        raise VLMResponseError("school anchor lies outside the proposed boundary")
    for collection in (
        "included_regions",
        "excluded_adjacent_regions",
        "shared_or_ambiguous_regions",
    ):
        for region in parsed[collection]:
            bbox = region["bbox"]
            if not (bbox["x_min"] < bbox["x_max"] and bbox["y_min"] < bbox["y_max"]):
                raise VLMResponseError(f"{collection} contains an inverted or empty bbox")

    reasons: list[str] = []
    if visibility != "complete":
        reasons.append("model reported partial campus visibility")
    if float(parsed["suggested_confidence"]) <= float(
        config["decision_policy"]["confidence_at_or_below_requires_review"]
    ):
        reasons.append("model boundary confidence is at or below the review threshold")
    if parsed["review_required"]:
        reasons.append("model requested boundary review")
    if parsed["shared_or_ambiguous_regions"]:
        reasons.append("model identified shared or ownership-ambiguous regions")
    for region in parsed["excluded_adjacent_regions"]:
        bbox = region["bbox"]
        if bbox["x_min"] <= anchor[0] <= bbox["x_max"] and bbox["y_min"] <= anchor[1] <= bbox["y_max"]:
            reasons.append("an excluded-region box contains the school anchor")
            break
    for region in parsed["included_regions"]:
        bbox = region["bbox"]
        center = ((bbox["x_min"] + bbox["x_max"]) / 2, (bbox["y_min"] + bbox["y_max"]) / 2)
        if not _point_in_polygon(center, points):
            reasons.append("an included-region center lies outside the proposed boundary")
            break
    if visibility == "complete":
        clearance = min(min(x, y, 1 - x, 1 - y) for x, y in points[:-1])
        if clearance < float(guards["complete_visibility_minimum_edge_clearance_fraction"]):
            reasons.append("complete boundary is too close to the context-image edge")
    candidate_passed = not reasons
    if not config["decision_policy"]["auto_confirmation_enabled"]:
        reasons.append("V1.0 Gemini boundary proposals require human approval pending validation")
    return {
        "candidate_quality_gate_passed": candidate_passed,
        "guarded_review_required": bool(reasons),
        "guard_reasons": reasons,
        "area_fraction": round(area, 6),
    }


def _polygon_to_wgs84(
    context_geotiff: Path, polygon: list[dict[str, float]]
) -> tuple[list[list[float]], list[float]]:
    with rasterio.open(context_geotiff) as dataset:
        if dataset.crs is None:
            raise VLMConfigurationError("context GeoTIFF has no CRS")
        projected = [
            dataset.transform @ (float(point["x"]) * dataset.width, float(point["y"]) * dataset.height)
            for point in polygon
        ]
        longitudes, latitudes = transform_coordinates(
            dataset.crs, "EPSG:4326", [point[0] for point in projected], [point[1] for point in projected]
        )
    geometry = [[round(lat, 8), round(lon, 8)] for lon, lat in zip(longitudes, latitudes)]
    bbox = [min(longitudes), min(latitudes), max(longitudes), max(latitudes)]
    return geometry, [round(value, 8) for value in bbox]


class GeminiBoundaryClient:
    def __init__(
        self,
        root: Path,
        create_interaction: InteractionCreator,
        *,
        close_client: Callable[[], None] | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = root.resolve()
        self.config, self.schema, _ = load_boundary_bundle(self.root)
        self.create_interaction = create_interaction
        self.close_client = close_client
        self.ledger = RequestLedger(
            self.root / self.config["outputs"]["request_ledger"],
            self.config,
            now=now,
            sleep=sleep,
        )

    @classmethod
    def from_environment(cls, root: Path) -> GeminiBoundaryClient:
        config, _, _ = load_boundary_bundle(root)
        from .credentials import CredentialError, load_api_key

        try:
            api_key, _ = load_api_key(
                environment_variable=config["credentials"]["api_key_environment_variable"],
                secrets_file=root / config["credentials"]["local_secrets_file"],
            )
        except CredentialError as error:
            raise VLMConfigurationError(str(error)) from error
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version=config["api_version"],
                timeout=int(config["request_limits"]["request_timeout_seconds"]) * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        return cls(root, client.interactions.create, close_client=client.close)

    def close(self) -> None:
        if self.close_client:
            self.close_client()

    def propose(self, school: BoundaryVLMInput) -> Path:
        request = build_boundary_request(self.root, school)
        output_path = self.root / self.config["outputs"]["raw_directory"] / f"{school.school_id}.json"
        if output_path.exists():
            raise VLMConfigurationError(f"boundary proposal already exists: {output_path}")
        self.ledger.reserve(school.school_id, "production", 1)
        try:
            response = self.create_interaction(
                **request,
                timeout=float(self.config["request_limits"]["request_timeout_seconds"]),
            )
        except Exception as error:
            status = getattr(error, "status_code", None)
            raise VLMError(f"Gemini boundary request failed with status {status or 'unknown'}") from error

        status = _response_value(response, "status")
        output_text = _response_value(response, "output_text")
        base_record = {
            "schema_version": "1.0.0",
            "configuration_id": self.config["configuration_id"],
            "school_id": school.school_id,
            "model": self.config["model"],
            "interaction_id": _response_value(response, "id"),
            "status": status,
            "created": _response_value(response, "created"),
            "usage": _serializable(_response_value(response, "usage")),
            "request_fingerprint_sha256": _request_fingerprint(request),
            "input_sha256": {
                "context.jpg": _sha256(school.context_path),
                "context.tif": _sha256(school.context_geotiff_path),
            },
            "output_text": output_text if isinstance(output_text, str) else None,
        }
        if status != "completed" or not isinstance(output_text, str):
            base_record["provider_diagnostics"] = _sanitized_provider_diagnostics(response)
            base_record["validation_error"] = "interaction did not return completed output text"
            rejected = self.root / self.config["outputs"]["rejected_directory"] / f"{school.school_id}.json"
            _atomic_json(rejected, base_record)
            raise VLMResponseError(f"boundary response rejected and preserved at {rejected}")
        try:
            parsed = json.loads(output_text)
            parsed, vocabulary_normalizations = normalize_boundary_vocabulary(parsed)
            guard = validate_boundary_output(parsed, self.schema, school.school_id, self.config)
            geometry: list[list[float]] = []
            bbox: list[float] | None = None
            extent = 600
            unclamped = 600
            clipped = False
            if parsed["boundary_polygon_normalized"]:
                geometry, bbox = _polygon_to_wgs84(
                    school.context_geotiff_path, parsed["boundary_polygon_normalized"]
                )
                extent, unclamped, clipped = detail_extent_plan(tuple(bbox))
        except (json.JSONDecodeError, VLMResponseError, VLMConfigurationError, rasterio.errors.RasterioError) as error:
            base_record["parsed_output"] = locals().get("parsed")
            base_record["validation_error"] = str(error)
            rejected = self.root / self.config["outputs"]["rejected_directory"] / f"{school.school_id}.json"
            _atomic_json(rejected, base_record)
            raise VLMResponseError(f"boundary response rejected and preserved at {rejected}: {error}") from error

        base_record.update(
            {
                "parsed_output": parsed,
                "provider_vocabulary_normalizations": vocabulary_normalizations,
                "deterministic_guard": guard,
                "proposal_status": "ready_for_human_review",
                "geometry_wgs84_lat_lon": geometry,
                "bbox_wgs84": bbox,
                "recommended_detail_extent_m": extent,
                "unclamped_detail_extent_m": unclamped,
                "detail_extent_clipped_at_maximum": clipped,
                "campus_resolution_was_not_overwritten": True,
            }
        )
        _atomic_json(output_path, base_record)
        return output_path


def reconcile_rejected_boundary(root: Path, school_id: str) -> Path:
    """Revalidate a preserved boundary response after whitelist normalization."""
    root = root.resolve()
    config, schema, _ = load_boundary_bundle(root)
    school = load_boundary_input(root, school_id)
    rejected_path = root / config["outputs"]["rejected_directory"] / f"{school_id}.json"
    rejected = _read_object(rejected_path)
    parsed, vocabulary_normalizations = normalize_boundary_vocabulary(
        rejected.get("parsed_output")
    )
    if not isinstance(parsed, dict):
        raise VLMResponseError("rejected boundary record has no parsed object")
    guard = validate_boundary_output(parsed, schema, school_id, config)
    geometry: list[list[float]] = []
    bbox: list[float] | None = None
    extent = 600
    unclamped = 600
    clipped = False
    if parsed["boundary_polygon_normalized"]:
        geometry, bbox = _polygon_to_wgs84(
            school.context_geotiff_path, parsed["boundary_polygon_normalized"]
        )
        extent, unclamped, clipped = detail_extent_plan(tuple(bbox))
    output_path = root / config["outputs"]["raw_directory"] / f"{school_id}.json"
    if output_path.exists():
        raise VLMConfigurationError(f"boundary proposal already exists: {output_path}")
    reconciled = {
        key: value
        for key, value in rejected.items()
        if key not in {"validation_error", "provider_diagnostics"}
    }
    reconciled.update(
        {
            "parsed_output": parsed,
            "provider_vocabulary_normalizations": vocabulary_normalizations,
            "deterministic_guard": guard,
            "proposal_status": "ready_for_human_review",
            "geometry_wgs84_lat_lon": geometry,
            "bbox_wgs84": bbox,
            "recommended_detail_extent_m": extent,
            "unclamped_detail_extent_m": unclamped,
            "detail_extent_clipped_at_maximum": clipped,
            "campus_resolution_was_not_overwritten": True,
            "reconciled_from_rejected_path": str(rejected_path.relative_to(root)),
        }
    )
    _atomic_json(output_path, reconciled)
    return output_path
