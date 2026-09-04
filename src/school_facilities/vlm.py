from __future__ import annotations

import json
import hashlib
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from jsonschema import Draft202012Validator
from PIL import Image
import rasterio
from rasterio.warp import transform as transform_coordinates

from .configuration import validate_configuration
from .campus import extent_plan_for_scope


RunMode = Literal["pilot", "production", "auditor"]
VLMProfile = Literal["final"]


class VLMError(RuntimeError):
    """Base error for the frozen Gemini workflow."""


class VLMConfigurationError(VLMError):
    """Raised when the frozen configuration or a request input is invalid."""


class VLMQuotaError(VLMError):
    """Raised before a request that would exceed a frozen request cap."""


class VLMResponseError(VLMError):
    """Raised when Gemini does not return a completed, schema-valid response."""


@dataclass(frozen=True)
class VLMImage:
    path: Path
    source: str
    capture_vintage: str
    metres_per_pixel: float
    role: str = ""


@dataclass(frozen=True)
class SchoolVLMInput:
    school_id: str
    school_name: str
    campus_resolution_notes: str
    context: VLMImage
    detail: VLMImage
    public_source_and_non_sensitive_confirmed: bool
    facility_crops: tuple[VLMImage, ...] = ()
    campus_boundary_detail_normalized: tuple[tuple[float, float], ...] = ()
    campus_boundary_source: str | None = None
    campus_scope_mode: str = "authoritative_polygon"
    scope_boundary_authority: str = "authoritative"
    measurement_search_scope: str = "inside_authoritative_polygon"


class InteractionCreator(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise VLMConfigurationError(f"missing required file: {path}") from error
    except json.JSONDecodeError as error:
        raise VLMConfigurationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise VLMConfigurationError(f"JSON root must be an object: {path}")
    return value


def _response_output_directories(
    root: Path,
    config: dict[str, Any],
    school_id: str,
) -> tuple[Path, Path, bool]:
    """Route frozen validation responses away from ordinary review outputs.

    This changes storage only. It deliberately does not modify the request,
    configuration identifier, prompt, model, or response schema.
    """
    pilot = _read_object(root / "config" / "pilot_schools.json")
    validation_ids = set(pilot.get("excluded_validation_school_ids", []))
    if school_id not in validation_ids:
        return (
            root / config["outputs"]["raw_directory"],
            root / config["outputs"]["rejected_directory"],
            False,
        )
    version = Path(config["outputs"]["raw_directory"]).name
    quarantine = root / "data" / "model_outputs" / "quarantine" / version
    return quarantine / "raw", quarantine / "rejected", True


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_image(path: Path, imagery_root: Path, expected_pixels: list[int]) -> None:
    if not _within(path, imagery_root):
        raise VLMConfigurationError(f"VLM image must be inside {imagery_root}: {path}")
    if not path.is_file():
        raise VLMConfigurationError(f"VLM image does not exist: {path}")
    try:
        with Image.open(path) as image:
            image_format = image.format
            dimensions = list(image.size)
            image.verify()
    except Exception as error:
        raise VLMConfigurationError(f"VLM image is unreadable: {path}: {error}") from error
    if image_format != "JPEG":
        raise VLMConfigurationError(f"VLM image must be JPEG: {path}")
    if dimensions != expected_pixels:
        raise VLMConfigurationError(
            f"VLM image must be {expected_pixels[0]}x{expected_pixels[1]} pixels: "
            f"{path} is {dimensions[0]}x{dimensions[1]}"
        )


def _normalized_detail_boundary(
    image_dir: Path,
    review: dict[str, Any],
) -> tuple[tuple[float, float], ...]:
    geometry = review.get("geometry")
    if not isinstance(geometry, list) or len(geometry) < 3:
        return ()
    try:
        latitudes = [float(point[0]) for point in geometry]
        longitudes = [float(point[1]) for point in geometry]
    except (IndexError, TypeError, ValueError) as error:
        raise VLMConfigurationError("campus boundary geometry is invalid") from error
    detail_geotiff = image_dir / "detail.tif"
    if not detail_geotiff.is_file():
        raise VLMConfigurationError(
            "automatic campus boundary metadata requires the approved detail GeoTIFF"
        )
    try:
        with rasterio.open(detail_geotiff) as dataset:
            if dataset.crs is None:
                raise VLMConfigurationError("approved detail GeoTIFF has no CRS")
            xs, ys = transform_coordinates(
                "EPSG:4326", dataset.crs, longitudes, latitudes
            )
            inverse = ~dataset.transform
            normalized_points = []
            for x, y in zip(xs, ys):
                column, row = inverse @ (x, y)
                normalized_points.append(
                    (float(column) / dataset.width, float(row) / dataset.height)
                )
            normalized = tuple(normalized_points)
    except rasterio.errors.RasterioError as error:
        raise VLMConfigurationError(f"cannot read approved detail GeoTIFF: {error}") from error
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in normalized):
        raise VLMConfigurationError(
            "resolved campus boundary is outside the approved adaptive detail image"
        )
    return tuple((round(x, 6), round(y, 6)) for x, y in normalized)


def load_vlm_bundle(
    root: Path,
    profile: VLMProfile = "final",
) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = root.resolve()
    result = validate_configuration(root)
    if not result.ok:
        raise VLMConfigurationError("; ".join(result.errors))
    if profile != "final":
        raise VLMConfigurationError(f"unknown VLM profile: {profile!r}")
    config = _read_object(root / "config" / "vlm.json")
    schema_relative = config["generation"]["response_format"]["schema_path"]
    schema_path = (root / str(schema_relative)).resolve()
    if not _within(schema_path, root / "config"):
        raise VLMConfigurationError("VLM response schema must remain inside config")
    schema = _read_object(schema_path)
    prompt = (root / "config" / "vlm_prompt.txt").read_text(encoding="utf-8")
    protocol_relative = config["evidence_policy"]["field_protocol_path"]
    protocol_path = (root / str(protocol_relative)).resolve()
    if not _within(protocol_path, root / "config"):
        raise VLMConfigurationError("VLM field protocol must remain inside config")
    field_protocol = _read_object(protocol_path)
    prompt += (
        "\n\nAUTHORITATIVE FIELD PROTOCOL (machine-readable frozen contract):\n"
        + json.dumps(field_protocol, ensure_ascii=False, sort_keys=True)
    )
    return config, schema, prompt


def load_approved_school_input(
    root: Path,
    school_id: str,
    profile: VLMProfile = "final",
) -> SchoolVLMInput:
    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", school_id):
        raise VLMConfigurationError(f"unsafe school_id: {school_id!r}")
    automatic_path = root / "data" / "campus_resolutions" / f"{school_id}.json"
    manual_path = root / "data" / "reviewed" / school_id / "campus.json"
    review_path = automatic_path if automatic_path.is_file() else manual_path
    review = _read_object(review_path)
    if review.get("school_id") != school_id:
        raise VLMConfigurationError("campus review school_id does not match the requested school")
    status = review.get("status", review.get("review_status"))
    if status not in {"confirmed", "probable"}:
        raise VLMConfigurationError("campus review must be confirmed or probable")
    pair_frozen = review.get("input_pair_frozen", review.get("image_pair_approved"))
    if pair_frozen is not True:
        raise VLMConfigurationError("the context/detail image pair has not been approved")
    if review.get("public_source_and_non_sensitive_confirmed") is not True:
        raise VLMConfigurationError("public-source and non-sensitive confirmation is missing")

    image_dir = root / "data" / "imagery" / school_id
    expected_relative = [
        f"data/imagery/{school_id}/context.jpg",
        f"data/imagery/{school_id}/detail.jpg",
    ]
    if review.get("approved_gemini_inputs") != expected_relative:
        raise VLMConfigurationError("approved Gemini input paths do not match the frozen pair")
    hashes = review.get("approved_gemini_input_sha256", {})
    if not isinstance(hashes, dict):
        raise VLMConfigurationError("approved Gemini input hashes are missing")
    supporting_hashes = review.get("approved_supporting_artifact_sha256", {})
    detail_geotiff = image_dir / "detail.tif"
    if (
        not isinstance(supporting_hashes, dict)
        or not isinstance(supporting_hashes.get("detail.tif"), str)
        or _sha256(detail_geotiff) != supporting_hashes["detail.tif"]
    ):
        raise VLMConfigurationError("approved detail GeoTIFF hash does not match")

    sidecars: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for role in ("context", "detail"):
        path = image_dir / f"{role}.jpg"
        expected_hash = hashes.get(f"{role}.jpg")
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise VLMConfigurationError(f"approved {role} image hash does not match")
        sidecar = _read_object(image_dir / f"{role}.json")
        if sidecar.get("school_id") != school_id or sidecar.get("product") != role:
            raise VLMConfigurationError(f"{role} provenance does not match the approved school/product")
        if sidecar.get("target_coverage_fraction", 0) < 0.995:
            raise VLMConfigurationError(f"{role} imagery coverage is below the frozen threshold")
        sidecars[role] = sidecar
        paths[role] = path

    if sidecars["context"].get("capture_datetime_or_vintage") != sidecars["detail"].get(
        "capture_datetime_or_vintage"
    ):
        raise VLMConfigurationError("approved context/detail images have different capture vintages")
    imagery_config = _read_object(root / "config" / "imagery.json")
    scope_mode = str(review.get("scope_mode") or "authoritative_polygon")
    if scope_mode not in {"authoritative_polygon", "soft_boundary", "center_only"}:
        raise VLMConfigurationError(f"unsupported campus scope mode: {scope_mode!r}")
    bbox = review.get("bbox_wgs84")
    if not isinstance(bbox, list) or len(bbox) != 4:
        geometry = review.get("geometry")
        if isinstance(geometry, list) and len(geometry) >= 3:
            try:
                latitudes = [float(point[0]) for point in geometry]
                longitudes = [float(point[1]) for point in geometry]
                bbox = [
                    min(longitudes),
                    min(latitudes),
                    max(longitudes),
                    max(latitudes),
                ]
            except (IndexError, TypeError, ValueError) as error:
                raise VLMConfigurationError("campus boundary geometry is invalid") from error
        elif scope_mode != "center_only":
            raise VLMConfigurationError("campus record lacks a valid boundary bbox")
        else:
            bbox = None
    try:
        numeric_bbox = (
            tuple(float(value) for value in bbox)
            if isinstance(bbox, list) and len(bbox) == 4
            else None
        )
        selected_extent, _unclamped_extent, clipped = extent_plan_for_scope(
            scope_mode, numeric_bbox
        )
    except (TypeError, ValueError) as error:
        raise VLMConfigurationError("campus boundary extent is invalid") from error
    if clipped:
        raise VLMConfigurationError(
            "buffered campus exceeds the single-image detail limit and needs tiled review"
        )
    detail_provenance = sidecars["detail"]
    requested_extent = detail_provenance.get("requested_extent_m", {})
    compatible_imagery_ids = {
        imagery_config.get("configuration_id"),
        *imagery_config.get("compatible_configuration_ids", []),
    }
    if detail_provenance.get("configuration_id") not in compatible_imagery_ids:
        raise VLMConfigurationError(
            "approved detail image was not generated by the active adaptive imagery configuration"
        )
    if scope_mode == "soft_boundary" and detail_provenance.get(
        "configuration_id"
    ) != imagery_config.get("configuration_id"):
        raise VLMConfigurationError(
            "soft-scope detail image was not generated by imagery Version 1.6"
        )
    if not isinstance(requested_extent, dict) or any(
        not isinstance(requested_extent.get(axis), (int, float))
        or not abs(float(requested_extent[axis]) - selected_extent) <= 1e-9
        for axis in ("width", "height")
    ):
        raise VLMConfigurationError(
            "approved detail image extent does not match the resolved campus boundary"
        )
    school_name = review.get("school_name")
    notes = review.get("boundary_notes")
    if not isinstance(school_name, str) or not isinstance(notes, str):
        raise VLMConfigurationError("campus review is missing school name or boundary notes")

    def image(role: str) -> VLMImage:
        sidecar = sidecars[role]
        return VLMImage(
            path=paths[role],
            source=str(sidecar["source"]),
            capture_vintage=str(sidecar["capture_datetime_or_vintage"]),
            metres_per_pixel=float(sidecar["output_resolution_m"]),
            role=role,
        )

    return SchoolVLMInput(
        school_id=school_id,
        school_name=school_name,
        campus_resolution_notes=notes,
        context=image("context"),
        detail=image("detail"),
        public_source_and_non_sensitive_confirmed=True,
        facility_crops=(),
        campus_boundary_detail_normalized=_normalized_detail_boundary(image_dir, review),
        campus_boundary_source=(
            str(review.get("soft_boundary_proposal", {}).get("path"))
            if scope_mode == "soft_boundary"
            and isinstance(review.get("soft_boundary_proposal"), dict)
            else (
                str(review["source_element"])
                if isinstance(review.get("source_element"), str)
                else None
            )
        ),
        campus_scope_mode=scope_mode,
        scope_boundary_authority=str(
            review.get("scope_boundary_authority")
            or ("authoritative" if scope_mode == "authoritative_polygon" else "none")
        ),
        measurement_search_scope=str(
            review.get("measurement_search_scope")
            or (
                "inside_authoritative_polygon"
                if scope_mode == "authoritative_polygon"
                else "entire_detail_image"
            )
        ),
    )


def build_interaction_request(
    root: Path,
    school: SchoolVLMInput,
    profile: VLMProfile = "final",
) -> dict[str, Any]:
    root = root.resolve()
    config, schema, prompt = load_vlm_bundle(root, profile)
    if not school.public_source_and_non_sensitive_confirmed:
        raise VLMConfigurationError(
            "public-source and non-sensitive input confirmation is required before a VLM request"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", school.school_id):
        raise VLMConfigurationError(f"unsafe school_id: {school.school_id!r}")
    if not school.school_name.strip():
        raise VLMConfigurationError("school_name must not be blank")
    if not school.campus_resolution_notes.strip():
        raise VLMConfigurationError("campus_resolution_notes must not be blank")

    inputs = config["inputs"]
    allowed_scope_modes = {"authoritative_polygon", "soft_boundary", "center_only"}
    if school.campus_scope_mode not in allowed_scope_modes:
        raise VLMConfigurationError(
            f"unsupported campus scope mode: {school.campus_scope_mode!r}"
        )
    if school.campus_scope_mode == "authoritative_polygon" and not (
        school.campus_boundary_detail_normalized
    ):
        raise VLMConfigurationError(
            "authoritative-polygon scope requires a resolved campus boundary"
        )
    expected_search_scope = (
        "inside_authoritative_polygon"
        if school.campus_scope_mode == "authoritative_polygon"
        else "entire_detail_image"
    )
    if school.measurement_search_scope != expected_search_scope:
        raise VLMConfigurationError("campus scope mode and measurement search scope disagree")
    imagery_root = root / "data" / "imagery"
    expected_pixels = list(inputs["image_dimensions"])
    all_images = (school.context, school.detail, *school.facility_crops)
    if len(all_images) != int(inputs["images_per_school"]):
        raise VLMConfigurationError(
            f"{profile} profile requires {inputs['images_per_school']} images, got {len(all_images)}"
        )
    expected_order = list(inputs["image_order"])
    actual_order = [image.role for image in all_images]
    if actual_order != expected_order:
        raise VLMConfigurationError(
            f"{profile} image order must be {expected_order}, got {actual_order}"
        )
    for image in all_images:
        _validate_image(image.path, imagery_root, expected_pixels)
        if not image.source.strip() or not image.capture_vintage.strip():
            raise VLMConfigurationError("each image requires source and capture-vintage metadata")
        if image.metres_per_pixel <= 0:
            raise VLMConfigurationError("metres_per_pixel must be greater than zero")

    metadata = {
        "school_id": school.school_id,
        "school_name": school.school_name,
        "campus_resolution_notes": school.campus_resolution_notes,
        "campus_scope": {
            "mode": school.campus_scope_mode,
            "boundary_authority": school.scope_boundary_authority,
            "measurement_search_scope": school.measurement_search_scope,
            "fencing_and_ownership_require_review_when_boundary_is_not_authoritative": True,
        },
        "images": [
            {
                "role": image.role,
                "source": image.source,
                "capture_vintage": image.capture_vintage,
                "metres_per_pixel": image.metres_per_pixel,
            }
            for image in all_images
        ],
    }
    if school.campus_boundary_detail_normalized:
        if len(school.campus_boundary_detail_normalized) < 3:
            raise VLMConfigurationError("normalized campus boundary needs at least three points")
        polygon = []
        for x, y in school.campus_boundary_detail_normalized:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise VLMConfigurationError("normalized campus boundary coordinates must be in [0,1]")
            polygon.append({"x": round(float(x), 6), "y": round(float(y), 6)})
        boundary_record = {
            "image_role": "detail",
            "coordinate_system": inputs["campus_boundary_coordinate_system"],
            "polygon": polygon,
            "source": school.campus_boundary_source or "approved campus-resolution record",
        }
        if school.campus_scope_mode == "authoritative_polygon":
            metadata["resolved_campus_boundary"] = boundary_record
        else:
            metadata["non_binding_boundary_guidance"] = {
                **boundary_record,
                "instruction": (
                    "Use only as a centering cue; inspect and measure across the entire detail image."
                ),
            }
    generation = config["generation"]
    resolution = inputs["per_image_media_resolution"]
    return {
        "model": config["model"],
        "input": [
            {
                "type": "text",
                "text": "Assess this school using the frozen protocol. Metadata:\n"
                + json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            },
            *[
                {
                    "type": "image",
                    "data": image.path.resolve(),
                    "mime_type": "image/jpeg",
                    "resolution": resolution,
                }
                for image in all_images
            ],
        ],
        "system_instruction": prompt,
        "generation_config": {
            "thinking_level": generation["thinking_level"],
            "max_output_tokens": generation["max_output_tokens"],
        },
        "response_format": {
            "type": generation["response_format"]["type"],
            "mime_type": generation["response_format"]["mime_type"],
            "schema": _gemini_response_schema(schema),
        },
        "service_tier": "standard",
        "store": False,
        "stream": False,
    }


def _request_fingerprint(request: dict[str, Any]) -> str:
    """Hash the exact provider-visible request while replacing image paths with content hashes."""
    normalized_input: list[dict[str, Any]] = []
    for item in request["input"]:
        if item["type"] == "image":
            normalized_input.append(
                {
                    "type": "image",
                    "mime_type": item["mime_type"],
                    "resolution": item["resolution"],
                    "sha256": _sha256(Path(item["data"])),
                }
            )
        else:
            normalized_input.append(item)
    provider_visible = {
        key: value
        for key, value in request.items()
        if key != "input"
    }
    provider_visible["input"] = normalized_input
    payload = json.dumps(
        provider_visible,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _response_value(response: Any, field: str) -> Any:
    if isinstance(response, dict):
        return response.get(field)
    return getattr(response, field, None)


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return str(value)


def _sanitized_provider_diagnostics(response: Any) -> dict[str, Any]:
    """Retain safe interaction diagnostics without copying prompts or images."""
    diagnostic_fields = {
        "status",
        "id",
        "created",
        "updated",
        "model",
        "usage",
        "errors",
        "error",
        "incomplete_details",
        "incomplete_reason",
        "failure_reason",
        "finish_reason",
        "status_details",
    }
    snapshot: dict[str, Any] = {}
    for field in diagnostic_fields:
        value = _response_value(response, field)
        if value is not None:
            snapshot[field] = _serializable(value)

    # The generated SDK permits extra response fields. Preserve any future
    # provider diagnostic whose name explicitly denotes an error, reason, or
    # detail, while excluding request inputs, instructions, and image content.
    if hasattr(response, "model_dump"):
        dumped = response.model_dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(dumped, dict):
            for field, value in dumped.items():
                normalized = str(field).lower()
                if field not in snapshot and any(
                    marker in normalized for marker in ("error", "reason", "detail")
                ):
                    snapshot[field] = value

    # Apply the same credential redaction used for provider exceptions while
    # preserving the structure for later diagnosis.
    serialized = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)
    serialized = re.sub(
        r"AIza[A-Za-z0-9_-]{20,}", "[REDACTED_API_KEY]", serialized
    )
    serialized = re.sub(
        r"(?i)(x-goog-api-key|api[_-]?key)(\s*[:=]\s*)[^\s,;\"'}]+",
        r"\1\2[REDACTED]",
        serialized,
    )
    return json.loads(serialized)


def _gemini_response_schema(value: Any) -> dict[str, Any]:
    """Return a reduced provider schema while retaining strict local validation.

    Gemini rejected the complete V1.8 and flattened V1.8.1 schemas before
    inference with ``400 invalid_request``. V1.8.2 proved that a top-level-only
    schema reaches inference, but it permitted empty nested objects. The prompt
    still specifies the full contract, and completed output is checked against
    the frozen local Draft 2020-12 schema. The provider projection below
    requires the complete property tree and basic types while omitting refs,
    unions, enums, bounds, fixed lengths, and additional-property constraints.
    """
    if not isinstance(value, dict) or value.get("type") != "object":
        raise VLMConfigurationError("provider response schema must have an object root")
    definitions = value.get("$defs")
    if not isinstance(definitions, dict):
        raise VLMConfigurationError("provider response schema requires local definitions")

    def project(node: Any) -> dict[str, Any]:
        if not isinstance(node, dict):
            raise VLMConfigurationError("response schema contains a non-object node")
        reference = node.get("$ref")
        if isinstance(reference, str):
            prefix = "#/$defs/"
            if not reference.startswith(prefix):
                raise VLMConfigurationError(f"unsupported local schema reference: {reference}")
            definition = definitions.get(reference[len(prefix) :])
            if not isinstance(definition, dict):
                raise VLMConfigurationError(f"missing local schema reference: {reference}")
            return project(definition)
        if "const" in node:
            constant = node["const"]
            if isinstance(constant, bool):
                return {"type": "boolean"}
            if isinstance(constant, int):
                return {"type": "integer"}
            if isinstance(constant, float):
                return {"type": "number"}
            if isinstance(constant, str):
                return {"type": "string"}
            return {}
        alternatives = node.get("anyOf") or node.get("oneOf")
        if isinstance(alternatives, list):
            # Measurement value unions are fully specified in the prompt and
            # enforced locally. An empty subschema permits either the numeric
            # or explicit-unknown representation without provider grammar cost.
            projected_alternatives = [project(item) for item in alternatives]
            types = {item.get("type") for item in projected_alternatives if item}
            if len(types) == 1:
                return {"type": types.pop()}
            return {}
        if isinstance(node.get("allOf"), list):
            projected_parts = [project(item) for item in node["allOf"]]
            if projected_parts and all(item.get("type") == "object" for item in projected_parts):
                merged: dict[str, Any] = {"type": "object", "properties": {}}
                required: list[str] = []
                for item in projected_parts:
                    merged["properties"].update(item.get("properties", {}))
                    for name in item.get("required", []):
                        if name not in required:
                            required.append(name)
                if required:
                    merged["required"] = required
                return merged
            return {}

        node_type = node.get("type")
        if node_type == "object":
            projected: dict[str, Any] = {"type": "object"}
            source_properties = node.get("properties")
            if isinstance(source_properties, dict):
                projected["properties"] = {
                    name: project(definition)
                    for name, definition in source_properties.items()
                }
                required = node.get("required")
                if isinstance(required, list):
                    projected["required"] = list(required)
            return projected
        if node_type == "array":
            return {"type": "array", "items": project(node.get("items", {}))}
        if node_type in {"string", "number", "integer", "boolean"}:
            return {"type": node_type}
        if not node:
            return {}
        raise VLMConfigurationError(f"unsupported local schema node type: {node_type!r}")

    return project(value)


def _normalize_provider_vocabulary(
    parsed: Any,
) -> tuple[Any, list[dict[str, str]]]:
    """Canonicalize only frozen, meaning-preserving provider synonyms.

    Raw ``output_text`` remains untouched. Each applied substitution is stored
    with the record so normalization is auditable and cannot silently alter a
    measurement value.
    """
    if not isinstance(parsed, dict):
        return parsed, []
    normalized = json.loads(json.dumps(parsed))
    changes: list[dict[str, str]] = []

    def replace(
        container: Any,
        key: str,
        mapping: dict[str, str],
        path: str,
    ) -> None:
        if not isinstance(container, dict):
            return
        current = container.get(key)
        replacement = mapping.get(current) if isinstance(current, str) else None
        if replacement is None or replacement == current:
            return
        container[key] = replacement
        changes.append({"path": path, "from": current, "to": replacement})

    replace(normalized, "schema_version", {"1.0": "1.10.0", "1.9.0": "1.10.0"}, "schema_version")
    campus = normalized.get("campus_assessment")
    replace(
        campus,
        "status",
        {
            "fully_visible": "confirmed",
            "fully_resolved": "confirmed",
            "partially_visible": "probable",
            "complete": "confirmed",
            "ambiguous_boundary": "probable",
            # Provider wording that explicitly says the campus could not be
            # verified must stay in the most conservative schema state.
            "unverified_or_ambiguous": "unresolved",
            "unverified_boundary_center_only": "unresolved",
            "assessed": "probable",
            "provisional_boundary_soft_guidance": "probable",
        },
        "campus_assessment.status",
    )
    solar = normalized.get("solar_inventory")
    replace(
        solar,
        "roof_visibility",
        {
            "clear": "adequate",
            "good": "adequate",
            "fully visible": "adequate",
            "fully_visible": "adequate",
            # A merely partial roof search is not adequate evidence for a
            # confident absence and must enter the conservative branch.
            "partial": "inadequate",
            "partially_clear": "inadequate",
            "partially_obscured_or_unclear": "inadequate",
        },
        "solar_inventory.roof_visibility",
    )
    fencing = normalized.get("fencing_inventory")
    replace(
        fencing,
        "boundary_visibility",
        {
            "clear": "adequate",
            "fully visible": "adequate",
            "fully_visible": "adequate",
            "partially_clear": "partial",
            "partially_obscured": "partial",
            "partially_visible": "partial",
            "mostly_visible": "partial",
            "mostly_obscured_or_absent": "inadequate",
            "unfenced_or_not_visible": "inadequate",
            "poor": "inadequate",
            "none": "inadequate",
        },
        "fencing_inventory.boundary_visibility",
    )
    if isinstance(fencing, dict):
        for index, segment in enumerate(fencing.get("segments", [])):
            replace(
                segment,
                "sector",
                {
                    "entire_perimeter": "irregular",
                    "entire_periphery": "irregular",
                    "perimeter": "irregular",
                },
                f"fencing_inventory.segments.{index}.sector",
            )
            sector = segment.get("sector") if isinstance(segment, dict) else None
            boundary_phrases = {
                f"{sector}ern edge along residential street": "yes"
                for sector in ("north", "east", "south", "west")
            }
            boundary_phrases.update(
                {
                    "southern edge along major arterial road": "yes",
                    "western edge along residential street": "yes",
                    "outer_boundary": "yes",
                    "direct": "yes",
                }
            )
            replace(
                segment,
                "outer_boundary_relation",
                {
                    **boundary_phrases,
                    "adjacent to street and tree line": "unknown",
                    "along athletic field edge": "unknown",
                    "along parking and local access road": "unknown",
                    "along thoroughfare road": "unknown",
                    "uncertain": "unknown",
                    "bordering open land": "unknown",
                    "bordering road and properties": "unknown",
                    "bordering access roads and structures": "unknown",
                    "bordering dense forest": "unknown",
                    "adjacent_to_roadway": "unknown",
                    "adjacent_to_residential": "unknown",
                    "adjacent_to_open_land": "unknown",
                    "adjacent to roadway and treeline": "unknown",
                    "adjacent to residential properties": "unknown",
                    "adjacent to roadway": "unknown",
                    "adjacent to roadway and open land": "unknown",
                    "coincident": "yes",
                    "authoritative_boundary": "yes",
                    "adjacent to residential properties and access roads": "unknown",
                    "adjacent to tree line and residential lots": "unknown",
                    "adjacent to local streets": "unknown",
                    "adjacent to athletic fields and residential neighborhood": "unknown",
                    "coincident_with_tree_line": "unknown",
                    "coincident_with_turf": "unknown",
                    "coincident_with_wooded_edge": "unknown",
                    "coincident_with_access_road": "unknown",
                    "property line adjacent to residential street": "unknown",
                    "exact": "yes",
                    "primary_street_edge": "unknown",
                    "residential_edge": "unknown",
                    "street_edge": "unknown",
                    "commercial_edge": "unknown",
                    "eastern edge along street": "yes",
                    "southern edge along major road": "yes",
                    "western edge along street": "yes",
                    "adjacent to northern perimeter road and solar parking": "unknown",
                    "adjacent to Avenue A and eastern residential areas": "unknown",
                    "adjacent to 24th Street": "unknown",
                    "adjacent to irrigation canal and open desert": "unknown",
                    "unverified": "unknown",
                    "school_perimeter": "yes",
                    "unclear": "unknown",
                },
                f"fencing_inventory.segments.{index}.outer_boundary_relation",
            )
            replace(
                segment,
                "shadow_support",
                {
                    "none": "no",
                    "no fence shadow visible": "no",
                    "no_shadow": "no",
                    "minor shadow near property line": "yes",
                    "weak": "unknown",
                },
                f"fencing_inventory.segments.{index}.shadow_support",
            )
    for index, packet in enumerate(normalized.get("evidence_packets", [])):
        replace(
            packet,
            "visibility",
            {
                "clear": "adequate",
                "good": "adequate",
                "fully visible": "adequate",
                "fully_visible": "adequate",
                "partially_clear": "partial",
                "partially_visible": "partial",
                "partially_obscured": "partial",
                "mostly_visible": "partial",
                "poor": "inadequate",
                "none": "inadequate",
            },
            f"evidence_packets.{index}.visibility",
        )
        replace(
            packet,
            "campus_relation",
            {
                "on-site": "inside",
                "inside_campus": "inside",
                "inside campus boundary": "inside",
                "outer campus boundary": "inside",
                "outer boundary": "inside",
                "inside_boundary": "inside",
                "inside_authoritative_polygon": "inside",
                "fully_contained": "inside",
                "outer_boundary": "inside",
                "internal": "inside",
                "ambiguous_boundary": "uncertain",
                "ambiguous boundary": "uncertain",
                "uncertain_boundary": "uncertain",
                # Under center-only scope, visually "on campus" is not a
                # verified containment claim; retain that uncertainty.
                "on_campus": "uncertain",
                "onsite": "uncertain",
                "unclear": "uncertain",
                "off-site": "outside",
                "exterior_nearby": "outside",
            },
            f"evidence_packets.{index}.campus_relation",
        )
    for feature_index, assessment in enumerate(
        normalized.get("feature_assessments", [])
    ):
        if not isinstance(assessment, dict):
            continue
        for candidate_index, candidate in enumerate(assessment.get("candidates", [])):
            replace(
                candidate,
                "visibility",
                {
                    "fully visible": "clear",
                    "fully_visible": "clear",
                    "good": "clear",
                },
                (
                    f"feature_assessments.{feature_index}.candidates."
                    f"{candidate_index}.visibility"
                ),
            )
    if isinstance(solar, dict):
        for index, candidate in enumerate(solar.get("candidates", [])):
            replace(
                candidate,
                "support_structure",
                {
                    "metal carport structure": "parking_carport",
                    "metal carport frame": "parking_carport",
                    "school building roof structure": "school_building",
                    # This phrase identifies a mounting form, not the support
                    # beneath it. Preserve the ambiguity so the independent
                    # mount/support consistency check can require review.
                    "flat roof mount": "uncertain",
                    "roof-mounted racking": "uncertain",
                    "metal framing": "uncertain",
                    "roof-mounted rack": "uncertain",
                    "metal canopy framing": "parking_carport",
                },
                f"solar_inventory.candidates.{index}.support_structure",
            )
            replace(
                candidate,
                "support_surface_form",
                {
                    "carport canopy": "canopy",
                    "parking lot shade structure": "canopy",
                    "flat/low-slope roof section": "low-slope",
                    "pitched/flat hybrid roof plane": "uncertain",
                    "flat school building roof": "flat",
                    "flat roof": "flat",
                    "pitched roof": "pitched",
                    "parking lot canopy": "canopy",
                    "flat carport canopy rows": "canopy",
                },
                f"solar_inventory.candidates.{index}.support_surface_form",
            )
    return normalized, changes


def _validate_unknown_handling(parsed: dict[str, Any]) -> None:
    """Apply the same frozen unknown rules to live and reconciled responses."""
    for field, suggestion in parsed["measurements"].items():
        if suggestion["value"] != "unknown":
            continue
        if suggestion["suggested_confidence"] != 0.2:
            raise VLMResponseError(
                f"Gemini output field {field} must use "
                "suggested_confidence 0.20 for unknown"
            )
        if suggestion["review_required"] is not True or field not in parsed["review_fields"]:
            raise VLMResponseError(
                f"Gemini output field {field} must flag every unknown for review"
            )


_QUALIFYING_SOLAR_MOUNTS = frozenset(
    {"school_building_roof", "portable_classroom_roof"}
)


def _solar_polygon_area_m2(candidate: dict[str, Any], school: SchoolVLMInput) -> float:
    image_by_role = {"context": school.context, "detail": school.detail}
    image = image_by_role[candidate["image_role"]]
    with Image.open(image.path) as opened:
        width_px, height_px = opened.size
    width_m = width_px * image.metres_per_pixel
    height_m = height_px * image.metres_per_pixel
    points = [
        (point["x"] * width_m, point["y"] * height_m)
        for point in candidate["footprint_polygon_normalized"]
    ]
    twice_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )
    return abs(twice_area) / 2.0


def _validate_solar_inventory(
    parsed: dict[str, Any], school: SchoolVLMInput, config: dict[str, Any]
) -> dict[str, Any]:
    """Enforce rooftop eligibility and polygon-based aggregation independently."""
    inventory = parsed["solar_inventory"]
    candidates = inventory["candidates"]
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise VLMResponseError("Gemini solar inventory candidate_id values must be unique")

    expected_support_by_mount = {
        "school_building_roof": "school_building",
        "portable_classroom_roof": "portable_classroom",
        "parking_carport_canopy": "parking_carport",
        "ground_mounted": "ground",
        "uncertain": "uncertain",
    }
    for candidate in candidates:
        bounds = candidate["bbox_normalized"]
        if bounds["x_min"] >= bounds["x_max"] or bounds["y_min"] >= bounds["y_max"]:
            raise VLMResponseError(
                f"Gemini solar candidate {candidate['candidate_id']} has an invalid bounding box"
            )
        polygon = candidate["footprint_polygon_normalized"]
        if any(
            point["x"] < bounds["x_min"]
            or point["x"] > bounds["x_max"]
            or point["y"] < bounds["y_min"]
            or point["y"] > bounds["y_max"]
            for point in polygon
        ):
            raise VLMResponseError(
                f"Gemini solar candidate {candidate['candidate_id']} polygon must lie inside its bounding box"
            )

    candidate_areas = {
        candidate["candidate_id"]: _solar_polygon_area_m2(candidate, school)
        for candidate in candidates
    }
    for candidate_id, candidate_area in candidate_areas.items():
        if candidate_area <= 0:
            raise VLMResponseError(
                f"Gemini solar candidate {candidate_id} has a zero-area footprint polygon"
            )

    qualifying = [
        candidate
        for candidate in candidates
        if candidate["mount_location"] in _QUALIFYING_SOLAR_MOUNTS
    ]
    uncertain_mount = any(
        candidate["mount_location"] == "uncertain" for candidate in candidates
    )
    inadequate_visibility = inventory["roof_visibility"] == "inadequate"
    presence = parsed["measurements"]["solar_present"]
    area = parsed["measurements"]["solar_area_m2"]
    semantic_issues: list[dict[str, Any]] = []
    for candidate in candidates:
        expected_support = expected_support_by_mount[candidate["mount_location"]]
        if candidate["support_structure"] != expected_support:
            semantic_issues.append(
                {
                    "code": "solar_mount_support_structure_mismatch",
                    "fields": ["solar_present", "solar_area_m2"],
                    "severity": "force_abstention",
                    "message": (
                        f"Solar candidate {candidate['candidate_id']} classifies mount_location "
                        f"as {candidate['mount_location']!r} but support_structure as "
                        f"{candidate['support_structure']!r}"
                    ),
                }
            )

    if qualifying:
        expected_presence = "yes"
    elif uncertain_mount or inadequate_visibility:
        expected_presence = "unknown"
    else:
        expected_presence = "no"
    if presence["value"] != expected_presence:
        semantic_issues.append(
            {
                "code": "solar_presence_inventory_mismatch",
                "fields": ["solar_present", "solar_area_m2"],
                "severity": "force_abstention",
                "message": (
                    "Gemini solar_present is inconsistent with the classified solar inventory: "
                    f"reported {presence['value']!r}, expected {expected_presence!r}"
                ),
            }
        )

    qualifying_areas = [candidate_areas[candidate["candidate_id"]] for candidate in qualifying]
    area_is_unresolved = (
        uncertain_mount
        or inadequate_visibility
    )
    if not qualifying:
        expected_area: float | str = (
            "unknown" if uncertain_mount or inadequate_visibility else 0.0
        )
    elif area_is_unresolved:
        expected_area = "unknown"
    else:
        expected_area = sum(qualifying_areas)

    reported_area = area["value"]
    if expected_area == "unknown":
        area_matches = reported_area == "unknown"
    else:
        tolerance = config["evidence_policy"]["solar_area_consistency"]
        area_matches = isinstance(reported_area, (int, float)) and abs(
            float(reported_area) - expected_area
        ) <= max(
            float(tolerance["absolute_tolerance_m2"]),
            expected_area * float(tolerance["relative_tolerance_fraction"]),
        )
    if not area_matches:
        semantic_issues.append(
            {
                "code": "solar_area_polygon_mismatch",
                "fields": ["solar_present", "solar_area_m2"],
                "severity": "force_abstention",
                "message": (
                    "Gemini solar_area_m2 does not match the qualifying rooftop-polygon sum: "
                    f"reported {reported_area!r}, expected {expected_area!r}"
                ),
            }
        )

    solar_review_required = bool(candidates) or uncertain_mount or inadequate_visibility
    if solar_review_required:
        review_fields = set(parsed["review_fields"])
        for field, suggestion in (("solar_present", presence), ("solar_area_m2", area)):
            if suggestion["review_required"] is not True or field not in review_fields:
                semantic_issues.append(
                    {
                        "code": "required_solar_review_flag_missing",
                        "fields": [field],
                        "severity": "review_required",
                        "message": (
                            f"Gemini did not flag {field} even though the solar inventory "
                            "requires human review"
                        ),
                    }
                )
    mounts = {candidate["mount_location"] for candidate in candidates}
    uncertainty_flags: list[dict[str, Any]] = []
    if candidates:
        uncertainty_flags.append(
            {
                "code": "solar_mount_classification_requires_human_verification",
                "fields": ["solar_present", "solar_area_m2"],
                "reason": "Every visible solar candidate relies on model-inferred physical support.",
            }
        )
    if mounts & _QUALIFYING_SOLAR_MOUNTS and mounts - _QUALIFYING_SOLAR_MOUNTS:
        uncertainty_flags.append(
            {
                "code": "mixed_solar_mount_context",
                "fields": ["solar_present", "solar_area_m2"],
                "reason": "Qualifying and excluded/uncertain mount classes coexist in one campus image.",
            }
        )

    pipeline_review_fields = set(parsed["review_fields"])
    for issue in semantic_issues:
        pipeline_review_fields.update(issue["fields"])
    for flag in uncertainty_flags:
        pipeline_review_fields.update(flag["fields"])
    guarded_measurements = {
        field: suggestion["value"]
        for field, suggestion in parsed["measurements"].items()
    }
    if any(issue["severity"] == "force_abstention" for issue in semantic_issues):
        guarded_measurements["solar_present"] = "unknown"
        guarded_measurements["solar_area_m2"] = "unknown"
    return {
        "semantic_validation_status": (
            "needs_review" if semantic_issues else "passed"
        ),
        "pipeline_status": "needs_review" if pipeline_review_fields else "accepted",
        "semantic_issues": semantic_issues,
        "uncertainty_flags": uncertainty_flags,
        "pipeline_review_fields": sorted(pipeline_review_fields),
        "guarded_measurements": guarded_measurements,
        "roof_visibility": inventory["roof_visibility"],
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "mount_location": candidate["mount_location"],
                "qualifies_for_rooftop_metric": candidate["mount_location"]
                in _QUALIFYING_SOLAR_MOUNTS,
                "polygon_area_m2": round(candidate_areas[candidate["candidate_id"]], 1),
            }
            for candidate in candidates
        ],
        "qualifying_rooftop_area_m2": (
            "unknown" if expected_area == "unknown" else round(expected_area, 1)
        ),
    }


_COUNT_EVIDENCE_FIELDS = frozenset(
    {"portable_classroom_count", "full_size_sports_fields", "hard_courts"}
)


def _validate_normalized_box(box: dict[str, Any], label: str) -> None:
    if box["x_min"] >= box["x_max"] or box["y_min"] >= box["y_max"]:
        raise VLMResponseError(f"Gemini evidence {label} has an invalid bounding box")


def _negative_measurement(field: str, value: Any) -> bool:
    if field in {"solar_present", "running_track", "pool"}:
        return value == "no"
    if field in _COUNT_EVIDENCE_FIELDS or field == "solar_area_m2":
        return isinstance(value, (int, float)) and value == 0
    if field in {"perimeter_fencing", "dominant_fence_type"}:
        return value == "none"
    return False


def _positive_measurement(field: str, value: Any) -> bool:
    if field in {"solar_present", "running_track", "pool"}:
        return value == "yes"
    if field in _COUNT_EVIDENCE_FIELDS or field == "solar_area_m2":
        return isinstance(value, (int, float)) and value > 0
    if field == "perimeter_fencing":
        return value in {"partial", "full"}
    if field == "dominant_fence_type":
        return value not in {"none", "unknown"}
    return False


def _feature_direction(feature: str, measurements: dict[str, Any]) -> str:
    """Reduce linked measurements to the direction checked by the question registry."""
    if feature == "solar":
        value = measurements["solar_present"]["value"]
        return "positive" if value == "yes" else "negative" if value == "no" else "unknown"
    if feature == "fencing":
        value = measurements["perimeter_fencing"]["value"]
        return "positive" if value in {"full", "partial"} else "negative" if value == "none" else "unknown"
    field_by_feature = {
        "portable_classrooms": "portable_classroom_count",
        "running_track": "running_track",
        "sports_fields": "full_size_sports_fields",
        "hard_courts": "hard_courts",
        "pool": "pool",
    }
    field = field_by_feature[feature]
    value = measurements[field]["value"]
    if value == "unknown":
        return "unknown"
    if isinstance(value, str):
        return "positive" if value == "yes" else "negative"
    return "positive" if value > 0 else "negative"


def _deterministic_evidence_checks(
    parsed: dict[str, Any],
    field_protocol: dict[str, Any] | None = None,
    *,
    scope_mode: str = "authoritative_polygon",
) -> dict[str, Any]:
    """Check observable evidence consistency without making visual judgments."""
    hard_conflicts: list[dict[str, Any]] = []
    soft_risks: list[dict[str, Any]] = []
    packet_rows = parsed["evidence_packets"]
    packet_fields = [packet["field"] for packet in packet_rows]
    if len(packet_fields) != len(set(packet_fields)):
        raise VLMResponseError("Gemini evidence packets contain duplicate field names")
    if set(packet_fields) != set(parsed["measurements"]):
        raise VLMResponseError("Gemini evidence packets must cover all measurement fields")
    packets = {packet["field"]: packet for packet in packet_rows}
    measurements = parsed["measurements"]

    def add_hard(code: str, fields: list[str], message: str) -> None:
        hard_conflicts.append(
            {"code": code, "fields": fields, "severity": "force_abstention", "message": message}
        )

    def add_soft(code: str, fields: list[str], message: str) -> None:
        soft_risks.append(
            {"code": code, "fields": fields, "severity": "review_required", "message": message}
        )

    if field_protocol is not None:
        protocol_features = field_protocol.get("features", {})
        assessment_rows = parsed["feature_assessments"]
        assessment_names = [row["feature"] for row in assessment_rows]
        if len(assessment_names) != len(set(assessment_names)):
            raise VLMResponseError("Gemini feature assessments contain duplicate features")
        if set(assessment_names) != set(protocol_features):
            raise VLMResponseError(
                "Gemini feature assessments must cover the exact authoritative feature set"
            )
        assessments = {row["feature"]: row for row in assessment_rows}
        for feature, specification in protocol_features.items():
            assessment = assessments[feature]
            fields = list(specification["measurement_fields"])
            expected_questions = {
                row["id"]: row for row in specification["questions"]
            }
            answers = assessment["question_answers"]
            answer_ids = [row["question_id"] for row in answers]
            if len(answer_ids) != len(set(answer_ids)):
                raise VLMResponseError(
                    f"Gemini feature {feature} contains duplicate question IDs"
                )
            if set(answer_ids) != set(expected_questions):
                raise VLMResponseError(
                    f"Gemini feature {feature} must answer the exact authoritative question set"
                )
            direction = _feature_direction(feature, measurements)
            for answer in answers:
                question_id = answer["question_id"]
                observed = answer["answer"]
                expected = expected_questions[question_id].get(
                    f"expected_for_{direction}"
                )
                if observed == "unknown":
                    add_soft(
                        "item_specific_question_unknown",
                        fields,
                        f"{feature} question {question_id} is unknown",
                    )
                if expected is not None and observed != expected:
                    add_hard(
                        "directional_question_conflict",
                        fields,
                        f"{feature} question {question_id} is {observed!r}; "
                        f"the proposed {direction} result requires {expected!r}",
                    )

            candidates = assessment["candidates"]
            candidate_ids = [row["candidate_id"] for row in candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise VLMResponseError(
                    f"Gemini feature {feature} contains duplicate candidate IDs"
                )
            if direction == "positive" and not candidates and feature != "fencing":
                add_hard(
                    "positive_feature_without_candidate_inventory",
                    fields,
                    f"{feature} is positive but has no candidate inventory",
                )
            for candidate in candidates:
                _validate_normalized_box(
                    candidate["bbox_normalized"],
                    f"{feature}.{candidate['candidate_id']}",
                )
                if candidate["qualifies"] == "yes" and candidate[
                    "inside_campus_boundary"
                ] == "no":
                    add_hard(
                        "qualifying_candidate_outside_campus",
                        fields,
                        f"{feature} candidate {candidate['candidate_id']} qualifies but is outside campus",
                    )
                elif candidate["qualifies"] == "yes" and candidate[
                    "inside_campus_boundary"
                ] == "uncertain":
                    add_soft(
                        "qualifying_candidate_ownership_uncertain",
                        fields,
                        f"{feature} candidate {candidate['candidate_id']} has uncertain campus ownership",
                    )
                if candidate["qualifies"] == "yes" and candidate["contradictory_cues"]:
                    add_soft(
                        "qualifying_candidate_has_contradictory_cues",
                        fields,
                        f"{feature} candidate {candidate['candidate_id']} retains contradictory cues",
                    )

    for field, packet in packets.items():
        value = measurements[field]["value"]
        observations = packet["located_observable_facts"]
        components = packet["count_components"]
        component_ids = [item["component_id"] for item in components]
        if len(component_ids) != len(set(component_ids)):
            raise VLMResponseError(f"Gemini evidence packet {field} has duplicate component IDs")

        if _negative_measurement(field, value) and packet["visibility"] != "adequate":
            add_hard(
                "negative_answer_without_adequate_visibility",
                [field],
                f"{field} is negative but its structured evidence visibility is not adequate",
            )
        if _positive_measurement(field, value) and not observations and not components:
            add_hard(
                "positive_answer_without_located_evidence",
                [field],
                f"{field} is positive but has no located observation or count component",
            )
        if _positive_measurement(field, value) and packet["campus_relation"] in {
            "adjacent",
            "outside",
            "uncertain",
        }:
            add_soft(
                "positive_facility_has_ownership_or_boundary_risk",
                [field],
                f"{field} is positive but its campus relation is {packet['campus_relation']}",
            )

        if field in _COUNT_EVIDENCE_FIELDS and isinstance(value, int):
            component_sum = sum(item["physical_footprint_count"] for item in components)
            if component_sum != value:
                add_hard(
                    "count_component_sum_mismatch",
                    [field],
                    f"{field} reports {value} but count components sum to {component_sum}",
                )

    fencing = parsed["fencing_inventory"]
    segments = fencing["segments"]
    segment_ids = [segment["segment_id"] for segment in segments]
    if len(segment_ids) != len(set(segment_ids)):
        raise VLMResponseError("Gemini fencing inventory has duplicate segment IDs")
    boundary_sum = sum(float(segment["boundary_fraction"]) for segment in segments)
    if abs(boundary_sum - 1.0) > 0.02:
        add_hard(
            "fencing_boundary_fraction_sum_mismatch",
            ["perimeter_fencing", "dominant_fence_type"],
            f"Fencing segment boundary fractions sum to {boundary_sum:.3f}, not 1",
        )
    computed_minimum = 0.0
    computed_unobservable = 0.0
    for segment in segments:
        disposition_sum = sum(
            float(segment[name])
            for name in ("barrier_fraction", "unfenced_fraction", "unobservable_fraction")
        )
        if abs(disposition_sum - 1.0) > 0.02:
            add_hard(
                "fencing_segment_fraction_sum_mismatch",
                ["perimeter_fencing", "dominant_fence_type"],
                f"Fencing segment {segment['segment_id']} disposition fractions sum to "
                f"{disposition_sum:.3f}, not 1",
            )
        weight = float(segment["boundary_fraction"])
        computed_minimum += weight * float(segment["barrier_fraction"])
        computed_unobservable += weight * float(segment["unobservable_fraction"])
        if segment["barrier_fraction"] > 0 and segment["outer_boundary_relation"] == "no":
            add_hard(
                "interior_barrier_counted_as_perimeter",
                ["perimeter_fencing", "dominant_fence_type"],
                f"Fencing segment {segment['segment_id']} counts a barrier not on the outer boundary",
            )
    computed_maximum = computed_minimum + computed_unobservable
    reported_minimum = float(fencing["minimum_barrier_coverage_fraction"])
    reported_maximum = float(fencing["maximum_barrier_coverage_fraction"])
    if reported_minimum > reported_maximum:
        add_hard(
            "fencing_coverage_interval_reversed",
            ["perimeter_fencing"],
            "Fencing minimum coverage exceeds maximum coverage",
        )
    if abs(reported_minimum - computed_minimum) > 0.02 or abs(
        reported_maximum - computed_maximum
    ) > 0.02:
        add_hard(
            "fencing_coverage_interval_mismatch",
            ["perimeter_fencing"],
            "Reported fencing coverage bounds do not match the boundary-weighted segments",
        )
    perimeter_value = measurements["perimeter_fencing"]["value"]
    fence_type_value = measurements["dominant_fence_type"]["value"]
    if computed_minimum >= 0.8:
        expected_perimeter = "full"
    elif computed_maximum < 0.2:
        expected_perimeter = "none"
    elif computed_minimum >= 0.2 and computed_maximum < 0.8:
        expected_perimeter = "partial"
    else:
        expected_perimeter = "unknown"
    if perimeter_value != expected_perimeter:
        add_hard(
            "fencing_coverage_classification_mismatch",
            ["perimeter_fencing"],
            f"Coverage interval [{computed_minimum:.3f}, {computed_maximum:.3f}] "
            f"implies {expected_perimeter!r}, not {perimeter_value!r}",
        )

    observed_type = fencing["dominant_observed_type"]
    if perimeter_value == "none" and fence_type_value != "none":
        add_hard(
            "fencing_none_type_mismatch",
            ["perimeter_fencing", "dominant_fence_type"],
            "No perimeter fencing cannot have a non-none dominant fence type",
        )
    if perimeter_value in {"partial", "full"} and fence_type_value == "none":
        add_hard(
            "fencing_present_type_none_mismatch",
            ["perimeter_fencing", "dominant_fence_type"],
            "Present perimeter fencing cannot have dominant fence type none",
        )
    if observed_type not in {"unknown", fence_type_value}:
        add_hard(
            "fencing_inventory_type_mismatch",
            ["dominant_fence_type"],
            f"Fencing inventory reports {observed_type!r}, not {fence_type_value!r}",
        )
    elif observed_type == "unknown" and fence_type_value not in {"unknown", "none"}:
        add_soft(
            "fencing_type_not_supported_by_inventory",
            ["dominant_fence_type"],
            "A known dominant fence type is proposed while the segment inventory type is unknown",
        )

    if scope_mode in {"soft_boundary", "center_only"}:
        complete_outer_boundary_trace = (
            parsed["campus_assessment"]["status"] == "confirmed"
            and fencing["boundary_visibility"] == "adequate"
            and all(segment["outer_boundary_relation"] == "yes" for segment in segments)
            and all(float(segment["unobservable_fraction"]) <= 0.02 for segment in segments)
        )
        if not complete_outer_boundary_trace and (
            perimeter_value != "unknown" or fence_type_value != "unknown"
        ):
            add_hard(
                "non_authoritative_scope_cannot_support_fencing",
                ["perimeter_fencing", "dominant_fence_type"],
                "The non-authoritative campus scope does not independently trace a complete "
                "observable outer boundary, so known fencing values are not defensible.",
            )

    guarded = {
        field: suggestion["value"] for field, suggestion in measurements.items()
    }
    for conflict in hard_conflicts:
        for field in conflict["fields"]:
            guarded[field] = "unknown"
    return {
        "hard_conflicts": hard_conflicts,
        "soft_risks": soft_risks,
        "guarded_measurements": guarded,
        "hard_conflict_count": len(hard_conflicts),
        "soft_risk_count": len(soft_risks),
    }


def _uncertainty_assessment(
    parsed: dict[str, Any],
    solar_summary: dict[str, Any],
    evidence_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Combine model uncertainty with frozen deterministic risk flags."""
    policy = config["uncertainty_policy"]
    review_fields = set(
        solar_summary.get("pipeline_review_fields", parsed.get("review_fields", []))
    )
    reasons: dict[str, set[str]] = {field: set() for field in parsed["measurements"]}

    for field, suggestion in parsed["measurements"].items():
        value = suggestion["value"]
        if policy["respect_model_review_required"] and suggestion["review_required"]:
            review_fields.add(field)
            reasons[field].add("model_review_required")
        if policy["review_model_unknown"] and value == "unknown":
            review_fields.add(field)
            reasons[field].add("model_abstained_unknown")
        if suggestion["suggested_confidence"] <= policy["review_model_confidence_at_or_below"]:
            review_fields.add(field)
            reasons[field].add("model_confidence_at_or_below_threshold")
        if field in policy["always_review_fields"]:
            review_fields.add(field)
            reasons[field].add("frozen_high_risk_field_policy")
        if field in policy["review_when_nonzero_fields"] and (
            isinstance(value, (int, float)) and value > 0
        ):
            review_fields.add(field)
            reasons[field].add("nonzero_count_requires_object_review")

    semantic_issues = solar_summary.get("semantic_issues", [])
    uncertainty_flags = solar_summary.get("uncertainty_flags", [])
    for issue in semantic_issues:
        for field in issue["fields"]:
            review_fields.add(field)
            reasons[field].add(issue["code"])
    for flag in uncertainty_flags:
        for field in flag["fields"]:
            review_fields.add(field)
            reasons[field].add(flag["code"])

    for conflict in evidence_summary["hard_conflicts"]:
        for field in conflict["fields"]:
            review_fields.add(field)
            reasons[field].add(conflict["code"])
    for risk in evidence_summary["soft_risks"]:
        for field in risk["fields"]:
            review_fields.add(field)
            reasons[field].add(risk["code"])

    guarded = dict(
        solar_summary.get(
            "guarded_measurements",
            {
                field: suggestion["value"]
                for field, suggestion in parsed["measurements"].items()
            },
        )
    )
    for conflict in evidence_summary["hard_conflicts"]:
        for field in conflict["fields"]:
            guarded[field] = "unknown"
    auto_accept_fields = sorted(
        field
        for field, suggestion in parsed["measurements"].items()
        if field not in review_fields
        and guarded[field] != "unknown"
        and suggestion["suggested_confidence"] >= 0.8
    )
    return {
        "pipeline_status": "needs_review" if review_fields else "accepted",
        "pipeline_review_fields": sorted(review_fields),
        "review_reasons_by_field": {
            field: sorted(field_reasons)
            for field, field_reasons in reasons.items()
            if field_reasons
        },
        "guarded_measurements": guarded,
        "auto_accept_candidate_fields": auto_accept_fields,
        "semantic_issue_count": len(semantic_issues),
        "hard_evidence_conflict_count": evidence_summary["hard_conflict_count"],
        "soft_evidence_risk_count": evidence_summary["soft_risk_count"],
        "raw_predictions_preserved": True,
        "auditor_may_overwrite_primary_value": False,
    }


def _sanitized_error_detail(error: Exception, maximum_length: int = 1200) -> str:
    """Extract useful provider diagnostics while redacting likely credentials."""
    candidates = (
        getattr(error, "body", None),
        getattr(error, "message", None),
        str(error),
    )
    detail = ""
    for candidate in candidates:
        if candidate in (None, "", {}):
            continue
        if isinstance(candidate, (dict, list)):
            detail = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        else:
            detail = str(candidate)
        if detail.strip():
            break
    detail = re.sub(r"AIza[A-Za-z0-9_-]{20,}", "[REDACTED_API_KEY]", detail)
    detail = re.sub(
        r"(?i)(x-goog-api-key|api[_-]?key)(\s*[:=]\s*)[^\s,;\"'}]+",
        r"\1\2[REDACTED]",
        detail,
    )
    detail = " ".join(detail.split())
    if len(detail) > maximum_length:
        detail = detail[: maximum_length - 3] + "..."
    return detail


class RequestLedger:
    """Conservative persistent request accounting for one frozen configuration."""

    def __init__(
        self,
        path: Path,
        config: dict[str, Any],
        *,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.path = path
        self.config = config
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep

    def _records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        value = _read_object(self.path)
        records = value.get("requests", [])
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise VLMConfigurationError("request ledger has invalid records")
        legacy_configuration_id = value.get("configuration_id")
        if isinstance(legacy_configuration_id, str):
            records = [
                {**row, "configuration_id": row.get("configuration_id", legacy_configuration_id)}
                for row in records
            ]
        records = [
            {
                **row,
                "model": row.get("model")
                or (
                    # Preserve the model identity of pre-v1.2 ledger rows for
                    # quota/audit accounting. This is not an executable model
                    # option; the active configuration has only one model.
                    "gemini-3.7-flash"
                    if str(row.get("configuration_id", "")).startswith(
                        "school-facilities-vlm-pilot-v1."
                    )
                    else "unknown"
                ),
            }
            for row in records
        ]
        return records

    def reserve(self, school_id: str, mode: RunMode, attempt: int) -> None:
        records = self._records()
        limits = self.config["request_limits"]
        now = self.now()
        if now.tzinfo is None:
            raise VLMConfigurationError("request-ledger clock must return a timezone-aware datetime")
        cutoff = now - timedelta(hours=24)
        recent = [
            row
            for row in records
            if row.get("model") == self.config["model"]
            and datetime.fromisoformat(str(row["reserved_at_utc"]).replace("Z", "+00:00"))
            >= cutoff
        ]
        if len(recent) >= int(limits["requests_per_day"]):
            raise VLMQuotaError(
                f"{limits['requests_per_day']}-request rolling-24-hour safety cap reached"
            )
        cap_field = {
            "pilot": "pilot_hard_request_cap_including_retries",
            "production": "production_hard_request_cap_including_retries",
            "auditor": "auditor_hard_request_cap_including_retries",
        }[mode]
        mode_count = sum(
            row.get("mode") == mode
            and row.get("configuration_id") == self.config["configuration_id"]
            for row in records
        )
        if mode_count >= int(limits[cap_field]):
            raise VLMQuotaError(f"{mode} hard request cap reached")

        school_request_count = sum(
            row.get("mode") == mode
            and row.get("school_id") == school_id
            and row.get("configuration_id") == self.config["configuration_id"]
            for row in records
        )
        school_request_cap = 1 + int(limits["maximum_retries_per_school"])
        if school_request_count >= school_request_cap:
            raise VLMQuotaError(
                f"{mode} per-school request cap reached for {school_id}: "
                f"{school_request_count} reservations already recorded"
            )

        if records:
            previous = max(
                datetime.fromisoformat(str(row["reserved_at_utc"]).replace("Z", "+00:00"))
                for row in records
            )
            remaining = float(limits["minimum_seconds_between_requests"]) - (
                now - previous
            ).total_seconds()
            if remaining > 0:
                self.sleep(remaining)
                now = self.now()

        records.append(
            {
                "configuration_id": self.config["configuration_id"],
                "model": self.config["model"],
                "school_id": school_id,
                "mode": mode,
                "attempt": school_request_count + 1,
                "reserved_at_utc": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
        _atomic_json(
            self.path,
            {
                "schema_version": "1.1",
                "provider_scope": "Gemini Developer API project",
                "requests": records,
            },
        )


class GeminiVLMClient:
    def __init__(
        self,
        root: Path,
        create_interaction: InteractionCreator,
        *,
        profile: VLMProfile = "final",
        close_client: Callable[[], None] | None = None,
        ledger_path: Path | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = root.resolve()
        self.profile = profile
        self.config, self.schema, _ = load_vlm_bundle(self.root, profile)
        self.field_protocol = _read_object(
            self.root / self.config["evidence_policy"]["field_protocol_path"]
        )
        self.create_interaction = create_interaction
        self.close_client = close_client
        self.sleep = sleep
        self.ledger = RequestLedger(
            ledger_path or self.root / "data" / "model_outputs" / "request_ledger.json",
            self.config,
            now=now,
            sleep=sleep,
        )

    @classmethod
    def from_environment(
        cls,
        root: Path,
        profile: VLMProfile = "final",
    ) -> GeminiVLMClient:
        config, _, _ = load_vlm_bundle(root, profile)
        from .credentials import CredentialError, load_api_key

        credential_config = config["credentials"]
        try:
            api_key, _credential_source = load_api_key(
                environment_variable=credential_config["api_key_environment_variable"],
                secrets_file=root / credential_config["local_secrets_file"],
            )
        except CredentialError as error:
            raise VLMConfigurationError(str(error)) from error

        from google import genai
        from google.genai import types

        limits = config["request_limits"]
        google_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version=config["api_version"],
                timeout=int(limits["request_timeout_seconds"]) * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        return cls(
            root,
            google_client.interactions.create,
            profile=profile,
            close_client=google_client.close,
        )

    def close(self) -> None:
        if self.close_client is not None:
            self.close_client()

    def assess(
        self,
        school: SchoolVLMInput,
        *,
        mode: RunMode,
        overwrite: bool = False,
        allow_retry: bool = False,
    ) -> Path:
        request = build_interaction_request(self.root, school, self.profile)
        request_fingerprint = _request_fingerprint(request)
        output_directory, rejected_directory, quarantined = (
            _response_output_directories(self.root, self.config, school.school_id)
        )
        output_path = output_directory / f"{school.school_id}.json"
        if output_path.exists() and not overwrite:
            raise VLMConfigurationError(f"raw response already exists: {output_path}")

        input_sha256 = {
            image.role: _sha256(image.path)
            for image in (school.context, school.detail, *school.facility_crops)
        }
        detail_geotiff = school.detail.path.with_suffix(".tif")
        supporting_sha256 = {
            "detail.tif": _sha256(detail_geotiff)
        } if detail_geotiff.is_file() else {}

        limits = self.config["request_limits"]
        maximum_attempts = (
            1 + int(limits["maximum_retries_per_school"])
            if allow_retry
            else 1
        )
        response: Any = None
        for attempt in range(1, maximum_attempts + 1):
            self.ledger.reserve(school.school_id, mode, attempt)
            try:
                response = self.create_interaction(
                    **request,
                    timeout=float(limits["request_timeout_seconds"]),
                )
                break
            except Exception as error:
                status = getattr(error, "status_code", None)
                if status is None and getattr(error, "response", None) is not None:
                    status = getattr(error.response, "status_code", None)
                retryable = status in set(limits["retry_statuses"])
                if attempt >= maximum_attempts or not retryable:
                    detail = _sanitized_error_detail(error)
                    suffix = f": {detail}" if detail else ""
                    raise VLMError(
                        f"Gemini request failed with status {status or 'unknown'}{suffix}"
                    ) from error
                self.sleep(float(limits["retry_initial_backoff_seconds"]))

        status = _response_value(response, "status")
        output_text = _response_value(response, "output_text")
        if status != "completed":
            provider_diagnostics = _sanitized_provider_diagnostics(response)
            interaction_id = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                str(_response_value(response, "id") or "unknown-interaction"),
            )
            rejected_path = rejected_directory / f"{school.school_id}-{interaction_id}.json"
            rejected_record = {
                "schema_version": "1.10.0",
                "configuration_id": self.config["configuration_id"],
                "profile": self.profile,
                "school_id": school.school_id,
                "model": self.config["model"],
                "interaction_id": _response_value(response, "id"),
                "status": status,
                "created": _response_value(response, "created"),
                "usage": _serializable(_response_value(response, "usage")),
                "request_fingerprint_sha256": request_fingerprint,
                "input_sha256": input_sha256,
                "supporting_artifact_sha256": supporting_sha256,
                "campus_boundary_detail_normalized": [
                    {"x": x, "y": y}
                    for x, y in school.campus_boundary_detail_normalized
                ],
                "campus_scope_mode": school.campus_scope_mode,
                "scope_boundary_authority": school.scope_boundary_authority,
                "measurement_search_scope": school.measurement_search_scope,
                "blind_validation_quarantined": quarantined,
                "output_text": output_text if isinstance(output_text, str) else None,
                "parsed_output": None,
                "provider_diagnostics": provider_diagnostics,
                "rejection_stage": "interaction_status",
                "validation_error": (
                    f"Gemini interaction status was {status!r}, not 'completed'"
                ),
            }
            _atomic_json(rejected_path, rejected_record)
            diagnostic_summary = json.dumps(
                provider_diagnostics, ensure_ascii=False, sort_keys=True
            )
            if len(diagnostic_summary) > 800:
                diagnostic_summary = diagnostic_summary[:797] + "..."
            raise VLMResponseError(
                f"Gemini interaction status was {status!r}, not 'completed'; "
                f"rejected response preserved at {rejected_path.relative_to(self.root)}; "
                f"provider diagnostics: {diagnostic_summary}"
            )
        if not isinstance(output_text, str) or not output_text.strip():
            raise VLMResponseError("Gemini interaction returned no output_text")

        provider_vocabulary_normalizations: list[dict[str, str]] = []

        def response_record(parsed_output: Any) -> dict[str, Any]:
            record = {
                "schema_version": "1.10.0",
                "configuration_id": self.config["configuration_id"],
                "profile": self.profile,
                "school_id": school.school_id,
                "model": self.config["model"],
                "interaction_id": _response_value(response, "id"),
                "status": status,
                "created": _response_value(response, "created"),
                "usage": _serializable(_response_value(response, "usage")),
                "request_fingerprint_sha256": request_fingerprint,
                "input_sha256": input_sha256,
                "supporting_artifact_sha256": supporting_sha256,
                "campus_boundary_detail_normalized": [
                    {"x": x, "y": y}
                    for x, y in school.campus_boundary_detail_normalized
                ],
                "campus_scope_mode": school.campus_scope_mode,
                "scope_boundary_authority": school.scope_boundary_authority,
                "measurement_search_scope": school.measurement_search_scope,
                "blind_validation_quarantined": quarantined,
                "output_text": output_text,
                "parsed_output": parsed_output,
            }
            if provider_vocabulary_normalizations:
                record["provider_vocabulary_normalizations"] = (
                    provider_vocabulary_normalizations
                )
            return record

        parsed: Any = None
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as error:
            validation_error = VLMResponseError(f"Gemini output is not valid JSON: {error}")
        else:
            parsed, provider_vocabulary_normalizations = (
                _normalize_provider_vocabulary(parsed)
            )
            try:
                errors = sorted(
                    Draft202012Validator(self.schema).iter_errors(parsed),
                    key=lambda item: list(item.path),
                )
                if errors:
                    first = errors[0]
                    location = ".".join(str(item) for item in first.path) or "<root>"
                    raise VLMResponseError(
                        f"Gemini output failed schema validation at {location}: {first.message}"
                    )
                if parsed.get("school_id") != school.school_id:
                    raise VLMResponseError(
                        f"Gemini output school_id {parsed.get('school_id')!r} "
                        f"does not match {school.school_id!r}"
                    )
                _validate_unknown_handling(parsed)
                derived_solar_summary = _validate_solar_inventory(
                    parsed, school, self.config
                )
                derived_evidence_summary = _deterministic_evidence_checks(
                    parsed, self.field_protocol, scope_mode=school.campus_scope_mode
                )
                uncertainty_assessment = _uncertainty_assessment(
                    parsed,
                    derived_solar_summary,
                    derived_evidence_summary,
                    self.config,
                )
            except VLMResponseError as error:
                validation_error = error
            else:
                validation_error = None

        if validation_error is not None:
            rejected_record = response_record(parsed)
            rejected_record["validation_error"] = str(validation_error)
            interaction_id = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                str(_response_value(response, "id") or "unknown-interaction"),
            )
            rejected_path = rejected_directory / f"{school.school_id}-{interaction_id}.json"
            _atomic_json(rejected_path, rejected_record)
            raise VLMResponseError(
                f"{validation_error}; rejected response preserved at "
                f"{rejected_path.relative_to(self.root)}"
            )

        raw_record = response_record(parsed)
        raw_record["derived_solar_summary"] = derived_solar_summary
        raw_record["derived_evidence_summary"] = derived_evidence_summary
        raw_record["uncertainty_assessment"] = uncertainty_assessment
        _atomic_json(output_path, raw_record)
        return output_path


def reconcile_rejected_response(root: Path, school_id: str) -> Path:
    """Promote a schema-valid rejected response into a flagged raw pipeline record."""
    root = root.resolve()
    config, schema, _prompt = load_vlm_bundle(root)
    school = load_approved_school_input(root, school_id)
    request_fingerprint = _request_fingerprint(build_interaction_request(root, school))
    output_directory, rejected_directory, quarantined = _response_output_directories(
        root, config, school_id
    )
    candidates = sorted(
        rejected_directory.glob(f"{school_id}-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise VLMConfigurationError(f"no rejected active-version response exists for {school_id}")
    source_path = candidates[0]
    record = _read_object(source_path)
    if record.get("configuration_id") != config["configuration_id"]:
        raise VLMConfigurationError("rejected response configuration does not match the active version")
    if record.get("school_id") != school_id:
        raise VLMConfigurationError("rejected response school_id mismatch")
    expected_input_hashes = {
        image.role: _sha256(image.path) for image in (school.context, school.detail)
    }
    if record.get("input_sha256") != expected_input_hashes:
        raise VLMConfigurationError("rejected response image hashes do not match current inputs")
    detail_geotiff = school.detail.path.with_suffix(".tif")
    if record.get("supporting_artifact_sha256") != {"detail.tif": _sha256(detail_geotiff)}:
        raise VLMConfigurationError("rejected response detail GeoTIFF hash does not match")
    expected_boundary = [
        {"x": x, "y": y} for x, y in school.campus_boundary_detail_normalized
    ]
    if record.get("campus_boundary_detail_normalized") != expected_boundary:
        raise VLMConfigurationError("rejected response campus boundary does not match")
    if record.get("campus_scope_mode") != school.campus_scope_mode:
        raise VLMConfigurationError("rejected response campus scope mode does not match")
    if record.get("measurement_search_scope") != school.measurement_search_scope:
        raise VLMConfigurationError("rejected response measurement search scope does not match")
    parsed, vocabulary_normalizations = _normalize_provider_vocabulary(
        record.get("parsed_output")
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(parsed), key=lambda item: list(item.path)
    )
    if errors:
        raise VLMResponseError("only a schema-valid rejected response can be reconciled")
    if parsed.get("school_id") != school_id:
        raise VLMResponseError("rejected response parsed school_id mismatch")
    _validate_unknown_handling(parsed)
    derived_solar_summary = _validate_solar_inventory(parsed, school, config)
    field_protocol = _read_object(
        root / config["evidence_policy"]["field_protocol_path"]
    )
    derived_evidence_summary = _deterministic_evidence_checks(
        parsed, field_protocol, scope_mode=school.campus_scope_mode
    )
    uncertainty_assessment = _uncertainty_assessment(
        parsed, derived_solar_summary, derived_evidence_summary, config
    )

    output_path = output_directory / f"{school_id}.json"
    if output_path.exists():
        existing = _read_object(output_path)
        if existing.get("reconciled_from_rejected") != source_path.relative_to(root).as_posix():
            raise VLMConfigurationError(f"raw response already exists: {output_path}")
    reconciled = dict(record)
    reconciled["schema_version"] = "1.10.0"
    reconciled["parsed_output"] = parsed
    existing_normalizations = record.get("provider_vocabulary_normalizations", [])
    reconciled["provider_vocabulary_normalizations"] = [
        *existing_normalizations,
        *vocabulary_normalizations,
    ]
    reconciled.pop("validation_error", None)
    reconciled["derived_solar_summary"] = derived_solar_summary
    reconciled["derived_evidence_summary"] = derived_evidence_summary
    reconciled["uncertainty_assessment"] = uncertainty_assessment
    reconciled["request_fingerprint_sha256"] = request_fingerprint
    reconciled["reconciled_from_rejected"] = source_path.relative_to(root).as_posix()
    reconciled["reconciliation_policy"] = (
        "schema-valid raw predictions retained; deterministic semantic contradictions "
        "force guarded unknown values and review flags"
    )
    reconciled["blind_validation_quarantined"] = quarantined
    _atomic_json(output_path, reconciled)
    return output_path
