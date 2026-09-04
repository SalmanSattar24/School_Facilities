from __future__ import annotations

import json
import hashlib
import math
import re
import time
from datetime import date
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import requests


OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
GENERIC_NAME_TOKENS = {
    "school",
    "elementary",
    "primary",
    "middle",
    "high",
    "academy",
    "alternative",
    "education",
    "ed",
    "hs",
    "es",
}


class CampusResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CampusCandidate:
    osm_type: str
    osm_id: int
    name: str
    operator: str | None
    center_latitude: float
    center_longitude: float
    bbox_wgs84: tuple[float, float, float, float] | None
    geometry: tuple[tuple[float, float], ...]
    name_similarity: float
    distance_m: float
    score: float


@dataclass(frozen=True)
class CampusResolution:
    school_id: str
    school_name: str
    status: str
    method: str
    requested_latitude: float
    requested_longitude: float
    resolved_latitude: float
    resolved_longitude: float
    bbox_wgs84: tuple[float, float, float, float] | None
    geometry: tuple[tuple[float, float], ...]
    matched_name: str | None
    source_element: str | None
    name_similarity: float | None
    distance_from_ccd_m: float | None
    candidate_margin: float | None
    recommended_detail_extent_m: int
    unclamped_detail_extent_m: int
    detail_extent_clipped_at_maximum: bool
    requires_human_review: bool
    reason: str
    candidates_considered: int


def _tokens(value: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    tokens = [token for token in normalized.split() if token not in GENERIC_NAME_TOKENS]
    return tokens or normalized.split()


def _name_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    left_core = " ".join(left_tokens)
    right_core = " ".join(right_tokens)
    sequence = SequenceMatcher(None, left_core, right_core).ratio()
    union = set(left_tokens) | set(right_tokens)
    jaccard = len(set(left_tokens) & set(right_tokens)) / len(union) if union else 0.0
    return max(sequence, jaccard)


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _element_geometry(element: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    geometry = element.get("geometry", [])
    if isinstance(geometry, list):
        points = tuple(
            (float(point["lat"]), float(point["lon"]))
            for point in geometry
            if isinstance(point, dict) and "lat" in point and "lon" in point
        )
        if points:
            return points
    if "lat" in element and "lon" in element:
        return ((float(element["lat"]), float(element["lon"])),)
    center = element.get("center", {})
    if isinstance(center, dict) and "lat" in center and "lon" in center:
        return ((float(center["lat"]), float(center["lon"])),)
    return ()


def _element_bbox(
    element: dict[str, Any], geometry: tuple[tuple[float, float], ...]
) -> tuple[float, float, float, float] | None:
    bounds = element.get("bounds", {})
    if isinstance(bounds, dict) and all(
        key in bounds for key in ("minlat", "minlon", "maxlat", "maxlon")
    ):
        return (
            float(bounds["minlon"]),
            float(bounds["minlat"]),
            float(bounds["maxlon"]),
            float(bounds["maxlat"]),
        )
    if len(geometry) >= 3:
        latitudes = [point[0] for point in geometry]
        longitudes = [point[1] for point in geometry]
        return (min(longitudes), min(latitudes), max(longitudes), max(latitudes))
    return None


def detail_extent_plan(
    bbox: tuple[float, float, float, float] | None,
    *,
    minimum_extent_m: int = 250,
    maximum_extent_m: int = 1200,
    buffer_each_side_m: float = 60.0,
    rounding_increment_m: int = 50,
) -> tuple[int, int, bool]:
    """Return (selected extent, unclamped extent, clipped-at-maximum flag)."""
    if minimum_extent_m <= 0 or maximum_extent_m < minimum_extent_m:
        raise ValueError("invalid adaptive detail extent limits")
    if buffer_each_side_m < 0 or rounding_increment_m <= 0:
        raise ValueError("invalid adaptive detail buffer or rounding increment")
    if bbox is None:
        return 600, 600, False
    minlon, minlat, maxlon, maxlat = bbox
    midlat = (minlat + maxlat) / 2
    midlon = (minlon + maxlon) / 2
    width = _distance_m(midlat, minlon, midlat, maxlon)
    height = _distance_m(minlat, midlon, maxlat, midlon)
    required = max(width, height) + 2 * buffer_each_side_m
    rounded = int(math.ceil(required / rounding_increment_m) * rounding_increment_m)
    selected = min(maximum_extent_m, max(minimum_extent_m, rounded))
    return selected, rounded, rounded > maximum_extent_m


def soft_detail_extent_plan(
    bbox: tuple[float, float, float, float] | None,
) -> tuple[int, int, bool]:
    """Plan a deliberately generous crop around non-authoritative boundary guidance."""
    return detail_extent_plan(
        bbox,
        minimum_extent_m=600,
        maximum_extent_m=1200,
        buffer_each_side_m=150.0,
        rounding_increment_m=50,
    )


def extent_plan_for_scope(
    scope_mode: str,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[int, int, bool]:
    if scope_mode == "soft_boundary":
        return soft_detail_extent_plan(bbox)
    if scope_mode == "center_only":
        return 800, 800, False
    if scope_mode != "authoritative_polygon":
        raise ValueError(f"unsupported campus scope mode: {scope_mode!r}")
    return detail_extent_plan(bbox)


def recommended_detail_extent_m(
    bbox: tuple[float, float, float, float] | None,
) -> int:
    """Return the frozen adaptive square detail extent for a campus boundary."""
    return detail_extent_plan(bbox)[0]


def overpass_school_candidates(
    latitude: float,
    longitude: float,
    *,
    radius_m: int = 1500,
    endpoint: str = OVERPASS_ENDPOINT,
    timeout_seconds: float = 60.0,
    attempts: int = 4,
    post: Callable[..., Any] = requests.post,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    query = (
        f'[out:json][timeout:25];nwr["amenity"="school"]'
        f"(around:{radius_m},{latitude},{longitude});out tags center geom;"
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": "school-facilities-research/0.1"},
                timeout=timeout_seconds,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise CampusResolutionError(f"Overpass returned HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            elements = payload.get("elements", [])
            if not isinstance(elements, list):
                raise CampusResolutionError("Overpass response elements must be a list")
            return [element for element in elements if isinstance(element, dict)]
        except (requests.RequestException, ValueError, CampusResolutionError) as error:
            last_error = error
            if attempt + 1 < attempts:
                sleep(min(8.0, 0.5 * (2**attempt)))
    raise CampusResolutionError(f"public campus-boundary query failed: {last_error}")


def select_campus(
    school: dict[str, str], elements: list[dict[str, Any]]
) -> CampusResolution:
    school_id = school["school_id"]
    school_name = school["school_name"]
    latitude = float(school["latitude"])
    longitude = float(school["longitude"])
    candidates: list[CampusCandidate] = []

    for element in elements:
        tags = element.get("tags", {})
        name = tags.get("name") if isinstance(tags, dict) else None
        if not isinstance(name, str) or not name.strip():
            continue
        geometry = _element_geometry(element)
        if not geometry:
            continue
        bbox = _element_bbox(element, geometry)
        if bbox is not None:
            center_latitude = (bbox[1] + bbox[3]) / 2
            center_longitude = (bbox[0] + bbox[2]) / 2
        else:
            center_latitude, center_longitude = geometry[0]
        similarity = _name_similarity(school_name, name)
        distance = _distance_m(latitude, longitude, center_latitude, center_longitude)
        distance_score = max(0.0, 1.0 - distance / 1500.0)
        polygon_bonus = 0.1 if bbox is not None else 0.0
        score = 0.7 * similarity + 0.2 * distance_score + polygon_bonus
        candidates.append(
            CampusCandidate(
                osm_type=str(element.get("type", "unknown")),
                osm_id=int(element.get("id", 0)),
                name=name,
                operator=(tags.get("operator") if isinstance(tags, dict) else None),
                center_latitude=center_latitude,
                center_longitude=center_longitude,
                bbox_wgs84=bbox,
                geometry=geometry,
                name_similarity=similarity,
                distance_m=distance,
                score=score,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    if not candidates:
        return CampusResolution(
            school_id=school_id,
            school_name=school_name,
            status="unresolved",
            method="ccd_fallback_no_public_match",
            requested_latitude=latitude,
            requested_longitude=longitude,
            resolved_latitude=latitude,
            resolved_longitude=longitude,
            bbox_wgs84=None,
            geometry=(),
            matched_name=None,
            source_element=None,
            name_similarity=None,
            distance_from_ccd_m=None,
            candidate_margin=None,
            recommended_detail_extent_m=600,
            unclamped_detail_extent_m=600,
            detail_extent_clipped_at_maximum=False,
            requires_human_review=True,
            reason="No named public school candidate was found within 1,500 metres.",
            candidates_considered=0,
        )

    best = candidates[0]
    margin = best.score - candidates[1].score if len(candidates) > 1 else best.score
    has_polygon = best.bbox_wgs84 is not None and len(best.geometry) >= 3
    confirmed = (
        has_polygon
        and best.name_similarity >= 0.8
        and best.distance_m <= 800
        and margin >= 0.12
    )
    probable = has_polygon and best.name_similarity >= 0.6 and best.distance_m <= 1000
    status = "confirmed" if confirmed else "probable" if probable else "unresolved"
    requires_review = status != "confirmed"
    reason = (
        f"Matched {best.name!r} from OSM {best.osm_type}/{best.osm_id}; "
        f"name similarity {best.name_similarity:.3f}, CCD-to-center distance "
        f"{best.distance_m:.1f} m, candidate margin {margin:.3f}."
    )
    if status == "unresolved":
        reason += " Match did not meet the automatic polygon/name/distance thresholds."
    elif status == "probable":
        reason += " Match is usable for a provisional crop but requires human review."
    else:
        reason += " Match passed the frozen automatic-resolution thresholds."

    selected_extent, unclamped_extent, extent_clipped = detail_extent_plan(best.bbox_wgs84)
    if extent_clipped:
        reason += (
            " The campus plus buffer exceeds the 1,200 m detail limit; "
            "the case is flagged for a future tiled-detail review."
        )
        requires_review = True

    return CampusResolution(
        school_id=school_id,
        school_name=school_name,
        status=status,
        method="openstreetmap_polygon_match",
        requested_latitude=latitude,
        requested_longitude=longitude,
        resolved_latitude=best.center_latitude,
        resolved_longitude=best.center_longitude,
        bbox_wgs84=best.bbox_wgs84,
        geometry=best.geometry,
        matched_name=best.name,
        source_element=f"https://www.openstreetmap.org/{best.osm_type}/{best.osm_id}",
        name_similarity=best.name_similarity,
        distance_from_ccd_m=best.distance_m,
        candidate_margin=margin,
        recommended_detail_extent_m=selected_extent,
        unclamped_detail_extent_m=unclamped_extent,
        detail_extent_clipped_at_maximum=extent_clipped,
        requires_human_review=requires_review,
        reason=reason,
        candidates_considered=len(candidates),
    )


def resolution_record(resolution: CampusResolution) -> dict[str, Any]:
    record = asdict(resolution)
    record["schema_version"] = "1.0"
    record["resolved_center"] = {
        "latitude": record.pop("resolved_latitude"),
        "longitude": record.pop("resolved_longitude"),
    }
    record["requested_ccd_coordinate"] = {
        "latitude": record.pop("requested_latitude"),
        "longitude": record.pop("requested_longitude"),
    }
    record["boundary_notes"] = record["reason"]
    has_polygon = bool(record.get("bbox_wgs84")) and len(record.get("geometry", [])) >= 3
    record["scope_mode"] = "authoritative_polygon" if has_polygon else "center_only"
    record["scope_boundary_authority"] = "authoritative" if has_polygon else "none"
    record["measurement_search_scope"] = (
        "inside_authoritative_polygon" if has_polygon else "entire_detail_image"
    )
    return record


def write_resolution(path: Path, resolution: CampusResolution) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(resolution_record(resolution), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def approve_flagged_campus_polygon(
    root: Path,
    school_id: str,
    *,
    review_note: str,
    reviewer_role: str = "user manual campus review",
    reviewed_at: str | None = None,
) -> Path:
    """Record an explicit human approval of a resolver-proposed campus polygon."""
    root = root.resolve()
    resolution_path = root / "data" / "campus_resolutions" / f"{school_id}.json"
    try:
        record = json.loads(resolution_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CampusResolutionError(f"invalid campus resolution: {error}") from error
    if record.get("school_id") != school_id:
        raise CampusResolutionError("campus resolution school_id mismatch")
    if record.get("input_pair_frozen"):
        raise CampusResolutionError("campus inputs are already frozen")
    if record.get("requires_human_review") is not True:
        raise CampusResolutionError("campus resolution is not awaiting human review")
    geometry = record.get("geometry")
    bbox = record.get("bbox_wgs84")
    source_element = str(record.get("source_element") or "")
    if not isinstance(geometry, list) or len(geometry) < 4:
        raise CampusResolutionError("manual approval requires a proposed polygon")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise CampusResolutionError("manual approval requires a polygon bounding box")
    if "/way/" not in source_element and "/relation/" not in source_element:
        raise CampusResolutionError("manual approval requires an OSM way or relation")
    if not review_note.strip():
        raise CampusResolutionError("manual approval requires a non-empty review note")

    review_date = reviewed_at or date.today().isoformat()
    original_status = str(record.get("status", "unknown"))
    original_reason = str(record.get("reason", "")).strip()
    approval_sentence = (
        f"User manually approved the proposed OSM campus polygon on {review_date}. "
        f"{review_note.strip()}"
    )
    record.update(
        {
            "status": "confirmed",
            "requires_human_review": False,
            "confirmation_method": "user_manual_polygon_review",
            "automatic_resolution_status_before_review": original_status,
            "manual_review": {
                "decision": "approved_proposed_polygon",
                "reviewed_at": review_date,
                "reviewer_role": reviewer_role,
                "review_note": review_note.strip(),
                "review_overlay": f"data/campus_reviews/{school_id}/context_overlay.jpg",
            },
            "reason": " ".join(part for part in (original_reason, approval_sentence) if part),
            "boundary_notes": " ".join(
                part
                for part in (str(record.get("boundary_notes", "")).strip(), approval_sentence)
                if part
            ),
            "input_pair_frozen": False,
            "scope_mode": "authoritative_polygon",
            "scope_boundary_authority": "authoritative",
            "measurement_search_scope": "inside_authoritative_polygon",
        }
    )
    temporary = resolution_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(resolution_path)
    return resolution_path


def activate_boundary_proposal_as_soft_scope(root: Path, school_id: str) -> Path:
    """Use a guarded Gemini polygon for crop guidance, never as a measurement mask."""
    root = root.resolve()
    resolution_path = root / "data" / "campus_resolutions" / f"{school_id}.json"
    proposal_path = (
        root / "data" / "campus_boundary_proposals" / "v1.0" / f"{school_id}.json"
    )
    try:
        record = json.loads(resolution_path.read_text(encoding="utf-8"))
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CampusResolutionError(f"invalid soft-scope input: {error}") from error
    if record.get("school_id") != school_id or proposal.get("school_id") != school_id:
        raise CampusResolutionError("soft-scope school_id mismatch")
    if record.get("input_pair_frozen"):
        raise CampusResolutionError("campus inputs are already frozen")
    if proposal.get("status") != "completed":
        raise CampusResolutionError("boundary proposal is not completed")
    if proposal.get("configuration_id") != "school-facilities-boundary-resolver-v1.0":
        raise CampusResolutionError("boundary proposal configuration is not approved")

    image_dir = root / "data" / "imagery" / school_id
    expected_hashes = proposal.get("input_sha256", {})
    for filename in ("context.jpg", "context.tif"):
        path = image_dir / filename
        expected = expected_hashes.get(filename) if isinstance(expected_hashes, dict) else None
        if not isinstance(expected, str) or not path.is_file():
            raise CampusResolutionError(f"soft-scope proposal lacks frozen {filename} provenance")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise CampusResolutionError(f"soft-scope {filename} hash does not match proposal")

    geometry = proposal.get("geometry_wgs84_lat_lon")
    bbox = proposal.get("bbox_wgs84")
    if not isinstance(geometry, list) or len(geometry) < 4:
        raise CampusResolutionError("soft-scope proposal lacks a valid polygon")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise CampusResolutionError("soft-scope proposal lacks a valid bounding box")
    try:
        numeric_bbox = tuple(float(value) for value in bbox)
        numeric_geometry = [[float(point[0]), float(point[1])] for point in geometry]
    except (IndexError, TypeError, ValueError) as error:
        raise CampusResolutionError("soft-scope proposal geometry is invalid") from error
    selected_extent, unclamped_extent, clipped = soft_detail_extent_plan(numeric_bbox)
    if clipped:
        raise CampusResolutionError(
            "soft boundary plus safety buffer exceeds the single-image detail limit"
        )

    original_status = str(record.get("status", "unknown"))
    original_reason = str(record.get("reason", "")).strip()
    guidance_note = (
        "Gemini boundary proposal is used only to center and size a generously buffered crop; "
        "it is not an authoritative campus boundary or measurement mask. The facility model "
        "must inspect the entire detail image and flag ownership or perimeter ambiguity."
    )
    record.update(
        {
            "status": "confirmed",
            "requires_human_review": False,
            "confirmation_method": "automated_gemini_soft_scope",
            "automatic_resolution_status_before_soft_scope": original_status,
            "scope_mode": "soft_boundary",
            "scope_boundary_authority": "soft_guidance",
            "measurement_search_scope": "entire_detail_image",
            "boundary_status": "approximate_non_authoritative",
            "original_resolver_geometry": record.get("geometry", []),
            "original_resolver_bbox_wgs84": record.get("bbox_wgs84"),
            "geometry": numeric_geometry,
            "bbox_wgs84": list(numeric_bbox),
            "soft_boundary_geometry": numeric_geometry,
            "soft_boundary_bbox_wgs84": list(numeric_bbox),
            "resolved_center": {
                "latitude": (numeric_bbox[1] + numeric_bbox[3]) / 2,
                "longitude": (numeric_bbox[0] + numeric_bbox[2]) / 2,
            },
            "recommended_detail_extent_m": selected_extent,
            "unclamped_detail_extent_m": unclamped_extent,
            "detail_extent_clipped_at_maximum": False,
            "soft_scope_safety_buffer_each_side_m": 150,
            "soft_boundary_proposal": {
                "path": str(proposal_path.relative_to(root)).replace("\\", "/"),
                "sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
                "configuration_id": proposal["configuration_id"],
                "model": proposal.get("model"),
                "request_fingerprint_sha256": proposal.get("request_fingerprint_sha256"),
                "candidate_quality_gate_passed": proposal.get(
                    "deterministic_guard", {}
                ).get("candidate_quality_gate_passed"),
            },
            "reason": " ".join(part for part in (original_reason, guidance_note) if part),
            "boundary_notes": " ".join(
                part
                for part in (str(record.get("boundary_notes", "")).strip(), guidance_note)
                if part
            ),
            "input_pair_frozen": False,
        }
    )
    temporary = resolution_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(resolution_path)
    return resolution_path


def activate_center_only_scope(root: Path, school_id: str) -> Path:
    """Freeze a verified public/CCD center when no usable polygon can be trusted."""
    root = root.resolve()
    resolution_path = root / "data" / "campus_resolutions" / f"{school_id}.json"
    try:
        record = json.loads(resolution_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CampusResolutionError(f"invalid center-only campus resolution: {error}") from error
    if record.get("school_id") != school_id:
        raise CampusResolutionError("center-only school_id mismatch")
    if record.get("input_pair_frozen"):
        raise CampusResolutionError("campus inputs are already frozen")
    center = record.get("resolved_center")
    if not isinstance(center, dict):
        raise CampusResolutionError("center-only scope requires a resolved center")
    try:
        latitude = float(center["latitude"])
        longitude = float(center["longitude"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampusResolutionError("center-only coordinates are invalid") from error
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise CampusResolutionError("center-only coordinates are outside WGS84 limits")

    image_dir = root / "data" / "imagery" / school_id
    context_path = image_dir / "context.jpg"
    context_tif = image_dir / "context.tif"
    if not context_path.is_file() or not context_tif.is_file():
        raise CampusResolutionError("center-only activation requires frozen context imagery")
    rejected_path = (
        root
        / "data"
        / "campus_boundary_proposals"
        / "rejected"
        / "v1.0"
        / f"{school_id}.json"
    )
    rejected_reference: dict[str, Any] | None = None
    if rejected_path.is_file():
        rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
        input_hashes = rejected.get("input_sha256", {})
        if (
            not isinstance(input_hashes, dict)
            or input_hashes.get("context.jpg")
            != hashlib.sha256(context_path.read_bytes()).hexdigest()
            or input_hashes.get("context.tif")
            != hashlib.sha256(context_tif.read_bytes()).hexdigest()
        ):
            raise CampusResolutionError("rejected boundary attempt does not match context imagery")
        rejected_reference = {
            "path": str(rejected_path.relative_to(root)).replace("\\", "/"),
            "sha256": hashlib.sha256(rejected_path.read_bytes()).hexdigest(),
            "configuration_id": rejected.get("configuration_id"),
            "model": rejected.get("model"),
            "validation_error": rejected.get("validation_error"),
        }

    original_status = str(record.get("status", "unknown"))
    original_reason = str(record.get("reason", "")).strip()
    guidance_note = (
        "No trustworthy polygon was available. The stored public/CCD center controls an "
        "800 m crop; the facility model must search the entire detail image and flag all "
        "ownership or perimeter ambiguity."
    )
    record.update(
        {
            "status": "confirmed",
            "requires_human_review": False,
            "confirmation_method": "automatic_center_only_fallback",
            "automatic_resolution_status_before_center_only": original_status,
            "scope_mode": "center_only",
            "scope_boundary_authority": "none",
            "measurement_search_scope": "entire_detail_image",
            "boundary_status": "unavailable_non_authoritative",
            "geometry": [],
            "bbox_wgs84": None,
            "recommended_detail_extent_m": 800,
            "unclamped_detail_extent_m": 800,
            "detail_extent_clipped_at_maximum": False,
            "failed_boundary_attempt": rejected_reference,
            "reason": " ".join(part for part in (original_reason, guidance_note) if part),
            "boundary_notes": " ".join(
                part
                for part in (str(record.get("boundary_notes", "")).strip(), guidance_note)
                if part
            ),
            "input_pair_frozen": False,
        }
    )
    temporary = resolution_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(resolution_path)
    return resolution_path


def freeze_automatic_vlm_inputs(root: Path, school_id: str) -> Path:
    root = root.resolve()
    resolution_path = root / "data" / "campus_resolutions" / f"{school_id}.json"
    try:
        record = json.loads(resolution_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CampusResolutionError(f"invalid automatic campus resolution: {error}") from error
    if record.get("school_id") != school_id:
        raise CampusResolutionError("automatic campus resolution school_id mismatch")
    if record.get("status") != "confirmed" or record.get("requires_human_review") is not False:
        raise CampusResolutionError("only a confirmed automatic campus match can freeze inputs")

    image_dir = root / "data" / "imagery" / school_id
    sidecars: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    paths: list[str] = []
    for role in ("context", "detail"):
        image_path = image_dir / f"{role}.jpg"
        sidecar_path = image_dir / f"{role}.json"
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise CampusResolutionError(f"invalid {role} provenance: {error}") from error
        if sidecar.get("school_id") != school_id or sidecar.get("product") != role:
            raise CampusResolutionError(f"{role} provenance does not match the school/product")
        if float(sidecar.get("target_coverage_fraction", 0)) < 0.995:
            raise CampusResolutionError(f"{role} coverage is below 99.5 percent")
        if not image_path.is_file():
            raise CampusResolutionError(f"missing {role} JPEG: {image_path}")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes[f"{role}.jpg"] = digest
        paths.append(f"data/imagery/{school_id}/{role}.jpg")
        sidecars[role] = sidecar
    if sidecars["context"].get("capture_datetime_or_vintage") != sidecars["detail"].get(
        "capture_datetime_or_vintage"
    ):
        raise CampusResolutionError("context and detail imagery do not have the same vintage")
    scope_mode = str(record.get("scope_mode") or "authoritative_polygon")
    soft_scope = scope_mode == "soft_boundary"
    try:
        imagery_config = json.loads((root / "config" / "imagery.json").read_text(encoding="utf-8"))
        bbox_value = record.get("bbox_wgs84")
        bbox = (
            tuple(float(value) for value in bbox_value)
            if isinstance(bbox_value, list) and len(bbox_value) == 4
            else None
        )
        if scope_mode in {"authoritative_polygon", "soft_boundary"} and bbox is None:
            raise ValueError("polygon scope requires a four-value bbox")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise CampusResolutionError(f"cannot verify adaptive detail extent: {error}") from error
    selected_extent, unclamped_extent, clipped = extent_plan_for_scope(scope_mode, bbox)
    if clipped:
        raise CampusResolutionError(
            "buffered campus exceeds the 1,200 m single-image limit and needs tiled review"
        )
    detail_sidecar = sidecars["detail"]
    requested_extent = detail_sidecar.get("requested_extent_m", {})
    compatible_ids = {
        imagery_config.get("configuration_id"),
        *imagery_config.get("compatible_configuration_ids", []),
    }
    if detail_sidecar.get("configuration_id") not in compatible_ids:
        raise CampusResolutionError(
            "detail imagery was not generated by the active adaptive configuration"
        )
    if soft_scope and detail_sidecar.get("configuration_id") != imagery_config.get(
        "configuration_id"
    ):
        raise CampusResolutionError(
            "soft-scope detail imagery requires the active Version 1.6 configuration"
        )
    if not isinstance(requested_extent, dict) or any(
        not isinstance(requested_extent.get(axis), (int, float))
        or abs(float(requested_extent[axis]) - selected_extent) > 1e-9
        for axis in ("width", "height")
    ):
        raise CampusResolutionError(
            "detail imagery extent does not match the resolved campus boundary"
        )

    manually_confirmed = record.get("confirmation_method") == "user_manual_polygon_review"
    record.update(
        {
            "recommended_detail_extent_m": selected_extent,
            "unclamped_detail_extent_m": unclamped_extent,
            "detail_extent_clipped_at_maximum": False,
            "input_pair_frozen": True,
            "input_pair_freeze_method": (
                "manual polygon approval plus provenance and hash checks"
                if manually_confirmed
                else (
                    "soft-boundary centering plus safety buffer, provenance, and hash checks"
                    if soft_scope
                    else "automatic provenance and hash checks"
                )
            ),
            "approved_gemini_inputs": paths,
            "approved_gemini_input_sha256": hashes,
            "approved_supporting_artifact_sha256": {
                "detail.tif": hashlib.sha256(
                    (image_dir / "detail.tif").read_bytes()
                ).hexdigest()
            },
            "public_source_and_non_sensitive_confirmed": True,
        }
    )
    temporary = resolution_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(resolution_path)
    return resolution_path
