from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

import requests
from jsonschema import Draft202012Validator
from PIL import Image

from .credentials import CredentialError, load_api_key
from .schema import read_csv


class StreetViewError(RuntimeError):
    """Base error for the guarded Street View workflow."""


class StreetViewConfigurationError(StreetViewError):
    pass


class StreetViewBudgetError(StreetViewError):
    pass


class StreetViewProviderError(StreetViewError):
    pass


class HttpGetter(Protocol):
    def __call__(
        self, url: str, *, params: Mapping[str, Any], timeout: float
    ) -> Any: ...


class HttpPoster(Protocol):
    def __call__(
        self,
        url: str,
        *,
        data: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Any: ...


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise StreetViewConfigurationError(f"missing required file: {path}") from error
    except json.JSONDecodeError as error:
        raise StreetViewConfigurationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise StreetViewConfigurationError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise StreetViewConfigurationError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise StreetViewConfigurationError(f"invalid UTC timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        raise StreetViewConfigurationError(f"timestamp lacks timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def load_streetview_config(root: Path) -> dict[str, Any]:
    config = _read_object(root.resolve() / "config" / "streetview_v1_11.json")
    if config.get("configuration_id") != "school-facilities-streetview-v1.11":
        raise StreetViewConfigurationError("unexpected Street View configuration ID")
    budget = config.get("budget")
    network = config.get("network")
    safeguards = config.get("safeguards")
    if not isinstance(budget, dict) or not isinstance(network, dict) or not isinstance(safeguards, dict):
        raise StreetViewConfigurationError("Street View safeguards are incomplete")
    required_budget = {
        "paid_budget_usd": 0,
        "billable_image_retries": 0,
        "allow_sku_fallback": False,
    }
    for name, expected in required_budget.items():
        if budget.get(name) != expected:
            raise StreetViewConfigurationError(f"{name} must remain {expected!r}")
    if network.get("image_retries") != 0:
        raise StreetViewConfigurationError("billable image retries must remain disabled")
    if safeguards.get("never_overwrite_v1_10") is not True:
        raise StreetViewConfigurationError("V1.11 must never overwrite V1.10")
    if safeguards.get("require_zero_paid_budget_argument") is not True:
        raise StreetViewConfigurationError("zero-paid-budget live gate is missing")
    if int(budget.get("free_cap_reserve", -1)) >= int(budget.get("monthly_free_cap", 0)):
        raise StreetViewConfigurationError("free-cap reserve must be below the monthly cap")
    return config


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    term = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(term))


def _bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta = math.radians(lon2 - lon1)
    y = math.sin(delta) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta)
    return round((math.degrees(math.atan2(y, x)) + 360) % 360, 3)


def _offset_degrees(latitude: float, metres: float) -> tuple[float, float]:
    lat_delta = metres / 111_320.0
    lon_delta = metres / max(1.0, 111_320.0 * math.cos(math.radians(latitude)))
    return lat_delta, lon_delta


def _safe_school_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise StreetViewConfigurationError(f"unsafe school_id: {value!r}")
    return value


def _campus_record(root: Path, school_id: str) -> dict[str, Any]:
    school_id = _safe_school_id(school_id)
    path = root / "data" / "campus_resolutions" / f"{school_id}.json"
    record = _read_object(path)
    if record.get("school_id") != school_id:
        raise StreetViewConfigurationError(f"campus record mismatch for {school_id}")
    if record.get("status") not in {"confirmed", "probable"}:
        raise StreetViewConfigurationError(f"campus is not resolved for {school_id}")
    center = record.get("resolved_center")
    if not isinstance(center, dict):
        raise StreetViewConfigurationError(f"resolved center is missing for {school_id}")
    return record


def _campus_bbox(record: Mapping[str, Any], buffer_m: float) -> list[float]:
    center = record["resolved_center"]
    latitude = float(center["latitude"])
    longitude = float(center["longitude"])
    raw_bbox = record.get("bbox_wgs84")
    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        west, south, east, north = (float(item) for item in raw_bbox)
    else:
        lat_delta, lon_delta = _offset_degrees(latitude, 300)
        west, south, east, north = (
            longitude - lon_delta,
            latitude - lat_delta,
            longitude + lon_delta,
            latitude + lat_delta,
        )
    lat_delta, lon_delta = _offset_degrees(latitude, buffer_m)
    return [west - lon_delta, south - lat_delta, east + lon_delta, north + lat_delta]


def _fallback_probe_points(bbox: list[float]) -> list[dict[str, float]]:
    west, south, east, north = bbox
    mid_lat = (south + north) / 2
    mid_lon = (west + east) / 2
    points = [
        (north, west), (north, mid_lon), (north, east),
        (mid_lat, east), (south, east), (south, mid_lon),
        (south, west), (mid_lat, west),
    ]
    return [
        {"latitude": round(lat, 7), "longitude": round(lon, 7)}
        for lat, lon in points
    ]


def _distance_to_bbox_m(latitude: float, longitude: float, bbox: list[float]) -> float:
    west, south, east, north = bbox
    nearest_latitude = min(max(latitude, south), north)
    nearest_longitude = min(max(longitude, west), east)
    return _haversine_m(latitude, longitude, nearest_latitude, nearest_longitude)


def create_probe_plan(
    root: Path,
    *,
    school_ids: Iterable[str] | None = None,
    schools_path: Path | None = None,
    output_path: Path | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> Path:
    root = root.resolve()
    config = load_streetview_config(root)
    schools = read_csv(schools_path or root / "schools_sample.csv")
    selected = set(school_ids or [row["school_id"] for row in schools])
    unknown = selected - {row["school_id"] for row in schools}
    if unknown:
        raise StreetViewConfigurationError(f"unknown school IDs: {', '.join(sorted(unknown))}")
    entries = []
    for school in schools:
        if school["school_id"] not in selected:
            continue
        record = _campus_record(root, school["school_id"])
        center = record["resolved_center"]
        campus_bbox = _campus_bbox(record, 0)
        bbox = _campus_bbox(record, float(config["discovery"]["road_search_buffer_m"]))
        entries.append(
            {
                "school_id": school["school_id"],
                "school_name": school["school_name"],
                "campus_center": {
                    "latitude": float(center["latitude"]),
                    "longitude": float(center["longitude"]),
                },
                "road_query_bbox_wgs84": bbox,
                "campus_bbox_wgs84": campus_bbox,
                "fallback_probe_points": _fallback_probe_points(bbox),
                "campus_resolution_source": f"data/campus_resolutions/{school['school_id']}.json",
            }
        )
    if not entries:
        raise StreetViewConfigurationError("probe plan contains no schools")
    output = output_path or root / config["storage"]["manifest_directory"] / "probe_plan.json"
    _atomic_json(
        output,
        {
            "schema_version": "1.0",
            "kind": "streetview_probe_plan",
            "configuration_id": config["configuration_id"],
            "created_at_utc": _utc_text(now()),
            "network_requests_made": False,
            "schools": entries,
        },
    )
    return output


def _default_get(url: str, *, params: Mapping[str, Any], timeout: float) -> Any:
    try:
        return requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as error:
        # requests exceptions may embed the fully prepared URL, including the
        # API key query parameter. Never propagate or log their string form.
        raise StreetViewProviderError(
            f"network request to {url.split('?', 1)[0]} failed ({type(error).__name__}); "
            "request parameters were redacted"
        ) from None


def _default_post(
    url: str,
    *,
    data: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> Any:
    try:
        return requests.post(url, data=data, headers=headers, timeout=timeout)
    except requests.RequestException as error:
        raise StreetViewProviderError(
            f"network request to {url.split('?', 1)[0]} failed ({type(error).__name__}); "
            "request parameters were redacted"
        ) from None


def _response_json(response: Any) -> dict[str, Any]:
    try:
        value = response.json()
    except Exception as error:
        raise StreetViewProviderError("provider returned invalid JSON") from error
    if not isinstance(value, dict):
        raise StreetViewProviderError("provider JSON root is not an object")
    return value


def _safe_provider_message(payload: Mapping[str, Any], api_key: str) -> str:
    message = str(payload.get("error_message") or "").strip()
    if api_key:
        message = message.replace(api_key, "[REDACTED_API_KEY]")
    message = re.sub(r"AIza[A-Za-z0-9_-]{20,}", "[REDACTED_API_KEY]", message)
    return " ".join(message.split())[:500]


def fetch_osm_road_points(
    bbox: list[float],
    *,
    endpoint: str,
    timeout: float,
    poster: HttpPoster = _default_post,
) -> list[dict[str, float]]:
    west, south, east, north = bbox
    query = (
        "[out:json][timeout:25];"
        f"way[\"highway\"]({south:.7f},{west:.7f},{north:.7f},{east:.7f});"
        "out geom;"
    )
    response = poster(
        endpoint,
        data={"data": query},
        headers={"User-Agent": "school-facilities-research/0.1"},
        timeout=timeout,
    )
    if int(getattr(response, "status_code", 0)) != 200:
        raise StreetViewProviderError(
            f"OpenStreetMap road query failed with status {getattr(response, 'status_code', 'unknown')}"
        )
    payload = _response_json(response)
    points: set[tuple[float, float]] = set()
    for element in payload.get("elements", []):
        if not isinstance(element, dict):
            continue
        for point in element.get("geometry", []):
            if not isinstance(point, dict):
                continue
            try:
                pair = (round(float(point["lat"]), 7), round(float(point["lon"]), 7))
            except (KeyError, TypeError, ValueError):
                continue
            points.add(pair)
    return [{"latitude": lat, "longitude": lon} for lat, lon in sorted(points)]


def _sample_road_points(
    points: list[dict[str, float]],
    center: Mapping[str, float],
    maximum: int,
) -> list[dict[str, float]]:
    ranked = sorted(
        points,
        key=lambda item: (
            _haversine_m(
                float(center["latitude"]), float(center["longitude"]),
                float(item["latitude"]), float(item["longitude"]),
            ),
            float(item["latitude"]), float(item["longitude"]),
        ),
    )
    if len(ranked) <= maximum:
        return ranked
    # Retain angular diversity instead of taking only the closest road segment.
    buckets: dict[int, list[dict[str, float]]] = {index: [] for index in range(8)}
    for point in ranked:
        bearing = _bearing_degrees(
            float(center["latitude"]), float(center["longitude"]),
            float(point["latitude"]), float(point["longitude"]),
        )
        buckets[int((bearing + 22.5) // 45) % 8].append(point)
    selected: list[dict[str, float]] = []
    while len(selected) < maximum and any(buckets.values()):
        for index in range(8):
            if buckets[index] and len(selected) < maximum:
                selected.append(buckets[index].pop(0))
    return selected


def _metadata(
    point: Mapping[str, float],
    *,
    config: Mapping[str, Any],
    api_key: str,
    getter: HttpGetter,
) -> dict[str, Any] | None:
    response = getter(
        str(config["metadata_endpoint"]),
        params={
            "location": f"{float(point['latitude']):.7f},{float(point['longitude']):.7f}",
            "radius": int(config["discovery"]["metadata_radius_m"]),
            "source": config["discovery"]["source"],
            "key": api_key,
        },
        timeout=float(config["network"]["timeout_seconds"]),
    )
    if int(getattr(response, "status_code", 0)) != 200:
        raise StreetViewProviderError(
            f"Street View metadata failed with status {getattr(response, 'status_code', 'unknown')}"
        )
    payload = _response_json(response)
    status = payload.get("status")
    if status in {"ZERO_RESULTS", "NOT_FOUND"}:
        return None
    if status != "OK":
        detail = _safe_provider_message(payload, api_key)
        suffix = f": {detail}" if detail else ""
        raise StreetViewProviderError(
            f"Street View metadata status was {status!r}{suffix}"
        )
    location = payload.get("location")
    if not isinstance(location, dict) or not payload.get("pano_id"):
        raise StreetViewProviderError("Street View metadata omitted panorama identity/location")
    return {
        "panorama_id": str(payload["pano_id"]),
        "latitude": float(location["lat"]),
        "longitude": float(location["lng"]),
        "capture_vintage": str(payload.get("date") or "unknown"),
        "copyright": str(payload.get("copyright") or ""),
        "metadata_status": "OK",
    }


def _select_panoramas(
    panoramas: list[dict[str, Any]], center: Mapping[str, float], maximum: int
) -> list[dict[str, Any]]:
    unique = {item["panorama_id"]: item for item in panoramas}
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            _haversine_m(
                float(center["latitude"]), float(center["longitude"]),
                float(item["latitude"]), float(item["longitude"]),
            ),
            item["panorama_id"],
        ),
    )
    sectors: dict[int, list[dict[str, Any]]] = {index: [] for index in range(4)}
    for item in ranked:
        bearing = _bearing_degrees(
            float(center["latitude"]), float(center["longitude"]),
            float(item["latitude"]), float(item["longitude"]),
        )
        sectors[int((bearing + 45) // 90) % 4].append(item)
    selected = [sectors[index].pop(0) for index in range(4) if sectors[index]]
    for item in ranked:
        if len(selected) >= maximum:
            break
        if item not in selected:
            selected.append(item)
    return selected[:maximum]


def _is_google_owned_panorama(panorama: Mapping[str, Any]) -> bool:
    return "google" in str(panorama.get("copyright") or "").casefold()


def _request_fingerprint(request: Mapping[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def probe_metadata(
    root: Path,
    *,
    plan_path: Path,
    output_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
    getter: HttpGetter = _default_get,
    road_poster: HttpPoster = _default_post,
    now: Callable[[], datetime] = _utc_now,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    root = root.resolve()
    config = load_streetview_config(root)
    plan = _read_object(plan_path)
    if plan.get("kind") != "streetview_probe_plan" or plan.get("configuration_id") != config["configuration_id"]:
        raise StreetViewConfigurationError("probe plan is incompatible with V1.11")
    credentials = config["credentials"]
    try:
        api_key, _ = load_api_key(
            environment_variable=credentials["api_key_environment_variable"],
            secrets_file=root / credentials["local_secrets_file"],
            environment=environment,
        )
    except CredentialError as error:
        raise StreetViewConfigurationError(str(error).replace("Gemini key", "Street View key")) from error

    school_manifests = []
    previous_metadata_request_at: datetime | None = None
    for school in plan.get("schools", []):
        center = school["campus_center"]
        road_query_error = None
        try:
            road_points = fetch_osm_road_points(
                school["road_query_bbox_wgs84"],
                endpoint=config["osm_overpass_endpoint"],
                timeout=float(config["network"]["timeout_seconds"]),
                poster=road_poster,
            )
        except StreetViewProviderError as error:
            road_points = []
            road_query_error = str(error)
        if not road_points:
            road_points = list(school["fallback_probe_points"])
            point_source = "campus_bbox_fallback"
        else:
            point_source = "openstreetmap_roads"
        sampled = _sample_road_points(
            road_points,
            center,
            int(config["discovery"]["maximum_road_probe_points_per_school"]),
        )
        panoramas = []
        metadata_probe_count = 0
        maximum_distance = float(
            config["discovery"]["maximum_panorama_distance_from_campus_bbox_m"]
        )
        minimum_before_stop = int(
            config["discovery"].get("minimum_metadata_probes_before_early_stop", len(sampled))
        )
        desired_unique = int(config["discovery"]["maximum_selected_panoramas_per_school"])
        for point in sampled:
            if previous_metadata_request_at is not None:
                elapsed = (now() - previous_metadata_request_at).total_seconds()
                remaining = float(
                    config["network"]["minimum_seconds_between_metadata_requests"]
                ) - elapsed
                if remaining > 0:
                    sleep(remaining)
            result = _metadata(
                point, config=config, api_key=api_key, getter=getter
            )
            metadata_probe_count += 1
            previous_metadata_request_at = now()
            if result is not None:
                distance_to_campus = _distance_to_bbox_m(
                    float(result["latitude"]),
                    float(result["longitude"]),
                    list(school["campus_bbox_wgs84"]),
                )
                result["distance_to_campus_bbox_m"] = round(distance_to_campus, 1)
                panoramas.append(result)
            if metadata_probe_count >= minimum_before_stop:
                eligible_ids = {
                    item["panorama_id"]
                    for item in panoramas
                    if float(item["distance_to_campus_bbox_m"]) <= maximum_distance
                    and (
                        not bool(config["discovery"].get("require_google_owned_panorama", False))
                        or _is_google_owned_panorama(item)
                    )
                }
                if len(eligible_ids) >= desired_unique:
                    break
        proximity_rejected = [
            item
            for item in panoramas
            if float(item["distance_to_campus_bbox_m"]) > maximum_distance
        ]
        eligible_panoramas = [
            item
            for item in panoramas
            if float(item["distance_to_campus_bbox_m"]) <= maximum_distance
        ]
        contributed_rejected = []
        if bool(config["discovery"].get("require_google_owned_panorama", False)):
            contributed_rejected = [
                item for item in eligible_panoramas
                if not _is_google_owned_panorama(item)
            ]
            eligible_panoramas = [
                item for item in eligible_panoramas
                if _is_google_owned_panorama(item)
            ]
        selected = _select_panoramas(
            eligible_panoramas,
            center,
            int(config["discovery"]["maximum_selected_panoramas_per_school"]),
        )
        requests_out = []
        for panorama in selected:
            base_heading = _bearing_degrees(
                panorama["latitude"], panorama["longitude"],
                float(center["latitude"]), float(center["longitude"]),
            )
            for offset in config["images"]["heading_offsets_degrees"]:
                request = {
                    "school_id": school["school_id"],
                    "panorama_id": panorama["panorama_id"],
                    "heading": round((base_heading + float(offset)) % 360, 3),
                    "pitch": float(config["images"]["pitch_degrees"]),
                    "field_of_view": float(config["images"]["field_of_view_degrees"]),
                    "size": list(config["images"]["size"]),
                    "source": config["discovery"]["source"],
                }
                fingerprint = _request_fingerprint(request)
                requests_out.append(
                    {
                        **request,
                        "request_fingerprint_sha256": fingerprint,
                        "image_id": f"sv-{fingerprint[:12]}",
                        "capture_vintage": panorama["capture_vintage"],
                        "copyright": panorama["copyright"],
                        "panorama_location": {
                            "latitude": panorama["latitude"],
                            "longitude": panorama["longitude"],
                        },
                        "distance_to_campus_bbox_m": panorama[
                            "distance_to_campus_bbox_m"
                        ],
                    }
                )
        maximum = int(config["budget"]["per_school_image_cap"])
        if len(requests_out) > maximum:
            raise StreetViewBudgetError(f"generated more than {maximum} views for {school['school_id']}")
        school_manifests.append(
            {
                "school_id": school["school_id"],
                "school_name": school["school_name"],
                "campus_center": center,
                "road_point_source": point_source,
                "road_query_error": road_query_error,
                "metadata_probe_count": metadata_probe_count,
                "unique_panorama_count": len({item["panorama_id"] for item in panoramas}),
                "proximity_eligible_panorama_count": len(
                    {item["panorama_id"] for item in eligible_panoramas}
                ),
                "proximity_rejected_panorama_count": len(
                    {item["panorama_id"] for item in proximity_rejected}
                ),
                "contributed_panorama_rejected_count": len(
                    {item["panorama_id"] for item in contributed_rejected}
                ),
                "selected_panorama_count": len(selected),
                "image_requests": requests_out,
            }
        )
    total = sum(len(item["image_requests"]) for item in school_manifests)
    if total > int(config["budget"]["project_run_cap"]):
        raise StreetViewBudgetError("image manifest exceeds the V1.11 project cap")
    output = output_path or root / config["storage"]["manifest_directory"] / "image_manifest.json"
    _atomic_json(
        output,
        {
            "schema_version": "1.0",
            "kind": "streetview_image_manifest",
            "configuration_id": config["configuration_id"],
            "created_at_utc": _utc_text(now()),
            "metadata_requests_are_non_image_discovery": True,
            "billable_image_requests_made": False,
            "maximum_billable_image_requests": total,
            "schools": school_manifests,
        },
    )
    return output


@dataclass(frozen=True)
class UsageSnapshot:
    month: str
    provider_reported_image_requests: int
    recorded_at_utc: datetime
    source: str


def record_usage_snapshot(
    root: Path,
    *,
    month: str,
    used_requests: int,
    source: str,
    now: Callable[[], datetime] = _utc_now,
) -> Path:
    config = load_streetview_config(root)
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise StreetViewConfigurationError("month must use YYYY-MM")
    if used_requests < 0:
        raise StreetViewConfigurationError("used request count cannot be negative")
    if not source.strip():
        raise StreetViewConfigurationError("usage source must not be blank")
    path = root / config["storage"]["usage_snapshot_path"]
    _atomic_json(
        path,
        {
            "schema_version": "1.0",
            "month": month,
            "provider_reported_image_requests": used_requests,
            "recorded_at_utc": _utc_text(now()),
            "source": source.strip(),
        },
    )
    return path


class StreetViewLedger:
    def __init__(self, path: Path, *, now: Callable[[], datetime] = _utc_now) -> None:
        self.path = path
        self.now = now

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise StreetViewConfigurationError(f"invalid ledger line {number}: {error}") from error
            if not isinstance(value, dict):
                raise StreetViewConfigurationError(f"ledger line {number} is not an object")
            records.append(value)
        return records

    def append(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"recorded_at_utc": _utc_text(self.now()), **event}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def fingerprints(self) -> set[str]:
        return {
            str(row["request_fingerprint_sha256"])
            for row in self.records()
            if row.get("event") == "image_request_reserved" and row.get("request_fingerprint_sha256")
        }


def _usage_snapshot(root: Path, config: Mapping[str, Any]) -> UsageSnapshot:
    value = _read_object(root / config["storage"]["usage_snapshot_path"])
    return UsageSnapshot(
        month=str(value.get("month")),
        provider_reported_image_requests=int(value.get("provider_reported_image_requests", -1)),
        recorded_at_utc=_parse_utc(str(value.get("recorded_at_utc"))),
        source=str(value.get("source") or ""),
    )


def budget_status(
    root: Path,
    *,
    proposed_requests: int = 0,
    now: Callable[[], datetime] = _utc_now,
    require_current_snapshot: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_streetview_config(root)
    current = now()
    month = current.strftime("%Y-%m")
    ledger = StreetViewLedger(root / config["storage"]["ledger_path"], now=now)
    records = ledger.records()
    reservations = [row for row in records if row.get("event") == "image_request_reserved"]
    snapshot: UsageSnapshot | None
    try:
        snapshot = _usage_snapshot(root, config)
    except (StreetViewConfigurationError, FileNotFoundError):
        snapshot = None
    reasons = []
    if snapshot is None:
        if require_current_snapshot:
            reasons.append("current provider usage snapshot is missing")
        provider_used = None
        local_after_snapshot = None
        effective_used = None
    else:
        if snapshot.month != month:
            reasons.append(f"provider usage snapshot is for {snapshot.month}, not {month}")
        age = current - snapshot.recorded_at_utc
        if age < timedelta(0) or age > timedelta(
            hours=float(config["budget"]["maximum_usage_snapshot_age_hours"])
        ):
            reasons.append("provider usage snapshot is stale")
        provider_used = snapshot.provider_reported_image_requests
        local_after_snapshot = sum(
            _parse_utc(str(row["recorded_at_utc"])) > snapshot.recorded_at_utc
            for row in reservations
        )
        effective_used = provider_used + local_after_snapshot
    usable_cap = int(config["budget"]["monthly_free_cap"]) - int(
        config["budget"]["free_cap_reserve"]
    )
    project_used = len(reservations)
    if proposed_requests < 0:
        reasons.append("proposed request count cannot be negative")
    if project_used + proposed_requests > int(config["budget"]["project_run_cap"]):
        reasons.append("V1.11 project request cap would be exceeded")
    if effective_used is not None and effective_used + proposed_requests > usable_cap:
        reasons.append("monthly free allowance after reserve would be exceeded")
    return {
        "month": month,
        "provider_reported_requests": provider_used,
        "local_reservations_after_snapshot": local_after_snapshot,
        "effective_requests_used": effective_used,
        "usable_free_cap_after_reserve": usable_cap,
        "v1_11_project_requests_reserved": project_used,
        "proposed_requests": proposed_requests,
        "worst_case_paid_cost_usd": 0.0 if not reasons else None,
        "allowed": not reasons,
        "refusal_reasons": reasons,
    }


def cost_estimate(root: Path, *, schools: int, views_per_school: int) -> dict[str, Any]:
    config = load_streetview_config(root)
    if schools < 0 or views_per_school < 0:
        raise StreetViewConfigurationError("schools and views per school must be non-negative")
    total = schools * views_per_school
    remaining = total
    lower = 0
    cost = 0.0
    breakdown = []
    pricing = config["pricing_snapshot"]
    for tier in pricing["tiers"]:
        upper = tier["through_events"]
        capacity = remaining if upper is None else max(0, int(upper) - lower)
        events = min(remaining, capacity)
        tier_cost = events / 1000 * float(tier["price_per_1000"])
        breakdown.append(
            {
                "from_event": lower + 1,
                "through_event": upper,
                "events": events,
                "price_per_1000_usd": float(tier["price_per_1000"]),
                "cost_usd": round(tier_cost, 2),
            }
        )
        cost += tier_cost
        remaining -= events
        if remaining <= 0:
            break
        if upper is not None:
            lower = int(upper)
    return {
        "schools": schools,
        "views_per_school": views_per_school,
        "monthly_image_requests": total,
        "estimated_monthly_cost_usd": round(cost, 2),
        "pricing_observed_at": pricing["observed_at"],
        "pricing_source": pricing["official_source"],
        "warning": "Pricing is a dated estimate; verify the official table before reporting or spending.",
        "breakdown": breakdown,
    }


def _flatten_requests(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    flattened = []
    for school in manifest.get("schools", []):
        for request in school.get("image_requests", []):
            flattened.append({"school_name": school.get("school_name"), **request})
    return flattened


def replace_previously_reserved_views(
    root: Path, manifest_path: Path, *, output_path: Path
) -> Path:
    """Deterministically rotate duplicate headings without another metadata query."""
    root = root.resolve()
    manifest = _read_object(manifest_path)
    config = load_streetview_config(root)
    ledger = StreetViewLedger(root / config["storage"]["ledger_path"])
    reserved = ledger.fingerprints()
    used = set(reserved)
    replacements = 0
    for school in manifest.get("schools", []):
        for request in school.get("image_requests", []):
            original = str(request["request_fingerprint_sha256"])
            if original not in reserved:
                used.add(original)
                continue
            original_heading = float(request["heading"])
            for delta in (12, -12, 24, -24, 36, -36, 48, -48):
                request["heading"] = round((original_heading + delta) % 360, 3)
                fingerprint = _request_fingerprint(
                    {
                        name: request[name]
                        for name in (
                            "school_id", "panorama_id", "heading", "pitch",
                            "field_of_view", "size", "source",
                        )
                    }
                )
                if fingerprint not in used:
                    request["request_fingerprint_sha256"] = fingerprint
                    request["image_id"] = f"sv-{fingerprint[:12]}"
                    request["replaces_reserved_fingerprint_sha256"] = original
                    used.add(fingerprint)
                    replacements += 1
                    break
            else:
                raise StreetViewBudgetError(
                    f"could not create a non-duplicate view for {request['school_id']}"
                )
    manifest["duplicate_views_replaced"] = replacements
    manifest["created_at_utc"] = _utc_text(_utc_now())
    _atomic_json(output_path, manifest)
    return output_path


def preflight_fetch(root: Path, manifest_path: Path, *, now: Callable[[], datetime] = _utc_now) -> dict[str, Any]:
    config = load_streetview_config(root)
    manifest = _read_object(manifest_path)
    if manifest.get("kind") != "streetview_image_manifest" or manifest.get("configuration_id") != config["configuration_id"]:
        raise StreetViewConfigurationError("image manifest is incompatible with V1.11")
    requests_out = _flatten_requests(manifest)
    if len(requests_out) != int(manifest.get("maximum_billable_image_requests", -1)):
        raise StreetViewConfigurationError("manifest request total is inconsistent")
    counts: dict[str, int] = {}
    fingerprints = set()
    for request in requests_out:
        school_id = _safe_school_id(str(request["school_id"]))
        counts[school_id] = counts.get(school_id, 0) + 1
        fingerprint = str(request.get("request_fingerprint_sha256") or "")
        expected = _request_fingerprint(
            {
                name: request[name]
                for name in ("school_id", "panorama_id", "heading", "pitch", "field_of_view", "size", "source")
            }
        )
        if fingerprint != expected:
            raise StreetViewConfigurationError("manifest request fingerprint mismatch")
        if fingerprint in fingerprints:
            raise StreetViewConfigurationError("manifest contains duplicate billable requests")
        fingerprints.add(fingerprint)
        if not str(request.get("copyright") or "").strip():
            raise StreetViewConfigurationError("manifest request lacks required attribution")
        if (
            bool(config["discovery"].get("require_google_owned_panorama", False))
            and not _is_google_owned_panorama(request)
        ):
            raise StreetViewConfigurationError(
                "manifest request uses a contributed panorama; V1.11 requires Google-owned road imagery"
            )
    per_school_cap = int(config["budget"]["per_school_image_cap"])
    if any(count > per_school_cap for count in counts.values()):
        raise StreetViewBudgetError("manifest exceeds the per-school image cap")
    ledger = StreetViewLedger(root / config["storage"]["ledger_path"], now=now)
    duplicates = fingerprints & ledger.fingerprints()
    status = budget_status(root, proposed_requests=len(requests_out), now=now, require_current_snapshot=True)
    if duplicates:
        status["allowed"] = False
        status["refusal_reasons"].append("one or more billable request fingerprints already exist in the ledger")
        status["worst_case_paid_cost_usd"] = None
    return {
        "configuration_id": config["configuration_id"],
        "schools": sorted(counts),
        "requests_by_school": counts,
        "maximum_billable_image_requests": len(requests_out),
        **status,
    }


def _validate_image_bytes(content: bytes, expected_size: list[int]) -> None:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        handle.write(content)
        path = Path(handle.name)
    try:
        with Image.open(path) as image:
            if image.format != "JPEG" or list(image.size) != expected_size:
                raise StreetViewProviderError(
                    f"unexpected image response: format={image.format}, size={image.size}"
                )
            image.verify()
    except StreetViewProviderError:
        raise
    except Exception as error:
        raise StreetViewProviderError("Street View response is not a valid JPEG") from error
    finally:
        path.unlink(missing_ok=True)


def fetch_images(
    root: Path,
    *,
    manifest_path: Path,
    live: bool,
    max_paid_usd: float | None,
    provider_quota_confirmed: bool,
    environment: Mapping[str, str] | None = None,
    getter: HttpGetter = _default_get,
    now: Callable[[], datetime] = _utc_now,
    sleep: Callable[[float], None] = time.sleep,
) -> Path | dict[str, Any]:
    root = root.resolve()
    preflight = preflight_fetch(root, manifest_path, now=now)
    if not live:
        return preflight
    if max_paid_usd != 0:
        raise StreetViewBudgetError("live retrieval requires --max-paid-usd 0")
    if not provider_quota_confirmed:
        raise StreetViewBudgetError("live retrieval requires provider quota confirmation")
    if not preflight["allowed"]:
        raise StreetViewBudgetError("; ".join(preflight["refusal_reasons"]))
    config = load_streetview_config(root)
    credentials = config["credentials"]
    try:
        api_key, _ = load_api_key(
            environment_variable=credentials["api_key_environment_variable"],
            secrets_file=root / credentials["local_secrets_file"],
            environment=environment,
        )
    except CredentialError as error:
        raise StreetViewConfigurationError(str(error).replace("Gemini key", "Street View key")) from error
    manifest = _read_object(manifest_path)
    requests_out = _flatten_requests(manifest)
    ledger = StreetViewLedger(root / config["storage"]["ledger_path"], now=now)
    temporary_root = root / config["storage"]["temporary_directory"]
    fetched = []
    previous_request_at: datetime | None = None
    output_manifest = Path(manifest_path).with_name(Path(manifest_path).stem + "_fetched.json")

    def preserve_fetched_manifest() -> None:
        _atomic_json(
            output_manifest,
            {
                "schema_version": "1.0",
                "kind": "streetview_fetched_manifest",
                "configuration_id": config["configuration_id"],
                "source_manifest_sha256": _sha256_file(manifest_path),
                "created_at_utc": _utc_text(now()),
                "complete": len(fetched) == len(requests_out),
                "expected_image_count": len(requests_out),
                "fetched_image_count": len(fetched),
                "images": fetched,
            },
        )

    for request_index, request in enumerate(requests_out, start=1):
        status = budget_status(root, proposed_requests=1, now=now, require_current_snapshot=True)
        if not status["allowed"]:
            raise StreetViewBudgetError("circuit breaker: " + "; ".join(status["refusal_reasons"]))
        fingerprint = request["request_fingerprint_sha256"]
        if fingerprint in ledger.fingerprints():
            raise StreetViewBudgetError("circuit breaker: duplicate billable request")
        if previous_request_at is not None:
            elapsed = (now() - previous_request_at).total_seconds()
            remaining = float(config["network"]["minimum_seconds_between_image_requests"]) - elapsed
            if remaining > 0:
                sleep(remaining)
        ledger.append(
            {
                "event": "image_request_reserved",
                "configuration_id": config["configuration_id"],
                "school_id": request["school_id"],
                "image_id": request["image_id"],
                "panorama_id": request["panorama_id"],
                "heading": request["heading"],
                "pitch": request["pitch"],
                "field_of_view": request["field_of_view"],
                "request_fingerprint_sha256": fingerprint,
            }
        )
        previous_request_at = now()
        response = getter(
            str(config["image_endpoint"]),
            params={
                "pano": request["panorama_id"],
                "heading": request["heading"],
                "pitch": request["pitch"],
                "fov": request["field_of_view"],
                "size": "x".join(str(item) for item in request["size"]),
                "source": request["source"],
                "return_error_code": "true",
                "key": api_key,
            },
            timeout=float(config["network"]["timeout_seconds"]),
        )
        status_code = int(getattr(response, "status_code", 0))
        if status_code != 200:
            ledger.append(
                {
                    "event": "image_request_failed",
                    "school_id": request["school_id"],
                    "request_fingerprint_sha256": fingerprint,
                    "http_status": status_code,
                    "retry_permitted": False,
                }
            )
            raise StreetViewProviderError(
                f"circuit breaker: image request failed with status {status_code}; no retry was made"
            )
        content = bytes(response.content)
        try:
            _validate_image_bytes(content, list(config["images"]["size"]))
        except StreetViewProviderError:
            ledger.append(
                {
                    "event": "image_request_failed",
                    "school_id": request["school_id"],
                    "request_fingerprint_sha256": fingerprint,
                    "http_status": 200,
                    "failure": "invalid_image_response",
                    "retry_permitted": False,
                }
            )
            preserve_fetched_manifest()
            raise
        output = temporary_root / request["school_id"] / f"{request['image_id']}.jpg"
        if output.exists():
            raise StreetViewBudgetError(f"temporary image already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        temporary.replace(output)
        digest = _sha256_file(output)
        fetched.append(
            {
                **request,
                "temporary_image_path": str(output.relative_to(root)).replace("\\", "/"),
                "image_sha256": digest,
                "retrieved_at_utc": _utc_text(now()),
            }
        )
        ledger.append(
            {
                "event": "image_request_succeeded",
                "school_id": request["school_id"],
                "request_fingerprint_sha256": fingerprint,
                "http_status": 200,
                "image_sha256": digest,
                "attribution": request["copyright"],
                "capture_vintage": request["capture_vintage"],
                "temporary_file_deleted": False,
            }
        )
        preserve_fetched_manifest()
        print(
            f"Street View fetch {request_index}/{len(requests_out)}: "
            f"{request['school_id']} {request['image_id']} completed",
            flush=True,
        )
    return output_manifest


def validate_street_response(
    response: Mapping[str, Any],
    *,
    school_id: str,
    expected_image_ids: set[str],
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    schema = _read_object(root / "config" / "streetview_vlm_response_schema_v1_11.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(response), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.path) or "<root>"
        raise StreetViewProviderError(f"V1.11 response schema error at {location}: {first.message}")
    if response.get("school_id") != school_id:
        raise StreetViewProviderError("V1.11 response school_id mismatch")
    actual_ids = [item["image_id"] for item in response["image_observations"]]
    if set(actual_ids) != expected_image_ids or len(actual_ids) != len(set(actual_ids)):
        raise StreetViewProviderError("V1.11 response does not cover each supplied street image exactly once")
    fields = response["candidate_fields"]
    fencing = fields["perimeter_fencing"]
    minimum = float(fencing["minimum_supported_coverage"])
    maximum = float(fencing["maximum_supported_coverage"])
    if minimum > maximum:
        raise StreetViewProviderError("fencing coverage interval is reversed")
    value = fencing["value"]
    conflict = (
        (value == "full" and minimum < 0.8)
        or (value == "partial" and not (minimum >= 0.2 and maximum < 0.8))
        or (value == "none" and maximum >= 0.2)
    )
    if conflict:
        fencing["value"] = "unknown"
        fencing["suggested_confidence"] = 0.2
        fencing["review_required"] = True
        response["pipeline_review_required"] = True
        reasons = set(response["review_reasons"])
        reasons.add("deterministic_fencing_coverage_conflict")
        response["review_reasons"] = sorted(reasons)
    for observation in response["image_observations"]:
        if not observation["school_visible"] or observation["adequately_visible_boundary_fraction"] == 0:
            for feature in ("fencing", "portable_classrooms", "athletics"):
                if observation[feature]["result"] == "negative_visible_segment":
                    observation[feature]["result"] = "unknown"
                    observation[feature]["limitations"] += " Deterministic guard: inadequate visibility for a negative observation."
                    response["pipeline_review_required"] = True
                    reasons = set(response["review_reasons"])
                    reasons.add("negative_without_adequate_visibility")
                    response["review_reasons"] = sorted(reasons)
    return dict(response)


def delete_temporary_images(
    root: Path,
    fetched_manifest_path: Path,
    *,
    school_id: str | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> int:
    config = load_streetview_config(root)
    manifest = _read_object(fetched_manifest_path)
    if manifest.get("kind") != "streetview_fetched_manifest":
        raise StreetViewConfigurationError("not a fetched Street View manifest")
    temporary_root = (root / config["storage"]["temporary_directory"]).resolve()
    ledger = StreetViewLedger(root / config["storage"]["ledger_path"], now=now)
    deleted = 0
    for item in manifest.get("images", []):
        if school_id is not None and item.get("school_id") != school_id:
            continue
        path = (root / item["temporary_image_path"]).resolve()
        try:
            path.relative_to(temporary_root)
        except ValueError as error:
            raise StreetViewConfigurationError("refusing to delete a path outside the Street View temporary root") from error
        path.unlink(missing_ok=True)
        if path.exists():
            raise StreetViewError(f"temporary image could not be deleted: {path}")
        deleted += 1
        ledger.append(
            {
                "event": "temporary_image_deleted",
                "school_id": item["school_id"],
                "request_fingerprint_sha256": item["request_fingerprint_sha256"],
                "image_sha256": item["image_sha256"],
                "temporary_file_deleted": True,
            }
        )
    return deleted
