from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ESRI_EXPORT_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
)


def bounds_around(latitude: float, longitude: float, half_width_m: float) -> tuple[float, float, float, float]:
    if half_width_m <= 0:
        raise ValueError("half_width_m must be positive")
    latitude_delta = half_width_m / 111_320.0
    longitude_scale = 111_320.0 * math.cos(math.radians(latitude))
    if abs(longitude_scale) < 1:
        raise ValueError("coordinates are too close to a pole")
    longitude_delta = half_width_m / longitude_scale
    return (
        longitude - longitude_delta,
        latitude - latitude_delta,
        longitude + longitude_delta,
        latitude + latitude_delta,
    )


def _retry_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _valid_cached_pair(image_path: Path, sidecar_path: Path, school_id: str) -> bool:
    if not image_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        with Image.open(image_path) as image:
            image.verify()
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return metadata.get("school_id") == school_id


def _atomic_write_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def fetch_esri_image(
    *,
    school_id: str,
    school_name: str,
    latitude: float,
    longitude: float,
    output_dir: Path,
    half_width_m: float,
    pixels: int,
    timeout_seconds: float = 60,
    overwrite: bool = False,
    session: requests.Session | None = None,
) -> tuple[Path, bool]:
    if not 256 <= pixels <= 4000:
        raise ValueError("pixels must be between 256 and 4000")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{school_id}.jpg"
    sidecar_path = output_dir / f"{school_id}.json"
    if not overwrite and _valid_cached_pair(image_path, sidecar_path, school_id):
        return image_path, False

    bounds = bounds_around(latitude, longitude, half_width_m)
    params = {
        "bbox": ",".join(f"{value:.8f}" for value in bounds),
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{pixels},{pixels}",
        "format": "jpg",
        "transparent": "false",
        "f": "image",
    }
    owns_session = session is None
    active_session = session or _retry_session()
    temporary_image: Path | None = None
    try:
        response = active_session.get(ESRI_EXPORT_URL, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type.lower():
            raise RuntimeError(f"imagery service returned unexpected content type: {content_type}")

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            prefix=f".{school_id}.",
            suffix=".jpg.tmp",
            delete=False,
        ) as handle:
            handle.write(response.content)
            temporary_image = Path(handle.name)
        with Image.open(temporary_image) as image:
            image.verify()
        temporary_image.replace(image_path)
        temporary_image = None

        request_url = response.url
    finally:
        if temporary_image is not None:
            temporary_image.unlink(missing_ok=True)
        if owns_session:
            active_session.close()

    retrieved_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "school_id": school_id,
        "school_name": school_name,
        "requested_coordinate": {"latitude": latitude, "longitude": longitude},
        "bbox_wgs84": bounds,
        "half_width_m": half_width_m,
        "pixels": pixels,
        "imagery_source": "Esri World Imagery",
        "capture_vintage": None,
        "retrieved_at_utc": retrieved_at,
        "request_url": request_url,
        "note": "Capture vintage is not reliably exposed by this export; resolve it during review or mark unknown.",
    }
    _atomic_write_text(sidecar_path, json.dumps(metadata, indent=2) + "\n")
    return image_path, True
