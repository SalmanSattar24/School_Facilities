from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds
from rasterio.warp import transform_bounds


class FacilityCropError(RuntimeError):
    """Raised when a facility crop cannot be reproduced or verified."""


@dataclass(frozen=True)
class FacilityCrop:
    role: str
    jpeg_path: Path
    sidecar_path: Path
    downloaded: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise FacilityCropError(f"cannot read facility-region specification {path}: {error}") from error
    if not isinstance(value, dict):
        raise FacilityCropError(f"facility-region specification must be an object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _region_fingerprint(source_hash: str, region: dict[str, Any], output_pixels: list[int]) -> str:
    payload = json.dumps(
        {"source_sha256": source_hash, "region": region, "output_pixels": output_pixels},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_valid(jpeg_path: Path, sidecar_path: Path, fingerprint: str) -> bool:
    if not jpeg_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        metadata = _read_object(sidecar_path)
        with Image.open(jpeg_path) as image:
            valid_image = image.format == "JPEG" and list(image.size) == metadata.get(
                "output_pixels"
            )
            image.verify()
    except (FacilityCropError, OSError, ValueError):
        return False
    return (
        valid_image
        and metadata.get("region_fingerprint") == fingerprint
        and metadata.get("jpeg_sha256") == _sha256(jpeg_path)
    )


def prepare_facility_crops(
    root: Path,
    school_id: str,
    *,
    overwrite: bool = False,
) -> list[FacilityCrop]:
    """Create verified model-view crops from an approved detail GeoTIFF."""
    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", school_id):
        raise FacilityCropError(f"unsafe school_id: {school_id!r}")
    spec_path = root / "data" / "reviewed" / school_id / "facility_regions.json"
    spec = _read_object(spec_path)
    if spec.get("school_id") != school_id or spec.get("review_status") != "confirmed":
        raise FacilityCropError("facility regions must be confirmed for the requested school")
    source_relative = spec.get("source_geotiff")
    if source_relative != f"data/imagery/{school_id}/detail.tif":
        raise FacilityCropError("facility crop source must be the approved detail GeoTIFF")
    source_path = root / str(source_relative)
    expected_source_hash = spec.get("source_sha256")
    actual_source_hash = _sha256(source_path)
    if expected_source_hash != actual_source_hash:
        raise FacilityCropError("facility crop source hash does not match the approved specification")
    output_pixels = spec.get("output_pixels")
    if output_pixels != [1600, 1600]:
        raise FacilityCropError("diagnostic facility crops must be 1600x1600 pixels")
    regions = spec.get("regions")
    if not isinstance(regions, list) or not regions:
        raise FacilityCropError("facility-region specification contains no regions")

    output_dir = root / "data" / "imagery" / school_id / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    products: list[FacilityCrop] = []
    with rasterio.open(source_path) as dataset:
        if dataset.count < 3 or dataset.crs is None:
            raise FacilityCropError("approved detail GeoTIFF lacks RGB bands or a CRS")
        for region in regions:
            if not isinstance(region, dict):
                raise FacilityCropError("each facility region must be an object")
            role = region.get("role")
            if not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", role):
                raise FacilityCropError(f"invalid facility crop role: {role!r}")
            raw_window = region.get("source_pixel_window")
            if (
                not isinstance(raw_window, list)
                or len(raw_window) != 4
                or not all(isinstance(value, int) for value in raw_window)
            ):
                raise FacilityCropError(f"{role} source_pixel_window must contain four integers")
            col_off, row_off, width, height = raw_window
            if min(col_off, row_off) < 0 or min(width, height) <= 0 or width != height:
                raise FacilityCropError(f"{role} crop window must be positive and square")
            if col_off + width > dataset.width or row_off + height > dataset.height:
                raise FacilityCropError(f"{role} crop window extends beyond the detail image")

            fingerprint = _region_fingerprint(actual_source_hash, region, output_pixels)
            jpeg_path = output_dir / f"{role}.jpg"
            sidecar_path = output_dir / f"{role}.json"
            if not overwrite and _cache_valid(jpeg_path, sidecar_path, fingerprint):
                products.append(FacilityCrop(role, jpeg_path, sidecar_path, downloaded=False))
                continue

            window = Window(col_off, row_off, width, height)
            pixels = dataset.read(
                indexes=(1, 2, 3),
                window=window,
                out_shape=(3, output_pixels[1], output_pixels[0]),
                resampling=Resampling.bilinear,
            )
            pixels = np.clip(pixels, 0, 255).astype(np.uint8, copy=False)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=output_dir,
                    prefix=f".{role}.",
                    suffix=".jpg.tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                Image.fromarray(np.moveaxis(pixels, 0, -1), mode="RGB").save(
                    temporary,
                    format="JPEG",
                    quality=95,
                )
                with Image.open(temporary) as image:
                    if image.format != "JPEG" or list(image.size) != output_pixels:
                        raise FacilityCropError(f"{role} JPEG failed verification")
                    image.verify()
                temporary.replace(jpeg_path)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

            projected_bounds = window_bounds(window, dataset.transform)
            bbox_wgs84 = transform_bounds(
                dataset.crs,
                "EPSG:4326",
                *projected_bounds,
                densify_pts=21,
            )
            source_gsd = max(abs(dataset.transform.a), abs(dataset.transform.e))
            metadata = {
                "schema_version": "1.0",
                "school_id": school_id,
                "role": role,
                "purpose": region.get("purpose"),
                "source_geotiff": source_relative,
                "source_sha256": actual_source_hash,
                "source_capture_vintage": spec.get("source_capture_vintage"),
                "source_pixel_window": raw_window,
                "source_ground_sample_distance_m": source_gsd,
                "output_pixels": output_pixels,
                "display_resampled": width != output_pixels[0],
                "projected_bbox": list(projected_bounds),
                "bbox_wgs84": list(bbox_wgs84),
                "target_crs": str(dataset.crs),
                "region_fingerprint": fingerprint,
                "jpeg_sha256": _sha256(jpeg_path),
                "configuration_id": "school-facilities-vlm-development-v0.2",
            }
            _atomic_json(sidecar_path, metadata)
            products.append(FacilityCrop(role, jpeg_path, sidecar_path, downloaded=True))
    return products
