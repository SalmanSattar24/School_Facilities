from __future__ import annotations

import json
import hashlib
import math
import re
import shutil
import tempfile
from urllib.parse import urlencode
from urllib.request import urlopen
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol

import numpy as np
import planetary_computer
import rasterio
from PIL import Image
from pystac import Item
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform, transform_bounds

from .configuration import validate_configuration
from .imagery import _atomic_write_text, bounds_around


ProductName = Literal["context", "detail"]
CampusStatus = Literal["confirmed", "probable", "unresolved"]


class NAIPError(RuntimeError):
    """Base error for dated NAIP acquisition."""


class NAIPCoverageError(NAIPError):
    """Raised when no single-date item set covers the required product extent."""


class SearchLike(Protocol):
    def item_collection(self) -> Iterable[Item]: ...


class StacClientLike(Protocol):
    def search(self, **kwargs: Any) -> SearchLike: ...


@dataclass(frozen=True)
class TargetGrid:
    crs: str
    bounds: tuple[float, float, float, float]
    bbox_wgs84: tuple[float, float, float, float]
    transform: Any
    width: int
    height: int
    metres_per_pixel: float


@dataclass(frozen=True)
class NAIPProduct:
    geotiff_path: Path
    jpeg_path: Path
    sidecar_path: Path
    downloaded: bool


def _load_imagery_config(root: Path) -> dict[str, Any]:
    result = validate_configuration(root)
    if not result.ok:
        raise NAIPError("; ".join(result.errors))
    path = root / "config" / "imagery.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise NAIPError(f"cannot load imagery configuration: {error}") from error
    if not isinstance(value, dict):
        raise NAIPError("imagery configuration root must be an object")
    return value


def _utm_crs(latitude: float, longitude: float) -> str:
    if not -80 <= latitude <= 84:
        raise NAIPError("automatic UTM selection requires latitude between -80 and 84 degrees")
    if not -180 <= longitude <= 180:
        raise NAIPError("longitude must be between -180 and 180 degrees")
    zone = min(60, max(1, math.floor((longitude + 180) / 6) + 1))
    epsg = (32600 if latitude >= 0 else 32700) + zone
    return f"EPSG:{epsg}"


def target_grid(
    latitude: float,
    longitude: float,
    width_m: float,
    height_m: float,
    output_pixels: list[int],
) -> TargetGrid:
    if width_m <= 0 or height_m <= 0:
        raise NAIPError("product width and height must be positive")
    if len(output_pixels) != 2 or min(output_pixels) <= 0:
        raise NAIPError("output_pixels must contain two positive integers")
    crs = _utm_crs(latitude, longitude)
    center_x, center_y = transform("EPSG:4326", crs, [longitude], [latitude])
    bounds = (
        center_x[0] - width_m / 2,
        center_y[0] - height_m / 2,
        center_x[0] + width_m / 2,
        center_y[0] + height_m / 2,
    )
    width, height = int(output_pixels[0]), int(output_pixels[1])
    output_transform = from_bounds(*bounds, width=width, height=height)
    bbox_wgs84 = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    return TargetGrid(
        crs=crs,
        bounds=bounds,
        bbox_wgs84=bbox_wgs84,
        transform=output_transform,
        width=width,
        height=height,
        metres_per_pixel=max(width_m / width, height_m / height),
    )


def _capture_datetime(item: Item) -> datetime:
    value = item.datetime
    if value is None:
        raw = item.properties.get("datetime")
        if not isinstance(raw, str):
            raise NAIPError(f"NAIP item {item.id} has no capture datetime")
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def candidate_date_groups(items: Iterable[Item], minimum_year: int) -> list[tuple[date, list[Item]]]:
    groups: dict[date, list[Item]] = {}
    for item in items:
        captured = _capture_datetime(item)
        if captured.year < minimum_year:
            continue
        groups.setdefault(captured.date(), []).append(item)
    return [
        (capture_date, sorted(group, key=lambda item: item.id))
        for capture_date, group in sorted(groups.items(), reverse=True)
    ]


def _footprint_mask(dataset: Any, grid: TargetGrid) -> np.ndarray:
    left, bottom, right, top = transform_bounds(
        dataset.crs,
        grid.crs,
        *dataset.bounds,
        densify_pts=21,
    )
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
            [left, bottom],
        ]],
    }
    return rasterize(
        [(geometry, 1)],
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def _mosaic_group(
    items: list[Item],
    *,
    asset_key: str,
    grid: TargetGrid,
    signer: Callable[[Item], Item],
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    mosaic = np.zeros((3, grid.height, grid.width), dtype=np.uint8)
    filled = np.zeros((grid.height, grid.width), dtype=bool)
    sources: list[dict[str, Any]] = []
    for unsigned_item in items:
        if asset_key not in unsigned_item.assets:
            continue
        signed_item = signer(unsigned_item)
        asset = signed_item.assets.get(asset_key)
        if asset is None:
            continue
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_TIMEOUT="60",
            GDAL_HTTP_MAX_RETRY="3",
            GDAL_HTTP_RETRY_DELAY="1",
        ):
            with rasterio.open(asset.href) as dataset:
                if dataset.crs is None or dataset.count < 3:
                    raise NAIPError(f"NAIP asset {unsigned_item.id} lacks RGB bands or a CRS")
                footprint = _footprint_mask(dataset, grid)
                take = footprint & ~filled
                if not np.any(take):
                    continue
                warped = np.zeros_like(mosaic)
                for band in range(1, 4):
                    reproject(
                        source=rasterio.band(dataset, band),
                        destination=warped[band - 1],
                        src_transform=dataset.transform,
                        src_crs=dataset.crs,
                        src_nodata=dataset.nodata,
                        dst_transform=grid.transform,
                        dst_crs=grid.crs,
                        dst_nodata=0,
                        resampling=Resampling.bilinear,
                        init_dest_nodata=True,
                    )
                mosaic[:, take] = warped[:, take]
                filled[take] = True
                native_resolution = unsigned_item.properties.get("gsd")
                if not isinstance(native_resolution, (int, float)):
                    native_resolution = max(abs(dataset.transform.a), abs(dataset.transform.e))
                sources.append(
                    {
                        "item_id": unsigned_item.id,
                        "capture_datetime": _capture_datetime(unsigned_item)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "native_resolution_m": float(native_resolution),
                    }
                )
        if filled.all():
            break
    return mosaic, float(filled.mean()), sources


def _fetch_url_bytes(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def _mosaic_group_via_data_api(
    items: list[Item],
    *,
    asset_key: str,
    collection: str,
    grid: TargetGrid,
    endpoint: str,
    fetch_bytes: Callable[[str], bytes] = _fetch_url_bytes,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """Render exact target grids through the official Planetary Computer data API."""
    mosaic = np.zeros((3, grid.height, grid.width), dtype=np.uint8)
    filled = np.zeros((grid.height, grid.width), dtype=bool)
    sources: list[dict[str, Any]] = []
    bbox = ",".join(f"{value:.12f}" for value in grid.bounds)
    for item in items:
        if asset_key not in item.assets:
            continue
        query = urlencode(
            {
                "collection": collection,
                "item": item.id,
                "assets": asset_key,
                "asset_bidx": f"{asset_key}|1,2,3",
                "coord_crs": grid.crs,
                "dst_crs": grid.crs,
                "reproject": "bilinear",
                "return_mask": "true",
            }
        )
        url = f"{endpoint.rstrip('/')}/{bbox}/{grid.width}x{grid.height}.tif?{query}"
        try:
            payload = fetch_bytes(url)
            with rasterio.MemoryFile(payload) as memory_file:
                with memory_file.open() as dataset:
                    if (
                        dataset.count < 3
                        or dataset.width != grid.width
                        or dataset.height != grid.height
                        or str(dataset.crs) != grid.crs
                    ):
                        raise NAIPError(
                            f"Planetary data API returned an invalid grid for {item.id}"
                        )
                    rendered = dataset.read([1, 2, 3])
                    footprint = dataset.dataset_mask() > 0
        except NAIPError:
            raise
        except Exception as error:
            raise NAIPError(
                f"Planetary data API render failed for {item.id}: {error}"
            ) from error
        take = footprint & ~filled
        if not np.any(take):
            continue
        mosaic[:, take] = rendered[:, take]
        filled[take] = True
        native_resolution = item.properties.get("gsd")
        if not isinstance(native_resolution, (int, float)):
            native_resolution = 0.0
        sources.append(
            {
                "item_id": item.id,
                "capture_datetime": _capture_datetime(item)
                .isoformat()
                .replace("+00:00", "Z"),
                "native_resolution_m": float(native_resolution),
            }
        )
        if filled.all():
            break
    return mosaic, float(filled.mean()), sources


def _cache_is_valid(
    paths: tuple[Path, Path, Path],
    school_id: str,
    product: ProductName,
    configuration_id: str,
    grid: TargetGrid,
) -> bool:
    geotiff_path, jpeg_path, sidecar_path = paths
    if not all(path.is_file() for path in paths):
        return False
    try:
        with rasterio.open(geotiff_path) as dataset:
            actual_bounds = tuple(float(value) for value in dataset.bounds)
            geotiff_ok = (
                dataset.count == 3
                and dataset.crs is not None
                and dataset.width == grid.width
                and dataset.height == grid.height
                and dataset.crs.to_string() == grid.crs
                and all(
                    math.isclose(actual, expected, abs_tol=0.01)
                    for actual, expected in zip(actual_bounds, grid.bounds)
                )
            )
        with Image.open(jpeg_path) as image:
            jpeg_ok = image.format == "JPEG" and image.size == (grid.width, grid.height)
            image.verify()
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError, rasterio.errors.RasterioError):
        return False
    return (
        geotiff_ok
        and jpeg_ok
        and metadata.get("school_id") == school_id
        and metadata.get("product") == product
        and metadata.get("configuration_id") == configuration_id
        and metadata.get("output_pixels") == [grid.width, grid.height]
        and math.isclose(
            float(metadata.get("output_resolution_m", -1)),
            grid.metres_per_pixel,
            abs_tol=1e-9,
        )
    )


def _write_products(
    mosaic: np.ndarray,
    grid: TargetGrid,
    geotiff_path: Path,
    jpeg_path: Path,
    jpeg_quality: int,
) -> None:
    temporary_geotiff: Path | None = None
    temporary_jpeg: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=geotiff_path.parent,
            prefix=f".{geotiff_path.stem}.",
            suffix=".tif.tmp",
            delete=False,
        ) as handle:
            temporary_geotiff = Path(handle.name)
        with rasterio.open(
            temporary_geotiff,
            "w",
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=3,
            dtype="uint8",
            crs=grid.crs,
            transform=grid.transform,
            tiled=True,
            compress="deflate",
            predictor=2,
            photometric="RGB",
        ) as destination:
            destination.write(mosaic)
            destination.colorinterp = (
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
            )
        with rasterio.open(temporary_geotiff) as dataset:
            if dataset.count != 3 or dataset.width != grid.width or dataset.height != grid.height:
                raise NAIPError("written GeoTIFF failed verification")

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=jpeg_path.parent,
            prefix=f".{jpeg_path.stem}.",
            suffix=".jpg.tmp",
            delete=False,
        ) as handle:
            temporary_jpeg = Path(handle.name)
        Image.fromarray(np.moveaxis(mosaic, 0, -1), mode="RGB").save(
            temporary_jpeg,
            format="JPEG",
            quality=jpeg_quality,
        )
        with Image.open(temporary_jpeg) as image:
            image.verify()

        temporary_geotiff.replace(geotiff_path)
        temporary_geotiff = None
        temporary_jpeg.replace(jpeg_path)
        temporary_jpeg = None
    finally:
        if temporary_geotiff is not None:
            temporary_geotiff.unlink(missing_ok=True)
        if temporary_jpeg is not None:
            temporary_jpeg.unlink(missing_ok=True)


def _preserve_existing_product(
    paths: tuple[Path, Path, Path],
    product: ProductName,
) -> Path | None:
    """Archive an existing product before a changed grid/configuration replaces it."""
    geotiff_path, jpeg_path, sidecar_path = paths
    if not all(path.is_file() for path in paths):
        return None
    try:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NAIPError(f"cannot preserve existing {product} provenance: {error}") from error
    configuration_id = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(metadata.get("configuration_id", "unknown-config"))
    )
    digest = hashlib.sha256(jpeg_path.read_bytes()).hexdigest()[:12]
    archive_dir = geotiff_path.parent / "history" / f"{product}-{configuration_id}-{digest}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        destination = archive_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    return archive_dir


def fetch_naip_product(
    *,
    root: Path,
    school_id: str,
    school_name: str,
    product: ProductName,
    center_latitude: float,
    center_longitude: float,
    requested_latitude: float,
    requested_longitude: float,
    campus_resolution_status: CampusStatus,
    campus_resolution_notes: str,
    extent_m: float | None = None,
    overwrite: bool = False,
    stac_client: StacClientLike | None = None,
    signer: Callable[[Item], Item] = planetary_computer.sign,
    retrieved_at: Callable[[], datetime] | None = None,
) -> NAIPProduct:
    root = root.resolve()
    config = _load_imagery_config(root)
    if not school_id or not school_name.strip():
        raise NAIPError("school_id and school_name are required")
    if campus_resolution_status not in set(config["campus_resolution"]["statuses"]):
        raise NAIPError(f"invalid campus-resolution status: {campus_resolution_status}")
    if not campus_resolution_notes.strip():
        raise NAIPError("campus-resolution notes are required")
    if product == "detail" and campus_resolution_status == "unresolved":
        raise NAIPError("detail imagery requires a confirmed or probable campus center")

    product_config = config["products"][product]
    if extent_m is not None and product != "detail":
        raise NAIPError("an adaptive extent override is allowed only for the detail product")
    width_m = float(product_config["width_m"])
    height_m = float(product_config["height_m"])
    if product == "detail" and extent_m is not None:
        if not product_config.get("adaptive_from_campus_polygon"):
            raise NAIPError("the imagery configuration does not enable adaptive detail extents")
        minimum = float(product_config["minimum_extent_m"])
        maximum = float(product_config["maximum_extent_m"])
        increment = float(product_config["rounding_increment_m"])
        if not minimum <= extent_m <= maximum:
            raise NAIPError(
                f"detail extent must be between {minimum:g} and {maximum:g} metres"
            )
        if not math.isclose(extent_m / increment, round(extent_m / increment), abs_tol=1e-9):
            raise NAIPError(f"detail extent must be a multiple of {increment:g} metres")
        width_m = height_m = float(extent_m)
    grid = target_grid(
        center_latitude,
        center_longitude,
        width_m,
        height_m,
        list(product_config["output_pixels"]),
    )
    school_dir = root / config["storage"]["root"] / school_id
    school_dir.mkdir(parents=True, exist_ok=True)
    geotiff_path = school_dir / f"{product}.tif"
    jpeg_path = school_dir / f"{product}.jpg"
    sidecar_path = school_dir / f"{product}.json"
    paths = (geotiff_path, jpeg_path, sidecar_path)
    if not overwrite and _cache_is_valid(
        paths, school_id, product, str(config["configuration_id"]), grid
    ):
        return NAIPProduct(geotiff_path, jpeg_path, sidecar_path, downloaded=False)

    primary = config["primary"]
    selection = primary["selection"]
    query_bbox = bounds_around(
        center_latitude,
        center_longitude,
        max(width_m, height_m) / 2,
    )
    use_data_api = stac_client is None
    active_client = stac_client or Client.open(primary["stac_endpoint"])
    search = active_client.search(
        collections=[primary["collection"]],
        bbox=query_bbox,
        datetime=f"{selection['minimum_capture_year']}-01-01/..",
    )
    items = list(search.item_collection())
    if not items:
        raise NAIPCoverageError("NAIP STAC search returned no candidates")

    chosen_date: date | None = None
    chosen_mosaic: np.ndarray | None = None
    chosen_coverage = 0.0
    chosen_sources: list[dict[str, Any]] = []
    threshold = float(selection["minimum_target_coverage_fraction"])
    for capture_date, group in candidate_date_groups(items, int(selection["minimum_capture_year"])):
        if use_data_api:
            mosaic, coverage, sources = _mosaic_group_via_data_api(
                group,
                asset_key=primary["asset_key"],
                collection=primary["collection"],
                grid=grid,
                endpoint=primary["data_api_bbox_endpoint"],
            )
        else:
            mosaic, coverage, sources = _mosaic_group(
                group,
                asset_key=primary["asset_key"],
                grid=grid,
                signer=signer,
            )
        if coverage >= threshold:
            chosen_date = capture_date
            chosen_mosaic = mosaic
            chosen_coverage = coverage
            chosen_sources = sources
            break
    if chosen_date is None or chosen_mosaic is None:
        raise NAIPCoverageError(
            f"no single-date NAIP item set reached {threshold:.1%} target coverage"
        )

    archived_previous_product = _preserve_existing_product(paths, product)
    _write_products(
        chosen_mosaic,
        grid,
        geotiff_path,
        jpeg_path,
        int(config["products"]["jpeg_quality"]),
    )
    now = (retrieved_at or (lambda: datetime.now(timezone.utc)))()
    if now.tzinfo is None:
        raise NAIPError("retrieval clock must return a timezone-aware datetime")
    item_ids = [source["item_id"] for source in chosen_sources]
    capture_datetimes = sorted({source["capture_datetime"] for source in chosen_sources})
    capture_value = capture_datetimes[0] if len(capture_datetimes) == 1 else chosen_date.isoformat()
    stable_references = [
        f"{primary['stac_endpoint']}/collections/{primary['collection']}/items/{item_id}"
        for item_id in item_ids
    ]
    sidecar = {
        "schema_version": "1.0",
        "school_id": school_id,
        "school_name": school_name,
        "product": product,
        "source": primary["dataset"],
        "source_collection_or_service": primary["collection"],
        "acquisition_transport": (
            "Planetary Computer data API bbox render"
            if use_data_api
            else "direct raster asset"
        ),
        "stac_item_id_or_request_url": item_ids,
        "capture_datetime_or_vintage": capture_value,
        "capture_datetimes_utc": capture_datetimes,
        "retrieved_at_utc": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "native_resolution_m": [source["native_resolution_m"] for source in chosen_sources],
        "output_resolution_m": grid.metres_per_pixel,
        "requested_extent_m": {"width": width_m, "height": height_m},
        "bbox_wgs84": list(grid.bbox_wgs84),
        "bbox_projected": list(grid.bounds),
        "target_crs": grid.crs,
        "output_pixels": [grid.width, grid.height],
        "target_coverage_fraction": chosen_coverage,
        "requested_ccd_coordinate": {
            "latitude": requested_latitude,
            "longitude": requested_longitude,
        },
        "resolved_center": {
            "latitude": center_latitude,
            "longitude": center_longitude,
        },
        "campus_resolution_status": campus_resolution_status,
        "campus_resolution_notes": campus_resolution_notes,
        "asset_reference_without_expired_credentials": stable_references,
        "configuration_id": config["configuration_id"],
        "geotiff_path": str(geotiff_path.relative_to(root)),
        "jpeg_path": str(jpeg_path.relative_to(root)),
        "superseded_product_archive": (
            archived_previous_product.relative_to(root).as_posix()
            if archived_previous_product is not None
            else None
        ),
    }
    _atomic_write_text(sidecar_path, json.dumps(sidecar, indent=2) + "\n")
    return NAIPProduct(geotiff_path, jpeg_path, sidecar_path, downloaded=True)
