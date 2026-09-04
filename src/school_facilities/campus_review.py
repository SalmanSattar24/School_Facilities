from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import rasterio
from PIL import Image, ImageDraw
from rasterio.warp import transform as transform_coordinates

from .campus import CampusResolutionError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_campus_review_overlay(root: Path, school_id: str) -> tuple[Path, Path]:
    """Render an auditable context overlay for a flagged campus resolution."""
    root = root.resolve()
    resolution_path = root / "data" / "campus_resolutions" / f"{school_id}.json"
    context_dir = root / "data" / "imagery" / school_id
    jpeg_path = context_dir / "context.jpg"
    geotiff_path = context_dir / "context.tif"
    try:
        resolution: dict[str, Any] = json.loads(
            resolution_path.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CampusResolutionError(f"invalid campus resolution: {error}") from error
    if resolution.get("school_id") != school_id:
        raise CampusResolutionError("campus resolution school_id mismatch")
    if not resolution.get("requires_human_review"):
        raise CampusResolutionError("review overlay is only for flagged campus resolutions")
    if not jpeg_path.is_file() or not geotiff_path.is_file():
        raise CampusResolutionError("context JPEG and GeoTIFF are required for review overlay")

    with rasterio.open(geotiff_path) as dataset:
        if dataset.crs is None:
            raise CampusResolutionError("context GeoTIFF has no coordinate reference system")
        inverse_transform = ~dataset.transform
        raster_width = dataset.width
        raster_height = dataset.height

        def pixel(latitude: float, longitude: float) -> tuple[float, float]:
            projected_x, projected_y = transform_coordinates(
                "EPSG:4326", dataset.crs, [longitude], [latitude]
            )
            column, row = inverse_transform @ (projected_x[0], projected_y[0])
            return float(column), float(row)

    with Image.open(jpeg_path) as opened:
        image = opened.convert("RGBA")
    if image.size != (raster_width, raster_height):
        raise CampusResolutionError("context JPEG and GeoTIFF dimensions do not match")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    geometry = resolution.get("geometry", [])
    polygon_pixels: list[tuple[float, float]] = []
    if isinstance(geometry, list):
        for point in geometry:
            if isinstance(point, list) and len(point) == 2:
                polygon_pixels.append(pixel(float(point[0]), float(point[1])))
    line_width = max(4, raster_width // 320)
    if len(polygon_pixels) >= 3:
        draw.polygon(
            polygon_pixels,
            fill=(255, 40, 40, 45),
            outline=(255, 30, 30, 255),
            width=line_width,
        )

    def marker(center: dict[str, Any], color: tuple[int, int, int, int]) -> None:
        x, y = pixel(float(center["latitude"]), float(center["longitude"]))
        radius = max(8, raster_width // 100)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=line_width,
        )
        draw.line((x - radius, y, x + radius, y), fill=color, width=line_width)
        draw.line((x, y - radius, x, y + radius), fill=color, width=line_width)

    marker(resolution["requested_ccd_coordinate"], (40, 120, 255, 255))
    marker(resolution["resolved_center"], (255, 220, 20, 255))

    label = (
        f"{resolution['school_name']} ({school_id})\n"
        f"status: {resolution['status']} | match: "
        f"{resolution.get('matched_name') or 'none'}\n"
        "red: proposed polygon | blue: supplied coordinate | yellow: resolved center"
    )
    padding = 10
    box = draw.multiline_textbbox((padding, padding), label, spacing=4)
    draw.rectangle(
        (box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding),
        fill=(0, 0, 0, 190),
    )
    draw.multiline_text((padding, padding), label, fill=(255, 255, 255, 255), spacing=4)

    output_dir = root / "data" / "campus_reviews" / school_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "context_overlay.jpg"
    temporary_image = output_dir / ".context_overlay.jpg.tmp"
    Image.alpha_composite(image, overlay).convert("RGB").save(
        temporary_image, format="JPEG", quality=92
    )
    temporary_image.replace(output_path)

    metadata = {
        "schema_version": "1.0",
        "school_id": school_id,
        "purpose": "human campus-resolution exception review only",
        "not_an_approved_vlm_input": True,
        "context_jpeg_sha256": _sha256(jpeg_path),
        "context_geotiff_sha256": _sha256(geotiff_path),
        "campus_resolution_sha256": _sha256(resolution_path),
        "overlay_legend": {
            "red": "proposed OSM campus polygon when available",
            "blue": "supplied CCD coordinate",
            "yellow": "automatic resolved center",
        },
        "source_element": resolution.get("source_element"),
    }
    metadata_path = output_dir / "context_overlay.json"
    temporary_metadata = output_dir / ".context_overlay.json.tmp"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return output_path, metadata_path


def prepare_boundary_proposal_overlay(root: Path, school_id: str) -> tuple[Path, Path]:
    """Render a Gemini boundary proposal and its included/excluded regions."""
    root = root.resolve()
    proposal_path = root / "data" / "campus_boundary_proposals" / "v1.0" / f"{school_id}.json"
    jpeg_path = root / "data" / "imagery" / school_id / "context.jpg"
    try:
        proposal: dict[str, Any] = json.loads(proposal_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise CampusResolutionError(f"invalid boundary proposal: {error}") from error
    parsed = proposal.get("parsed_output")
    if proposal.get("school_id") != school_id or not isinstance(parsed, dict):
        raise CampusResolutionError("boundary proposal does not match the school")
    if proposal.get("proposal_status") != "ready_for_human_review":
        raise CampusResolutionError("boundary proposal is not ready for review")
    if not jpeg_path.is_file():
        raise CampusResolutionError("context JPEG is required for boundary-proposal review")

    with Image.open(jpeg_path) as opened:
        image = opened.convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(4, width // 320)

    def point(value: dict[str, Any]) -> tuple[float, float]:
        return float(value["x"]) * width, float(value["y"]) * height

    polygon = parsed.get("boundary_polygon_normalized", [])
    if isinstance(polygon, list) and len(polygon) >= 4:
        pixels = [point(value) for value in polygon if isinstance(value, dict)]
        draw.polygon(
            pixels,
            fill=(255, 40, 40, 45),
            outline=(255, 30, 30, 255),
            width=line_width,
        )

    region_styles = {
        "included_regions": (30, 210, 80, 255),
        "excluded_adjacent_regions": (40, 130, 255, 255),
        "shared_or_ambiguous_regions": (255, 150, 20, 255),
    }
    for collection, color in region_styles.items():
        regions = parsed.get(collection, [])
        if not isinstance(regions, list):
            continue
        for region in regions:
            bbox = region.get("bbox") if isinstance(region, dict) else None
            if not isinstance(bbox, dict):
                continue
            draw.rectangle(
                (
                    float(bbox["x_min"]) * width,
                    float(bbox["y_min"]) * height,
                    float(bbox["x_max"]) * width,
                    float(bbox["y_max"]) * height,
                ),
                outline=color,
                width=line_width,
            )

    anchor = parsed.get("school_anchor")
    if isinstance(anchor, dict):
        x, y = point(anchor)
        radius = max(8, width // 100)
        yellow = (255, 220, 20, 255)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=yellow, width=line_width)
        draw.line((x - radius, y, x + radius, y), fill=yellow, width=line_width)
        draw.line((x, y - radius, x, y + radius), fill=yellow, width=line_width)

    guard = proposal.get("deterministic_guard", {})
    label = (
        f"Gemini boundary proposal: {school_id}\n"
        f"confidence: {parsed.get('suggested_confidence')} | candidate gate: "
        f"{guard.get('candidate_quality_gate_passed')}\n"
        "red: boundary | green: included | blue: excluded | orange: ambiguous | yellow: anchor"
    )
    padding = 10
    box = draw.multiline_textbbox((padding, padding), label, spacing=4)
    draw.rectangle(
        (box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding),
        fill=(0, 0, 0, 195),
    )
    draw.multiline_text((padding, padding), label, fill=(255, 255, 255, 255), spacing=4)

    output_dir = root / "data" / "campus_reviews" / school_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gemini_boundary_overlay.jpg"
    temporary_image = output_dir / ".gemini_boundary_overlay.jpg.tmp"
    Image.alpha_composite(image, overlay).convert("RGB").save(
        temporary_image, format="JPEG", quality=92
    )
    temporary_image.replace(output_path)
    metadata = {
        "schema_version": "1.0",
        "school_id": school_id,
        "purpose": "human review of a guarded Gemini campus-boundary proposal",
        "not_an_approved_vlm_input": True,
        "context_jpeg_sha256": _sha256(jpeg_path),
        "boundary_proposal_sha256": _sha256(proposal_path),
        "overlay_legend": {
            "red": "Gemini proposed campus boundary",
            "green": "model-included region boxes",
            "blue": "model-excluded adjacent region boxes",
            "orange": "model-shared or ownership-ambiguous region boxes",
            "yellow": "model school anchor",
        },
    }
    metadata_path = output_dir / "gemini_boundary_overlay.json"
    temporary_metadata = output_dir / ".gemini_boundary_overlay.json.tmp"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return output_path, metadata_path
