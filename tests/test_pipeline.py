from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from PIL import Image
from pystac import Asset, Item
from rasterio.transform import from_bounds
from rasterio.io import MemoryFile

from school_facilities.calibration import observations, summarize
from school_facilities.auditor import (
    GeminiEvidenceAuditorClient,
    _auditor_safety_overrides,
    _normalize_auditor_output,
)
from school_facilities.campus import (
    CampusResolutionError,
    activate_boundary_proposal_as_soft_scope,
    activate_center_only_scope,
    approve_flagged_campus_polygon,
    detail_extent_plan,
    extent_plan_for_scope,
    soft_detail_extent_plan,
    select_campus,
)
from school_facilities.campus_review import (
    prepare_boundary_proposal_overlay,
    prepare_campus_review_overlay,
)
from school_facilities.boundary_vlm import (
    BoundaryVLMInput,
    GeminiBoundaryClient,
    build_boundary_request,
    load_boundary_bundle,
    normalize_boundary_vocabulary,
    validate_boundary_output,
)
from school_facilities.credentials import load_api_key, save_api_key
from school_facilities.facility_crops import prepare_facility_crops
from school_facilities.imagery import bounds_around, fetch_esri_image
from school_facilities.naip import (
    _mosaic_group_via_data_api,
    candidate_date_groups,
    fetch_naip_product,
    target_grid,
)
from school_facilities.configuration import validate_configuration
from school_facilities.schema import (
    CONFIDENCE_COLUMN,
    MEASUREMENT_COLUMNS,
    MEASUREMENT_FIELDS,
    measurement_template,
    read_csv,
    validate_measurements,
    validate_schools,
    write_csv,
)
from school_facilities.vlm import (
    GeminiVLMClient,
    RequestLedger,
    SchoolVLMInput,
    VLMError,
    VLMConfigurationError,
    VLMImage,
    VLMQuotaError,
    VLMResponseError,
    build_interaction_request,
    load_approved_school_input,
    _deterministic_evidence_checks,
    _normalize_provider_vocabulary,
    _response_output_directories,
    _validate_solar_inventory,
)
from school_facilities.vlm_evaluation import (
    pilot_auditor_diagnostic,
    pilot_uncertainty_diagnostic,
    raw_observations,
    summarize_raw_vlm,
)


ROOT = Path(__file__).resolve().parents[1]


def test_validation_outputs_are_routed_to_blind_quarantine(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "pilot_schools.json").write_text(
        json.dumps({"excluded_validation_school_ids": ["validation-school"]}),
        encoding="utf-8",
    )
    config = {
        "outputs": {
            "raw_directory": "data/model_outputs/final/v1.10",
            "rejected_directory": "data/model_outputs/rejected/v1.10",
        }
    }

    raw, rejected, quarantined = _response_output_directories(
        tmp_path, config, "validation-school"
    )
    ordinary_raw, ordinary_rejected, ordinary_quarantined = (
        _response_output_directories(tmp_path, config, "ordinary-school")
    )

    assert raw == tmp_path / "data/model_outputs/quarantine/v1.10/raw"
    assert rejected == tmp_path / "data/model_outputs/quarantine/v1.10/rejected"
    assert quarantined is True
    assert ordinary_raw == tmp_path / "data/model_outputs/final/v1.10"
    assert ordinary_rejected == tmp_path / "data/model_outputs/rejected/v1.10"
    assert ordinary_quarantined is False


def test_api_key_environment_override_precedes_local_secrets_file() -> None:
    key, source = load_api_key(
        environment_variable="GEMINI_API_KEY",
        secrets_file=Path("unused.env"),
        environment={"GEMINI_API_KEY": " environment-key "},
    )
    assert key == "environment-key"
    assert source == "environment"


def test_api_key_can_be_saved_and_loaded_from_local_secrets_file(tmp_path: Path) -> None:
    secrets_file = tmp_path / "secrets.local.env"
    save_api_key(
        " stored-key ",
        environment_variable="GEMINI_API_KEY",
        secrets_file=secrets_file,
    )
    key, source = load_api_key(
        environment_variable="GEMINI_API_KEY",
        secrets_file=secrets_file,
        environment={},
    )
    assert key == "stored-key"
    assert source == "local_secrets_file"


def test_automatic_campus_resolution_matches_named_polygon() -> None:
    school = {
        "school_id": "061734009378",
        "school_name": "Cerra Vista Elementary",
        "latitude": "36.829411",
        "longitude": "-121.372331",
    }
    elements = [
        {
            "type": "way",
            "id": 27420396,
            "bounds": {
                "minlat": 36.8282335,
                "minlon": -121.3735033,
                "maxlat": 36.8301477,
                "maxlon": -121.3707674,
            },
            "geometry": [
                {"lat": 36.8301477, "lon": -121.3733070},
                {"lat": 36.8300392, "lon": -121.3707674},
                {"lat": 36.8282335, "lon": -121.3708070},
                {"lat": 36.8282666, "lon": -121.3735033},
                {"lat": 36.8301477, "lon": -121.3733070},
            ],
            "tags": {"amenity": "school", "name": "Cerra Vista School"},
        },
        {
            "type": "node",
            "id": 1,
            "lat": 36.831,
            "lon": -121.375,
            "tags": {"amenity": "school", "name": "Enterprise School"},
        },
    ]

    resolution = select_campus(school, elements)

    assert resolution.status == "confirmed"
    assert resolution.requires_human_review is False
    assert resolution.matched_name == "Cerra Vista School"
    assert resolution.source_element == "https://www.openstreetmap.org/way/27420396"
    assert resolution.recommended_detail_extent_m == 400
    assert resolution.unclamped_detail_extent_m == 400
    assert resolution.detail_extent_clipped_at_maximum is False
    assert resolution.resolved_latitude == pytest.approx(36.8291906)
    assert resolution.resolved_longitude == pytest.approx(-121.37213535)


def test_automatic_campus_resolution_flags_missing_name_match() -> None:
    school = {
        "school_id": "402277001928",
        "school_name": "EMERSON ALTERNATIVE ED. (HS)",
        "latitude": "36.142973",
        "longitude": "-95.964045",
    }
    elements = [
        {
            "type": "way",
            "id": 192369209,
            "bounds": {
                "minlat": 36.142,
                "minlon": -95.965,
                "maxlat": 36.144,
                "maxlon": -95.963,
            },
            "geometry": [
                {"lat": 36.142, "lon": -95.965},
                {"lat": 36.144, "lon": -95.963},
                {"lat": 36.142, "lon": -95.965},
            ],
            "tags": {"amenity": "school", "name": "Mayo Demonstration School"},
        }
    ]

    resolution = select_campus(school, elements)

    assert resolution.status == "unresolved"
    assert resolution.requires_human_review is True


def test_adaptive_detail_extent_clamps_small_and_flags_oversized_campuses() -> None:
    tiny_bbox = (-97.5222, 35.4745, -97.5219, 35.4748)
    selected, unclamped, clipped = detail_extent_plan(tiny_bbox)
    assert selected == 250
    assert unclamped <= 250
    assert clipped is False

    huge_bbox = (-118.43, 34.05, -118.40, 34.08)
    selected, unclamped, clipped = detail_extent_plan(huge_bbox)
    assert selected == 1200
    assert unclamped > 1200
    assert clipped is True


def test_soft_boundary_extent_adds_safety_margin_and_center_only_uses_fixed_crop() -> None:
    bbox = (-82.4932778, 33.79477207, -82.49091088, 33.79787636)
    authoritative, _, _ = detail_extent_plan(bbox)
    soft, _, clipped = soft_detail_extent_plan(bbox)
    assert authoritative == 500
    assert soft == 650
    assert soft > authoritative
    assert clipped is False
    assert extent_plan_for_scope("center_only", None) == (800, 800, False)


def test_center_only_activation_preserves_failed_boundary_provenance(tmp_path: Path) -> None:
    school_id = "example-school"
    imagery_dir = tmp_path / "data" / "imagery" / school_id
    resolution_dir = tmp_path / "data" / "campus_resolutions"
    rejected_dir = tmp_path / "data" / "campus_boundary_proposals" / "rejected" / "v1.0"
    imagery_dir.mkdir(parents=True)
    resolution_dir.mkdir(parents=True)
    rejected_dir.mkdir(parents=True)
    (imagery_dir / "context.jpg").write_bytes(b"context-jpeg")
    (imagery_dir / "context.tif").write_bytes(b"context-geotiff")
    resolution_path = resolution_dir / f"{school_id}.json"
    resolution_path.write_text(
        json.dumps(
            {
                "school_id": school_id,
                "school_name": "Example School",
                "status": "unresolved",
                "requires_human_review": True,
                "resolved_center": {"latitude": 40.0, "longitude": -75.0},
                "reason": "No polygon match.",
                "boundary_notes": "No polygon match.",
            }
        ),
        encoding="utf-8",
    )
    rejected_path = rejected_dir / f"{school_id}.json"
    rejected_path.write_text(
        json.dumps(
            {
                "school_id": school_id,
                "configuration_id": "school-facilities-boundary-resolver-v1.0",
                "model": "gemini-3.5-flash-lite",
                "validation_error": "malformed polygon",
                "input_sha256": {
                    "context.jpg": hashlib.sha256(b"context-jpeg").hexdigest(),
                    "context.tif": hashlib.sha256(b"context-geotiff").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    activate_center_only_scope(tmp_path, school_id)

    activated = json.loads(resolution_path.read_text(encoding="utf-8"))
    assert activated["scope_mode"] == "center_only"
    assert activated["scope_boundary_authority"] == "none"
    assert activated["measurement_search_scope"] == "entire_detail_image"
    assert activated["recommended_detail_extent_m"] == 800
    assert activated["failed_boundary_attempt"]["validation_error"] == "malformed polygon"


def test_boundary_proposal_activation_records_non_binding_scope(tmp_path: Path) -> None:
    school_id = "example-school"
    imagery_dir = tmp_path / "data" / "imagery" / school_id
    resolution_dir = tmp_path / "data" / "campus_resolutions"
    proposal_dir = tmp_path / "data" / "campus_boundary_proposals" / "v1.0"
    imagery_dir.mkdir(parents=True)
    resolution_dir.mkdir(parents=True)
    proposal_dir.mkdir(parents=True)
    (imagery_dir / "context.jpg").write_bytes(b"context-jpeg")
    (imagery_dir / "context.tif").write_bytes(b"context-geotiff")
    resolution_path = resolution_dir / f"{school_id}.json"
    resolution_path.write_text(
        json.dumps(
            {
                "school_id": school_id,
                "school_name": "Example School",
                "status": "unresolved",
                "requires_human_review": True,
                "geometry": [[40.0, -75.0]],
                "bbox_wgs84": None,
                "reason": "Public match is point-only.",
                "boundary_notes": "Public match is point-only.",
            }
        ),
        encoding="utf-8",
    )
    proposal_path = proposal_dir / f"{school_id}.json"
    proposal_path.write_text(
        json.dumps(
            {
                "school_id": school_id,
                "status": "completed",
                "configuration_id": "school-facilities-boundary-resolver-v1.0",
                "model": "gemini-3.5-flash-lite",
                "input_sha256": {
                    "context.jpg": hashlib.sha256(b"context-jpeg").hexdigest(),
                    "context.tif": hashlib.sha256(b"context-geotiff").hexdigest(),
                },
                "bbox_wgs84": [-75.002, 39.998, -74.998, 40.002],
                "geometry_wgs84_lat_lon": [
                    [39.998, -75.002],
                    [39.998, -74.998],
                    [40.002, -74.998],
                    [39.998, -75.002],
                ],
                "request_fingerprint_sha256": "a" * 64,
                "deterministic_guard": {"candidate_quality_gate_passed": False},
            }
        ),
        encoding="utf-8",
    )

    activate_boundary_proposal_as_soft_scope(tmp_path, school_id)

    activated = json.loads(resolution_path.read_text(encoding="utf-8"))
    assert activated["status"] == "confirmed"
    assert activated["scope_mode"] == "soft_boundary"
    assert activated["scope_boundary_authority"] == "soft_guidance"
    assert activated["measurement_search_scope"] == "entire_detail_image"
    assert activated["boundary_status"] == "approximate_non_authoritative"
    assert activated["recommended_detail_extent_m"] >= 600
    assert activated["input_pair_frozen"] is False
    assert activated["soft_boundary_proposal"]["candidate_quality_gate_passed"] is False


def test_campus_review_overlay_marks_flagged_polygon_and_centers(tmp_path: Path) -> None:
    root = tmp_path
    school_id = "example-school"
    resolution_dir = root / "data" / "campus_resolutions"
    imagery_dir = root / "data" / "imagery" / school_id
    resolution_dir.mkdir(parents=True)
    imagery_dir.mkdir(parents=True)
    resolution = {
        "school_id": school_id,
        "school_name": "Example School",
        "status": "probable",
        "requires_human_review": True,
        "matched_name": "Example Public School",
        "source_element": "https://www.openstreetmap.org/way/1",
        "requested_ccd_coordinate": {"latitude": 40.005, "longitude": -75.005},
        "resolved_center": {"latitude": 40.0055, "longitude": -75.0045},
        "geometry": [
            [40.003, -75.007],
            [40.003, -75.002],
            [40.008, -75.002],
            [40.008, -75.007],
            [40.003, -75.007],
        ],
    }
    (resolution_dir / f"{school_id}.json").write_text(
        json.dumps(resolution), encoding="utf-8"
    )
    Image.new("RGB", (100, 100), "white").save(imagery_dir / "context.jpg")
    with rasterio.open(
        imagery_dir / "context.tif",
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(-75.01, 40.0, -75.0, 40.01, 100, 100),
    ) as destination:
        destination.write(np.full((3, 100, 100), 255, dtype=np.uint8))

    overlay_path, metadata_path = prepare_campus_review_overlay(root, school_id)

    assert overlay_path.is_file()
    assert metadata_path.is_file()
    with Image.open(overlay_path) as overlay:
        assert overlay.size == (100, 100)
        assert overlay.convert("RGB").getpixel((50, 50)) != (255, 255, 255)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["not_an_approved_vlm_input"] is True
    assert metadata["source_element"].endswith("/way/1")
    assert len(metadata["campus_resolution_sha256"]) == 64


def test_flagged_polygon_approval_records_auditable_manual_confirmation(
    tmp_path: Path,
) -> None:
    school_id = "example-school"
    resolution_dir = tmp_path / "data" / "campus_resolutions"
    resolution_dir.mkdir(parents=True)
    resolution_path = resolution_dir / f"{school_id}.json"
    resolution_path.write_text(
        json.dumps(
            {
                "school_id": school_id,
                "status": "probable",
                "requires_human_review": True,
                "source_element": "https://www.openstreetmap.org/way/1",
                "bbox_wgs84": [-75.01, 40.0, -75.0, 40.01],
                "geometry": [
                    [40.0, -75.01],
                    [40.0, -75.0],
                    [40.01, -75.0],
                    [40.0, -75.01],
                ],
                "reason": "Automatic score margin was below threshold.",
                "boundary_notes": "Automatic score margin was below threshold.",
            }
        ),
        encoding="utf-8",
    )

    approve_flagged_campus_polygon(
        tmp_path,
        school_id,
        review_note="Reviewer approved the visible campus scope.",
        reviewed_at="2026-09-03",
    )

    approved = json.loads(resolution_path.read_text(encoding="utf-8"))
    assert approved["status"] == "confirmed"
    assert approved["requires_human_review"] is False
    assert approved["input_pair_frozen"] is False
    assert approved["confirmation_method"] == "user_manual_polygon_review"
    assert approved["automatic_resolution_status_before_review"] == "probable"
    assert approved["manual_review"]["decision"] == "approved_proposed_polygon"
    assert approved["manual_review"]["reviewed_at"] == "2026-09-03"

    with pytest.raises(CampusResolutionError, match="not awaiting human review"):
        approve_flagged_campus_polygon(
            tmp_path,
            school_id,
            review_note="Duplicate approval should fail.",
            reviewed_at="2026-09-03",
        )


def valid_boundary_output() -> dict[str, object]:
    cue = {
        "visibility": "clear",
        "cue_types": ["road", "tree_line"],
        "observable_evidence": "A road and tree line form a visible edge.",
    }
    region = {
        "label": "main school complex",
        "region_type": "school_building",
        "bbox": {"x_min": 0.35, "y_min": 0.35, "x_max": 0.65, "y_max": 0.65},
        "observable_evidence": "A large institutional building surrounds the supplied point.",
    }
    return {
        "schema_version": "1.0.0",
        "school_id": "example-school",
        "campus_visibility": "complete",
        "school_anchor": {
            "x": 0.5,
            "y": 0.5,
            "observable_evidence": "The supplied point falls on the institutional roof complex.",
        },
        "boundary_polygon_normalized": [
            {"x": 0.2, "y": 0.2},
            {"x": 0.8, "y": 0.2},
            {"x": 0.8, "y": 0.8},
            {"x": 0.2, "y": 0.8},
            {"x": 0.2, "y": 0.2},
        ],
        "included_regions": [region],
        "excluded_adjacent_regions": [],
        "shared_or_ambiguous_regions": [],
        "boundary_cues": {side: dict(cue) for side in ("north", "east", "south", "west")},
        "suggested_confidence": 0.8,
        "review_required": False,
        "review_reasons": [],
        "summary": "The complete campus has visible perimeter cues.",
    }


def test_boundary_request_is_isolated_to_one_context_image() -> None:
    school = BoundaryVLMInput(
        school_id="example-school",
        school_name="Example School",
        context_path=Path("context.jpg"),
        context_geotiff_path=Path("context.tif"),
        source="NAIP",
        capture_vintage="2023-01-01T00:00:00Z",
        metres_per_pixel=0.625,
        requested_coordinate=(40.0, -75.0),
        public_match_name="Example School",
        public_match_source="https://www.openstreetmap.org/node/1",
        public_match_coordinate=(40.0, -75.0),
        resolver_reason="Exact-name public point without a polygon.",
    )
    request = build_boundary_request(ROOT, school)

    assert request["model"] == "gemini-3.5-flash-lite"
    assert [item["type"] for item in request["input"]] == ["text", "image"]
    assert request["input"][1]["data"] == Path("context.jpg").resolve()
    assert "Do not report rooftop solar" in request["system_instruction"]
    assert request["store"] is False


def test_boundary_geometry_guard_requires_review_without_auto_confirmation() -> None:
    config, schema, _ = load_boundary_bundle(ROOT)
    guarded = validate_boundary_output(
        valid_boundary_output(), schema, "example-school", config
    )

    assert guarded["candidate_quality_gate_passed"] is True
    assert guarded["guarded_review_required"] is True
    assert guarded["area_fraction"] == pytest.approx(0.36)
    assert "require human approval" in guarded["guard_reasons"][-1]


def test_boundary_geometry_guard_rejects_self_intersection() -> None:
    config, schema, _ = load_boundary_bundle(ROOT)
    output = valid_boundary_output()
    output["boundary_polygon_normalized"] = [
        {"x": 0.2, "y": 0.2},
        {"x": 0.8, "y": 0.8},
        {"x": 0.8, "y": 0.2},
        {"x": 0.2, "y": 0.8},
        {"x": 0.2, "y": 0.2},
    ]

    with pytest.raises(VLMResponseError, match="self-intersects"):
        validate_boundary_output(output, schema, "example-school", config)


def test_boundary_vocabulary_normalization_is_logged_and_measurement_neutral() -> None:
    output = valid_boundary_output()
    output["schema_version"] = "1.0"
    output["campus_visibility"] = "full"
    output["included_regions"][0]["region_type"] = "building_complex"
    output["excluded_adjacent_regions"] = [
        {
            "label": "Public road",
            "region_type": "public_road",
            "bbox": {"x_min": 0.0, "y_min": 0.0, "x_max": 0.1, "y_max": 0.1},
            "observable_evidence": "Road outside the campus.",
        }
    ]
    output["boundary_cues"]["north"]["visibility"] = "visible"
    output["boundary_cues"]["east"]["visibility"] = "partial"
    output["boundary_cues"]["north"]["cue_types"] = ["roads", "tree lines"]

    normalized, changes = normalize_boundary_vocabulary(output)

    assert normalized["schema_version"] == "1.0.0"
    assert normalized["campus_visibility"] == "complete"
    assert normalized["included_regions"][0]["region_type"] == "school_building"
    assert normalized["excluded_adjacent_regions"][0]["region_type"] == "road"
    assert normalized["boundary_cues"]["north"]["visibility"] == "clear"
    assert normalized["boundary_cues"]["east"]["visibility"] == "weak"
    assert normalized["boundary_cues"]["north"]["cue_types"] == ["road", "tree_line"]
    assert len(changes) == 8


def test_boundary_client_preserves_guarded_geospatial_proposal(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    imagery_dir = tmp_path / "data" / "imagery" / "example-school"
    config_dir.mkdir(parents=True)
    imagery_dir.mkdir(parents=True)
    for name in (
        "boundary_vlm.json",
        "boundary_vlm_prompt.txt",
        "boundary_vlm_response_schema.json",
    ):
        shutil.copy2(ROOT / "config" / name, config_dir / name)
    context_path = imagery_dir / "context.jpg"
    geotiff_path = imagery_dir / "context.tif"
    Image.new("RGB", (100, 100), "white").save(context_path)
    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(-75.01, 40.0, -75.0, 40.01, 100, 100),
    ) as destination:
        destination.write(np.full((3, 100, 100), 255, dtype=np.uint8))
    school = BoundaryVLMInput(
        school_id="example-school",
        school_name="Example School",
        context_path=context_path,
        context_geotiff_path=geotiff_path,
        source="NAIP",
        capture_vintage="2023-01-01T00:00:00Z",
        metres_per_pixel=1.0,
        requested_coordinate=(40.005, -75.005),
        public_match_name="Example School",
        public_match_source="https://www.openstreetmap.org/node/1",
        public_match_coordinate=(40.005, -75.005),
        resolver_reason="Exact public point without polygon.",
    )

    def fake_create(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            status="completed",
            id="boundary-test",
            output_text=json.dumps(valid_boundary_output()),
            created="2026-09-03T00:00:00Z",
            usage={"total_tokens": 100},
        )

    saved_path = GeminiBoundaryClient(tmp_path, fake_create).propose(school)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved["proposal_status"] == "ready_for_human_review"
    assert saved["campus_resolution_was_not_overwritten"] is True
    assert saved["deterministic_guard"]["candidate_quality_gate_passed"] is True
    assert saved["deterministic_guard"]["guarded_review_required"] is True
    assert len(saved["geometry_wgs84_lat_lon"]) == 5
    assert saved["bbox_wgs84"] == pytest.approx([-75.008, 40.002, -75.002, 40.008])
    ledger = json.loads(
        (tmp_path / "data" / "model_outputs" / "boundary_request_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(ledger["requests"]) == 1
    overlay_path, overlay_metadata = prepare_boundary_proposal_overlay(
        tmp_path, "example-school"
    )
    assert overlay_path.is_file()
    assert overlay_metadata.is_file()
    overlay_record = json.loads(overlay_metadata.read_text(encoding="utf-8"))
    assert overlay_record["not_an_approved_vlm_input"] is True


def complete_measurement_rows() -> list[dict[str, str]]:
    schools = read_csv(ROOT / "schools_sample.csv")
    rows = measurement_template(schools)
    for row in rows:
        row.update(
            {
                "imagery_source": "test imagery",
                "imagery_vintage": "2024-06",
                "campus_resolution_notes": "confirmed for test",
                "solar_present": "no",
                "solar_area_m2": "0",
                "portable_classroom_count": "0",
                "perimeter_fencing": "none",
                "dominant_fence_type": "none",
                "running_track": "no",
                "full_size_sports_fields": "0",
                "hard_courts": "0",
                "pool": "no",
                "review_status": "reviewed",
            }
        )
        for field in MEASUREMENT_FIELDS:
            row[CONFIDENCE_COLUMN[field]] = "0.8"
    return rows


def test_supplied_school_file_is_valid() -> None:
    result = validate_schools(ROOT / "schools_sample.csv")
    assert result.ok, result.errors
    assert len(read_csv(ROOT / "schools_sample.csv")) == 25


def test_template_has_all_schools_and_columns(tmp_path: Path) -> None:
    schools = read_csv(ROOT / "schools_sample.csv")
    rows = measurement_template(schools)
    output = tmp_path / "measurements.csv"
    write_csv(output, MEASUREMENT_COLUMNS, rows)
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == MEASUREMENT_COLUMNS
        assert len(list(reader)) == 25
    result = validate_measurements(output, ROOT / "schools_sample.csv", final=False)
    assert result.ok, result.errors


def test_final_validation_rejects_unreviewed_template(tmp_path: Path) -> None:
    schools = read_csv(ROOT / "schools_sample.csv")
    output = tmp_path / "measurements.csv"
    write_csv(output, MEASUREMENT_COLUMNS, measurement_template(schools))
    result = validate_measurements(output, ROOT / "schools_sample.csv", final=True)
    assert not result.ok
    assert any("review_status" in error for error in result.errors)


def test_complete_measurements_pass_final_validation(tmp_path: Path) -> None:
    output = tmp_path / "measurements.csv"
    write_csv(output, MEASUREMENT_COLUMNS, complete_measurement_rows())
    result = validate_measurements(output, ROOT / "schools_sample.csv", final=True)
    assert result.ok, result.errors


def test_final_validation_requires_failure_notes_and_consistency(tmp_path: Path) -> None:
    rows = complete_measurement_rows()
    rows[0]["portable_classroom_count"] = "unknown"
    rows[0]["portable_classroom_count_confidence"] = "0.2"
    rows[1]["perimeter_fencing"] = "none"
    rows[1]["dominant_fence_type"] = "chain-link"
    rows[2]["imagery_vintage"] = "summer 2024"
    output = tmp_path / "measurements.csv"
    write_csv(output, MEASUREMENT_COLUMNS, rows)
    result = validate_measurements(output, ROOT / "schools_sample.csv", final=True)
    assert not result.ok
    assert any("failure_notes is required" in error for error in result.errors)
    assert any("dominant_fence_type must be none" in error for error in result.errors)
    assert any("imagery_vintage must be" in error for error in result.errors)


def test_calibration_excludes_unknown_truth_and_solar_negative_rows() -> None:
    prediction = {
        "school_id": "1",
        "portable_classroom_count": "unknown",
        "portable_classroom_count_confidence": "0.2",
        "solar_area_m2": "0",
        "solar_area_m2_confidence": "0.9",
    }
    truth = {"school_id": "1", "portable_classroom_count": "unknown", "solar_area_m2": "0"}
    assert observations([prediction], [truth]) == []


def test_calibration_reports_field_specific_errors() -> None:
    prediction = {
        "school_id": "1",
        "portable_classroom_count": "3",
        "portable_classroom_count_confidence": "0.8",
        "solar_area_m2": "120",
        "solar_area_m2_confidence": "0.8",
    }
    truth = {"school_id": "1", "portable_classroom_count": "2", "solar_area_m2": "100"}
    items = observations([prediction], [truth])
    report = summarize(items)
    assert report["n"] == 2
    assert report["school_n"] == 1
    assert report["correct_n"] == 1
    assert report["incorrect_n"] == 1
    assert report["accuracy_95pct_wilson"]["lower"] < report["accuracy"]
    assert report["accuracy_95pct_wilson"]["upper"] > report["accuracy"]
    assert report["calibration_status"] == "descriptive_only"
    assert report["warnings"]
    assert report["by_field"]["portable_classroom_count"]["mean_absolute_error"] == 1
    assert report["by_field"]["solar_area_m2"]["mean_absolute_error_m2"] == 20
    assert report["by_field"]["solar_area_m2"]["median_absolute_percentage_error"] == 0.2
    assert report["by_field"]["solar_area_m2"]["within_25_percent_rate"] == 1.0


def test_bounds_are_centered() -> None:
    west, south, east, north = bounds_around(34.0, -118.0, 450)
    assert west < -118.0 < east
    assert south < 34.0 < north


class FakeResponse:
    def __init__(self, content: bytes, content_type: str = "image/jpeg") -> None:
        self.content = content
        self.headers = {"content-type": content_type}
        self.url = "https://example.test/export?bbox=1"

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls = 0

    def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
        self.calls += 1
        return self.response


def jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_imagery_download_is_verified_and_cached(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(jpeg_bytes()))
    image_path, downloaded = fetch_esri_image(
        school_id="010282001124",
        school_name="Test School",
        latitude=34.0,
        longitude=-118.0,
        output_dir=tmp_path,
        half_width_m=450,
        pixels=1200,
        session=session,  # type: ignore[arg-type]
    )
    assert downloaded
    assert image_path.is_file()
    metadata = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["school_id"] == "010282001124"
    assert metadata["capture_vintage"] is None

    _, downloaded_again = fetch_esri_image(
        school_id="010282001124",
        school_name="Test School",
        latitude=34.0,
        longitude=-118.0,
        output_dir=tmp_path,
        half_width_m=450,
        pixels=1200,
        session=session,  # type: ignore[arg-type]
    )
    assert not downloaded_again
    assert session.calls == 1


def test_invalid_image_leaves_no_final_or_temporary_file(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(b"not an image"))
    with pytest.raises(Exception):
        fetch_esri_image(
            school_id="010282001124",
            school_name="Test School",
            latitude=34.0,
            longitude=-118.0,
            output_dir=tmp_path,
            half_width_m=450,
            pixels=1200,
            session=session,  # type: ignore[arg-type]
        )
    assert list(tmp_path.iterdir()) == []


def test_frozen_technical_configuration_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_configuration(root)
    assert result.errors == []


def vlm_test_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    shutil.copytree(ROOT / "config", root / "config")
    shutil.copy2(ROOT / "schools_sample.csv", root / "schools_sample.csv")
    (root / "data" / "imagery").mkdir(parents=True)
    return root


def vlm_school(root: Path, *, confirmed: bool = True) -> SchoolVLMInput:
    context = root / "data" / "imagery" / "test_context.jpg"
    detail = root / "data" / "imagery" / "test_detail.jpg"
    Image.new("RGB", (1600, 1600), color="white").save(context, format="JPEG")
    Image.new("RGB", (1600, 1600), color="gray").save(detail, format="JPEG")
    return SchoolVLMInput(
        school_id="060483000471",
        school_name="Beverly Hills High",
        campus_resolution_notes="Public map evidence confirms the campus.",
        context=VLMImage(context, "USDA NAIP", "2022-05-11", 0.625, "context"),
        detail=VLMImage(detail, "USDA NAIP", "2022-05-11", 0.375, "detail"),
        public_source_and_non_sensitive_confirmed=confirmed,
        facility_crops=(),
        campus_boundary_detail_normalized=(
            (0.25, 0.25),
            (0.75, 0.25),
            (0.75, 0.75),
            (0.25, 0.75),
            (0.25, 0.25),
        ),
        campus_boundary_source="https://www.openstreetmap.org/way/123",
    )


def valid_vlm_output() -> dict[str, object]:
    def suggestion(value: object) -> dict[str, object]:
        confidence = 0.2 if value == "unknown" else 0.8
        return {
            "value": value,
            "evidence": "Visible in the test image.",
            "suggested_confidence": confidence,
            "confidence_reason": "The supplied imagery supports this diagnostic score.",
            "review_required": True,
        }

    def packet(
        field: str,
        *,
        visibility: str = "adequate",
        observations: list[str] | None = None,
        components: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "field": field,
            "visibility": visibility,
            "campus_relation": "inside",
            "searched_scope": "The full resolved campus was inspected.",
            "located_observable_facts": observations or [],
            "count_components": components or [],
            "ambiguity_notes": "none",
        }

    directions = {
        "solar": "negative",
        "portable_classrooms": "negative",
        "fencing": "unknown",
        "running_track": "positive",
        "sports_fields": "positive",
        "hard_courts": "positive",
        "pool": "negative",
    }

    def candidate(candidate_id: str, classification: str) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "location": "detail image, central campus",
            "bbox_normalized": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.4, "y_max": 0.4},
            "inside_campus_boundary": "yes",
            "visibility": "clear",
            "classification": classification,
            "qualifies": "yes",
            "positive_cues": ["Test geometry and equipment are visible."],
            "contradictory_cues": [],
            "review_reason": "none",
        }

    protocol = json.loads((ROOT / "config" / "vlm_field_protocol.json").read_text())
    feature_candidates = {
        "solar": [],
        "portable_classrooms": [],
        "fencing": [],
        "running_track": [candidate("track-1", "purpose-built running track")],
        "sports_fields": [candidate("field-1", "formal full-size field")],
        "hard_courts": [candidate("courts-1", "two independent full courts")],
        "pool": [],
    }
    feature_assessments = []
    for feature, specification in protocol["features"].items():
        direction = directions[feature]
        answers = []
        for question in specification["questions"]:
            expected = question.get(f"expected_for_{direction}")
            answers.append(
                {
                    "question_id": question["id"],
                    "answer": expected or "yes",
                    "observation": "The test image contains the required observable cue.",
                    "location": "detail image, resolved campus",
                }
            )
        feature_assessments.append(
            {
                "feature": feature,
                "question_answers": answers,
                "candidates": feature_candidates[feature],
                "derivation_summary": "The test value follows from the listed observations.",
            }
        )

    return {
        "schema_version": "1.10.0",
        "school_id": "060483000471",
        "campus_assessment": {"status": "confirmed", "evidence": "Campus is visible."},
        "solar_inventory": {
            "roof_visibility": "adequate",
            "search_evidence": "All plausible campus roofs are visible and no array is present.",
            "candidates": [],
        },
        "fencing_inventory": {
            "boundary_visibility": "partial",
            "minimum_barrier_coverage_fraction": 0.3,
            "maximum_barrier_coverage_fraction": 0.8,
            "dominant_observed_type": "unknown",
            "coverage_evidence": "Some boundary segments are obscured.",
            "segments": [
                {
                    "segment_id": "boundary-1",
                    "sector": "irregular",
                    "boundary_fraction": 1.0,
                    "barrier_fraction": 0.3,
                    "unfenced_fraction": 0.2,
                    "unobservable_fraction": 0.5,
                    "barrier_type": "unknown",
                    "shadow_support": "unknown",
                    "outer_boundary_relation": "unknown",
                    "evidence": "Partial boundary evidence in the test image."
                }
            ],
        },
        "feature_assessments": feature_assessments,
        "evidence_packets": [
            packet("solar_present"),
            packet("solar_area_m2"),
            packet("portable_classroom_count"),
            packet("perimeter_fencing", visibility="partial"),
            packet("dominant_fence_type", visibility="partial"),
            packet(
                "running_track",
                observations=["Detail image, central campus: oval track footprint."],
            ),
            packet(
                "full_size_sports_fields",
                components=[
                    {
                        "component_id": "field-group-1",
                        "location": "detail image, central campus",
                        "physical_footprint_count": 1,
                        "observable_fact": "One formal field footprint is visible.",
                    }
                ]
            ),
            packet(
                "hard_courts",
                components=[
                    {
                        "component_id": "court-group-1",
                        "location": "detail image, east campus",
                        "physical_footprint_count": 2,
                        "observable_fact": "Two separately playable footprints are visible.",
                    }
                ]
            ),
            packet("pool"),
        ],
        "measurements": {
            "solar_present": suggestion("no"),
            "solar_area_m2": suggestion(0),
            "portable_classroom_count": suggestion(0),
            "perimeter_fencing": suggestion("unknown"),
            "dominant_fence_type": suggestion("unknown"),
            "running_track": suggestion("yes"),
            "full_size_sports_fields": suggestion(1),
            "hard_courts": suggestion(2),
            "pool": suggestion("no"),
        },
        "visibility_flags": ["test"],
        "ownership_flags": [],
        "review_fields": ["perimeter_fencing", "dominant_fence_type"],
    }


def test_vlm_request_matches_interactions_sdk_types(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    request = build_interaction_request(root, vlm_school(root))
    assert request["model"] == "gemini-3.5-flash-lite"
    assert request["generation_config"] == {
        "thinking_level": "minimal",
        "max_output_tokens": 8192,
    }
    assert [item["type"] for item in request["input"]] == ["text", "image", "image"]
    assert request["input"][1]["resolution"] == "high"
    assert request["input"][2]["resolution"] == "high"
    assert request["response_format"]["mime_type"] == "application/json"
    serialized_schema = json.dumps(request["response_format"]["schema"])
    assert "minLength" not in serialized_schema
    assert "uniqueItems" not in serialized_schema
    assert "$defs" not in serialized_schema
    assert "$ref" not in serialized_schema
    assert "anyOf" not in serialized_schema
    assert '"minimum":' not in serialized_schema
    assert '"maximum":' not in serialized_schema
    assert len(serialized_schema) < 14000
    local_schema = json.loads((root / "config" / "vlm_response_schema.json").read_text())
    assert local_schema["properties"]["review_fields"]["uniqueItems"] is True
    assert local_schema["$defs"]["confidenceScore"]["enum"] == [0.8, 0.6, 0.4, 0.2]
    assert "0.95" not in serialized_schema
    assert request["store"] is False
    assert "labels" not in request
    metadata = json.loads(request["input"][0]["text"].split("Metadata:\n", 1)[1])
    assert metadata["resolved_campus_boundary"]["image_role"] == "detail"
    assert metadata["resolved_campus_boundary"]["polygon"][0] == {"x": 0.25, "y": 0.25}
    assert metadata["campus_scope"]["mode"] == "authoritative_polygon"
    assert metadata["campus_scope"]["measurement_search_scope"] == "inside_authoritative_polygon"

    from google.genai._gaos.types.interactions.createmodelinteraction import (
        CreateModelInteraction,
    )

    parsed = CreateModelInteraction.model_validate(request)
    assert parsed.model == "gemini-3.5-flash-lite"


def test_soft_boundary_request_searches_full_detail_image(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    base = vlm_school(root)
    school = SchoolVLMInput(
        **{
            **base.__dict__,
            "campus_scope_mode": "soft_boundary",
            "scope_boundary_authority": "soft_guidance",
            "measurement_search_scope": "entire_detail_image",
        }
    )
    request = build_interaction_request(root, school)
    metadata = json.loads(request["input"][0]["text"].split("Metadata:\n", 1)[1])
    assert metadata["campus_scope"]["mode"] == "soft_boundary"
    assert metadata["campus_scope"]["measurement_search_scope"] == "entire_detail_image"
    assert "resolved_campus_boundary" not in metadata
    assert metadata["non_binding_boundary_guidance"]["polygon"]
    assert "entire detail image" in metadata["non_binding_boundary_guidance"]["instruction"]


def test_center_only_request_does_not_require_polygon(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    base = vlm_school(root)
    school = SchoolVLMInput(
        **{
            **base.__dict__,
            "campus_boundary_detail_normalized": (),
            "campus_boundary_source": None,
            "campus_scope_mode": "center_only",
            "scope_boundary_authority": "none",
            "measurement_search_scope": "entire_detail_image",
        }
    )
    request = build_interaction_request(root, school)
    metadata = json.loads(request["input"][0]["text"].split("Metadata:\n", 1)[1])
    assert metadata["campus_scope"]["mode"] == "center_only"
    assert "resolved_campus_boundary" not in metadata
    assert "non_binding_boundary_guidance" not in metadata


def test_vlm_request_requires_public_non_sensitive_confirmation(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    with pytest.raises(VLMConfigurationError, match="public-source and non-sensitive"):
        build_interaction_request(root, vlm_school(root, confirmed=False))


def test_provider_vocabulary_normalization_is_whitelisted_and_auditable() -> None:
    original = {
        "schema_version": "1.0",
        "campus_assessment": {"status": "fully_visible"},
        "solar_inventory": {
            "roof_visibility": "clear",
            "candidates": [
                {
                    "support_structure": "metal carport structure",
                    "support_surface_form": "carport canopy",
                }
            ],
        },
        "fencing_inventory": {"boundary_visibility": "clear"},
        "evidence_packets": [
            {"visibility": "clear", "campus_relation": "on-site"}
        ],
        "measurements": {"hard_courts": {"value": 4}},
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert original["schema_version"] == "1.0"
    assert normalized["schema_version"] == "1.10.0"
    assert normalized["campus_assessment"]["status"] == "confirmed"
    assert normalized["solar_inventory"]["roof_visibility"] == "adequate"
    assert normalized["fencing_inventory"]["boundary_visibility"] == "adequate"
    assert normalized["evidence_packets"][0] == {
        "visibility": "adequate",
        "campus_relation": "inside",
    }
    assert normalized["solar_inventory"]["candidates"][0] == {
        "support_structure": "parking_carport",
        "support_surface_form": "canopy",
    }
    assert normalized["measurements"] == original["measurements"]
    assert {change["path"] for change in changes} == {
        "schema_version",
        "campus_assessment.status",
        "solar_inventory.roof_visibility",
        "fencing_inventory.boundary_visibility",
        "evidence_packets.0.visibility",
        "evidence_packets.0.campus_relation",
        "solar_inventory.candidates.0.support_structure",
        "solar_inventory.candidates.0.support_surface_form",
    }


def test_provider_vocabulary_normalization_preserves_ambiguous_solar_support() -> None:
    original = {
        "schema_version": "1.10.0",
        "campus_assessment": {"status": "complete"},
        "solar_inventory": {
            "roof_visibility": "fully visible",
            "candidates": [
                {
                    "support_structure": "flat roof mount",
                    "support_surface_form": "pitched/flat hybrid roof plane",
                },
                {
                    "support_structure": "metal carport frame",
                    "support_surface_form": "parking lot shade structure",
                },
            ],
        },
        "fencing_inventory": {
            "boundary_visibility": "fully visible",
            "segments": [
                {
                    "sector": "north",
                    "outer_boundary_relation": "northern edge along residential street",
                    "shadow_support": "no fence shadow visible",
                },
                {
                    "sector": "east",
                    "outer_boundary_relation": "eastern edge along residential street",
                    "shadow_support": "minor shadow near property line",
                },
            ],
        },
        "evidence_packets": [
            {"visibility": "fully visible", "campus_relation": "outer campus boundary"}
        ],
        "feature_assessments": [
            {"candidates": [{"visibility": "fully visible"}]}
        ],
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert normalized["campus_assessment"]["status"] == "confirmed"
    assert normalized["solar_inventory"]["roof_visibility"] == "adequate"
    assert normalized["solar_inventory"]["candidates"][0] == {
        "support_structure": "uncertain",
        "support_surface_form": "uncertain",
    }
    assert normalized["solar_inventory"]["candidates"][1] == {
        "support_structure": "parking_carport",
        "support_surface_form": "canopy",
    }
    assert normalized["fencing_inventory"]["segments"][0][
        "outer_boundary_relation"
    ] == "yes"
    assert normalized["fencing_inventory"]["segments"][0]["shadow_support"] == "no"
    assert normalized["fencing_inventory"]["segments"][1]["shadow_support"] == "yes"
    assert normalized["evidence_packets"][0] == {
        "visibility": "adequate",
        "campus_relation": "inside",
    }
    assert normalized["feature_assessments"][0]["candidates"][0][
        "visibility"
    ] == "clear"
    assert original["solar_inventory"]["candidates"][0]["support_structure"] == (
        "flat roof mount"
    )
    assert len(changes) == 14


def test_provider_vocabulary_normalizes_soft_scope_descriptors_conservatively() -> None:
    original = {
        "campus_assessment": {"status": "ambiguous_boundary"},
        "fencing_inventory": {"segments": [{"sector": "entire_perimeter"}]},
        "evidence_packets": [
            {"campus_relation": "inside_boundary"},
            {"campus_relation": "ambiguous_boundary"},
            {
                "campus_relation": "uncertain_boundary",
                "visibility": "partially_obscured",
            },
        ],
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert normalized["campus_assessment"]["status"] == "probable"
    assert normalized["fencing_inventory"]["segments"][0]["sector"] == "irregular"
    assert normalized["evidence_packets"][0]["campus_relation"] == "inside"
    assert normalized["evidence_packets"][1]["campus_relation"] == "uncertain"
    assert normalized["evidence_packets"][2]["campus_relation"] == "uncertain"
    assert normalized["evidence_packets"][2]["visibility"] == "partial"
    assert {change["path"] for change in changes} == {
        "campus_assessment.status",
        "fencing_inventory.segments.0.sector",
        "evidence_packets.0.campus_relation",
        "evidence_packets.1.campus_relation",
        "evidence_packets.2.campus_relation",
        "evidence_packets.2.visibility",
    }


def test_provider_vocabulary_keeps_unverified_campus_unresolved() -> None:
    original = {
        "campus_assessment": {"status": "unverified_boundary_center_only"},
        "fencing_inventory": {
            "segments": [{"outer_boundary_relation": "uncertain"}]
        },
        "evidence_packets": [{"campus_relation": "on_campus"}],
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert normalized["campus_assessment"]["status"] == "unresolved"
    assert normalized["fencing_inventory"]["segments"][0][
        "outer_boundary_relation"
    ] == "unknown"
    assert normalized["evidence_packets"][0]["campus_relation"] == "uncertain"
    assert {change["path"] for change in changes} == {
        "campus_assessment.status",
        "fencing_inventory.segments.0.outer_boundary_relation",
        "evidence_packets.0.campus_relation",
    }


def test_provider_vocabulary_normalization_uses_conservative_partial_roof_rule() -> None:
    original = {
        "campus_assessment": {"status": "fully_resolved"},
        "solar_inventory": {"roof_visibility": "partial", "candidates": []},
        "fencing_inventory": {
            "segments": [
                {
                    "sector": "irregular",
                    "outer_boundary_relation": "outer_boundary",
                }
            ]
        },
        "evidence_packets": [{"campus_relation": "inside_campus"}],
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert normalized["campus_assessment"]["status"] == "confirmed"
    assert normalized["solar_inventory"]["roof_visibility"] == "inadequate"
    assert normalized["fencing_inventory"]["segments"][0][
        "outer_boundary_relation"
    ] == "yes"
    assert normalized["evidence_packets"][0]["campus_relation"] == "inside"
    assert len(changes) == 4


def test_provider_vocabulary_normalization_handles_visibility_spelling_variants() -> None:
    original = {
        "fencing_inventory": {
            "boundary_visibility": "partially_visible",
            "segments": [{"outer_boundary_relation": "direct"}],
        },
        "evidence_packets": [
            {"visibility": "fully_visible"},
            {"visibility": "partially_visible"},
        ],
        "feature_assessments": [
            {"candidates": [{"visibility": "fully_visible"}]}
        ],
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert normalized["fencing_inventory"]["boundary_visibility"] == "partial"
    assert normalized["fencing_inventory"]["segments"][0][
        "outer_boundary_relation"
    ] == "yes"
    assert normalized["evidence_packets"][0]["visibility"] == "adequate"
    assert normalized["evidence_packets"][1]["visibility"] == "partial"
    assert normalized["feature_assessments"][0]["candidates"][0][
        "visibility"
    ] == "clear"
    assert len(changes) == 5


def test_provider_vocabulary_normalizes_production_descriptors_conservatively() -> None:
    original = {
        "solar_inventory": {
            "roof_visibility": "fully_visible",
            "candidates": [
                {
                    "support_structure": "metal framing",
                    "support_surface_form": "flat roof",
                }
            ],
        },
        "fencing_inventory": {
            "boundary_visibility": "mostly_obscured_or_absent",
            "segments": [
                {
                    "sector": "perimeter",
                    "outer_boundary_relation": "adjacent_to_roadway",
                    "shadow_support": "no_shadow",
                },
                {
                    "sector": "north",
                    "outer_boundary_relation": "authoritative_boundary",
                },
            ],
        },
        "evidence_packets": [
            {"campus_relation": "inside_authoritative_polygon"},
            {"campus_relation": "fully_contained", "visibility": "mostly_visible"},
        ],
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert normalized["solar_inventory"]["roof_visibility"] == "adequate"
    assert normalized["solar_inventory"]["candidates"][0][
        "support_structure"
    ] == "uncertain"
    assert normalized["solar_inventory"]["candidates"][0][
        "support_surface_form"
    ] == "flat"
    assert normalized["fencing_inventory"]["boundary_visibility"] == "inadequate"
    assert normalized["fencing_inventory"]["segments"][0]["sector"] == "irregular"
    assert normalized["fencing_inventory"]["segments"][0][
        "outer_boundary_relation"
    ] == "unknown"
    assert normalized["fencing_inventory"]["segments"][0]["shadow_support"] == "no"
    assert normalized["fencing_inventory"]["segments"][1][
        "outer_boundary_relation"
    ] == "yes"
    assert normalized["evidence_packets"][0]["campus_relation"] == "inside"
    assert normalized["evidence_packets"][1]["campus_relation"] == "inside"
    assert normalized["evidence_packets"][1]["visibility"] == "partial"
    assert len(changes) == 11


def test_provider_vocabulary_normalizes_boundary_relation_prose() -> None:
    original = {
        "campus_assessment": {"status": "assessed"},
        "fencing_inventory": {
            "boundary_visibility": "unfenced_or_not_visible",
            "segments": [
                {"outer_boundary_relation": "coincident_with_tree_line"},
                {"outer_boundary_relation": "exact"},
                {
                    "outer_boundary_relation": (
                        "property line adjacent to residential street"
                    )
                },
            ],
        },
        "evidence_packets": [
            {"campus_relation": "outer_boundary"},
            {"campus_relation": "exterior_nearby"},
        ],
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert normalized["campus_assessment"]["status"] == "probable"
    assert normalized["fencing_inventory"]["boundary_visibility"] == "inadequate"
    assert normalized["fencing_inventory"]["segments"][0][
        "outer_boundary_relation"
    ] == "unknown"
    assert normalized["fencing_inventory"]["segments"][1][
        "outer_boundary_relation"
    ] == "yes"
    assert normalized["fencing_inventory"]["segments"][2][
        "outer_boundary_relation"
    ] == "unknown"
    assert normalized["evidence_packets"][0]["campus_relation"] == "inside"
    assert normalized["evidence_packets"][1]["campus_relation"] == "outside"
    assert len(changes) == 7


def test_provider_vocabulary_normalizes_pilot_descriptors_conservatively() -> None:
    original = {
        "solar_inventory": {
            "roof_visibility": "partially_obscured_or_unclear",
            "candidates": [
                {
                    "support_structure": "roof-mounted rack",
                    "support_surface_form": "pitched roof",
                },
                {"support_surface_form": "parking lot canopy"},
            ],
        },
        "fencing_inventory": {
            "segments": [
                {"outer_boundary_relation": "primary_street_edge"},
                {"outer_boundary_relation": "eastern edge along street"},
            ]
        },
        "evidence_packets": [
            {"campus_relation": "internal"},
            {"campus_relation": "outer boundary"},
        ],
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert normalized["solar_inventory"]["roof_visibility"] == "inadequate"
    assert normalized["solar_inventory"]["candidates"][0][
        "support_structure"
    ] == "uncertain"
    assert normalized["solar_inventory"]["candidates"][0][
        "support_surface_form"
    ] == "pitched"
    assert normalized["solar_inventory"]["candidates"][1][
        "support_surface_form"
    ] == "canopy"
    assert normalized["fencing_inventory"]["segments"][0][
        "outer_boundary_relation"
    ] == "unknown"
    assert normalized["fencing_inventory"]["segments"][1][
        "outer_boundary_relation"
    ] == "yes"
    assert normalized["evidence_packets"][0]["campus_relation"] == "inside"
    assert normalized["evidence_packets"][1]["campus_relation"] == "inside"
    assert len(changes) == 8


def test_provider_vocabulary_normalizes_blind_validation_descriptors() -> None:
    original = {
        "campus_assessment": {"status": "provisional_boundary_soft_guidance"},
        "solar_inventory": {
            "candidates": [
                {
                    "support_structure": "metal canopy framing",
                    "support_surface_form": "flat carport canopy rows",
                }
            ]
        },
        "fencing_inventory": {
            "boundary_visibility": "poor",
            "segments": [
                {
                    "sector": "entire_periphery",
                    "outer_boundary_relation": "unverified",
                },
                {"sector": "north", "outer_boundary_relation": "school_perimeter"},
            ],
        },
        "evidence_packets": [
            {"campus_relation": "onsite", "visibility": "none"},
            {"campus_relation": "unclear", "visibility": "poor"},
        ],
    }

    normalized, changes = _normalize_provider_vocabulary(original)

    assert normalized["campus_assessment"]["status"] == "probable"
    assert normalized["solar_inventory"]["candidates"][0][
        "support_structure"
    ] == "parking_carport"
    assert normalized["solar_inventory"]["candidates"][0][
        "support_surface_form"
    ] == "canopy"
    assert normalized["fencing_inventory"]["boundary_visibility"] == "inadequate"
    assert normalized["fencing_inventory"]["segments"][0]["sector"] == "irregular"
    assert normalized["fencing_inventory"]["segments"][0][
        "outer_boundary_relation"
    ] == "unknown"
    assert normalized["fencing_inventory"]["segments"][1][
        "outer_boundary_relation"
    ] == "yes"
    assert normalized["evidence_packets"][0]["campus_relation"] == "uncertain"
    assert normalized["evidence_packets"][0]["visibility"] == "inadequate"
    assert normalized["evidence_packets"][1]["campus_relation"] == "uncertain"
    assert normalized["evidence_packets"][1]["visibility"] == "inadequate"
    assert len(changes) == 11


def test_optional_diagnostic_facility_crops_are_reproducible(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    approved_vlm_pair(root)
    source = root / "data" / "imagery" / "060483000471" / "detail.tif"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    review_dir = root / "data" / "reviewed" / "060483000471"
    specification = {
        "schema_version": "1.0",
        "school_id": "060483000471",
        "review_status": "confirmed",
        "crop_pair_approved": False,
        "source_geotiff": "data/imagery/060483000471/detail.tif",
        "source_sha256": source_hash,
        "source_capture_vintage": "2022-05-11T16:00:00Z",
        "output_pixels": [1600, 1600],
        "regions": [
            {
                "role": "athletics_overview",
                "purpose": "Test athletics crop.",
                "source_pixel_window": [100, 100, 1000, 1000],
            },
            {
                "role": "hard_courts_detail",
                "purpose": "Test hard-court crop.",
                "source_pixel_window": [200, 200, 800, 800],
            },
        ],
    }
    (review_dir / "facility_regions.json").write_text(
        json.dumps(specification), encoding="utf-8"
    )

    first = prepare_facility_crops(root, "060483000471")
    assert [product.role for product in first] == ["athletics_overview", "hard_courts_detail"]
    assert all(product.downloaded for product in first)
    second = prepare_facility_crops(root, "060483000471")
    assert not any(product.downloaded for product in second)



def test_vlm_client_validates_and_preserves_raw_output_offline(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    calls: list[dict[str, object]] = []
    output = valid_vlm_output()

    def fake_create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(output),
            id="test-interaction",
            created="2026-08-29T12:00:00Z",
            usage={"total_tokens": 123},
        )

    client = GeminiVLMClient(root, fake_create)
    output_path = client.assess(vlm_school(root), mode="pilot")
    assert len(calls) == 1
    assert calls[0]["timeout"] == 300.0
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["configuration_id"] == "school-facilities-vlm-final-v1.10"
    assert saved["profile"] == "final"
    assert saved["model"] == "gemini-3.5-flash-lite"
    assert set(saved["input_sha256"]) == {"context", "detail"}
    assert len(saved["request_fingerprint_sha256"]) == 64
    assert len(saved["campus_boundary_detail_normalized"]) == 5
    assert saved["output_text"] == json.dumps(output)
    assert saved["parsed_output"] == output
    assert saved["derived_solar_summary"]["semantic_issues"] == []
    assert saved["derived_solar_summary"]["guarded_measurements"]["solar_present"] == "no"
    assert "api_key" not in output_path.read_text(encoding="utf-8").lower()


def test_vlm_client_rejects_invalid_output_without_final_file(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="completed", output_text='{"schema_version":"1.1"}')

    client = GeminiVLMClient(root, fake_create)
    with pytest.raises(VLMResponseError, match="schema validation"):
        client.assess(vlm_school(root), mode="pilot")
    assert not (
        root / "data" / "model_outputs" / "final" / "v1.10" / "060483000471.json"
    ).exists()
    rejected = list(
        (root / "data" / "model_outputs" / "rejected" / "v1.10").glob("*.json")
    )
    assert len(rejected) == 1
    rejected_record = json.loads(rejected[0].read_text(encoding="utf-8"))
    assert "schema validation" in rejected_record["validation_error"]
    assert rejected_record["output_text"] == '{"schema_version":"1.1"}'


def test_vlm_client_preserves_incomplete_provider_diagnostics(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)

    class FakeIncompleteResponse:
        status = "incomplete"
        output_text = '{"partial":true}'
        id = "incomplete-interaction"
        created = "2026-09-03T17:18:05Z"
        usage = {"total_input_tokens": 4000, "total_output_tokens": 8192}
        errors = [{"code": "MAX_OUTPUT_TOKENS", "message": "Output limit reached"}]
        incomplete_details = {
            "reason": "max_output_tokens",
            "api_key": "AIzaTHIS_IS_A_SECRET_API_KEY_VALUE_123456",
        }

        def model_dump(self, **_kwargs: object) -> dict[str, object]:
            return {
                "status": self.status,
                "id": self.id,
                "input": [{"type": "image", "data": "must-not-be-saved"}],
                "system_instruction": "must-not-be-saved",
                "incomplete_details": self.incomplete_details,
                "provider_reason": "max_output_tokens",
            }

    def fake_create(**_kwargs: object) -> FakeIncompleteResponse:
        return FakeIncompleteResponse()

    client = GeminiVLMClient(root, fake_create)
    with pytest.raises(VLMResponseError, match="rejected response preserved") as captured:
        client.assess(vlm_school(root), mode="pilot")

    assert "max_output_tokens" in str(captured.value)
    assert "AIzaTHIS" not in str(captured.value)
    assert not (
        root / "data" / "model_outputs" / "final" / "v1.10" / "060483000471.json"
    ).exists()
    rejected = list(
        (root / "data" / "model_outputs" / "rejected" / "v1.10").glob("*.json")
    )
    assert len(rejected) == 1
    rejected_record = json.loads(rejected[0].read_text(encoding="utf-8"))
    assert rejected_record["status"] == "incomplete"
    assert rejected_record["rejection_stage"] == "interaction_status"
    assert rejected_record["output_text"] == '{"partial":true}'
    assert (
        rejected_record["provider_diagnostics"]["incomplete_details"]["reason"]
        == "max_output_tokens"
    )
    serialized = json.dumps(rejected_record)
    assert "must-not-be-saved" not in serialized
    assert "AIzaTHIS" not in serialized
    assert "[REDACTED_API_KEY]" in serialized


def test_vlm_client_rejects_model_confidence_above_single_source_cap(
    tmp_path: Path,
) -> None:
    root = vlm_test_root(tmp_path)
    output = valid_vlm_output()
    output["measurements"]["hard_courts"]["suggested_confidence"] = 0.95

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="completed", output_text=json.dumps(output))

    client = GeminiVLMClient(root, fake_create)
    with pytest.raises(VLMResponseError, match="schema validation"):
        client.assess(vlm_school(root), mode="pilot")


def test_vlm_client_excludes_canopy_inventory_from_rooftop_measurements(
    tmp_path: Path,
) -> None:
    root = vlm_test_root(tmp_path)
    output = valid_vlm_output()
    output["solar_inventory"]["candidates"] = [
        {
            "candidate_id": "solar-1",
            "image_role": "detail",
            "bbox_normalized": {
                "x_min": 0.1,
                "y_min": 0.2,
                "x_max": 0.2,
                "y_max": 0.22555556,
            },
            "footprint_polygon_normalized": [
                {"x": 0.1, "y": 0.2},
                {"x": 0.2, "y": 0.2},
                {"x": 0.2, "y": 0.22555556},
                {"x": 0.1, "y": 0.22555556},
            ],
            "mount_location": "parking_carport_canopy",
            "support_structure": "parking_carport",
            "support_surface_color": "dark gray",
            "support_surface_form": "canopy",
            "surrounding_cues": ["vehicles and parking rows beneath the panels"],
            "mount_evidence": "Parking rows and vehicles are visible beneath the array.",
        }
    ]
    output["measurements"]["solar_present"].update(
        {"value": "no", "review_required": True}
    )
    output["measurements"]["solar_area_m2"].update(
        {"value": 0, "review_required": True}
    )
    output["review_fields"] = [
        "solar_present",
        "solar_area_m2",
        *output["review_fields"],
    ]

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="completed", output_text=json.dumps(output))

    saved_path = GeminiVLMClient(root, fake_create).assess(
        vlm_school(root), mode="pilot"
    )
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["parsed_output"]["measurements"]["solar_present"]["value"] == "no"
    assert saved["parsed_output"]["measurements"]["solar_area_m2"]["value"] == 0
    assert saved["derived_solar_summary"]["candidates"][0]["polygon_area_m2"] == pytest.approx(
        920.0, abs=0.1
    )


def test_vlm_client_flags_and_abstains_on_canopy_area_in_rooftop_total(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    output = valid_vlm_output()
    output["solar_inventory"]["candidates"] = [
        {
            "candidate_id": "roof-array",
            "image_role": "detail",
            "bbox_normalized": {
                "x_min": 0.1,
                "y_min": 0.1,
                "x_max": 0.2,
                "y_max": 0.11388889,
            },
            "footprint_polygon_normalized": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.12, "y": 0.1},
                {"x": 0.12, "y": 0.11388889},
                {"x": 0.1, "y": 0.11388889},
            ],
            "mount_location": "school_building_roof",
            "support_structure": "school_building",
            "support_surface_color": "light gray",
            "support_surface_form": "flat",
            "surrounding_cues": ["continuous building roof plane"],
            "mount_evidence": "The array boundary lies on a traceable building roof plane.",
        },
        {
            "candidate_id": "canopy-array",
            "image_role": "detail",
            "bbox_normalized": {
                "x_min": 0.5,
                "y_min": 0.5,
                "x_max": 0.55,
                "y_max": 0.55,
            },
            "footprint_polygon_normalized": [
                {"x": 0.5, "y": 0.5},
                {"x": 0.55, "y": 0.5},
                {"x": 0.55, "y": 0.55},
                {"x": 0.5, "y": 0.55},
            ],
            "mount_location": "parking_carport_canopy",
            "support_structure": "parking_carport",
            "support_surface_color": "dark gray",
            "support_surface_form": "canopy",
            "surrounding_cues": ["vehicles and parking lanes beneath the panels"],
            "mount_evidence": "Vehicles and parking lanes are visible beneath the array.",
        },
    ]
    output["measurements"]["solar_present"].update(
        {"value": "yes", "review_required": True}
    )
    output["measurements"]["solar_area_m2"].update(
        {"value": 1000, "review_required": True}
    )
    output["review_fields"] = [
        "solar_present",
        "solar_area_m2",
        *output["review_fields"],
    ]

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="completed", output_text=json.dumps(output))

    saved_path = GeminiVLMClient(root, fake_create).assess(vlm_school(root), mode="pilot")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    summary = saved["derived_solar_summary"]
    assert "solar_area_polygon_mismatch" in {
        issue["code"] for issue in summary["semantic_issues"]
    }
    assert summary["guarded_measurements"]["solar_present"] == "unknown"
    assert summary["guarded_measurements"]["solar_area_m2"] == "unknown"
    assert {"solar_present", "solar_area_m2"} <= set(summary["pipeline_review_fields"])


def test_vlm_client_forces_guarded_unknown_for_uncertain_solar_mount(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    output = valid_vlm_output()
    output["solar_inventory"]["candidates"] = [
        {
            "candidate_id": "ambiguous-array",
            "image_role": "detail",
            "bbox_normalized": {
                "x_min": 0.2,
                "y_min": 0.2,
                "x_max": 0.22,
                "y_max": 0.22,
            },
            "footprint_polygon_normalized": [
                {"x": 0.2, "y": 0.2},
                {"x": 0.22, "y": 0.2},
                {"x": 0.22, "y": 0.22},
                {"x": 0.2, "y": 0.22},
            ],
            "mount_location": "uncertain",
            "support_structure": "uncertain",
            "support_surface_color": "dark",
            "support_surface_form": "uncertain",
            "surrounding_cues": ["support is occluded"],
            "mount_evidence": "The support beneath the panels cannot be resolved.",
        }
    ]
    output["measurements"]["solar_present"].update(
        {"value": "no", "review_required": True}
    )
    output["measurements"]["solar_area_m2"].update(
        {"value": 0, "review_required": True}
    )
    output["review_fields"] = [
        "solar_present",
        "solar_area_m2",
        *output["review_fields"],
    ]

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="completed", output_text=json.dumps(output))

    saved_path = GeminiVLMClient(root, fake_create).assess(vlm_school(root), mode="pilot")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    summary = saved["derived_solar_summary"]
    assert "solar_presence_inventory_mismatch" in {
        issue["code"] for issue in summary["semantic_issues"]
    }
    assert summary["guarded_measurements"]["solar_present"] == "unknown"
    assert summary["guarded_measurements"]["solar_area_m2"] == "unknown"


def test_vlm_client_requires_unknown_to_use_lowest_confidence_and_review_flag(
    tmp_path: Path,
) -> None:
    root = vlm_test_root(tmp_path)
    output = valid_vlm_output()
    output["measurements"]["perimeter_fencing"]["suggested_confidence"] = 0.6

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="completed", output_text=json.dumps(output))

    client = GeminiVLMClient(root, fake_create)
    with pytest.raises(VLMResponseError, match="must use suggested_confidence 0.20"):
        client.assess(vlm_school(root), mode="pilot")


def test_vlm_client_requires_every_unknown_in_review_fields(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    output = valid_vlm_output()
    output["review_fields"].remove("perimeter_fencing")

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="completed", output_text=json.dumps(output))

    client = GeminiVLMClient(root, fake_create)
    with pytest.raises(VLMResponseError, match="must flag every unknown for review"):
        client.assess(vlm_school(root), mode="pilot")


def test_raw_vlm_evaluation_separates_correct_wrong_and_abstained(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    output = valid_vlm_output()
    output["measurements"]["solar_present"].update(
        {"value": "no", "suggested_confidence": 0.8, "review_required": False}
    )
    output["measurements"]["running_track"].update(
        {"value": "yes", "suggested_confidence": 0.8, "review_required": False}
    )
    output["measurements"]["pool"].update(
        {"value": "unknown", "suggested_confidence": 0.2, "review_required": True}
    )
    (raw_directory / "060483000471.json").write_text(
        json.dumps(
            {
                "parsed_output": output,
                "derived_solar_summary": {
                    "pipeline_review_fields": ["running_track"],
                    "guarded_measurements": {"running_track": "unknown"},
                },
            }
        ),
        encoding="utf-8",
    )
    truth = {
        "school_id": "060483000471",
        **{field: "unknown" for field in MEASUREMENT_FIELDS},
        "solar_present": "no",
        "running_track": "no",
        "pool": "no",
    }

    items, exclusions = raw_observations(raw_directory, [truth], {"060483000471"})
    report = summarize_raw_vlm(items, exclusions)

    assert report["correct_n"] == 1
    assert report["wrong_n"] == 1
    assert report["abstained_unknown_n"] == 1
    assert report["answered_coverage"] == pytest.approx(2 / 3)
    assert report["selective_accuracy"] == pytest.approx(0.5)
    assert report["problem_capture_rate"] == pytest.approx(1.0)
    assert report["silent_error_n"] == 0
    assert report["guarded_pipeline"]["wrong_n"] == 0
    assert report["guarded_pipeline"]["abstained_unknown_n"] == 2
    assert report["guarded_pipeline"]["answered_coverage"] == pytest.approx(1 / 3)


def test_pilot_uncertainty_diagnostic_compares_model_and_pipeline_flags(
    tmp_path: Path,
) -> None:
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    output = valid_vlm_output()
    for suggestion in output["measurements"].values():
        suggestion["review_required"] = False
    output["measurements"]["perimeter_fencing"].update(
        {"value": "none", "suggested_confidence": 0.8}
    )
    output["measurements"]["dominant_fence_type"].update(
        {"value": "unknown", "suggested_confidence": 0.2}
    )
    uncertainty = {
        "pipeline_review_fields": ["dominant_fence_type", "hard_courts"],
        "guarded_measurements": {
            field: suggestion["value"]
            for field, suggestion in output["measurements"].items()
        },
        "auto_accept_candidate_fields": [
            field
            for field in MEASUREMENT_FIELDS
            if field not in {"dominant_fence_type", "hard_courts"}
        ],
    }
    (raw_directory / "060483000471.json").write_text(
        json.dumps(
            {
                "parsed_output": output,
                "uncertainty_assessment": uncertainty,
            }
        ),
        encoding="utf-8",
    )
    reviewed = {
        "school_id": "060483000471",
        **{
            field: str(suggestion["value"])
            for field, suggestion in output["measurements"].items()
        },
        "dominant_fence_type": "chain-link",
        "hard_courts": "0",
    }

    report = pilot_uncertainty_diagnostic(
        raw_directory,
        [reviewed],
        {"060483000471"},
    )
    aggregate = report["aggregate"]

    assert aggregate["raw_correct_n"] == 7
    assert aggregate["raw_wrong_n"] == 1
    assert aggregate["raw_abstained_unknown_n"] == 1
    assert aggregate["model_only_problem_capture_rate"] == pytest.approx(0.5)
    assert aggregate["model_only_silent_wrong_n"] == 1
    assert aggregate["pipeline_problem_capture_rate"] == pytest.approx(1.0)
    assert aggregate["pipeline_silent_wrong_n"] == 0
    assert aggregate["auto_accept_candidate_n"] == 7
    assert aggregate["auto_accept_precision"] == pytest.approx(1.0)


def test_pilot_auditor_diagnostic_measures_misses_and_safety_overrides(
    tmp_path: Path,
) -> None:
    raw_directory = tmp_path / "raw"
    audit_directory = tmp_path / "audits"
    raw_directory.mkdir()
    audit_directory.mkdir()
    output = valid_vlm_output()
    output["measurements"]["perimeter_fencing"]["value"] = "none"
    output["measurements"]["dominant_fence_type"]["value"] = "none"
    raw_record = {
        "parsed_output": output,
        "uncertainty_assessment": {
            "pipeline_review_fields": ["hard_courts"],
            "auto_accept_candidate_fields": [
                field for field in MEASUREMENT_FIELDS if field != "hard_courts"
            ],
        },
    }
    (raw_directory / "060483000471.json").write_text(
        json.dumps(raw_record), encoding="utf-8"
    )
    audit_record = {
        "audited_uncertainty_assessment": {
            "auditor_review_fields": ["pool"],
            "final_review_fields": ["hard_courts", "pool"],
            "final_auto_accept_candidate_fields": [
                field
                for field in MEASUREMENT_FIELDS
                if field not in {"hard_courts", "pool"}
            ],
        },
        "auditor_safety_overrides": [{"field": "hard_courts"}],
    }
    (audit_directory / "060483000471.json").write_text(
        json.dumps(audit_record), encoding="utf-8"
    )
    reviewed = {
        "school_id": "060483000471",
        **{
            field: str(suggestion["value"])
            for field, suggestion in output["measurements"].items()
        },
        "hard_courts": "99",
    }

    report = pilot_auditor_diagnostic(
        raw_directory,
        audit_directory,
        [reviewed],
        {"060483000471"},
    )
    aggregate = report["aggregate"]

    assert aggregate["problem_n"] == 1
    assert aggregate["auditor_problem_capture_n"] == 0
    assert aggregate["auditor_silent_wrong_n"] == 1
    assert aggregate["auditor_safety_override_n"] == 1
    assert aggregate["final_problem_capture_n"] == 1
    assert aggregate["final_silent_wrong_n"] == 0
    assert aggregate["final_auto_accept_n"] == 7
    assert aggregate["final_auto_accept_precision"] == pytest.approx(1.0)


def test_deterministic_evidence_checks_separate_hard_conflicts_from_soft_risks() -> None:
    output = valid_vlm_output()
    output["measurements"]["hard_courts"]["value"] = 3
    next(
        packet
        for packet in output["evidence_packets"]
        if packet["field"] == "running_track"
    )["campus_relation"] = "adjacent"

    summary = _deterministic_evidence_checks(output)

    assert any(
        issue["code"] == "count_component_sum_mismatch"
        and issue["fields"] == ["hard_courts"]
        for issue in summary["hard_conflicts"]
    )
    assert summary["guarded_measurements"]["hard_courts"] == "unknown"
    assert any(
        issue["code"] == "positive_facility_has_ownership_or_boundary_risk"
        and issue["fields"] == ["running_track"]
        for issue in summary["soft_risks"]
    )
    assert summary["guarded_measurements"]["running_track"] == "yes"


def test_soft_scope_forces_fencing_unknown_without_complete_outer_boundary() -> None:
    output = valid_vlm_output()
    output["campus_assessment"]["status"] = "probable"
    output["fencing_inventory"].update(
        {
            "boundary_visibility": "partial",
            "minimum_barrier_coverage_fraction": 0.0,
            "maximum_barrier_coverage_fraction": 0.0,
            "dominant_observed_type": "none",
        }
    )
    output["fencing_inventory"]["segments"][0].update(
        {
            "barrier_fraction": 0.0,
            "unfenced_fraction": 1.0,
            "unobservable_fraction": 0.0,
            "barrier_type": "none",
            "outer_boundary_relation": "unknown",
        }
    )
    output["measurements"]["perimeter_fencing"]["value"] = "none"
    output["measurements"]["dominant_fence_type"]["value"] = "none"
    protocol = json.loads((ROOT / "config" / "vlm_field_protocol.json").read_text())

    summary = _deterministic_evidence_checks(
        output, protocol, scope_mode="soft_boundary"
    )

    codes = {row["code"] for row in summary["hard_conflicts"]}
    assert "non_authoritative_scope_cannot_support_fencing" in codes
    assert summary["guarded_measurements"]["perimeter_fencing"] == "unknown"
    assert summary["guarded_measurements"]["dominant_fence_type"] == "unknown"


def test_item_specific_court_conflict_forces_guarded_unknown() -> None:
    output = valid_vlm_output()
    court_assessment = next(
        row for row in output["feature_assessments"] if row["feature"] == "hard_courts"
    )
    next(
        row
        for row in court_assessment["question_answers"]
        if row["question_id"] == "COURT-06"
    )["answer"] = "no"
    protocol = json.loads((ROOT / "config" / "vlm_field_protocol.json").read_text())

    summary = _deterministic_evidence_checks(output, protocol)

    assert any(
        issue["code"] == "directional_question_conflict"
        and issue["fields"] == ["hard_courts"]
        for issue in summary["hard_conflicts"]
    )
    assert summary["guarded_measurements"]["hard_courts"] == "unknown"


def test_fence_shadow_absence_is_not_negative_evidence() -> None:
    output = valid_vlm_output()
    output["fencing_inventory"].update(
        {
            "boundary_visibility": "adequate",
            "minimum_barrier_coverage_fraction": 0.5,
            "maximum_barrier_coverage_fraction": 0.5,
            "dominant_observed_type": "chain-link",
            "segments": [
                {
                    "segment_id": "all-boundary",
                    "sector": "irregular",
                    "boundary_fraction": 1.0,
                    "barrier_fraction": 0.5,
                    "unfenced_fraction": 0.5,
                    "unobservable_fraction": 0.0,
                    "barrier_type": "chain-link",
                    "shadow_support": "no",
                    "outer_boundary_relation": "yes",
                    "evidence": "Posts and mesh trace half of the outer boundary; no shadow is visible.",
                }
            ],
        }
    )
    output["measurements"]["perimeter_fencing"]["value"] = "partial"
    output["measurements"]["dominant_fence_type"]["value"] = "chain-link"
    for field in ("perimeter_fencing", "dominant_fence_type"):
        packet = next(row for row in output["evidence_packets"] if row["field"] == field)
        packet["visibility"] = "adequate"
        packet["located_observable_facts"] = ["Posts and mesh follow the outer boundary."]
    fencing_assessment = next(
        row for row in output["feature_assessments"] if row["feature"] == "fencing"
    )
    expected = {
        "FENCE-01": "yes",
        "FENCE-02": "yes",
        "FENCE-03": "yes",
        "FENCE-04": "yes",
        "FENCE-05": "no",
        "FENCE-06": "no",
        "FENCE-07": "yes",
        "FENCE-08": "yes",
    }
    for answer in fencing_assessment["question_answers"]:
        answer["answer"] = expected[answer["question_id"]]
    protocol = json.loads((ROOT / "config" / "vlm_field_protocol.json").read_text())

    summary = _deterministic_evidence_checks(output, protocol)

    assert not any(
        "shadow" in issue["code"] or "shadow" in issue["message"].lower()
        for issue in [*summary["hard_conflicts"], *summary["soft_risks"]]
    )
    assert summary["guarded_measurements"]["perimeter_fencing"] == "partial"


def test_solar_area_polygon_tolerance_is_twenty_five_percent(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    school = vlm_school(root)
    config = json.loads((root / "config" / "vlm.json").read_text(encoding="utf-8"))
    output = valid_vlm_output()
    output["solar_inventory"]["candidates"] = [
        {
            "candidate_id": "roof-array",
            "image_role": "detail",
            "bbox_normalized": {
                "x_min": 0.1,
                "y_min": 0.1,
                "x_max": 0.12,
                "y_max": 0.11388889,
            },
            "footprint_polygon_normalized": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.12, "y": 0.1},
                {"x": 0.12, "y": 0.11388889},
                {"x": 0.1, "y": 0.11388889},
            ],
            "mount_location": "school_building_roof",
            "support_structure": "school_building",
            "support_surface_color": "light gray",
            "support_surface_form": "flat",
            "surrounding_cues": ["continuous building roof plane"],
            "mount_evidence": "The panels lie on a traceable school roof.",
        }
    ]
    output["measurements"]["solar_present"].update(
        {"value": "yes", "review_required": True}
    )
    output["measurements"]["solar_area_m2"].update(
        {"value": 124, "review_required": True}
    )
    output["review_fields"] = sorted(
        set(output["review_fields"]) | {"solar_present", "solar_area_m2"}
    )

    within = _validate_solar_inventory(output, school, config)
    assert not any(
        issue["code"] == "solar_area_polygon_mismatch"
        for issue in within["semantic_issues"]
    )

    output["measurements"]["solar_area_m2"]["value"] = 126
    outside = _validate_solar_inventory(output, school, config)
    assert any(
        issue["code"] == "solar_area_polygon_mismatch"
        for issue in outside["semantic_issues"]
    )


def test_text_only_auditor_adds_flags_without_overwriting_primary_values(
    tmp_path: Path,
) -> None:
    root = vlm_test_root(tmp_path)
    output = valid_vlm_output()
    raw_directory = root / "data" / "model_outputs" / "final" / "v1.10"
    raw_directory.mkdir(parents=True)
    primary_record = {
        "configuration_id": "school-facilities-vlm-final-v1.10",
        "school_id": "060483000471",
        "parsed_output": output,
        "derived_solar_summary": {"semantic_issues": []},
        "derived_evidence_summary": {
            "hard_conflicts": [],
            "soft_risks": [],
        },
        "uncertainty_assessment": {
            "pipeline_review_fields": ["hard_courts"],
            "auto_accept_candidate_fields": ["pool", "running_track"],
            "guarded_measurements": {
                field: suggestion["value"]
                for field, suggestion in output["measurements"].items()
            },
        },
    }
    (raw_directory / "060483000471.json").write_text(
        json.dumps(primary_record), encoding="utf-8"
    )

    def audit(status: str = "consistent", action: str = "accept_candidate") -> dict[str, object]:
        return {
            "status": status,
            "issue_codes": [] if status == "consistent" else ["missing_evidence"],
            "question_ids_requiring_review": [] if status == "consistent" else ["POOL-02"],
            "evidence_assessment": "Checked only against the supplied evidence packet.",
            "recommended_action": action,
        }

    audit_output = {
        "schema_version": "1.10.0",
        "school_id": "060483000471",
        "field_audits": {field: audit() for field in MEASUREMENT_FIELDS},
        "overall_issue_codes": ["missing_evidence"],
    }
    audit_output["field_audits"]["pool"] = audit(
        "insufficient_evidence", "targeted_visual_recheck"
    )

    def fake_create(**kwargs: object) -> SimpleNamespace:
        assert [item["type"] for item in kwargs["input"]] == ["text"]
        assert "measurements.csv" not in kwargs["input"][0]["text"]
        assert "authoritative_field_protocol" in kwargs["input"][0]["text"]
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(audit_output),
            id="audit-interaction",
        )

    saved_path = GeminiEvidenceAuditorClient(
        root,
        fake_create,
        ledger_path=tmp_path / "auditor-ledger.json",
    ).audit("060483000471")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved["parsed_output"]["schema_version"] == "1.2"
    assert saved["auditor_vocabulary_normalizations"] == [
        {"path": "schema_version", "from": "1.10.0", "to": "1.2"}
    ]

    assert saved["audited_uncertainty_assessment"]["final_review_fields"] == [
        "hard_courts",
        "pool",
    ]
    assert saved["audited_uncertainty_assessment"][
        "final_auto_accept_candidate_fields"
    ] == ["running_track"]
    assert saved["audited_uncertainty_assessment"][
        "auditor_overwrote_primary_values"
    ] is False
    assert saved["audited_uncertainty_assessment"]["guarded_measurements"] == (
        primary_record["uncertainty_assessment"]["guarded_measurements"]
    )


def test_vlm_client_sends_provider_supported_schema_but_validates_strictly_locally(
    tmp_path: Path,
) -> None:
    root = vlm_test_root(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(valid_vlm_output()),
            id="test-interaction",
        )

    GeminiVLMClient(root, fake_create).assess(vlm_school(root), mode="pilot")
    serialized = json.dumps(calls[0]["response_format"])
    assert "minLength" not in serialized
    assert "uniqueItems" not in serialized
    assert "$defs" not in serialized
    assert "$ref" not in serialized
    assert "anyOf" not in serialized
    local_schema = json.loads((root / "config" / "vlm_response_schema.json").read_text())
    assert local_schema["properties"]["review_fields"]["uniqueItems"] is True


def test_auditor_schema_version_normalization_accepts_known_echoes_only() -> None:
    for source_version in ("1.0", "1.10.0"):
        normalized, changes = _normalize_auditor_output(
            {"schema_version": source_version, "school_id": "example"}
        )
        assert normalized["schema_version"] == "1.2"
        assert changes == [
            {"path": "schema_version", "from": source_version, "to": "1.2"}
        ]

    unchanged, changes = _normalize_auditor_output(
        {"schema_version": "unexpected", "school_id": "example"}
    )
    assert unchanged["schema_version"] == "unexpected"
    assert changes == []


def test_auditor_review_status_normalization_never_promotes_to_consistent() -> None:
    original = {
        "schema_version": "1.2",
        "field_audits": {
            "solar_present": {
                "status": "review",
                "issue_codes": ["solar_presence_inventory_mismatch"],
                "recommended_action": "review",
            },
            "dominant_fence_type": {
                "status": "review",
                "issue_codes": ["negative_answer_without_adequate_visibility"],
                "recommended_action": "review",
            },
        },
    }

    normalized, changes = _normalize_auditor_output(original)

    assert normalized["field_audits"]["solar_present"]["status"] == "contradictory"
    assert normalized["field_audits"]["dominant_fence_type"]["status"] == (
        "insufficient_evidence"
    )
    assert all(
        audit["status"] != "consistent"
        for audit in normalized["field_audits"].values()
    )
    assert len(changes) == 2
    assert original["field_audits"]["solar_present"]["status"] == "review"


def test_auditor_acceptance_cannot_remove_primary_hard_conflict_review() -> None:
    primary = {
        "derived_evidence_summary": {
            "hard_conflicts": [
                {"code": "half_court_exclusion_conflict", "fields": ["hard_courts"]}
            ]
        }
    }
    parsed_audit = {
        "field_audits": {
            "hard_courts": {
                "status": "consistent",
                "recommended_action": "accept_candidate",
            }
        }
    }

    overrides = _auditor_safety_overrides(primary, parsed_audit)

    assert overrides == [
        {
            "field": "hard_courts",
            "auditor_status": "consistent",
            "auditor_recommended_action": "accept_candidate",
            "primary_hard_conflict_codes": ["half_court_exclusion_conflict"],
            "enforced_action": "retain_primary_review_flag",
            "auditor_output_preserved": True,
        }
    ]


def test_vlm_client_exposes_sanitized_provider_error_detail(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)

    class FakeBadRequest(Exception):
        status_code = 400
        body = {
            "error": {
                "code": 400,
                "message": "Unsupported schema keyword uniqueItems",
                "api_key": "AIzaTHIS_IS_A_SECRET_API_KEY_VALUE_123456",
            }
        }

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        raise FakeBadRequest("request rejected")

    with pytest.raises(VLMError) as captured:
        GeminiVLMClient(root, fake_create).assess(vlm_school(root), mode="pilot")
    message = str(captured.value)
    assert "Unsupported schema keyword uniqueItems" in message
    assert "AIzaTHIS" not in message
    assert "[REDACTED" in message


def test_vlm_client_does_not_retry_without_separate_authorization(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    call_count = 0

    class FakeRateLimit(Exception):
        status_code = 429

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        raise FakeRateLimit("rate limited")

    with pytest.raises(VLMError, match="status 429"):
        GeminiVLMClient(root, fake_create).assess(vlm_school(root), mode="pilot")
    assert call_count == 1


def test_auditor_client_does_not_retry_without_separate_authorization(
    tmp_path: Path,
) -> None:
    root = vlm_test_root(tmp_path)
    raw_directory = root / "data" / "model_outputs" / "final" / "v1.10"
    raw_directory.mkdir(parents=True)
    output = valid_vlm_output()
    primary_record = {
        "configuration_id": "school-facilities-vlm-final-v1.10",
        "school_id": "060483000471",
        "parsed_output": output,
        "derived_solar_summary": {"semantic_issues": []},
        "derived_evidence_summary": {"hard_conflicts": [], "soft_risks": []},
        "uncertainty_assessment": {
            "pipeline_review_fields": [],
            "auto_accept_candidate_fields": list(MEASUREMENT_FIELDS),
            "guarded_measurements": {
                field: suggestion["value"]
                for field, suggestion in output["measurements"].items()
            },
        },
    }
    (raw_directory / "060483000471.json").write_text(
        json.dumps(primary_record), encoding="utf-8"
    )
    call_count = 0

    class FakeRateLimit(Exception):
        status_code = 429

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        raise FakeRateLimit("rate limited")

    with pytest.raises(VLMError, match="status 429"):
        GeminiEvidenceAuditorClient(
            root,
            fake_create,
            ledger_path=tmp_path / "auditor-ledger.json",
        ).audit("060483000471")
    assert call_count == 1


def test_auditor_client_preserves_incomplete_provider_diagnostics(
    tmp_path: Path,
) -> None:
    root = vlm_test_root(tmp_path)
    raw_directory = root / "data" / "model_outputs" / "final" / "v1.10"
    raw_directory.mkdir(parents=True)
    output = valid_vlm_output()
    primary_record = {
        "configuration_id": "school-facilities-vlm-final-v1.10",
        "school_id": "060483000471",
        "parsed_output": output,
        "derived_solar_summary": {"semantic_issues": []},
        "derived_evidence_summary": {"hard_conflicts": [], "soft_risks": []},
        "uncertainty_assessment": {
            "pipeline_review_fields": [],
            "auto_accept_candidate_fields": list(MEASUREMENT_FIELDS),
            "guarded_measurements": {
                field: suggestion["value"]
                for field, suggestion in output["measurements"].items()
            },
        },
    }
    (raw_directory / "060483000471.json").write_text(
        json.dumps(primary_record), encoding="utf-8"
    )

    def fake_create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            status="incomplete",
            output_text="partial",
            id="incomplete-auditor",
            usage={"total_output_tokens": 4096},
            errors=[{"code": "MAX_OUTPUT_TOKENS", "message": "limit reached"}],
        )

    client = GeminiEvidenceAuditorClient(
        root,
        fake_create,
        ledger_path=tmp_path / "auditor-ledger.json",
    )
    with pytest.raises(VLMResponseError, match="rejected response preserved"):
        client.audit("060483000471")

    rejected_path = (
        root
        / "data"
        / "model_outputs"
        / "audits"
        / "rejected"
        / "v1.10"
        / "060483000471-incomplete-auditor.json"
    )
    rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
    assert rejected["status"] == "incomplete"
    assert rejected["output_text"] == "partial"
    assert rejected["provider_diagnostics"]["errors"][0]["code"] == (
        "MAX_OUTPUT_TOKENS"
    )


def test_request_ledger_enforces_cross_invocation_per_school_cap(tmp_path: Path) -> None:
    config = json.loads((ROOT / "config" / "vlm.json").read_text(encoding="utf-8"))
    moments = iter(
        [
            datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 30, 12, 0, 13, tzinfo=timezone.utc),
            datetime(2026, 8, 30, 12, 0, 26, tzinfo=timezone.utc),
        ]
    )
    ledger = RequestLedger(tmp_path / "ledger.json", config, now=lambda: next(moments))
    ledger.reserve("060483000471", "pilot", 1)
    ledger.reserve("060483000471", "pilot", 1)
    with pytest.raises(VLMQuotaError, match="per-school request cap"):
        ledger.reserve("060483000471", "pilot", 1)
    records = json.loads((tmp_path / "ledger.json").read_text())["requests"]
    assert [row["attempt"] for row in records] == [1, 2]


class FakeItemSearch:
    def __init__(self, items: list[Item]) -> None:
        self.items = items

    def item_collection(self) -> list[Item]:
        return self.items


class FakeStacClient:
    def __init__(self, items: list[Item]) -> None:
        self.items = items
        self.search_kwargs: dict[str, object] = {}

    def search(self, **kwargs: object) -> FakeItemSearch:
        self.search_kwargs = kwargs
        return FakeItemSearch(self.items)


def local_naip_item(root: Path, captured: datetime) -> Item:
    grid = target_grid(34.061692, -118.412181, 1200, 1200, [1200, 1200])
    raster_path = root / "source_naip.tif"
    data = np.zeros((4, 1200, 1200), dtype=np.uint8)
    data[0] = 80
    data[1] = 120
    data[2] = 160
    data[3] = 200
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=1200,
        height=1200,
        count=4,
        dtype="uint8",
        crs=grid.crs,
        transform=from_bounds(*grid.bounds, width=1200, height=1200),
    ) as destination:
        destination.write(data)
    west, south, east, north = grid.bbox_wgs84
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }
    item = Item(
        id=f"naip-{captured.date().isoformat()}",
        geometry=geometry,
        bbox=[west, south, east, north],
        datetime=captured,
        properties={"gsd": 1.0},
    )
    item.add_asset("image", Asset(href=str(raster_path), media_type="image/tiff"))
    return item


def test_naip_candidate_groups_are_newest_first_and_never_mix_dates(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    old = local_naip_item(root, datetime(2021, 5, 1, tzinfo=timezone.utc))
    new = local_naip_item(root, datetime(2023, 6, 1, tzinfo=timezone.utc))
    groups = candidate_date_groups([old, new], minimum_year=2010)
    assert [str(capture_date) for capture_date, _ in groups] == ["2023-06-01", "2021-05-01"]
    assert all(len({item.datetime.date() for item in items}) == 1 for _, items in groups)


def test_planetary_data_api_transport_returns_exact_target_grid(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    item = local_naip_item(root, datetime(2023, 6, 19, 16, 0, tzinfo=timezone.utc))
    grid = target_grid(35.47467665, -97.52204925, 100, 100, [16, 16])
    data = np.zeros((3, grid.height, grid.width), dtype=np.uint8)
    data[0] = 40
    data[1] = 80
    data[2] = 120
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=3,
            dtype="uint8",
            crs=grid.crs,
            transform=grid.transform,
        ) as destination:
            destination.write(data)
        payload = memory_file.read()
    requested_urls: list[str] = []

    def fake_fetch(url: str) -> bytes:
        requested_urls.append(url)
        return payload

    mosaic, coverage, sources = _mosaic_group_via_data_api(
        [item],
        asset_key="image",
        collection="naip",
        grid=grid,
        endpoint="https://example.test/api/data/v1/item/bbox",
        fetch_bytes=fake_fetch,
    )
    assert mosaic.shape == (3, 16, 16)
    assert coverage == 1.0
    assert sources[0]["item_id"] == item.id
    assert "/16x16.tif?" in requested_urls[0]
    assert "asset_bidx=image%7C1%2C2%2C3" in requested_urls[0]
    assert "dst_crs=EPSG%3A32614" in requested_urls[0]


def test_naip_context_acquisition_is_georeferenced_dated_and_reproducible(
    tmp_path: Path,
) -> None:
    root = vlm_test_root(tmp_path)
    item = local_naip_item(root, datetime(2022, 5, 11, 16, 0, tzinfo=timezone.utc))
    stac = FakeStacClient([item])
    result = fetch_naip_product(
        root=root,
        school_id="060483000471",
        school_name="Beverly Hills High",
        product="context",
        center_latitude=34.061692,
        center_longitude=-118.412181,
        requested_latitude=34.061692,
        requested_longitude=-118.412181,
        campus_resolution_status="unresolved",
        campus_resolution_notes="Initial context; human review pending.",
        stac_client=stac,
        signer=lambda value: value,
        retrieved_at=lambda: datetime(2026, 8, 30, tzinfo=timezone.utc),
    )
    assert result.downloaded
    assert stac.search_kwargs["collections"] == ["naip"]
    with rasterio.open(result.geotiff_path) as dataset:
        assert dataset.count == 3
        assert (dataset.width, dataset.height) == (1600, 1600)
        assert dataset.crs.to_epsg() == 32611
        assert pytest.approx(abs(dataset.transform.a)) == 0.625
    with Image.open(result.jpeg_path) as image:
        assert image.format == "JPEG"
        assert image.size == (1600, 1600)
    metadata = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert metadata["configuration_id"] == "school-facilities-imagery-pilot-v1.6"
    assert metadata["capture_datetime_or_vintage"] == "2022-05-11T16:00:00Z"
    assert metadata["target_coverage_fraction"] >= 0.995
    assert "?" not in metadata["asset_reference_without_expired_credentials"][0]

    cached = fetch_naip_product(
        root=root,
        school_id="060483000471",
        school_name="Beverly Hills High",
        product="context",
        center_latitude=34.061692,
        center_longitude=-118.412181,
        requested_latitude=34.061692,
        requested_longitude=-118.412181,
        campus_resolution_status="unresolved",
        campus_resolution_notes="Initial context; human review pending.",
        stac_client=stac,
        signer=lambda value: value,
    )
    assert not cached.downloaded


def test_naip_detail_uses_adaptive_extent_and_rejects_stale_extent_cache(
    tmp_path: Path,
) -> None:
    root = vlm_test_root(tmp_path)
    item = local_naip_item(root, datetime(2022, 5, 11, 16, 0, tzinfo=timezone.utc))
    stac = FakeStacClient([item])
    common = {
        "root": root,
        "school_id": "060483000471",
        "school_name": "Beverly Hills High",
        "product": "detail",
        "center_latitude": 34.061692,
        "center_longitude": -118.412181,
        "requested_latitude": 34.061692,
        "requested_longitude": -118.412181,
        "campus_resolution_status": "confirmed",
        "campus_resolution_notes": "Confirmed public campus boundary.",
        "stac_client": stac,
        "signer": lambda value: value,
    }
    first = fetch_naip_product(**common, extent_m=250)
    assert first.downloaded is True
    with rasterio.open(first.geotiff_path) as dataset:
        assert abs(dataset.transform.a) == pytest.approx(250 / 1600)
    metadata = json.loads(first.sidecar_path.read_text(encoding="utf-8"))
    assert metadata["requested_extent_m"] == {"width": 250.0, "height": 250.0}

    cached = fetch_naip_product(**common, extent_m=250)
    assert cached.downloaded is False

    replaced = fetch_naip_product(**common, extent_m=300)
    assert replaced.downloaded is True
    with rasterio.open(replaced.geotiff_path) as dataset:
        assert abs(dataset.transform.a) == pytest.approx(300 / 1600)
    history = list((replaced.geotiff_path.parent / "history").glob("detail-*"))
    assert len(history) == 1
    assert (history[0] / "detail.tif").is_file()
    assert (history[0] / "detail.jpg").is_file()
    assert (history[0] / "detail.json").is_file()
    replacement_metadata = json.loads(replaced.sidecar_path.read_text(encoding="utf-8"))
    assert replacement_metadata["superseded_product_archive"].startswith(
        "data/imagery/060483000471/history/detail-"
    )


def approved_vlm_pair(root: Path) -> None:
    item = local_naip_item(root, datetime(2022, 5, 11, 16, 0, tzinfo=timezone.utc))
    stac = FakeStacClient([item])
    common = {
        "root": root,
        "school_id": "060483000471",
        "school_name": "Beverly Hills High",
        "center_latitude": 34.061692,
        "center_longitude": -118.412181,
        "requested_latitude": 34.061692,
        "requested_longitude": -118.412181,
        "stac_client": stac,
        "signer": lambda value: value,
    }
    fetch_naip_product(
        **common,
        product="context",
        campus_resolution_status="unresolved",
        campus_resolution_notes="Context review pending.",
    )
    fetch_naip_product(
        **common,
        product="detail",
        campus_resolution_status="confirmed",
        campus_resolution_notes="Confirmed public campus boundary.",
        extent_m=250,
    )
    image_dir = root / "data" / "imagery" / "060483000471"

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    review_dir = root / "data" / "reviewed" / "060483000471"
    review_dir.mkdir(parents=True)
    review = {
        "school_id": "060483000471",
        "school_name": "Beverly Hills High",
        "review_status": "confirmed",
        "boundary_notes": "Confirmed public campus boundary.",
        "source_element": "https://www.openstreetmap.org/way/123",
        "geometry": [
            [34.0622, -118.4127],
            [34.0622, -118.4117],
            [34.0612, -118.4117],
            [34.0612, -118.4127],
            [34.0622, -118.4127],
        ],
        "image_pair_approved": True,
        "public_source_and_non_sensitive_confirmed": True,
        "approved_gemini_inputs": [
            "data/imagery/060483000471/context.jpg",
            "data/imagery/060483000471/detail.jpg",
        ],
        "approved_gemini_input_sha256": {
            "context.jpg": sha256(image_dir / "context.jpg"),
            "detail.jpg": sha256(image_dir / "detail.jpg"),
        },
        "approved_supporting_artifact_sha256": {
            "detail.tif": sha256(image_dir / "detail.tif"),
        },
    }
    (review_dir / "campus.json").write_text(json.dumps(review), encoding="utf-8")


def test_approved_image_pair_loads_as_frozen_vlm_input(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    approved_vlm_pair(root)
    school = load_approved_school_input(root, "060483000471")
    request = build_interaction_request(root, school)
    assert school.context.capture_vintage == school.detail.capture_vintage
    assert len(school.campus_boundary_detail_normalized) == 5
    assert [item["data"].name for item in request["input"] if item["type"] == "image"] == [
        "context.jpg",
        "detail.jpg",
    ]


def test_approved_image_pair_hash_prevents_silent_replacement(tmp_path: Path) -> None:
    root = vlm_test_root(tmp_path)
    approved_vlm_pair(root)
    detail = root / "data" / "imagery" / "060483000471" / "detail.jpg"
    detail.write_bytes(detail.read_bytes() + b"changed")
    with pytest.raises(VLMConfigurationError, match="detail image hash does not match"):
        load_approved_school_input(root, "060483000471")
