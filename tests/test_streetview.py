from __future__ import annotations

import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests
from PIL import Image

from school_facilities.credentials import load_api_key, upsert_api_key
from school_facilities.streetview import (
    StreetViewBudgetError,
    StreetViewConfigurationError,
    StreetViewLedger,
    _request_fingerprint,
    budget_status,
    cost_estimate,
    create_probe_plan,
    delete_temporary_images,
    fetch_images,
    preflight_fetch,
    probe_metadata,
    record_usage_snapshot,
    validate_street_response,
    _default_get,
)
from school_facilities.vlm import _gemini_response_schema
from school_facilities.streetview_vlm import (
    _apply_v1_11_uncertainty_guards,
    _normalize_v1_11_vocabulary,
    reconcile_rejected_v1_11,
)


REPOSITORY = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload=None, content: bytes = b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload


def _root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPOSITORY / "config" / "streetview_v1_11.json", tmp_path / "config")
    return tmp_path


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (640, 640), "green").save(buffer, format="JPEG")
    return buffer.getvalue()


def _image_manifest(root: Path, *, school_id: str = "abc") -> Path:
    request = {
        "school_id": school_id,
        "panorama_id": "pano-1",
        "heading": 123.0,
        "pitch": 5.0,
        "field_of_view": 75.0,
        "size": [640, 640],
        "source": "outdoor",
    }
    fingerprint = _request_fingerprint(request)
    request.update(
        {
            "request_fingerprint_sha256": fingerprint,
            "image_id": f"sv-{fingerprint[:12]}",
            "capture_vintage": "2025-06",
            "copyright": "Copyright Google",
            "panorama_location": {"latitude": 1.0, "longitude": 2.0},
        }
    )
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "streetview_image_manifest",
                "configuration_id": "school-facilities-streetview-v1.11",
                "maximum_billable_image_requests": 1,
                "schools": [
                    {
                        "school_id": school_id,
                        "school_name": "Example School",
                        "image_requests": [request],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_upsert_key_preserves_existing_key(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.local.env"
    secrets.write_text("GEMINI_API_KEY=gemini-secret\n", encoding="utf-8")
    upsert_api_key(
        "street-secret",
        environment_variable="GOOGLE_STREET_VIEW_API_KEY",
        secrets_file=secrets,
        provider_name="Google Street View",
    )
    assert load_api_key(
        environment_variable="GEMINI_API_KEY", secrets_file=secrets, environment={}
    )[0] == "gemini-secret"
    assert load_api_key(
        environment_variable="GOOGLE_STREET_VIEW_API_KEY",
        secrets_file=secrets,
        environment={},
    )[0] == "street-secret"


def test_v1_11_schema_projects_const_one_of_and_all_of() -> None:
    schema = json.loads(
        (REPOSITORY / "config" / "streetview_vlm_response_schema_v1_11.json")
        .read_text(encoding="utf-8")
    )
    projected = _gemini_response_schema(schema)
    properties = projected["properties"]
    assert properties["schema_version"] == {"type": "string"}
    count_value = properties["candidate_fields"]["properties"][
        "portable_classroom_count"
    ]["properties"]["value"]
    assert count_value == {}
    binary = properties["candidate_fields"]["properties"]["running_track"]
    assert binary["type"] == "object"
    assert set(binary["required"]) == {
        "value", "suggested_confidence", "evidence", "review_required"
    }


def test_probe_plan_is_no_network_and_deterministic(tmp_path: Path) -> None:
    first = create_probe_plan(
        REPOSITORY,
        school_ids=["060483000471"],
        output_path=tmp_path / "first.json",
        now=lambda: NOW,
    )
    second = create_probe_plan(
        REPOSITORY,
        school_ids=["060483000471"],
        output_path=tmp_path / "second.json",
        now=lambda: NOW,
    )
    assert first.read_bytes() == second.read_bytes()
    value = json.loads(first.read_text(encoding="utf-8"))
    assert value["network_requests_made"] is False
    assert len(value["schools"][0]["fallback_probe_points"]) == 8


def test_metadata_probe_uses_osm_and_deduplicates_panorama(tmp_path: Path) -> None:
    plan = create_probe_plan(
        REPOSITORY,
        school_ids=["060483000471"],
        output_path=tmp_path / "plan.json",
        now=lambda: NOW,
    )

    def road_poster(url, *, data, headers, timeout):
        assert "overpass" in url
        assert headers["User-Agent"]
        return FakeResponse(
            payload={
                "elements": [
                    {
                        "geometry": [
                            {"lat": 34.061, "lon": -118.414},
                            {"lat": 34.062, "lon": -118.414},
                        ]
                    }
                ]
            }
        )

    def getter(url, *, params, timeout):
        assert params["key"] == "secret"
        return FakeResponse(
            payload={
                "status": "OK",
                "pano_id": "same-panorama",
                "date": "2025-06",
                "copyright": "Copyright Google",
                "location": {"lat": 34.0615, "lng": -118.414},
            }
        )

    output = probe_metadata(
        REPOSITORY,
        plan_path=plan,
        output_path=tmp_path / "images.json",
        environment={"GOOGLE_STREET_VIEW_API_KEY": "secret"},
        getter=getter,
        road_poster=road_poster,
        now=lambda: NOW,
        sleep=lambda _: None,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    school = value["schools"][0]
    assert school["road_point_source"] == "openstreetmap_roads"
    assert school["unique_panorama_count"] == 1
    assert school["selected_panorama_count"] == 1
    assert len(school["image_requests"]) == 2
    assert value["billable_image_requests_made"] is False


def test_metadata_probe_rejects_contributed_panorama(tmp_path: Path) -> None:
    plan = create_probe_plan(
        REPOSITORY,
        school_ids=["060483000471"],
        output_path=tmp_path / "plan.json",
        now=lambda: NOW,
    )

    def road_poster(url, *, data, headers, timeout):
        return FakeResponse(
            payload={"elements": [{"geometry": [{"lat": 34.061, "lon": -118.414}]}]}
        )

    def getter(url, *, params, timeout):
        return FakeResponse(
            payload={
                "status": "OK",
                "pano_id": "contributed-panorama",
                "date": "2025-06",
                "copyright": "Copyright Local Photographer",
                "location": {"lat": 34.0615, "lng": -118.414},
            }
        )

    output = probe_metadata(
        REPOSITORY,
        plan_path=plan,
        output_path=tmp_path / "images.json",
        environment={"GOOGLE_STREET_VIEW_API_KEY": "secret"},
        getter=getter,
        road_poster=road_poster,
        now=lambda: NOW,
        sleep=lambda _: None,
    )
    school = json.loads(output.read_text(encoding="utf-8"))["schools"][0]
    assert school["contributed_panorama_rejected_count"] == 1
    assert school["selected_panorama_count"] == 0
    assert school["image_requests"] == []


def test_preflight_rejects_contributed_panorama(tmp_path: Path) -> None:
    root = _root(tmp_path)
    manifest = _image_manifest(root)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["schools"][0]["image_requests"][0]["copyright"] = "Copyright Contributor"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(StreetViewConfigurationError, match="contributed panorama"):
        preflight_fetch(root, manifest, now=lambda: NOW)


def test_preflight_fails_closed_without_current_usage_snapshot(tmp_path: Path) -> None:
    root = _root(tmp_path)
    manifest = _image_manifest(root)
    result = preflight_fetch(root, manifest, now=lambda: NOW)
    assert result["allowed"] is False
    assert "snapshot is missing" in result["refusal_reasons"][0]


def test_live_fetch_requires_zero_paid_budget_and_provider_quota(tmp_path: Path) -> None:
    root = _root(tmp_path)
    manifest = _image_manifest(root)
    record_usage_snapshot(
        root,
        month="2026-09",
        used_requests=0,
        source="test console",
        now=lambda: NOW,
    )
    with pytest.raises(StreetViewBudgetError, match="max-paid-usd 0"):
        fetch_images(
            root,
            manifest_path=manifest,
            live=True,
            max_paid_usd=None,
            provider_quota_confirmed=True,
            environment={"GOOGLE_STREET_VIEW_API_KEY": "secret"},
            now=lambda: NOW,
        )
    with pytest.raises(StreetViewBudgetError, match="quota confirmation"):
        fetch_images(
            root,
            manifest_path=manifest,
            live=True,
            max_paid_usd=0,
            provider_quota_confirmed=False,
            environment={"GOOGLE_STREET_VIEW_API_KEY": "secret"},
            now=lambda: NOW,
        )


def test_live_fetch_records_once_and_blocks_duplicate(tmp_path: Path) -> None:
    root = _root(tmp_path)
    manifest = _image_manifest(root)
    record_usage_snapshot(
        root,
        month="2026-09",
        used_requests=5,
        source="test console",
        now=lambda: NOW,
    )

    def getter(url, *, params, timeout):
        assert url.endswith("/streetview")
        assert params["key"] == "secret"
        return FakeResponse(content=_jpeg())

    fetched = fetch_images(
        root,
        manifest_path=manifest,
        live=True,
        max_paid_usd=0,
        provider_quota_confirmed=True,
        environment={"GOOGLE_STREET_VIEW_API_KEY": "secret"},
        getter=getter,
        now=lambda: NOW,
        sleep=lambda _: None,
    )
    value = json.loads(Path(fetched).read_text(encoding="utf-8"))
    assert (root / value["images"][0]["temporary_image_path"]).is_file()
    ledger = StreetViewLedger(root / "data/streetview/control/request_ledger_v1.11.jsonl")
    assert len([r for r in ledger.records() if r["event"] == "image_request_reserved"]) == 1
    with pytest.raises(StreetViewBudgetError, match="fingerprints already exist"):
        fetch_images(
            root,
            manifest_path=manifest,
            live=True,
            max_paid_usd=0,
            provider_quota_confirmed=True,
            environment={"GOOGLE_STREET_VIEW_API_KEY": "secret"},
            getter=getter,
            now=lambda: NOW,
            sleep=lambda _: None,
        )
    assert delete_temporary_images(root, Path(fetched), now=lambda: NOW) == 1


def _valid_response(image_id: str) -> dict:
    observation = {
        "image_id": image_id,
        "school_visible": False,
        "campus_sector": "west",
        "boundary_segment": "west edge",
        "adequately_visible_boundary_fraction": 0,
        "occlusion": "trees",
        "resolution": "adequate only for large objects",
        "fence_cues": "none",
        "portable_cues": "none",
        "athletic_cues": "none",
        "fencing": {"result": "negative_visible_segment", "observable_evidence": "none", "limitations": "trees"},
        "portable_classrooms": {"result": "unknown", "observable_evidence": "none", "limitations": "trees"},
        "athletics": {"result": "unknown", "observable_evidence": "none", "limitations": "trees"},
    }
    base = {"suggested_confidence": 0.2, "evidence": "insufficient", "review_required": True}
    return {
        "schema_version": "1.11.0",
        "school_id": "abc",
        "image_observations": [observation],
        "candidate_fields": {
            "portable_classroom_count": {"value": "unknown", **base},
            "perimeter_fencing": {"value": "full", "minimum_supported_coverage": 0.3, "maximum_supported_coverage": 0.6, **base},
            "dominant_fence_type": {"value": "unknown", **base},
            "running_track": {"value": "unknown", **base},
            "full_size_sports_fields": {"value": "unknown", **base},
            "hard_courts": {"value": "unknown", **base},
            "pool": {"value": "unknown", **base},
        },
        "pipeline_review_required": True,
        "review_reasons": [],
    }


def test_deterministic_response_guards_fencing_and_weak_negatives() -> None:
    guarded = validate_street_response(
        _valid_response("sv-one"), school_id="abc", expected_image_ids={"sv-one"}
    )
    assert guarded["candidate_fields"]["perimeter_fencing"]["value"] == "unknown"
    assert guarded["image_observations"][0]["fencing"]["result"] == "unknown"
    assert set(guarded["review_reasons"]) == {
        "deterministic_fencing_coverage_conflict",
        "negative_without_adequate_visibility",
    }


def test_v1_11_normalization_is_narrow_and_preserves_raw_value() -> None:
    raw = _valid_response("sv-one")
    raw["candidate_fields"]["dominant_fence_type"]["value"] = "chain_link"
    normalized, changes = _normalize_v1_11_vocabulary(raw)
    assert raw["candidate_fields"]["dominant_fence_type"]["value"] == "chain_link"
    assert normalized["candidate_fields"]["dominant_fence_type"]["value"] == "chain-link"
    assert changes == [
        {
            "path": "candidate_fields.dominant_fence_type.value",
            "from": "chain_link",
            "to": "chain-link",
        }
    ]


def test_v1_11_binary_none_normalizes_and_reconciles_without_request(tmp_path: Path) -> None:
    root = _root(tmp_path)
    output_dir = root / "data/model_outputs/streetview/v1.11"
    output_dir.mkdir(parents=True)
    raw = _valid_response("sv-one")
    raw["schema_version"] = "1.11"
    raw["candidate_fields"]["pool"]["value"] = "none"
    rejected = {
        "configuration_id": "school-facilities-streetview-vlm-v1.11",
        "school_id": "abc",
        "street_image_ids": ["sv-one"],
        "parsed_output": raw,
        "validation_error": "pool used none",
    }
    (output_dir / "abc-rejected.json").write_text(
        json.dumps(rejected), encoding="utf-8"
    )
    output = reconcile_rejected_v1_11(root, school_id="abc")
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["parsed_output"]["candidate_fields"]["pool"]["value"] == "none"
    assert value["parsed_output"]["schema_version"] == "1.11"
    assert value["normalized_output"]["candidate_fields"]["pool"]["value"] == "no"
    assert value["normalized_output"]["schema_version"] == "1.11.0"
    assert value["offline_reconciled"] is True
    ledger = StreetViewLedger(output_dir / "gemini_request_ledger.jsonl")
    assert ledger.records()[-1]["provider_request_made"] is False


def test_v1_11_uncertainty_flags_raw_disagreement_and_small_play_area(tmp_path: Path) -> None:
    root = _root(tmp_path)
    baseline_path = root / "data/model_outputs/final/v1.10/abc.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        json.dumps(
            {
                "parsed_output": {
                    "measurements": {
                        name: {"value": field["value"]}
                        for name, field in _valid_response("sv-one")["candidate_fields"].items()
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    response = _valid_response("sv-one")
    response["candidate_fields"]["full_size_sports_fields"]["value"] = 1
    response["candidate_fields"]["hard_courts"].update(
        {"value": 1, "evidence": "A small basketball play area is visible."}
    )
    guarded, comparison = _apply_v1_11_uncertainty_guards(
        root, response, school_id="abc"
    )
    assert guarded["pipeline_review_required"] is True
    assert guarded["candidate_fields"]["full_size_sports_fields"]["review_required"] is True
    assert guarded["candidate_fields"]["hard_courts"]["review_required"] is True
    assert "aerial_v1_10_raw_disagreement:full_size_sports_fields" in guarded["review_reasons"]
    assert "hard_court_evidence_may_describe_excluded_partial_play_area" in guarded["review_reasons"]
    assert len(comparison) >= 2


def test_budget_counts_only_local_reservations_after_snapshot(tmp_path: Path) -> None:
    root = _root(tmp_path)
    record_usage_snapshot(
        root,
        month="2026-09",
        used_requests=10,
        source="test console",
        now=lambda: NOW,
    )
    ledger = StreetViewLedger(
        root / "data/streetview/control/request_ledger_v1.11.jsonl",
        now=lambda: NOW.replace(hour=13),
    )
    ledger.append({"event": "image_request_reserved", "request_fingerprint_sha256": "x"})
    result = budget_status(root, proposed_requests=2, now=lambda: NOW.replace(hour=14), require_current_snapshot=True)
    assert result["effective_requests_used"] == 11
    assert result["allowed"] is True


def test_scale_cost_estimator_applies_progressive_tiers() -> None:
    result = cost_estimate(REPOSITORY, schools=130_000, views_per_school=8)
    assert result["monthly_image_requests"] == 1_040_000
    assert result["estimated_monthly_cost_usd"] == 5054.0


def test_take_home_cost_estimate_is_zero() -> None:
    result = cost_estimate(REPOSITORY, schools=25, views_per_school=8)
    assert result["estimated_monthly_cost_usd"] == 0.0


def test_network_exception_never_exposes_query_parameters(monkeypatch) -> None:
    secret = "AIza-test-secret-that-must-never-appear"

    def fail(*args, **kwargs):
        raise requests.exceptions.SSLError(
            f"failed URL https://maps.googleapis.com/example?key={secret}"
        )

    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(Exception) as captured:
        _default_get(
            "https://maps.googleapis.com/maps/api/streetview/metadata",
            params={"key": secret},
            timeout=1,
        )
    assert secret not in str(captured.value)
    assert "parameters were redacted" in str(captured.value)
