from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from school_facilities.streetview_evaluation import STREET_FIELDS, evaluate_stability, evaluate_v1_11


REPOSITORY = Path(__file__).resolve().parents[1]


def _candidate_fields(*, hard_courts=0, confidence=0.8, flagged=False):
    base = {"suggested_confidence": confidence, "evidence": "synthetic test", "review_required": flagged}
    return {
        "portable_classroom_count": {"value": 0, **base},
        "perimeter_fencing": {"value": "none", **base},
        "dominant_fence_type": {"value": "none", **base},
        "running_track": {"value": "no", **base},
        "full_size_sports_fields": {"value": 0, **base},
        "hard_courts": {"value": hard_courts, **base},
        "pool": {"value": "no", **base},
    }


def _write_prediction(path: Path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"guarded_output": {"candidate_fields": _candidate_fields(**kwargs)}}),
        encoding="utf-8",
    )


def test_frozen_evaluator_reports_wrong_unknown_and_silent_error(tmp_path: Path) -> None:
    pilot = json.loads((REPOSITORY / "config/pilot_schools.json").read_text(encoding="utf-8"))
    ids = pilot["excluded_validation_school_ids"]
    reference = tmp_path / "reference.csv"
    columns = [
        "school_id", "school_name", "city", "state", "solar_present", "solar_area_m2",
        *STREET_FIELDS, "verification_notes",
    ]
    with reference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for school_id in ids:
            writer.writerow(
                {
                    "school_id": school_id,
                    "school_name": "Synthetic",
                    "city": "Synthetic",
                    "state": "ZZ",
                    "solar_present": "no",
                    "solar_area_m2": 0,
                    "portable_classroom_count": 0,
                    "perimeter_fencing": "none",
                    "dominant_fence_type": "none",
                    "running_track": "no",
                    "full_size_sports_fields": 0,
                    "hard_courts": 0,
                    "pool": "no",
                    "verification_notes": "synthetic",
                }
            )
    predictions = tmp_path / "predictions"
    for index, school_id in enumerate(ids):
        _write_prediction(
            predictions / f"{school_id}.json",
            hard_courts=(1 if index == 0 else 0),
            flagged=False,
        )
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    result = evaluate_v1_11(
        REPOSITORY,
        reference_path=reference,
        reference_sha256=digest,
        prediction_directory=predictions,
    )
    assert result["overall"]["wrong_n"] == 1
    assert result["overall"]["silent_error_n"] == 1
    assert result["overall"]["known_reference_n"] == len(ids) * len(STREET_FIELDS)


def test_stability_reports_value_confidence_and_flag_drift(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    repeat = tmp_path / "repeat"
    _write_prediction(primary / "one.json")
    _write_prediction(repeat / "one.json", hard_courts=1, confidence=0.6, flagged=True)
    result = evaluate_stability(
        school_ids=["one"], primary_directory=primary, repeat_directories=[repeat]
    )
    assert result["comparison_n"] == len(STREET_FIELDS)
    assert result["exact_value_agreement"] == 6 / 7
    assert result["mean_confidence_absolute_drift"] == pytest.approx(0.2)
    assert result["flag_agreement"] == 0.0
