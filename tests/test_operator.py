from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from school_facilities.operator import (
    _prompt_confidence,
    authorize_validation_unblinding,
    prepare_blind_review_packet,
    prepare_review_packet,
    workflow_rows,
)
from school_facilities.schema import (
    GROUND_TRUTH_COLUMNS,
    MEASUREMENT_COLUMNS,
    MEASUREMENT_FIELDS,
)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _validation_root(tmp_path: Path) -> tuple[Path, list[str]]:
    ids = ["000000000001", "000000000002"]
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "pilot_schools.json").write_text(
        json.dumps({"excluded_validation_school_ids": ids}), encoding="utf-8"
    )
    _write_csv(
        tmp_path / "schools_sample.csv",
        ["school_id", "school_name", "city", "state"],
        [
            {
                "school_id": school_id,
                "school_name": f"School {school_id}",
                "city": "City",
                "state": "CA",
            }
            for school_id in ids
        ],
    )
    rows = []
    for school_id in ids:
        row = {column: "" for column in GROUND_TRUTH_COLUMNS}
        row.update(
            {
                "school_id": school_id,
                "school_name": f"School {school_id}",
                "city": "City",
                "state": "CA",
            }
        )
        rows.append(row)
    _write_csv(
        tmp_path / "data" / "validation" / "ground_truth.csv",
        GROUND_TRUTH_COLUMNS,
        rows,
    )
    return tmp_path, ids


def test_validation_unblinding_requires_all_rows_and_exact_hash(tmp_path: Path) -> None:
    root, ids = _validation_root(tmp_path)
    with pytest.raises(ValueError, match="must be complete and valid"):
        authorize_validation_unblinding(root, ids[0], "not-a-hash")

    truth_path = root / "data" / "validation" / "ground_truth.csv"
    rows = list(csv.DictReader(truth_path.open(encoding="utf-8")))
    for row in rows:
        for field in MEASUREMENT_FIELDS:
            row[field] = "unknown"
        row["verification_notes"] = "Imagery was insufficient for a defensible label."
    _write_csv(truth_path, GROUND_TRUTH_COLUMNS, rows)
    digest = hashlib.sha256(truth_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="does not match"):
        authorize_validation_unblinding(root, ids[0], "0" * 64)
    authorize_validation_unblinding(root, ids[0], digest)


def test_blind_packet_contains_images_but_no_model_material(tmp_path: Path) -> None:
    root, ids = _validation_root(tmp_path)
    school_id = ids[0]
    _write_csv(
        root / "schools_sample.csv",
        ["school_id", "school_name", "city", "state"],
        [{"school_id": school_id, "school_name": "Blind School", "city": "City", "state": "CA"}],
    )
    image_dir = root / "data" / "imagery" / school_id
    image_dir.mkdir(parents=True)
    (image_dir / "context.jpg").write_bytes(b"context-image")
    (image_dir / "detail.jpg").write_bytes(b"detail-image")
    quarantine = root / "data" / "model_outputs" / "quarantine" / "v1.10" / "raw"
    quarantine.mkdir(parents=True)
    (quarantine / f"{school_id}.json").write_text(
        '{"secret_prediction":"DO_NOT_LEAK"}', encoding="utf-8"
    )

    output = prepare_blind_review_packet(root, school_id)
    text = output.read_text(encoding="utf-8")
    assert "context.jpg" in text
    assert "detail.jpg" in text
    assert "DO_NOT_LEAK" not in text
    assert "Raw VLM" not in text


def test_review_packet_expands_authoritative_question_text(tmp_path: Path) -> None:
    root = tmp_path
    school_id = "000000000003"
    (root / "config").mkdir()
    (root / "config" / "pilot_schools.json").write_text(
        '{"excluded_validation_school_ids":[]}', encoding="utf-8"
    )
    (root / "config" / "vlm_field_protocol.json").write_text(
        json.dumps(
            {
                "features": {
                    "solar": {
                        "questions": [
                            {"id": "SOL-01", "question": "Are all roofs visible?"}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        root / "schools_sample.csv",
        ["school_id", "school_name", "city", "state"],
        [{"school_id": school_id, "school_name": "Review School", "city": "City", "state": "CA"}],
    )
    image_dir = root / "data" / "imagery" / school_id
    image_dir.mkdir(parents=True)
    (image_dir / "context.jpg").write_bytes(b"context")
    (image_dir / "detail.jpg").write_bytes(b"detail")
    (image_dir / "detail.json").write_text(
        json.dumps(
            {
                "source": "USDA National Agriculture Imagery Program (NAIP)",
                "capture_datetime_or_vintage": "2024-01-02T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    raw_dir = root / "data" / "model_outputs" / "final" / "v1.10"
    raw_dir.mkdir(parents=True)
    measurements = {
        field: {
            "value": "unknown",
            "suggested_confidence": 0.2,
            "evidence": "insufficient",
        }
        for field in MEASUREMENT_FIELDS
    }
    (raw_dir / f"{school_id}.json").write_text(
        json.dumps(
            {
                "parsed_output": {
                    "measurements": measurements,
                    "feature_assessments": [
                        {
                            "feature": "solar",
                            "derivation_summary": "Checked roofs.",
                            "question_answers": [
                                {
                                    "question_id": "SOL-01",
                                    "answer": "no",
                                    "location": "campus",
                                    "observation": "tree cover",
                                }
                            ],
                        }
                    ],
                },
                "uncertainty_assessment": {
                    "guarded_measurements": {field: "unknown" for field in MEASUREMENT_FIELDS},
                    "pipeline_review_fields": MEASUREMENT_FIELDS,
                    "review_reasons_by_field": {},
                },
            }
        ),
        encoding="utf-8",
    )

    output = prepare_review_packet(root, school_id)
    text = output.read_text(encoding="utf-8")
    assert "Are all roofs visible?" in text
    assert "Raw VLM" in text


def test_unknown_confidence_reprompts_until_point_two() -> None:
    answers = iter(["not-a-number", "0.80", "0.20"])
    score, note = _prompt_confidence(
        "pool", "unknown", input_func=lambda _: next(answers)
    )
    assert score == "0.20"
    assert note == ""


def test_validation_waits_for_all_nonvalidation_reviews(tmp_path: Path) -> None:
    validation_id = "000000000010"
    ordinary_id = "000000000011"
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "pilot_schools.json").write_text(
        json.dumps({"excluded_validation_school_ids": [validation_id]}),
        encoding="utf-8",
    )
    _write_csv(
        tmp_path / "schools_sample.csv",
        ["school_id", "school_name", "city", "state"],
        [
            {"school_id": validation_id, "school_name": "Validation", "city": "C", "state": "CA"},
            {"school_id": ordinary_id, "school_name": "Ordinary", "city": "C", "state": "CA"},
        ],
    )
    measurement_rows = []
    for school_id, name in ((validation_id, "Validation"), (ordinary_id, "Ordinary")):
        row = {column: "" for column in MEASUREMENT_COLUMNS}
        row.update(
            {
                "school_id": school_id,
                "school_name": name,
                "city": "C",
                "state": "CA",
                "review_status": "unreviewed",
            }
        )
        measurement_rows.append(row)
    _write_csv(tmp_path / "measurements.csv", MEASUREMENT_COLUMNS, measurement_rows)
    truth_rows = []
    for school_id, name in ((validation_id, "Validation"), (ordinary_id, "Ordinary")):
        row = {column: "" for column in GROUND_TRUTH_COLUMNS}
        row.update({"school_id": school_id, "school_name": name, "city": "C", "state": "CA"})
        truth_rows.append(row)
    _write_csv(
        tmp_path / "data" / "validation" / "ground_truth.csv",
        GROUND_TRUTH_COLUMNS,
        truth_rows,
    )
    quarantine = tmp_path / "data" / "model_outputs" / "quarantine" / "v1.10" / "raw"
    quarantine.mkdir(parents=True)
    (quarantine / f"{validation_id}.json").write_text("{}", encoding="utf-8")
    ordinary_raw = tmp_path / "data" / "model_outputs" / "final" / "v1.10"
    ordinary_raw.mkdir(parents=True)
    (ordinary_raw / f"{ordinary_id}.json").write_text("{}", encoding="utf-8")

    rows = {row["school_id"]: row for row in workflow_rows(tmp_path)}
    assert rows[validation_id]["stage"] == "WAITING"

    measurement_rows[1]["review_status"] = "reviewed"
    _write_csv(tmp_path / "measurements.csv", MEASUREMENT_COLUMNS, measurement_rows)
    rows = {row["school_id"]: row for row in workflow_rows(tmp_path)}
    assert rows[validation_id]["stage"] == "BLIND LABEL"
