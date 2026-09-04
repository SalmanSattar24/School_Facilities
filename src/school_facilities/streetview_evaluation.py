from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schema import read_csv
from .streetview import StreetViewConfigurationError, _read_object


STREET_FIELDS = (
    "portable_classroom_count",
    "perimeter_fencing",
    "dominant_fence_type",
    "running_track",
    "full_size_sports_fields",
    "hard_courts",
    "pool",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalize(value: Any) -> str | int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    return text


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row["reference"] != "unknown"]
    answered = [row for row in known if row["prediction"] != "unknown"]
    wrong = [row for row in answered if not row["correct"]]
    abstained = [row for row in known if row["prediction"] == "unknown"]
    flagged = [row for row in known if row["flagged"]]
    problematic = [row for row in known if row["prediction"] == "unknown" or not row["correct"]]
    captured = [row for row in problematic if row["flagged"]]
    flagged_problematic = [row for row in flagged if row in problematic]
    brier_rows = [row for row in answered if row["confidence"] is not None]
    return {
        "known_reference_n": len(known),
        "answered_n": len(answered),
        "correct_n": sum(bool(row["correct"]) for row in answered),
        "wrong_n": len(wrong),
        "unknown_n": len(abstained),
        "coverage": len(answered) / len(known) if known else None,
        "selective_accuracy": (
            sum(bool(row["correct"]) for row in answered) / len(answered)
            if answered else None
        ),
        "unknown_rate": len(abstained) / len(known) if known else None,
        "silent_error_n": sum(not row["flagged"] for row in wrong),
        "silent_error_rate_among_answered": (
            sum(not row["flagged"] for row in wrong) / len(answered)
            if answered else None
        ),
        "error_or_abstention_flag_recall": len(captured) / len(problematic) if problematic else None,
        "review_precision_for_error_or_abstention": (
            len(flagged_problematic) / len(flagged) if flagged else None
        ),
        "brier_score_answered": (
            sum((row["confidence"] - (1.0 if row["correct"] else 0.0)) ** 2 for row in brier_rows)
            / len(brier_rows)
            if brier_rows else None
        ),
    }


def _calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["reference"] == "unknown" or row["prediction"] == "unknown" or row["confidence"] is None:
            continue
        bins[float(row["confidence"])].append(row)
    return [
        {
            "confidence": confidence,
            "n": len(items),
            "correct_n": sum(bool(item["correct"]) for item in items),
            "observed_accuracy": sum(bool(item["correct"]) for item in items) / len(items),
            "calibration_gap": (
                sum(bool(item["correct"]) for item in items) / len(items) - confidence
            ),
        }
        for confidence, items in sorted(bins.items(), reverse=True)
    ]


def _load_prediction(path: Path) -> dict[str, Any]:
    record = _read_object(path)
    output = record.get("guarded_output")
    if not isinstance(output, dict) or not isinstance(output.get("candidate_fields"), dict):
        raise StreetViewConfigurationError(f"missing guarded V1.11 output: {path}")
    return output


def evaluate_v1_11(
    root: Path,
    *,
    reference_path: Path,
    reference_sha256: str,
    prediction_directory: Path,
) -> dict[str, Any]:
    root = root.resolve()
    actual_hash = _sha256(reference_path)
    if actual_hash != reference_sha256.strip().upper():
        raise StreetViewConfigurationError(
            f"reference SHA-256 mismatch: expected {reference_sha256.strip().upper()}, got {actual_hash}"
        )
    pilot = _read_object(root / "config" / "pilot_schools.json")
    validation_ids = set(pilot.get("excluded_validation_school_ids", []))
    references = {
        row["school_id"]: row
        for row in read_csv(reference_path)
        if row["school_id"] in validation_ids
    }
    if set(references) != validation_ids:
        raise StreetViewConfigurationError("reference file does not contain the exact validation-school set")
    observations: list[dict[str, Any]] = []
    for school_id in sorted(validation_ids):
        output = _load_prediction(prediction_directory / f"{school_id}.json")
        fields = output["candidate_fields"]
        for field in STREET_FIELDS:
            candidate = fields[field]
            prediction = _normalize(candidate["value"])
            reference = _normalize(references[school_id][field])
            observations.append(
                {
                    "school_id": school_id,
                    "field": field,
                    "prediction": prediction,
                    "reference": reference,
                    "correct": prediction != "unknown" and prediction == reference,
                    "abstained_unknown": prediction == "unknown",
                    "confidence": float(candidate["suggested_confidence"]),
                    "flagged": bool(candidate["review_required"]) or prediction == "unknown",
                }
            )
    return {
        "schema_version": "1.0",
        "evaluation": "frozen_v1_11_street_supplement",
        "reference_sha256": actual_hash,
        "validation_school_n": len(validation_ids),
        "field_n_per_school": len(STREET_FIELDS),
        "overall": _summary(observations),
        "by_field": {
            field: _summary([row for row in observations if row["field"] == field])
            for field in STREET_FIELDS
        },
        "calibration_bins_answered": _calibration(observations),
        "observations": observations,
        "interpretation_note": (
            "Six schools provide descriptive error and calibration estimates only; "
            "confidence mappings were not fitted on this reference set."
        ),
    }


def evaluate_stability(
    *,
    school_ids: Iterable[str],
    primary_directory: Path,
    repeat_directories: Iterable[Path],
) -> dict[str, Any]:
    comparisons = []
    for repeat_index, repeat_directory in enumerate(repeat_directories, start=1):
        for school_id in school_ids:
            primary = _load_prediction(primary_directory / f"{school_id}.json")
            repeat = _load_prediction(repeat_directory / f"{school_id}.json")
            for field in STREET_FIELDS:
                first = primary["candidate_fields"][field]
                second = repeat["candidate_fields"][field]
                comparisons.append(
                    {
                        "repeat": repeat_index,
                        "school_id": school_id,
                        "field": field,
                        "value_agreement": _normalize(first["value"]) == _normalize(second["value"]),
                        "confidence_absolute_drift": abs(
                            float(first["suggested_confidence"])
                            - float(second["suggested_confidence"])
                        ),
                        "flag_agreement": bool(first["review_required"]) == bool(second["review_required"]),
                    }
                )
    return {
        "schema_version": "1.0",
        "comparison_n": len(comparisons),
        "exact_value_agreement": (
            sum(item["value_agreement"] for item in comparisons) / len(comparisons)
            if comparisons else None
        ),
        "mean_confidence_absolute_drift": (
            sum(item["confidence_absolute_drift"] for item in comparisons) / len(comparisons)
            if comparisons else None
        ),
        "flag_agreement": (
            sum(item["flag_agreement"] for item in comparisons) / len(comparisons)
            if comparisons else None
        ),
        "comparisons": comparisons,
    }
