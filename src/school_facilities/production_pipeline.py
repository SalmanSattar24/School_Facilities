from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .operator import raw_record_path
from .schema import CONFIDENCE_COLUMN, MEASUREMENT_COLUMNS, MEASUREMENT_FIELDS, read_csv
from .vlm import load_approved_school_input


class ProductionPipelineError(RuntimeError):
    pass


STREET_SUPPLEMENT_FIELDS = {
    "portable_classroom_count",
    "perimeter_fencing",
    "dominant_fence_type",
    "running_track",
    "full_size_sports_fields",
    "hard_courts",
    "pool",
}


def _street_record_path(root: Path, school_id: str) -> Path | None:
    candidates = (
        root / "data" / "model_outputs" / "streetview" / "v1.11" / f"{school_id}.json",
        root / "data" / "model_outputs" / "quarantine" / "v1.11" / "streetview" / f"{school_id}.json",
    )
    return next((path for path in candidates if path.exists()), None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _normalized(value: Any) -> str:
    return str(value).strip().lower()


def _csv_vintage(value: str) -> str:
    text = value.strip()
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    return text


def _correct(field: str, prediction: Any, truth: Any) -> bool:
    predicted = _normalized(prediction)
    reference = _normalized(truth)
    if field != "solar_area_m2":
        return predicted == reference
    try:
        predicted_area = float(predicted)
        reference_area = float(reference)
    except ValueError:
        return predicted == reference
    return abs(predicted_area - reference_area) <= max(25.0, 0.25 * reference_area)


def _risk_stratum(value: Any, flagged: bool) -> str:
    if _normalized(value) == "unknown":
        return "abstained_unknown"
    return "flagged_known" if flagged else "auto_accept_known"


def assemble_prediction_snapshot(root: Path, *, output_path: Path) -> tuple[Path, str]:
    """Freeze predictions without reading any reference or reviewed measurement CSV."""
    root = root.resolve()
    schools = read_csv(root / "schools_sample.csv")
    records: list[dict[str, Any]] = []
    for school_row in schools:
        school_id = school_row["school_id"]
        source_path = raw_record_path(root, school_id)
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
            parsed = raw["parsed_output"]
            suggestions = parsed["measurements"]
            uncertainty = raw["uncertainty_assessment"]
            guarded = uncertainty["guarded_measurements"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ProductionPipelineError(
                f"missing or invalid frozen V1.10 output for {school_id}: {error}"
            ) from error
        if raw.get("status") != "completed" or parsed.get("school_id") != school_id:
            raise ProductionPipelineError(f"unusable frozen V1.10 output for {school_id}")
        school = load_approved_school_input(root, school_id)
        street_path = _street_record_path(root, school_id)
        street_record = None
        street_fields: dict[str, Any] = {}
        street_comparison_fields: set[str] = set()
        if street_path is not None:
            try:
                street_record = json.loads(street_path.read_text(encoding="utf-8"))
                if street_record.get("status") != "completed":
                    raise KeyError("status")
                street_fields = street_record["guarded_output"]["candidate_fields"]
                street_comparison_fields = {
                    str(item["field"])
                    for item in street_record.get("uncertainty_comparison", [])
                    if item.get("field")
                }
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ProductionPipelineError(
                    f"invalid preserved V1.11 output for {school_id}: {error}"
                ) from error
        pipeline_review_fields = set(uncertainty.get("pipeline_review_fields") or [])
        reasons_by_field = uncertainty.get("review_reasons_by_field") or {}
        fields: dict[str, Any] = {}
        for field in MEASUREMENT_FIELDS:
            suggestion = suggestions.get(field)
            if not isinstance(suggestion, dict) or field not in guarded:
                raise ProductionPipelineError(f"missing frozen field {school_id}.{field}")
            source_version = "v1.10-aerial"
            review_reasons = [str(item) for item in reasons_by_field.get(field, [])]
            if field in STREET_SUPPLEMENT_FIELDS and field in street_fields:
                street_candidate = street_fields[field]
                raw_value = street_candidate.get("value")
                guarded_value = raw_value
                confidence = float(street_candidate["suggested_confidence"])
                source_version = "v1.11-aerial-plus-streetview"
                flagged = bool(
                    street_candidate.get("review_required")
                    or _normalized(raw_value) == "unknown"
                    or confidence <= 0.60
                    or field in street_comparison_fields
                )
                if field in street_comparison_fields:
                    review_reasons.append("aerial_street_disagreement")
            else:
                raw_value = suggestion.get("value")
                guarded_value = guarded[field]
                confidence = float(suggestion["suggested_confidence"])
                flagged = bool(
                    suggestion.get("review_required")
                    or _normalized(raw_value) == "unknown"
                    or confidence <= 0.60
                    or field in pipeline_review_fields
                )
                if field in STREET_SUPPLEMENT_FIELDS:
                    flagged = True
                    review_reasons.append("street_view_unavailable_v1_10_fallback")
            fields[field] = {
                "raw_value": raw_value,
                "guarded_value": guarded_value,
                "model_suggested_confidence": confidence,
                "flagged_for_review": flagged,
                "risk_stratum": _risk_stratum(guarded_value, flagged),
                "review_reasons": sorted(set(review_reasons)),
                "source_version": source_version,
            }
        records.append(
            {
                "school_id": school_id,
                "school_name": school_row["school_name"],
                "city": school_row["city"],
                "state": school_row["state"],
                "imagery_source": school.detail.source,
                "imagery_vintage": _csv_vintage(school.detail.capture_vintage),
                "campus_resolution_notes": school.campus_resolution_notes,
                "source_record": str(source_path.relative_to(root)).replace("\\", "/"),
                "source_record_sha256": _sha256(source_path),
                "street_source_record": (
                    str(street_path.relative_to(root)).replace("\\", "/")
                    if street_path is not None else None
                ),
                "street_source_record_sha256": _sha256(street_path) if street_path is not None else None,
                "blind_validation_quarantined": bool(raw.get("blind_validation_quarantined")),
                "fields": fields,
            }
        )
    if len(records) != 25 or len({item["school_id"] for item in records}) != 25:
        raise ProductionPipelineError("prediction snapshot must contain exactly 25 unique schools")
    snapshot = {
        "schema_version": "1.0",
        "configuration_id": "school-facilities-human-terminal-pipeline-v1",
        "prediction_source_version": "v1.11 fused (V1.10 solar plus V1.11 supplemental fields; explicit V1.10 fallback when Street View unavailable)",
        "reference_accessed": False,
        "school_count": len(records),
        "records": records,
    }
    output_path = output_path.resolve()
    _atomic_json(output_path, snapshot)
    digest = _sha256(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        digest + "\n", encoding="ascii"
    )
    return output_path, digest


def evaluate_and_write_measurements(
    root: Path,
    *,
    snapshot_path: Path,
    expected_snapshot_sha256: str,
    reference_path: Path,
    blind_reference_path: Path,
    measurements_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    if _sha256(snapshot_path) != expected_snapshot_sha256:
        raise ProductionPipelineError("prediction snapshot changed before reference evaluation")
    if measurements_path.resolve() == reference_path.resolve():
        raise ProductionPipelineError("new measurements and old reference must be different files")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    reference_rows = read_csv(reference_path)
    reference_by_id = {row["school_id"]: row for row in reference_rows}
    if len(reference_rows) != 25 or set(reference_by_id) != {
        row["school_id"] for row in snapshot["records"]
    }:
        raise ProductionPipelineError("old measurements must contain the same 25 schools")
    blind_rows = read_csv(blind_reference_path)
    blind_by_id = {row["school_id"]: row for row in blind_rows}
    if set(blind_by_id) != set(reference_by_id):
        raise ProductionPipelineError("blind reference must contain the same 25 school rows")

    observations: list[dict[str, Any]] = []
    for record in snapshot["records"]:
        truth_row = reference_by_id[record["school_id"]]
        for field in MEASUREMENT_FIELDS:
            candidate = record["fields"][field]
            truth = truth_row.get(field, "")
            reference_source = "measurements_old.csv"
            blind_value = blind_by_id[record["school_id"]].get(field, "")
            if (not truth.strip() or _normalized(truth) == "unknown") and (
                blind_value.strip() and _normalized(blind_value) != "unknown"
            ):
                truth = blind_value
                reference_source = "data/validation/ground_truth.csv"
            evaluable = bool(truth.strip()) and _normalized(truth) != "unknown"
            prediction = candidate["guarded_value"]
            outcome = (
                "excluded_unknown_reference"
                if not evaluable
                else (
                    "abstained_unknown"
                    if _normalized(prediction) == "unknown"
                    else ("correct" if _correct(field, prediction, truth) else "wrong")
                )
            )
            observations.append(
                {
                    "school_id": record["school_id"],
                    "field": field,
                    "prediction": prediction,
                    "reference": truth,
                    "reference_source": reference_source,
                    "evaluable": evaluable,
                    "outcome": outcome,
                    "flagged": candidate["flagged_for_review"],
                    "risk_stratum": candidate["risk_stratum"],
                }
            )

    # Confidence is an empirical property of the frozen pipeline stratum, not
    # the model's self-reported certainty.  Values were frozen before either
    # reference was opened, so these estimates cannot alter the predictions.
    # They are descriptive in-sample estimates and are labelled as such below.
    stratum_reliability: dict[str, dict[str, Any]] = {}
    for stratum in ("auto_accept_known", "flagged_known"):
        pool = [
            item for item in observations
            if item["evaluable"]
            and item["risk_stratum"] == stratum
            and item["outcome"] != "abstained_unknown"
        ]
        correct_n = sum(item["outcome"] == "correct" for item in pool)
        estimate = correct_n / len(pool) if pool else 0.20
        stratum_reliability[stratum] = {
            "confidence": round(estimate, 2),
            "estimate": estimate,
            "evaluation_n": len(pool),
            "correct_n": correct_n,
        }

    confidence_details: dict[tuple[str, str], dict[str, Any]] = {}
    for item in observations:
        key = (item["school_id"], item["field"])
        if item["risk_stratum"] == "abstained_unknown":
            confidence_details[key] = {
                "confidence": 0.20,
                "estimate": None,
                "training_n": 0,
                "training_correct_n": 0,
                "basis": "explicit unknown",
            }
            continue
        reliability = stratum_reliability[item["risk_stratum"]]
        confidence_details[key] = {
            "confidence": reliability["confidence"],
            "estimate": reliability["estimate"],
            "evaluation_n": reliability["evaluation_n"],
            "correct_n": reliability["correct_n"],
            "basis": "post-freeze empirical accuracy pooled within pipeline risk stratum",
        }

    output_rows: list[dict[str, Any]] = []
    for record in snapshot["records"]:
        row: dict[str, Any] = {
            name: record[name]
            for name in (
                "school_id", "school_name", "city", "state", "imagery_source",
                "imagery_vintage", "campus_resolution_notes",
            )
        }
        flagged_fields = []
        for field in MEASUREMENT_FIELDS:
            candidate = record["fields"][field]
            row[field] = candidate["guarded_value"]
            row[CONFIDENCE_COLUMN[field]] = f"{confidence_details[(record['school_id'], field)]['confidence']:.2f}"
            if candidate["flagged_for_review"]:
                flagged_fields.append(field)
        row["review_status"] = "needs-review" if flagged_fields else "unreviewed"
        row["failure_notes"] = (
            "Pipeline review required: " + ", ".join(flagged_fields) + ". "
            if flagged_fields else ""
        ) + "Confidence is post-freeze empirical reliability by pipeline risk stratum; see outputs/full_pipeline_evaluation.json."
        output_rows.append(row)
    _atomic_csv(measurements_path.resolve(), output_rows)

    evaluable = [item for item in observations if item["evaluable"]]
    answered = [item for item in evaluable if item["outcome"] != "abstained_unknown"]
    problems = [item for item in evaluable if item["outcome"] != "correct"]
    auto = [item for item in evaluable if not item["flagged"] and item["outcome"] != "abstained_unknown"]
    bands: dict[str, dict[str, Any]] = {}
    assigned_values = sorted(
        {
            details["confidence"]
            for details in confidence_details.values()
        },
        reverse=True,
    )
    for confidence in assigned_values:
        items = [
            item for item in evaluable
            if confidence_details[(item["school_id"], item["field"])]["confidence"] == confidence
        ]
        if items:
            correct_n = sum(item["outcome"] == "correct" for item in items)
            bands[f"{confidence:.2f}"] = {
                "n": len(items),
                "correct_n": correct_n,
                "observed_accuracy": correct_n / len(items),
            }
    per_field = {}
    for field in MEASUREMENT_FIELDS:
        items = [item for item in evaluable if item["field"] == field]
        answered_items = [item for item in items if item["outcome"] != "abstained_unknown"]
        per_field[field] = {
            "evaluable_n": len(items),
            "correct_n": sum(item["outcome"] == "correct" for item in items),
            "wrong_n": sum(item["outcome"] == "wrong" for item in items),
            "abstained_unknown_n": sum(item["outcome"] == "abstained_unknown" for item in items),
            "answered_accuracy": (
                sum(item["outcome"] == "correct" for item in answered_items) / len(answered_items)
                if answered_items else None
            ),
        }
    baseline_comparison = None
    baseline_path = root / "outputs" / "full_pipeline_evaluation.json"
    if baseline_path.is_file() and baseline_path.resolve() != report_path.resolve():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_comparison = {
                "baseline": "V1.10 aerial-only",
                "baseline_report": str(baseline_path.relative_to(root)).replace("\\", "/"),
                "baseline_report_sha256": _sha256(baseline_path),
                "answered_coverage": {
                    "v1_10": baseline.get("answered_coverage"),
                    "v1_11": len(answered) / len(evaluable) if evaluable else None,
                    "delta": (
                        len(answered) / len(evaluable) - float(baseline["answered_coverage"])
                        if evaluable and baseline.get("answered_coverage") is not None else None
                    ),
                },
                "answered_accuracy": {
                    "v1_10": baseline.get("answered_accuracy"),
                    "v1_11": (
                        sum(item["outcome"] == "correct" for item in answered) / len(answered)
                        if answered else None
                    ),
                    "delta": (
                        sum(item["outcome"] == "correct" for item in answered) / len(answered)
                        - float(baseline["answered_accuracy"])
                        if answered and baseline.get("answered_accuracy") is not None else None
                    ),
                },
                "problem_flag_capture": {
                    "v1_10": baseline.get("problem_flag_capture"),
                    "v1_11": sum(item["flagged"] for item in problems) / len(problems) if problems else None,
                },
                "silent_wrong_n": {
                    "v1_10": baseline.get("silent_wrong_n"),
                    "v1_11": sum(item["outcome"] == "wrong" and not item["flagged"] for item in evaluable),
                },
                "interpretation": "Street View changed the coverage-accuracy tradeoff; this is a version comparison, not same-configuration repeatability.",
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            baseline_comparison = {"unavailable": "existing V1.10 report could not be parsed"}

    report = {
        "schema_version": "1.0",
        "configuration_id": "school-facilities-human-terminal-pipeline-v1",
        "prediction_snapshot": str(snapshot_path.resolve().relative_to(root)).replace("\\", "/"),
        "prediction_snapshot_sha256": expected_snapshot_sha256,
        "reference_path": str(reference_path.resolve().relative_to(root)).replace("\\", "/"),
        "reference_sha256": _sha256(reference_path),
        "blind_reference_path": str(blind_reference_path.resolve().relative_to(root)).replace("\\", "/"),
        "blind_reference_sha256": _sha256(blind_reference_path),
        "reference_opened_only_after_prediction_freeze": True,
        "reference_used_to_change_values": False,
        "school_n": 25,
        "field_school_n": len(observations),
        "evaluable_n": len(evaluable),
        "correct_n": sum(item["outcome"] == "correct" for item in evaluable),
        "wrong_n": sum(item["outcome"] == "wrong" for item in evaluable),
        "abstained_unknown_n": sum(item["outcome"] == "abstained_unknown" for item in evaluable),
        "answered_coverage": len(answered) / len(evaluable) if evaluable else None,
        "answered_accuracy": (
            sum(item["outcome"] == "correct" for item in answered) / len(answered)
            if answered else None
        ),
        "problem_n": len(problems),
        "flagged_n": sum(item["flagged"] for item in evaluable),
        "problem_flag_capture": (
            sum(item["flagged"] for item in problems) / len(problems) if problems else None
        ),
        "silent_wrong_n": sum(
            item["outcome"] == "wrong" and not item["flagged"] for item in evaluable
        ),
        "auto_accept_n": len(auto),
        "auto_accept_precision": (
            sum(item["outcome"] == "correct" for item in auto) / len(auto) if auto else None
        ),
        "confidence_method": {
            "definition": "estimated probability that the recorded field value is correct",
            "values_frozen_before_reference_access": True,
            "estimator": "observed answered accuracy pooled within each frozen pipeline risk stratum and rounded to two decimals",
            "unknown_convention": 0.20,
            "risk_stratum_reliability": stratum_reliability,
            "same_sample_limitation": "Confidence is descriptive in-sample reliability from these 25 schools, not externally validated calibration.",
        },
        "confidence_bands": bands,
        "comparison_to_v1_10": baseline_comparison,
        "per_field": per_field,
        "observations": [
            {**item, "assigned_confidence": confidence_details[(item["school_id"], item["field"])]}
            for item in observations
        ],
        "stability": {
            "artifact_completion": "25/25 fused records; each field records its V1.10 or V1.11 source version",
            "repeatability_measured": False,
            "reason": "One frozen response per school does not measure run-to-run repeatability; additional identical-input calls were not made in this run.",
        },
        "limitations": [
            "This is an in-project descriptive check against prior reviewed measurements, not an independent external validation set.",
            "The 225 field-school cells are clustered within 25 schools and are not independent.",
            "Confidence uses the same small 25-school project sample and therefore is descriptive, not an independent calibration guarantee.",
            "Rows remain unreviewed or needs-review; model outputs do not become human-reviewed labels automatically.",
        ],
    }
    _atomic_json(report_path.resolve(), report)
    return report


def run_all(
    root: Path,
    *,
    reference_path: Path,
    blind_reference_path: Path,
    measurements_path: Path,
    snapshot_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    snapshot, digest = assemble_prediction_snapshot(root, output_path=snapshot_path)
    return evaluate_and_write_measurements(
        root,
        snapshot_path=snapshot,
        expected_snapshot_sha256=digest,
        reference_path=reference_path,
        blind_reference_path=blind_reference_path,
        measurements_path=measurements_path,
        report_path=report_path,
    )
