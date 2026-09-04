from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import MEASUREMENT_FIELDS


COUNT_FIELDS = {"portable_classroom_count", "full_size_sports_fields", "hard_courts"}
CONFIDENCE_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)


@dataclass(frozen=True)
class RawVLMObservation:
    school_id: str
    field: str
    prediction: str
    truth: str
    suggested_confidence: float
    review_required: bool
    outcome: str
    flagged: bool
    auto_accept_candidate: bool
    guarded_prediction: str
    guarded_outcome: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: object) -> str:
    return str(value).strip().lower()


def _correct(field: str, prediction: str, truth: str) -> bool:
    if field != "solar_area_m2":
        return prediction == truth
    try:
        predicted_area = float(prediction)
        true_area = float(truth)
    except ValueError:
        return prediction == truth
    return abs(predicted_area - true_area) <= max(25.0, 0.25 * true_area)


def raw_observations(
    raw_directory: Path,
    ground_truth: list[dict[str, str]],
    validation_school_ids: set[str],
) -> tuple[list[RawVLMObservation], dict[str, int]]:
    truth_by_id = {row["school_id"]: row for row in ground_truth}
    observations: list[RawVLMObservation] = []
    exclusions = {
        "blank_reference": 0,
        "unknown_reference": 0,
        "solar_nonpositive_reference": 0,
    }

    for school_id in sorted(validation_school_ids):
        truth_row = truth_by_id.get(school_id)
        if truth_row is None:
            raise ValueError(f"validation school is missing from reference CSV: {school_id}")
        raw_path = raw_directory / f"{school_id}.json"
        try:
            raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
            parsed = raw_record["parsed_output"]
            measurements = parsed["measurements"]
            derived = raw_record.get(
                "uncertainty_assessment",
                raw_record.get("derived_solar_summary", {}),
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid or missing raw VLM record {raw_path}: {error}") from error
        if parsed.get("school_id") != school_id:
            raise ValueError(f"raw VLM school_id does not match filename for {school_id}")

        for field in MEASUREMENT_FIELDS:
            truth = _normalized(truth_row.get(field, ""))
            if not truth:
                exclusions["blank_reference"] += 1
                continue
            if truth == "unknown":
                exclusions["unknown_reference"] += 1
                continue
            if field == "solar_area_m2":
                try:
                    if float(truth) <= 0:
                        exclusions["solar_nonpositive_reference"] += 1
                        continue
                except ValueError as error:
                    raise ValueError(f"invalid numeric solar reference for {school_id}: {truth}") from error

            try:
                suggestion = measurements[field]
                prediction = _normalized(suggestion["value"])
                confidence = float(suggestion["suggested_confidence"])
                review_required = suggestion["review_required"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid raw suggestion for {school_id}.{field}: {error}") from error
            if prediction == "unknown":
                outcome = "abstained_unknown"
            else:
                outcome = "correct" if _correct(field, prediction, truth) else "wrong"
            pipeline_review_fields = set(derived.get("pipeline_review_fields", []))
            guarded_values = derived.get("guarded_measurements", {})
            guarded_prediction = _normalized(guarded_values.get(field, prediction))
            if guarded_prediction == "unknown":
                guarded_outcome = "abstained_unknown"
            else:
                guarded_outcome = (
                    "correct" if _correct(field, guarded_prediction, truth) else "wrong"
                )
            flagged = (
                bool(review_required)
                or prediction == "unknown"
                or confidence <= 0.6
                or field in pipeline_review_fields
            )
            auto_accept = (
                prediction != "unknown"
                and confidence >= 0.8
                and review_required is False
                and field not in pipeline_review_fields
            )
            observations.append(
                RawVLMObservation(
                    school_id=school_id,
                    field=field,
                    prediction=prediction,
                    truth=truth,
                    suggested_confidence=confidence,
                    review_required=bool(review_required),
                    outcome=outcome,
                    flagged=flagged,
                    auto_accept_candidate=auto_accept,
                    guarded_prediction=guarded_prediction,
                    guarded_outcome=guarded_outcome,
                )
            )
    return observations, exclusions


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summary(items: list[RawVLMObservation]) -> dict[str, object]:
    n = len(items)
    correct_n = sum(item.outcome == "correct" for item in items)
    wrong_n = sum(item.outcome == "wrong" for item in items)
    abstained_n = sum(item.outcome == "abstained_unknown" for item in items)
    answered_n = correct_n + wrong_n
    flagged_n = sum(item.flagged for item in items)
    problem_n = wrong_n + abstained_n
    captured_n = sum(item.flagged and item.outcome != "correct" for item in items)
    silent_error_n = sum(not item.flagged and item.outcome == "wrong" for item in items)
    auto_items = [item for item in items if item.auto_accept_candidate]
    auto_correct_n = sum(item.outcome == "correct" for item in auto_items)
    flagged_problem_n = sum(item.flagged and item.outcome != "correct" for item in items)
    guarded_correct_n = sum(item.guarded_outcome == "correct" for item in items)
    guarded_wrong_n = sum(item.guarded_outcome == "wrong" for item in items)
    guarded_abstained_n = sum(
        item.guarded_outcome == "abstained_unknown" for item in items
    )
    guarded_answered_n = guarded_correct_n + guarded_wrong_n
    guarded_silent_error_n = sum(
        not item.flagged and item.guarded_outcome == "wrong" for item in items
    )

    report: dict[str, object] = {
        "evaluable_n": n,
        "school_n": len({item.school_id for item in items}),
        "correct_n": correct_n,
        "wrong_n": wrong_n,
        "abstained_unknown_n": abstained_n,
        "overall_success_rate": _ratio(correct_n, n),
        "wrong_rate": _ratio(wrong_n, n),
        "abstention_rate": _ratio(abstained_n, n),
        "answered_n": answered_n,
        "answered_coverage": _ratio(answered_n, n),
        "selective_accuracy": _ratio(correct_n, answered_n),
        "flagged_n": flagged_n,
        "flag_rate": _ratio(flagged_n, n),
        "problem_n": problem_n,
        "problem_captured_n": captured_n,
        "problem_capture_rate": _ratio(captured_n, problem_n),
        "silent_error_n": silent_error_n,
        "silent_error_rate": _ratio(silent_error_n, n),
        "auto_accept_candidate_n": len(auto_items),
        "auto_accept_correct_n": auto_correct_n,
        "auto_accept_precision": _ratio(auto_correct_n, len(auto_items)),
        "flag_problem_n": flagged_problem_n,
        "flag_precision": _ratio(flagged_problem_n, flagged_n),
        "guarded_pipeline": {
            "correct_n": guarded_correct_n,
            "wrong_n": guarded_wrong_n,
            "abstained_unknown_n": guarded_abstained_n,
            "answered_n": guarded_answered_n,
            "answered_coverage": _ratio(guarded_answered_n, n),
            "selective_accuracy": _ratio(guarded_correct_n, guarded_answered_n),
            "silent_error_n": guarded_silent_error_n,
            "silent_error_rate": _ratio(guarded_silent_error_n, n),
        },
        "mean_suggested_confidence": (
            sum(item.suggested_confidence for item in items) / n if n else None
        ),
        "brier_score": (
            sum(
                (item.suggested_confidence - (1 if item.outcome == "correct" else 0)) ** 2
                for item in items
            )
            / n
            if n
            else None
        ),
    }

    if items and items[0].field in COUNT_FIELDS:
        numeric_pairs: list[tuple[int, int]] = []
        for item in items:
            if item.outcome == "abstained_unknown":
                continue
            try:
                numeric_pairs.append((int(item.prediction), int(item.truth)))
            except ValueError:
                continue
        report["numeric_answered_n"] = len(numeric_pairs)
        report["mean_absolute_error"] = (
            sum(abs(prediction - truth) for prediction, truth in numeric_pairs) / len(numeric_pairs)
            if numeric_pairs
            else None
        )
    return report


def summarize_raw_vlm(
    items: list[RawVLMObservation], exclusions: dict[str, int]
) -> dict[str, object]:
    report = _summary(items)
    report["evaluation_target"] = "raw_vlm_before_human_adjudication"
    report["outcomes_are_mutually_exclusive"] = ["correct", "wrong", "abstained_unknown"]
    report["flag_rule"] = "review_required OR unknown OR suggested_confidence <= 0.60"
    report["exclusions"] = exclusions
    report["by_field"] = {
        field: _summary([item for item in items if item.field == field])
        for field in MEASUREMENT_FIELDS
        if any(item.field == field for item in items)
    }
    report["confidence_scores"] = {
        str(score): _summary([item for item in items if item.suggested_confidence == score])
        for score in CONFIDENCE_THRESHOLDS
        if any(item.suggested_confidence == score for item in items)
    }
    risk_coverage: list[dict[str, object]] = []
    for threshold in CONFIDENCE_THRESHOLDS:
        selected = [
            item
            for item in items
            if item.prediction != "unknown" and item.suggested_confidence >= threshold
        ]
        wrong_n = sum(item.outcome == "wrong" for item in selected)
        risk_coverage.append(
            {
                "minimum_confidence": threshold,
                "answered_n": len(selected),
                "coverage_of_evaluable": _ratio(len(selected), len(items)),
                "wrong_n": wrong_n,
                "error_rate_among_selected": _ratio(wrong_n, len(selected)),
            }
        )
    report["risk_coverage"] = risk_coverage
    report["limitations"] = [
        "Six purposively selected schools are too few for strong calibration claims.",
        "Fields within a school are correlated and are not independent schools.",
        "The reference labels are blind same-reviewer labels, not independent expert annotations.",
    ]
    return report


def pilot_uncertainty_diagnostic(
    raw_directory: Path,
    reviewed_rows: list[dict[str, str]],
    pilot_school_ids: set[str],
) -> dict[str, object]:
    """Compare V1.7 uncertainty routing with prior reviewed pilot rows."""
    reviewed_by_id = {row["school_id"]: row for row in reviewed_rows}
    observations: list[dict[str, object]] = []
    for school_id in sorted(pilot_school_ids):
        reviewed = reviewed_by_id.get(school_id)
        if reviewed is None:
            raise ValueError(f"pilot school is missing from reviewed measurements: {school_id}")
        raw_path = raw_directory / f"{school_id}.json"
        try:
            raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
            measurements = raw_record["parsed_output"]["measurements"]
            uncertainty = raw_record["uncertainty_assessment"]
            pipeline_review_fields = set(uncertainty["pipeline_review_fields"])
            guarded = uncertainty["guarded_measurements"]
            auto_accept_fields = set(uncertainty["auto_accept_candidate_fields"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid pilot VLM record {raw_path}: {error}") from error
        for field in MEASUREMENT_FIELDS:
            suggestion = measurements[field]
            prediction = _normalized(suggestion["value"])
            truth = _normalized(reviewed[field])
            if prediction == "unknown":
                outcome = "abstained_unknown"
            else:
                outcome = "correct" if _correct(field, prediction, truth) else "wrong"
            model_flagged = (
                bool(suggestion["review_required"])
                or prediction == "unknown"
                or float(suggestion["suggested_confidence"]) <= 0.6
            )
            guarded_prediction = _normalized(guarded.get(field, prediction))
            if guarded_prediction == "unknown":
                guarded_outcome = "abstained_unknown"
            else:
                guarded_outcome = (
                    "correct" if _correct(field, guarded_prediction, truth) else "wrong"
                )
            observations.append(
                {
                    "school_id": school_id,
                    "field": field,
                    "outcome": outcome,
                    "model_flagged": model_flagged,
                    "pipeline_flagged": field in pipeline_review_fields,
                    "auto_accept_candidate": field in auto_accept_fields,
                    "guarded_outcome": guarded_outcome,
                }
            )

    def summary(rows: list[dict[str, object]]) -> dict[str, object]:
        n = len(rows)
        correct_n = sum(row["outcome"] == "correct" for row in rows)
        wrong_n = sum(row["outcome"] == "wrong" for row in rows)
        abstained_n = sum(row["outcome"] == "abstained_unknown" for row in rows)
        problems = [row for row in rows if row["outcome"] != "correct"]
        model_flagged = [row for row in rows if row["model_flagged"]]
        pipeline_flagged = [row for row in rows if row["pipeline_flagged"]]
        auto_accept = [row for row in rows if row["auto_accept_candidate"]]
        guarded_correct_n = sum(row["guarded_outcome"] == "correct" for row in rows)
        guarded_wrong_n = sum(row["guarded_outcome"] == "wrong" for row in rows)
        guarded_abstained_n = sum(
            row["guarded_outcome"] == "abstained_unknown" for row in rows
        )
        guarded_answered_n = guarded_correct_n + guarded_wrong_n
        return {
            "field_school_n": n,
            "raw_correct_n": correct_n,
            "raw_wrong_n": wrong_n,
            "raw_abstained_unknown_n": abstained_n,
            "raw_overall_success_rate": _ratio(correct_n, n),
            "raw_answered_coverage": _ratio(correct_n + wrong_n, n),
            "raw_selective_accuracy": _ratio(correct_n, correct_n + wrong_n),
            "problem_n": len(problems),
            "model_only_flag_n": len(model_flagged),
            "model_only_problem_capture_n": sum(
                row["model_flagged"] for row in problems
            ),
            "model_only_problem_capture_rate": _ratio(
                sum(row["model_flagged"] for row in problems), len(problems)
            ),
            "model_only_silent_wrong_n": sum(
                row["outcome"] == "wrong" and not row["model_flagged"] for row in rows
            ),
            "pipeline_flag_n": len(pipeline_flagged),
            "pipeline_review_rate": _ratio(len(pipeline_flagged), n),
            "pipeline_problem_capture_n": sum(
                row["pipeline_flagged"] for row in problems
            ),
            "pipeline_problem_capture_rate": _ratio(
                sum(row["pipeline_flagged"] for row in problems), len(problems)
            ),
            "pipeline_silent_wrong_n": sum(
                row["outcome"] == "wrong" and not row["pipeline_flagged"] for row in rows
            ),
            "pipeline_flag_precision": _ratio(
                sum(row["outcome"] != "correct" for row in pipeline_flagged),
                len(pipeline_flagged),
            ),
            "auto_accept_candidate_n": len(auto_accept),
            "auto_accept_coverage": _ratio(len(auto_accept), n),
            "auto_accept_correct_n": sum(
                row["outcome"] == "correct" for row in auto_accept
            ),
            "auto_accept_precision": _ratio(
                sum(row["outcome"] == "correct" for row in auto_accept),
                len(auto_accept),
            ),
            "guarded_correct_n": guarded_correct_n,
            "guarded_wrong_n": guarded_wrong_n,
            "guarded_abstained_unknown_n": guarded_abstained_n,
            "guarded_answered_coverage": _ratio(guarded_answered_n, n),
            "guarded_selective_accuracy": _ratio(
                guarded_correct_n, guarded_answered_n
            ),
        }

    return {
        "evaluation_target": "non_blind_prior_reviewed_pilot_diagnostic",
        "school_n": len(pilot_school_ids),
        "aggregate": summary(observations),
        "by_school": {
            school_id: summary(
                [row for row in observations if row["school_id"] == school_id]
            )
            for school_id in sorted(pilot_school_ids)
        },
        "limitations": [
            "The same pilot rows informed pipeline development, so this is not blind validation.",
            "Three schools and 27 correlated field-school observations are too small for calibration claims.",
            "Review efficiency must be re-estimated on the six frozen blind-reference schools.",
        ],
    }


def pilot_auditor_diagnostic(
    raw_directory: Path,
    audit_directory: Path,
    reviewed_rows: list[dict[str, str]],
    pilot_school_ids: set[str],
) -> dict[str, object]:
    """Measure text-auditor capture and its guarded combination on pilot rows."""
    reviewed_by_id = {row["school_id"]: row for row in reviewed_rows}
    observations: list[dict[str, object]] = []
    for school_id in sorted(pilot_school_ids):
        reviewed = reviewed_by_id.get(school_id)
        if reviewed is None:
            raise ValueError(f"pilot school is missing from reviewed measurements: {school_id}")
        raw_path = raw_directory / f"{school_id}.json"
        audit_path = audit_directory / f"{school_id}.json"
        try:
            raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
            audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
            measurements = raw_record["parsed_output"]["measurements"]
            deterministic_flags = set(
                raw_record["uncertainty_assessment"]["pipeline_review_fields"]
            )
            audited = audit_record["audited_uncertainty_assessment"]
            auditor_flags = set(audited["auditor_review_fields"])
            final_flags = set(audited["final_review_fields"])
            final_auto_accept = set(audited["final_auto_accept_candidate_fields"])
            override_fields = {
                row["field"] for row in audit_record.get("auditor_safety_overrides", [])
            }
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"invalid pilot auditor inputs for {school_id}: {error}"
            ) from error
        for field in MEASUREMENT_FIELDS:
            prediction = _normalized(measurements[field]["value"])
            truth = _normalized(reviewed[field])
            if prediction == "unknown":
                outcome = "abstained_unknown"
            else:
                outcome = "correct" if _correct(field, prediction, truth) else "wrong"
            observations.append(
                {
                    "school_id": school_id,
                    "field": field,
                    "outcome": outcome,
                    "deterministic_flagged": field in deterministic_flags,
                    "auditor_flagged": field in auditor_flags,
                    "final_flagged": field in final_flags,
                    "final_auto_accept": field in final_auto_accept,
                    "auditor_safety_override": field in override_fields,
                }
            )

    def summary(rows: list[dict[str, object]]) -> dict[str, object]:
        n = len(rows)
        problems = [row for row in rows if row["outcome"] != "correct"]
        auditor_flagged = [row for row in rows if row["auditor_flagged"]]
        final_flagged = [row for row in rows if row["final_flagged"]]
        final_auto = [row for row in rows if row["final_auto_accept"]]
        return {
            "field_school_n": n,
            "problem_n": len(problems),
            "auditor_flag_n": len(auditor_flagged),
            "auditor_review_rate": _ratio(len(auditor_flagged), n),
            "auditor_problem_capture_n": sum(
                row["auditor_flagged"] for row in problems
            ),
            "auditor_problem_capture_rate": _ratio(
                sum(row["auditor_flagged"] for row in problems), len(problems)
            ),
            "auditor_silent_wrong_n": sum(
                row["outcome"] == "wrong" and not row["auditor_flagged"]
                for row in rows
            ),
            "auditor_flag_precision": _ratio(
                sum(row["outcome"] != "correct" for row in auditor_flagged),
                len(auditor_flagged),
            ),
            "auditor_incremental_problem_capture_n": sum(
                row["auditor_flagged"] and not row["deterministic_flagged"]
                for row in problems
            ),
            "auditor_safety_override_n": sum(
                row["auditor_safety_override"] for row in rows
            ),
            "final_flag_n": len(final_flagged),
            "final_review_rate": _ratio(len(final_flagged), n),
            "final_problem_capture_n": sum(row["final_flagged"] for row in problems),
            "final_problem_capture_rate": _ratio(
                sum(row["final_flagged"] for row in problems), len(problems)
            ),
            "final_silent_wrong_n": sum(
                row["outcome"] == "wrong" and not row["final_flagged"]
                for row in rows
            ),
            "final_flag_precision": _ratio(
                sum(row["outcome"] != "correct" for row in final_flagged),
                len(final_flagged),
            ),
            "final_auto_accept_n": len(final_auto),
            "final_auto_accept_correct_n": sum(
                row["outcome"] == "correct" for row in final_auto
            ),
            "final_auto_accept_precision": _ratio(
                sum(row["outcome"] == "correct" for row in final_auto),
                len(final_auto),
            ),
        }

    return {
        "evaluation_target": "non_blind_text_auditor_pilot_diagnostic",
        "school_n": len(pilot_school_ids),
        "aggregate": summary(observations),
        "by_school": {
            school_id: summary(
                [row for row in observations if row["school_id"] == school_id]
            )
            for school_id in sorted(pilot_school_ids)
        },
        "limitations": [
            "The pilot rows informed pipeline development, so this is not blind validation.",
            "The auditor sees primary reasoning but no imagery and cannot verify visual claims.",
            "Three schools and 27 correlated field-school observations are too small for general accuracy claims.",
        ],
    }
