from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import median

from .schema import CONFIDENCE_COLUMN, MEASUREMENT_FIELDS


COUNT_FIELDS = {"portable_classroom_count", "full_size_sports_fields", "hard_courts"}
CALIBRATION_BINS = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.0)]
SPARSE_VALIDATION_SCHOOL_THRESHOLD = 10


@dataclass(frozen=True)
class CalibrationObservation:
    school_id: str
    field: str
    confidence: float
    correct: int
    prediction: str
    truth: str


def _correct(field: str, prediction: str, truth: str) -> bool:
    prediction = prediction.strip().lower()
    truth = truth.strip().lower()
    if field != "solar_area_m2":
        return prediction == truth
    try:
        predicted_area = float(prediction)
        true_area = float(truth)
    except ValueError:
        return prediction == truth
    tolerance = max(25.0, 0.25 * true_area)
    return abs(predicted_area - true_area) <= tolerance


def observations(
    predictions: list[dict[str, str]],
    ground_truth: list[dict[str, str]],
) -> list[CalibrationObservation]:
    truth_by_id = {row["school_id"]: row for row in ground_truth}
    result: list[CalibrationObservation] = []
    for prediction in predictions:
        school_id = prediction["school_id"]
        truth = truth_by_id.get(school_id)
        if not truth:
            continue
        for field in MEASUREMENT_FIELDS:
            truth_value = truth.get(field, "").strip().lower()
            predicted_value = prediction.get(field, "").strip().lower()
            confidence_value = prediction.get(CONFIDENCE_COLUMN[field], "").strip()
            if not truth_value or truth_value == "unknown":
                continue
            if not predicted_value or not confidence_value:
                continue
            if field == "solar_area_m2":
                try:
                    true_area = float(truth_value)
                except ValueError:
                    continue
                # Solar-negative rows do not provide an area-estimation test.
                if true_area <= 0:
                    continue
            try:
                confidence = float(confidence_value)
            except ValueError:
                continue
            if not 0 <= confidence <= 1:
                continue
            result.append(
                CalibrationObservation(
                    school_id=school_id,
                    field=field,
                    confidence=confidence,
                    correct=int(_correct(field, predicted_value, truth_value)),
                    prediction=predicted_value,
                    truth=truth_value,
                )
            )
    return result


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> dict[str, float] | None:
    if n == 0:
        return None
    proportion = successes / n
    denominator = 1 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / denominator
    margin = z * sqrt((proportion * (1 - proportion) + z**2 / (4 * n)) / n) / denominator
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def _base_summary(items: list[CalibrationObservation]) -> dict[str, object]:
    if not items:
        return {
            "n": 0,
            "school_n": 0,
            "correct_n": 0,
            "incorrect_n": 0,
            "accuracy": None,
            "accuracy_95pct_wilson": None,
            "mean_confidence": None,
            "brier_score": None,
        }
    n = len(items)
    correct_n = sum(item.correct for item in items)
    return {
        "n": n,
        "school_n": len({item.school_id for item in items}),
        "correct_n": correct_n,
        "incorrect_n": n - correct_n,
        "accuracy": correct_n / n,
        "accuracy_95pct_wilson": _wilson_interval(correct_n, n),
        "mean_confidence": sum(item.confidence for item in items) / n,
        "brier_score": sum((item.confidence - item.correct) ** 2 for item in items) / n,
    }


def _calibration_bins(items: list[CalibrationObservation]) -> list[dict[str, object]]:
    bins: list[dict[str, object]] = []
    for lower, upper in CALIBRATION_BINS:
        selected = [
            item for item in items
            if item.confidence >= lower and (item.confidence < upper or (upper == 1 and item.confidence <= 1))
        ]
        if selected:
            bin_report = _base_summary(selected)
            bin_report["range"] = f"[{lower:.2f}, {upper:.2f}{']' if upper == 1 else ')'}"
            bins.append(bin_report)
    return bins


def _field_summary(field: str, items: list[CalibrationObservation]) -> dict[str, object]:
    report = _base_summary(items)
    if field in COUNT_FIELDS:
        numeric_pairs: list[tuple[int, int]] = []
        for item in items:
            try:
                numeric_pairs.append((int(item.prediction), int(item.truth)))
            except ValueError:
                continue
        report["numeric_n"] = len(numeric_pairs)
        report["mean_absolute_error"] = (
            sum(abs(prediction - truth) for prediction, truth in numeric_pairs) / len(numeric_pairs)
            if numeric_pairs
            else None
        )
    elif field == "solar_area_m2":
        numeric_pairs: list[tuple[float, float]] = []
        for item in items:
            try:
                numeric_pairs.append((float(item.prediction), float(item.truth)))
            except ValueError:
                continue
        absolute_errors = [abs(prediction - truth) for prediction, truth in numeric_pairs]
        percentage_errors = [error / truth for error, (_, truth) in zip(absolute_errors, numeric_pairs)]
        report.update(
            {
                "numeric_n": len(numeric_pairs),
                "mean_absolute_error_m2": (
                    sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
                ),
                "median_absolute_percentage_error": (
                    median(percentage_errors) if percentage_errors else None
                ),
                "within_25_percent_rate": (
                    sum(error <= 0.25 for error in percentage_errors) / len(percentage_errors)
                    if percentage_errors
                    else None
                ),
            }
        )
    return report


def summarize(items: list[CalibrationObservation]) -> dict[str, object]:
    report = _base_summary(items)
    bins = _calibration_bins(items)
    report["expected_calibration_error"] = (
        sum(
            bin_result["n"]
            * abs(float(bin_result["accuracy"]) - float(bin_result["mean_confidence"]))
            for bin_result in bins
        )
        / len(items)
        if items
        else None
    )
    report["bins"] = bins
    warnings: list[str] = []
    school_n = int(report["school_n"])
    if 0 < school_n < SPARSE_VALIDATION_SCHOOL_THRESHOLD:
        warnings.append(
            f"Only {school_n} schools contribute to validation. Confidence bins and error rates are "
            "descriptive only; this software warning threshold is not a professor-specified minimum."
        )
    sparse_bins = [str(bin_result["range"]) for bin_result in bins if int(bin_result["school_n"]) < 3]
    if sparse_bins:
        warnings.append(
            "Confidence bins with fewer than three contributing schools should normally be omitted from "
            f"the memo: {', '.join(sparse_bins)}."
        )
    report["calibration_status"] = "descriptive_only" if warnings else "sample_size_caution_not_triggered"
    report["field_observations_are_clustered_by_school"] = True
    report["warnings"] = warnings
    report["by_field"] = {
        field: _field_summary(field, [item for item in items if item.field == field])
        for field in MEASUREMENT_FIELDS
        if any(item.field == field for item in items)
    }
    return report
