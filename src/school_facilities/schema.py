from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


IDENTITY_COLUMNS = ["school_id", "school_name", "city", "state"]

MEASUREMENT_FIELDS = [
    "solar_present",
    "solar_area_m2",
    "portable_classroom_count",
    "perimeter_fencing",
    "dominant_fence_type",
    "running_track",
    "full_size_sports_fields",
    "hard_courts",
    "pool",
]

CONFIDENCE_COLUMN = {field: f"{field}_confidence" for field in MEASUREMENT_FIELDS}

MEASUREMENT_COLUMNS = [
    *IDENTITY_COLUMNS,
    "imagery_source",
    "imagery_vintage",
    "campus_resolution_notes",
]
for _field in MEASUREMENT_FIELDS:
    MEASUREMENT_COLUMNS.extend([_field, CONFIDENCE_COLUMN[_field]])
MEASUREMENT_COLUMNS.extend(["review_status", "failure_notes"])

GROUND_TRUTH_COLUMNS = [*IDENTITY_COLUMNS, *MEASUREMENT_FIELDS, "verification_notes"]

YES_NO_UNKNOWN = {"yes", "no", "unknown"}
FENCING = {"full", "partial", "none", "unknown"}
FENCE_TYPES = {
    "chain-link",
    "wrought-iron",
    "wall",
    "other",
    "mixed",
    "none",
    "unknown",
}
REVIEW_STATUS = {"unreviewed", "reviewed", "needs-review"}
UNKNOWN_VINTAGE = re.compile(r"unknown \(retrieved (\d{4}-\d{2}-\d{2})\)")


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_schools(path: Path) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    rows = read_csv(path)
    required = {
        "school_id",
        "school_name",
        "district",
        "city",
        "state",
        "level",
        "enrollment_2024",
        "latitude",
        "longitude",
    }
    actual = set(rows[0]) if rows else set()
    missing = sorted(required - actual)
    if missing:
        errors.append(f"school input is missing columns: {', '.join(missing)}")
    if len(rows) != 25:
        warnings.append(f"expected 25 schools from the supplied sample; found {len(rows)}")
    ids = [row.get("school_id", "") for row in rows]
    if any(not value for value in ids):
        errors.append("one or more school_id values are blank")
    if len(ids) != len(set(ids)):
        errors.append("school_id values are not unique")
    for number, row in enumerate(rows, start=2):
        school_id = row.get("school_id", f"row {number}")
        if len(school_id) != 12 or not school_id.isdigit():
            errors.append(f"{school_id}: school_id must be a 12-digit string")
        try:
            latitude = float(row.get("latitude", ""))
            longitude = float(row.get("longitude", ""))
        except ValueError:
            errors.append(f"{school_id}: latitude/longitude must be numeric")
            continue
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            errors.append(f"{school_id}: latitude/longitude is outside valid bounds")
    return ValidationResult(errors, warnings)


def measurement_template(school_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for school in school_rows:
        row = {column: "" for column in MEASUREMENT_COLUMNS}
        for column in IDENTITY_COLUMNS:
            row[column] = school[column]
        row["review_status"] = "unreviewed"
        output.append(row)
    return output


def ground_truth_template(school_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for school in school_rows:
        row = {column: "" for column in GROUND_TRUTH_COLUMNS}
        for column in IDENTITY_COLUMNS:
            row[column] = school[column]
        output.append(row)
    return output


def _is_nonnegative_number(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return math.isfinite(number) and number >= 0


def _is_nonnegative_integer_or_unknown(value: str) -> bool:
    if value == "unknown":
        return True
    try:
        number = int(value)
    except ValueError:
        return False
    return number >= 0 and str(number) == value


def _is_valid_imagery_vintage(value: str) -> bool:
    """Accept a capture year/date or an explicitly documented unknown vintage."""
    normalized = value.strip().lower()
    if normalized == "unknown":
        return True
    unknown_match = UNKNOWN_VINTAGE.fullmatch(normalized)
    if unknown_match:
        try:
            retrieval_date = date.fromisoformat(unknown_match.group(1))
        except ValueError:
            return False
        return retrieval_date <= date.today()

    formats = {
        4: "%Y",
        7: "%Y-%m",
        10: "%Y-%m-%d",
    }
    date_format = formats.get(len(normalized))
    if not date_format:
        return False
    try:
        parsed = datetime.strptime(normalized, date_format)
    except ValueError:
        return False
    return 1900 <= parsed.year <= date.today().year


def validate_measurements(
    measurements_path: Path,
    schools_path: Path,
    *,
    final: bool,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    schools = read_csv(schools_path)
    rows = read_csv(measurements_path)
    expected_ids = [row["school_id"] for row in schools]

    actual_columns = set(rows[0]) if rows else set()
    missing_columns = sorted(set(MEASUREMENT_COLUMNS) - actual_columns)
    if missing_columns:
        errors.append(f"measurements file is missing columns: {', '.join(missing_columns)}")
        return ValidationResult(errors, warnings)

    actual_ids = [row["school_id"] for row in rows]
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("measurements contain duplicate school_id values")
    missing_ids = sorted(set(expected_ids) - set(actual_ids))
    extra_ids = sorted(set(actual_ids) - set(expected_ids))
    if missing_ids:
        errors.append(f"measurements are missing {len(missing_ids)} supplied school IDs")
    if extra_ids:
        errors.append(f"measurements contain {len(extra_ids)} unexpected school IDs")

    for row in rows:
        school_id = row["school_id"]
        normalized = {key: value.strip().lower() for key, value in row.items() if value is not None}
        status = normalized.get("review_status", "")
        if status and status not in REVIEW_STATUS:
            errors.append(f"{school_id}: invalid review_status {status!r}")
        if not final:
            continue

        if status != "reviewed":
            errors.append(f"{school_id}: review_status must be 'reviewed' for final validation")
        if not normalized.get("imagery_source"):
            errors.append(f"{school_id}: imagery_source is blank")
        imagery_vintage = normalized.get("imagery_vintage", "")
        if not imagery_vintage:
            errors.append(f"{school_id}: imagery_vintage is blank")
        elif not _is_valid_imagery_vintage(imagery_vintage):
            errors.append(
                f"{school_id}: imagery_vintage must be YYYY, YYYY-MM, YYYY-MM-DD, "
                "'unknown', or 'unknown (retrieved YYYY-MM-DD)'"
            )

        categorical = {
            "solar_present": YES_NO_UNKNOWN,
            "perimeter_fencing": FENCING,
            "dominant_fence_type": FENCE_TYPES,
            "running_track": YES_NO_UNKNOWN,
            "pool": YES_NO_UNKNOWN,
        }
        for field, allowed in categorical.items():
            value = normalized.get(field, "")
            if value not in allowed:
                errors.append(f"{school_id}: {field} must be one of {sorted(allowed)}")

        solar_area = normalized.get("solar_area_m2", "")
        if solar_area != "unknown" and not _is_nonnegative_number(solar_area):
            errors.append(f"{school_id}: solar_area_m2 must be non-negative or 'unknown'")
        solar_present = normalized.get("solar_present")
        if solar_present == "no" and (
            not _is_nonnegative_number(solar_area) or float(solar_area) != 0
        ):
            errors.append(f"{school_id}: solar_area_m2 must be 0 when solar_present is no")
        if solar_present == "yes" and solar_area == "unknown":
            warnings.append(f"{school_id}: solar is present but its area is unknown")
        if solar_present == "yes" and _is_nonnegative_number(solar_area) and float(solar_area) == 0:
            errors.append(f"{school_id}: solar_area_m2 must be positive when solar_present is yes")
        if solar_present == "unknown" and solar_area != "unknown":
            errors.append(f"{school_id}: solar_area_m2 must be unknown when solar_present is unknown")

        fencing = normalized.get("perimeter_fencing")
        fence_type = normalized.get("dominant_fence_type")
        if fencing == "none" and fence_type != "none":
            errors.append(f"{school_id}: dominant_fence_type must be none when perimeter_fencing is none")
        if fencing in {"full", "partial"} and fence_type == "none":
            errors.append(
                f"{school_id}: dominant_fence_type cannot be none when perimeter_fencing is {fencing}"
            )

        for field in ["portable_classroom_count", "full_size_sports_fields", "hard_courts"]:
            if not _is_nonnegative_integer_or_unknown(normalized.get(field, "")):
                errors.append(f"{school_id}: {field} must be a non-negative integer or 'unknown'")

        for field, confidence_column in CONFIDENCE_COLUMN.items():
            raw = normalized.get(confidence_column, "")
            try:
                confidence = float(raw)
            except ValueError:
                errors.append(f"{school_id}: {confidence_column} must be a number from 0 to 1")
                continue
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                errors.append(f"{school_id}: {confidence_column} must be from 0 to 1")
            if normalized.get(field) == "unknown" and confidence > 0.5:
                warnings.append(f"{school_id}: {field} is unknown but confidence is above 0.5")

        unknown_fields = [field for field in MEASUREMENT_FIELDS if normalized.get(field) == "unknown"]
        vintage_unknown = imagery_vintage.startswith("unknown")
        if (unknown_fields or vintage_unknown) and not normalized.get("failure_notes"):
            reasons: list[str] = []
            if unknown_fields:
                reasons.append(f"unknown fields: {', '.join(unknown_fields)}")
            if vintage_unknown:
                reasons.append("unknown imagery vintage")
            errors.append(f"{school_id}: failure_notes is required for {'; '.join(reasons)}")

    return ValidationResult(errors, warnings)


def validate_ground_truth(
    ground_truth_path: Path,
    schools_path: Path,
    required_school_ids: set[str],
) -> ValidationResult:
    """Validate only the frozen blind-reference rows; other template rows stay blank."""
    errors: list[str] = []
    warnings: list[str] = []
    schools = read_csv(schools_path)
    known_ids = {row["school_id"] for row in schools}
    unknown_required = sorted(required_school_ids - known_ids)
    if unknown_required:
        errors.append("required reference IDs are absent from schools_sample.csv: " + ", ".join(unknown_required))

    rows = read_csv(ground_truth_path)
    actual_columns = set(rows[0]) if rows else set()
    missing_columns = sorted(set(GROUND_TRUTH_COLUMNS) - actual_columns)
    if missing_columns:
        errors.append("ground truth is missing columns: " + ", ".join(missing_columns))
        return ValidationResult(errors, warnings)
    ids = [row["school_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("ground truth contains duplicate school_id values")
    missing_rows = sorted(required_school_ids - set(ids))
    if missing_rows:
        errors.append("ground truth is missing required rows: " + ", ".join(missing_rows))

    categorical = {
        "solar_present": YES_NO_UNKNOWN,
        "perimeter_fencing": FENCING,
        "dominant_fence_type": FENCE_TYPES,
        "running_track": YES_NO_UNKNOWN,
        "pool": YES_NO_UNKNOWN,
    }
    for row in rows:
        school_id = row["school_id"]
        if school_id not in required_school_ids:
            continue
        normalized = {key: (value or "").strip().lower() for key, value in row.items()}
        blank = [field for field in MEASUREMENT_FIELDS if not normalized.get(field)]
        if blank:
            errors.append(f"{school_id}: blind reference fields are blank: {', '.join(blank)}")
            continue
        for field, allowed in categorical.items():
            if normalized[field] not in allowed:
                errors.append(f"{school_id}: {field} must be one of {sorted(allowed)}")
        solar_present = normalized["solar_present"]
        solar_area = normalized["solar_area_m2"]
        if solar_area != "unknown" and not _is_nonnegative_number(solar_area):
            errors.append(f"{school_id}: solar_area_m2 must be non-negative or 'unknown'")
        if solar_present == "no" and (
            not _is_nonnegative_number(solar_area) or float(solar_area) != 0
        ):
            errors.append(f"{school_id}: solar_area_m2 must be 0 when solar_present is no")
        if solar_present == "yes" and _is_nonnegative_number(solar_area) and float(solar_area) == 0:
            errors.append(f"{school_id}: solar_area_m2 must be positive or unknown when solar_present is yes")
        if solar_present == "unknown" and solar_area != "unknown":
            errors.append(f"{school_id}: solar_area_m2 must be unknown when solar_present is unknown")
        fencing = normalized["perimeter_fencing"]
        fence_type = normalized["dominant_fence_type"]
        if fencing == "none" and fence_type != "none":
            errors.append(f"{school_id}: dominant_fence_type must be none when perimeter_fencing is none")
        if fencing in {"full", "partial"} and fence_type == "none":
            errors.append(f"{school_id}: dominant_fence_type cannot be none when fencing is {fencing}")
        for field in ("portable_classroom_count", "full_size_sports_fields", "hard_courts"):
            if not _is_nonnegative_integer_or_unknown(normalized[field]):
                errors.append(f"{school_id}: {field} must be a non-negative integer or 'unknown'")
        unknown_fields = [field for field in MEASUREMENT_FIELDS if normalized[field] == "unknown"]
        if unknown_fields and not normalized.get("verification_notes"):
            errors.append(
                f"{school_id}: verification_notes is required for unknown fields: "
                + ", ".join(unknown_fields)
            )
    return ValidationResult(errors, warnings)
