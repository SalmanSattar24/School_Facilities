from __future__ import annotations

import hashlib
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .configuration import validate_configuration
from .credentials import CredentialError, load_api_key
from .schema import (
    CONFIDENCE_COLUMN,
    MEASUREMENT_COLUMNS,
    MEASUREMENT_FIELDS,
    read_csv,
    validate_ground_truth,
    validate_schools,
    write_csv,
)


CONFIDENCE_SCORES = {"0.95", "0.80", "0.60", "0.40", "0.20"}
FIELD_CHOICES: dict[str, set[str] | None] = {
    "solar_present": {"yes", "no", "unknown"},
    "solar_area_m2": None,
    "portable_classroom_count": None,
    "perimeter_fencing": {"full", "partial", "none", "unknown"},
    "dominant_fence_type": {
        "chain-link",
        "wrought-iron",
        "wall",
        "other",
        "mixed",
        "none",
        "unknown",
    },
    "running_track": {"yes", "no", "unknown"},
    "full_size_sports_fields": None,
    "hard_courts": None,
    "pool": {"yes", "no", "unknown"},
}
COUNT_FIELDS = {
    "portable_classroom_count",
    "full_size_sports_fields",
    "hard_courts",
}


@dataclass(frozen=True)
class OperatorCheck:
    name: str
    ok: bool
    detail: str


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_school_ids(root: Path) -> set[str]:
    pilot = _load_json(root / "config" / "pilot_schools.json")
    values = pilot.get("excluded_validation_school_ids", [])
    if not isinstance(values, list):
        raise ValueError("excluded_validation_school_ids must be a list")
    return {str(value) for value in values}


def blind_reference_is_complete(root: Path, school_id: str) -> bool:
    path = root / "data" / "validation" / "ground_truth.csv"
    rows = [row for row in read_csv(path) if row["school_id"] == school_id]
    return len(rows) == 1 and all(rows[0].get(field, "").strip() for field in MEASUREMENT_FIELDS)


def authorize_validation_unblinding(
    root: Path, school_id: str, reference_sha256: str | None
) -> None:
    validation_ids = validation_school_ids(root)
    if school_id not in validation_ids:
        return
    validation = validate_ground_truth(
        root / "data" / "validation" / "ground_truth.csv",
        root / "schools_sample.csv",
        validation_ids,
    )
    if not validation.ok:
        raise ValueError(
            "all six blind reference rows must be complete and valid before any validation "
            "output is opened: " + "; ".join(validation.errors)
        )
    if not reference_sha256:
        raise ValueError(
            "validation review requires --reference-sha256 recorded after all blind labels were frozen"
        )
    truth_path = root / "data" / "validation" / "ground_truth.csv"
    if _sha256(truth_path).lower() != reference_sha256.lower():
        raise ValueError("reference SHA-256 does not match the current blind-label file")


def raw_record_path(root: Path, school_id: str) -> Path:
    if school_id in validation_school_ids(root):
        return root / "data" / "model_outputs" / "quarantine" / "v1.10" / "raw" / f"{school_id}.json"
    return root / "data" / "model_outputs" / "final" / "v1.10" / f"{school_id}.json"


def run_doctor(root: Path, *, require_key: bool) -> list[OperatorCheck]:
    checks: list[OperatorCheck] = []
    required = [
        "pyproject.toml",
        "schools_sample.csv",
        "measurements.csv",
        "config/vlm.json",
        "config/imagery.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    checks.append(
        OperatorCheck(
            "project files",
            not missing,
            "all required files present" if not missing else "missing: " + ", ".join(missing),
        )
    )

    school_result = validate_schools(root / "schools_sample.csv")
    checks.append(
        OperatorCheck(
            "school input",
            school_result.ok,
            "25-school source is valid" if school_result.ok else "; ".join(school_result.errors),
        )
    )

    config_result = validate_configuration(root)
    checks.append(
        OperatorCheck(
            "frozen configuration",
            config_result.ok,
            "configuration and dependency pins are valid"
            if config_result.ok
            else "; ".join(config_result.errors),
        )
    )

    key_ok = False
    key_detail = "not configured (run: school-facilities save-gemini-key)"
    try:
        config = _load_json(root / "config" / "vlm.json")
        credentials = config["credentials"]
        if not isinstance(credentials, dict):
            raise KeyError("credentials")
        _, source = load_api_key(
            environment_variable=str(credentials["api_key_environment_variable"]),
            secrets_file=root / str(credentials["local_secrets_file"]),
        )
        key_ok = True
        key_detail = f"configured via {source}; value was not displayed"
    except (CredentialError, FileNotFoundError, KeyError, TypeError, ValueError):
        pass
    checks.append(OperatorCheck("Gemini credential", key_ok or not require_key, key_detail))

    validation_ids = validation_school_ids(root)
    leaked = [
        school_id
        for school_id in sorted(validation_ids)
        if (root / "data" / "model_outputs" / "final" / "v1.10" / f"{school_id}.json").exists()
    ]
    checks.append(
        OperatorCheck(
            "blind-validation isolation",
            not leaked,
            "no validation response is present in the ordinary output directory"
            if not leaked
            else "validation IDs exposed outside quarantine: " + ", ".join(leaked),
        )
    )
    return checks


def workflow_rows(root: Path) -> list[dict[str, str]]:
    schools = read_csv(root / "schools_sample.csv")
    measurement_rows = {
        row["school_id"]: row for row in read_csv(root / "measurements.csv")
    }
    truth_rows = {
        row["school_id"]: row
        for row in read_csv(root / "data" / "validation" / "ground_truth.csv")
    }
    validation_ids = validation_school_ids(root)
    nonvalidation_reviews_pending = any(
        school["school_id"] not in validation_ids
        and measurement_rows.get(school["school_id"], {}).get("review_status") != "reviewed"
        for school in schools
    )
    rows: list[dict[str, str]] = []
    for school in schools:
        school_id = school["school_id"]
        reviewed = measurement_rows.get(school_id, {}).get("review_status") == "reviewed"
        is_validation = school_id in validation_ids
        truth_complete = all(
            truth_rows.get(school_id, {}).get(field, "").strip()
            for field in MEASUREMENT_FIELDS
        )
        context = root / "data" / "imagery" / school_id / "context.json"
        resolution = root / "data" / "campus_resolutions" / f"{school_id}.json"
        detail = root / "data" / "imagery" / school_id / "detail.json"
        raw = raw_record_path(root, school_id)
        audit = root / "data" / "model_outputs" / "audits" / "v1.10" / f"{school_id}.json"
        resolution_status = ""
        needs_boundary_review = False
        if resolution.is_file():
            try:
                resolved = _load_json(resolution)
                resolution_status = str(resolved.get("status", ""))
                needs_boundary_review = bool(resolved.get("requires_human_review"))
            except (OSError, ValueError, json.JSONDecodeError):
                resolution_status = "invalid"

        if reviewed:
            stage, action = "REVIEWED", "none"
        elif (
            is_validation
            and raw.is_file()
            and not truth_complete
            and nonvalidation_reviews_pending
        ):
            stage, action = (
                "WAITING",
                "finish all 19 non-validation reviews before blind reference labeling",
            )
        elif is_validation and raw.is_file() and not truth_complete:
            stage, action = (
                "BLIND LABEL",
                "label ground_truth.csv from imagery only; do not open quarantined output",
            )
        elif is_validation and raw.is_file() and truth_complete:
            stage, action = (
                "READY TO UNBLIND",
                "hash ground_truth.csv, run evaluate-vlm, then prepare-review",
            )
        elif raw.is_file():
            stage, action = "FIELD REVIEW", f"prepare-review --school-id {school_id}"
            if not is_validation and not audit.is_file():
                action = f"audit-vlm --school-id {school_id} (optional), then {action}"
        elif detail.is_file():
            stage, action = (
                "INFERENCE",
                f"assess-school --school-id {school_id} --live --confirm-free-tier-production",
            )
        elif resolution.is_file() and resolution_status == "confirmed" and not needs_boundary_review:
            stage, action = "DETAIL IMAGERY", f"fetch-naip-detail --school-id {school_id}"
        elif resolution.is_file():
            stage, action = "CAMPUS REVIEW", f"prepare-campus-review --school-id {school_id}"
        elif context.is_file():
            stage, action = "CAMPUS RESOLUTION", f"resolve-campus --school-id {school_id}"
        else:
            stage, action = "CONTEXT IMAGERY", f"fetch-naip-context --school-id {school_id}"
        rows.append(
            {
                "school_id": school_id,
                "school_name": school["school_name"],
                "validation": "yes" if is_validation else "no",
                "stage": stage,
                "next_action": action,
            }
        )
    return rows


def _protocol_questions(root: Path) -> dict[str, str]:
    protocol = _load_json(root / "config" / "vlm_field_protocol.json")
    result: dict[str, str] = {}
    features = protocol.get("features", {})
    if isinstance(features, dict):
        for specification in features.values():
            if not isinstance(specification, dict):
                continue
            for question in specification.get("questions", []):
                if isinstance(question, dict):
                    question_id = str(question.get("question_id", question.get("id", "")))
                    text = str(question.get("question", question.get("text", "")))
                    if question_id:
                        result[question_id] = text
    return result


def _relative_url(path: Path, output_path: Path) -> str:
    return Path(os.path.relpath(path, output_path.parent)).as_posix()


def prepare_blind_review_packet(
    root: Path,
    school_id: str,
    *,
    output_path: Path | None = None,
) -> Path:
    if school_id not in validation_school_ids(root):
        raise ValueError("blind-review packets are only for the six frozen validation schools")
    schools = [row for row in read_csv(root / "schools_sample.csv") if row["school_id"] == school_id]
    if len(schools) != 1:
        raise ValueError(f"school_id not found exactly once: {school_id}")
    context_path = root / "data" / "imagery" / school_id / "context.jpg"
    detail_path = root / "data" / "imagery" / school_id / "detail.jpg"
    for path in (context_path, detail_path):
        if not path.is_file():
            raise FileNotFoundError(f"blind-review image is missing: {path}")
    if output_path is None:
        output_path = root / "data" / "blind_review_packets" / f"{school_id}.html"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    school = schools[0]
    field_rows = "".join(
        f"<tr><td><code>{html.escape(field)}</code></td><td></td><td></td></tr>"
        for field in MEASUREMENT_FIELDS
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Blind reference {html.escape(school_id)}</title><style>
body{{font:16px/1.45 system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#1f2937}}
.warning{{background:#fee2e2;border:2px solid #b91c1c;padding:1rem}} .images{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
img{{width:100%;height:auto;border:1px solid #666}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #aaa;padding:.5rem;height:1.5rem}}
@media(max-width:900px){{.images{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Blind reference: {html.escape(school['school_name'])}</h1><p><strong>School ID:</strong> {html.escape(school_id)}</p>
<div class="warning"><strong>Leakage control:</strong> This packet contains imagery only. Do not open model-output, audit, review-packet, or reviewed-CSV material while assigning the reference label. Record uncertainty as <code>unknown</code>; never consult the prediction to resolve doubt.</div>
<h2>Frozen imagery</h2><div class="images"><figure><img src="{html.escape(_relative_url(context_path, output_path))}" alt="Context"><figcaption>Context</figcaption></figure><figure><img src="{html.escape(_relative_url(detail_path, output_path))}" alt="Detail"><figcaption>Detail</figcaption></figure></div>
<h2>Reference worksheet</h2><table><thead><tr><th>Field</th><th>Blind label</th><th>Verification note</th></tr></thead><tbody>{field_rows}</tbody></table>
<p>Enter the labels in <code>data/validation/ground_truth.csv</code>. Complete all six schools before hashing the file or revealing any quarantined output.</p>
</body></html>"""
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def prepare_review_packet(
    root: Path,
    school_id: str,
    *,
    output_path: Path | None = None,
    reference_sha256: str | None = None,
) -> Path:
    authorize_validation_unblinding(root, school_id, reference_sha256)
    schools = [row for row in read_csv(root / "schools_sample.csv") if row["school_id"] == school_id]
    if len(schools) != 1:
        raise ValueError(f"school_id not found exactly once: {school_id}")
    raw_path = raw_record_path(root, school_id)
    raw = _load_json(raw_path)
    audit_path = root / "data" / "model_outputs" / "audits" / "v1.10" / f"{school_id}.json"
    audit = _load_json(audit_path) if audit_path.is_file() else {}
    context_path = root / "data" / "imagery" / school_id / "context.jpg"
    detail_path = root / "data" / "imagery" / school_id / "detail.jpg"
    sidecar = _load_json(root / "data" / "imagery" / school_id / "detail.json")
    for path in (context_path, detail_path):
        if not path.is_file():
            raise FileNotFoundError(f"review image is missing: {path}")
    if output_path is None:
        output_path = root / "data" / "review_packets" / f"{school_id}.html"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = raw.get("parsed_output", {})
    measurements = parsed.get("measurements", {}) if isinstance(parsed, dict) else {}
    uncertainty = raw.get("uncertainty_assessment", {})
    guarded = uncertainty.get("guarded_measurements", {}) if isinstance(uncertainty, dict) else {}
    review_reasons = uncertainty.get("review_reasons_by_field", {}) if isinstance(uncertainty, dict) else {}
    if audit:
        audited = audit.get("audited_uncertainty_assessment", {})
        if isinstance(audited, dict):
            guarded = audited.get("guarded_measurements", guarded)
            final_review = set(audited.get("final_review_fields", []))
        else:
            final_review = set()
        audit_parsed = audit.get("parsed_output", {})
        field_audits = audit_parsed.get("field_audits", {}) if isinstance(audit_parsed, dict) else {}
    else:
        final_review = set(uncertainty.get("pipeline_review_fields", [])) if isinstance(uncertainty, dict) else set()
        field_audits = {}
    questions = _protocol_questions(root)

    table_rows: list[str] = []
    for field in MEASUREMENT_FIELDS:
        suggestion = measurements.get(field, {}) if isinstance(measurements, dict) else {}
        raw_value = suggestion.get("value", "") if isinstance(suggestion, dict) else ""
        confidence = suggestion.get("suggested_confidence", "") if isinstance(suggestion, dict) else ""
        evidence = suggestion.get("evidence", "") if isinstance(suggestion, dict) else ""
        reasons = review_reasons.get(field, []) if isinstance(review_reasons, dict) else []
        field_audit = field_audits.get(field, {}) if isinstance(field_audits, dict) else {}
        audit_status = field_audit.get("status", "not run") if isinstance(field_audit, dict) else "not run"
        audit_text = field_audit.get("evidence_assessment", "") if isinstance(field_audit, dict) else ""
        table_rows.append(
            "<tr>"
            f"<td><code>{html.escape(field)}</code></td>"
            f"<td>{html.escape(str(raw_value))}</td>"
            f"<td>{html.escape(str(confidence))}</td>"
            f"<td>{html.escape(str(guarded.get(field, '')) if isinstance(guarded, dict) else '')}</td>"
            f"<td>{'YES' if field in final_review else 'no'}</td>"
            f"<td>{html.escape(', '.join(str(item) for item in reasons))}</td>"
            f"<td>{html.escape(str(evidence))}</td>"
            f"<td>{html.escape(str(audit_status))}: {html.escape(str(audit_text))}</td>"
            "</tr>"
        )

    feature_blocks: list[str] = []
    assessments = parsed.get("feature_assessments", []) if isinstance(parsed, dict) else []
    if isinstance(assessments, list):
        for assessment in assessments:
            if not isinstance(assessment, dict):
                continue
            answers: list[str] = []
            for answer in assessment.get("question_answers", []):
                if not isinstance(answer, dict):
                    continue
                qid = str(answer.get("question_id", ""))
                answers.append(
                    "<li>"
                    f"<strong>{html.escape(qid)}</strong> {html.escape(questions.get(qid, ''))}<br>"
                    f"Answer: {html.escape(str(answer.get('answer', '')))}; "
                    f"location: {html.escape(str(answer.get('location', '')))}; "
                    f"observation: {html.escape(str(answer.get('observation', '')))}"
                    "</li>"
                )
            feature_blocks.append(
                f"<details><summary>{html.escape(str(assessment.get('feature', 'feature')))}</summary>"
                f"<p>{html.escape(str(assessment.get('derivation_summary', '')))}</p>"
                f"<ol>{''.join(answers)}</ol></details>"
            )

    capture = str(sidecar.get("capture_datetime_or_vintage", "unknown"))[:10]
    source = str(sidecar.get("source", "unknown"))
    school = schools[0]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Review {html.escape(school_id)}</title>
<style>
body{{font:16px/1.45 system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#1f2937}}
.warning{{background:#fff4ce;border:2px solid #b45309;padding:1rem}} .images{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
img{{width:100%;height:auto;border:1px solid #666}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #aaa;padding:.45rem;vertical-align:top}} th{{background:#eee;position:sticky;top:0}}
code{{white-space:nowrap}} details{{margin:.6rem 0;padding:.5rem;border:1px solid #bbb}} @media(max-width:900px){{.images{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{html.escape(school['school_name'])}</h1>
<p><strong>School ID:</strong> {html.escape(school_id)} · <strong>Imagery:</strong> {html.escape(source)}, {html.escape(capture)}</p>
<div class="warning"><strong>Human decision required.</strong> Raw and guarded values are suggestions, not final labels. Inspect both images, apply the frozen definitions, use <code>unknown</code> instead of guessing, and assign human confidence independently. Then run <code>school-facilities review-school --school-id {html.escape(school_id)}</code>.</div>
<h2>Frozen imagery</h2><div class="images"><figure><img src="{html.escape(_relative_url(context_path, output_path))}" alt="Context"><figcaption>Context</figcaption></figure><figure><img src="{html.escape(_relative_url(detail_path, output_path))}" alt="Detail"><figcaption>Detail</figcaption></figure></div>
<h2>Prediction and uncertainty review</h2>
<table><thead><tr><th>Field</th><th>Raw VLM</th><th>Model confidence</th><th>Guarded suggestion</th><th>Routed to review</th><th>Deterministic reasons</th><th>VLM evidence</th><th>Text auditor</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Item-specific observable answers</h2>{''.join(feature_blocks)}
<h2>Required operator procedure</h2><ol><li>Inspect the images before accepting any suggestion.</li><li>Pay special attention to every row marked YES.</li><li>Run the interactive review command shown above.</li><li>Run <code>school-facilities validate</code> after saving.</li></ol>
</body></html>"""
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def _prompt_value(
    field: str,
    suggested: object,
    *,
    input_func: Callable[[str], str],
) -> str:
    choices = FIELD_CHOICES[field]
    while True:
        suffix = f" [{suggested}]" if suggested != "" else ""
        raw = input_func(f"{field}{suffix}: ").strip().lower()
        value = raw or str(suggested).strip().lower()
        if choices is not None and value in choices:
            return value
        if field in COUNT_FIELDS:
            if value == "unknown" or (value.isdigit() and int(value) >= 0):
                return value
        elif field == "solar_area_m2":
            if value == "unknown":
                return value
            try:
                if float(value) >= 0:
                    return value
            except ValueError:
                pass
        print("Invalid value. Use a listed category, a non-negative number/count, or unknown.")


def _prompt_confidence(
    field: str,
    value: str,
    *,
    input_func: Callable[[str], str],
) -> tuple[str, str]:
    while True:
        score = input_func(
            f"{field} human confidence (0.95/0.80/0.60/0.40/0.20): "
        ).strip()
        try:
            normalized = f"{float(score):.2f}" if score else ""
        except ValueError:
            normalized = ""
        if normalized not in CONFIDENCE_SCORES:
            print("Use exactly one frozen confidence anchor: 0.95, 0.80, 0.60, 0.40, or 0.20.")
            continue
        if value == "unknown" and normalized != "0.20":
            print("An unknown value must use confidence 0.20.")
            continue
        corroboration = ""
        if normalized == "0.95":
            corroboration = input_func(
                "0.95 requires genuinely independent corroboration. Describe that source, or leave blank to choose another score: "
            ).strip()
            if not corroboration:
                continue
        return normalized, corroboration


def review_school_interactively(
    root: Path,
    school_id: str,
    *,
    reference_sha256: str | None = None,
    input_func: Callable[[str], str] = input,
) -> Path:
    authorize_validation_unblinding(root, school_id, reference_sha256)
    raw = _load_json(raw_record_path(root, school_id))
    uncertainty = raw.get("uncertainty_assessment", {})
    guarded = uncertainty.get("guarded_measurements", {}) if isinstance(uncertainty, dict) else {}
    audit_path = root / "data" / "model_outputs" / "audits" / "v1.10" / f"{school_id}.json"
    if audit_path.is_file():
        audit = _load_json(audit_path)
        audited = audit.get("audited_uncertainty_assessment", {})
        if isinstance(audited, dict):
            guarded = audited.get("guarded_measurements", guarded)
    if not isinstance(guarded, dict):
        raise ValueError("guarded measurement suggestions are missing")

    measurement_path = root / "measurements.csv"
    rows = read_csv(measurement_path)
    matches = [row for row in rows if row["school_id"] == school_id]
    if len(matches) != 1:
        raise ValueError(f"measurement row not found exactly once: {school_id}")
    row = matches[0]
    if row.get("review_status") == "reviewed":
        answer = input_func("This row is already reviewed. Replace it? Type YES to continue: ").strip()
        if answer != "YES":
            raise ValueError("existing reviewed row was preserved")

    sidecar = _load_json(root / "data" / "imagery" / school_id / "detail.json")
    row["imagery_source"] = "USDA NAIP" if "NAIP" in str(sidecar.get("source", "")) else str(sidecar.get("source", "unknown"))
    row["imagery_vintage"] = str(sidecar.get("capture_datetime_or_vintage", "unknown"))[:10]
    row["campus_resolution_notes"] = str(sidecar.get("campus_resolution_notes", ""))

    print("Inspect the generated review packet before answering. Press Ctrl+C to cancel without saving.")
    unknown_notes: list[str] = []
    corroboration_notes: list[str] = []
    for field in MEASUREMENT_FIELDS:
        if field == "solar_area_m2" and row.get("solar_present") == "no":
            value = "0"
            print("solar_area_m2 is structurally set to 0 because solar_present is no.")
        elif field == "solar_area_m2" and row.get("solar_present") == "unknown":
            value = "unknown"
            print("solar_area_m2 is structurally set to unknown because solar_present is unknown.")
        elif field == "dominant_fence_type" and row.get("perimeter_fencing") == "none":
            value = "none"
            print("dominant_fence_type is structurally set to none because perimeter_fencing is none.")
        else:
            value = _prompt_value(field, guarded.get(field, ""), input_func=input_func)
        row[field] = value
        if value == "unknown":
            reason = input_func(f"One-line reason why {field} is unknown: ").strip()
            while not reason:
                reason = input_func("A reason is required: ").strip()
            unknown_notes.append(f"{field}: {reason}")
        confidence, corroboration = _prompt_confidence(
            field, value, input_func=input_func
        )
        row[CONFIDENCE_COLUMN[field]] = confidence
        if corroboration:
            corroboration_notes.append(f"{field} corroboration: {corroboration}")

    if row["solar_present"] == "no":
        row["solar_area_m2"] = "0"
    elif row["solar_present"] == "unknown":
        row["solar_area_m2"] = "unknown"
        row["solar_area_m2_confidence"] = "0.20"
    elif row["solar_present"] == "yes" and row["solar_area_m2"] not in {"unknown"}:
        if float(row["solar_area_m2"]) <= 0:
            raise ValueError("solar_area_m2 must be positive when solar_present is yes")
    if row["perimeter_fencing"] == "none":
        row["dominant_fence_type"] = "none"
    if row["perimeter_fencing"] in {"full", "partial"} and row["dominant_fence_type"] == "none":
        raise ValueError("dominant_fence_type cannot be none when fencing is full or partial")

    general_note = input_func("Optional additional review/failure note: ").strip()
    notes = [*unknown_notes, *corroboration_notes]
    if general_note:
        notes.append(general_note)
    row["failure_notes"] = "; ".join(notes)
    row["review_status"] = "reviewed"

    print("\nReview summary:")
    for field in MEASUREMENT_FIELDS:
        print(f"  {field}: {row[field]} ({row[CONFIDENCE_COLUMN[field]]})")
    if input_func("Type SAVE to write this row to measurements.csv: ").strip() != "SAVE":
        raise ValueError("review was not saved")

    backup = measurement_path.with_suffix(".csv.bak")
    backup.write_bytes(measurement_path.read_bytes())
    write_csv(measurement_path, MEASUREMENT_COLUMNS, rows)
    return measurement_path


def print_doctor(checks: list[OperatorCheck]) -> int:
    for check in checks:
        print(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    if all(check.ok for check in checks):
        print("Operator setup is ready. Run: school-facilities workflow-status")
        return 0
    print("Setup is not ready. Correct the FAIL items and rerun doctor.", file=sys.stderr)
    return 1
