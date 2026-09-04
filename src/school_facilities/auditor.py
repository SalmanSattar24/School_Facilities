from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .configuration import validate_configuration
from .vlm import (
    InteractionCreator,
    RequestLedger,
    VLMConfigurationError,
    VLMError,
    VLMResponseError,
    _atomic_json,
    _gemini_response_schema,
    _read_object,
    _request_fingerprint,
    _response_value,
    _sanitized_error_detail,
    _sanitized_provider_diagnostics,
    _serializable,
    _sha256,
    _within,
)


def load_auditor_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = root.resolve()
    result = validate_configuration(root)
    if not result.ok:
        raise VLMConfigurationError("; ".join(result.errors))
    config = _read_object(root / "config" / "vlm_auditor.json")
    schema = _read_object(root / config["generation"]["response_schema_path"])
    prompt = (root / config["generation"]["prompt_path"]).read_text(encoding="utf-8")
    return config, schema, prompt


def build_auditor_request(root: Path, school_id: str) -> tuple[dict[str, Any], Path]:
    root = root.resolve()
    primary_config = _read_object(root / "config" / "vlm.json")
    auditor_config, auditor_schema, auditor_prompt = load_auditor_bundle(root)
    raw_path = root / primary_config["outputs"]["raw_directory"] / f"{school_id}.json"
    if not _within(raw_path, root / "data" / "model_outputs"):
        raise VLMConfigurationError("primary output path escaped the model-output directory")
    raw_record = _read_object(raw_path)
    if raw_record.get("configuration_id") != auditor_config["inputs"]["primary_configuration_id"]:
        raise VLMConfigurationError("primary response does not use the auditor's frozen VLM version")
    if raw_record.get("school_id") != school_id:
        raise VLMConfigurationError("primary response school_id mismatch")
    field_protocol = _read_object(root / primary_config["evidence_policy"]["field_protocol_path"])
    payload = {
        "school_id": school_id,
        "primary_configuration_id": raw_record["configuration_id"],
        "authoritative_field_protocol": field_protocol,
        "parsed_output": raw_record["parsed_output"],
        "derived_solar_summary": raw_record["derived_solar_summary"],
        "derived_evidence_summary": raw_record["derived_evidence_summary"],
        "uncertainty_assessment": raw_record["uncertainty_assessment"],
    }
    generation = auditor_config["generation"]
    request = {
        "model": auditor_config["model"],
        "input": [
            {
                "type": "text",
                "text": "Audit this frozen primary evidence packet:\n"
                + json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
        ],
        "system_instruction": auditor_prompt,
        "generation_config": {
            "thinking_level": generation["thinking_level"],
            "max_output_tokens": generation["max_output_tokens"],
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": _gemini_response_schema(auditor_schema),
        },
        "service_tier": "standard",
        "store": False,
        "stream": False,
    }
    return request, raw_path


def _audit_summary(
    primary_record: dict[str, Any], parsed_audit: dict[str, Any]
) -> dict[str, Any]:
    primary_uncertainty = primary_record["uncertainty_assessment"]
    auditor_review_fields = sorted(
        field
        for field, audit in parsed_audit["field_audits"].items()
        if audit["status"] != "consistent"
        or audit["recommended_action"] != "accept_candidate"
    )
    final_review_fields = sorted(
        set(primary_uncertainty["pipeline_review_fields"]) | set(auditor_review_fields)
    )
    auto_accept = sorted(
        set(primary_uncertainty["auto_accept_candidate_fields"])
        - set(auditor_review_fields)
    )
    return {
        "auditor_review_fields": auditor_review_fields,
        "final_review_fields": final_review_fields,
        "final_auto_accept_candidate_fields": auto_accept,
        "guarded_measurements": primary_uncertainty["guarded_measurements"],
        "auditor_overwrote_primary_values": False,
    }


def _auditor_safety_overrides(
    primary_record: dict[str, Any], parsed_audit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Record auditor misses that the primary hard-conflict gate supersedes."""
    hard_conflicts = primary_record["derived_evidence_summary"]["hard_conflicts"]
    reasons_by_field: dict[str, list[str]] = {}
    for conflict in hard_conflicts:
        for field in conflict["fields"]:
            reasons_by_field.setdefault(field, []).append(conflict["code"])
    overrides: list[dict[str, Any]] = []
    for field, reasons in sorted(reasons_by_field.items()):
        audit = parsed_audit["field_audits"][field]
        if audit["recommended_action"] != "accept_candidate":
            continue
        overrides.append(
            {
                "field": field,
                "auditor_status": audit["status"],
                "auditor_recommended_action": audit["recommended_action"],
                "primary_hard_conflict_codes": sorted(set(reasons)),
                "enforced_action": "retain_primary_review_flag",
                "auditor_output_preserved": True,
            }
        )
    return overrides


def _normalize_auditor_output(
    parsed: Any,
) -> tuple[Any, list[dict[str, str]]]:
    """Normalize only known schema-version echoes from the auditor."""
    if not isinstance(parsed, dict):
        return parsed, []
    normalized = json.loads(json.dumps(parsed))
    changes: list[dict[str, str]] = []
    source_version = normalized.get("schema_version")
    if source_version in {"1.0", "1.1", "1.9.0", "1.10.0"}:
        normalized["schema_version"] = "1.2"
        changes.append(
            {"path": "schema_version", "from": source_version, "to": "1.2"}
        )

    # Some responses use the action word ``review`` as a status. Resolve it
    # conservatively from the already supplied issue codes: explicit internal
    # conflicts are contradictory; all other review cases remain insufficient
    # evidence. This never promotes a field to consistent or accept_candidate.
    field_audits = normalized.get("field_audits")
    if isinstance(field_audits, dict):
        for field, audit in field_audits.items():
            if not isinstance(audit, dict) or audit.get("status") != "review":
                continue
            issue_codes = audit.get("issue_codes", [])
            contradictory = isinstance(issue_codes, list) and any(
                isinstance(code, str)
                and any(
                    marker in code
                    for marker in ("mismatch", "conflict", "contradiction")
                )
                for code in issue_codes
            )
            replacement = "contradictory" if contradictory else "insufficient_evidence"
            audit["status"] = replacement
            changes.append(
                {
                    "path": f"field_audits.{field}.status",
                    "from": "review",
                    "to": replacement,
                }
            )
    return normalized, changes


class GeminiEvidenceAuditorClient:
    def __init__(
        self,
        root: Path,
        create_interaction: InteractionCreator,
        *,
        close_client: Callable[[], None] | None = None,
        ledger_path: Path | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = root.resolve()
        self.config, self.schema, _ = load_auditor_bundle(self.root)
        self.create_interaction = create_interaction
        self.close_client = close_client
        self.sleep = sleep
        self.ledger = RequestLedger(
            ledger_path
            or self.root / "data" / "model_outputs" / "auditor_request_ledger.json",
            self.config,
            now=now,
            sleep=sleep,
        )

    @classmethod
    def from_environment(cls, root: Path) -> GeminiEvidenceAuditorClient:
        config, _, _ = load_auditor_bundle(root)
        from .credentials import CredentialError, load_api_key

        try:
            api_key, _ = load_api_key(
                environment_variable=config["credentials"]["api_key_environment_variable"],
                secrets_file=root / config["credentials"]["local_secrets_file"],
            )
        except CredentialError as error:
            raise VLMConfigurationError(str(error)) from error
        from google import genai
        from google.genai import types

        limits = config["request_limits"]
        google_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version=config["api_version"],
                timeout=int(limits["request_timeout_seconds"]) * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        return cls(
            root,
            google_client.interactions.create,
            close_client=google_client.close,
        )

    def close(self) -> None:
        if self.close_client is not None:
            self.close_client()

    def audit(
        self,
        school_id: str,
        *,
        overwrite: bool = False,
        allow_retry: bool = False,
    ) -> Path:
        request, primary_path = build_auditor_request(self.root, school_id)
        output_path = self.root / self.config["outputs"]["directory"] / f"{school_id}.json"
        if output_path.exists() and not overwrite:
            raise VLMConfigurationError(f"auditor response already exists: {output_path}")
        primary_record = _read_object(primary_path)
        limits = self.config["request_limits"]
        response: Any = None
        maximum_attempts = (
            1 + int(limits["maximum_retries_per_school"])
            if allow_retry
            else 1
        )
        for attempt in range(1, maximum_attempts + 1):
            self.ledger.reserve(school_id, "auditor", attempt)
            try:
                response = self.create_interaction(
                    **request,
                    timeout=float(limits["request_timeout_seconds"]),
                )
                break
            except Exception as error:
                status = getattr(error, "status_code", None)
                retryable = status in set(limits["retry_statuses"])
                if attempt >= maximum_attempts or not retryable:
                    detail = _sanitized_error_detail(error)
                    suffix = f": {detail}" if detail else ""
                    raise VLMError(
                        f"Gemini auditor request failed with status {status or 'unknown'}{suffix}"
                    ) from error
                self.sleep(float(limits["retry_initial_backoff_seconds"]))

        status = _response_value(response, "status")
        output_text = _response_value(response, "output_text")
        base_record = {
            "schema_version": "1.2",
            "configuration_id": self.config["configuration_id"],
            "school_id": school_id,
            "model": self.config["model"],
            "status": status,
            "interaction_id": _response_value(response, "id"),
            "created": _response_value(response, "created"),
            "usage": _serializable(_response_value(response, "usage")),
            "request_fingerprint_sha256": _request_fingerprint(request),
            "primary_record_sha256": _sha256(primary_path),
            "output_text": output_text if isinstance(output_text, str) else None,
        }
        if status != "completed" or not isinstance(output_text, str) or not output_text.strip():
            provider_diagnostics = _sanitized_provider_diagnostics(response)
            rejected_record = {
                **base_record,
                "parsed_output": None,
                "provider_diagnostics": provider_diagnostics,
                "rejection_stage": "interaction_status_or_output",
                "validation_error": "Gemini auditor did not return a completed text response",
            }
            interaction_id = str(base_record["interaction_id"] or "unknown-interaction")
            safe_interaction_id = "".join(
                character if character.isalnum() or character in "_.-" else "_"
                for character in interaction_id
            )
            rejected_path = (
                self.root
                / self.config["outputs"]["rejected_directory"]
                / f"{school_id}-{safe_interaction_id}.json"
            )
            _atomic_json(rejected_path, rejected_record)
            diagnostic_summary = json.dumps(
                provider_diagnostics, ensure_ascii=False, sort_keys=True
            )
            if len(diagnostic_summary) > 800:
                diagnostic_summary = diagnostic_summary[:797] + "..."
            raise VLMResponseError(
                "Gemini auditor did not return a completed text response; "
                f"rejected response preserved at {rejected_path.relative_to(self.root)}; "
                f"provider diagnostics: {diagnostic_summary}"
            )
        auditor_vocabulary_normalizations: list[dict[str, str]] = []
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as error:
            validation_error = VLMResponseError(
                f"Gemini auditor output is not valid JSON: {error}"
            )
            parsed = None
        else:
            parsed, auditor_vocabulary_normalizations = _normalize_auditor_output(parsed)
            errors = sorted(
                Draft202012Validator(self.schema).iter_errors(parsed),
                key=lambda item: list(item.path),
            )
            if errors:
                first = errors[0]
                location = ".".join(str(item) for item in first.path) or "<root>"
                validation_error = VLMResponseError(
                    f"Gemini auditor output failed schema validation at {location}: {first.message}"
                )
            elif parsed["school_id"] != school_id:
                validation_error = VLMResponseError("Gemini auditor school_id mismatch")
            else:
                validation_error = None
                for field, audit in parsed["field_audits"].items():
                    if (
                        audit["status"] != "consistent"
                        and audit["recommended_action"] == "accept_candidate"
                    ):
                        validation_error = VLMResponseError(
                            f"Gemini auditor cannot accept inconsistent field {field}"
                        )
                        break
        if validation_error is not None:
            rejected_record = dict(base_record)
            rejected_record["parsed_output"] = parsed
            if auditor_vocabulary_normalizations:
                rejected_record["auditor_vocabulary_normalizations"] = (
                    auditor_vocabulary_normalizations
                )
            rejected_record["validation_error"] = str(validation_error)
            interaction_id = str(base_record["interaction_id"] or "unknown-interaction")
            safe_interaction_id = "".join(
                character if character.isalnum() or character in "_.-" else "_"
                for character in interaction_id
            )
            rejected_path = (
                self.root
                / self.config["outputs"]["rejected_directory"]
                / f"{school_id}-{safe_interaction_id}.json"
            )
            _atomic_json(rejected_path, rejected_record)
            raise VLMResponseError(
                f"{validation_error}; rejected auditor response preserved at "
                f"{rejected_path.relative_to(self.root)}"
            )
        record = dict(base_record)
        record["parsed_output"] = parsed
        if auditor_vocabulary_normalizations:
            record["auditor_vocabulary_normalizations"] = (
                auditor_vocabulary_normalizations
            )
        record["audited_uncertainty_assessment"] = _audit_summary(
            primary_record, parsed
        )
        auditor_safety_overrides = _auditor_safety_overrides(primary_record, parsed)
        if auditor_safety_overrides:
            record["auditor_safety_overrides"] = auditor_safety_overrides
        _atomic_json(output_path, record)
        return output_path


def reconcile_rejected_auditor_response(root: Path, school_id: str) -> Path:
    """Promote a normalized, schema-valid rejected auditor response offline."""
    root = root.resolve()
    config, schema, _prompt = load_auditor_bundle(root)
    request, primary_path = build_auditor_request(root, school_id)
    primary_record = _read_object(primary_path)
    rejected_directory = root / config["outputs"]["rejected_directory"]
    candidates = sorted(
        rejected_directory.glob(f"{school_id}-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise VLMConfigurationError(
            f"no rejected active-version auditor response exists for {school_id}"
        )
    source_path = candidates[0]
    record = _read_object(source_path)
    if record.get("configuration_id") != config["configuration_id"]:
        raise VLMConfigurationError("rejected auditor configuration does not match")
    if record.get("school_id") != school_id:
        raise VLMConfigurationError("rejected auditor school_id mismatch")
    if record.get("primary_record_sha256") != _sha256(primary_path):
        raise VLMConfigurationError("rejected auditor primary-record hash mismatch")

    parsed, normalizations = _normalize_auditor_output(record.get("parsed_output"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(parsed),
        key=lambda item: list(item.path),
    )
    if errors:
        raise VLMResponseError("only a schema-valid rejected auditor response can be reconciled")
    if parsed["school_id"] != school_id:
        raise VLMResponseError("rejected auditor parsed school_id mismatch")
    for field, audit in parsed["field_audits"].items():
        if (
            audit["status"] != "consistent"
            and audit["recommended_action"] == "accept_candidate"
        ):
            raise VLMResponseError(
                f"Gemini auditor cannot accept inconsistent field {field}"
            )
    output_path = root / config["outputs"]["directory"] / f"{school_id}.json"
    if output_path.exists():
        raise VLMConfigurationError(f"auditor response already exists: {output_path}")
    reconciled = dict(record)
    reconciled.pop("validation_error", None)
    reconciled["parsed_output"] = parsed
    reconciled["request_fingerprint_sha256"] = _request_fingerprint(request)
    reconciled["auditor_vocabulary_normalizations"] = [
        *record.get("auditor_vocabulary_normalizations", []),
        *normalizations,
    ]
    reconciled["audited_uncertainty_assessment"] = _audit_summary(
        primary_record, parsed
    )
    auditor_safety_overrides = _auditor_safety_overrides(primary_record, parsed)
    if auditor_safety_overrides:
        reconciled["auditor_safety_overrides"] = auditor_safety_overrides
    reconciled["reconciled_from_rejected"] = source_path.relative_to(root).as_posix()
    _atomic_json(output_path, reconciled)
    return output_path
