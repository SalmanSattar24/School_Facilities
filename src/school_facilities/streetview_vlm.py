from __future__ import annotations

import json
import tempfile
import copy
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator, SchemaError

from .credentials import CredentialError, load_api_key
from .streetview import (
    StreetViewConfigurationError,
    StreetViewLedger,
    StreetViewProviderError,
    _read_object,
    _sha256_file,
    delete_temporary_images,
    load_streetview_config,
    validate_street_response,
)
from .vlm import _gemini_response_schema, load_approved_school_input


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _response_value(response: Any, name: str) -> Any:
    if isinstance(response, dict):
        return response.get(name)
    return getattr(response, name, None)


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _serializable(value.model_dump())
    return str(value)


def _normalize_v1_11_vocabulary(
    parsed: Any,
) -> tuple[Any, list[dict[str, str]]]:
    """Normalize only frozen, meaning-preserving V1.11 enum spellings.

    The provider-visible schema intentionally omits enums to keep its grammar
    small. Raw provider text remains untouched in the audit record.
    """
    if not isinstance(parsed, dict):
        return parsed, []
    normalized = copy.deepcopy(parsed)
    changes: list[dict[str, str]] = []
    if normalized.get("schema_version") in {
        "1.11", "v1.11", "V1.11", "v1.11.0", "V1.11.0"
    }:
        original_version = normalized["schema_version"]
        normalized["schema_version"] = "1.11.0"
        changes.append(
            {"path": "schema_version", "from": original_version, "to": "1.11.0"}
        )
    fields = normalized.get("candidate_fields")
    fence = fields.get("dominant_fence_type") if isinstance(fields, dict) else None
    if isinstance(fence, dict):
        current = fence.get("value")
        replacement = {
            "chain_link": "chain-link",
            "wrought_iron": "wrought-iron",
            "metal_bar": "wrought-iron",
            "metal_rod": "wrought-iron",
            "metal-rod": "wrought-iron",
        }.get(current)
        if replacement is not None:
            fence["value"] = replacement
            changes.append(
                {
                    "path": "candidate_fields.dominant_fence_type.value",
                    "from": current,
                    "to": replacement,
                }
            )
    for field_name in ("running_track", "pool"):
        field = fields.get(field_name) if isinstance(fields, dict) else None
        if not isinstance(field, dict) or field.get("value") != "none":
            continue
        field["value"] = "no"
        changes.append(
            {
                "path": f"candidate_fields.{field_name}.value",
                "from": "none",
                "to": "no",
            }
        )
    for field_name in (
        "portable_classroom_count", "full_size_sports_fields", "hard_courts"
    ):
        field = fields.get(field_name) if isinstance(fields, dict) else None
        if not isinstance(field, dict):
            continue
        current = field.get("value")
        if current == "no":
            field["value"] = 0
            replacement = "0"
        elif current == "yes":
            # Presence does not determine a defensible count.
            field["value"] = "unknown"
            field["suggested_confidence"] = 0.2
            field["review_required"] = True
            replacement = "unknown"
        else:
            continue
        changes.append(
            {
                "path": f"candidate_fields.{field_name}.value",
                "from": current,
                "to": replacement,
            }
        )
    observations = normalized.get("image_observations")
    if isinstance(observations, list):
        for index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                continue
            for feature in ("fencing", "portable_classrooms", "athletics"):
                detail = observation.get(feature)
                if not isinstance(detail, dict) or detail.get("result") != "negative_visible_session":
                    continue
                detail["result"] = "negative_visible_segment"
                changes.append(
                    {
                        "path": f"image_observations.{index}.{feature}.result",
                        "from": "negative_visible_session",
                        "to": "negative_visible_segment",
                    }
                )
    return normalized, changes


def reconcile_rejected_v1_11(root: Path, *, school_id: str) -> Path:
    """Revalidate a preserved response after narrow offline normalization."""
    root = root.resolve()
    street_config = load_streetview_config(root)
    output_directories = [
        root / street_config["storage"]["derived_output_directory"],
        root / street_config["storage"]["validation_quarantine_directory"],
    ]
    output_dir = next(
        (directory for directory in output_directories if (directory / f"{school_id}-rejected.json").is_file()),
        output_directories[0],
    )
    rejected_path = output_dir / f"{school_id}-rejected.json"
    output_path = output_dir / f"{school_id}.json"
    if output_path.exists():
        raise StreetViewConfigurationError(f"V1.11 response already exists: {output_path}")
    if not rejected_path.is_file():
        raise StreetViewConfigurationError(
            f"preserved rejected V1.11 response is missing: {rejected_path}"
        )
    rejected = _read_object(rejected_path)
    parsed = rejected.get("parsed_output")
    expected_ids = set(rejected.get("street_image_ids") or [])
    if not isinstance(parsed, dict) or not expected_ids:
        raise StreetViewConfigurationError(
            "rejected response lacks parsed output or Street View image identities"
        )
    normalized, changes = _normalize_v1_11_vocabulary(parsed)
    if not changes:
        raise StreetViewConfigurationError(
            "rejected response has no approved meaning-preserving normalization"
        )
    guarded = validate_street_response(
        copy.deepcopy(normalized), school_id=school_id, expected_image_ids=expected_ids
    )
    guarded, comparison = _apply_v1_11_uncertainty_guards(
        root, guarded, school_id=school_id
    )
    reconciled = copy.deepcopy(rejected)
    reconciled["original_validation_error"] = reconciled.pop("validation_error", None)
    reconciled["normalization_changes"] = changes
    reconciled["normalized_output"] = normalized
    reconciled["guarded_output"] = guarded
    reconciled["uncertainty_comparison"] = comparison
    reconciled["guard_policy_version"] = "1.11.1"
    reconciled["offline_reconciled"] = True
    reconciled["source_rejected_record"] = str(rejected_path.relative_to(root)).replace("\\", "/")
    _atomic_json(output_path, reconciled)
    StreetViewLedger(output_dir / "gemini_request_ledger.jsonl").append(
        {
            "event": "gemini_response_reconciled_offline",
            "configuration_id": rejected.get("configuration_id"),
            "school_id": school_id,
            "source_rejected_record": str(rejected_path.relative_to(root)).replace("\\", "/"),
            "provider_request_made": False,
        }
    )
    return output_path


def _v1_10_raw_measurements(root: Path, school_id: str) -> dict[str, Any] | None:
    path = root / "data" / "model_outputs" / "final" / "v1.10" / f"{school_id}.json"
    if not path.is_file():
        return None
    record = _read_object(path)
    measurements = record.get("parsed_output", {}).get("measurements")
    if not isinstance(measurements, dict):
        return None
    return {
        name: value.get("value")
        for name, value in measurements.items()
        if isinstance(value, dict) and "value" in value
    }


def _apply_v1_11_uncertainty_guards(
    root: Path,
    response: Mapping[str, Any],
    *,
    school_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Add review-only cross-source and evidence-ambiguity flags."""
    guarded = copy.deepcopy(response)
    fields = guarded["candidate_fields"]
    reasons = set(guarded.get("review_reasons") or [])
    comparison: list[dict[str, Any]] = []
    baseline = _v1_10_raw_measurements(root, school_id)
    field_names = (
        "portable_classroom_count",
        "perimeter_fencing",
        "dominant_fence_type",
        "running_track",
        "full_size_sports_fields",
        "hard_courts",
        "pool",
    )
    if baseline is None:
        reasons.add("aerial_v1_10_raw_baseline_missing")
        for name in field_names:
            fields[name]["review_required"] = True
        comparison.append({"reason": "aerial_v1_10_raw_baseline_missing"})
    else:
        for name in field_names:
            street_value = fields[name]["value"]
            aerial_value = baseline.get(name)
            if aerial_value != street_value:
                reason = f"aerial_v1_10_raw_disagreement:{name}"
                reasons.add(reason)
                fields[name]["review_required"] = True
                comparison.append(
                    {
                        "field": name,
                        "aerial_v1_10_raw_value": aerial_value,
                        "street_v1_11_value": street_value,
                        "reason": reason,
                    }
                )
    hard_courts = fields["hard_courts"]
    hard_value = hard_courts.get("value")
    evidence = str(hard_courts.get("evidence") or "").casefold()
    ambiguous_terms = [
        term for term in ("small", "half", "play area", "partial", "single hoop", "one hoop")
        if term in evidence
    ]
    complete_terms = any(
        term in evidence
        for term in ("full court", "full-size court", "complete court", "independently playable")
    )
    if isinstance(hard_value, int) and hard_value > 0 and ambiguous_terms and not complete_terms:
        reason = "hard_court_evidence_may_describe_excluded_partial_play_area"
        reasons.add(reason)
        hard_courts["review_required"] = True
        comparison.append(
            {
                "field": "hard_courts",
                "street_v1_11_value": hard_value,
                "evidence_terms": ambiguous_terms,
                "reason": reason,
            }
        )
    if reasons:
        guarded["pipeline_review_required"] = True
    guarded["review_reasons"] = sorted(reasons)
    return guarded, comparison


def refresh_v1_11_guards(root: Path, *, school_id: str) -> Path:
    """Recompute derived guards on a preserved valid record without a request."""
    root = root.resolve()
    street_config = load_streetview_config(root)
    output_path = root / street_config["storage"]["derived_output_directory"] / f"{school_id}.json"
    if not output_path.is_file():
        raise StreetViewConfigurationError(f"V1.11 response is missing: {output_path}")
    record = _read_object(output_path)
    source = (
        record.get("normalized_output")
        or record.get("guarded_output")
        or record.get("parsed_output")
    )
    expected_ids = set(record.get("street_image_ids") or [])
    if not isinstance(source, dict) or not expected_ids:
        raise StreetViewConfigurationError("V1.11 record lacks response or image identities")
    validated = validate_street_response(
        copy.deepcopy(source), school_id=school_id, expected_image_ids=expected_ids
    )
    guarded, comparison = _apply_v1_11_uncertainty_guards(
        root, validated, school_id=school_id
    )
    record["guarded_output"] = guarded
    record["uncertainty_comparison"] = comparison
    record["guard_policy_version"] = "1.11.1"
    record["guards_refreshed_offline"] = True
    _atomic_json(output_path, record)
    return output_path


def load_streetview_vlm_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    config = _read_object(root / "config" / "streetview_vlm_v1_11.json")
    if config.get("configuration_id") != "school-facilities-streetview-vlm-v1.11":
        raise StreetViewConfigurationError("unexpected V1.11 VLM configuration ID")
    if config.get("model") != "gemini-3.5-flash-lite":
        raise StreetViewConfigurationError("V1.11 must use frozen Gemini 3.5 Flash Lite")
    if config.get("request_limits", {}).get("maximum_retries") != 0:
        raise StreetViewConfigurationError("V1.11 VLM retries must remain disabled")
    schema = _read_object(root / config["generation"]["response_schema_path"])
    prompt = (root / config["generation"]["prompt_path"]).read_text(encoding="utf-8")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise StreetViewConfigurationError(
            f"V1.11 response JSON Schema is invalid: {error.message}"
        ) from error
    required_phrases = (
        "observable facts, not private chain-of-thought",
        "Never estimate rooftop-solar",
        "negative applies only",
        "full fencing requires",
        "extra images do not automatically increase confidence",
    )
    for phrase in required_phrases:
        if phrase not in prompt:
            raise StreetViewConfigurationError(
                f"V1.11 prompt is missing required safeguard: {phrase!r}"
            )
    return config, schema, prompt


def _manifest_school(manifest: Mapping[str, Any], school_id: str) -> list[dict[str, Any]]:
    images = [item for item in manifest.get("images", []) if item.get("school_id") == school_id]
    if not images:
        raise StreetViewConfigurationError(f"fetched manifest has no images for {school_id}")
    return images


def build_v1_11_request(
    root: Path, fetched_manifest_path: Path, school_id: str
) -> tuple[dict[str, Any], set[str]]:
    root = root.resolve()
    street_config = load_streetview_config(root)
    config, schema, prompt = load_streetview_vlm_bundle(root)
    manifest = _read_object(fetched_manifest_path)
    if manifest.get("kind") != "streetview_fetched_manifest":
        raise StreetViewConfigurationError("V1.11 inference requires a fetched manifest")
    if manifest.get("configuration_id") != street_config["configuration_id"]:
        raise StreetViewConfigurationError("fetched manifest configuration mismatch")
    street_images = _manifest_school(manifest, school_id)
    if len(street_images) > int(config["inputs"]["maximum_street_images"]):
        raise StreetViewConfigurationError("too many Street View images for one V1.11 request")
    school = load_approved_school_input(root, school_id)
    image_inputs = []
    aerial_metadata = []
    for image in (school.context, school.detail):
        image_inputs.append(
            {
                "type": "image",
                "data": image.path.resolve(),
                "mime_type": "image/jpeg",
                "resolution": config["inputs"]["aerial_image_resolution"],
            }
        )
        aerial_metadata.append(
            {
                "role": image.role,
                "source": image.source,
                "capture_vintage": image.capture_vintage,
                "sha256": _sha256_file(image.path),
            }
        )
    street_metadata = []
    expected_ids = set()
    temporary_root = (root / street_config["storage"]["temporary_directory"]).resolve()
    for item in street_images:
        image_id = str(item["image_id"])
        if image_id in expected_ids:
            raise StreetViewConfigurationError("duplicate image ID in fetched manifest")
        expected_ids.add(image_id)
        path = (root / item["temporary_image_path"]).resolve()
        try:
            path.relative_to(temporary_root)
        except ValueError as error:
            raise StreetViewConfigurationError("Street View image lies outside temporary storage") from error
        if not path.is_file() or _sha256_file(path) != item["image_sha256"]:
            raise StreetViewConfigurationError(f"Street View image is missing or changed: {image_id}")
        image_inputs.append(
            {
                "type": "image",
                "data": path,
                "mime_type": "image/jpeg",
                "resolution": config["inputs"]["street_image_resolution"],
            }
        )
        street_metadata.append(
            {
                "image_id": image_id,
                "panorama_id": item["panorama_id"],
                "capture_vintage": item["capture_vintage"],
                "retrieved_at_utc": item["retrieved_at_utc"],
                "location": item["panorama_location"],
                "heading": item["heading"],
                "pitch": item["pitch"],
                "field_of_view": item["field_of_view"],
                "attribution": item["copyright"],
                "sha256": item["image_sha256"],
            }
        )
    metadata = {
        "school_id": school.school_id,
        "school_name": school.school_name,
        "campus_resolution_notes": school.campus_resolution_notes,
        "campus_scope_mode": school.campus_scope_mode,
        "aerial_images": aerial_metadata,
        "street_images_in_input_order": street_metadata,
        "street_negative_scope": "visible_segment_only",
        "solar_fields_are_out_of_scope": True,
    }
    generation = config["generation"]
    request = {
        "model": config["model"],
        "input": [
            {
                "type": "text",
                "text": "Assess supplemental street-level evidence. Metadata:\n"
                + json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            },
            *image_inputs,
        ],
        "system_instruction": prompt,
        "generation_config": {
            "thinking_level": generation["thinking_level"],
            "max_output_tokens": generation["max_output_tokens"],
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": _gemini_response_schema(schema),
        },
        "service_tier": "standard",
        "store": False,
        "stream": False,
    }
    return request, expected_ids


def assess_v1_11(
    root: Path,
    *,
    fetched_manifest_path: Path,
    school_id: str,
    live: bool,
    confirmed: bool,
    create_interaction: Callable[..., Any] | None = None,
    request_ledger_name: str = "gemini_request_ledger.jsonl",
) -> dict[str, Any] | Path:
    root = root.resolve()
    config, _, _ = load_streetview_vlm_bundle(root)
    request, expected_ids = build_v1_11_request(root, fetched_manifest_path, school_id)
    if not live:
        return {
            "configuration_id": config["configuration_id"],
            "model": config["model"],
            "school_id": school_id,
            "aerial_image_count": 2,
            "street_image_count": len(expected_ids),
            "street_image_ids": sorted(expected_ids),
            "gemini_request_will_be_made": False,
            "street_images_will_be_deleted": False,
        }
    if not confirmed:
        raise StreetViewConfigurationError("live V1.11 inference requires explicit Gemini confirmation")
    street_config = load_streetview_config(root)
    pilot = _read_object(root / "config" / "pilot_schools.json")
    validation_ids = set(pilot.get("excluded_validation_school_ids", []))
    if school_id in validation_ids:
        output_dir = root / street_config["storage"]["validation_quarantine_directory"]
        quarantined = True
    else:
        output_dir = root / street_config["storage"]["derived_output_directory"]
        quarantined = False
    output_path = output_dir / f"{school_id}.json"
    if output_path.exists():
        raise StreetViewConfigurationError(f"V1.11 response already exists: {output_path}")
    ledger = StreetViewLedger(output_dir / request_ledger_name)
    if any(
        row.get("event") == "gemini_request_reserved" and row.get("school_id") == school_id
        for row in ledger.records()
    ):
        raise StreetViewConfigurationError("V1.11 permits only one Gemini request per school")
    ledger.append(
        {
            "event": "gemini_request_reserved",
            "configuration_id": config["configuration_id"],
            "school_id": school_id,
            "model": config["model"],
            "retry_permitted": False,
        }
    )
    close_client = None
    if create_interaction is None:
        credentials = config["credentials"]
        try:
            api_key, _ = load_api_key(
                environment_variable=credentials["api_key_environment_variable"],
                secrets_file=root / credentials["local_secrets_file"],
            )
        except CredentialError as error:
            raise StreetViewConfigurationError(str(error)) from error
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version=config["api_version"],
                timeout=int(config["request_limits"]["request_timeout_seconds"]) * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        create_interaction = client.interactions.create
        close_client = client.close
    response = None
    try:
        response = create_interaction(
            **request,
            timeout=float(config["request_limits"]["request_timeout_seconds"]),
        )
        status = _response_value(response, "status")
        output_text = _response_value(response, "output_text")
        record: dict[str, Any] = {
            "schema_version": "1.11.0",
            "configuration_id": config["configuration_id"],
            "school_id": school_id,
            "model": config["model"],
            "interaction_id": _response_value(response, "id"),
            "status": status,
            "created": _response_value(response, "created"),
            "usage": _serializable(_response_value(response, "usage")),
            "blind_validation_quarantined": quarantined,
            "fetched_manifest_sha256": _sha256_file(fetched_manifest_path),
            "street_image_ids": sorted(expected_ids),
            "output_text": output_text if isinstance(output_text, str) else None,
            "parsed_output": None,
        }
        if status != "completed" or not isinstance(output_text, str):
            record["validation_error"] = "Gemini response was not completed with output text"
            _atomic_json(output_path.with_name(f"{school_id}-rejected.json"), record)
            raise StreetViewProviderError(record["validation_error"])
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as error:
            record["validation_error"] = f"Gemini output is not valid JSON: {error}"
            _atomic_json(output_path.with_name(f"{school_id}-rejected.json"), record)
            raise StreetViewProviderError(record["validation_error"]) from error
        normalized, normalization_changes = _normalize_v1_11_vocabulary(parsed)
        record["parsed_output"] = parsed
        record["normalization_changes"] = normalization_changes
        record["normalized_output"] = normalized
        try:
            guarded = validate_street_response(
                copy.deepcopy(normalized),
                school_id=school_id,
                expected_image_ids=expected_ids,
            )
        except StreetViewProviderError as error:
            record["normalized_output"] = normalized
            record["validation_error"] = str(error)
            _atomic_json(output_path.with_name(f"{school_id}-rejected.json"), record)
            ledger.append(
                {
                    "event": "gemini_request_completed",
                    "configuration_id": config["configuration_id"],
                    "school_id": school_id,
                    "interaction_id": _response_value(response, "id"),
                    "schema_valid": False,
                }
            )
            raise
        guarded, comparison = _apply_v1_11_uncertainty_guards(
            root, guarded, school_id=school_id
        )
        record["guarded_output"] = guarded
        record["uncertainty_comparison"] = comparison
        record["guard_policy_version"] = "1.11.1"
        _atomic_json(output_path, record)
        ledger.append(
            {
                "event": "gemini_request_completed",
                "configuration_id": config["configuration_id"],
                "school_id": school_id,
                "interaction_id": _response_value(response, "id"),
                "schema_valid": True,
            }
        )
        return output_path
    finally:
        if close_client is not None:
            close_client()
        # Retention is an explicit configuration decision. Preserved source
        # images remain local and Git-ignored so a human can review the exact
        # evidence without authorizing an implicit provider retry.
        if response is not None and config["storage"]["delete_images_after_inference"]:
            delete_temporary_images(root, fetched_manifest_path, school_id=school_id)


def assess_v1_11_batch(
    root: Path,
    *,
    fetched_manifest_path: Path,
    live: bool,
    confirmed: bool,
) -> dict[str, Any]:
    """Run or preview a resumable, one-request-per-school V1.11 batch."""
    root = root.resolve()
    manifest = _read_object(fetched_manifest_path)
    if manifest.get("kind") != "streetview_fetched_manifest":
        raise StreetViewConfigurationError("not a fetched Street View manifest")
    school_ids = sorted(
        {
            str(item["school_id"])
            for item in manifest.get("images", [])
            if item.get("school_id")
        }
    )
    if not live:
        return {
            "mode": "dry-run",
            "school_count": len(school_ids),
            "school_ids": school_ids,
            "gemini_requests_will_be_made": 0,
        }
    if not confirmed:
        raise StreetViewConfigurationError("live V1.11 batch requires explicit Gemini confirmation")

    street_config = load_streetview_config(root)
    output_directories = [
        root / street_config["storage"]["derived_output_directory"],
        root / street_config["storage"]["validation_quarantine_directory"],
    ]
    results: list[dict[str, Any]] = []
    previous_started = 0.0
    minimum_interval = float(
        _read_object(root / "config" / "streetview_vlm_v1_11.json")["request_limits"][
            "minimum_seconds_between_requests"
        ]
    )
    for index, school_id in enumerate(school_ids, start=1):
        existing = next(
            (directory / f"{school_id}.json" for directory in output_directories if (directory / f"{school_id}.json").exists()),
            None,
        )
        if existing is not None:
            results.append({"school_id": school_id, "status": "reused", "output": str(existing)})
            continue
        wait_seconds = minimum_interval - (time.monotonic() - previous_started)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        previous_started = time.monotonic()
        try:
            output = assess_v1_11(
                root,
                fetched_manifest_path=fetched_manifest_path,
                school_id=school_id,
                live=True,
                confirmed=True,
                request_ledger_name="gemini_request_ledger_production.jsonl",
            )
            results.append({"school_id": school_id, "status": "completed", "output": str(output)})
        except Exception as error:
            results.append({"school_id": school_id, "status": "failed", "error": str(error)})
        print(f"V1.11 batch {index}/{len(school_ids)}: {school_id} {results[-1]['status']}", flush=True)
    return {
        "mode": "live",
        "school_count": len(school_ids),
        "completed_n": sum(row["status"] == "completed" for row in results),
        "reused_n": sum(row["status"] == "reused" for row in results),
        "failed_n": sum(row["status"] == "failed" for row in results),
        "results": results,
    }
