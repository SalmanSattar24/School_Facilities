from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

from jsonschema import Draft202012Validator, SchemaError

from .schema import MEASUREMENT_FIELDS, ValidationResult, read_csv


FROZEN_FILES = (
    "imagery.json",
    "vlm.json",
    "vlm_response_schema.json",
    "vlm_prompt.txt",
    "vlm_field_protocol.json",
    "vlm_auditor.json",
    "vlm_auditor_prompt.txt",
    "vlm_auditor_response_schema.json",
    "boundary_vlm.json",
    "boundary_vlm_prompt.txt",
    "boundary_vlm_response_schema.json",
    "pilot_schools.json",
)

EXPECTED_PACKAGES = {
    "google-genai": "2.20.0",
    "jsonschema": "4.26.0",
    "numpy": "2.5.2",
    "planetary-computer": "1.0.0",
    "pystac-client": "0.9.0",
    "rasterio": "1.5.1",
}


def _load_json(path: Path, errors: list[str]) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing configuration file: {path}")
        return {}
    except json.JSONDecodeError as error:
        errors.append(f"invalid JSON in {path}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"configuration root must be an object: {path}")
        return {}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(config_dir: Path, errors: list[str]) -> None:
    manifest_path = config_dir / "frozen_config.sha256"
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        errors.append(f"missing configuration freeze manifest: {manifest_path}")
        return
    expected: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"invalid manifest line {number}: {line!r}")
            continue
        digest, name = parts
        expected[name.lstrip("*")] = digest.lower()
    if set(expected) != set(FROZEN_FILES):
        errors.append(
            f"freeze manifest must list exactly the {len(FROZEN_FILES)} frozen configuration files"
        )
        return
    for name, digest in expected.items():
        path = config_dir / name
        if not path.is_file():
            errors.append(f"manifest target is missing: {path}")
        elif _sha256(path) != digest:
            errors.append(f"frozen configuration changed without a versioned re-freeze: {path}")


def _validate_packages(errors: list[str]) -> None:
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"pilot dependency is not installed: {package}=={expected}")
            continue
        if actual != expected:
            errors.append(f"pilot dependency mismatch: {package} is {actual}, expected {expected}")


def validate_configuration(root: Path) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    config_dir = root / "config"
    imagery = _load_json(config_dir / "imagery.json", errors)
    vlm = _load_json(config_dir / "vlm.json", errors)
    response_schema = _load_json(config_dir / "vlm_response_schema.json", errors)
    field_protocol = _load_json(config_dir / "vlm_field_protocol.json", errors)
    auditor = _load_json(config_dir / "vlm_auditor.json", errors)
    auditor_schema = _load_json(
        config_dir / "vlm_auditor_response_schema.json", errors
    )
    boundary_vlm = _load_json(config_dir / "boundary_vlm.json", errors)
    boundary_schema = _load_json(
        config_dir / "boundary_vlm_response_schema.json", errors
    )
    pilot = _load_json(config_dir / "pilot_schools.json", errors)

    primary = imagery.get("primary", {})
    if not isinstance(primary, dict) or primary.get("collection") != "naip":
        errors.append("imagery primary collection must be NAIP")
    if imagery.get("schema_version") != "1.6":
        errors.append("imagery configuration schema version must be 1.6")
    if imagery.get("configuration_id") != "school-facilities-imagery-pilot-v1.6":
        errors.append("imagery configuration ID must be the frozen Version 1.6 ID")
    if imagery.get("compatible_configuration_ids") != [
        "school-facilities-imagery-pilot-v1.5"
    ]:
        errors.append("imagery Version 1.6 must retain read compatibility with Version 1.5")
    if isinstance(primary, dict):
        if (
            primary.get("data_api_bbox_endpoint")
            != "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox"
        ):
            errors.append("NAIP data API bbox endpoint must remain frozen")
        selection = primary.get("selection", {})
        if (
            not isinstance(selection, dict)
            or selection.get("minimum_target_coverage_fraction") != 0.995
        ):
            errors.append("NAIP same-date target coverage threshold must remain 0.995")
    street_level = imagery.get("street_level", {})
    if not isinstance(street_level, dict) or street_level.get("enabled_for_pilot") is not False:
        errors.append("street-level imagery must remain disabled for the pilot")
    manual = imagery.get("manual_adjudication", {})
    if not isinstance(manual, dict):
        errors.append("manual imagery-adjudication policy is missing")
    else:
        if manual.get("provider") != "Google Earth Pro":
            errors.append("manual adjudication provider must remain Google Earth Pro")
        for field in ("automated_extraction", "bulk_download", "may_be_sent_to_vlm"):
            if manual.get(field) is not False:
                errors.append(f"manual imagery policy requires {field}=false")
    campus_resolution = imagery.get("campus_resolution", {})
    if not isinstance(campus_resolution, dict):
        errors.append("automatic campus-resolution policy is missing")
    else:
        if campus_resolution.get("human_review_only_when_flagged") is not True:
            errors.append("campus resolution must use exception-based human review")
        thresholds = campus_resolution.get("automatic_confirmed_thresholds", {})
        if not isinstance(thresholds, dict) or thresholds.get("minimum_name_similarity") != 0.8:
            errors.append("automatic campus name-similarity threshold must remain 0.8")
        if campus_resolution.get("gemini_boundary_proposals_are_never_hard_masks") is not True:
            errors.append("Gemini boundary proposals must never become hard measurement masks")
        if campus_resolution.get("soft_scope_requires_ownership_and_fencing_review") is not True:
            errors.append("soft scope must retain ownership and fencing review safeguards")

    products = imagery.get("products", {})
    detail_product = products.get("detail", {}) if isinstance(products, dict) else {}
    expected_adaptive_detail = {
        "adaptive_from_campus_polygon": True,
        "minimum_extent_m": 250,
        "maximum_extent_m": 1200,
        "buffer_each_side_m": 60,
        "soft_boundary_minimum_extent_m": 600,
        "soft_boundary_buffer_each_side_m": 150,
        "rounding_increment_m": 50,
        "fallback_extent_without_polygon_m": 800,
        "flag_if_maximum_clips_buffered_campus": True,
    }
    if not isinstance(detail_product, dict):
        errors.append("adaptive detail-product configuration is missing")
    else:
        for field, expected in expected_adaptive_detail.items():
            if detail_product.get(field) != expected:
                errors.append(f"adaptive detail configuration requires {field}={expected!r}")

    if vlm.get("schema_version") != "1.10":
        errors.append("VLM configuration schema version must be 1.10")
    if vlm.get("configuration_id") != "school-facilities-vlm-final-v1.10":
        errors.append("VLM configuration ID must be the frozen final Version 1.10 ID")
    if vlm.get("model") != "gemini-3.5-flash-lite":
        errors.append("VLM model must be the frozen gemini-3.5-flash-lite ID")
    client = vlm.get("client", {})
    if not isinstance(client, dict) or client.get("version") != EXPECTED_PACKAGES["google-genai"]:
        errors.append("VLM client version does not match the frozen dependency")
    elif client.get("method") != "interactions.create":
        errors.append("VLM client must use the frozen interactions.create method")
    credentials = vlm.get("credentials", {})
    if not isinstance(credentials, dict):
        errors.append("VLM credentials configuration is missing")
    else:
        if credentials.get("api_key_environment_variable") != "GEMINI_API_KEY":
            errors.append("VLM environment-key override must remain GEMINI_API_KEY")
        if credentials.get("lookup_order") != ["environment", "local_secrets_file"]:
            errors.append("VLM credential lookup must prefer environment then local secrets file")
        if credentials.get("local_secrets_file") != "secrets.local.env":
            errors.append("VLM local secrets filename changed")
        if credentials.get("require_project_without_active_billing") is not True:
            errors.append("Version 1.8 must require a project without active billing")
    usage = vlm.get("usage_mode", {})
    if not isinstance(usage, dict):
        errors.append("VLM usage-mode configuration is missing")
    else:
        for field in ("allow_paid_overage", "allow_model_fallback", "allow_latest_alias"):
            if usage.get(field) is not False:
                errors.append(f"{field} must remain false")
    generation = vlm.get("generation", {})
    if not isinstance(generation, dict):
        errors.append("VLM generation configuration is missing")
    else:
        if generation.get("thinking_level") != "minimal":
            errors.append("VLM thinking level must remain minimal")
        if generation.get("max_output_tokens") != 8192:
            errors.append("VLM maximum output tokens must remain 8192")
        unsupported = {
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "thinking_budget",
        }
        present = unsupported & set(generation)
        if present:
            errors.append(
                "Version 1.8 generation config contains unsupported legacy settings: "
                + ", ".join(sorted(present))
            )
        response_format = generation.get("response_format", {})
        if not isinstance(response_format, dict):
            errors.append("VLM structured response format is missing")
        else:
            expected_response_format = {
                "type": "text",
                "mime_type": "application/json",
                "schema_path": "config/vlm_response_schema.json",
                "provider_projection": "required_property_tree_basic_types",
                "strict_validation_location": "local_after_response",
            }
            if response_format != expected_response_format:
                errors.append("VLM structured response format does not match Version 1.8.3")
    inputs = vlm.get("inputs", {})
    if not isinstance(inputs, dict):
        errors.append("VLM input configuration is missing")
    else:
        if inputs.get("images_per_school") != 2:
            errors.append("VLM requests must contain exactly two images per school")
        if inputs.get("per_image_media_resolution") != "high":
            errors.append("Gemini media resolution must remain high for each image")
        if inputs.get("include_resolved_campus_boundary_when_authoritative") is not True:
            errors.append("VLM must include authoritative campus boundaries")
        if inputs.get("include_non_binding_boundary_guidance_when_available") is not True:
            errors.append("VLM must include available non-binding boundary guidance")
        if inputs.get("scope_modes") != [
            "authoritative_polygon",
            "soft_boundary",
            "center_only",
        ]:
            errors.append("VLM scope-mode contract changed")
        if inputs.get("soft_and_center_only_measurement_search_scope") != "entire_detail_image":
            errors.append("soft and center-only cases must search the entire detail image")
        if inputs.get("campus_boundary_coordinate_system") != (
            "detail-image normalized coordinates with top-left (0,0) and bottom-right (1,1)"
        ):
            errors.append("VLM campus-boundary coordinate system changed")
    limits = vlm.get("request_limits", {})
    if not isinstance(limits, dict):
        errors.append("VLM request limits are missing")
    else:
        if limits.get("requests_per_minute") != 15:
            errors.append("pilot rate must remain capped at 15 requests per minute")
        if limits.get("tokens_per_minute") != 250000:
            errors.append("recorded Gemini project quota must remain 250000 tokens per minute")
        if limits.get("requests_per_day") != 500:
            errors.append("daily request cap must remain at the observed 500 requests per day")
        if limits.get("request_timeout_seconds") != 300:
            errors.append("request timeout must remain 300 seconds")
        if limits.get("pilot_hard_request_cap_including_retries") != 6:
            errors.append("pilot hard request cap must remain 6")
        if limits.get("production_hard_request_cap_including_retries") != 30:
            errors.append("production hard request cap must remain 30")
    outputs = vlm.get("outputs", {})
    if not isinstance(outputs, dict):
        errors.append("VLM output configuration is missing")
    else:
        if outputs.get("raw_directory") != "data/model_outputs/final/v1.10":
            errors.append("VLM raw-response directory must remain version-isolated")
        if outputs.get("rejected_directory") != "data/model_outputs/rejected/v1.10":
            errors.append("VLM rejected-response directory must remain version-isolated")
        suggested = outputs.get("model_suggested_confidence", {})
        if (
            not isinstance(suggested, dict)
            or suggested.get("allowed_scores") != [0.8, 0.6, 0.4, 0.2]
        ):
            errors.append("VLM suggested confidence must exclude 0.95")
    uncertainty_policy = vlm.get("uncertainty_policy", {})
    if not isinstance(uncertainty_policy, dict):
        errors.append("VLM uncertainty policy is missing")
    else:
        if uncertainty_policy.get("always_review_fields") != [
            "perimeter_fencing",
            "dominant_fence_type",
            "hard_courts",
        ]:
            errors.append("VLM always-review field policy changed")
        if uncertainty_policy.get("review_when_nonzero_fields") != [
            "portable_classroom_count",
            "full_size_sports_fields",
        ]:
            errors.append("VLM nonzero-count review policy changed")
        if uncertainty_policy.get("semantic_conflict_action") != (
            "preserve_raw_and_force_guarded_unknown"
        ):
            errors.append("VLM semantic-conflict action changed")
        if uncertainty_policy.get("hard_conflict_action") != (
            "preserve_raw_and_force_guarded_unknown"
        ):
            errors.append("VLM hard-conflict action changed")
        if uncertainty_policy.get("soft_risk_action") != (
            "preserve_raw_and_require_review"
        ):
            errors.append("VLM soft-risk action changed")
    evidence_policy = vlm.get("evidence_policy", {})
    if not isinstance(evidence_policy, dict):
        errors.append("VLM structured-evidence policy is missing")
    else:
        tolerance = evidence_policy.get("solar_area_consistency", {})
        if tolerance != {
            "absolute_tolerance_m2": 25,
            "relative_tolerance_fraction": 0.25,
        }:
            errors.append("solar-area consistency tolerance must be max(25 m2, 25%)")
        if evidence_policy.get("auditor_may_overwrite_primary_value") is not False:
            errors.append("the auditor must never overwrite a primary measurement")
        if evidence_policy.get("field_protocol_path") != "config/vlm_field_protocol.json":
            errors.append("VLM field-protocol path changed")
    expected_features = {
        "solar",
        "portable_classrooms",
        "fencing",
        "running_track",
        "sports_fields",
        "hard_courts",
        "pool",
    }
    protocol_features = field_protocol.get("features", {})
    if field_protocol.get("protocol_id") != "school-facilities-observable-feature-protocol-v1.9":
        errors.append("observable-feature protocol ID changed")
    if not isinstance(protocol_features, dict) or set(protocol_features) != expected_features:
        errors.append("observable-feature protocol must contain the exact seven features")
    elif any(
        not isinstance(specification, dict)
        or not specification.get("definition")
        or not specification.get("measurement_fields")
        or not specification.get("questions")
        for specification in protocol_features.values()
    ):
        errors.append("every observable-feature protocol entry requires definitions and questions")
    data_terms = vlm.get("data_terms", {})
    if not isinstance(data_terms, dict):
        errors.append("VLM data-terms configuration is missing")
    else:
        for field in (
            "sensitive_or_confidential_data_prohibited",
            "personal_data_prohibited",
            "operational_metadata_is_outside_project_content_allowlist",
        ):
            if data_terms.get(field) is not True:
                errors.append(f"{field} must remain true")

    properties = response_schema.get("properties", {})
    measurement_schema = properties.get("measurements", {}) if isinstance(properties, dict) else {}
    schema_properties = (
        measurement_schema.get("properties", {}) if isinstance(measurement_schema, dict) else {}
    )
    if set(schema_properties) != set(MEASUREMENT_FIELDS):
        errors.append("VLM response schema must contain exactly the nine measurement fields")
    schema_version = properties.get("schema_version", {}) if isinstance(properties, dict) else {}
    if not isinstance(schema_version, dict) or schema_version.get("enum") != ["1.10.0"]:
        errors.append("VLM response schema version must be 1.10.0")
    solar_inventory = properties.get("solar_inventory", {}) if isinstance(properties, dict) else {}
    if not isinstance(solar_inventory, dict):
        errors.append("VLM response schema must require a structured solar inventory")
    elif "solar_inventory" not in response_schema.get("required", []):
        errors.append("VLM response schema must require the structured solar inventory")
    definitions = response_schema.get("$defs", {})
    confidence_definition = (
        definitions.get("confidenceScore", {}) if isinstance(definitions, dict) else {}
    )
    if (
        not isinstance(confidence_definition, dict)
        or confidence_definition.get("enum") != [0.8, 0.6, 0.4, 0.2]
    ):
        errors.append(
            "VLM diagnostic confidence must exclude 0.95 and use exactly 0.80, 0.60, 0.40, and 0.20"
        )
    solar_candidate = (
        definitions.get("solarCandidate", {}) if isinstance(definitions, dict) else {}
    )
    solar_candidate_required = (
        set(solar_candidate.get("required", []))
        if isinstance(solar_candidate, dict)
        else set()
    )
    if not {
        "candidate_id",
        "image_role",
        "bbox_normalized",
        "footprint_polygon_normalized",
        "mount_location",
        "support_structure",
        "support_surface_color",
        "support_surface_form",
        "surrounding_cues",
        "mount_evidence",
    } <= solar_candidate_required:
        errors.append("VLM solar candidates must require location, footprint, mount, and evidence")
    evidence_packets = properties.get("evidence_packets", {}) if isinstance(properties, dict) else {}
    if not isinstance(evidence_packets, dict) or evidence_packets.get("minItems") != 9 or evidence_packets.get("maxItems") != 9:
        errors.append("VLM response schema must require exactly nine evidence packets")
    if "fencing_inventory" not in response_schema.get("required", []):
        errors.append("VLM response schema must require a fencing inventory")
    feature_assessments = properties.get("feature_assessments", {}) if isinstance(properties, dict) else {}
    if (
        not isinstance(feature_assessments, dict)
        or feature_assessments.get("minItems") != 7
        or feature_assessments.get("maxItems") != 7
        or "feature_assessments" not in response_schema.get("required", [])
    ):
        errors.append("VLM response schema must require exactly seven feature assessments")
    solar_candidate_properties = (
        solar_candidate.get("properties", {}) if isinstance(solar_candidate, dict) else {}
    )
    mount_schema = (
        solar_candidate_properties.get("mount_location", {})
        if isinstance(solar_candidate_properties, dict)
        else {}
    )
    if not isinstance(mount_schema, dict) or mount_schema.get("enum") != [
        "school_building_roof",
        "portable_classroom_roof",
        "parking_carport_canopy",
        "ground_mounted",
        "uncertain",
    ]:
        errors.append("VLM solar mount classes do not match the frozen eligibility policy")
    for field, field_schema in schema_properties.items():
        reference = field_schema.get("$ref") if isinstance(field_schema, dict) else None
        definition_name = reference.rsplit("/", 1)[-1] if isinstance(reference, str) else ""
        definition = definitions.get(definition_name, {}) if isinstance(definitions, dict) else {}
        required = definition.get("required", []) if isinstance(definition, dict) else []
        if "suggested_confidence" not in required or "confidence_reason" not in required:
            errors.append(f"VLM response schema field {field} must request diagnostic confidence")
    try:
        Draft202012Validator.check_schema(response_schema)
    except SchemaError as error:
        errors.append(f"VLM response JSON Schema is invalid: {error.message}")
    prompt_path = config_dir / "vlm_prompt.txt"
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"missing frozen prompt: {prompt_path}")
    else:
        for phrase in (
            "Return unknown rather than guessing",
            "suggested_confidence",
            "never the final project confidence",
            "Never return 0.95",
            "footprint_polygon_normalized",
            "authoritative spatial scope",
            "Never add excluded candidate area",
            "solar_present no",
            "observable facts, not private chain-of-thought",
            "count_components",
            "fencing_inventory",
            "AUTHORITATIVE FIELD PROTOCOL",
            "question_answers",
            "minimum_barrier_coverage_fraction",
            "maximum_barrier_coverage_fraction",
            "A fence shadow is supporting evidence only",
            "isolated half basketball court",
            "soft_boundary",
            "Search the entire detail image",
        ):
            if phrase not in prompt:
                errors.append(f"frozen prompt is missing required instruction: {phrase!r}")

    if auditor.get("configuration_id") != "school-facilities-evidence-auditor-v1.2":
        errors.append("evidence-auditor configuration ID changed")
    if auditor.get("model") != "gemini-3.1-flash-lite":
        errors.append("evidence auditor must use the frozen gemini-3.1-flash-lite ID")
    auditor_inputs = auditor.get("inputs", {})
    if not isinstance(auditor_inputs, dict) or auditor_inputs.get(
        "primary_configuration_id"
    ) != "school-facilities-vlm-final-v1.10":
        errors.append("evidence auditor must consume only frozen VLM 1.10 records")
    elif auditor_inputs.get("include_authoritative_field_protocol") is not True:
        errors.append("evidence auditor must receive the authoritative field protocol")
    elif any(
        auditor_inputs.get(field) is not False
        for field in (
            "include_images",
            "include_raw_output_text",
            "include_human_reviewed_values",
            "include_blind_reference_values",
        )
    ):
        errors.append("evidence auditor input isolation changed")
    auditor_outputs = auditor.get("outputs", {})
    if not isinstance(auditor_outputs, dict) or auditor_outputs.get(
        "auditor_may_overwrite_primary_value"
    ) is not False:
        errors.append("evidence auditor must not overwrite primary values")
    try:
        Draft202012Validator.check_schema(auditor_schema)
    except SchemaError as error:
        errors.append(f"VLM auditor response JSON Schema is invalid: {error.message}")

    if boundary_vlm.get("configuration_id") != "school-facilities-boundary-resolver-v1.0":
        errors.append("boundary-resolver configuration ID changed")
    if boundary_vlm.get("model") != "gemini-3.5-flash-lite":
        errors.append("boundary resolver must use gemini-3.5-flash-lite")
    boundary_inputs = boundary_vlm.get("inputs", {})
    if not isinstance(boundary_inputs, dict) or boundary_inputs.get("images_per_school") != 1:
        errors.append("boundary resolver must use exactly one context image")
    elif boundary_inputs.get("image_role") != "context":
        errors.append("boundary resolver input must be the dated context image")
    boundary_decision = boundary_vlm.get("decision_policy", {})
    if not isinstance(boundary_decision, dict) or boundary_decision.get(
        "auto_confirmation_enabled"
    ) is not False:
        errors.append("unvalidated boundary resolver V1.0 must not auto-confirm polygons")
    boundary_outputs = boundary_vlm.get("outputs", {})
    if not isinstance(boundary_outputs, dict) or boundary_outputs.get(
        "never_overwrite_campus_resolution_automatically"
    ) is not True:
        errors.append("boundary resolver must not overwrite campus resolution automatically")
    try:
        Draft202012Validator.check_schema(boundary_schema)
    except SchemaError as error:
        errors.append(f"boundary-resolver response JSON Schema is invalid: {error.message}")
    boundary_prompt_path = config_dir / "boundary_vlm_prompt.txt"
    try:
        boundary_prompt = boundary_prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"missing boundary-resolver prompt: {boundary_prompt_path}")
    else:
        for phrase in (
            "not a school-facility measurement task",
            "Do not report rooftop solar",
            "shared or ownership-ambiguous",
            "The polygon must be simple, closed",
            "Do not provide hidden chain-of-thought",
        ):
            if phrase not in boundary_prompt:
                errors.append(
                    f"boundary-resolver prompt is missing required instruction: {phrase!r}"
                )
    pilot_schools = pilot.get("schools", [])
    if not isinstance(pilot_schools, list) or len(pilot_schools) != 3:
        errors.append("pilot configuration must contain exactly three schools")
        pilot_schools = []
    pilot_ids = {
        row.get("school_id") for row in pilot_schools if isinstance(row, dict) and row.get("school_id")
    }
    validation_ids = set(pilot.get("excluded_validation_school_ids", []))
    if pilot_ids & validation_ids:
        errors.append("pilot schools overlap the frozen validation set")
    school_path = root / "schools_sample.csv"
    if school_path.is_file():
        known_ids = {row["school_id"] for row in read_csv(school_path)}
        if not pilot_ids <= known_ids:
            errors.append("one or more pilot school IDs are not in schools_sample.csv")
    else:
        errors.append(f"missing school source file: {school_path}")

    _validate_manifest(config_dir, errors)
    _validate_packages(errors)
    return ValidationResult(errors, warnings)
