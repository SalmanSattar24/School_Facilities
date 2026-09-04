from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from .calibration import observations, summarize
from .configuration import validate_configuration
from .imagery import fetch_esri_image
from .schema import (
    GROUND_TRUTH_COLUMNS,
    MEASUREMENT_COLUMNS,
    ground_truth_template,
    measurement_template,
    read_csv,
    validate_measurements,
    validate_ground_truth,
    validate_schools,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHOOLS = ROOT / "schools_sample.csv"
DEFAULT_MEASUREMENTS = ROOT / "measurements.csv"
DEFAULT_GROUND_TRUTH = ROOT / "data" / "validation" / "ground_truth.csv"
DEFAULT_IMAGERY = ROOT / "data" / "imagery"
DEFAULT_RAW_VLM = ROOT / "data" / "model_outputs" / "quarantine" / "v1.10" / "raw"


def _print_validation(errors: list[str], warnings: list[str]) -> None:
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


def command_prepare(args: argparse.Namespace) -> int:
    schools_path = Path(args.schools)
    result = validate_schools(schools_path)
    _print_validation(result.errors, result.warnings)
    if not result.ok:
        return 1

    schools = read_csv(schools_path)
    outputs = [
        (Path(args.measurements), MEASUREMENT_COLUMNS, measurement_template(schools)),
        (Path(args.ground_truth), GROUND_TRUTH_COLUMNS, ground_truth_template(schools)),
    ]
    for path, columns, rows in outputs:
        if path.exists() and not args.force:
            print(f"Preserved existing {path}")
            continue
        write_csv(path, columns, rows)
        print(f"Wrote {path} ({len(rows)} schools)")
    return 0


def command_fetch(args: argparse.Namespace) -> int:
    schools_path = Path(args.schools)
    result = validate_schools(schools_path)
    _print_validation(result.errors, result.warnings)
    if not result.ok:
        return 1
    schools = read_csv(schools_path)
    if args.limit is not None:
        schools = schools[: args.limit]
    failures = 0
    for index, school in enumerate(schools, start=1):
        try:
            path, downloaded = fetch_esri_image(
                school_id=school["school_id"],
                school_name=school["school_name"],
                latitude=float(school["latitude"]),
                longitude=float(school["longitude"]),
                output_dir=Path(args.output_dir),
                half_width_m=args.half_width_m,
                pixels=args.pixels,
                overwrite=args.overwrite,
            )
            action = "downloaded" if downloaded else "cached"
            print(f"[{index}/{len(schools)}] {action}: {path.name}")
        except Exception as error:  # keep the batch auditable and continue to the next school
            failures += 1
            print(f"[{index}/{len(schools)}] ERROR {school['school_id']}: {error}", file=sys.stderr)
    return 1 if failures else 0


def command_fetch_naip_context(args: argparse.Namespace) -> int:
    from .naip import fetch_naip_product

    schools_path = Path(args.schools)
    result = validate_schools(schools_path)
    _print_validation(result.errors, result.warnings)
    if not result.ok:
        return 1
    matches = [row for row in read_csv(schools_path) if row["school_id"] == args.school_id]
    if len(matches) != 1:
        print(f"ERROR: school_id not found exactly once: {args.school_id}", file=sys.stderr)
        return 1
    school = matches[0]
    try:
        result = fetch_naip_product(
            root=Path(args.root),
            school_id=school["school_id"],
            school_name=school["school_name"],
            product="context",
            center_latitude=float(school["latitude"]),
            center_longitude=float(school["longitude"]),
            requested_latitude=float(school["latitude"]),
            requested_longitude=float(school["longitude"]),
            campus_resolution_status="unresolved",
            campus_resolution_notes=(
                "Initial context centered on the supplied CCD coordinate; automatic public-polygon resolution pending."
            ),
            overwrite=args.overwrite,
        )
    except Exception as error:
        print(f"ERROR: NAIP context acquisition failed: {error}", file=sys.stderr)
        return 1
    action = "downloaded" if result.downloaded else "cached"
    print(f"NAIP context {action}: {result.jpeg_path}")
    print(f"Georeferenced master: {result.geotiff_path}")
    print(f"Provenance: {result.sidecar_path}")
    print("Next step: run resolve-campus; human review is requested only if the automatic match is flagged.")
    return 0


def command_fetch_naip_detail(args: argparse.Namespace) -> int:
    from .campus import extent_plan_for_scope
    from .naip import fetch_naip_product

    root = Path(args.root)
    schools_path = Path(args.schools)
    result = validate_schools(schools_path)
    _print_validation(result.errors, result.warnings)
    if not result.ok:
        return 1
    matches = [row for row in read_csv(schools_path) if row["school_id"] == args.school_id]
    if len(matches) != 1:
        print(f"ERROR: school_id not found exactly once: {args.school_id}", file=sys.stderr)
        return 1
    school = matches[0]
    automatic_path = root / "data" / "campus_resolutions" / f"{args.school_id}.json"
    manual_path = root / "data" / "reviewed" / args.school_id / "campus.json"
    review_path = automatic_path if automatic_path.is_file() else manual_path
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        center = review["resolved_center"]
        status = review.get("status", review.get("review_status"))
        notes = review["boundary_notes"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
        print(f"ERROR: invalid or missing campus review {review_path}: {error}", file=sys.stderr)
        return 1
    if review.get("school_id") != args.school_id or status not in {"confirmed", "probable"}:
        print("ERROR: detail imagery requires a matching confirmed/probable campus review", file=sys.stderr)
        return 1
    extent_m = review.get("recommended_detail_extent_m", 600)
    if review_path == automatic_path:
        bbox = review.get("bbox_wgs84")
        try:
            boundary_bbox = (
                tuple(float(value) for value in bbox)
                if isinstance(bbox, list) and len(bbox) == 4
                else None
            )
            scope_mode = str(review.get("scope_mode") or "authoritative_polygon")
            extent_m, unclamped_extent_m, clipped = extent_plan_for_scope(
                scope_mode, boundary_bbox
            )
        except (TypeError, ValueError) as error:
            print(f"ERROR: invalid automatic campus boundary extent: {error}", file=sys.stderr)
            return 1
        if clipped:
            print(
                "ERROR: buffered campus exceeds the 1,200 m single-image limit; "
                "USER ACTION REQUIRED: review this large-campus case before tiled acquisition.",
                file=sys.stderr,
            )
            print(f"Unclamped required extent: {unclamped_extent_m} m", file=sys.stderr)
            return 1
    try:
        result = fetch_naip_product(
            root=root,
            school_id=school["school_id"],
            school_name=school["school_name"],
            product="detail",
            center_latitude=float(center["latitude"]),
            center_longitude=float(center["longitude"]),
            requested_latitude=float(school["latitude"]),
            requested_longitude=float(school["longitude"]),
            campus_resolution_status=status,
            campus_resolution_notes=str(notes),
            extent_m=float(extent_m),
            overwrite=args.overwrite,
        )
    except Exception as error:
        print(f"ERROR: NAIP detail acquisition failed: {error}", file=sys.stderr)
        return 1
    action = "downloaded" if result.downloaded else "cached"
    print(f"NAIP detail {action}: {result.jpeg_path}")
    print(f"Adaptive detail extent: {float(extent_m):g} m square")
    print(f"Georeferenced master: {result.geotiff_path}")
    print(f"Provenance: {result.sidecar_path}")
    if review_path == automatic_path and review.get("status") == "confirmed":
        from .campus import freeze_automatic_vlm_inputs

        try:
            frozen_path = freeze_automatic_vlm_inputs(root, args.school_id)
        except Exception as error:
            print(f"ERROR: automatic input freeze failed: {error}", file=sys.stderr)
            return 1
        if review.get("scope_mode") == "soft_boundary":
            print(f"Frozen soft-scope context/detail inputs: {frozen_path}")
            print(
                "The proposal controlled centering and crop size only; the facility model "
                "will search the entire detail image."
            )
        elif review.get("scope_mode") == "center_only":
            print(f"Frozen center-only context/detail inputs: {frozen_path}")
            print(
                "No campus polygon is asserted; the facility model will search the "
                "entire detail image and perimeter claims remain uncertainty-guarded."
            )
        elif review.get("confirmation_method") == "user_manual_polygon_review":
            print(f"Frozen manually approved context/detail inputs: {frozen_path}")
            print("The required human campus approval is recorded in the resolution artifact.")
        else:
            print(f"Automatically frozen context/detail inputs: {frozen_path}")
            print("Routine human centering approval is not required for this confirmed match.")
    else:
        print("USER ACTION REQUIRED: review this flagged campus/image pair before Gemini inference.")
    return 0


def command_resolve_campus(args: argparse.Namespace) -> int:
    from .campus import overpass_school_candidates, select_campus, write_resolution

    root = Path(args.root).resolve()
    schools_path = Path(args.schools)
    result = validate_schools(schools_path)
    _print_validation(result.errors, result.warnings)
    if not result.ok:
        return 1
    matches = [row for row in read_csv(schools_path) if row["school_id"] == args.school_id]
    if len(matches) != 1:
        print(f"ERROR: school_id not found exactly once: {args.school_id}", file=sys.stderr)
        return 1
    output_path = root / "data" / "campus_resolutions" / f"{args.school_id}.json"
    if output_path.exists() and not args.overwrite:
        print(f"Cached automatic campus resolution: {output_path}")
        return 0
    school = matches[0]
    try:
        elements = overpass_school_candidates(
            float(school["latitude"]),
            float(school["longitude"]),
        )
        resolution = select_campus(school, elements)
        write_resolution(output_path, resolution)
    except Exception as error:
        print(f"ERROR: automatic campus resolution failed: {error}", file=sys.stderr)
        return 1
    print(f"Automatic campus resolution saved: {output_path}")
    print(f"Status: {resolution.status}")
    print(f"Method: {resolution.method}")
    print(
        "Resolved center: "
        f"{resolution.resolved_latitude:.7f}, {resolution.resolved_longitude:.7f}"
    )
    print(f"Recommended detail extent: {resolution.recommended_detail_extent_m} m square")
    print(resolution.reason)
    if resolution.requires_human_review:
        print("USER ACTION REQUIRED: automatic campus match is not confirmed; review the flagged case.")
    else:
        print("Automatic campus match passed; routine human centering is not required.")
    return 0


def command_prepare_campus_review(args: argparse.Namespace) -> int:
    from .campus_review import prepare_campus_review_overlay

    try:
        overlay_path, metadata_path = prepare_campus_review_overlay(
            Path(args.root), args.school_id
        )
    except Exception as error:
        print(f"ERROR: campus review overlay failed: {error}", file=sys.stderr)
        return 1
    print(f"Campus review overlay: {overlay_path}")
    print(f"Overlay provenance: {metadata_path}")
    print("USER ACTION REQUIRED: approve or correct the flagged campus boundary.")
    return 0


def command_approve_campus(args: argparse.Namespace) -> int:
    from .campus import approve_flagged_campus_polygon

    try:
        resolution_path = approve_flagged_campus_polygon(
            Path(args.root),
            args.school_id,
            review_note=args.review_note,
            reviewed_at=args.reviewed_at,
        )
    except Exception as error:
        print(f"ERROR: campus approval failed: {error}", file=sys.stderr)
        return 1
    print(f"Manually confirmed campus polygon: {resolution_path}")
    print("No imagery or Gemini request was made by this approval command.")
    return 0


def command_activate_soft_scope(args: argparse.Namespace) -> int:
    from .campus import activate_boundary_proposal_as_soft_scope

    try:
        resolution_path = activate_boundary_proposal_as_soft_scope(
            Path(args.root), args.school_id
        )
    except Exception as error:
        print(f"ERROR: soft-scope activation failed: {error}", file=sys.stderr)
        return 1
    print(f"Soft boundary activated for crop guidance: {resolution_path}")
    print("No Gemini request was made and the polygon was not promoted to a hard mask.")
    print("Next step: fetch-naip-detail will create and freeze the buffered detail image.")
    return 0


def command_activate_center_only(args: argparse.Namespace) -> int:
    from .campus import activate_center_only_scope

    try:
        resolution_path = activate_center_only_scope(Path(args.root), args.school_id)
    except Exception as error:
        print(f"ERROR: center-only activation failed: {error}", file=sys.stderr)
        return 1
    print(f"Center-only scope activated: {resolution_path}")
    print("No Gemini request was made and no polygon will be used as a measurement mask.")
    print("Next step: fetch-naip-detail will create and freeze the 800 m detail image.")
    return 0


def command_propose_campus_boundary(args: argparse.Namespace) -> int:
    from .boundary_vlm import (
        GeminiBoundaryClient,
        build_boundary_request,
        load_boundary_bundle,
        load_boundary_input,
    )
    from .vlm import _request_fingerprint, _sha256

    root = Path(args.root).resolve()
    try:
        school = load_boundary_input(root, args.school_id)
        config, _, _ = load_boundary_bundle(root)
        request = build_boundary_request(root, school)
    except Exception as error:
        print(f"ERROR: boundary proposal preparation failed: {error}", file=sys.stderr)
        return 1
    manifest = {
        "configuration_id": config["configuration_id"],
        "model": config["model"],
        "school_id": school.school_id,
        "school_name": school.school_name,
        "image_role": "context",
        "context_path": str(school.context_path),
        "context_sha256": _sha256(school.context_path),
        "request_fingerprint_sha256": _request_fingerprint(request),
        "output_path": str(root / config["outputs"]["raw_directory"] / f"{school.school_id}.json"),
        "auto_confirmation_enabled": config["decision_policy"]["auto_confirmation_enabled"],
        "gemini_request_will_be_made": bool(args.live),
    }
    print(json.dumps(manifest, indent=2))
    if not args.live:
        print("Dry run only: no credential loaded and no Gemini request made.")
        return 0
    if not args.confirm_free_tier_boundary:
        print(
            "ERROR: live boundary proposal requires --confirm-free-tier-boundary",
            file=sys.stderr,
        )
        return 1
    client = None
    try:
        client = GeminiBoundaryClient.from_environment(root)
        output_path = client.propose(school)
    except Exception as error:
        print(f"ERROR: controlled Gemini boundary call failed: {error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()
    print(f"Boundary proposal saved: {output_path}")
    print("USER ACTION REQUIRED: review the guarded proposal before it can define VLM scope.")
    return 0


def command_reconcile_boundary(args: argparse.Namespace) -> int:
    from .boundary_vlm import reconcile_rejected_boundary

    try:
        output_path = reconcile_rejected_boundary(Path(args.root), args.school_id)
    except Exception as error:
        print(f"ERROR: boundary reconciliation failed: {error}", file=sys.stderr)
        return 1
    print(f"Boundary proposal reconciled offline: {output_path}")
    print("No Gemini request was made. USER ACTION REQUIRED: review the guarded proposal.")
    return 0


def command_prepare_boundary_review(args: argparse.Namespace) -> int:
    from .campus_review import prepare_boundary_proposal_overlay

    try:
        overlay_path, metadata_path = prepare_boundary_proposal_overlay(
            Path(args.root), args.school_id
        )
    except Exception as error:
        print(f"ERROR: boundary review overlay failed: {error}", file=sys.stderr)
        return 1
    print(f"Boundary proposal overlay: {overlay_path}")
    print(f"Overlay provenance: {metadata_path}")
    print("USER ACTION REQUIRED: approve, correct, or reject the proposed campus boundary.")
    return 0


def _command_assess(args: argparse.Namespace, *, mode: str) -> int:
    from .vlm import (
        GeminiVLMClient,
        build_interaction_request,
        load_approved_school_input,
        load_vlm_bundle,
    )

    root = Path(args.root).resolve()
    try:
        if mode == "pilot":
            pilot = json.loads((root / "config" / "pilot_schools.json").read_text(encoding="utf-8"))
            pilot_ids = {row["school_id"] for row in pilot["schools"]}
            if args.school_id not in pilot_ids:
                raise ValueError("school is not in the frozen three-school pilot")
        school = load_approved_school_input(root, args.school_id)
        request = build_interaction_request(root, school)
        vlm_config, response_schema, _ = load_vlm_bundle(root)
        field_protocol = json.loads(
            (root / vlm_config["evidence_policy"]["field_protocol_path"]).read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        print(f"ERROR: cannot prepare {mode} request: {error}", file=sys.stderr)
        return 1

    if not args.live:
        images = [item for item in request["input"] if item["type"] == "image"]
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "network_request_made": False,
                    "school_id": school.school_id,
                    "school_name": school.school_name,
                    "profile": "final",
                    "model": request["model"],
                    "configuration_id": vlm_config["configuration_id"],
                    "response_schema_version": response_schema["properties"][
                        "schema_version"
                    ]["enum"][0],
                    "observable_feature_protocol": field_protocol["protocol_id"],
                    "feature_assessments_required": len(field_protocol["features"]),
                    "item_specific_questions_required": sum(
                        len(specification["questions"])
                        for specification in field_protocol["features"].values()
                    ),
                    "fencing_uses_coverage_interval": True,
                    "fence_shadow_absence_is_negative_evidence": False,
                    "structured_evidence_packets_required": True,
                    "hard_conflicts_force_guarded_unknown": True,
                    "soft_risks_only_add_review": True,
                    "solar_area_relative_tolerance": vlm_config["evidence_policy"][
                        "solar_area_consistency"
                    ]["relative_tolerance_fraction"],
                    "image_order": [image["data"].name for image in images],
                    "image_resolution": [image["resolution"] for image in images],
                    "detail_metres_per_pixel": school.detail.metres_per_pixel,
                    "campus_boundary_points": len(
                        school.campus_boundary_detail_normalized
                    ),
                    "campus_scope_mode": school.campus_scope_mode,
                    "scope_boundary_authority": school.scope_boundary_authority,
                    "measurement_search_scope": school.measurement_search_scope,
                    "structured_output": request["response_format"]["mime_type"],
                    "store_interaction": request["store"],
                    "next_step": (
                        "request contract is verified; explicit approval is required before --live"
                    ),
                },
                indent=2,
            )
        )
        return 0

    client: GeminiVLMClient | None = None
    try:
        client = GeminiVLMClient.from_environment(root)
        output_path = client.assess(school, mode=mode)  # type: ignore[arg-type]
    except Exception as error:
        print(f"ERROR: controlled Gemini {mode} call failed: {error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()
    pilot = json.loads(
        (root / "config" / "pilot_schools.json").read_text(encoding="utf-8")
    )
    validation_ids = set(pilot.get("excluded_validation_school_ids", []))
    if school.school_id in validation_ids:
        print(f"Blind validation response quarantined: {output_path}")
        print(
            "DO NOT OPEN: freeze the blind reference row before revealing this response."
        )
    else:
        print(f"Raw schema-valid Gemini output saved: {output_path}")
        print(
            "USER ACTION REQUIRED: review every suggested field before assigning "
            "final values/confidence."
        )
    return 0


def command_assess_pilot(args: argparse.Namespace) -> int:
    return _command_assess(args, mode="pilot")


def command_assess_school(args: argparse.Namespace) -> int:
    if args.live and not args.confirm_free_tier_production:
        print(
            "ERROR: live production assessment requires --confirm-free-tier-production",
            file=sys.stderr,
        )
        return 1
    return _command_assess(args, mode="production")


def command_prepare_vlm_crops(args: argparse.Namespace) -> int:
    from .facility_crops import prepare_facility_crops

    root = Path(args.root).resolve()
    try:
        products = prepare_facility_crops(
            root,
            args.school_id,
            overwrite=args.overwrite,
        )
    except Exception as error:
        print(f"ERROR: facility crop preparation failed: {error}", file=sys.stderr)
        return 1
    for product in products:
        action = "generated" if product.downloaded else "cached"
        print(f"{product.role} {action}: {product.jpeg_path}")
        print(f"Provenance: {product.sidecar_path}")
    print("No model request was made. These diagnostic crops are not final-model inputs.")
    return 0


def command_reconcile_rejected(args: argparse.Namespace) -> int:
    from .vlm import reconcile_rejected_response

    root = Path(args.root).resolve()
    try:
        output_path = reconcile_rejected_response(root, args.school_id)
    except Exception as error:
        print(f"ERROR: rejected response could not be reconciled: {error}", file=sys.stderr)
        return 1
    pilot = json.loads(
        (root / "config" / "pilot_schools.json").read_text(encoding="utf-8")
    )
    validation_ids = set(pilot.get("excluded_validation_school_ids", []))
    if args.school_id in validation_ids:
        print(f"Blind validation response reconciled in quarantine: {output_path}")
        print("No network request was made. DO NOT OPEN before reference freeze.")
    else:
        print(f"Flagged raw VLM response saved: {output_path}")
        print("No network request was made.")
        print(
            "USER ACTION REQUIRED: review all pipeline_review_fields before final "
            "adjudication."
        )
    return 0


def command_reconcile_rejected_auditor(args: argparse.Namespace) -> int:
    from .auditor import reconcile_rejected_auditor_response

    root = Path(args.root).resolve()
    try:
        output_path = reconcile_rejected_auditor_response(root, args.school_id)
    except Exception as error:
        print(f"ERROR: auditor reconciliation failed: {error}", file=sys.stderr)
        return 1
    print(f"Guarded auditor response saved: {output_path}")
    print("No network request was made.")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    school_result = validate_schools(Path(args.schools))
    _print_validation(school_result.errors, school_result.warnings)
    if not school_result.ok:
        return 1
    path = Path(args.measurements)
    if not path.exists():
        print(f"ERROR: measurements file does not exist: {path}", file=sys.stderr)
        return 1
    result = validate_measurements(path, Path(args.schools), final=args.final)
    _print_validation(result.errors, result.warnings)
    if result.ok:
        mode = "final" if args.final else "draft"
        print(f"Validation passed ({mode} mode): {path}")
        return 0
    return 1


def command_calibrate(args: argparse.Namespace) -> int:
    measurements_path = Path(args.measurements)
    truth_path = Path(args.ground_truth)
    if not measurements_path.exists() or not truth_path.exists():
        print("ERROR: run prepare and fill measurements plus ground truth first", file=sys.stderr)
        return 1
    items = observations(read_csv(measurements_path), read_csv(truth_path))
    report = summarize(items)
    print(json.dumps(report, indent=2))
    if not items:
        print("ERROR: no comparable hand-labeled fields were found", file=sys.stderr)
        return 1
    return 0


def command_run_all(args: argparse.Namespace) -> int:
    from .production_pipeline import run_all

    try:
        report = run_all(
            Path(args.root),
            reference_path=Path(args.reference),
            blind_reference_path=Path(args.blind_reference),
            measurements_path=Path(args.measurements),
            snapshot_path=Path(args.snapshot),
            report_path=Path(args.report),
        )
    except Exception as error:
        print(f"ERROR: full pipeline failed: {error}", file=sys.stderr)
        return 1
    print(f"Frozen 25-school prediction snapshot: {args.snapshot}")
    print(f"New measurements CSV: {args.measurements}")
    print(f"Evaluation report: {args.report}")
    print(
        "Result: "
        f"{report['correct_n']} correct, {report['wrong_n']} wrong, "
        f"{report['abstained_unknown_n']} unknown across {report['evaluable_n']} evaluable fields"
    )
    print(
        f"Flag capture={report['problem_flag_capture']}; "
        f"silent wrong={report['silent_wrong_n']}; "
        f"auto-accept precision={report['auto_accept_precision']}"
    )
    print("USER ACTION REQUIRED: review every row marked needs-review before submission.")
    return 0


def command_evaluate_vlm(args: argparse.Namespace) -> int:
    from .vlm_evaluation import raw_observations, sha256, summarize_raw_vlm

    root = Path(args.root).resolve()
    truth_path = Path(args.ground_truth).resolve()
    raw_directory = Path(args.raw_directory).resolve()
    if not truth_path.is_file():
        print(f"ERROR: blind reference file does not exist: {truth_path}", file=sys.stderr)
        return 1
    try:
        pilot = json.loads((root / "config" / "pilot_schools.json").read_text(encoding="utf-8"))
        validation_ids = set(pilot["excluded_validation_school_ids"])
    except Exception as error:
        print(f"ERROR: validation-school configuration is invalid: {error}", file=sys.stderr)
        return 1
    reference_validation = validate_ground_truth(
        truth_path, root / "schools_sample.csv", validation_ids
    )
    _print_validation(reference_validation.errors, reference_validation.warnings)
    if not reference_validation.ok:
        return 1
    actual_hash = sha256(truth_path)
    if actual_hash.lower() != args.reference_sha256.lower():
        print("ERROR: reference SHA-256 does not match; freeze labels before unblinding", file=sys.stderr)
        return 1
    try:
        items, exclusions = raw_observations(
            raw_directory,
            read_csv(truth_path),
            validation_ids,
        )
    except Exception as error:
        print(f"ERROR: raw VLM evaluation failed: {error}", file=sys.stderr)
        return 1
    report = summarize_raw_vlm(items, exclusions)
    report["reference_sha256"] = actual_hash
    report["raw_directory"] = str(raw_directory)
    print(json.dumps(report, indent=2))
    if not items:
        print("ERROR: no evaluable raw VLM fields were found", file=sys.stderr)
        return 1
    return 0


def command_validate_reference(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    try:
        pilot = json.loads((root / "config" / "pilot_schools.json").read_text(encoding="utf-8"))
        validation_ids = set(pilot["excluded_validation_school_ids"])
        result = validate_ground_truth(
            Path(args.ground_truth).resolve(),
            Path(args.schools).resolve(),
            validation_ids,
        )
    except Exception as error:
        print(f"ERROR: blind-reference validation failed: {error}", file=sys.stderr)
        return 1
    _print_validation(result.errors, result.warnings)
    if not result.ok:
        return 1
    print("Blind-reference validation passed for all six frozen schools.")
    return 0


def command_evaluate_pilot_uncertainty(args: argparse.Namespace) -> int:
    from .vlm_evaluation import pilot_uncertainty_diagnostic

    root = Path(args.root).resolve()
    try:
        pilot = json.loads((root / "config" / "pilot_schools.json").read_text(encoding="utf-8"))
        pilot_ids = {row["school_id"] for row in pilot["schools"]}
        report = pilot_uncertainty_diagnostic(
            Path(args.raw_directory),
            read_csv(Path(args.measurements)),
            pilot_ids,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    except Exception as error:
        print(f"ERROR: pilot uncertainty diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    print(f"Diagnostic report saved: {output_path}")
    print("This is non-blind pilot debugging, not validation accuracy.")
    return 0


def command_evaluate_pilot_auditor(args: argparse.Namespace) -> int:
    from .vlm_evaluation import pilot_auditor_diagnostic

    root = Path(args.root).resolve()
    try:
        pilot = json.loads((root / "config" / "pilot_schools.json").read_text(encoding="utf-8"))
        pilot_ids = {row["school_id"] for row in pilot["schools"]}
        report = pilot_auditor_diagnostic(
            Path(args.raw_directory),
            Path(args.audit_directory),
            read_csv(Path(args.measurements)),
            pilot_ids,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    except Exception as error:
        print(f"ERROR: pilot auditor diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    print(f"Diagnostic report saved: {output_path}")
    print("This is non-blind pilot debugging, not validation accuracy.")
    return 0


def command_audit_vlm(args: argparse.Namespace) -> int:
    from .auditor import (
        GeminiEvidenceAuditorClient,
        build_auditor_request,
        load_auditor_bundle,
    )

    root = Path(args.root).resolve()
    try:
        request, primary_path = build_auditor_request(root, args.school_id)
        config, _, _ = load_auditor_bundle(root)
    except Exception as error:
        print(f"ERROR: cannot prepare evidence-auditor request: {error}", file=sys.stderr)
        return 1
    if not args.live:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "network_request_made": False,
                    "school_id": args.school_id,
                    "primary_record": str(primary_path),
                    "model": request["model"],
                    "text_only": True,
                    "images_sent": 0,
                    "human_reviewed_values_sent": False,
                    "blind_reference_values_sent": False,
                    "auditor_may_overwrite_primary_value": False,
                    "output_directory": config["outputs"]["directory"],
                },
                indent=2,
            )
        )
        return 0
    if not args.confirm_free_tier_auditor:
        print(
            "ERROR: live auditing requires --confirm-free-tier-auditor after reviewing "
            "the exact dry-run request summary",
            file=sys.stderr,
        )
        return 1
    client: GeminiEvidenceAuditorClient | None = None
    try:
        client = GeminiEvidenceAuditorClient.from_environment(root)
        output_path = client.audit(args.school_id)
    except Exception as error:
        print(f"ERROR: controlled Gemini evidence audit failed: {error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()
    print(f"Schema-valid evidence audit saved: {output_path}")
    print("The auditor did not overwrite any primary value or confidence.")
    return 0


def command_validate_config(args: argparse.Namespace) -> int:
    result = validate_configuration(Path(args.root))
    _print_validation(result.errors, result.warnings)
    if not result.ok:
        return 1
    try:
        from .streetview import load_streetview_config
        from .streetview_vlm import load_streetview_vlm_bundle

        load_streetview_config(Path(args.root))
        load_streetview_vlm_bundle(Path(args.root))
    except Exception as error:
        print(f"ERROR: V1.11 supplemental configuration is invalid: {error}", file=sys.stderr)
        return 1
    print(f"Frozen technical configuration is valid: {Path(args.root) / 'config'}")
    print("The separately versioned V1.11 Street View configuration and schema are also valid.")
    return 0


def command_save_gemini_key(args: argparse.Namespace) -> int:
    from .credentials import CredentialError, upsert_api_key

    root = Path(args.root).resolve()
    try:
        config = json.loads((root / "config" / "vlm.json").read_text(encoding="utf-8"))
        credentials = config["credentials"]
        value = getpass.getpass("Paste Gemini API key (hidden): ")
        upsert_api_key(
            value,
            environment_variable=credentials["api_key_environment_variable"],
            secrets_file=root / credentials["local_secrets_file"],
            provider_name="Gemini",
        )
    except (CredentialError, FileNotFoundError, json.JSONDecodeError, KeyError) as error:
        print(f"ERROR: could not save Gemini credential: {error}", file=sys.stderr)
        return 1
    print(f"Gemini API key saved in ignored local secrets file: {credentials['local_secrets_file']}")
    print("Any separately saved Street View key was preserved.")
    print("It was not written to tracked configuration, output, or command history.")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    from .operator import print_doctor, run_doctor

    try:
        checks = run_doctor(Path(args.root).resolve(), require_key=args.require_key)
    except Exception as error:
        print(f"ERROR: operator readiness check failed: {error}", file=sys.stderr)
        return 1
    return print_doctor(checks)


def command_workflow_status(args: argparse.Namespace) -> int:
    from .operator import workflow_rows

    try:
        rows = workflow_rows(Path(args.root).resolve())
    except Exception as error:
        print(f"ERROR: workflow status failed: {error}", file=sys.stderr)
        return 1
    headers = ("school_id", "validation", "stage", "next_action")
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    print("\nSummary: " + ", ".join(f"{stage}={count}" for stage, count in sorted(counts.items())))
    if any(row["stage"] == "BLIND LABEL" for row in rows):
        print(
            "USER ACTION REQUIRED at the blind-label stage: inspect imagery only. "
            "Do not open data/model_outputs/quarantine/v1.10."
        )
    return 0


def command_prepare_review(args: argparse.Namespace) -> int:
    from .operator import prepare_review_packet

    root = Path(args.root).resolve()
    try:
        output = prepare_review_packet(
            root,
            args.school_id,
            output_path=Path(args.output).resolve() if args.output else None,
            reference_sha256=args.reference_sha256,
        )
    except Exception as error:
        print(f"ERROR: review packet preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Human review packet: {output}")
    print("Open the HTML file in a browser, inspect both images, then run review-school.")
    return 0


def command_prepare_blind_review(args: argparse.Namespace) -> int:
    from .operator import prepare_blind_review_packet

    try:
        output = prepare_blind_review_packet(
            Path(args.root).resolve(),
            args.school_id,
            output_path=Path(args.output).resolve() if args.output else None,
        )
    except Exception as error:
        print(f"ERROR: blind-review packet preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Imagery-only blind-reference packet: {output}")
    print("Do not open quarantined predictions. Enter the label in data/validation/ground_truth.csv.")
    return 0


def command_review_school(args: argparse.Namespace) -> int:
    from .operator import review_school_interactively

    try:
        output = review_school_interactively(
            Path(args.root).resolve(),
            args.school_id,
            reference_sha256=args.reference_sha256,
        )
    except (KeyboardInterrupt, EOFError):
        print("\nReview cancelled; measurements.csv was not changed.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"ERROR: school review was not saved: {error}", file=sys.stderr)
        return 1
    print(f"Reviewed row saved: {output}")
    print(f"Backup of the prior file: {output.with_suffix('.csv.bak')}")
    print("Next step: run school-facilities validate")
    return 0


def command_save_streetview_key(args: argparse.Namespace) -> int:
    from .credentials import CredentialError, upsert_api_key

    key = getpass.getpass("Paste Google Street View API key (hidden): ")
    try:
        upsert_api_key(
            key,
            environment_variable="GOOGLE_STREET_VIEW_API_KEY",
            secrets_file=Path(args.root).resolve() / "secrets.local.env",
            provider_name="Google Street View",
        )
    except CredentialError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Street View key saved in ignored secrets.local.env; the existing Gemini key was preserved.")
    print("Never commit, attach, print, or submit this file.")
    return 0


def _streetview_school_ids(args: argparse.Namespace) -> list[str] | None:
    if getattr(args, "all", False):
        return None
    values = list(getattr(args, "school_id", []) or [])
    if not values:
        raise ValueError("select --all or provide at least one --school-id")
    return values


def command_streetview_plan(args: argparse.Namespace) -> int:
    from .streetview import create_probe_plan

    try:
        output = create_probe_plan(
            Path(args.root),
            school_ids=_streetview_school_ids(args),
            schools_path=Path(args.schools),
            output_path=Path(args.output) if args.output else None,
        )
    except Exception as error:
        print(f"ERROR: Street View plan failed: {error}", file=sys.stderr)
        return 1
    print(f"No-network Street View probe plan: {output}")
    print("No API key was loaded and no provider request was made.")
    print("Next step after provider setup: run streetview-probe --metadata-only.")
    return 0


def command_streetview_probe(args: argparse.Namespace) -> int:
    from .streetview import probe_metadata

    if not args.metadata_only:
        print("ERROR: streetview-probe currently requires --metadata-only", file=sys.stderr)
        return 1
    try:
        output = probe_metadata(
            Path(args.root),
            plan_path=Path(args.manifest),
            output_path=Path(args.output) if args.output else None,
        )
    except Exception as error:
        print(f"ERROR: Street View metadata probe failed: {error}", file=sys.stderr)
        return 1
    value = json.loads(output.read_text(encoding="utf-8"))
    print(f"Metadata-derived image manifest: {output}")
    print(f"Maximum later image requests: {value['maximum_billable_image_requests']}")
    print("No Street View image was retrieved by this metadata-only command.")
    return 0


def command_streetview_record_usage(args: argparse.Namespace) -> int:
    from .streetview import record_usage_snapshot

    try:
        output = record_usage_snapshot(
            Path(args.root),
            month=args.month,
            used_requests=args.used_requests,
            source=args.source,
        )
    except Exception as error:
        print(f"ERROR: usage snapshot was not saved: {error}", file=sys.stderr)
        return 1
    print(f"Provider usage snapshot saved locally: {output}")
    print("This snapshot expires after 24 hours and does not replace the provider-side API quota.")
    return 0


def command_streetview_budget_status(args: argparse.Namespace) -> int:
    from .streetview import budget_status, preflight_fetch

    try:
        value = (
            preflight_fetch(Path(args.root), Path(args.manifest))
            if args.manifest
            else budget_status(Path(args.root), require_current_snapshot=False)
        )
    except Exception as error:
        print(f"ERROR: Street View budget status failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["allowed"] else 1


def command_streetview_cost_estimate(args: argparse.Namespace) -> int:
    from .streetview import cost_estimate

    try:
        result = cost_estimate(
            Path(args.root), schools=args.schools, views_per_school=args.views_per_school
        )
    except Exception as error:
        print(f"ERROR: Street View cost estimate failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_streetview_fetch(args: argparse.Namespace) -> int:
    from .streetview import fetch_images

    try:
        result = fetch_images(
            Path(args.root),
            manifest_path=Path(args.manifest),
            live=args.live,
            max_paid_usd=args.max_paid_usd,
            provider_quota_confirmed=args.confirm_provider_quota,
        )
    except Exception as error:
        print(f"ERROR: Street View retrieval stopped: {error}", file=sys.stderr)
        return 1
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Dry run only: no credential loaded and no image request made.")
        return 0 if result["allowed"] else 1
    print(f"Fetched-manifest saved: {result}")
    print("Temporary images must now be consumed by assess-v1-11; do not copy them into the submission.")
    return 0


def command_streetview_deduplicate_manifest(args: argparse.Namespace) -> int:
    from .streetview import replace_previously_reserved_views

    try:
        output = replace_previously_reserved_views(
            Path(args.root), Path(args.manifest), output_path=Path(args.output)
        )
    except Exception as error:
        print(f"ERROR: Street View manifest deduplication failed: {error}", file=sys.stderr)
        return 1
    print(f"Deduplicated Street View manifest: {output}")
    return 0


def command_assess_v1_11(args: argparse.Namespace) -> int:
    from .streetview_vlm import (
        assess_v1_11,
        reconcile_rejected_v1_11,
        refresh_v1_11_guards,
    )

    try:
        if args.reconcile_rejected:
            if args.live or args.confirm_gemini_v1_11:
                raise ValueError("offline reconciliation cannot be combined with live flags")
            result = reconcile_rejected_v1_11(
                Path(args.root), school_id=args.school_id
            )
        elif args.refresh_guards:
            if args.live or args.confirm_gemini_v1_11:
                raise ValueError("offline guard refresh cannot be combined with live flags")
            result = refresh_v1_11_guards(Path(args.root), school_id=args.school_id)
        else:
            result = assess_v1_11(
                Path(args.root),
                fetched_manifest_path=Path(args.manifest),
                school_id=args.school_id,
                live=args.live,
                confirmed=args.confirm_gemini_v1_11,
            )
    except Exception as error:
        print(f"ERROR: V1.11 assessment failed: {error}", file=sys.stderr)
        return 1
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Dry run only: no Gemini request was made and no image was deleted.")
    elif args.reconcile_rejected:
        print(f"V1.11 response reconciled offline: {result}")
        print("No Gemini request was made; the preserved raw provider output remains unchanged.")
        print("USER ACTION REQUIRED: review the supplemental candidate and uncertainty flags.")
    elif args.refresh_guards:
        print(f"V1.11 uncertainty guards refreshed offline: {result}")
        print("No Gemini request was made and raw provider output was not changed.")
    else:
        print(f"V1.11 supplemental response saved: {result}")
        print("Street View source images were retained locally for human review and remain ignored by Git.")
        print("USER ACTION REQUIRED: review the supplemental candidate and uncertainty flags.")
    return 0


def command_assess_v1_11_batch(args: argparse.Namespace) -> int:
    from .streetview_vlm import assess_v1_11_batch

    try:
        result = assess_v1_11_batch(
            Path(args.root),
            fetched_manifest_path=Path(args.manifest),
            live=args.live,
            confirmed=args.confirm_gemini_v1_11,
        )
    except Exception as error:
        print(f"ERROR: V1.11 batch assessment failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("failed_n", 0) == 0 else 1


def command_evaluate_v1_11(args: argparse.Namespace) -> int:
    from .streetview_evaluation import evaluate_stability, evaluate_v1_11

    try:
        result = evaluate_v1_11(
            Path(args.root),
            reference_path=Path(args.ground_truth),
            reference_sha256=args.reference_sha256,
            prediction_directory=Path(args.prediction_directory),
        )
        if args.repeat_directory:
            pilot = json.loads(
                (Path(args.root) / "config" / "pilot_schools.json").read_text(encoding="utf-8")
            )
            result["stability"] = evaluate_stability(
                school_ids=pilot["excluded_validation_school_ids"],
                primary_directory=Path(args.prediction_directory),
                repeat_directories=[Path(item) for item in args.repeat_directory],
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as error:
        print(f"ERROR: V1.11 evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"V1.11 evaluation saved: {output}")
    print(json.dumps(result["overall"], indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="School facilities imagery review pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="validate schools and create review templates")
    prepare.add_argument("--schools", default=DEFAULT_SCHOOLS)
    prepare.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    prepare.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    prepare.add_argument("--force", action="store_true", help="replace existing templates")
    prepare.set_defaults(func=command_prepare)

    fetch = subparsers.add_parser("fetch-imagery", help="download no-key aerial images from Esri")
    fetch.add_argument("--schools", default=DEFAULT_SCHOOLS)
    fetch.add_argument("--output-dir", default=DEFAULT_IMAGERY)
    fetch.add_argument("--half-width-m", type=float, default=450)
    fetch.add_argument("--pixels", type=int, default=1200)
    fetch.add_argument("--limit", type=int, default=None, help="download only the first N rows")
    fetch.add_argument("--overwrite", action="store_true")
    fetch.set_defaults(func=command_fetch)

    fetch_naip = subparsers.add_parser(
        "fetch-naip-context",
        help="download the frozen dated-NAIP context product for one school",
    )
    fetch_naip.add_argument("--root", default=ROOT)
    fetch_naip.add_argument("--schools", default=DEFAULT_SCHOOLS)
    fetch_naip.add_argument("--school-id", required=True)
    fetch_naip.add_argument("--overwrite", action="store_true")
    fetch_naip.set_defaults(func=command_fetch_naip_context)

    fetch_naip_detail = subparsers.add_parser(
        "fetch-naip-detail",
        help="download the frozen dated-NAIP detail product after campus review",
    )
    fetch_naip_detail.add_argument("--root", default=ROOT)
    fetch_naip_detail.add_argument("--schools", default=DEFAULT_SCHOOLS)
    fetch_naip_detail.add_argument("--school-id", required=True)
    fetch_naip_detail.add_argument("--overwrite", action="store_true")
    fetch_naip_detail.set_defaults(func=command_fetch_naip_detail)

    resolve_campus = subparsers.add_parser(
        "resolve-campus",
        help="automatically match a school to a public campus polygon and derive its center",
    )
    resolve_campus.add_argument("--root", default=ROOT)
    resolve_campus.add_argument("--schools", default=DEFAULT_SCHOOLS)
    resolve_campus.add_argument("--school-id", required=True)
    resolve_campus.add_argument("--overwrite", action="store_true")
    resolve_campus.set_defaults(func=command_resolve_campus)

    campus_review = subparsers.add_parser(
        "prepare-campus-review",
        help="render a flagged campus polygon and centers over its context image",
    )
    campus_review.add_argument("--root", default=ROOT)
    campus_review.add_argument("--school-id", required=True)
    campus_review.set_defaults(func=command_prepare_campus_review)

    approve_campus = subparsers.add_parser(
        "approve-campus",
        help="record explicit human approval of a flagged resolver-proposed polygon",
    )
    approve_campus.add_argument("--root", default=ROOT)
    approve_campus.add_argument("--school-id", required=True)
    approve_campus.add_argument("--review-note", required=True)
    approve_campus.add_argument("--reviewed-at", default=None)
    approve_campus.set_defaults(func=command_approve_campus)

    activate_soft_scope = subparsers.add_parser(
        "activate-soft-scope",
        help="use a guarded Gemini polygon only for buffered centering and crop sizing",
    )
    activate_soft_scope.add_argument("--root", default=ROOT)
    activate_soft_scope.add_argument("--school-id", required=True)
    activate_soft_scope.set_defaults(func=command_activate_soft_scope)

    activate_center_only = subparsers.add_parser(
        "activate-center-only",
        help="use a verified center and full-image search when no polygon is trustworthy",
    )
    activate_center_only.add_argument("--root", default=ROOT)
    activate_center_only.add_argument("--school-id", required=True)
    activate_center_only.set_defaults(func=command_activate_center_only)

    propose_boundary = subparsers.add_parser(
        "propose-campus-boundary",
        help="dry-run or execute the Gemini fallback for a flagged campus boundary",
    )
    propose_boundary.add_argument("--root", default=ROOT)
    propose_boundary.add_argument("--school-id", required=True)
    propose_boundary.add_argument("--live", action="store_true")
    propose_boundary.add_argument(
        "--confirm-free-tier-boundary",
        action="store_true",
        help="confirm that this one public-imagery request may use the free-tier key",
    )
    propose_boundary.set_defaults(func=command_propose_campus_boundary)

    reconcile_boundary = subparsers.add_parser(
        "reconcile-boundary",
        help="revalidate a preserved boundary response after whitelist normalization",
    )
    reconcile_boundary.add_argument("--root", default=ROOT)
    reconcile_boundary.add_argument("--school-id", required=True)
    reconcile_boundary.set_defaults(func=command_reconcile_boundary)

    boundary_review = subparsers.add_parser(
        "prepare-boundary-review",
        help="render a guarded Gemini boundary proposal for human review",
    )
    boundary_review.add_argument("--root", default=ROOT)
    boundary_review.add_argument("--school-id", required=True)
    boundary_review.set_defaults(func=command_prepare_boundary_review)

    assess_pilot = subparsers.add_parser(
        "assess-pilot",
        help="dry-run or execute one approved Gemini pilot-school request",
    )
    assess_pilot.add_argument("--root", default=ROOT)
    assess_pilot.add_argument("--school-id", required=True)
    assess_pilot.add_argument(
        "--live",
        action="store_true",
        help="make the controlled API request; without this flag no network request occurs",
    )
    assess_pilot.set_defaults(func=command_assess_pilot)

    assess_school = subparsers.add_parser(
        "assess-school",
        help="dry-run or execute one frozen production-school Gemini request",
    )
    assess_school.add_argument("--root", default=ROOT)
    assess_school.add_argument("--school-id", required=True)
    assess_school.add_argument("--live", action="store_true")
    assess_school.add_argument(
        "--confirm-free-tier-production",
        action="store_true",
        help="confirm that this production request may use the free-tier key",
    )
    assess_school.set_defaults(func=command_assess_school)

    reconcile_rejected = subparsers.add_parser(
        "reconcile-rejected",
        help="convert a schema-valid rejected response into a guarded review record offline",
    )
    reconcile_rejected.add_argument("--root", default=ROOT)
    reconcile_rejected.add_argument("--school-id", required=True)
    reconcile_rejected.set_defaults(func=command_reconcile_rejected)

    reconcile_auditor = subparsers.add_parser(
        "reconcile-rejected-auditor",
        help="convert a schema-valid rejected auditor response offline",
    )
    reconcile_auditor.add_argument("--root", default=ROOT)
    reconcile_auditor.add_argument("--school-id", required=True)
    reconcile_auditor.set_defaults(func=command_reconcile_rejected_auditor)

    prepare_crops = subparsers.add_parser(
        "prepare-vlm-crops",
        help="generate optional human-review diagnostic crops from the approved detail GeoTIFF",
    )
    prepare_crops.add_argument("--root", default=ROOT)
    prepare_crops.add_argument("--school-id", required=True)
    prepare_crops.add_argument("--overwrite", action="store_true")
    prepare_crops.set_defaults(func=command_prepare_vlm_crops)

    validate = subparsers.add_parser("validate", help="validate the measurement deliverable")
    validate.add_argument("--schools", default=DEFAULT_SCHOOLS)
    validate.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    validate.add_argument("--final", action="store_true", help="require every field and confidence")
    validate.set_defaults(func=command_validate)

    calibrate = subparsers.add_parser("calibrate", help="compare measurements with hand labels")
    calibrate.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    calibrate.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    calibrate.set_defaults(func=command_calibrate)

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="freeze all 25 existing predictions, then evaluate and build a new measurements CSV",
    )
    run_all_parser.add_argument("--root", default=ROOT)
    run_all_parser.add_argument("--reference", default=ROOT / "measurements_old.csv")
    run_all_parser.add_argument(
        "--blind-reference", default=ROOT / "data" / "validation" / "ground_truth.csv"
    )
    run_all_parser.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    run_all_parser.add_argument(
        "--snapshot", default=ROOT / "outputs" / "all25_predictions_frozen.json"
    )
    run_all_parser.add_argument(
        "--report", default=ROOT / "outputs" / "full_pipeline_evaluation.json"
    )
    run_all_parser.set_defaults(func=command_run_all)

    validate_reference = subparsers.add_parser(
        "validate-reference",
        help="validate all six blind-reference rows before hashing or unblinding",
    )
    validate_reference.add_argument("--root", default=ROOT)
    validate_reference.add_argument("--schools", default=DEFAULT_SCHOOLS)
    validate_reference.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    validate_reference.set_defaults(func=command_validate_reference)

    evaluate_vlm = subparsers.add_parser(
        "evaluate-vlm",
        help="evaluate quarantined raw VLM responses after blind-reference freeze",
    )
    evaluate_vlm.add_argument("--root", default=ROOT)
    evaluate_vlm.add_argument("--raw-directory", default=DEFAULT_RAW_VLM)
    evaluate_vlm.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    evaluate_vlm.add_argument(
        "--reference-sha256",
        required=True,
        help="SHA-256 recorded before raw validation outputs are revealed",
    )
    evaluate_vlm.set_defaults(func=command_evaluate_vlm)

    evaluate_pilot = subparsers.add_parser(
        "evaluate-pilot-uncertainty",
        help="compare V1.7 uncertainty routing with prior reviewed pilot rows",
    )
    evaluate_pilot.add_argument("--root", default=ROOT)
    evaluate_pilot.add_argument(
        "--raw-directory", default=ROOT / "data" / "model_outputs" / "final" / "v1.7"
    )
    evaluate_pilot.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    evaluate_pilot.add_argument(
        "--output", default=ROOT / "outputs" / "pilot_uncertainty_diagnostic.json"
    )
    evaluate_pilot.set_defaults(func=command_evaluate_pilot_uncertainty)

    evaluate_auditor = subparsers.add_parser(
        "evaluate-pilot-auditor",
        help="measure text-auditor capture against prior reviewed pilot rows",
    )
    evaluate_auditor.add_argument("--root", default=ROOT)
    evaluate_auditor.add_argument(
        "--raw-directory", default=ROOT / "data" / "model_outputs" / "final" / "v1.10"
    )
    evaluate_auditor.add_argument(
        "--audit-directory", default=ROOT / "data" / "model_outputs" / "audits" / "v1.10"
    )
    evaluate_auditor.add_argument("--measurements", default=DEFAULT_MEASUREMENTS)
    evaluate_auditor.add_argument(
        "--output", default=ROOT / "outputs" / "pilot_auditor_diagnostic_v1.10.json"
    )
    evaluate_auditor.set_defaults(func=command_evaluate_pilot_auditor)

    audit_vlm = subparsers.add_parser(
        "audit-vlm",
        help="dry-run or execute the text-only evidence-consistency auditor",
    )
    audit_vlm.add_argument("--root", default=ROOT)
    audit_vlm.add_argument("--school-id", required=True)
    audit_vlm.add_argument(
        "--live",
        action="store_true",
        help="make one controlled auditor request; without this flag no network request occurs",
    )
    audit_vlm.add_argument(
        "--confirm-free-tier-auditor",
        action="store_true",
        help="confirm the separate free-tier text-only auditor call",
    )
    audit_vlm.set_defaults(func=command_audit_vlm)

    validate_config = subparsers.add_parser(
        "validate-config", help="verify frozen imagery/VLM configuration and dependency versions"
    )
    validate_config.add_argument("--root", default=ROOT)
    validate_config.set_defaults(func=command_validate_config)

    save_key = subparsers.add_parser(
        "save-gemini-key",
        help="store the Gemini key once in the ignored local secrets file",
    )
    save_key.add_argument("--root", default=ROOT)
    save_key.set_defaults(func=command_save_gemini_key)

    doctor = subparsers.add_parser(
        "doctor",
        help="check whether a human operator can run the project without exposing secrets",
    )
    doctor.add_argument("--root", default=ROOT)
    doctor.add_argument(
        "--require-key",
        action="store_true",
        help="fail when no Gemini key is configured (the key value is never displayed)",
    )
    doctor.set_defaults(func=command_doctor)

    workflow_status = subparsers.add_parser(
        "workflow-status",
        help="show each school's current stage and exact next operator action",
    )
    workflow_status.add_argument("--root", default=ROOT)
    workflow_status.set_defaults(func=command_workflow_status)

    prepare_review = subparsers.add_parser(
        "prepare-review",
        help="generate a local HTML packet for human review of one VLM result",
    )
    prepare_review.add_argument("--root", default=ROOT)
    prepare_review.add_argument("--school-id", required=True)
    prepare_review.add_argument("--output", default=None)
    prepare_review.add_argument(
        "--reference-sha256",
        default=None,
        help="required for validation schools after blind labels are frozen",
    )
    prepare_review.set_defaults(func=command_prepare_review)

    prepare_blind_review = subparsers.add_parser(
        "prepare-blind-review",
        help="generate an imagery-only packet for one frozen validation school",
    )
    prepare_blind_review.add_argument("--root", default=ROOT)
    prepare_blind_review.add_argument("--school-id", required=True)
    prepare_blind_review.add_argument("--output", default=None)
    prepare_blind_review.set_defaults(func=command_prepare_blind_review)

    review_school = subparsers.add_parser(
        "review-school",
        help="interactively record final human values and confidence for one school",
    )
    review_school.add_argument("--root", default=ROOT)
    review_school.add_argument("--school-id", required=True)
    review_school.add_argument(
        "--reference-sha256",
        default=None,
        help="required for validation schools after blind labels are frozen",
    )
    review_school.set_defaults(func=command_review_school)

    save_streetview_key = subparsers.add_parser(
        "save-streetview-key",
        help="store a restricted Street View key without replacing the saved Gemini key",
    )
    save_streetview_key.add_argument("--root", default=ROOT)
    save_streetview_key.set_defaults(func=command_save_streetview_key)

    streetview_plan = subparsers.add_parser(
        "streetview-plan",
        help="create a deterministic no-network Street View discovery plan",
    )
    streetview_plan.add_argument("--root", default=ROOT)
    streetview_plan.add_argument("--schools", default=DEFAULT_SCHOOLS)
    selection = streetview_plan.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--school-id", action="append")
    streetview_plan.add_argument("--output", default=None)
    streetview_plan.set_defaults(func=command_streetview_plan)

    streetview_probe = subparsers.add_parser(
        "streetview-probe",
        help="resolve panoramas using OSM roads and the non-image metadata endpoint",
    )
    streetview_probe.add_argument("--root", default=ROOT)
    streetview_probe.add_argument("--manifest", required=True)
    streetview_probe.add_argument("--output", default=None)
    streetview_probe.add_argument("--metadata-only", action="store_true", required=True)
    streetview_probe.set_defaults(func=command_streetview_probe)

    record_usage = subparsers.add_parser(
        "streetview-record-usage",
        help="record a fresh provider-console image-usage snapshot for fail-closed budgeting",
    )
    record_usage.add_argument("--root", default=ROOT)
    record_usage.add_argument("--month", required=True, help="billing month as YYYY-MM")
    record_usage.add_argument("--used-requests", type=int, required=True)
    record_usage.add_argument("--source", required=True, help="for example: Google Cloud console")
    record_usage.set_defaults(func=command_streetview_record_usage)

    budget_status_parser = subparsers.add_parser(
        "streetview-budget-status",
        help="show strict fee barriers and optionally preflight an image manifest",
    )
    budget_status_parser.add_argument("--root", default=ROOT)
    budget_status_parser.add_argument("--manifest", default=None)
    budget_status_parser.set_defaults(func=command_streetview_budget_status)

    cost_parser = subparsers.add_parser(
        "streetview-cost-estimate",
        help="estimate monthly Static Street View cost from the dated official pricing snapshot",
    )
    cost_parser.add_argument("--root", default=ROOT)
    cost_parser.add_argument("--schools", type=int, required=True)
    cost_parser.add_argument("--views-per-school", type=int, required=True)
    cost_parser.set_defaults(func=command_streetview_cost_estimate)

    streetview_fetch = subparsers.add_parser(
        "streetview-fetch",
        help="dry-run or fetch a frozen Street View image manifest under zero-paid-overage gates",
    )
    streetview_fetch.add_argument("--root", default=ROOT)
    streetview_fetch.add_argument("--manifest", required=True)
    streetview_fetch.add_argument("--live", action="store_true")
    streetview_fetch.add_argument("--max-paid-usd", type=float, default=None)
    streetview_fetch.add_argument("--confirm-provider-quota", action="store_true")
    streetview_fetch.set_defaults(func=command_streetview_fetch)

    streetview_deduplicate = subparsers.add_parser(
        "streetview-deduplicate-manifest",
        help="rotate any previously reserved panorama headings before the billable preflight",
    )
    streetview_deduplicate.add_argument("--root", default=ROOT)
    streetview_deduplicate.add_argument("--manifest", required=True)
    streetview_deduplicate.add_argument("--output", required=True)
    streetview_deduplicate.set_defaults(func=command_streetview_deduplicate_manifest)

    assess_streetview = subparsers.add_parser(
        "assess-v1-11",
        help="dry-run or execute one frozen aerial-plus-Street-View Gemini assessment",
    )
    assess_streetview.add_argument("--root", default=ROOT)
    assess_streetview.add_argument("--manifest", required=True)
    assess_streetview.add_argument("--school-id", required=True)
    assess_streetview.add_argument("--live", action="store_true")
    assess_streetview.add_argument("--confirm-gemini-v1-11", action="store_true")
    assess_streetview.add_argument("--reconcile-rejected", action="store_true")
    assess_streetview.add_argument("--refresh-guards", action="store_true")
    assess_streetview.set_defaults(func=command_assess_v1_11)

    assess_streetview_batch = subparsers.add_parser(
        "assess-v1-11-batch",
        help="run a resumable V1.11 assessment batch from one fetched manifest",
    )
    assess_streetview_batch.add_argument("--root", default=ROOT)
    assess_streetview_batch.add_argument("--manifest", required=True)
    assess_streetview_batch.add_argument("--live", action="store_true")
    assess_streetview_batch.add_argument("--confirm-gemini-v1-11", action="store_true")
    assess_streetview_batch.set_defaults(func=command_assess_v1_11_batch)

    evaluate_streetview = subparsers.add_parser(
        "evaluate-v1-11",
        help="score frozen V1.11 predictions only after explicit reference unblinding",
    )
    evaluate_streetview.add_argument("--root", default=ROOT)
    evaluate_streetview.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    evaluate_streetview.add_argument("--reference-sha256", required=True)
    evaluate_streetview.add_argument(
        "--prediction-directory",
        default=ROOT / "data" / "model_outputs" / "quarantine" / "v1.11" / "streetview",
    )
    evaluate_streetview.add_argument("--repeat-directory", action="append", default=[])
    evaluate_streetview.add_argument(
        "--output", default=ROOT / "outputs" / "streetview_v1_11_evaluation.json"
    )
    evaluate_streetview.set_defaults(func=command_evaluate_v1_11)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
