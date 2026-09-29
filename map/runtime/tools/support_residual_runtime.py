#!/usr/bin/env python3
"""Adapters for the Validation-only SSRR support-residual recurrence.

The adapters preserve MAP public state while exposing it to the generic
source-registered deficit and support-coverage executors.  They never read a
benchmark task key, reference answer, annotation, metric, or score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FINAL_HINTS = {
    "SCALAR": "Return exactly `Final answer: N`, where N is one non-negative integer.",
    "OPTION": "Return exactly `Final answer: X`, where X is one valid option letter.",
    "SINGLE_GEOMETRY": "Return exactly one source-image base1000 geometry after `Final answer:`.",
    "GEOMETRY_SET": "Return all source-image base1000 geometries after `Final answer:`, or null when unsupported.",
}

TASK_ROUTES = {
    "SCALAR": "count",
    "OPTION": "choice",
    "SINGLE_GEOMETRY": "single_bbox",
    "GEOMETRY_SET": "multi_bbox",
    "MASK_SUPPORT": "segmentation",
}


def observer_template_key(row: dict[str, Any]) -> str:
    """Compile the existing MAP observer interface from public evidence state.

    This deliberately mirrors the MAP typed-observer contract.  It does
    not inspect a benchmark task name: answer shape, support extent,
    composition, and required resolution are already public physical state.
    """

    state = row["evidence_state"]
    schema = str(state["answer_schema"])
    extent = str(state["support_extent"])
    composition = str(state["evidence_composition"])
    resolution = str(state["resolution_requirement"])
    initial = str(state["initial_operator"])
    explicit = int(state.get("explicit_region_count") or 0) > 0
    if schema == "SCALAR":
        if composition == "NUMERIC_DIFFERENCE":
            return "cross_region_compare"
        if explicit:
            return "region_count"
        return "global_count" if extent == "FULL_SCENE" else "conditional_count"
    if schema == "OPTION":
        if resolution == "FINE_SUBTYPE":
            return "fine_grained_recognition"
        return "classification" if initial == "ACQUIRE_TARGET_SUPPORT" else "relationship"
    if schema == "SINGLE_GEOMETRY":
        return "basic_grounding"
    if schema == "GEOMETRY_SET":
        if extent == "FULL_SCENE":
            return "global_detection"
        return "region_detection" if explicit else "multi_condition_retrieval"
    if schema == "MASK_SUPPORT":
        return "segmentation"
    raise ValueError(f"unknown public answer schema: {schema}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def keyed(rows: Iterable[dict[str, Any]], key: str = "sample_id") -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity or identity in output:
            raise ValueError(f"missing or duplicate {key}: {identity!r}")
        output[identity] = row
    return output


def adapt(args: argparse.Namespace) -> int:
    manifest = read_jsonl(args.manifest)
    closure = keyed(read_jsonl(args.closure))
    canvases = keyed(read_jsonl(args.source_canvases))
    open_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    for source in manifest:
        sid = str(source["sample_id"])
        if closure[sid]["closure_status"] != "OPEN":
            continue
        row = copy.deepcopy(source)
        schema = str(row["evidence_state"]["answer_schema"])
        row["special_operator"] = "NONE"
        row["final_format_hint"] = FINAL_HINTS[schema]
        open_rows.append(row)
        canvas = canvases[sid]
        overlay_rows.append(
            {
                "sample_id": sid,
                "overlay_image_path": canvas["source_image_path"],
                "overlay_image_sha256": canvas["source_canvas_sha256"],
                "method_generated_support_canvas": bool(
                    canvas["source_canvas_is_method_generated"]
                ),
                "dataset_annotation_used": False,
            }
        )
    audit = {
        "schema_version": "ssrr-support-residual-adapter-audit-1",
        "input_rows": len(manifest),
        "map_open_rows": len(open_rows),
        "map_closed_rows_excluded": len(manifest) - len(open_rows),
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
        "passed": len(open_rows) > 0,
    }
    write_jsonl(args.out_manifest, open_rows)
    write_jsonl(args.out_overlays, overlay_rows)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"support residual adapter failed: {audit}")
    return 0


def sanitize_controller(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    output: list[dict[str, Any]] = []
    forbidden_hits: list[str] = []
    for source in rows:
        prompt = str(source["model_input_prompt"])
        lowered = prompt.lower()
        for term in ("task_key", "qa_relpath", "ground truth", "reference answer"):
            if term in lowered:
                forbidden_hits.append(f"{source['task_id']}:{term}")
        output.append(
            {
                "task_id": source["task_id"],
                "sample_id": source["sample_id"],
                "model_image_path": source["model_image_path"],
                "model_input_prompt": prompt,
                "observer_generation_profile": "STANDARD",
                "stage_name": source["stage_name"],
                "image_abs_path": source["image_abs_path"],
                "image_sha256": source["image_sha256"],
            }
        )
    audit = {
        "schema_version": "ssrr-deficit-controller-boundary-audit-1",
        "rows": len(output),
        "unique_task_ids": len({row["task_id"] for row in output}),
        "forbidden_prompt_hits": forbidden_hits,
        "allowlisted_fields_only": True,
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        bool(output)
        and len(output) == audit["unique_task_ids"]
        and not forbidden_hits
    )
    write_jsonl(args.out, output)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"controller boundary failed: {audit}")
    return 0


def sanitize_generic(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    allowed = {
        "task_id", "sample_id", "model_image_path", "model_input_prompt",
        "observer_generation_profile", "stage_name", "image_abs_path", "image_sha256",
        "roi_image_sha256", "crop_image_sha256", "min_pixels", "max_pixels",
    }
    output = [{key: value for key, value in row.items() if key in allowed} for row in rows]
    ids = [str(row.get("task_id") or "") for row in output]
    audit = {
        "schema_version": "ssrr-generic-model-boundary-audit-1",
        "rows": len(output),
        "unique_task_ids": len(set(ids)),
        "allowlisted_fields": sorted(allowed),
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
        "passed": bool(output) and "" not in ids and len(ids) == len(set(ids)),
    }
    write_jsonl(args.out, output)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"generic boundary failed: {audit}")
    return 0


def enforce_budget(args: argparse.Namespace) -> int:
    decisions = keyed(read_jsonl(args.decisions))
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.rois):
        grouped[str(row["sample_id"])].append(row)
    output_rois: list[dict[str, Any]] = []
    output_decisions: list[dict[str, Any]] = []
    rejected = 0
    admitted = 0
    for sid, source in decisions.items():
        row = copy.deepcopy(source)
        tiles = sorted(grouped.get(sid, []), key=lambda item: int(item["roi_index"]))
        if row.get("admitted") and len(tiles) > args.max_views:
            row["admitted"] = False
            row["reason"] = "complete_support_exceeds_fixed_view_budget"
            row["required_complete_views"] = len(tiles)
            row["fixed_view_budget"] = args.max_views
            rejected += 1
        elif row.get("admitted"):
            if not tiles:
                raise ValueError(f"admitted support has no tiled views: {sid}")
            output_rois.extend(tiles)
            admitted += 1
        output_decisions.append(row)
    audit = {
        "schema_version": "ssrr-support-view-budget-audit-1",
        "decision_rows": len(output_decisions),
        "admitted_supports_within_budget": admitted,
        "supports_rejected_over_budget": rejected,
        "retained_complete_tiles": len(output_rois),
        "maximum_views_per_sample": max(
            (len(value) for sid, value in grouped.items() if decisions[sid].get("admitted") and len(value) <= args.max_views),
            default=0,
        ),
        "fixed_view_budget": args.max_views,
        "partial_coverage_admitted": False,
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        len(output_decisions) == len(decisions)
        and admitted > 0
        and audit["maximum_views_per_sample"] <= args.max_views
    )
    write_jsonl(args.out_rois, output_rois)
    write_jsonl(args.out_decisions, output_decisions)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"support budget gate failed: {audit}")
    return 0


def compile_observer_profiles(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    metadata_updates = 0
    for source in rows:
        row = copy.deepcopy(source)
        key = observer_template_key(row)
        route = TASK_ROUTES[str(row["evidence_state"]["answer_schema"])]
        row["stage3_template_key"] = key
        row["task_route"] = route
        row["observer_profile_compiled_from_public_evidence_state"] = True
        if args.crop_dir is not None:
            metadata_path = args.crop_dir / str(row["roi_id"]) / "roi_meta.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"missing ROI metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["stage3_template_key"] = key
            metadata["task_route"] = route
            metadata["observer_profile_compiled_from_public_evidence_state"] = True
            temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(metadata_path)
            metadata_updates += 1
        output.append(row)
        counts[key] += 1
    audit = {
        "schema_version": "ssrr-public-observer-profile-audit-1",
        "rows": len(output),
        "unique_roi_ids": len({str(row.get("roi_id") or "") for row in output}),
        "template_counts": dict(sorted(counts.items())),
        "roi_metadata_updates": metadata_updates,
        "compiler_inputs": [
            "answer_schema",
            "support_extent",
            "evidence_composition",
            "resolution_requirement",
            "initial_operator",
            "explicit_region_count",
        ],
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        bool(output)
        and len(output) == audit["unique_roi_ids"]
        and "" not in {str(row.get("roi_id") or "") for row in output}
        and (args.crop_dir is None or metadata_updates == len(output))
    )
    write_jsonl(args.out, output)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"observer profile compilation failed: {audit}")
    return 0


def rehydrate_observer_results(args: argparse.Namespace) -> int:
    """Restore non-model ROI registration fields after the safe runner boundary."""

    tasks = keyed(read_jsonl(args.tasks), "task_id")
    results = keyed(read_jsonl(args.results), "task_id")
    if set(tasks) != set(results):
        raise ValueError("observer task/result IDs differ")
    output: list[dict[str, Any]] = []
    mismatches: list[str] = []
    errors: list[str] = []
    for task_id, source in tasks.items():
        result = results[task_id]
        for field in ("sample_id", "model_image_path", "model_input_prompt"):
            if str(source.get(field)) != str(result.get(field)):
                mismatches.append(f"{task_id}:{field}")
        if result.get("error"):
            errors.append(task_id)
        row = copy.deepcopy(source)
        row.update(copy.deepcopy(result))
        output.append(row)
    audit = {
        "schema_version": "ssrr-observer-result-rehydration-audit-1",
        "task_rows": len(tasks),
        "result_rows": len(results),
        "output_rows": len(output),
        "model_boundary_mismatches": mismatches,
        "infrastructure_error_rows": errors,
        "registration_fields_restored": [
            "roi_id",
            "roi_abs_xyxy",
            "roi_full1000_xyxy",
            "original_width_px",
            "original_height_px",
            "task_route",
        ],
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        len(tasks) == len(results) == len(output)
        and not mismatches
        and not errors
    )
    write_jsonl(args.out, output)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"observer result rehydration failed: {audit}")
    return 0


def enforce_observation_validity(args: argparse.Namespace) -> int:
    """Make malformed geometry an explicit no-op while retaining raw evidence."""

    rows = read_jsonl(args.input)
    output: list[dict[str, Any]] = []
    invalid: Counter[str] = Counter()
    for source in rows:
        row = copy.deepcopy(source)
        if not bool(row.get("stage3_parse_ok", True)):
            reason = str(row.get("stage3_parse_error") or "unknown_parse_error")
            invalid[reason] += 1
            row["invalid_observation_raw_response"] = row.get("model_response")
            row["model_response"] = None
            row["stage3_mapped_response_full1000"] = None
            row["stage3_mapped_coord_lists_full1000"] = []
            row["feedback_mapped_response"] = None
            row["observation_update_admitted"] = False
        else:
            row["observation_update_admitted"] = True
        output.append(row)
    audit = {
        "schema_version": "ssrr-observation-validity-audit-1",
        "rows": len(output),
        "unique_task_ids": len({str(row.get("task_id") or "") for row in output}),
        "unique_sample_ids": len({str(row.get("sample_id") or "") for row in output}),
        "invalid_observations_noop": sum(invalid.values()),
        "invalid_reason_counts": dict(sorted(invalid.items())),
        "raw_invalid_responses_retained": sum(
            "invalid_observation_raw_response" in row for row in output
        ),
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        bool(output)
        and len(output) == audit["unique_task_ids"]
        and audit["invalid_observations_noop"]
        == audit["raw_invalid_responses_retained"]
    )
    write_jsonl(args.out, output)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"observation validity gate failed: {audit}")
    return 0


def prepare_terminal_base(args: argparse.Namespace) -> int:
    """Attach the public output contract to unchanged MAP composer tasks."""

    base = keyed(read_jsonl(args.base))
    manifest = keyed(read_jsonl(args.manifest))
    if not set(manifest).issubset(base):
        raise ValueError("support manifest is not a subset of MAP composer tasks")
    output: list[dict[str, Any]] = []
    for sid in manifest:
        row = copy.deepcopy(base[sid])
        hint = str(manifest[sid].get("final_format_hint") or "").strip()
        if not hint:
            raise ValueError(f"missing final format hint: {sid}")
        row["final_format_hint"] = hint
        row["terminal_contract_compiled_from_public_answer_schema"] = True
        output.append(row)
    audit = {
        "schema_version": "ssrr-terminal-base-adapter-audit-1",
        "base_rows": len(base),
        "manifest_rows": len(manifest),
        "output_rows": len(output),
        "base_prompt_bytes_changed": sum(
            str(base[row["sample_id"]]["model_input_prompt"])
            != str(row["model_input_prompt"])
            for row in output
        ),
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        len(output) == len(manifest)
        and audit["base_prompt_bytes_changed"] == 0
    )
    write_jsonl(args.out, output)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"terminal base adapter failed: {audit}")
    return 0


def support_residual_is_admissible(row: dict[str, Any]) -> tuple[bool, str]:
    """Return whether one support-owned closure update is physically defined."""

    state = row["evidence_state"]
    schema = str(state["answer_schema"])
    composition = str(state["evidence_composition"])
    if schema == "SINGLE_GEOMETRY":
        return False, "single_target_identity_has_no_additive_support_residual"
    if composition == "NUMERIC_DIFFERENCE":
        return False, "paired_operands_require_separate_owned_residuals"
    if composition not in {"AGGREGATION", "SPATIAL_CONFIGURATION"}:
        return False, "composition_has_no_defined_support_owned_update"
    return True, "support_owned_update_defined"


def filter_controller_admissible(args: argparse.Namespace) -> int:
    """Apply the physical support algebra before spending controller calls.

    The deficit controller has its own generic eligibility gate.  This second
    gate is deliberately applied to its public eligible manifest before model
    inference so SINGLE_GEOMETRY and paired NUMERIC_DIFFERENCE states do not
    consume calls for an update that the current one-support recurrence cannot
    represent correctly.
    """

    manifest = keyed(read_jsonl(args.manifest))
    tasks = keyed(read_jsonl(args.tasks), "sample_id")
    if set(manifest) != set(tasks):
        raise ValueError("controller manifest/task sample IDs differ")
    admitted: list[dict[str, Any]] = []
    admitted_tasks: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for sid, source in manifest.items():
        keep, reason = support_residual_is_admissible(source)
        reasons[reason] += 1
        if keep:
            admitted.append(copy.deepcopy(source))
            admitted_tasks.append(copy.deepcopy(tasks[sid]))
    audit = {
        "schema_version": "ssrr-v2.1-precontroller-support-admissibility-1",
        "input_rows": len(manifest),
        "admissible_rows": len(admitted),
        "excluded_rows": len(manifest) - len(admitted),
        "reason_counts": dict(sorted(reasons.items())),
        "selection_inputs": ["answer_schema", "evidence_composition"],
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        bool(admitted)
        and len(admitted) == len(admitted_tasks)
        and len(admitted) < len(manifest)
    )
    write_jsonl(args.out_manifest, admitted)
    write_jsonl(args.out_tasks, admitted_tasks)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"pre-controller support gate failed: {audit}")
    return 0


def enforce_support_admissibility(args: argparse.Namespace) -> int:
    manifest = keyed(read_jsonl(args.manifest))
    states = keyed(read_jsonl(args.states))
    terminal = keyed(read_jsonl(args.terminal_results))
    controller = keyed(read_jsonl(args.controller_tasks))
    mapped_rows = read_jsonl(args.mapped)
    reasons: Counter[str] = Counter()
    allowed: set[str] = set()
    for sid, row in manifest.items():
        admitted, reason = support_residual_is_admissible(row)
        reasons[reason] += 1
        if admitted:
            allowed.add(sid)
    output_states: list[dict[str, Any]] = []
    deactivated = 0
    for sid, source in states.items():
        row = copy.deepcopy(source)
        if sid not in allowed and row.get("admitted"):
            row["admitted"] = False
            row["reason"] = "support_residual_physically_inadmissible"
            deactivated += 1
        output_states.append(row)
    active = {sid for sid, row in keyed(output_states).items() if row.get("admitted")}
    if not active.issubset(allowed):
        raise RuntimeError("inadmissible support update remained active")
    output_terminal = [copy.deepcopy(terminal[sid]) for sid in terminal if sid in active]
    output_controller = [copy.deepcopy(controller[sid]) for sid in controller if sid in allowed]
    output_mapped = [
        copy.deepcopy(row)
        for row in mapped_rows
        if str(row["sample_id"]) in allowed
    ]
    audit = {
        "schema_version": "ssrr-support-residual-admissibility-audit-1",
        "manifest_rows": len(manifest),
        "admissible_before_controller": len(allowed),
        "inadmissible_before_controller": len(manifest) - len(allowed),
        "admissibility_reason_counts": dict(sorted(reasons.items())),
        "previous_active_updates": len(terminal),
        "deactivated_updates": deactivated,
        "retained_active_updates": len(active),
        "retained_controller_calls": len(output_controller),
        "retained_observation_calls": len(output_mapped),
        "task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        len(states) == len(manifest)
        and len(output_terminal) == len(active)
        and set(terminal).issuperset(active)
    )
    write_jsonl(args.out_states, output_states)
    write_jsonl(args.out_terminal_results, output_terminal)
    write_jsonl(args.out_controller_tasks, output_controller)
    write_jsonl(args.out_mapped, output_mapped)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"support admissibility gate failed: {audit}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    item = commands.add_parser("adapt")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--closure", type=Path, required=True)
    item.add_argument("--source-canvases", type=Path, required=True)
    item.add_argument("--out-manifest", type=Path, required=True)
    item.add_argument("--out-overlays", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=adapt)
    for name, function in (("sanitize-controller", sanitize_controller), ("sanitize-generic", sanitize_generic)):
        item = commands.add_parser(name)
        item.add_argument("--input", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        item.add_argument("--audit", type=Path, required=True)
        item.set_defaults(func=function)
    item = commands.add_parser("enforce-budget")
    item.add_argument("--rois", type=Path, required=True)
    item.add_argument("--decisions", type=Path, required=True)
    item.add_argument("--max-views", type=int, default=4)
    item.add_argument("--out-rois", type=Path, required=True)
    item.add_argument("--out-decisions", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=enforce_budget)
    item = commands.add_parser("compile-observer-profiles")
    item.add_argument("--input", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.add_argument("--crop-dir", type=Path)
    item.set_defaults(func=compile_observer_profiles)
    item = commands.add_parser("rehydrate-observer-results")
    item.add_argument("--tasks", type=Path, required=True)
    item.add_argument("--results", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=rehydrate_observer_results)
    item = commands.add_parser("enforce-observation-validity")
    item.add_argument("--input", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=enforce_observation_validity)
    item = commands.add_parser("prepare-terminal-base")
    item.add_argument("--base", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=prepare_terminal_base)
    item = commands.add_parser("enforce-support-admissibility")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--states", type=Path, required=True)
    item.add_argument("--terminal-results", type=Path, required=True)
    item.add_argument("--controller-tasks", type=Path, required=True)
    item.add_argument("--mapped", type=Path, required=True)
    item.add_argument("--out-states", type=Path, required=True)
    item.add_argument("--out-terminal-results", type=Path, required=True)
    item.add_argument("--out-controller-tasks", type=Path, required=True)
    item.add_argument("--out-mapped", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=enforce_support_admissibility)
    item = commands.add_parser("filter-controller-admissible")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--tasks", type=Path, required=True)
    item.add_argument("--out-manifest", type=Path, required=True)
    item.add_argument("--out-tasks", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=filter_controller_admissible)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.func(arguments))
