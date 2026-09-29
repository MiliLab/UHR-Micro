#!/usr/bin/env python3
"""Materialize every SSRR v2.1 ablation from one primitive evidence state.

This command never accepts a completed prediction file.  It recomputes the
MAP closure and the scale-residual successor from public typed evidence, joins
same-run support recurrence receipts, and emits all four variants in one pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import composer_runtime as composer_runtime
import scale_residual_runtime as scale_runtime
from io_utils import keyed, read_jsonl, sha256_file, write_json, write_jsonl


VARIANTS = (
    "recurrence_disabled",
    "scale_residual_only",
    "support_residual_only",
    "ssrr_full",
)


def raw_response(row: dict[str, Any]) -> str:
    value = row.get("raw_backend_response")
    return str(value if value is not None else row.get("model_response") or "")


def grouped_rows(
    rows: Iterable[dict[str, Any]], key: str = "sample_id"
) -> dict[str, list[dict[str, Any]]]:
    output: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity:
            raise ValueError(f"grouped row lacks {key}")
        output[identity].append(row)
    return dict(output)


def deterministic_generation_errors(
    stage_rows: dict[str, Iterable[dict[str, Any]]]
) -> list[str]:
    errors: list[str] = []
    for stage, rows in stage_rows.items():
        for index, row in enumerate(rows):
            config = row.get("generation_config")
            if not isinstance(config, dict):
                continue
            if float(config.get("temperature", -1.0)) != 0.0:
                errors.append(f"{stage}:{index}:temperature")
            if float(config.get("top_p", -1.0)) != 1.0:
                errors.append(f"{stage}:{index}:top_p")
            if int(config.get("seed", -1)) != 42:
                errors.append(f"{stage}:{index}:seed")
    return errors


def add_call_accounting(
    row: dict[str, Any],
    *,
    controller_result: dict[str, Any] | None,
    observations: list[dict[str, Any]],
    terminal_result: dict[str, Any] | None,
) -> None:
    controller_attempts = (
        [composer_runtime.required_attempt(controller_result, "support_residual_controller")]
        if controller_result is not None
        else []
    )
    observation_attempts = [
        composer_runtime.required_attempt(item, "support_residual_observer")
        for item in observations
    ]
    terminal_attempts = (
        [composer_runtime.required_attempt(terminal_result, "support_residual_terminal")]
        if terminal_result is not None
        else []
    )
    logical = len(controller_attempts) + len(observation_attempts) + len(terminal_attempts)
    physical = sum(controller_attempts + observation_attempts + terminal_attempts)
    row["logical_vlm_calls"] = int(row.get("logical_vlm_calls") or 0) + logical
    accounting = row.setdefault("diagnostics", {}).setdefault("call_accounting", {})
    accounting["logical_vlm_calls"] = int(
        accounting.get("logical_vlm_calls") or 0
    ) + logical
    accounting["known_physical_vlm_attempts"] = int(
        accounting.get("known_physical_vlm_attempts") or 0
    ) + physical
    accounting["physical_vlm_attempts"] = int(
        accounting.get("physical_vlm_attempts") or 0
    ) + physical
    logical_by_stage = accounting.setdefault("logical_calls_by_stage", {})
    logical_by_stage["support_residual_controller"] = len(controller_attempts)
    logical_by_stage["support_residual_observers"] = len(observation_attempts)
    logical_by_stage["support_residual_terminal"] = len(terminal_attempts)
    attempts_by_stage = accounting.setdefault("attempt_numbers_by_stage", {})
    attempts_by_stage["support_residual_controller"] = controller_attempts
    attempts_by_stage["support_residual_observers"] = observation_attempts
    attempts_by_stage["support_residual_terminal"] = terminal_attempts
    all_attempts = controller_attempts + observation_attempts + terminal_attempts
    if all_attempts:
        row["attempt"] = max(int(row.get("attempt") or 1), max(all_attempts))


def apply_support_recurrence(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    support_state: dict[str, Any] | None,
    controller_result: dict[str, Any] | None,
    observations: list[dict[str, Any]],
    terminal_result: dict[str, Any] | None,
) -> dict[str, Any]:
    output = copy.deepcopy(row)
    admitted = bool(support_state and support_state.get("admitted"))
    if admitted != (terminal_result is not None):
        raise ValueError(
            f"support terminal/admission mismatch: {source['sample_id']}"
        )
    if controller_result is not None and controller_result.get("error"):
        raise RuntimeError("unresolved infrastructure error in support controller")
    if any(item.get("error") for item in observations):
        raise RuntimeError("unresolved infrastructure error in support observation")
    if terminal_result is not None and terminal_result.get("error"):
        raise RuntimeError("unresolved infrastructure error in support terminal")

    add_call_accounting(
        output,
        controller_result=controller_result,
        observations=observations,
        terminal_result=terminal_result,
    )
    diagnostics = output.setdefault("diagnostics", {})
    diagnostics["support_residual_state"] = copy.deepcopy(support_state)
    diagnostics["support_residual_observation_count"] = len(observations)
    diagnostics["support_residual_controller_called"] = controller_result is not None
    diagnostics["support_residual_update_admitted"] = admitted
    diagnostics["ground_truth_or_metric_used"] = False
    output.setdefault("trajectory", []).extend(
        [
            {
                "stage": "support_residual_controller",
                "called": controller_result is not None,
                "raw_response": (
                    raw_response(controller_result)
                    if controller_result is not None
                    else None
                ),
            },
            {
                "stage": "source_registered_support_residual_observations",
                "count": len(observations),
                "observations": [
                    {
                        "task_id": item.get("task_id"),
                        "source_support_base1000": item.get("roi_full1000_xyxy"),
                        "raw_response": raw_response(item),
                        "update_admitted": item.get("observation_update_admitted"),
                    }
                    for item in observations
                ],
            },
        ]
    )
    if terminal_result is not None:
        response = raw_response(terminal_result)
        output["raw_response"] = response
        output["final_response"] = response.strip()
        output["api_mode"] = terminal_result.get("api_mode")
        output["prompt_id"] = "ssrr-v2.1:support-residual-typed-update"
        prompt = str(terminal_result["model_input_prompt"])
        output["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        output["generation_config"] = copy.deepcopy(
            terminal_result["generation_config"]
        )
        size = composer_runtime.smart_resize_size(source, terminal_result)
        output["model_input_size"] = size
        output["transmitted_image_size"] = copy.deepcopy(size)
        output["endpoint_id"] = terminal_result.get("endpoint_id")
        output["request_id"] = terminal_result.get("request_id")
        output["requested_model_id"] = terminal_result.get("requested_model_id")
        diagnostics["final_response_origin"] = "support_residual_typed_update"
        output["trajectory"].append(
            {
                "stage": "support_residual_typed_update",
                "corrective_observations": len(observations),
                "raw_response": response,
            }
        )
    return output


def materialize_all(args: argparse.Namespace) -> int:
    manifest_rows = read_jsonl(args.manifest)
    native_rows = read_jsonl(args.native)
    planner_rows = read_jsonl(args.planner)
    mapped_rows = read_jsonl(args.mapped)
    state_rows = read_jsonl(args.states)
    base_closure_rows = read_jsonl(args.base_closure)
    scale_closure_rows = read_jsonl(args.scale_closure)
    composer_rows = read_jsonl(args.composer_union)
    binding_rows = read_jsonl(args.binding)
    controller_task_rows = read_jsonl(args.support_controller_tasks)
    controller_result_rows = read_jsonl(args.support_controller_results)
    support_state_rows = read_jsonl(args.support_states)
    support_terminal_rows = read_jsonl(args.support_terminal_results)
    support_mapped_rows = read_jsonl(args.support_mapped)

    manifest = keyed(manifest_rows)
    native = keyed(native_rows)
    planner = keyed(planner_rows)
    states = keyed(state_rows)
    base_closure = keyed(base_closure_rows)
    scale_closure = keyed(scale_closure_rows)
    composer_union = keyed(composer_rows)
    binding = keyed(binding_rows)
    controller_tasks = keyed(controller_task_rows)
    controller_results = keyed(controller_result_rows)
    support_states = keyed(support_state_rows)
    support_terminal = keyed(support_terminal_rows)
    local_grouped = grouped_rows(mapped_rows)
    support_grouped = grouped_rows(support_mapped_rows)
    inference_ids = set(manifest)

    if any(
        set(table) != inference_ids
        for table in (native, planner, states, base_closure, scale_closure, binding)
    ):
        raise ValueError("fixed-denominator primitive evidence join failed")
    if set(local_grouped) - inference_ids or set(support_grouped) - inference_ids:
        raise ValueError("observation references an unknown sample")
    if set(controller_tasks) != set(controller_results):
        raise ValueError("support controller task/result sets differ")
    if set(support_states) != set(controller_tasks):
        raise ValueError("support state/controller sets differ")
    active_support = {
        sid for sid, state in support_states.items() if state.get("admitted")
    }
    if set(support_terminal) != active_support:
        raise ValueError("support terminal set differs from admitted states")

    expected_base: dict[str, dict[str, Any]] = {}
    expected_scale: dict[str, dict[str, Any]] = {}
    scale_active: set[str] = set()
    for sid in inference_ids:
        local = local_grouped.get(sid, [])
        expected_base[sid] = composer_runtime.seal_closure_decision(
            composer_runtime.evidence_closure_decision(
                manifest[sid], states[sid], local
            )
        )
        if expected_base[sid] != base_closure[sid]:
            raise ValueError(f"base closure is not reproducible: {sid}")
        expected_scale[sid], activated = scale_runtime.expected_reopened_decision(
            states[sid], expected_base[sid]
        )
        if expected_scale[sid] != scale_closure[sid]:
            raise ValueError(f"scale residual decision is not reproducible: {sid}")
        if activated:
            scale_active.add(sid)
    scale_open = {
        sid
        for sid, decision in expected_scale.items()
        if decision.get("closure_status") == "OPEN"
    }
    if set(composer_union) != scale_open:
        raise ValueError("composer union differs from recomputed scale-open set")
    if scale_active & active_support:
        raise ValueError("scale and support terminal updates unexpectedly overlap")

    phase0 = json.loads(args.phase0.read_text(encoding="utf-8"))
    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    ordered_ids = sorted(
        inference_ids, key=lambda value: int(binding[value]["split_position"])
    )
    for sid in ordered_ids:
        local = local_grouped.get(sid, [])
        core_rows: dict[bool, dict[str, Any]] = {}
        for scale_enabled in (False, True):
            decision = expected_scale[sid] if scale_enabled else expected_base[sid]
            core_row = composer_runtime.build_prediction_row(
                source=manifest[sid],
                private_binding=binding[sid],
                native=native[sid],
                planner=planner[sid],
                local_rows=local,
                state=states[sid],
                terminal=(
                    composer_union.get(sid)
                    if decision.get("closure_status") == "OPEN"
                    else None
                ),
                closure_decision=decision,
                expected_closure_decision=decision,
                phase0=phase0,
                run_id=str(args.run_id),
            )
            core_row["resolved_model_id"] = (
                "method:ScaleSupportResidualRecurrenceV2.1;"
                "answering_model:Qwen3-VL-8B-Instruct"
            )
            diagnostics = core_row.setdefault("diagnostics", {})
            diagnostics["residual_state_machine_version"] = "2.1"
            diagnostics["scale_residual_enabled"] = scale_enabled
            diagnostics["scale_residual_activated"] = sid in scale_active
            diagnostics["scale_residual_certificate"] = copy.deepcopy(
                expected_scale[sid].get("scale_residual_certificate")
            )
            diagnostics["completed_prediction_input_used"] = False
            if scale_enabled and sid in scale_active:
                core_row["prompt_id"] = "ssrr-v2.1:scale-residual-joint-synthesis"
                core_row.setdefault("trajectory", []).insert(
                    -1,
                    {
                        "stage": "scale_residual_certification",
                        "logical_vlm_calls": 0,
                        "certificate": copy.deepcopy(
                            expected_scale[sid]["scale_residual_certificate"]
                        ),
                    },
                )
            core_rows[scale_enabled] = core_row

        support_kwargs = {
            "source": manifest[sid],
            "support_state": support_states.get(sid),
            "controller_result": controller_results.get(sid),
            "observations": support_grouped.get(sid, []),
            "terminal_result": support_terminal.get(sid),
        }
        rows = {
            "recurrence_disabled": copy.deepcopy(core_rows[False]),
            "scale_residual_only": copy.deepcopy(core_rows[True]),
            "support_residual_only": apply_support_recurrence(
                core_rows[False], **support_kwargs
            ),
            "ssrr_full": apply_support_recurrence(
                core_rows[True], **support_kwargs
            ),
        }
        for variant, row in rows.items():
            row["run_id"] = f"{args.run_id}_{variant}"
            row.setdefault("diagnostics", {})["support_residual_enabled"] = (
                variant in {"support_residual_only", "ssrr_full"}
            )
            outputs[variant].append(row)

    deterministic_errors = deterministic_generation_errors(
        {
            "native": native_rows,
            "planner": planner_rows,
            "local_observers": mapped_rows,
            "composer_union": composer_rows,
            "support_controller": controller_result_rows,
            "support_observers": support_mapped_rows,
            "support_terminal": support_terminal_rows,
        }
    )
    variant_audits: dict[str, dict[str, Any]] = {}
    for variant, rows in outputs.items():
        path = args.output_root / variant / "canonical_predictions.jsonl"
        write_jsonl(path, rows)
        audit = {
            "schema_version": "ssrr-v2.1-unified-materialization-audit-1",
            "variant": variant,
            "rows": len(rows),
            "unique_qas": len({row["qa_relpath"] for row in rows}),
            "completed_prediction_inputs": 0,
            "same_run_primitive_evidence_only": True,
            "single_residual_state_machine": True,
            "single_materializer_for_all_variants": True,
            "base_closures_recomputed": len(expected_base),
            "scale_residual_decisions_recomputed": len(expected_scale),
            "scale_residual_activations": len(scale_active),
            "support_controller_rows": len(controller_results),
            "support_corrective_observations": len(support_mapped_rows),
            "support_typed_updates": len(active_support),
            "deterministic_generation_errors": deterministic_errors,
            "response_or_metric_selector_used": False,
            "task_identity_used_during_inference": False,
            "ground_truth_or_metric_used": False,
            "standalone_primitive_evidence_materialization": True,
            "test_used": False,
        }
        audit["passed"] = (
            audit["rows"] == audit["unique_qas"]
            and not deterministic_errors
            and all(
                int(row.get("logical_vlm_calls") or 0)
                == int(
                    ((row.get("diagnostics") or {}).get("call_accounting") or {}).get(
                        "logical_vlm_calls", -1
                    )
                )
                for row in rows
            )
        )
        write_json(args.audit_root / f"MATERIALIZATION_{variant}.json", audit)
        if not audit["passed"]:
            raise RuntimeError(f"unified materialization failed: {audit}")
        variant_audits[variant] = audit

    joint_audit = {
        "schema_version": "ssrr-v2.1-unified-materialization-joint-audit-1",
        "variants": list(VARIANTS),
        "variant_audits": variant_audits,
        "output_sha256": {
            variant: sha256_file(
                args.output_root / variant / "canonical_predictions.jsonl"
            )
            for variant in VARIANTS
        },
        "completed_prediction_input_interface_exists": False,
        "all_variants_emitted_in_one_process": True,
        "test_used": False,
        "passed": all(value["passed"] for value in variant_audits.values()),
    }
    write_json(args.audit_root / "MATERIALIZATION_ALL.json", joint_audit)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--manifest", type=Path, required=True)
    root.add_argument("--binding", type=Path, required=True)
    root.add_argument("--native", type=Path, required=True)
    root.add_argument("--planner", type=Path, required=True)
    root.add_argument("--mapped", type=Path, required=True)
    root.add_argument("--states", type=Path, required=True)
    root.add_argument("--base-closure", type=Path, required=True)
    root.add_argument("--scale-closure", type=Path, required=True)
    root.add_argument("--composer-union", type=Path, required=True)
    root.add_argument("--support-controller-tasks", type=Path, required=True)
    root.add_argument("--support-controller-results", type=Path, required=True)
    root.add_argument("--support-states", type=Path, required=True)
    root.add_argument("--support-terminal-results", type=Path, required=True)
    root.add_argument("--support-mapped", type=Path, required=True)
    root.add_argument("--phase0", type=Path, required=True)
    root.add_argument("--run-id", required=True)
    root.add_argument("--output-root", type=Path, required=True)
    root.add_argument("--audit-root", type=Path, required=True)
    return root


if __name__ == "__main__":
    raise SystemExit(materialize_all(parser().parse_args()))
