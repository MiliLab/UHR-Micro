from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .constants import CHOICE_TASKS, COUNT_TASKS, MULTI_BOX_TASKS, SEGMENTATION_TASKS, SINGLE_BOX_TASKS, TASK_TYPES
from .geometry import transform_boxes_to_source
from .gt import GroundTruthSample, decode_coco_rle, gt_segmentation_mask
from .io import sha256_sources
from .metrics import box_score_s_v2, mask_score_s_v2, multi_box_score_v2, relative_count_score
from .parsing import parse_boxes, parse_choice, parse_count


def parser_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    return sha256_sources(root.glob("*.py"))


def _base_parsed(
    sample: GroundTruthSample, raw: dict[str, Any] | None, *, run_id: str,
    model_registry_id: str, phase0_contract_sha256: str, benchmark_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "uhr-bench-v4-parsed-prediction-1",
        "run_id": run_id,
        "phase0_contract_sha256": phase0_contract_sha256,
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "qa_relpath": sample.qa_relpath,
        "task_key": sample.task_key,
        "model_registry_id": model_registry_id,
        "parser_id": "uhr-bench-v4-strict-final-parser-v1",
        "parser_source_sha256": parser_source_sha256(),
        "parse_status": "not_parsed",
        "protocol_status": "not_applicable" if sample.task_key in COUNT_TASKS | CHOICE_TASKS else "not_checked",
        "prediction_type": TASK_TYPES[sample.task_key],
        "parsed_count": None,
        "parsed_choice": None,
        "raw_boxes": [],
        "source_pixel_boxes": [],
        "diagnostics": {
            "raw_response_was_not_parsed": True,
            "output_format_status": "not_checked",
            "runtime_status": "not_attempted" if raw is None else str(raw.get("runtime_status") or ""),
        },
    }


def parse_prediction(
    sample: GroundTruthSample, raw: dict[str, Any] | None, *, run_id: str,
    model_registry_id: str, phase0_contract_sha256: str, benchmark_manifest_sha256: str,
) -> dict[str, Any]:
    out = _base_parsed(sample, raw, run_id=run_id, model_registry_id=model_registry_id,
                       phase0_contract_sha256=phase0_contract_sha256,
                       benchmark_manifest_sha256=benchmark_manifest_sha256)
    if raw is None:
        return out
    if str(raw.get("run_id") or "") != run_id or str(raw.get("model_registry_id") or "") != model_registry_id:
        raise ValueError(f"run/model identity mismatch: {sample.qa_relpath}")
    if str(raw.get("phase0_contract_sha256") or "") != phase0_contract_sha256:
        raise ValueError(f"phase0 contract mismatch: {sample.qa_relpath}")
    if str(raw.get("benchmark_manifest_sha256") or "") != benchmark_manifest_sha256:
        raise ValueError(f"benchmark manifest mismatch: {sample.qa_relpath}")
    if str(raw.get("task_key") or "") != sample.task_key:
        raise ValueError(f"task mismatch: {sample.qa_relpath}")
    if raw.get("source_image_size") != {"width_px": sample.image_width_px, "height_px": sample.image_height_px}:
        raise ValueError(f"source image size mismatch: {sample.qa_relpath}")
    runtime = str(raw.get("runtime_status") or "")
    if runtime != "success":
        return out
    final_response = raw.get("final_response")
    if sample.task_key in COUNT_TASKS:
        value, status, diagnostics = parse_count(final_response)
        out.update(parse_status=status, parsed_count=value)
        out["diagnostics"].update(diagnostics, output_format_status="ok" if status == "ok" else "invalid")
        return out
    if sample.task_key in CHOICE_TASKS:
        value, status, diagnostics = parse_choice(final_response)
        out.update(parse_status=status, parsed_choice=value)
        out["diagnostics"].update(diagnostics, output_format_status="ok" if status == "ok" else "invalid")
        return out
    boxes, status, diagnostics = parse_boxes(final_response)
    out["raw_boxes"] = boxes
    out["diagnostics"].update(diagnostics)
    if status != "ok" or (sample.task_key in SINGLE_BOX_TASKS and not boxes):
        out.update(parse_status="syntax_failure", protocol_status="not_checked")
        out["diagnostics"]["output_format_status"] = "invalid"
        return out
    # The frozen single-box contract scores only the first box.  Extra boxes
    # are retained in raw_boxes for compliance diagnostics, but must never
    # invalidate or otherwise change the first box's coordinate transform.
    boxes_to_transform = boxes[:1] if sample.task_key in SINGLE_BOX_TASKS else boxes
    transformed, protocol_status, transform_diag = transform_boxes_to_source(
        boxes_to_transform, str(raw.get("requested_coordinate_protocol") or ""),
        dict(raw.get("source_image_size") or {}), raw.get("model_input_size"),
        dict(raw.get("diagnostics") or {}),
    )
    out.update(parse_status="ok", protocol_status=protocol_status, source_pixel_boxes=transformed)
    out["diagnostics"].update(transform_diag)
    gt_box_len = None
    if sample.task_key not in SEGMENTATION_TASKS and sample.answer_value:
        gt_box_len = len(sample.answer_value[0])
    shape_mismatches = sum(1 for box in boxes if gt_box_len and len(box) != gt_box_len)
    if sample.task_key in SINGLE_BOX_TASKS and len(boxes) > 1:
        out["diagnostics"]["output_format_status"] = "multi_prediction_noncompliant_first_scored"
    elif shape_mismatches:
        out["diagnostics"]["output_format_status"] = "box_shape_mismatch_scored_geometrically"
    else:
        out["diagnostics"]["output_format_status"] = "ok"
    out["diagnostics"]["box_shape_mismatch_count"] = shape_mismatches
    out["diagnostics"]["raw_box_count"] = len(boxes)
    out["diagnostics"]["ignored_extra_box_count"] = (
        max(0, len(boxes) - 1) if sample.task_key in SINGLE_BOX_TASKS else 0
    )
    return out


def _metric_id(task: str, multi_gate: str) -> str:
    if task in COUNT_TASKS:
        return "relative_count_score_v2"
    if task in CHOICE_TASKS:
        return "choice_accuracy_v2_strict_final"
    if task in SEGMENTATION_TASKS:
        return "mask_score_s_v2_fixed_denominator"
    if task in SINGLE_BOX_TASKS:
        return "box_score_s_v2_polygon_first_final"
    return f"hungarian_soft_f1_box_score_s_v2_tp_at_{multi_gate}"


def score_sample(
    sample: GroundTruthSample, parsed: dict[str, Any], raw: dict[str, Any] | None,
    mask_row: dict[str, Any] | None, *, run_id: str, model_registry_id: str,
    phase0_contract_sha256: str, benchmark_manifest_sha256: str,
    metric_spec_sha256: str, multi_gate: str, sam2_contract_sha256: str,
) -> dict[str, Any]:
    runtime_status = "not_attempted" if raw is None else str(raw.get("runtime_status") or "")
    parse_status, protocol_status = parsed["parse_status"], parsed["protocol_status"]
    valid = runtime_status == "success" and parse_status == "ok" and protocol_status in {"ok", "not_applicable"}
    score = 0.0
    diagnostics: dict[str, Any] = {
        "split_position": sample.split_position,
        "output_format_status": parsed["diagnostics"].get("output_format_status"),
    }
    if sample.task_key in MULTI_BOX_TASKS:
        diagnostics.update(
            n_pred=len(parsed.get("source_pixel_boxes") or []) if valid else 0,
            n_gt=len(sample.answer_value),
            valid_matches=0,
            matched_quality_sum=0.0,
            pairs=[],
            multi_gate=multi_gate,
        )
    if valid and sample.task_key in COUNT_TASKS:
        score = relative_count_score(parsed["parsed_count"], int(sample.answer_value))
        pred_count, gt_count = int(parsed["parsed_count"]), int(sample.answer_value)
        absolute_error = abs(pred_count - gt_count)
        diagnostics.update(
            pred_count=pred_count, gt_count=gt_count,
            count_exact_match=int(pred_count == gt_count),
            count_absolute_error=absolute_error,
            count_relative_error=(absolute_error / gt_count if gt_count > 0 else None),
        )
    elif valid and sample.task_key in CHOICE_TASKS:
        score = 1.0 if parsed["parsed_choice"] == sample.answer_value else 0.0
    elif valid and sample.task_key in SINGLE_BOX_TASKS - SEGMENTATION_TASKS:
        boxes = parsed["source_pixel_boxes"]
        score = box_score_s_v2(boxes[0], sample.answer_value[0]) if boxes else 0.0
        diagnostics.update(n_pred=len(boxes), n_gt=1, first_prediction_scored=True)
    elif valid and sample.task_key in MULTI_BOX_TASKS:
        result = multi_box_score_v2(parsed["source_pixel_boxes"], sample.answer_value, gate=multi_gate)
        score = float(result.pop("score"))
        diagnostics.update(result, n_pred=len(parsed["source_pixel_boxes"]), n_gt=len(sample.answer_value), multi_gate=multi_gate)
    elif valid and sample.task_key in SEGMENTATION_TASKS:
        valid = False
        if mask_row is not None:
            if str(mask_row.get("qa_relpath") or "") != sample.qa_relpath:
                raise ValueError(f"SAM2 QA mismatch: {sample.qa_relpath}")
            if str(mask_row.get("run_id") or "") != run_id:
                raise ValueError(f"SAM2 run mismatch: {sample.qa_relpath}")
            if str(mask_row.get("phase0_contract_sha256") or "") != phase0_contract_sha256:
                raise ValueError(f"SAM2 phase0 mismatch: {sample.qa_relpath}")
            if str(mask_row.get("benchmark_manifest_sha256") or "") != benchmark_manifest_sha256:
                raise ValueError(f"SAM2 manifest mismatch: {sample.qa_relpath}")
            if str(mask_row.get("sam2_contract_sha256") or "") != sam2_contract_sha256:
                raise ValueError(f"SAM2 contract mismatch: {sample.qa_relpath}")
            if str(mask_row.get("task_key") or "") != sample.task_key or str(mask_row.get("image_sha256") or "") != sample.image_sha256:
                raise ValueError(f"SAM2 task/image identity mismatch: {sample.qa_relpath}")
            boxes = parsed.get("source_pixel_boxes") or []
            expected_bbox_sha = hashlib.sha256(json.dumps(
                boxes[0] if boxes else None, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            if str(mask_row.get("bbox_sha256") or "") != expected_bbox_sha:
                raise ValueError(f"SAM2 bbox cache key mismatch: {sample.qa_relpath}")
            if mask_row.get("status") == "success":
                if mask_row.get("mask_size_hw") != [sample.image_height_px, sample.image_width_px]:
                    raise ValueError(f"SAM2 mask canvas mismatch: {sample.qa_relpath}")
                pred = decode_coco_rle(str(mask_row.get("pred_mask_rle") or ""), sample.image_height_px, sample.image_width_px)
                gt = gt_segmentation_mask(sample)
                if pred is not None and gt is not None:
                    score, miou = mask_score_s_v2(gt, pred)
                    diagnostics.update(mask_iou=miou, sam2_status="success")
                    valid = True
        if not valid:
            diagnostics["sam2_status"] = "missing_or_failure"
    return {
        "schema_version": "uhr-bench-v4-metric-record-1",
        "run_id": run_id,
        "phase0_contract_sha256": phase0_contract_sha256,
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "qa_relpath": sample.qa_relpath,
        "task_key": sample.task_key,
        "model_registry_id": model_registry_id,
        "metric_id": _metric_id(sample.task_key, multi_gate),
        "metric_spec_sha256": metric_spec_sha256,
        "fixed_denominator_member": True,
        "runtime_status": runtime_status,
        "parse_status": parse_status,
        "protocol_status": protocol_status,
        "valid_prediction": bool(valid),
        "score": float(max(0.0, min(1.0, score))),
        "diagnostics": diagnostics,
    }


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_task[str(row["task_key"])].append(row)
    tasks: dict[str, Any] = {}
    for task in sorted(by_task):
        rows = by_task[task]
        runtime = Counter(str(row["runtime_status"]) for row in rows)
        n = len(rows)
        attempted = n - runtime.get("not_attempted", 0)
        parse_ok = sum(row["parse_status"] == "ok" for row in rows)
        protocol_ok = sum(row["protocol_status"] in {"ok", "not_applicable"} for row in rows)
        valid = sum(bool(row["valid_prediction"]) for row in rows)
        score_sum = sum(float(row["score"]) for row in rows)
        tasks[task] = {
            "n_gt": n,
            "attempted": attempted,
            "response_ok": runtime.get("success", 0),
            "parse_ok": parse_ok,
            "protocol_ok": protocol_ok,
            "scored_valid": valid,
            "invalid": n - valid,
            "missing": runtime.get("not_attempted", 0),
            "runtime_status_counts": dict(sorted(runtime.items())),
            "score_sum": score_sum,
            "score_mean_fixed_denominator": score_sum / n if n else 0.0,
        }
        if task in SEGMENTATION_TASKS:
            miou_sum = sum(float(row.get("diagnostics", {}).get("mask_iou") or 0.0) for row in rows)
            tasks[task]["mask_iou_sum"] = miou_sum
            tasks[task]["mask_iou_mean_fixed_denominator"] = miou_sum / n if n else 0.0
        if task in COUNT_TASKS:
            exact_sum = sum(int(row.get("diagnostics", {}).get("count_exact_match") or 0) for row in rows)
            valid_absolute_errors = [
                float(row["diagnostics"]["count_absolute_error"])
                for row in rows if row.get("diagnostics", {}).get("count_absolute_error") is not None
            ]
            tasks[task]["count_exact_match_sum"] = exact_sum
            tasks[task]["count_exact_match_fixed_denominator"] = exact_sum / n if n else 0.0
            tasks[task]["count_mae_valid_predictions_only"] = (
                sum(valid_absolute_errors) / len(valid_absolute_errors) if valid_absolute_errors else None
            )
        if task in MULTI_BOX_TASKS:
            tp = sum(int(row.get("diagnostics", {}).get("valid_matches") or 0) for row in rows)
            n_pred = sum(int(row.get("diagnostics", {}).get("n_pred") or 0) for row in rows)
            n_gt_instances = sum(int(row.get("diagnostics", {}).get("n_gt") or 0) for row in rows)
            fp, fn = n_pred - tp, n_gt_instances - tp
            tasks[task].update({
                "tp_instances": tp,
                "fp_instances": fp,
                "fn_instances": fn,
                "pred_instances": n_pred,
                "gt_instances": n_gt_instances,
                "hard_precision": tp / n_pred if n_pred else 0.0,
                "hard_recall": tp / n_gt_instances if n_gt_instances else 0.0,
                "hard_f1": 2 * tp / (n_pred + n_gt_instances) if n_pred + n_gt_instances else 0.0,
                "matching_gate": rows[0].get("diagnostics", {}).get("multi_gate") if rows else None,
            })
    family_values: dict[str, list[float]] = defaultdict(list)
    for task, entry in tasks.items():
        family_values[task.split("/", 1)[0]].append(entry["score_mean_fixed_denominator"])
    family_macro = {key: sum(values) / len(values) for key, values in sorted(family_values.items())}
    task_values = [tasks[task]["score_mean_fixed_denominator"] for task in sorted(tasks)]
    return {
        "task_count": len(tasks),
        "sample_count": len(records),
        "tasks": tasks,
        "family_macro": family_macro,
        "overall_macro_16_tasks": sum(task_values) / len(task_values) if len(task_values) == 16 else None,
        "aggregation_uses_unrounded_values": True,
    }
