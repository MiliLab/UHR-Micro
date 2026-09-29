from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import CHOICE_TASKS, COUNT_TASKS, SEGMENTATION_TASKS, SINGLE_BOX_TASKS, TASK_TYPES
from .geometry import transform_boxes_to_source
from .gt import GroundTruthSample
from .io import sha256_sources

from .parsing_v2 import parse_boxes, parse_choice, parse_count


PARSER_ID = "uhr-bench-v4-final-answer-extractor-v2"


def parser_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    # Freeze only answer-extraction semantics.  CLI orchestration and reporting
    # may add registry compatibility without changing the parser identity.
    return sha256_sources([root / "parsing.py", root / "evaluator.py"])


def _base_parsed(
    sample: GroundTruthSample,
    raw: dict[str, Any] | None,
    *,
    run_id: str,
    model_registry_id: str,
    phase0_contract_sha256: str,
    benchmark_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "uhr-bench-v4-parsed-prediction-1",
        "run_id": run_id,
        "phase0_contract_sha256": phase0_contract_sha256,
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "qa_relpath": sample.qa_relpath,
        "task_key": sample.task_key,
        "model_registry_id": model_registry_id,
        "parser_id": PARSER_ID,
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
    sample: GroundTruthSample,
    raw: dict[str, Any] | None,
    *,
    run_id: str,
    model_registry_id: str,
    phase0_contract_sha256: str,
    benchmark_manifest_sha256: str,
) -> dict[str, Any]:
    out = _base_parsed(
        sample,
        raw,
        run_id=run_id,
        model_registry_id=model_registry_id,
        phase0_contract_sha256=phase0_contract_sha256,
        benchmark_manifest_sha256=benchmark_manifest_sha256,
    )
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
    if str(raw.get("runtime_status") or "") != "success":
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
    boxes_to_transform = boxes[:1] if sample.task_key in SINGLE_BOX_TASKS else boxes
    transformed, protocol_status, transform_diag = transform_boxes_to_source(
        boxes_to_transform,
        str(raw.get("requested_coordinate_protocol") or ""),
        dict(raw.get("source_image_size") or {}),
        raw.get("model_input_size"),
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
    out["diagnostics"]["ignored_extra_box_count"] = max(0, len(boxes) - 1) if sample.task_key in SINGLE_BOX_TASKS else 0
    return out
