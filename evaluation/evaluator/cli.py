from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .evaluator import aggregate_metrics, score_sample
from .gt import load_ground_truth
from .io import read_json, read_jsonl, sha256_file, unique_by_qa, write_json, write_jsonl
from .registry import load_model_registry
from .schema_validation import validation_errors

from .evaluator_v2 import PARSER_ID, parse_prediction


def _validate_schema(rows: list[dict[str, Any]], schema_path: Path, source: str) -> None:
    schema = read_json(schema_path)
    for index, row in enumerate(rows):
        errors = validation_errors(row, schema)
        if errors:
            raise ValueError(f"{source} row {index}: {errors[0]}")


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "task_key", "n_gt", "attempted", "response_ok", "parse_ok", "protocol_ok",
        "scored_valid", "invalid", "missing", "score_sum", "score_mean_fixed_denominator",
        "mask_iou_sum", "mask_iou_mean_fixed_denominator", "count_exact_match_sum",
        "count_exact_match_fixed_denominator", "count_mae_valid_predictions_only",
        "tp_instances", "fp_instances", "fn_instances", "pred_instances", "gt_instances",
        "hard_precision", "hard_recall", "hard_f1", "matching_gate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task, entry in sorted(summary["tasks"].items()):
            writer.writerow({"task_key": task, **{key: entry.get(key) for key in fields[1:]}})


def main() -> int:
    eval_root_default = Path(__file__).resolve().parents[1]
    workspace_default = eval_root_default.parents[1]
    parser = argparse.ArgumentParser(description="UHR-Bench v4 evaluator with final-answer extractor v2")
    parser.add_argument("--split", choices=["Development", "Validation", "Test"], required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sam2-masks", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-registry-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=eval_root_default)
    parser.add_argument("--dataset-root", type=Path, default=workspace_default / "UHR-Bench-v4")
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--phase0-contract", type=Path)
    parser.add_argument("--verify-source-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-test-after-parser-v2-freeze", action="store_true")
    args = parser.parse_args()

    eval_root, dataset_root = args.eval_root.resolve(), args.dataset_root.resolve()
    phase0_path = (args.phase0_contract or (eval_root / "configs/phase0_contract.lock.json")).resolve()
    manifest_dir = (args.manifest_dir or (eval_root / "manifests")).resolve()
    phase0 = read_json(phase0_path)
    sam2_contract = read_json(eval_root / "configs/sam2_contract.json")

    registry_by_id, _ = load_model_registry(eval_root)
    closed_contract_path = eval_root / "configs/closed_api_eval_contract.v2.json"
    if closed_contract_path.is_file():
        closed_contract = read_json(closed_contract_path)
        closed_protocol = str(closed_contract.get("coordinate_policy", {}).get("prediction_protocol") or "")
        if closed_protocol == "base1000_source_full_image":
            closed_protocol = "base1000"
        for row in closed_contract.get("models", []):
            model_id = str(row.get("model_registry_id") or "")
            if model_id and model_id not in registry_by_id:
                registry_by_id[model_id] = {
                    **row,
                    "kind": "closed_api",
                    "coordinate_protocol": closed_protocol,
                }
    if args.model_registry_id not in registry_by_id:
        raise ValueError(f"unknown model_registry_id: {args.model_registry_id}")
    raw_rows = read_jsonl(args.predictions)
    _validate_schema(raw_rows, eval_root / "schemas/prediction.schema.json", str(args.predictions))
    raw_by_qa = unique_by_qa(raw_rows, str(args.predictions))
    expected_protocol = str(registry_by_id[args.model_registry_id]["coordinate_protocol"])
    bad_protocol = [row.get("qa_relpath") for row in raw_rows if row.get("requested_coordinate_protocol") != expected_protocol]
    if bad_protocol:
        raise ValueError(f"prediction protocol differs from model registry; first={bad_protocol[0]}")

    samples = load_ground_truth(
        eval_root,
        dataset_root,
        args.split,
        verify_hashes=args.verify_source_files,
        manifest_dir=manifest_dir,
    )
    manifest_keys = {sample.qa_relpath for sample in samples}
    extra = sorted(set(raw_by_qa) - manifest_keys)
    if extra:
        raise ValueError(f"predictions contain {len(extra)} extra qa_relpath; first={extra[0]}")

    mask_by_qa: dict[str, dict[str, Any]] = {}
    if args.sam2_masks:
        mask_rows = read_jsonl(args.sam2_masks)
        _validate_schema(mask_rows, eval_root / "schemas/sam2_mask.schema.json", str(args.sam2_masks))
        mask_by_qa = unique_by_qa(mask_rows, str(args.sam2_masks))
        allowed = {sample.qa_relpath for sample in samples if "Segmentation" in sample.task_key}
        extra_masks = sorted(set(mask_by_qa) - allowed)
        if extra_masks:
            raise ValueError(f"SAM2 masks contain non-segmentation/extra QA: {extra_masks[0]}")

    parsed_rows = [
        parse_prediction(
            sample,
            raw_by_qa.get(sample.qa_relpath),
            run_id=args.run_id,
            model_registry_id=args.model_registry_id,
            phase0_contract_sha256=str(phase0["phase0_contract_sha256"]),
            benchmark_manifest_sha256=str(phase0["benchmark_manifest_sha256"]),
        )
        for sample in samples
    ]
    _validate_schema(parsed_rows, eval_root / "schemas/parsed_prediction.schema.json", "generated parser-v2 predictions")
    out = args.out_dir.resolve()
    write_jsonl(out / "parsed_predictions.jsonl", parsed_rows)

    metric_spec_path = eval_root / "configs/metric_spec.v2.json"
    variants: dict[str, Any] = {}
    metric_rows_by_gate: dict[str, list[dict[str, Any]]] = {}
    for gate, label, slug in (
        ("iou_0.3", "TP@IoU0.3", "tp_iou0.3"),
        ("score_s_0.3", "TP@Score0.3", "tp_score0.3"),
    ):
        metric_rows = [
            score_sample(
                sample,
                parsed,
                raw_by_qa.get(sample.qa_relpath),
                mask_by_qa.get(sample.qa_relpath),
                run_id=args.run_id,
                model_registry_id=args.model_registry_id,
                phase0_contract_sha256=str(phase0["phase0_contract_sha256"]),
                benchmark_manifest_sha256=str(phase0["benchmark_manifest_sha256"]),
                metric_spec_sha256=sha256_file(metric_spec_path),
                multi_gate=gate,
                sam2_contract_sha256=str(sam2_contract["sam2_contract_sha256"]),
            )
            for sample, parsed in zip(samples, parsed_rows)
        ]
        _validate_schema(metric_rows, eval_root / "schemas/metric_record.schema.json", f"generated {label} records")
        summary = aggregate_metrics(metric_rows)
        summary.update({
            "schema_version": "uhr-bench-v4-evaluation-summary-1",
            "run_id": args.run_id,
            "model_registry_id": args.model_registry_id,
            "split": args.split,
            "phase0_contract_sha256": phase0["phase0_contract_sha256"],
            "benchmark_manifest_sha256": phase0["benchmark_manifest_sha256"],
            "metric_spec_sha256": sha256_file(metric_spec_path),
            "parser_id": PARSER_ID,
            "multi_metric_label": label,
            "multi_gate": gate,
            "formal_result": args.split == "Test",
        })
        metric_rows_by_gate[gate] = metric_rows
        write_jsonl(out / f"metric_records.{slug}.jsonl", metric_rows)
        write_json(out / f"summary.{slug}.json", summary)
        _write_summary_csv(out / f"summary_by_task.{slug}.csv", summary)
        variants[label] = summary

    multi_tasks = {"Grounding/GlobalDetection", "Grounding/RegionDetection", "Grounding/MultiConditionRetrieval"}
    non_multi_equal = all(
        left["score"] == right["score"] and left["metric_id"] == right["metric_id"]
        for left, right in zip(metric_rows_by_gate["iou_0.3"], metric_rows_by_gate["score_s_0.3"])
        if left["task_key"] not in multi_tasks
    )
    if not non_multi_equal:
        raise AssertionError("dual multi-metric variants changed a non-multi task")
    envelope = {
        "schema_version": "uhr-bench-v4-dual-multi-metric-summary-1",
        "run_id": args.run_id,
        "model_registry_id": args.model_registry_id,
        "split": args.split,
        "parser_id": PARSER_ID,
        "formal_multi_metric_mode": "dual_report_iou_0.3_and_score_s_0.3",
        "unique_primary_multi_metric": None,
        "selection_deferred_without_inference_rerun": True,
        "non_multi_task_records_identical": non_multi_equal,
        "variants": variants,
    }
    write_json(out / "summary.json", envelope)
    print(json.dumps({
        "out_dir": str(out),
        "sample_count_per_variant": len(parsed_rows),
        "task_count": 16,
        "parser_id": PARSER_ID,
        "overall_macro_16_tasks": {label: value["overall_macro_16_tasks"] for label, value in variants.items()},
        "formal_result": all(value["formal_result"] for value in variants.values()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
