#!/usr/bin/env python3
"""Evaluate method-research predictions with frozen v4.1 evaluators.

This shared implementation is independent of every active or rejected method
candidate. All generated artifacts remain under the caller's method-research
run directory; v4_eval supplies only the frozen Parser V2, Metric V2, SAM2,
benchmark manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve()
EVAL_ROOT = HERE
DATASET = Path(os.environ.get("MAP_DATASET", ""))
EVAL_PYTHON = Path(os.environ.get("MAP_EVAL_PYTHON", sys.executable))
SAM2_PYTHON = Path(os.environ.get("MAP_SAM2_PYTHON", ""))
FILES = (
    "parsed_predictions.jsonl",
    "metric_records.tp_iou0.3.jsonl",
    "metric_records.tp_score0.3.jsonl",
    "summary.json",
    "summary.tp_iou0.3.json",
    "summary.tp_score0.3.json",
    "summary_by_task.tp_iou0.3.csv",
    "summary_by_task.tp_score0.3.csv",
)
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))



def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def segmentation_rows(path: Path) -> list[dict]:
    rows = [
        row
        for row in read_jsonl(path)
        if str(row.get("task_key") or "")
        in {
            "Understanding/ComponentSegmentation",
            "Understanding/ReferSegmentation",
        }
    ]
    return sorted(rows, key=lambda row: str(row["qa_relpath"]))


def bind_identical_sam2_reuse(run_dir: Path, target: str, source: str) -> None:
    target_root = run_dir / "evaluation" / target
    source_root = run_dir / "evaluation" / source
    source_parsed = source_root / "evaluation_no_sam" / "parsed_predictions.jsonl"
    target_parsed = target_root / "evaluation_no_sam" / "parsed_predictions.jsonl"
    source_masks = source_root / "sam2_masks.jsonl"
    source_summary_path = source_root / "sam2_parallel_summary.json"
    for required in (source_parsed, target_parsed, source_masks, source_summary_path):
        if not required.is_file():
            raise FileNotFoundError(f"SAM2 reuse input is missing: {required}")
    source_rows = segmentation_rows(source_parsed)
    target_rows = segmentation_rows(target_parsed)
    if len(source_rows) != 200 or source_rows != target_rows:
        raise PermissionError(
            "SAM2 reuse requires 200 byte-equivalent parsed segmentation records"
        )
    source_summary = read_json(source_summary_path)
    if not source_summary.get("passed"):
        raise PermissionError("source SAM2 evaluation did not pass")
    target_masks = target_root / "sam2_masks.jsonl"
    target_masks.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_masks, target_masks)
    binding = {
        "schema_version": "method-research-identical-sam2-reuse-binding-1",
        "passed": True,
        "source_variant": source,
        "target_variant": target,
        "segmentation_rows": len(target_rows),
        "parsed_segmentation_records_identical": True,
        "source_parsed_predictions_sha256": sha256_file(source_parsed),
        "target_parsed_predictions_sha256": sha256_file(target_parsed),
        "source_sam2_summary_sha256": sha256_file(source_summary_path),
        "source_sam2_masks_sha256": sha256_file(source_masks),
        "target_sam2_masks_sha256": sha256_file(target_masks),
        "frozen_tool_sha256": source_summary.get("frozen_tool_sha256"),
        "assignment_sha256": source_summary.get("assignment_sha256"),
        "test_used": False,
    }
    if binding["source_sam2_masks_sha256"] != binding["target_sam2_masks_sha256"]:
        raise RuntimeError("SAM2 mask copy hash mismatch")
    write_json(target_root / "sam2_reuse_binding.json", binding)
    reused_summary = dict(source_summary)
    reused_summary["schema_version"] = "uhr-bench-v4-sam2-identical-input-reuse-summary-1"
    reused_summary["reuse_binding"] = binding
    write_json(target_root / "sam2_parallel_summary.json", reused_summary)


def execute(command: list[str], cwd: Path = EVAL_ROOT, *, retries: int = 4) -> None:
    delay = 3.0
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(command, cwd=cwd, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            print(f"retrying subprocess after transient failure ({attempt}/{retries - 1}): {command[1:3]}", flush=True)
            time.sleep(delay)
            delay *= 2.0


def parser_command(
    split: str,
    predictions: Path,
    run_id: str,
    model_registry_id: str,
    out: Path,
    sam2: Path | None,
    *,
    manifest_dir: Path,
    dataset_root: Path,
    phase0_contract: Path,
) -> list[str]:
    command = [
        str(EVAL_PYTHON), "-m", "evaluator.cli",
        "--split", split,
        "--predictions", str(predictions),
        "--run-id", run_id,
        "--model-registry-id", model_registry_id,
        "--out-dir", str(out),
        "--eval-root", str(EVAL_ROOT),
        "--dataset-root", str(dataset_root),
        "--manifest-dir", str(manifest_dir),
        "--phase0-contract", str(phase0_contract),
        "--no-verify-source-files",
    ]
    if sam2 is not None:
        command.extend(["--sam2-masks", str(sam2)])
    if split == "Test":
        command.append("--allow-test-after-parser-v2-freeze")
    return command


def task_scores(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as stream:
        return {
            str(row["task_key"]): float(row["score_mean_fixed_denominator"])
            for row in csv.DictReader(stream)
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["Validation", "Test"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variants", required=True, help="comma-separated canonical variant directories")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--phase0-contract", type=Path, default=None)
    parser.add_argument(
        "--sam2-reuse-from",
        help="source variant with passed SAM2 output; requires exactly one target variant and identical parsed segmentation records",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    variants = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    manifest_dir = (args.manifest_dir or Path(os.environ.get("MAP_MANIFEST_DIR", ""))).resolve()
    dataset_root = (args.dataset_root or DATASET).resolve()
    phase0_contract = (args.phase0_contract or (EVAL_ROOT / "configs/phase0_contract_v4_1.lock.json")).resolve()
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be non-empty and unique")
    if args.sam2_reuse_from and len(variants) != 1:
        raise ValueError("--sam2-reuse-from requires exactly one target variant")

    report = {
        "schema_version": "method-research-formal-evaluation-1",
        "split": args.split,
        "fixed_denominator": expected,
        "variants": {},
        "parser_v2_repeat_byte_identical": {},
        "shared_evaluator_sha256": sha256_file(HERE),
    }
    for variant in variants:
        predictions = run_dir / "variants" / variant / "canonical_predictions.jsonl"
        rows = read_jsonl(predictions)
        if len(rows) != len({str(row["qa_relpath"]) for row in rows}):
            raise ValueError(f"{variant}: canonical predictions contain duplicate qa_relpath")
        if Counter(str(row.get("runtime_status")) for row in rows) != {"success": expected}:
            raise ValueError(f"{variant}: non-success canonical prediction")
        canonical_run_ids = {str(row.get("run_id") or "") for row in rows}
        canonical_model_ids = {str(row.get("model_registry_id") or "") for row in rows}
        if len(canonical_run_ids) != 1 or "" in canonical_run_ids:
            raise ValueError(f"{variant}: canonical run identity is not unique")
        if not canonical_model_ids <= {"qwen3_vl_8b", "internvl3_5_8b"}:
            raise ValueError(f"{variant}: unexpected canonical model identity {sorted(canonical_model_ids)}")
        run_id = next(iter(canonical_run_ids))
        model_registry_id = next(iter(canonical_model_ids))
        variant_root = run_dir / "evaluation" / variant
        no_sam = variant_root / "evaluation_no_sam"
        execute(parser_command(args.split, predictions, run_id, model_registry_id, no_sam, None,
                          manifest_dir=manifest_dir, dataset_root=dataset_root, phase0_contract=phase0_contract))
        if args.sam2_reuse_from:
            bind_identical_sam2_reuse(run_dir, variant, args.sam2_reuse_from)
        elif any(str(row.get("task_key") or "") in {"Understanding/ReferSegmentation", "Understanding/ComponentSegmentation"} for row in rows):
            execute([
                str(SAM2_PYTHON), str(EVAL_ROOT / "tools/run_parallel_frozen_sam2.py"),
                "--run-root", str(variant_root),
                "--split", args.split,
                "--gpus", args.gpus,
                "--manifest-dir", str(manifest_dir),
                "--dataset-root", str(dataset_root),
                "--phase0-contract", str(phase0_contract),
            ])
        else:
            (variant_root / "sam2_masks.jsonl").write_text("", encoding="utf-8")
            write_json(variant_root / "sam2_parallel_summary.json", {
                "schema_version": "sam2-parallel-summary-1",
                "skipped_choice_only_subset": True,
                "rows": 0,
            })
        sam2 = variant_root / "sam2_masks.jsonl"
        final = variant_root / "evaluation_parser_v2_final"
        repeat = variant_root / "evaluation_parser_v2_repeat"
        execute(parser_command(args.split, predictions, run_id, model_registry_id, final, sam2,
                          manifest_dir=manifest_dir, dataset_root=dataset_root, phase0_contract=phase0_contract))
        execute(parser_command(args.split, predictions, run_id, model_registry_id, repeat, sam2,
                          manifest_dir=manifest_dir, dataset_root=dataset_root, phase0_contract=phase0_contract))
        repeat_checks = {name: sha256_file(final / name) == sha256_file(repeat / name) for name in FILES}
        report["parser_v2_repeat_byte_identical"][variant] = repeat_checks

        score = read_json(final / "summary.tp_score0.3.json")
        iou = read_json(final / "summary.tp_iou0.3.json")
        if score.get("overall_macro_16_tasks") is None:
            score = dict(score)
            iou = dict(iou)
            rr = (score.get("tasks") or {}).get("Relationship/RelationshipRecognition") or {}
            rr_mean = rr.get("score_mean_fixed_denominator")
            if rr_mean is not None:
                score["overall_macro_16_tasks"] = rr_mean
                iou["overall_macro_16_tasks"] = rr_mean
            else:
                # Subset run without RelationshipRecognition: macro over the
                # task means that are present in this manifest denominator.
                score_values = [
                    float(entry["score_mean_fixed_denominator"])
                    for entry in (score.get("tasks") or {}).values()
                    if entry.get("score_mean_fixed_denominator") is not None
                ]
                iou_values = [
                    float(entry["score_mean_fixed_denominator"])
                    for entry in (iou.get("tasks") or {}).values()
                    if entry.get("score_mean_fixed_denominator") is not None
                ]
                score["overall_macro_16_tasks"] = sum(score_values) / len(score_values) if score_values else 0.0
                iou["overall_macro_16_tasks"] = sum(iou_values) / len(iou_values) if iou_values else 0.0
        parsed = read_jsonl(final / "parsed_predictions.jsonl")
        report["variants"][variant] = {
            "TP@Score0.3": float(score["overall_macro_16_tasks"]),
            "TP@IoU0.3": float(iou["overall_macro_16_tasks"]),
            "family_macro_TP@Score0.3": score["family_macro"],
            "family_macro_TP@IoU0.3": iou["family_macro"],
            "per_task_TP@Score0.3": task_scores(final / "summary_by_task.tp_score0.3.csv"),
            "per_task_TP@IoU0.3": task_scores(final / "summary_by_task.tp_iou0.3.csv"),
            "parse_status_counts": dict(sorted(Counter(str(row.get("parse_status")) for row in parsed).items())),
            "mean_logical_vlm_calls": sum(float(row.get("logical_vlm_calls") or 0) for row in rows) / expected,
            "canonical_predictions_sha256": sha256_file(predictions),
            "sam2_summary": read_json(variant_root / "sam2_parallel_summary.json"),
        }
    report["all_repeat_checks_passed"] = all(
        value for group in (report["parser_v2_repeat_byte_identical"],)
        for checks in group.values() for value in checks.values()
    )
    write_json(run_dir / "FORMAL_EVALUATION.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_repeat_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
