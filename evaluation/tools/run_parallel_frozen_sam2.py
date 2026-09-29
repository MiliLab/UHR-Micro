#!/usr/bin/env python3
"""Run the frozen SAM2 entrypoint on manifest-ordered GPU shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
FROZEN_TOOL = EVAL_ROOT / "tools" / "run_frozen_sam2.py"
SAM_PYTHON = Path(os.environ.get("MAP_SAM2_PYTHON", ""))
SEGMENTATION_TASKS = {
    "Understanding/ReferSegmentation",
    "Understanding/ComponentSegmentation",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split", choices=["Development", "Validation", "Test"], required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Optional frozen subset-manifest directory. Defaults to v4_eval/manifests.",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--phase0-contract", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    gpu_ids = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("a formal parallel SAM2 run requires one or more distinct physical GPU IDs")
    parsed_path = run_root / "evaluation_no_sam" / "parsed_predictions.jsonl"
    manifest_root = args.manifest_dir.resolve() if args.manifest_dir else EVAL_ROOT / "manifests"
    manifest_path = manifest_root / f"{args.split}.jsonl"
    work_root = run_root / "sam2_shards"
    final_out = run_root / "sam2_masks.jsonl"
    if not FROZEN_TOOL.is_file() or not parsed_path.is_file() or not SAM_PYTHON.is_file():
        raise FileNotFoundError("frozen tool, parsed predictions, or SAM2 Python is missing")

    parsed_rows = read_jsonl(parsed_path)
    parsed_by_qa = {str(row["qa_relpath"]): row for row in parsed_rows}
    if len(parsed_by_qa) != len(parsed_rows):
        raise ValueError("parsed predictions contain duplicate QA IDs")
    manifest = read_jsonl(manifest_path)
    seg_rows = [row for row in manifest if row["task_key"] in SEGMENTATION_TASKS]
    missing_parsed = [row["qa_relpath"] for row in seg_rows if row["qa_relpath"] not in parsed_by_qa]
    if not seg_rows or missing_parsed:
        raise ValueError(f"segmentation manifest/parsed coverage failure: missing={len(missing_parsed)}")

    assignments: list[dict] = []
    shard_inputs: list[Path] = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        assigned = [row for index, row in enumerate(seg_rows) if index % len(gpu_ids) == shard_index]
        shard_rows = [parsed_by_qa[str(row["qa_relpath"])] for row in assigned]
        shard_input = work_root / "input" / f"shard_{shard_index:02d}.jsonl"
        write_jsonl(shard_input, shard_rows)
        shard_inputs.append(shard_input)
        for row in assigned:
            assignments.append({
                "qa_relpath": row["qa_relpath"],
                "task_key": row["task_key"],
                "shard_index": shard_index,
                "physical_gpu_id": gpu_id,
            })
    assignment_by_qa = {str(row["qa_relpath"]): row for row in assignments}
    if len(assignment_by_qa) != len(seg_rows):
        raise ValueError("duplicate or missing segmentation assignment")
    assignment_path = work_root / "assignment.json"
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text(json.dumps({
        "schema_version": "uhr-bench-v4-sam2-parallel-assignment-2",
        "split": args.split,
        "strategy": f"manifest_index_mod_{len(gpu_ids)}",
        "gpu_ids": gpu_ids,
        "frozen_tool_sha256": sha256_file(FROZEN_TOOL),
        "assignments": assignments,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    wrapper = work_root / "run_frozen_with_large_image_support.py"
    wrapper.write_text(
        "import runpy\nfrom PIL import Image\nImage.MAX_IMAGE_PIXELS = None\n"
        f"runpy.run_path({str(FROZEN_TOOL)!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    processes: list[tuple[int, str, subprocess.Popen, object, Path, Path]] = []
    for shard_index, (gpu_id, shard_input) in enumerate(zip(gpu_ids, shard_inputs)):
        shard_out = work_root / "output" / f"shard_{shard_index:02d}.jsonl"
        log_path = work_root / "logs" / f"shard_{shard_index:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        shard_out.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        command = [
            str(SAM_PYTHON), str(wrapper),
            "--parsed", str(shard_input),
            "--split", args.split,
            "--out", str(shard_out),
            "--device", "cuda",
            "--manifest", str(manifest_path),
        ]
        if args.dataset_root:
            command.extend(["--dataset-root", str(args.dataset_root.resolve())])
        if args.phase0_contract:
            command.extend(["--phase0-contract", str(args.phase0_contract.resolve())])
        if args.split == "Test":
            command.append("--allow-test-after-freeze")
        process = subprocess.Popen(
            command,
            cwd=str(EVAL_ROOT.parent),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_index, gpu_id, process, log_handle, log_path, shard_out))

    failures: list[dict] = []
    shard_summaries: list[dict] = []
    for shard_index, gpu_id, process, log_handle, log_path, shard_out in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode != 0 or not shard_out.is_file():
            failures.append({
                "shard_index": shard_index,
                "gpu_id": gpu_id,
                "returncode": returncode,
                "log": str(log_path),
            })
            continue
        rows = read_jsonl(shard_out)
        shard_summaries.append({
            "shard_index": shard_index,
            "gpu_id": gpu_id,
            "returncode": returncode,
            "output_rows": len(rows),
            "success_rows": sum(row["status"] == "success" for row in rows),
            "log": str(log_path),
            "output": str(shard_out),
            "output_sha256": sha256_file(shard_out),
        })
    if failures:
        raise RuntimeError(f"SAM2 shard failures: {failures}")

    shard_rows_by_index = {
        item[0]: {str(row["qa_relpath"]): row for row in read_jsonl(item[5])}
        for item in processes
    }
    merged: list[dict] = []
    for manifest_row in seg_rows:
        qa_relpath = str(manifest_row["qa_relpath"])
        assignment = assignment_by_qa[qa_relpath]
        merged.append(shard_rows_by_index[int(assignment["shard_index"])][qa_relpath])
    if len(merged) != len(seg_rows) or len({str(row["qa_relpath"]) for row in merged}) != len(seg_rows):
        raise ValueError("merged SAM2 output is not uniquely aligned with the segmentation manifest")
    write_jsonl(final_out, merged)
    status_counts: dict[str, int] = {}
    for row in merged:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema_version": "uhr-bench-v4-sam2-parallel-summary-2",
        "passed": status_counts.get("sam2_error", 0) == 0 and status_counts.get("image_error", 0) == 0,
        "split": args.split,
        "segmentation_rows": len(merged),
        "unique_qa": len({row["qa_relpath"] for row in merged}),
        "status_counts": status_counts,
        "gpu_ids": gpu_ids,
        "shards": shard_summaries,
        "assignment_sha256": sha256_file(assignment_path),
        "sam2_masks_sha256": sha256_file(final_out),
        "frozen_tool_sha256": sha256_file(FROZEN_TOOL),
        "large_image_wrapper_effect": "PIL.Image.MAX_IMAGE_PIXELS=None only; frozen tool and SAM2 parameters unchanged",
    }
    summary_path = run_root / "sam2_parallel_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
