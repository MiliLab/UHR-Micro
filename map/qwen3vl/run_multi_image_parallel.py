#!/usr/bin/env python3
"""Run frozen multi-image receipts over deterministic GPU shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-infrastructure-attempts", type=int, default=5)
    args = parser.parse_args()

    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    tasks = read_jsonl(args.tasks)
    shards: list[list[dict]] = [[] for _ in gpus]
    for row in tasks:
        digest = hashlib.sha256(str(row["task_id"]).encode("utf-8")).digest()
        shards[int.from_bytes(digest[:8], "big") % len(gpus)].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, Path, subprocess.Popen]] = []
    for index, (gpu, rows) in enumerate(zip(gpus, shards)):
        if not rows:
            continue
        root = args.output_dir / f"shard_{index:02d}_gpu{gpu}"
        task_path = root / "tasks.jsonl"
        output_path = root / "outputs.jsonl"
        write_jsonl(task_path, rows)
        root.mkdir(parents=True, exist_ok=True)
        log = (root / "runner.log").open("a", encoding="utf-8")
        command = [
            str(args.python), str(args.runner),
            "--tasks", str(task_path),
            "--contract", str(args.contract),
            "--out", str(output_path),
            "--attempt-ledger", str(root / "attempts.jsonl"),
            "--summary", str(root / "summary.json"),
            "--gpu", str(gpu),
            "--max-infrastructure-attempts", str(args.max_infrastructure_attempts),
        ]
        processes.append((index, output_path, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)))

    failed: list[int] = []
    for index, _, process in processes:
        if process.wait() != 0:
            failed.append(index)
    if failed:
        raise RuntimeError(f"multi-image shards failed: {failed}")

    by_id: dict[str, dict] = {}
    for _, output_path, _ in processes:
        for row in read_jsonl(output_path):
            ident = str(row["task_id"])
            if ident in by_id:
                raise ValueError(f"duplicate output task id: {ident}")
            by_id[ident] = row
    expected = [str(row["task_id"]) for row in tasks]
    if set(by_id) != set(expected):
        raise RuntimeError("parallel output coverage mismatch")
    write_jsonl(args.output_dir / "outputs.jsonl", [by_id[ident] for ident in expected])
    summary = {
        "schema_version": "map-multi-image-parallel-summary-1",
        "task_rows": len(tasks),
        "success_rows": len(by_id),
        "gpu_ids": gpus,
        "nonempty_shards": len(processes),
        "max_infrastructure_attempts": args.max_infrastructure_attempts,
        "test_used": False,
        "passed": len(by_id) == len(tasks),
    }
    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

