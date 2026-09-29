from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .constants import PKG_ROOT
from .runner import merge_attempts, run_parallel
from .manifest import load_input_manifest
from .io import read_json


def parse_gpu_ids(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("GPU list is empty")
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("GPU IDs must be unique")
    if any(not item.isdigit() for item in ids):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative integers")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="UHR-Bench v4 unified inference runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--input-manifest", type=Path, required=True)
    run.add_argument("--run-config", type=Path, required=True)
    run.add_argument("--adapter-config", type=Path, default=PKG_ROOT / "configs/qwen3_vl_8b.json")
    run.add_argument("--model-path", type=Path, required=True, help="path to the local model checkpoint")
    run.add_argument("--image-root", type=Path, required=True, help="path to the UHR-Bench Datasource root")
    run.add_argument("--gpus", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3,4,5,6,7"))
    run.add_argument(
        "--scheduler",
        choices=["dynamic_queue", "static_round_robin", "static_image_hash"],
        default="dynamic_queue",
    )
    run.add_argument("--dry-run", action="store_true")

    merge = sub.add_parser("merge")
    merge.add_argument("--input-manifest", type=Path, required=True)
    merge.add_argument("--run-config", type=Path, required=True)
    merge.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    if args.command == "run":
        result = run_parallel(
            args.input_manifest,
            args.run_config,
            args.adapter_config,
            args.model_path,
            args.image_root,
            args.gpus,
            args.scheduler,
            args.dry_run,
        )
    else:
        rows = load_input_manifest(args.input_manifest)
        config = read_json(args.run_config)
        result = merge_attempts(
            rows,
            Path(config["output_root"]),
            int(config["retry_policy"]["max_attempts"]),
            args.allow_incomplete,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
