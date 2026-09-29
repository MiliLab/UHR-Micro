#!/usr/bin/env python3
"""InternVL3.5-8B runner for the MAP pipeline (backbone-transfer experiment).

Contract-compatible with the frozen Qwen runner
(adapter_runtime/tools/run_qwen_tasks.py): same CLI, same per-task request
binding verification (request_digest), same attempt ledger / summary schemas,
same output row fields. Backend differences:

  - transformers (not vLLM), one model copy per GPU via CUDA_VISIBLE_DEVICES;
  - InternVL dynamic tiling with max_num=12 tiles of 448px (the historical
    formal InternVL3.5 preprocessing profile);
  - generation via model.chat with do_sample=False at temperature 0;
  - api_mode = "map_internvl3_5_8b_transformers_local".

This file lives outside every frozen tree of map_v1.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ADAPTER_TOOLS = Path(__file__).resolve().parents[1] / "runtime" / "tools"
INTERNVL_REF_DIR = Path(os.environ.get("MAP_INTERNVL_REF_DIR", ""))
DEFAULT_MODEL = Path(os.environ.get("MAP_INTERNVL_MODEL_PATH") or Path(__file__).resolve().parents[3] / "models/InternVL3_5-8B")
INTERNVL_MAX_NUM_TILES = 12

sys.path.insert(0, str(ADAPTER_TOOLS))
sys.path.insert(0, str(INTERNVL_REF_DIR))

from io_utils import read_jsonl, recover_truncated_jsonl_tail, request_digest, sha256_file  # noqa: E402


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def worker(
    worker_index: int,
    gpu_id: int,
    tasks: mp.Queue,
    events: mp.Queue,
    model_path: str,
    config: dict[str, Any],
    error_log: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    time.sleep(worker_index * 1.5)
    try:
        import torch
        from PIL import Image
        from internvl_grounding import load_image, dynamic_preprocess, build_transform

        Image.MAX_IMAGE_PIXELS = None

        def load_image_cfg(path: str, max_num: int, max_side: int):
            image = Image.open(path).convert("RGB")
            if max_side and max(image.size) > max_side:
                scale = max_side / max(image.size)
                image = image.resize(
                    (round(image.size[0] * scale), round(image.size[1] * scale)),
                    Image.LANCZOS,
                )
            transform = build_transform(448)
            images = dynamic_preprocess(
                image, image_size=448, use_thumbnail=True, max_num=max_num
            )
            return torch.stack([transform(img) for img in images])

        torch.cuda.set_device(0)
        from transformers import AutoModel, AutoTokenizer

        model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .to("cuda")
        )
        # Infra fix (manual-review resolution 2026-08-31): the remote code
        # forces eager attention when flash-attn is absent, which OOMs on the
        # longest composer prompts. sdpa is memory-efficient and does not
        # change any generation parameter, prompt, image, or scientific
        # element of the frozen method.
        if getattr(model.language_model.config, "_attn_implementation", "eager") == "eager":
            model.language_model.config._attn_implementation = "sdpa"
            model.language_model._attn_implementation = "sdpa"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        events.put({"status": "worker_ready", "worker": worker_index, "gpu_id": gpu_id})
    except Exception as error:
        events.put(
            {
                "status": "worker_init_error",
                "worker": worker_index,
                "gpu_id": gpu_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        task = tasks.get()
        if task is None:
            break
        task_id = str(task["task_id"])
        events.put(
            {
                "status": "started",
                "worker": worker_index,
                "gpu_id": gpu_id,
                "task_id": task_id,
                "request_sha256": task["request_sha256"],
                "request_contract_sha256": task["request_contract_sha256"],
                "attempt_number": task["attempt_number"],
                "started_at_unix": int(time.time()),
            }
        )
        started = time.perf_counter()
        try:
            actual_request_sha256 = request_digest(
                task, str(task["request_contract_sha256"])
            )
            if actual_request_sha256 != str(task["request_sha256"]):
                raise PermissionError(
                    "prompt or model image bytes changed after request binding"
                )
            pixel_values = (
                load_image_cfg(
                    str(task["model_image_path"]),
                    max_num=int(config.get("max_num_tiles", INTERNVL_MAX_NUM_TILES)),
                    max_side=int(config.get("global_view_max_side", 0) or 0),
                )
                .to(torch.bfloat16)
                .to("cuda")
            )
            generation_config = {"max_new_tokens": int(config["max_new_tokens"])}
            if float(config["temperature"]) > 0:
                generation_config["do_sample"] = True
                generation_config["temperature"] = float(config["temperature"])
                if float(config.get("top_p", 1.0)) < 1.0:
                    generation_config["top_p"] = float(config["top_p"])
            else:
                generation_config["do_sample"] = False
            response = model.chat(
                tokenizer,
                pixel_values,
                str(task["model_input_prompt"]),
                generation_config,
                return_history=False,
                verbose=False,
            )
            if isinstance(response, tuple):
                response = response[0]
            raw_backend_response = str(response)
            row = dict(task)
            row.update(
                {
                    "raw_backend_response": raw_backend_response,
                    "model_response": raw_backend_response,
                    "api_mode": "map_internvl3_5_8b_transformers_local",
                    "worker": worker_index,
                    "gpu_id": gpu_id,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "completed_at_unix": int(time.time()),
                }
            )
            events.put({"status": "success", "worker": worker_index, "row": row})
        except Exception as error:
            with Path(error_log).open("a", encoding="utf-8") as stream:
                stream.write(
                    f"task_id={task_id}\n{type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}\n{'=' * 80}\n"
                )
            row = dict(task)
            row.update(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "error_type": "infrastructure",
                    "api_mode": "map_internvl3_5_8b_transformers_local",
                    "worker": worker_index,
                    "gpu_id": gpu_id,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "completed_at_unix": int(time.time()),
                }
            )
            events.put({"status": "error", "worker": worker_index, "row": row})
    del model
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--request-contract", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--no-progress-timeout-sec", type=int, default=3600)
    parser.add_argument("--global-view-max-side", type=int, default=0)
    parser.add_argument("--max-num-tiles", type=int, default=INTERNVL_MAX_NUM_TILES)
    args = parser.parse_args()
    if args.summary_out.exists():
        raise FileExistsError(f"launch summary already exists: {args.summary_out}")
    recover_truncated_jsonl_tail(args.out)
    recover_truncated_jsonl_tail(args.attempt_ledger)
    rows = read_jsonl(args.tasks)
    ids = [str(row.get("task_id") or "") for row in rows]
    if not rows or "" in ids or len(ids) != len(set(ids)):
        raise ValueError("runner input must contain unique non-empty task IDs")
    contract = json.loads(args.request_contract.read_text(encoding="utf-8"))
    contract_hash = str(contract["request_contract_sha256"])
    required_observer_profile = contract.get(
        "required_observer_generation_profile"
    )
    if required_observer_profile not in {None, "STANDARD", "QUERY_NATIVE"}:
        raise ValueError(
            "request contract declares an unsupported observer generation profile"
        )
    expected_runtime = {
        "model_path": str(args.model.resolve()),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
    }
    declared_runtime = {
        "model_path": str(Path(contract["model_path"]).resolve()),
        "max_new_tokens": int(contract["max_new_tokens"]),
        "temperature": float(contract["temperature"]),
        "top_p": float(contract["top_p"]),
        "gpu_memory_utilization": float(contract["gpu_memory_utilization"]),
    }
    if expected_runtime != declared_runtime:
        raise PermissionError(
            f"runner CLI differs from frozen request contract: "
            f"actual={expected_runtime}, declared={declared_runtime}"
        )
    for row in rows:
        if row.get("request_contract_sha256") != contract_hash:
            raise PermissionError("task/request contract mismatch")
        if not row.get("request_sha256") or int(row.get("attempt_number") or 0) < 1:
            raise ValueError("task lacks request hash or persistent attempt number")
        if request_digest(row, contract_hash) != str(row["request_sha256"]):
            raise PermissionError("task request hash does not match current prompt/image bytes")
        if (
            required_observer_profile is not None
            and row.get("observer_generation_profile")
            != required_observer_profile
        ):
            raise PermissionError(
                "task observer generation profile differs from request contract"
            )
    rows.sort(
        key=lambda row: (
            -Path(str(row["model_image_path"])).stat().st_size,
            str(row["task_id"]),
        )
    )
    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    if (
        not gpu_ids
        or len(gpu_ids) != len(set(gpu_ids))
        or any(gpu_id < 0 or gpu_id > 7 for gpu_id in gpu_ids)
    ):
        raise RuntimeError(f"GPU IDs must be unique integers in [0,7]: {gpu_ids}")
    config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "global_view_max_side": args.global_view_max_side,
        "max_num_tiles": args.max_num_tiles,
    }
    context = mp.get_context("spawn")
    task_queue: mp.Queue = context.Queue()
    event_queue: mp.Queue = context.Queue()
    logs = args.out.parent / "worker_logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: dict[int, mp.Process] = {}
    for index, gpu_id in enumerate(gpu_ids):
        process = context.Process(
            target=worker,
            args=(
                index,
                gpu_id,
                task_queue,
                event_queue,
                str(args.model.resolve()),
                config,
                str(logs / f"worker_{index}_gpu{gpu_id}.log"),
            ),
        )
        process.start()
        processes[index] = process
    task_map = {str(row["task_id"]): row for row in rows}
    for row in rows:
        task_queue.put(row)
    for _ in processes:
        task_queue.put(None)

    completed: set[str] = set()
    started: set[str] = set()
    in_flight: dict[int, str] = {}
    dead_recorded: set[int] = set()
    success = errors = init_errors = 0
    last_event = time.monotonic()
    interrupted = False
    while len(completed) < len(rows):
        try:
            event = event_queue.get(timeout=2)
            last_event = time.monotonic()
        except queue.Empty:
            event = None
        if event is not None:
            state = str(event["status"])
            if state == "worker_ready":
                append_jsonl(args.attempt_ledger, {**event, "event_at_unix": int(time.time())})
            elif state == "worker_init_error":
                init_errors += 1
                append_jsonl(args.attempt_ledger, {**event, "event_at_unix": int(time.time())})
            elif state == "started":
                ident = str(event["task_id"])
                started.add(ident)
                in_flight[int(event["worker"])] = ident
                append_jsonl(args.attempt_ledger, event)
            elif state in {"success", "error"}:
                row = dict(event["row"])
                ident = str(row["task_id"])
                if ident not in completed:
                    completed.add(ident)
                    append_jsonl(args.out, row)
                    append_jsonl(
                        args.attempt_ledger,
                        {
                            "status": state,
                            "task_id": ident,
                            "request_sha256": row["request_sha256"],
                            "request_contract_sha256": row["request_contract_sha256"],
                            "attempt_number": row["attempt_number"],
                            "worker": row["worker"],
                            "gpu_id": row["gpu_id"],
                            "latency_ms": row["latency_ms"],
                            "error": row.get("error"),
                            "event_at_unix": int(time.time()),
                        },
                    )
                    success += int(state == "success")
                    errors += int(state == "error")
                in_flight.pop(int(event["worker"]), None)
        for index, process in processes.items():
            if process.is_alive() or index in dead_recorded:
                continue
            dead_recorded.add(index)
            ident = in_flight.pop(index, None)
            if ident and ident not in completed:
                task = dict(task_map[ident])
                task.update(
                    {
                        "error": f"WorkerDied: worker {index} exited with code {process.exitcode}",
                        "error_type": "infrastructure",
                        "worker": index,
                        "gpu_id": gpu_ids[index],
                        "latency_ms": None,
                        "completed_at_unix": int(time.time()),
                    }
                )
                completed.add(ident)
                errors += 1
                append_jsonl(args.out, task)
                append_jsonl(
                    args.attempt_ledger,
                    {
                        "status": "error",
                        "task_id": ident,
                        "request_sha256": task["request_sha256"],
                        "request_contract_sha256": task["request_contract_sha256"],
                        "attempt_number": task["attempt_number"],
                        "worker": index,
                        "gpu_id": gpu_ids[index],
                        "error": task["error"],
                        "event_at_unix": int(time.time()),
                    },
                )
        alive = [process for process in processes.values() if process.is_alive()]
        if not alive and event is None and len(completed) < len(rows):
            interrupted = True
            break
        if time.monotonic() - last_event > max(60, args.no_progress_timeout_sec):
            interrupted = True
            for process in alive:
                process.terminate()
            break
    for process in processes.values():
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    summary = {
        "schema_version": "map-internvl-runner-summary-1",
        "launch_id": args.launch_id,
        "task_manifest_sha256": sha256_file(args.tasks),
        "tasks": len(rows),
        "started": len(started),
        "completed": len(completed),
        "success": success,
        "errors": errors,
        "worker_init_errors": init_errors,
        "unstarted": len(rows) - len(started),
        "interrupted": interrupted,
        "request_contract_sha256": contract_hash,
        "required_observer_generation_profile": required_observer_profile,
        "gpu_ids": gpu_ids,
        "test_used": True,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 2 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
