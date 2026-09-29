#!/usr/bin/env python3
"""Run one exact-contract MAP v2.2 shard with liveness monitoring."""

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

from io_utils import (
    read_jsonl,
    recover_truncated_jsonl_tail,
    request_digest,
    sha256_file,
)


DEFAULT_MODEL = Path(os.environ.get("MAP_MODEL_PATH") or Path(__file__).resolve().parents[3] / "models/Qwen3-VL-main/Qwen3-VL-8B-Instruct")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def automatic_pixels(
    min_pixels: int | None, max_pixels: int | None
) -> tuple[int, int]:
    patch_size, merge_base = 16, 2
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    return (
        min_pixels or pixels_per_token * 768,
        max_pixels or pixels_per_token * 12800,
    )


def messages_for(
    image_path: str, prompt: str, min_pixels: int | None, max_pixels: int | None
) -> list[dict[str, Any]]:
    minimum, maximum = automatic_pixels(min_pixels, max_pixels)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                    "min_pixels": minimum,
                    "max_pixels": maximum,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_vllm_input(messages: list[dict[str, Any]], processor: Any) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    multimodal: dict[str, Any] = {}
    if image_inputs is not None:
        multimodal["image"] = image_inputs
    if video_inputs is not None:
        multimodal["video"] = video_inputs
    return {
        "prompt": prompt,
        "multi_modal_data": multimodal,
        "mm_processor_kwargs": video_kwargs,
    }


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
    time.sleep(worker_index * 1.5)
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        torch.cuda.set_device(0)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        engine = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=float(config["gpu_memory_utilization"]),
            trust_remote_code=True,
            max_model_len=128000,
            limit_mm_per_prompt={"image": 1},
            seed=42,
        )
        sampling = SamplingParams(
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            max_tokens=int(config["max_new_tokens"]),
            stop_token_ids=[],
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
            messages = messages_for(
                str(task["model_image_path"]),
                str(task["model_input_prompt"]),
                task.get("min_pixels"),
                task.get("max_pixels"),
            )
            generated = engine.generate(
                [prepare_vllm_input(messages, processor)],
                sampling_params=sampling,
                use_tqdm=False,
            )
            raw_backend_response = generated[0].outputs[0].text
            row = dict(task)
            row.update(
                {
                    # Preserve the backend output byte-for-byte.  All parsers
                    # normalize locally and never overwrite this evidence.
                    "raw_backend_response": raw_backend_response,
                    "model_response": raw_backend_response,
                    "api_mode": "map_v2_2_qwen3vl_local_vllm",
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
                    "api_mode": "map_v2_2_qwen3vl_local_vllm",
                    "worker": worker_index,
                    "gpu_id": gpu_id,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "completed_at_unix": int(time.time()),
                }
            )
            events.put({"status": "error", "worker": worker_index, "row": row})
    del engine
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
    # Dynamic workers share one queue. Enqueue the largest visual inputs first
    # so unusually large UHR examples do not become a single-GPU tail after
    # the other workers have already drained their shorter requests.
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
        # A multiprocessing Queue can deliver the last success shortly after a
        # worker exits.  Only declare interruption after an empty timed poll;
        # otherwise a queued terminal event would be lost.
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
        "schema_version": "map-v2.2-qwen-runner-summary-1",
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
        "test_used": False,
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
