#!/usr/bin/env python3
"""Checkpointed single-GPU Qwen3-VL runner for separate multi-image tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def request_digest(row: dict[str, Any], contract_hash: str) -> str:
    hashes = [sha256_file(Path(path).resolve()) for path in row["model_image_paths"]]
    if hashes != row["model_image_sha256"]:
        raise ValueError("multi-image bytes differ from frozen task hashes")
    return sha256_json({
        "task_id": row["task_id"],
        "image_sha256": hashes,
        "prompt": row["model_input_prompt"],
        "observer_generation_profile": row["observer_generation_profile"],
        "request_contract_sha256": contract_hash,
    })


def messages_for(paths: list[str], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for path in paths:
        content.append({
            "type": "image",
            "image": f"file://{path}",
            "min_pixels": 16 * 16 * 2 * 2 * 768,
            "max_pixels": 16 * 16 * 2 * 2 * 12800,
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_hash = str(contract["request_contract_sha256"])
    tasks = read_jsonl(args.tasks)
    if len(tasks) != len({row["task_id"] for row in tasks}):
        raise ValueError("task ids are not unique")
    for row in tasks:
        if row["request_contract_sha256"] != contract_hash:
            raise PermissionError("task/contract mismatch")
        if request_digest(row, contract_hash) != row["request_sha256"]:
            raise PermissionError("request hash mismatch")
    completed_rows = read_jsonl(args.out)
    completed = {row["task_id"]: row for row in completed_rows if not row.get("error")}

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    torch.cuda.set_device(0)
    processor = AutoProcessor.from_pretrained(contract["model_path"], trust_remote_code=True)
    engine = LLM(
        model=contract["model_path"],
        tensor_parallel_size=1,
        gpu_memory_utilization=0.78,
        trust_remote_code=True,
        max_model_len=int(contract["max_model_len"]),
        limit_mm_per_prompt={"image": int(contract["maximum_images_per_prompt"])},
        seed=int(contract["seed"]),
    )
    sampling = SamplingParams(
        temperature=float(contract["temperature"]),
        top_p=float(contract["top_p"]),
        max_tokens=int(contract["max_new_tokens"]),
        stop_token_ids=[],
    )
    failures: dict[str, int] = {}
    for task in tasks:
        if task["task_id"] in completed:
            continue
        for attempt in range(1, args.max_infrastructure_attempts + 1):
            started = time.perf_counter()
            try:
                messages = messages_for(task["model_image_paths"], task["model_input_prompt"])
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                multimodal: dict[str, Any] = {"image": image_inputs}
                if video_inputs is not None:
                    multimodal["video"] = video_inputs
                result = engine.generate([{
                    "prompt": prompt,
                    "multi_modal_data": multimodal,
                    "mm_processor_kwargs": video_kwargs,
                }], sampling_params=sampling, use_tqdm=False)
                response = result[0].outputs[0].text
                output = dict(task)
                output.update({
                    "attempt_number": attempt,
                    "raw_backend_response": response,
                    "model_response": response,
                    "api_mode": "qwen3vl8b_nonlocal_multi_image_vllm",
                    "gpu_id": args.gpu,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "completed_at_unix": int(time.time()),
                })
                append_jsonl(args.out, output)
                append_jsonl(args.attempt_ledger, {
                    "task_id": task["task_id"], "status": "success", "attempt_number": attempt,
                    "latency_ms": output["latency_ms"], "completed_at_unix": output["completed_at_unix"],
                })
                completed[task["task_id"]] = output
                break
            except Exception as error:
                failures[task["task_id"]] = attempt
                append_jsonl(args.attempt_ledger, {
                    "task_id": task["task_id"], "status": "infrastructure_error", "attempt_number": attempt,
                    "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                    "completed_at_unix": int(time.time()),
                })
        if task["task_id"] not in completed:
            break

    summary = {
        "schema_version": "nonlocal-support-pairwise-pilot-run-summary-1",
        "task_rows": len(tasks),
        "success_rows": len(completed),
        "unfinished_rows": len(tasks) - len(completed),
        "infrastructure_attempts_exhausted": sorted(
            ident for ident, attempt in failures.items()
            if attempt >= args.max_infrastructure_attempts and ident not in completed
        ),
        "gpu_id": args.gpu,
        "test_used": False,
    }
    summary["passed"] = summary["success_rows"] == len(tasks) and not summary["infrastructure_attempts_exhausted"]
    write_json(args.summary, summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
