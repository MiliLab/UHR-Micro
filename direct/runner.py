from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import traceback
from pathlib import Path
from typing import Any

from .constants import FORBIDDEN_GT_KEYS, PREDICTION_SCHEMA_VERSION
from .io import assert_no_forbidden_keys, read_json, write_json, write_jsonl
from .manifest import load_input_manifest, resolve_image


def _safe_stem(qa_relpath: str) -> str:
    import hashlib

    digest = hashlib.sha256(qa_relpath.encode("utf-8")).hexdigest()
    return digest


def _attempt_path(run_root: Path, qa_relpath: str, attempt: int) -> Path:
    return run_root / "attempts" / f"{_safe_stem(qa_relpath)}.attempt-{attempt}.json"


def _latest_attempt(run_root: Path, qa_relpath: str, max_attempts: int) -> dict[str, Any] | None:
    for attempt in range(max_attempts, 0, -1):
        path = _attempt_path(run_root, qa_relpath, attempt)
        if path.is_file():
            row = read_json(path)
            if row.get("qa_relpath") != qa_relpath or row.get("attempt") != attempt:
                raise ValueError(f"attempt identity mismatch: {path}")
            return row
    return None


def _build_prediction(
    task: dict[str, Any],
    result,
    run_config: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "source_question_sha256": task["source_question_sha256"],
            "model_question_sha256": task["model_question_sha256"],
            "question_coordinate_protocol": task["question_coordinate_protocol"],
            "coordinate_transforms": task["coordinate_transforms"],
            "image_sha256": task["image_sha256"],
            "split_position": task["split_position"],
        }
    )
    row = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "run_id": run_config["run_id"],
        "phase0_contract_sha256": run_config["phase0_contract_sha256"],
        "benchmark_manifest_sha256": run_config["benchmark_manifest_sha256"],
        "qa_relpath": task["qa_relpath"],
        "task_key": task["task_key"],
        "model_registry_id": run_config["model_registry_id"],
        "attempt": attempt,
        "runtime_status": result.runtime_status,
        "raw_response": result.raw_response,
        "final_response": result.final_response,
        "source_image_size": task["source_image_size"],
        "transmitted_image_size": result.transmitted_image_size,
        "model_input_size": result.model_input_size,
        "requested_coordinate_protocol": run_config["requested_coordinate_protocol"],
        "prompt_id": task["prompt_id"],
        "prompt_sha256": task["prompt_sha256"],
        "generation_config": run_config["generation_config"],
        "requested_model_id": None,
        "resolved_model_id": str(run_config["resolved_model_id"]),
        "endpoint_id": None,
        "api_mode": str(run_config.get("api_mode", "local_vllm")),
        "request_id": None,
        "logical_vlm_calls": result.logical_vlm_calls,
        "tool_calls": result.tool_calls,
        "trajectory": result.trajectory,
        "diagnostics": diagnostics,
    }
    assert_no_forbidden_keys(row, FORBIDDEN_GT_KEYS)
    return row


def _error_prediction(
    task: dict[str, Any], run_config: dict[str, Any], attempt: int, exc: BaseException,
    physical_gpu_ids: list[str], adapter_name: str,
) -> dict[str, Any]:
    from .adapters.base import AdapterResult

    result = AdapterResult(
        raw_response=None,
        final_response=None,
        runtime_status="runtime_or_api_error",
        transmitted_image_size=None,
        model_input_size=None,
        logical_vlm_calls=int(getattr(exc, "logical_vlm_calls", 0)),
        diagnostics={
            "backend": str(run_config.get("api_mode", "local_vllm")).removeprefix("local_"),
            "adapter": adapter_name,
            "physical_gpu_ids": physical_gpu_ids,
            "physical_gpu_id": physical_gpu_ids[0] if len(physical_gpu_ids) == 1 else None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    return _build_prediction(task, result, run_config, attempt)


def _worker(
    worker_id: int,
    physical_gpu_ids: list[str],
    task_queue,
    result_queue,
    run_config: dict[str, Any],
    adapter_config: dict[str, Any],
    model_path: str,
    image_root: str,
    run_root: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_gpu_ids)
    adapter_kind = str(adapter_config.get("adapter_class", "Qwen3VLAdapter"))
    if adapter_kind == "Qwen3VLAdapter":
        from .adapters.qwen3_vl import Qwen3VLAdapter

        adapter = Qwen3VLAdapter(Path(model_path), adapter_config, physical_gpu_ids)
    elif adapter_kind == "InternVLAdapter":
        from .adapters.internvl import InternVLAdapter

        adapter = InternVLAdapter(Path(model_path), adapter_config, physical_gpu_ids)
    else:
        raise ValueError(f"unsupported adapter_class: {adapter_kind}")

    try:
        adapter.load()
        result_queue.put(
            {
                "event": "worker_ready",
                "worker_id": worker_id,
                "gpu": physical_gpu_ids[0],
                "physical_gpu_ids": physical_gpu_ids,
                "pid": os.getpid(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "event": "worker_fatal",
                "worker_id": worker_id,
                "gpu": physical_gpu_ids[0],
                "physical_gpu_ids": physical_gpu_ids,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        task = task_queue.get()
        if task is None:
            break
        attempt = int(task.pop("_attempt"))
        try:
            task["image_abs_path"] = str(resolve_image(Path(image_root), task["image_relpath"]))
            result = adapter.generate(task)
            row = _build_prediction(task, result, run_config, attempt)
        except Exception as exc:
            row = _error_prediction(task, run_config, attempt, exc, physical_gpu_ids, adapter_kind)
        path = _attempt_path(Path(run_root), task["qa_relpath"], attempt)
        write_json(path, row)
        result_queue.put(
            {
                "event": "sample_complete",
                "worker_id": worker_id,
                "gpu": physical_gpu_ids[0],
                "physical_gpu_ids": physical_gpu_ids,
                "qa_relpath": task["qa_relpath"],
                "attempt": attempt,
                "runtime_status": row["runtime_status"],
            }
        )
    adapter.close()
    result_queue.put({
        "event": "worker_stopped", "worker_id": worker_id,
        "gpu": physical_gpu_ids[0], "physical_gpu_ids": physical_gpu_ids,
    })


def plan_pending(
    rows: list[dict[str, Any]], run_root: Path, max_attempts: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pending: list[dict[str, Any]] = []
    counts = {
        "total": len(rows), "success": 0, "terminal_model_outcome": 0,
        "failed_exhausted": 0, "pending": 0,
    }
    for row in rows:
        latest = _latest_attempt(run_root, row["qa_relpath"], max_attempts)
        if latest is not None:
            status = latest["runtime_status"]
            if status == "success":
                counts["success"] += 1
                continue
            if status in {"tool_max_turns", "empty_response", "postprocess_failure"}:
                counts["terminal_model_outcome"] += 1
                continue
        next_attempt = 1 if latest is None else int(latest["attempt"]) + 1
        if next_attempt > max_attempts:
            counts["failed_exhausted"] += 1
            continue
        item = dict(row)
        item["_attempt"] = next_attempt
        pending.append(item)
    counts["pending"] = len(pending)
    return pending, counts


def merge_attempts(
    rows: list[dict[str, Any]], run_root: Path, max_attempts: int, allow_incomplete: bool
) -> dict[str, Any]:
    canonical: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in rows:
        latest = _latest_attempt(run_root, task["qa_relpath"], max_attempts)
        if latest is None:
            missing.append(task["qa_relpath"])
        else:
            canonical.append(latest)
    if missing and not allow_incomplete:
        raise RuntimeError(f"cannot finalize: {len(missing)} samples have no attempt")
    write_jsonl(run_root / "canonical_predictions.jsonl", canonical)
    summary = {
        "total_manifest_rows": len(rows),
        "canonical_rows": len(canonical),
        "missing_rows": len(missing),
        "coverage_complete": not missing,
        "runtime_status_counts": {},
    }
    for row in canonical:
        status = row["runtime_status"]
        summary["runtime_status_counts"][status] = summary["runtime_status_counts"].get(status, 0) + 1
    write_json(run_root / "inference_summary.json", summary)
    return summary


def run_parallel(
    input_manifest: Path,
    run_config_path: Path,
    adapter_config_path: Path,
    model_path: Path,
    image_root: Path,
    physical_gpu_ids: list[str],
    scheduler: str,
    dry_run: bool,
) -> dict[str, Any]:
    rows = load_input_manifest(input_manifest)
    run_config = read_json(run_config_path)
    adapter_config = read_json(adapter_config_path)
    run_root = Path(run_config["output_root"]).resolve()
    max_attempts = int(run_config["retry_policy"]["max_attempts"])
    pending, counts = plan_pending(rows, run_root, max_attempts)
    if scheduler not in {"dynamic_queue", "static_round_robin", "static_image_hash"}:
        raise ValueError(f"unsupported scheduler: {scheduler}")
    if not physical_gpu_ids:
        raise ValueError("at least one GPU is required")
    if run_config["model_registry_id"] != adapter_config["model_registry_id"]:
        raise ValueError("run config and adapter config model_registry_id differ")
    if any(row["model_registry_id"] != run_config["model_registry_id"] for row in rows):
        raise ValueError("input manifest and run config model_registry_id differ")
    if any(row["requested_coordinate_protocol"] != run_config["requested_coordinate_protocol"] for row in rows):
        raise ValueError("input manifest and run config coordinate protocol differ")
    tensor_parallel_size = int(adapter_config["tensor_parallel_size"])
    if tensor_parallel_size < 1 or len(physical_gpu_ids) % tensor_parallel_size:
        raise ValueError("GPU count must be a positive multiple of tensor_parallel_size")
    worker_gpu_groups = [
        physical_gpu_ids[index:index + tensor_parallel_size]
        for index in range(0, len(physical_gpu_ids), tensor_parallel_size)
    ]
    if dry_run:
        return {
            "dry_run": True,
            "input_rows": len(rows),
            "pending": len(pending),
            "workers": len(worker_gpu_groups),
            "physical_gpu_ids": physical_gpu_ids,
            "worker_gpu_groups": worker_gpu_groups,
            "tensor_parallel_size": tensor_parallel_size,
            "scheduler": scheduler,
            "resume_counts": counts,
            "model_will_load": False,
            "generation_will_run": False,
        }
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_config.json", run_config)
    execution_plan = {
        "schema_version": "uhr-bench-v4-execution-plan-1",
        "run_id": run_config["run_id"],
        "input_manifest_realpath": str(input_manifest.resolve()),
        "input_row_count": len(rows),
        "pending_at_launch": len(pending),
        "resume_counts_at_launch": counts,
        "scheduler": scheduler,
        "physical_gpu_ids": physical_gpu_ids,
        "worker_gpu_groups": worker_gpu_groups,
        "tensor_parallel_size": tensor_parallel_size,
        "model_realpath": str(model_path.resolve()),
        "image_root_realpath": str(image_root.resolve()),
        "adapter_config_realpath": str(adapter_config_path.resolve()),
        "run_config_realpath": str(run_config_path.resolve()),
    }
    write_json(run_root / "execution_plan.json", execution_plan)
    if not pending:
        summary = merge_attempts(rows, run_root, max_attempts, allow_incomplete=False)
        summary.update(
            {
                "workers": 0,
                "physical_gpu_ids": physical_gpu_ids,
                "worker_gpu_groups": worker_gpu_groups,
                "tensor_parallel_size": tensor_parallel_size,
                "ready_workers": 0,
                "scheduler": scheduler,
                "resume_without_model_load": True,
            }
        )
        write_json(run_root / "inference_summary.json", summary)
        return summary

    ctx = mp.get_context("spawn")
    if scheduler in {"static_round_robin", "static_image_hash"}:
        task_queues = [ctx.Queue(maxsize=max(4, (len(pending) // len(worker_gpu_groups)) + 2)) for _ in worker_gpu_groups]
    else:
        shared_queue = ctx.Queue(maxsize=max(32, len(worker_gpu_groups) * 4))
        task_queues = [shared_queue for _ in worker_gpu_groups]
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_worker,
            args=(
                index,
                gpu_group,
                task_queues[index],
                result_queue,
                run_config,
                adapter_config,
                str(model_path.resolve()),
                str(image_root.resolve()),
                str(run_root),
            ),
            name=f"{adapter_config.get('model_registry_id', 'vlm')}-worker-{index}-gpus-{'-'.join(gpu_group)}",
        )
        for index, gpu_group in enumerate(worker_gpu_groups)
    ]
    for worker in workers:
        worker.start()
    ready = 0
    ready_events: list[dict[str, Any]] = []
    fatal: list[dict[str, Any]] = []
    startup_events = 0
    while startup_events < len(workers):
        try:
            event = result_queue.get(timeout=60)
        except queue.Empty:
            dead = [worker.name for worker in workers if not worker.is_alive() and worker.exitcode not in (None, 0)]
            if dead:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                raise RuntimeError(f"worker(s) died before startup report: {dead}")
            print(json.dumps({"event": "waiting_for_model_load", "ready": ready, "workers": len(workers)}), flush=True)
            continue
        startup_events += 1
        if event["event"] == "worker_ready":
            ready += 1
            ready_events.append(event)
        elif event["event"] == "worker_fatal":
            fatal.append(event)
        else:
            raise RuntimeError(f"unexpected event during worker startup: {event}")
    if fatal:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join()
        write_json(run_root / "worker_fatal_errors.json", {"errors": fatal})
        raise RuntimeError(f"{len(fatal)} worker(s) failed during model load; inference not started")
    gpu_snapshot = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip().splitlines()
    process_snapshot = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip().splitlines()
    write_json(
        run_root / "gpu_startup_snapshot.json",
        {
            "worker_ready_events": sorted(ready_events, key=lambda item: int(item["worker_id"])),
            "nvidia_smi_gpus": gpu_snapshot,
            "nvidia_smi_compute_apps": process_snapshot,
        },
    )
    if scheduler in {"static_round_robin", "static_image_hash"}:
        import hashlib
        for index, task in enumerate(pending):
            if scheduler == "static_image_hash":
                worker_index = int(hashlib.sha1(
                    task["image_relpath"].encode("utf-8")
                ).hexdigest()[:16], 16) % len(task_queues)
            else:
                worker_index = index % len(task_queues)
            task_queues[worker_index].put(task)
        for task_queue in task_queues:
            task_queue.put(None)
    else:
        for task in pending:
            shared_queue.put(task)
        for _ in workers:
            shared_queue.put(None)

    completed = 0
    while completed < len(pending):
        try:
            event = result_queue.get(timeout=60)
        except queue.Empty:
            dead = [worker.name for worker in workers if not worker.is_alive() and worker.exitcode not in (None, 0)]
            if dead:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                raise RuntimeError(f"worker(s) died during inference: {dead}")
            print(json.dumps({"event": "waiting_for_samples", "completed": completed, "pending_total": len(pending)}), flush=True)
            continue
        if event["event"] == "sample_complete":
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                print(json.dumps({"completed": completed, "pending_total": len(pending), **event}, ensure_ascii=False), flush=True)
    for worker in workers:
        worker.join()
    summary = merge_attempts(rows, run_root, max_attempts, allow_incomplete=False)
    summary.update(
        {
            "workers": len(workers),
            "physical_gpu_ids": physical_gpu_ids,
            "worker_gpu_groups": worker_gpu_groups,
            "tensor_parallel_size": tensor_parallel_size,
            "ready_workers": ready,
            "scheduler": scheduler,
        }
    )
    write_json(run_root / "inference_summary.json", summary)
    return summary
