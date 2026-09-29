#!/usr/bin/env python3
"""Execute the MAP core inside standalone SSRR v2.1 Validation.

The supervisor deliberately has no benchmark-task dispatch.  It compiles one
public-contract graph, executes every observation as evidence, and admits at
most one terminal composer only for evidence states that remain open.  Existing
outputs are retained only when task identity, prompt bytes, image content, and
the deterministic generation contract match.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve()
TOOLS = HERE.parent

RUN = Path(os.environ.get("MAP_RUN_DIR", ""))
WORK = RUN / "work"
RAW = WORK / "raw"
AUDITS = RUN / "audits"
LOGS = RUN / "logs"
STATUS = RUN / "supervisor_status.json"

DATASET = Path(os.environ.get("MAP_DATASET", ""))
MANIFEST = Path(os.environ.get("MAP_MANIFEST", ""))
PHASE0 = Path(os.environ.get("MAP_PHASE0", ""))
EVALUATE = Path(os.environ.get("MAP_EVALUATE", ""))
MODEL_PATH = Path(os.environ.get("MAP_MODEL_PATH", ""))

SUPPORT_RUNTIME = TOOLS / "support_runtime.py"
CLEAN_RUNNER = TOOLS / "run_qwen_tasks.py"

RUN_ID = os.environ.get("MAP_RUN_ID", "map_validation")
VARIANT_ID = os.environ.get("MAP_VARIANT_ID", "map_full")

OBSERVER_GENERATION_CONFIGS: dict[str, dict[str, float | int]] = {
    "STANDARD": {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "QUERY_NATIVE": {
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
    },
}

QPY = Path(os.environ.get("MAP_PYTHON", sys.executable))
GPU_STRING = os.environ.get("MAP_GPUS", "0,1,2,3,4,5,6,7")
GPU_IDS = [int(value) for value in GPU_STRING.split(",") if value.strip()]

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from io_utils import (  # noqa: E402
    read_jsonl,
    request_digest,
    sha256_file,
    write_json,
    write_jsonl,
)
RESULT_FIELDS = (
    "raw_backend_response",
    "model_response",
    "api_mode",
    "worker",
    "gpu_id",
    "latency_ms",
    "attempt_number",
    "completed_at_unix",
)


def keyed_by_task(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ident = str(row.get("task_id") or "")
        if ident:
            output.setdefault(ident, []).append(row)
    return output


class MapSupervisor:
    def __init__(self) -> None:
        LOGS.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        AUDITS.mkdir(parents=True, exist_ok=True)
        RAW.mkdir(parents=True, exist_ok=True)
        if STATUS.is_file():
            try:
                prior = json.loads(STATUS.read_text(encoding="utf-8"))
                self.records = list(prior.get("records") or [])
            except (OSError, ValueError, TypeError):
                self.records = []

    def status(self, stage: str, state: str, **extra: Any) -> None:
        payload = {
            "run_id": RUN_ID,
            "stage": stage,
            "state": state,
            "updated_at_unix": int(time.time()),
            "records": self.records,
            **extra,
        }
        write_json(STATUS, payload)
        print(
            json.dumps({"stage": stage, "state": state, **extra}, ensure_ascii=False),
            flush=True,
        )

    def execute(self, stage: str, command: list[str]) -> None:
        log = LOGS / f"{stage}.log"
        self.status(stage, "running", command=command, log=str(log))
        started = time.time()
        with log.open("a", encoding="utf-8") as stream:
            stream.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            stream.flush()
            result = subprocess.run(
                command,
                cwd=RUN.parent,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        elapsed = round(time.time() - started, 3)
        if result.returncode:
            self.status(
                stage,
                "failed",
                returncode=result.returncode,
                elapsed_sec=elapsed,
                log=str(log),
            )
            raise RuntimeError(f"stage failed: {stage}; see {log}")
        self.records.append(
            {"stage": stage, "elapsed_sec": elapsed, "returncode": 0, "log": str(log)}
        )
        self.status(stage, "completed", elapsed_sec=elapsed, log=str(log))

    def compile_public_graph(self) -> None:
        self.execute(
            "compile_public_contract",
            [
                str(QPY),
                str(TOOLS / "prepare_public.py"),
                "--manifest",
                str(MANIFEST),
                "--dataset-root",
                str(DATASET),
                "--public-out",
                str(WORK / "public_contract.jsonl"),
                "--private-binding-out",
                str(RUN / "private_reporting_binding.jsonl"),
                "--audit-out",
                str(RUN / "PUBLIC_BOUNDARY_AUDIT.json"),
                "--identity-map",
                str(RUN / "private_opaque_identity_map.jsonl"),
            ],
        )
        self.execute(
            "compile_one_runtime_graph",
            [
                str(QPY),
                str(TOOLS / "compile_runtime.py"),
                "--public",
                str(WORK / "public_contract.jsonl"),
                "--out",
                str(WORK / "runtime_manifest.jsonl"),
                "--audit",
                str(RUN / "RUNTIME_CONTRACT_AUDIT.json"),
            ],
        )

    def sanitize(self, stage: str, source: Path, target: Path) -> None:
        self.execute(
            f"sanitize_{stage}",
            [
                str(QPY),
                str(TOOLS / "sanitize_tasks.py"),
                "--input",
                str(source),
                "--out",
                str(target),
                "--audit",
                str(AUDITS / f"{stage}_model_boundary.json"),
            ],
        )

    def bind_request_contract(
        self,
        stage: str,
        tasks_path: Path,
        tokens: int,
        temperature: float,
        top_p: float,
        required_observer_profile: str | None = None,
    ) -> tuple[Path, str, Path]:
        if required_observer_profile is not None:
            if required_observer_profile not in OBSERVER_GENERATION_CONFIGS:
                raise ValueError(
                    f"unsupported observer generation profile: {required_observer_profile}"
                )
            expected = OBSERVER_GENERATION_CONFIGS[required_observer_profile]
            actual = {
                "max_new_tokens": int(tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
            if actual != expected:
                raise ValueError(
                    f"{stage}: observer profile parameters differ: {actual} != {expected}"
                )
        value = {
            "stage": stage,
            "model_path": str(MODEL_PATH.resolve()),
            "max_new_tokens": int(tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "gpu_memory_utilization": 0.70,
            "seed": 42,
        }
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        contract_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        value["request_contract_sha256"] = contract_sha
        contract_path = RUN / "request_contracts" / f"{stage}.json"
        write_json(contract_path, value)
        tasks = read_jsonl(tasks_path)
        task_ids = [str(row.get("task_id") or "") for row in tasks]
        if len(tasks) != len(set(task_ids)) or "" in task_ids:
            raise ValueError(f"{stage}: task IDs are missing or duplicated")
        output: list[dict[str, Any]] = []
        for source in tasks:
            row = dict(source)
            row["request_contract_sha256"] = contract_sha
            row["generation_config"] = {
                "max_new_tokens": int(tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "seed": 42,
                "max_model_len": 128000,
            }
            row["request_sha256"] = request_digest(row, contract_sha)
            output.append(row)
        bound_tasks_path = RUN / "bound_tasks" / f"{stage}.jsonl"
        write_jsonl(bound_tasks_path, output)
        return contract_path, contract_sha, bound_tasks_path
    def canonicalize_results(
        self,
        stage: str,
        tasks_path: Path,
        output_path: Path,
        contract_sha: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tasks = read_jsonl(tasks_path)
        results = keyed_by_task(read_jsonl(output_path, recover_truncated_tail=True))
        selected: list[dict[str, Any]] = []
        for task in tasks:
            ident = str(task["task_id"])
            valid = [
                row
                for row in results.get(ident, [])
                if not row.get("error") and row.get("model_response") is not None
            ]
            if not valid:
                continue
            row = dict(task)
            for key in RESULT_FIELDS:
                if key in valid[-1]:
                    row[key] = valid[-1][key]
            selected.append(row)
        write_jsonl(output_path, selected)
        missing = [
            row for row in tasks
            if str(row["task_id"]) not in {str(x["task_id"]) for x in selected}
        ]
        return selected, missing
    def verify_exact_output(
        self, stage: str, tasks_path: Path, output_path: Path, contract_sha: str
    ) -> None:
        selected, missing = self.canonicalize_results(stage, tasks_path, output_path, contract_sha)
        if missing:
            raise RuntimeError(f"{stage}: {len(missing)} results missing")
    def gpu(
        self,
        stage: str,
        tasks: Path,
        output: Path,
        tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        required_observer_profile: str | None = None,
    ) -> None:
        contract_path, contract_sha, bound_tasks = self.bind_request_contract(
            stage, tasks, tokens, temperature, top_p, required_observer_profile
        )
        successes, missing = self.canonicalize_results(stage, bound_tasks, output, contract_sha)
        launch_number = 1
        while missing:
            if launch_number > 5:
                raise RuntimeError(f"{stage}: five infrastructure attempts exhausted")
            batch = []
            for source in missing:
                row = dict(source)
                row["attempt_number"] = launch_number
                batch.append(row)
            batch_path = (
                RUN / "attempt_batches" / f"{stage}.launch{launch_number:03d}.jsonl"
            )
            write_jsonl(batch_path, batch)
            launch_id = f"{stage}.launch{launch_number:03d}"
            summary_path = RUN / "summaries" / f"{launch_id}.json"
            command = [
                str(QPY),
                str(CLEAN_RUNNER),
                "--tasks", str(batch_path),
                "--out", str(output),
                "--attempt-ledger", str(RUN / "attempts" / f"{stage}.jsonl"),
                "--summary-out", str(summary_path),
                "--launch-id", launch_id,
                "--request-contract", str(contract_path),
                "--model", str(MODEL_PATH),
                "--gpus", GPU_STRING,
                "--max-new-tokens", str(tokens),
                "--temperature", str(temperature),
                "--top-p", str(top_p),
                "--gpu-memory-utilization", "0.70",
                "--no-progress-timeout-sec", "3600",
            ]
            try:
                self.execute(f"{stage}_launch{launch_number:03d}", command)
            except RuntimeError:
                pass
            successes, missing = self.canonicalize_results(
                stage, bound_tasks, output, contract_sha
            )
            self.status(
                stage,
                "launch_checked",
                expected=len(read_jsonl(bound_tasks)),
                exact_successes=len(successes),
                remaining=len(missing),
                launch_number=launch_number,
            )
            launch_number += 1
        self.verify_exact_output(stage, bound_tasks, output, contract_sha)
    def prepare_main_observations(self) -> None:
        manifest = WORK / "runtime_manifest.jsonl"
        self.execute(
            "prepare_global_and_support_plan",
            [
                str(QPY),
                str(SUPPORT_RUNTIME),
                "prepare",
                "--manifest",
                str(manifest),
                "--out-dir",
                str(RAW / "main"),
            ],
        )
        self.sanitize(
            "native", RAW / "main/native_tasks.jsonl", WORK / "native_tasks.jsonl"
        )
        self.sanitize(
            "planner",
            RAW / "main/planner_tasks.jsonl",
            WORK / "planner_tasks.jsonl",
        )

    def partition_stage3_observer_profiles(self) -> dict[str, Path]:
        """Partition local observers only by the compiled public profile."""

        source_path = WORK / "stage3_tasks.jsonl"
        rows = read_jsonl(source_path)
        task_ids = [str(row.get("task_id") or "") for row in rows]
        if not rows or "" in task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("stage3 observer tasks lack unique task IDs")
        grouped: dict[str, list[dict[str, Any]]] = {
            name: [] for name in OBSERVER_GENERATION_CONFIGS
        }
        errors: list[dict[str, str]] = []
        for row in rows:
            profile = str(row.get("observer_generation_profile") or "")
            if profile not in grouped:
                errors.append(
                    {
                        "task_id": str(row.get("task_id") or ""),
                        "error": f"unsupported_profile:{profile}",
                    }
                )
                continue
            grouped[profile].append(row)
        paths: dict[str, Path] = {}
        for profile, profile_rows in grouped.items():
            path = (
                WORK
                / "stage3_profiles"
                / f"{profile.lower()}_tasks.jsonl"
            )
            write_jsonl(path, profile_rows)
            paths[profile] = path
        observed_union = {
            str(row["task_id"])
            for profile_rows in grouped.values()
            for row in profile_rows
        }
        audit = {
            "schema_version": "map-v2.2-observer-partition-audit-1",
            "source_tasks": len(rows),
            "source_unique_task_ids": len(set(task_ids)),
            "profile_counts": {
                profile: len(profile_rows)
                for profile, profile_rows in sorted(grouped.items())
            },
            "profile_generation_contracts": OBSERVER_GENERATION_CONFIGS,
            "partition_errors": errors,
            "partitions_disjoint": sum(len(value) for value in grouped.values())
            == len(observed_union),
            "partition_union_equals_source": observed_union == set(task_ids),
            "task_or_metric_selector_used": False,
            "test_used": False,
        }
        audit["passed"] = (
            not errors
            and all(audit["profile_counts"].values())
            and audit["partitions_disjoint"]
            and audit["partition_union_equals_source"]
        )
        write_json(AUDITS / "stage3_observer_partition.json", audit)
        if not audit["passed"]:
            raise RuntimeError(f"stage3 observer partition failed: {audit}")
        return paths

    def merge_stage3_observer_profiles(
        self, profile_outputs: dict[str, Path]
    ) -> Path:
        """Merge complete profile partitions by task identity without selection."""

        task_rows = read_jsonl(WORK / "stage3_tasks.jsonl")
        ordered_task_ids = [str(row["task_id"]) for row in task_rows]
        merged_by_id: dict[str, dict[str, Any]] = {}
        duplicates: list[str] = []
        profile_counts: dict[str, int] = {}
        profile_errors: list[dict[str, str]] = []
        for profile, path in sorted(profile_outputs.items()):
            rows = read_jsonl(path)
            profile_counts[profile] = len(rows)
            for row in rows:
                task_id = str(row.get("task_id") or "")
                if row.get("observer_generation_profile") != profile:
                    profile_errors.append(
                        {
                            "task_id": task_id,
                            "expected": profile,
                            "observed": str(
                                row.get("observer_generation_profile") or ""
                            ),
                        }
                    )
                if task_id in merged_by_id:
                    duplicates.append(task_id)
                else:
                    merged_by_id[task_id] = row
        source_ids = set(ordered_task_ids)
        merged_ids = set(merged_by_id)
        output_path = WORK / "stage3_results.jsonl"
        if not duplicates and not profile_errors and merged_ids == source_ids:
            write_jsonl(
                output_path,
                [merged_by_id[task_id] for task_id in ordered_task_ids],
            )
        audit = {
            "schema_version": "map-v2.2-observer-merge-audit-1",
            "source_tasks": len(task_rows),
            "merged_rows": len(merged_by_id),
            "profile_counts": dict(sorted(profile_counts.items())),
            "duplicate_task_ids": sorted(set(duplicates)),
            "profile_errors": profile_errors,
            "missing_task_ids": sorted(source_ids - merged_ids),
            "unexpected_task_ids": sorted(merged_ids - source_ids),
            "merge_key": "task_id",
            "response_selector_used": False,
            "historical_result_input_used": False,
            "test_used": False,
        }
        audit["passed"] = (
            not duplicates
            and not profile_errors
            and merged_ids == source_ids
            and len(task_rows) == len(merged_by_id)
        )
        write_json(AUDITS / "stage3_observer_merge.json", audit)
        if not audit["passed"]:
            raise RuntimeError(f"stage3 observer merge failed: {audit}")
        return output_path

    def observe_native_views(self) -> None:
        manifest = WORK / "runtime_manifest.jsonl"
        self.gpu(
            "native",
            WORK / "native_tasks.jsonl",
            WORK / "native_results.jsonl",
            8192,
        )
        self.gpu(
            "planner",
            WORK / "planner_tasks.jsonl",
            WORK / "planner_results.jsonl",
            256,
        )
        self.execute(
            "complete_public_support_plan",
            [
                str(QPY),
                str(SUPPORT_RUNTIME),
                "complete-planner",
                "--manifest",
                str(manifest),
                "--results",
                str(WORK / "planner_results.jsonl"),
                "--skipped",
                str(RAW / "main/planner_skipped.jsonl"),
                "--out",
                str(WORK / "planner_complete.jsonl"),
                "--audit",
                str(AUDITS / "planner_completion.json"),
            ],
        )
        self.execute(
            "construct_native_view_supports",
            [
                str(QPY),
                str(SUPPORT_RUNTIME),
                "build-rois",
                "--planner-complete",
                str(WORK / "planner_complete.jsonl"),
                "--out",
                str(RAW / "roi_manifest.jsonl"),
                "--audit",
                str(AUDITS / "roi_construction.json"),
                "--nms-iou",
                "0.2",
            ],
        )
        self.execute(
            "enforce_public_view_budget",
            [
                str(QPY),
                str(TOOLS / "cap_views.py"),
                "--manifest",
                str(manifest),
                "--input",
                str(RAW / "roi_manifest.jsonl"),
                "--out",
                str(WORK / "rois/roi_manifest.jsonl"),
                "--audit",
                str(AUDITS / "view_budget.json"),
            ],
        )
        self.execute(
            "export_registered_native_views",
            [
                str(QPY),
                str(TOOLS / "export_views.py"),
                "--roi-manifest",
                str(WORK / "rois/roi_manifest.jsonl"),
                "--out-dir",
                str(WORK / "roi_crops"),
                "--workers",
                "8",
                "--resume",
            ],
        )
        self.execute(
            "prepare_native_local_observers",
            [
                str(QPY),
                str(SUPPORT_RUNTIME),
                "build-local",
                "--rois",
                str(WORK / "rois/roi_manifest.jsonl"),
                "--crop-dir",
                str(WORK / "roi_crops"),
                "--out",
                str(RAW / "stage3_tasks.jsonl"),
                "--audit",
                str(AUDITS / "local_observer_prepare.json"),
            ],
        )
        self.sanitize(
            "stage3", RAW / "stage3_tasks.jsonl", WORK / "stage3_tasks.jsonl"
        )
        profile_tasks = self.partition_stage3_observer_profiles()
        profile_outputs: dict[str, Path] = {}
        for profile in ("STANDARD", "QUERY_NATIVE"):
            config = OBSERVER_GENERATION_CONFIGS[profile]
            stage = f"stage3_{profile.lower()}"
            output = (
                WORK / "stage3_profiles" / f"{profile.lower()}_results.jsonl"
            )
            self.gpu(
                stage,
                profile_tasks[profile],
                output,
                int(config["max_new_tokens"]),
                float(config["temperature"]),
                float(config["top_p"]),
                required_observer_profile=profile,
            )
            profile_outputs[profile] = output
        stage3_results = self.merge_stage3_observer_profiles(profile_outputs)
        self.execute(
            "strict_source_registration",
            [
                str(QPY),
                str(TOOLS / "register_outputs.py"),
                "--results",
                str(stage3_results),
                "--out",
                str(WORK / "stage3_mapped.jsonl"),
                "--audit",
                str(AUDITS / "source_registration.json"),
            ],
        )
        self.execute(
            "prepare_method_support_canvas",
            [
                str(QPY),
                str(TOOLS / "prepare_source_canvas.py"),
                "--manifest",
                str(manifest),
                "--rois",
                str(WORK / "rois/roi_manifest.jsonl"),
                "--out-dir",
                str(WORK / "source_canvases/images"),
                "--out",
                str(WORK / "source_canvases/source_canvas_manifest.jsonl"),
                "--audit",
                str(AUDITS / "source_canvas.json"),
            ],
        )

    def compose_and_evaluate(self) -> None:
        manifest = WORK / "runtime_manifest.jsonl"
        self.execute(
            "aggregate_typed_source_evidence",
            [
                str(QPY),
                str(TOOLS / "composer_runtime.py"),
                "aggregate",
                "--manifest",
                str(manifest),
                "--native",
                str(WORK / "native_results.jsonl"),
                "--mapped",
                str(WORK / "stage3_mapped.jsonl"),
                "--out",
                str(WORK / "typed_evidence_states.jsonl"),
                "--audit",
                str(AUDITS / "typed_aggregation.json"),
            ],
        )
        closure = WORK / "evidence_closure_decisions.jsonl"
        self.execute(
            "decide_public_evidence_closure",
            [
                str(QPY),
                str(TOOLS / "composer_runtime.py"),
                "closure",
                "--manifest",
                str(manifest),
                "--states",
                str(WORK / "typed_evidence_states.jsonl"),
                "--mapped",
                str(WORK / "stage3_mapped.jsonl"),
                "--out",
                str(closure),
                "--audit",
                str(AUDITS / "evidence_closure.json"),
            ],
        )
        self.execute(
            "prepare_sole_terminal_composer",
            [
                str(QPY),
                str(TOOLS / "composer_runtime.py"),
                "prepare",
                "--manifest",
                str(manifest),
                "--states",
                str(WORK / "typed_evidence_states.jsonl"),
                "--closure",
                str(closure),
                "--source-canvases",
                str(WORK / "source_canvases/source_canvas_manifest.jsonl"),
                "--out",
                str(RAW / "composer_tasks.jsonl"),
                "--audit",
                str(AUDITS / "composer_prepare.json"),
            ],
        )
        self.sanitize(
            "composer", RAW / "composer_tasks.jsonl", WORK / "composer_tasks.jsonl"
        )
        self.gpu(
            "composer",
            WORK / "composer_tasks.jsonl",
            WORK / "composer_results.jsonl",
            1024,
        )
        canonical = RUN / "variants" / VARIANT_ID / "canonical_predictions.jsonl"
        self.execute(
            "materialize_posthoc_reporting_identity",
            [
                str(QPY),
                str(TOOLS / "composer_runtime.py"),
                "materialize",
                "--manifest",
                str(manifest),
                "--binding",
                str(RUN / "private_reporting_binding.jsonl"),
                "--native",
                str(WORK / "native_results.jsonl"),
                "--planner",
                str(WORK / "planner_complete.jsonl"),
                "--mapped",
                str(WORK / "stage3_mapped.jsonl"),
                "--states",
                str(WORK / "typed_evidence_states.jsonl"),
                "--closure",
                str(closure),
                "--composer",
                str(WORK / "composer_results.jsonl"),
                "--phase0",
                str(PHASE0),
                "--run-id",
                RUN_ID,
                "--out",
                str(canonical),
                "--audit",
                str(RUN / "MATERIALIZATION_AUDIT.json"),
            ],
        )
        self.execute(
            "formal_validation_evaluation",
            [
                str(QPY),
                str(EVALUATE),
                "--split",
                "Validation",
                "--run-dir",
                str(RUN),
                "--variants",
                VARIANT_ID,
                "--gpus",
                GPU_STRING,
            ],
        )
        report = json.loads((RUN / "FORMAL_EVALUATION.json").read_text(encoding="utf-8"))
        if not report.get("all_repeat_checks_passed"):
            raise RuntimeError("formal Validation repeat-evaluation audit failed")

    def run(self, *, preflight_only: bool = False) -> None:
        if not GPU_IDS or len(GPU_IDS) != len(set(GPU_IDS)):
            raise ValueError(f"GPU IDs must be unique integers: {GPU_IDS}")
        self.compile_public_graph()
        self.prepare_main_observations()
        self.observe_native_views()
        self.compose_and_evaluate()
        self.status("validation", "complete")
