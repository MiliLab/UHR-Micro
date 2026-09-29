#!/usr/bin/env python3
"""Run the MAP method on InternVL3.5-8B (Validation split).

Reuses the Qwen MAP supervisor (map/qwen3vl/run_validation.py) and swaps only
the execution boundary: the GPU runner becomes run_internvl_tasks.py under the
InternVL python environment, and the scalar-operand multi-image stage becomes
run_multi_image_tasks_internvl.py.  All prompts, contracts, budgets, operators
and fallback logic are shared with the Qwen supervisor.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
RUNTIME_TOOLS = HERE.parents[1] / "runtime" / "tools"
QWEN_TOOLS = HERE.parents[1] / "qwen3vl"
for path in (RUNTIME_TOOLS, QWEN_TOOLS, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import importlib.util  # noqa: E402
import map_core_supervisor as core  # noqa: E402
from io_utils import read_jsonl, write_jsonl  # noqa: E402

_qwen_spec = importlib.util.spec_from_file_location(
    "qwen_map_validation", QWEN_TOOLS / "run_validation.py"
)
qwen_val = importlib.util.module_from_spec(_qwen_spec)
_qwen_spec.loader.exec_module(qwen_val)


INTERNVL_PY = Path(os.environ.get("MAP_INTERNVL_PYTHON", sys.executable))
INTERNVL_MODEL = Path(os.environ.get("MAP_INTERNVL_MODEL_PATH", ""))
INTERNVL_RUNNER = HERE / "run_internvl_tasks.py"
INTERNVL_MULTI = HERE / "run_multi_image_tasks_internvl.py"
INTERNVL_GLOBAL_VIEW_MAX_SIDE = int(os.environ.get("MAP_INTERNVL_GLOBAL_VIEW_MAX_SIDE", "0"))


class InternVLSupervisor(qwen_val.MAPSupervisor):
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
                core.RUN / "attempt_batches" / f"{stage}.launch{launch_number:03d}.jsonl"
            )
            write_jsonl(batch_path, batch)
            launch_id = f"{stage}.launch{launch_number:03d}"
            summary_path = core.RUN / "summaries" / f"{launch_id}.json"
            command = [
                str(INTERNVL_PY),
                str(INTERNVL_RUNNER),
                "--tasks", str(batch_path),
                "--out", str(output),
                "--attempt-ledger", str(core.RUN / "attempts" / f"{stage}.jsonl"),
                "--summary-out", str(summary_path),
                "--launch-id", launch_id,
                "--request-contract", str(contract_path),
                "--model", str(INTERNVL_MODEL),
                "--gpus", core.GPU_STRING,
                "--max-new-tokens", str(tokens),
                "--temperature", str(temperature),
                "--top-p", str(top_p),
                "--gpu-memory-utilization", "0.70",
                "--no-progress-timeout-sec", "3600",
                "--global-view-max-side", str(INTERNVL_GLOBAL_VIEW_MAX_SIDE),
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

    def compile_and_run_algebra(self, support: Path) -> None:
        root = core.WORK / "map"
        complexes = root / "support_complexes"
        actions = root / "action_inputs"
        operands = root / "pairwise_operands"
        self.execute("compile_source_support_complexes", [
            str(core.QPY), str(QWEN_TOOLS / "compile_support_complexes.py"),
            "--open-manifest", str(core.WORK / "support_recurrence/open_manifest.jsonl"),
            "--mapped", str(core.WORK / "stage3_mapped.jsonl"),
            "--overlays", str(core.WORK / "support_recurrence/overlays.jsonl"),
            "--controller-admitted", str(support / "decisions.jsonl"),
            "--output-dir", str(complexes),
        ])
        self.execute("materialize_source_support_actions", [
            str(core.QPY), str(QWEN_TOOLS / "materialize_action_inputs.py"),
            "--actions", str(complexes / "excluded_state_actions.jsonl"),
            "--output-dir", str(actions),
        ])
        self.execute("prepare_map_scalar_operands", [
            str(core.QPY), str(QWEN_TOOLS / "prepare_scalar_operands.py"),
            "--materialized-actions", str(actions / "materialized_actions.jsonl"),
            "--output-dir", str(operands),
        ])
        self.execute("run_map_scalar_operands", [
            str(core.QPY), str(QWEN_TOOLS / "run_multi_image_parallel.py"),
            "--python", str(INTERNVL_PY),
            "--runner", str(INTERNVL_MULTI),
            "--tasks", str(operands / "operand_tasks.jsonl"),
            "--contract", str(operands / "REQUEST_CONTRACT.json"),
            "--output-dir", str(operands / "parallel"),
            "--gpus", core.GPU_STRING,
            "--max-infrastructure-attempts", "5",
        ])
        baseline = core.RUN / "variants/support_recurrence_only/canonical_predictions.jsonl"
        scalar = core.RUN / "variants/scalar_algebra_only/canonical_predictions.jsonl"
        self.execute("materialize_map_scalar_algebra", [
            str(core.QPY), str(QWEN_TOOLS / "build_scalar_algebra_canonical.py"),
            "--baseline", str(baseline),
            "--operand-outputs", str(operands / "parallel/outputs.jsonl"),
            "--out", str(scalar),
            "--audit", str(root / "SCALAR_ALGEBRA_CANONICAL_AUDIT.json"),
        ])
        full = core.RUN / "variants/map_full/canonical_predictions.jsonl"
        geometry_audit = root / "GEOMETRY_ALGEBRA_CANONICAL_AUDIT.json"
        self.execute("materialize_map_geometry_algebra", [
            str(core.QPY), str(QWEN_TOOLS / "build_geometry_algebra.py"),
            "--carrier-baseline", str(scalar),
            "--terminal-base", str(support / "terminal_base.jsonl"),
            "--terminal-tasks", str(support / "terminal_tasks.jsonl"),
            "--mapped-observations", str(support / "mapped.jsonl"),
            "--out", str(full),
            "--audit", str(geometry_audit),
            "--iou-threshold", "0.5",
        ])
        rows = read_jsonl(full)
        for row in rows:
            if row.get("prompt_id") == "nonlocal-support-closure-v1:source-registered-geometry-algebra":
                row["prompt_id"] = "map-v1:geometry-set-union-nms"
        write_jsonl(full, rows)

    def run(self, *, preflight_only: bool = False) -> None:
        if not core.GPU_IDS or len(core.GPU_IDS) != len(set(core.GPU_IDS)):
            raise ValueError(f"GPU IDs must be unique integers: {core.GPU_IDS}")
        self.compile_public_graph()
        self.prepare_main_observations()
        self.observe_native_views()
        self.compose_map_and_evaluate()
        self.status("validation", "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    InternVLSupervisor().run(preflight_only=args.preflight_only)
