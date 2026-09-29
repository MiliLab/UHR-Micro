#!/usr/bin/env python3
"""Run the MAP method once on the UHR-Bench Test split."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
RUNTIME_TOOLS = HERE.parents[1] / "runtime" / "tools"
for path in (RUNTIME_TOOLS, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_validation as val  # noqa: E402
import map_core_supervisor as core  # noqa: E402
from io_utils import read_jsonl, write_jsonl  # noqa: E402


TEST_MANIFEST = Path(os.environ.get("MAP_TEST_MANIFEST", core.MANIFEST))
FINAL_VARIANT = "map_full"


class TestSupervisor(val.MAPSupervisor):
    def compile_public_graph(self) -> None:
        self.execute("compile_test_public_contract", [
            str(core.QPY), str(core.TOOLS / "prepare_public.py"),
            "--manifest", str(TEST_MANIFEST), "--dataset-root", str(core.DATASET),
            "--public-out", str(core.WORK / "public_contract.jsonl"),
            "--private-binding-out", str(core.RUN / "private_reporting_binding.jsonl"),
            "--audit-out", str(core.RUN / "PUBLIC_BOUNDARY_AUDIT.json"),
            "--identity-map", str(core.RUN / "private_opaque_identity_map.jsonl"),
        ])
        self.execute("compile_test_runtime_graph", [
            str(core.QPY), str(core.TOOLS / "compile_runtime.py"),
            "--public", str(core.WORK / "public_contract.jsonl"),
            "--out", str(core.WORK / "runtime_manifest.jsonl"),
            "--audit", str(core.RUN / "RUNTIME_CONTRACT_AUDIT.json"),
        ])

    def compile_and_run_algebra(self, support: Path) -> None:
        root = core.WORK / "map"
        complexes = root / "support_complexes"
        actions = root / "action_inputs"
        operands = root / "pairwise_operands"
        self.execute("compile_source_support_complexes", [
            str(core.QPY), str(HERE.parent / "compile_support_complexes.py"),
            "--open-manifest", str(core.WORK / "support_recurrence/open_manifest.jsonl"),
            "--mapped", str(core.WORK / "stage3_mapped.jsonl"),
            "--overlays", str(core.WORK / "support_recurrence/overlays.jsonl"),
            "--controller-admitted", str(support / "decisions.jsonl"),
            "--output-dir", str(complexes),
        ])
        self.execute("materialize_source_support_actions", [
            str(core.QPY), str(HERE.parent / "materialize_action_inputs.py"),
            "--actions", str(complexes / "excluded_state_actions.jsonl"),
            "--output-dir", str(actions),
        ])
        self.execute("prepare_map_scalar_operands", [
            str(core.QPY), str(HERE.parent / "prepare_scalar_operands.py"),
            "--materialized-actions", str(actions / "materialized_actions.jsonl"),
            "--output-dir", str(operands),
        ])
        self.execute("run_map_scalar_operands", [
            str(core.QPY), str(HERE.parent / "run_multi_image_parallel.py"),
            "--python", str(core.QPY),
            "--runner", str(HERE.parent / "run_multi_image_tasks.py"),
            "--tasks", str(operands / "operand_tasks.jsonl"),
            "--contract", str(operands / "REQUEST_CONTRACT.json"),
            "--output-dir", str(operands / "parallel"),
            "--gpus", core.GPU_STRING, "--max-infrastructure-attempts", "5",
        ])
        baseline = core.RUN / "variants/support_recurrence_only/canonical_predictions.jsonl"
        scalar = core.RUN / "variants/scalar_algebra_only/canonical_predictions.jsonl"
        self.execute("materialize_map_scalar_algebra", [
            str(core.QPY), str(HERE.parent / "build_scalar_algebra_canonical.py"),
            "--baseline", str(baseline),
            "--operand-outputs", str(operands / "parallel/outputs.jsonl"),
            "--out", str(scalar), "--audit", str(root / "SCALAR_ALGEBRA_CANONICAL_AUDIT.json"),
        ])
        full = core.RUN / "variants/map_full/canonical_predictions.jsonl"
        geometry_audit = root / "GEOMETRY_ALGEBRA_CANONICAL_AUDIT.json"
        self.execute("materialize_map_geometry_algebra", [
            str(core.QPY), str(HERE.parent / "build_geometry_algebra.py"),
            "--carrier-baseline", str(scalar),
            "--terminal-base", str(support / "terminal_base.jsonl"),
            "--terminal-tasks", str(support / "terminal_tasks.jsonl"),
            "--mapped-observations", str(support / "mapped.jsonl"),
            "--out", str(full), "--audit", str(geometry_audit),
            "--iou-threshold", "0.5",
        ])
        rows = read_jsonl(full)
        for row in rows:
            if row.get("prompt_id") == "nonlocal-support-closure-v1:source-registered-geometry-algebra":
                row["prompt_id"] = "map-v1:geometry-set-union-nms"
        write_jsonl(full, rows)

    def compose_test(self) -> None:
        _, closure, scale_closure = self.build_shared_state()
        scale_results, scale_tasks = self.run_shared_composer(closure, scale_closure)
        controller_tasks, controller_results, support_states, terminal, mapped = (
            self.run_support_recurrence(closure, scale_tasks)
        )
        support = core.WORK / "support_recurrence/support_complete"
        self.materialize_base_variants(
            closure, scale_closure, scale_results, controller_tasks,
            controller_results, support_states, terminal, mapped,
        )
        self.compile_and_run_algebra(support)
        self.execute("formal_test_evaluation", [
            str(core.QPY), str(core.EVALUATE), "--split", "Test",
            "--run-dir", str(core.RUN), "--variants", FINAL_VARIANT,
            "--gpus", core.GPU_STRING,
        ])

    def run_test(self, *, preflight_only: bool = False) -> None:
        if not core.GPU_IDS or len(core.GPU_IDS) != len(set(core.GPU_IDS)):
            raise ValueError(f"GPU IDs must be unique integers: {core.GPU_IDS}")
        self.compile_public_graph()
        self.prepare_main_observations()
        self.observe_native_views()
        self.compose_test()
        self.status("test", "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    TestSupervisor().run_test(preflight_only=args.preflight_only)
