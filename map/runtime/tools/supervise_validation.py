#!/usr/bin/env python3
"""Run standalone SSRR v2.1 and same-implementation Validation ablations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import map_core_supervisor as core
from io_utils import read_jsonl, sha256_file, write_json, write_jsonl


SCALE = core.TOOLS / "scale_residual_runtime.py"
SUPPORT = core.TOOLS / "support_residual_runtime.py"
DEFICIT = core.TOOLS / "deficit_controller_runtime.py"
RECURRENCE = core.TOOLS / "support_recurrence_runtime.py"
STRONG = core.TOOLS / "strong_path_runtime.py"
EXPORT = core.TOOLS / "export_roi_crops.py"
MAP_OLD = core.TOOLS / "register_outputs.py"  # overridden by qwen3vl/internvl entries
MATERIALIZER = core.TOOLS / "residual_materializer.py"
VARIANTS = (
    "recurrence_disabled",
    "scale_residual_only",
    "support_residual_only",
    "ssrr_full",
)


def keyed(path: Path, key: str = "sample_id") -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        ident = str(row.get(key) or "")
        if not ident or ident in output:
            raise ValueError(f"missing or duplicate {key}: {ident!r}")
        output[ident] = row
    return output


class SSRRSupervisor(core.MapSupervisor):
    @staticmethod
    def filter_composer_for_closure(
        source_results: Path, closure: Path, output: Path
    ) -> None:
        decisions = keyed(closure)
        expected = {
            sid
            for sid, row in decisions.items()
            if row.get("closure_status") == "OPEN"
        }
        results = keyed(source_results)
        if not expected.issubset(results):
            raise ValueError("composer results do not cover closure OPEN states")
        write_jsonl(output, [results[sid] for sid in sorted(expected)])

    def prepare_composer(self, name: str, closure: Path) -> tuple[Path, Path]:
        raw = core.RAW / f"composer_{name}.jsonl"
        tasks = core.WORK / f"composer_{name}.jsonl"
        self.execute(
            f"prepare_composer_{name}",
            [
                str(core.QPY), str(core.TOOLS / "composer_runtime.py"), "prepare",
                "--manifest", str(core.WORK / "runtime_manifest.jsonl"),
                "--states", str(core.WORK / "typed_evidence_states.jsonl"),
                "--closure", str(closure),
                "--source-canvases", str(core.WORK / "source_canvases/source_canvas_manifest.jsonl"),
                "--out", str(raw),
                "--audit", str(core.AUDITS / f"composer_{name}_prepare.json"),
            ],
        )
        self.sanitize(f"composer_{name}", raw, tasks)
        return raw, tasks

    def build_shared_state(self) -> tuple[Path, Path, Path]:
        manifest = core.WORK / "runtime_manifest.jsonl"
        states = core.WORK / "typed_evidence_states.jsonl"
        closure = core.WORK / "closure_recurrence_disabled.jsonl"
        scale_closure = core.WORK / "closure_scale_residual.jsonl"
        self.execute(
            "aggregate_typed_source_evidence",
            [
                str(core.QPY), str(core.TOOLS / "composer_runtime.py"), "aggregate",
                "--manifest", str(manifest),
                "--native", str(core.WORK / "native_results.jsonl"),
                "--mapped", str(core.WORK / "stage3_mapped.jsonl"),
                "--out", str(states),
                "--audit", str(core.AUDITS / "typed_aggregation.json"),
            ],
        )
        self.execute(
            "decide_recurrence_disabled_closure",
            [
                str(core.QPY), str(core.TOOLS / "composer_runtime.py"), "closure",
                "--manifest", str(manifest), "--states", str(states),
                "--mapped", str(core.WORK / "stage3_mapped.jsonl"),
                "--out", str(closure),
                "--audit", str(core.AUDITS / "closure_recurrence_disabled.json"),
            ],
        )
        self.execute(
            "certify_scale_residual",
            [
                str(core.QPY), str(SCALE), "reopen",
                "--states", str(states), "--closure", str(closure),
                "--out", str(scale_closure),
                "--activated", str(core.WORK / "scale_residual_activated.jsonl"),
                "--audit", str(core.AUDITS / "scale_residual.json"),
            ],
        )
        return states, closure, scale_closure

    def run_shared_composer(
        self, closure: Path, scale_closure: Path
    ) -> tuple[Path, Path]:
        _, all_tasks = self.prepare_composer("scale_union", scale_closure)
        all_results = core.WORK / "composer_scale_union_results.jsonl"
        self.gpu("composer_scale_union", all_tasks, all_results, 1024)
        base_results = core.WORK / "composer_recurrence_disabled_results.jsonl"
        self.filter_composer_for_closure(all_results, closure, base_results)
        write_json(
            core.AUDITS / "composer_same_run_partition.json",
            {
                "schema_version": "ssrr-v2.1-composer-same-run-partition-1",
                "union_results": len(read_jsonl(all_results)),
                "recurrence_disabled_results": len(read_jsonl(base_results)),
                "scale_residual_results": len(read_jsonl(all_results)) - len(read_jsonl(base_results)),
                "response_or_score_selector_used": False,
                "test_used": False,
                "passed": True,
            },
        )
        return all_results, all_tasks

    def run_support_recurrence(
        self, closure: Path, composer_tasks: Path
    ) -> tuple[Path, Path, Path, Path, Path]:
        root = core.WORK / "support_recurrence"
        controller = root / "controller"
        support = root / "support_complete"
        self.execute(
            "adapt_map_open_states_for_support_residual",
            [
                str(core.QPY), str(SUPPORT), "adapt",
                "--manifest", str(core.WORK / "runtime_manifest.jsonl"),
                "--closure", str(closure),
                "--source-canvases", str(core.WORK / "source_canvases/source_canvas_manifest.jsonl"),
                "--out-manifest", str(root / "open_manifest.jsonl"),
                "--out-overlays", str(root / "overlays.jsonl"),
                "--audit", str(core.AUDITS / "support_adapter.json"),
            ],
        )
        self.execute(
            "prepare_support_deficit_controller",
            [
                str(core.QPY), str(DEFICIT), "prepare",
                "--manifest", str(root / "open_manifest.jsonl"),
                "--mapped", str(core.WORK / "stage3_mapped.jsonl"),
                "--overlays", str(root / "overlays.jsonl"),
                "--out-dir", str(controller),
            ],
        )
        self.execute(
            "filter_physically_admissible_support_updates",
            [
                str(core.QPY), str(SUPPORT), "filter-controller-admissible",
                "--manifest", str(controller / "eligible_manifest.jsonl"),
                "--tasks", str(controller / "deficit_tasks.jsonl"),
                "--out-manifest", str(controller / "eligible_admissible.jsonl"),
                "--out-tasks", str(controller / "deficit_tasks_admissible.raw.jsonl"),
                "--audit", str(core.AUDITS / "support_admissibility.json"),
            ],
        )
        self.execute(
            "sanitize_support_controller",
            [
                str(core.QPY), str(SUPPORT), "sanitize-controller",
                "--input", str(controller / "deficit_tasks_admissible.raw.jsonl"),
                "--out", str(controller / "deficit_tasks_admissible.jsonl"),
                "--audit", str(core.AUDITS / "support_controller_boundary.json"),
            ],
        )
        controller_tasks = controller / "deficit_tasks_admissible.jsonl"
        controller_results = controller / "deficit_results.jsonl"
        self.gpu("support_deficit_controller", controller_tasks, controller_results, 512)
        self.execute(
            "materialize_support_deficit",
            [
                str(core.QPY), str(DEFICIT), "materialize",
                "--manifest", str(controller / "eligible_admissible.jsonl"),
                "--base-rois", str(core.WORK / "rois/roi_manifest.jsonl"),
                "--results", str(controller_results),
                "--out", str(controller / "proposed_rois.jsonl"),
            ],
        )
        decisions_raw = controller / "deficit_decisions.jsonl"
        self.execute(
            "build_complete_support_views",
            [
                str(core.QPY), str(RECURRENCE), "build-rois",
                "--manifest", str(controller / "eligible_admissible.jsonl"),
                "--controller-decisions", str(decisions_raw),
                "--tile-side", "640", "--coverage", "complete",
                "--out", str(support / "rois.raw.jsonl"),
                "--audit-out", str(core.AUDITS / "support_rois_raw.json"),
            ],
        )
        self.execute(
            "enforce_four_view_support_budget",
            [
                str(core.QPY), str(SUPPORT), "enforce-budget",
                "--rois", str(support / "rois.raw.jsonl"),
                "--decisions", str(decisions_raw), "--max-views", "4",
                "--out-rois", str(support / "rois.jsonl"),
                "--out-decisions", str(support / "decisions.jsonl"),
                "--audit", str(core.AUDITS / "support_view_budget.json"),
            ],
        )
        self.execute(
            "compile_public_observer_profiles",
            [
                str(core.QPY), str(SUPPORT), "compile-observer-profiles",
                "--input", str(support / "rois.jsonl"),
                "--out", str(support / "rois.profiled.jsonl"),
                "--audit", str(core.AUDITS / "support_observer_profiles.json"),
            ],
        )
        self.execute(
            "export_support_views",
            [
                str(core.QPY), str(EXPORT), "--roi-manifest", str(support / "rois.profiled.jsonl"),
                "--out-dir", str(support / "crops"), "--workers", "24",
                "--image-format", "png", "--resume",
            ],
        )
        self.execute(
            "build_support_observers",
            [
                str(core.QPY), str(STRONG), "build-stage3",
                "--rois", str(support / "rois.profiled.jsonl"),
                "--crop-dir", str(support / "crops"),
                "--out", str(support / "observer_tasks.raw.jsonl"),
            ],
        )
        self.execute(
            "sanitize_support_observers",
            [
                str(core.QPY), str(SUPPORT), "sanitize-generic",
                "--input", str(support / "observer_tasks.raw.jsonl"),
                "--out", str(support / "observer_tasks.jsonl"),
                "--audit", str(core.AUDITS / "support_observer_boundary.json"),
            ],
        )
        observer_tasks = support / "observer_tasks.jsonl"
        observer_results = support / "observer_results.jsonl"
        self.gpu("support_observers", observer_tasks, observer_results, 512)
        self.execute(
            "rehydrate_support_observers",
            [
                str(core.QPY), str(SUPPORT), "rehydrate-observer-results",
                "--tasks", str(support / "observer_tasks.raw.jsonl"),
                "--results", str(observer_results),
                "--out", str(support / "observer_results.rehydrated.jsonl"),
                "--audit", str(core.AUDITS / "support_observer_rehydration.json"),
            ],
        )
        self.execute(
            "register_support_observations",
            [
                str(core.QPY), str(MAP_OLD),
                "--stage3-run", str(support / "observer_results.rehydrated.jsonl"),
                "--out", str(support / "mapped.raw.jsonl"),
            ],
        )
        self.execute(
            "annotate_support_observations",
            [
                str(core.QPY), str(RECURRENCE), "annotate",
                "--mapped", str(support / "mapped.raw.jsonl"),
                "--rois", str(support / "rois.profiled.jsonl"),
                "--out", str(support / "mapped.annotated.jsonl"),
            ],
        )
        self.execute(
            "enforce_support_observation_validity",
            [
                str(core.QPY), str(SUPPORT), "enforce-observation-validity",
                "--input", str(support / "mapped.annotated.jsonl"),
                "--out", str(support / "mapped.jsonl"),
                "--audit", str(core.AUDITS / "support_observation_validity.json"),
            ],
        )
        self.execute(
            "prepare_support_terminal_base",
            [
                str(core.QPY), str(SUPPORT), "prepare-terminal-base",
                "--base", str(composer_tasks),
                "--manifest", str(controller / "eligible_admissible.jsonl"),
                "--out", str(support / "terminal_base.jsonl"),
                "--audit", str(core.AUDITS / "support_terminal_base.json"),
            ],
        )
        self.execute(
            "build_support_terminal",
            [
                str(core.QPY), str(RECURRENCE), "build-terminal",
                "--base-tasks", str(support / "terminal_base.jsonl"),
                "--manifest", str(controller / "eligible_admissible.jsonl"),
                "--controller-decisions", str(support / "decisions.jsonl"),
                "--mapped", str(support / "mapped.jsonl"),
                "--coverage", "complete", "--assimilation", "typed",
                "--out", str(support / "terminal_all.jsonl"),
                "--active-out", str(support / "terminal_tasks.raw.jsonl"),
                "--state-out", str(support / "states.jsonl"),
                "--audit-out", str(core.AUDITS / "support_terminal_prepare.json"),
            ],
        )
        self.execute(
            "sanitize_support_terminal",
            [
                str(core.QPY), str(SUPPORT), "sanitize-generic",
                "--input", str(support / "terminal_tasks.raw.jsonl"),
                "--out", str(support / "terminal_tasks.jsonl"),
                "--audit", str(core.AUDITS / "support_terminal_boundary.json"),
            ],
        )
        terminal_results = support / "terminal_results.jsonl"
        self.gpu("support_terminal", support / "terminal_tasks.jsonl", terminal_results, 1024)
        return (
            controller_tasks,
            controller_results,
            support / "states.jsonl",
            terminal_results,
            support / "mapped.jsonl",
        )

    def compose_ssrr_and_evaluate(self) -> None:
        _, closure, scale_closure = self.build_shared_state()
        scale_results, scale_tasks = self.run_shared_composer(closure, scale_closure)
        recurrence = self.run_support_recurrence(closure, scale_tasks)
        controller_tasks, controller_results, support_states, terminal, mapped = recurrence
        self.execute(
            "materialize_all_variants_from_residual_state",
            [
                str(core.QPY), str(MATERIALIZER),
                "--manifest", str(core.WORK / "runtime_manifest.jsonl"),
                "--binding", str(core.RUN / "private_reporting_binding.jsonl"),
                "--native", str(core.WORK / "native_results.jsonl"),
                "--planner", str(core.WORK / "planner_complete.jsonl"),
                "--mapped", str(core.WORK / "stage3_mapped.jsonl"),
                "--states", str(core.WORK / "typed_evidence_states.jsonl"),
                "--base-closure", str(closure),
                "--scale-closure", str(scale_closure),
                "--composer-union", str(scale_results),
                "--support-controller-tasks", str(controller_tasks),
                "--support-controller-results", str(controller_results),
                "--support-states", str(support_states),
                "--support-terminal-results", str(terminal),
                "--support-mapped", str(mapped),
                "--phase0", str(core.PHASE0),
                "--run-id", core.RUN_ID,
                "--output-root", str(core.RUN / "variants"),
                "--audit-root", str(core.RUN),
            ],
        )
        self.execute(
            "formal_validation_evaluation",
            [
                str(core.QPY), str(core.EVALUATE), "--split", "Validation",
                "--run-dir", str(core.RUN), "--variants", ",".join(VARIANTS),
                "--gpus", core.GPU_STRING,
            ],
        )
        formal = json.loads((core.RUN / "FORMAL_EVALUATION.json").read_text(encoding="utf-8"))
        if not formal.get("all_repeat_checks_passed"):
            raise RuntimeError("formal repeat evaluation failed")

    def run_ssrr(self, *, preflight_only: bool = False) -> None:
        if not core.GPU_IDS or len(core.GPU_IDS) != len(set(core.GPU_IDS)):
            raise ValueError(f"GPU IDs must be unique integers: {core.GPU_IDS}")
        self.compile_public_graph()
        self.prepare_main_observations()
        self.observe_native_views()
        self.compose_ssrr_and_evaluate()
        self.status("validation", "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    SSRRSupervisor().run_ssrr(preflight_only=args.preflight_only)
