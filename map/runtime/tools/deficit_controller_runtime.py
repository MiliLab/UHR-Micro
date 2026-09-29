#!/usr/bin/env python3
"""Standalone complete-Validation runtime for executable evidence deficits."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def keyed(rows: Iterable[dict[str, Any]], key: str = "sample_id") -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        ident = str(row.get(key) or "")
        if not ident or ident in output:
            raise ValueError(f"missing or duplicate {key}: {ident}")
        output[ident] = row
    return output


def grouped(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["sample_id"])].append(row)
    return output


def receipt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "region_id": f"R{int(row.get('roi_index') or 0)}",
            "source_box_1000": row.get("roi_full1000_xyxy"),
            "visible_observation": str(row.get("stage3_mapped_response_full1000") or row.get("model_response") or "").strip()[:500],
        }
        for row in sorted(rows, key=lambda item: int(item.get("roi_index") or 0))
    ]


def eligible(source: dict[str, Any]) -> tuple[bool, str]:
    state = source["evidence_state"]
    if state.get("answer_schema") == "SCALAR" and state.get("support_extent") == "FULL_SCENE":
        return False, "bounded_observation_cannot_close_full_scene_scalar"
    if source.get("special_operator") != "NONE" or bool(source.get("execution_plan", {}).get("grounding_candidate_fusion")):
        return False, "public_terminal_operator_owns_closure"
    if state.get("evidence_composition") == "SPATIAL_CONFIGURATION" and state.get("support_extent") == "MULTIPLE_SUPPORTS":
        return False, "single_view_cannot_close_multi_support_configuration"
    return True, "controller_eligible"


def controller_prompt(source: dict[str, Any], receipts: list[dict[str, Any]]) -> str:
    return f"""Inspect whether the current source-registered evidence is physically sufficient before final synthesis of one ultra-high-resolution remote-sensing question.

Public question:
{source['question_original']}

Public evidence requirement:
{json.dumps(source['evidence_state'], ensure_ascii=False, separators=(',', ':'))}

Already acquired native observation receipts:
{json.dumps(receipts, ensure_ascii=False, separators=(',', ':'))}

The displayed complete scene contains only method-generated white observation footprints. Do not answer the question, assess a candidate answer, or output confidence. Identify at most one answer-critical visible requirement that remains physically unsupported.

Return exactly one JSON object with:
- decision: STOP or OBSERVE;
- deficit_kind: none, RESOLUTION_DEFICIT, or COVERAGE_DEFICIT;
- visible_requirement: one short answer-free visible fact;
- cited_region_ids: receipt ids relevant to the requirement;
- support_box_1000: one [x1,y1,x2,y2] source-image box enclosing the visible support, or null;
- fixation_1000: one [x,y] source-image point, or null.

RESOLUTION_DEFICIT means the support is inside a cited receipt but too small, dense, or fine to resolve and needs one nested native view. COVERAGE_DEFICIT means a specific visible part of the required support lies outside the acquired receipts and needs one context view. Use STOP if the receipts are sufficient, if no specific support is defensible, or if the requirement is only uncertainty about an answer. Coordinates use the complete image's 0-1000 grid. Do not name a benchmark task, family, split, file, answer, annotation, metric, or worked example. JSON only."""


def prepare(args: argparse.Namespace) -> int:
    manifest_rows = read_jsonl(args.manifest)
    mapped = grouped(read_jsonl(args.mapped))
    overlays = keyed(read_jsonl(args.overlays))
    eligible_manifest: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for source in manifest_rows:
        allowed, reason = eligible(source)
        reasons[reason] += 1
        if not allowed:
            continue
        sid = str(source["sample_id"])
        receipts = receipt_rows(mapped.get(sid, []))
        if not receipts:
            raise ValueError(f"eligible sample has no registered receipt: {sid}")
        public = copy.deepcopy(source)
        public["source_registered_receipts"] = receipts
        eligible_manifest.append(public)
        task = copy.deepcopy(public)
        task.update({
            "task_id": f"{source['qa_slug']}_evidence_deficit",
            "stage_name": "preterminal_source_registered_deficit_check",
            "model_image_path": overlays[sid]["overlay_image_path"],
            "model_input_prompt": controller_prompt(public, receipts),
            "model_response": None,
        })
        tasks.append(task)
    forbidden = [row["sample_id"] for row in tasks if "task_key" in row or "split" in row]
    audit = {
        "input_rows": len(manifest_rows),
        "controller_rows": len(tasks),
        "precheck_reason_counts": dict(sorted(reasons.items())),
        "private_fields_in_runtime_rows": bool(forbidden),
        "candidate_answer_in_prompt": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
        "passed": bool(manifest_rows) and bool(tasks) and not forbidden,
    }
    write_jsonl(args.out_dir / "eligible_manifest.jsonl", eligible_manifest)
    write_jsonl(args.out_dir / "deficit_tasks.jsonl", tasks)
    write_json(args.out_dir / "deficit_prepare_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"complete-Validation preparation gate failed: {audit}")
    return 0


def parse_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    for candidate in (raw, raw[raw.find("{"):raw.rfind("}") + 1] if "{" in raw and "}" in raw else ""):
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def finite_list(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    try:
        output = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return output if all(math.isfinite(item) and 0 <= item <= 1000 for item in output) else None


def inside(point: list[float], box: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def intersection_area(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: list[float], b: list[float]) -> float:
    inter = intersection_area(a, b)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, aa + ab - inter)


def point_to_roi(point: list[float], width: int, height: int, side: int) -> list[int]:
    side = max(1, min(side, width, height))
    cx, cy = point[0] * width / 1000.0, point[1] * height / 1000.0
    x1 = min(max(int(round(cx - side / 2)), 0), width - side)
    y1 = min(max(int(round(cy - side / 2)), 0), height - side)
    return [x1, y1, x1 + side, y1 + side]


def to_1000(box: list[int], width: int, height: int) -> list[float]:
    return [round(box[0] * 1000 / width, 1), round(box[1] * 1000 / height, 1), round(box[2] * 1000 / width, 1), round(box[3] * 1000 / height, 1)]


def materialize(args: argparse.Namespace) -> int:
    manifest = keyed(read_jsonl(args.manifest))
    base_rois = grouped(read_jsonl(args.base_rois))
    results = keyed(read_jsonl(args.results), "task_id")
    outputs: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    for sid, source in manifest.items():
        result = results.get(f"{source['qa_slug']}_evidence_deficit")
        payload = parse_object(None if result is None else result.get("model_response"))
        decision = str(payload.get("decision") or "STOP").upper()
        kind = str(payload.get("deficit_kind") or "none").upper()
        point = finite_list(payload.get("fixation_1000"), 2)
        support = finite_list(payload.get("support_box_1000"), 4)
        if support is not None:
            support = [min(support[0], support[2]), min(support[1], support[3]), max(support[0], support[2]), max(support[1], support[3])]
            if support[2] - support[0] < 1 or support[3] - support[1] < 1:
                support = None
        receipt_by_id = {str(item["region_id"]): item for item in source.get("source_registered_receipts") or []}
        citations: list[str] = []
        for value in payload.get("cited_region_ids") or []:
            value = str(value)
            if value in receipt_by_id and value not in citations:
                citations.append(value)
        old_boxes = [[float(v) for v in row["roi_full1000_xyxy"]] for row in base_rois.get(sid, [])]
        reason = "admitted"
        side = 0
        if result is None or result.get("error"):
            reason = "missing_or_failed_controller"
        elif decision != "OBSERVE":
            reason = "controller_stop"
        elif kind not in {"RESOLUTION_DEFICIT", "COVERAGE_DEFICIT"}:
            reason = "invalid_deficit_kind"
        elif not citations:
            reason = "missing_valid_citations"
        elif point is None or support is None or not inside(point, support):
            reason = "invalid_support_geometry"
        elif kind == "RESOLUTION_DEFICIT":
            cited_boxes = [[float(v) for v in receipt_by_id[c]["source_box_1000"]] for c in citations]
            if not any(inside(point, box) for box in cited_boxes):
                reason = "resolution_not_inside_cited_support"
            else:
                side = 512
        else:
            support_area = max(1e-9, (support[2] - support[0]) * (support[3] - support[1]))
            covered = min(1.0, sum(intersection_area(support, box) for box in old_boxes) / support_area)
            if any(inside(point, box) for box in old_boxes) or covered >= 0.5:
                reason = "coverage_not_geometrically_missing"
            else:
                side = 1024
        if reason == "admitted":
            width, height = int(source["original_width_px"]), int(source["original_height_px"])
            roi_abs = point_to_roi(point, width, height, side)
            roi_1000 = to_1000(roi_abs, width, height)
            if any(iou(roi_1000, box) >= 0.8 for box in old_boxes):
                reason = "duplicate_source_view"
            else:
                max_index = max((int(row.get("roi_index") or 0) for row in base_rois.get(sid, [])), default=0)
                row = copy.deepcopy(source)
                row.update({
                    "roi_id": f"{source['qa_slug']}_deficit_roi01_{side}",
                    "roi_index": max_index + 1,
                    "roi_abs_xyxy": roi_abs,
                    "roi_full1000_xyxy": roi_1000,
                    "roi_width_px": roi_abs[2] - roi_abs[0],
                    "roi_height_px": roi_abs[3] - roi_abs[1],
                    "roi_area_fraction": round((roi_abs[2] - roi_abs[0]) * (roi_abs[3] - roi_abs[1]) / max(1, width * height), 8),
                    "roi_source": "EXECUTABLE_SOURCE_REGISTERED_DEFICIT",
                    "feedback_kind": "EVIDENCE_DEFICIT_CLOSURE",
                    "deficit_kind": kind,
                    "deficit_visible_requirement": str(payload.get("visible_requirement") or "").strip()[:300],
                    "deficit_cited_region_ids": citations,
                    "deficit_support_box_1000": support,
                    "deficit_fixation_1000": point,
                    "model_response": None,
                })
                outputs.append(row)
                actions[kind] += 1
        reasons[reason] += 1
        decisions.append({
            "sample_id": sid,
            "parse_ok": bool(payload),
            "proposed_decision": decision,
            "proposed_kind": kind,
            "admitted": reason == "admitted",
            "reason": reason,
            "cited_region_ids": citations,
            "support_box_1000": support,
            "fixation_1000": point,
            "visible_requirement": str(payload.get("visible_requirement") or "").strip()[:300],
        })
    outputs.sort(key=lambda row: -int(row["roi_width_px"]) * int(row["roi_height_px"]))
    write_jsonl(args.out, outputs)
    write_jsonl(args.out.with_name("deficit_decisions.jsonl"), decisions)
    audit = {
        "rows": len(decisions),
        "parse_ok": sum(bool(row["parse_ok"]) for row in decisions),
        "admitted_qas": len(outputs),
        "new_views": len(outputs),
        "action_counts": dict(sorted(actions.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "max_views_per_sample": 1,
        "task_key_used": False,
        "candidate_answer_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    write_json(args.out.with_name("deficit_materialize_audit.json"), audit)
    return 0


def annotate(args: argparse.Namespace) -> int:
    decisions = keyed(read_jsonl(args.decisions))
    rows = read_jsonl(args.mapped)
    for row in rows:
        state = decisions[str(row["sample_id"])]
        row["feedback_kind"] = "EVIDENCE_DEFICIT_CLOSURE"
        row["feedback_mapped_response"] = row.get("stage3_mapped_response_full1000") or row.get("model_response")
        row["deficit_kind"] = state["proposed_kind"]
        row["deficit_visible_requirement"] = state["visible_requirement"]
        row["deficit_support_box_1000"] = state["support_box_1000"]
    write_jsonl(args.out, rows)
    return 0


def combine_results(args: argparse.Namespace) -> int:
    base_rows = read_jsonl(args.base)
    candidate = keyed(read_jsonl(args.candidate))
    decisions = keyed(read_jsonl(args.decisions))
    active = {sid for sid, row in decisions.items() if row["admitted"]}
    if set(candidate) != active:
        raise ValueError(f"candidate terminal results do not equal admitted set: {len(candidate)} vs {len(active)}")
    output = [copy.deepcopy(candidate[row["sample_id"]] if row["sample_id"] in active else row) for row in base_rows]
    write_jsonl(args.out, output)
    write_json(args.out.with_name("combined_stage4_audit.json"), {
        "rows": len(output),
        "unique_samples": len({row["sample_id"] for row in output}),
        "replaced_active_rows": len(active),
        "inactive_exact_base_rows": len(output) - len(active),
        "test_used": False,
        "passed": len(output) == len({row["sample_id"] for row in output}),
    })
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--mapped", type=Path, required=True)
    item.add_argument("--overlays", type=Path, required=True)
    item.add_argument("--out-dir", type=Path, required=True)
    item.set_defaults(func=prepare)
    item = sub.add_parser("materialize")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--base-rois", type=Path, required=True)
    item.add_argument("--results", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.set_defaults(func=materialize)
    item = sub.add_parser("annotate")
    item.add_argument("--mapped", type=Path, required=True)
    item.add_argument("--decisions", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.set_defaults(func=annotate)
    item = sub.add_parser("combine-results")
    item.add_argument("--base", type=Path, required=True)
    item.add_argument("--candidate", type=Path, required=True)
    item.add_argument("--decisions", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.set_defaults(func=combine_results)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
