#!/usr/bin/env python3
"""Materialize and assimilate support-complete source-coordinate observations."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
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


def axis_windows(lo: int, hi: int, limit: int, side: int) -> list[int]:
    length = max(1, hi - lo)
    if length <= side:
        return [min(max(int(round((lo + hi - side) / 2)), 0), max(0, limit - side))]
    count = int(math.ceil(length / side))
    first = max(0, min(lo, limit - side))
    last = max(0, min(hi - side, limit - side))
    if count == 1:
        return [first]
    return sorted({int(round(first + (last - first) * index / (count - 1))) for index in range(count)})


def to_abs(box: list[float], width: int, height: int) -> list[int]:
    return [
        max(0, min(width, int(math.floor(box[0] * width / 1000)))),
        max(0, min(height, int(math.floor(box[1] * height / 1000)))),
        max(0, min(width, int(math.ceil(box[2] * width / 1000)))),
        max(0, min(height, int(math.ceil(box[3] * height / 1000)))),
    ]


def to_1000(box: list[int], width: int, height: int) -> list[float]:
    return [round(box[0] * 1000 / width, 1), round(box[1] * 1000 / height, 1), round(box[2] * 1000 / width, 1), round(box[3] * 1000 / height, 1)]


def centered_window(support: list[int], width: int, height: int, side: int) -> list[int]:
    cx = (support[0] + support[2]) / 2
    cy = (support[1] + support[3]) / 2
    x1 = min(max(int(round(cx - side / 2)), 0), max(0, width - side))
    y1 = min(max(int(round(cy - side / 2)), 0), max(0, height - side))
    return [x1, y1, min(width, x1 + side), min(height, y1 + side)]


def build_rois(args: argparse.Namespace) -> int:
    manifest = keyed(read_jsonl(args.manifest))
    controller = keyed(read_jsonl(args.controller_decisions))
    rows: list[dict[str, Any]] = []
    per_sample: Counter[str] = Counter()
    for sid, state in controller.items():
        if not state.get("admitted"):
            continue
        source = manifest[sid]
        support = controller[sid].get("support_box_1000")
        if not isinstance(support, list) or len(support) != 4:
            raise ValueError(f"missing declared support for admitted sample: {sid}")
        width, height = int(source["original_width_px"]), int(source["original_height_px"])
        absolute = to_abs([float(v) for v in support], width, height)
        if args.coverage == "complete":
            xs = axis_windows(absolute[0], absolute[2], width, args.tile_side)
            ys = axis_windows(absolute[1], absolute[3], height, args.tile_side)
            boxes = [[x, y, min(width, x + args.tile_side), min(height, y + args.tile_side)] for y in ys for x in xs]
        else:
            boxes = [centered_window(absolute, width, height, args.tile_side)]
        index = 0
        for box in boxes:
            index += 1
            row = copy.deepcopy(source)
            row.update({
                    "roi_id": f"{source['qa_slug']}_{args.coverage}_support_tile{index:02d}",
                    "roi_index": 100 + index,
                    "roi_abs_xyxy": box,
                    "roi_full1000_xyxy": to_1000(box, width, height),
                    "roi_width_px": box[2] - box[0],
                    "roi_height_px": box[3] - box[1],
                    "roi_area_fraction": round((box[2] - box[0]) * (box[3] - box[1]) / max(1, width * height), 8),
                    "roi_source": "SUPPORT_COMPLETE_DEFICIT_TILING" if args.coverage == "complete" else "SINGLE_CENTER_DEFICIT_FIXATION",
                    "feedback_kind": "SUPPORT_COMPLETE_EVIDENCE_CLOSURE" if args.coverage == "complete" else "SINGLE_FIXATION_EVIDENCE",
                    "declared_support_box_1000": support,
                    "support_tile_index": index,
                    "support_tile_count": len(boxes),
                    "model_response": None,
            })
            rows.append(row)
        per_sample[sid] = index
    rows.sort(key=lambda row: (-int(row["roi_width_px"]) * int(row["roi_height_px"]), row["sample_id"], int(row["roi_index"])))
    write_jsonl(args.out, rows)
    audit = {
        "residual_rows": len(manifest),
        "active_supports": len(per_sample),
        "tile_rows": len(rows),
        "tile_side_px": args.tile_side,
        "coverage_mode": args.coverage,
        "tile_count_distribution": dict(sorted(Counter(per_sample.values()).items())),
        "gap_free_by_construction": args.coverage == "complete",
        "task_key_used": False,
        "answer_or_metric_used": False,
        "test_used": False,
        "passed": bool(manifest) and bool(per_sample) and bool(rows),
    }
    write_json(args.audit_out, audit)
    if not audit["passed"]:
        raise RuntimeError(f"support closure ROI gate failed: {audit}")
    return 0


def annotate(args: argparse.Namespace) -> int:
    rois = {str(row["roi_id"]): row for row in read_jsonl(args.rois)}
    output = []
    for row in read_jsonl(args.mapped):
        source = rois[str(row["roi_id"])]
        row["feedback_kind"] = source["feedback_kind"]
        row["declared_support_box_1000"] = source["declared_support_box_1000"]
        row["support_tile_index"] = source["support_tile_index"]
        row["support_tile_count"] = source["support_tile_count"]
        row["feedback_mapped_response"] = row.get("stage3_mapped_response_full1000") or row.get("model_response")
        output.append(row)
    write_jsonl(args.out, output)
    return 0


def empty(value: Any) -> bool:
    return str(value or "").strip().lower().replace("`", "") in {"", "null", "none", "final answer: null", "final answer:null"}


TYPE_RULES = {
    "SCALAR": "Aggregate only the evidence actually covered by this receipt without double-counting overlap; do not convert a bounded receipt into a complete-scene total.",
    "OPTION": "Reconcile appearance cues across all support tiles and the complete-scene context. Do not prefer a cue merely because it is newer or higher resolution.",
    "GEOMETRY_SET": "Treat non-null tile geometries as additive candidates, deduplicate overlap in source coordinates, and preserve prior positive candidates outside the tiled support.",
    "SINGLE_GEOMETRY": "Select the source-localized candidate that best satisfies the question using both tiled detail and complete-scene configuration.",
    "MASK_SUPPORT": "Select one source-registered mask-prompt candidate using tiled detail and complete-scene configuration; empty tiles cannot erase positive support.",
}


def build_terminal(args: argparse.Namespace) -> int:
    base = keyed(read_jsonl(args.base_tasks))
    manifest = keyed(read_jsonl(args.manifest))
    decisions = keyed(read_jsonl(args.controller_decisions))
    mapped = grouped(read_jsonl(args.mapped))
    all_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for sid, source in manifest.items():
        row = copy.deepcopy(base[sid])
        reason = "inactive"
        receipt = None
        if decisions[sid].get("admitted"):
            observations = []
            for item in sorted(mapped.get(sid, []), key=lambda value: int(value["support_tile_index"])):
                content = item.get("feedback_mapped_response") or item.get("stage3_mapped_response_full1000") or item.get("model_response")
                observations.append({
                    "tile_index": item["support_tile_index"],
                    "source_box_1000": item["roi_full1000_xyxy"],
                    "observed_content": None if empty(content) else content,
                })
            nonempty = sum(item["observed_content"] is not None for item in observations)
            if not observations or nonempty == 0:
                reason = "all_support_tiles_empty_noop"
            else:
                schema = str(source["evidence_state"]["answer_schema"])
                receipt = {
                    "answer_schema": schema,
                    "evidence_composition": source["evidence_state"].get("evidence_composition"),
                    "declared_support_box_1000": observations[0] and mapped[sid][0]["declared_support_box_1000"],
                    "support_coverage": args.coverage.upper(),
                    "tile_count": len(observations),
                    "nonempty_tile_count": nonempty,
                    "tile_observations": observations,
                }
                typed_addition = f"""

Support-complete typed evidence receipt:
{json.dumps(receipt, ensure_ascii=False, separators=(',', ':'))}

Evidence-closure contract:
- Each observation is valid only in its own source box. The receipt coverage is `{args.coverage.upper()}`.
- The acquisition query is not evidence and is deliberately absent.
- Empty tiles express no new positive evidence and cannot delete prior evidence.
- Preserve all earlier source-registered evidence outside the declared support.
- {TYPE_RULES[schema]}
- Do not expose this internal receipt. Produce the answer once.

Final output contract:
{row['final_format_hint']}"""
                untyped_addition = f"""

Additional support-complete source observations:
{json.dumps(receipt, ensure_ascii=False, separators=(',', ':'))}

Use these observations together with the prior global and local evidence to produce the answer once. Do not expose the internal observation record.

Final output contract:
{row['final_format_hint']}"""
                addition = typed_addition if args.assimilation == "typed" else untyped_addition
                row["task_id"] = f"{source['qa_slug']}_{args.coverage}_{args.assimilation}_recurrence_synthesis_v2"
                row["stage_name"] = f"{args.coverage}_{args.assimilation}_evidence_recurrence_synthesis"
                row["model_input_prompt"] = str(row["model_input_prompt"]) + addition
                row["support_complete_receipt"] = receipt
                row["model_response"] = None
                active_rows.append(row)
                reason = "integrate_support_complete_receipt"
        all_rows.append(row)
        states.append({"sample_id": sid, "admitted": reason == "integrate_support_complete_receipt", "reason": reason})
        reasons[reason] += 1
    active_rows.sort(key=lambda row: -int(row.get("original_width_px") or 0) * int(row.get("original_height_px") or 0))
    write_jsonl(args.out, all_rows)
    write_jsonl(args.active_out, active_rows)
    write_jsonl(args.state_out, states)
    audit = {
        "rows": len(all_rows),
        "active_rows": len(active_rows),
        "reason_counts": dict(sorted(reasons.items())),
        "inactive_prompt_mismatches": sum(base[row["sample_id"]]["model_input_prompt"] != row["model_input_prompt"] for row in all_rows if row["sample_id"] not in {item["sample_id"] for item in active_rows}),
        "support_coverage": args.coverage.upper(),
        "assimilation": args.assimilation,
        "acquisition_query_serialized": False,
        "task_key_used": False,
        "answer_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = len(all_rows) == len(manifest) and len(active_rows) > 0 and audit["inactive_prompt_mismatches"] == 0
    write_json(args.audit_out, audit)
    if not audit["passed"]:
        raise RuntimeError(f"support terminal gate failed: {audit}")
    return 0


def combine(args: argparse.Namespace) -> int:
    base = read_jsonl(args.base)
    candidate = keyed(read_jsonl(args.candidate))
    states = keyed(read_jsonl(args.states))
    active = {sid for sid, state in states.items() if state["admitted"]}
    if set(candidate) != active:
        raise ValueError("candidate result set does not match active support closures")
    output = [copy.deepcopy(candidate[row["sample_id"]] if row["sample_id"] in active else row) for row in base]
    write_jsonl(args.out, output)
    write_json(args.audit_out, {"rows": len(output), "unique_samples": len({row["sample_id"] for row in output}), "active_replacements": len(active), "inactive_exact_base_rows": len(output) - len(active), "passed": len(output) == len(base)})
    return 0


def closure_guard(args: argparse.Namespace) -> int:
    base_rows = read_jsonl(args.base)
    full = keyed(read_jsonl(args.full))
    manifest = keyed(read_jsonl(args.manifest))
    output = []
    guarded = 0
    changed_guarded = 0
    for base in base_rows:
        sid = str(base["sample_id"])
        candidate = full[sid]
        source = manifest.get(sid)
        guard = bool(
            source
            and source["evidence_state"].get("answer_schema") == "SCALAR"
            and source["evidence_state"].get("support_extent") == "FULL_SCENE"
        )
        if guard:
            guarded += 1
            changed_guarded += str(candidate.get("model_response")) != str(base.get("model_response"))
            output.append(copy.deepcopy(base))
        else:
            output.append(copy.deepcopy(candidate))
    write_jsonl(args.out, output)
    audit = {
        "rows": len(output),
        "guarded_public_states": guarded,
        "changed_outputs_reverted": changed_guarded,
        "guard_basis": "answer_schema=SCALAR and support_extent=FULL_SCENE",
        "task_key_used": False,
        "answer_or_metric_used": False,
        "test_used": False,
        "passed": len(output) == len(base_rows) and guarded > 0,
    }
    write_json(args.audit_out, audit)
    if not audit["passed"]:
        raise RuntimeError(f"closure guard gate failed: {audit}")
    return 0


def closure_admission(args: argparse.Namespace) -> int:
    manifest_rows = read_jsonl(args.manifest)
    decisions = keyed(read_jsonl(args.decisions))
    mapped_rows = read_jsonl(args.mapped)
    states = keyed(read_jsonl(args.states))
    terminal = keyed(read_jsonl(args.terminal))
    excluded = {
        str(row["sample_id"])
        for row in manifest_rows
        if row["evidence_state"].get("answer_schema") == "SCALAR"
        and row["evidence_state"].get("support_extent") == "FULL_SCENE"
    }
    kept_manifest = [row for row in manifest_rows if str(row["sample_id"]) not in excluded]
    kept_decisions = [row for sid, row in decisions.items() if sid not in excluded]
    kept_mapped = [row for row in mapped_rows if str(row["sample_id"]) not in excluded]
    kept_states = [row for sid, row in states.items() if sid not in excluded]
    kept_terminal = [row for sid, row in terminal.items() if sid not in excluded]
    write_jsonl(args.out_dir / "eligible_manifest.jsonl", kept_manifest)
    write_jsonl(args.out_dir / "decisions.jsonl", kept_decisions)
    write_jsonl(args.out_dir / "mapped.jsonl", kept_mapped)
    write_jsonl(args.out_dir / "states.jsonl", kept_states)
    write_jsonl(args.out_dir / "terminal.jsonl", kept_terminal)
    removed_mapped = len(mapped_rows) - len(kept_mapped)
    removed_terminal = len(terminal) - len(kept_terminal)
    audit = {
        "excluded_public_states": len(excluded),
        "excluded_basis": "answer_schema=SCALAR and support_extent=FULL_SCENE",
        "controller_calls_avoided": len(excluded),
        "observation_calls_avoided": removed_mapped,
        "terminal_calls_avoided": removed_terminal,
        "total_logical_calls_avoided": len(excluded) + removed_mapped + removed_terminal,
        "task_key_used": False,
        "answer_or_metric_used": False,
        "test_used": False,
        "passed": len(excluded) > 0 and len(kept_decisions) + len(excluded) == len(decisions),
    }
    write_json(args.audit_out, audit)
    if not audit["passed"]:
        raise RuntimeError(f"closure admission gate failed: {audit}")
    return 0


def extract_final_boxes(text: Any) -> list[list[float]]:
    value = str(text or "")
    if "Final answer:" in value:
        value = value.rsplit("Final answer:", 1)[-1]
    boxes = []
    for group in re.findall(r"\[([^\[\]]+)\]", value):
        numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", group)
        if len(numbers) not in {4, 8}:
            continue
        boxes.append([float(number) for number in numbers])
    return boxes


def envelope(box: list[float]) -> list[float]:
    if len(box) == 4:
        return [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
    xs, ys = box[0::2], box[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def box_iou(a: list[float], b: list[float]) -> float:
    a, b = envelope(a), envelope(b)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


def executable_geometry_state(args: argparse.Namespace) -> int:
    base_rows = read_jsonl(args.base)
    full = keyed(read_jsonl(args.full))
    manifest = keyed(read_jsonl(args.manifest))
    states = keyed(read_jsonl(args.states))
    output = []
    eligible = 0
    merged = 0
    base_preserved_on_null = 0
    for base in base_rows:
        sid = str(base["sample_id"])
        candidate = copy.deepcopy(full[sid])
        source = manifest.get(sid)
        active = bool(states.get(sid, {}).get("admitted"))
        if source and active and source["evidence_state"].get("answer_schema") == "GEOMETRY_SET":
            eligible += 1
            old_boxes = extract_final_boxes(base.get("model_response"))
            new_boxes = extract_final_boxes(candidate.get("model_response"))
            if old_boxes:
                union = list(old_boxes)
                for box in new_boxes:
                    if not any(box_iou(box, prior) >= args.dedup_iou for prior in union):
                        union.append(box)
                if not new_boxes:
                    base_preserved_on_null += 1
                rendered = ", ".join("[" + ", ".join(format_number(value) for value in box) + "]" for box in union)
                candidate["model_response"] = "Final answer: " + rendered
                merged += str(candidate.get("model_response")) != str(full[sid].get("model_response"))
        output.append(candidate)
    write_jsonl(args.out, output)
    audit = {
        "rows": len(output),
        "eligible_active_geometry_sets": eligible,
        "outputs_materially_merged": merged,
        "base_sets_preserved_on_null": base_preserved_on_null,
        "dedup_iou": args.dedup_iou,
        "task_key_used": False,
        "answer_or_metric_used": False,
        "test_used": False,
        "passed": eligible > 0 and merged > 0,
    }
    write_json(args.audit_out, audit)
    if not audit["passed"]:
        raise RuntimeError(f"executable geometry state gate failed: {audit}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    item = sub.add_parser("build-rois")
    item.add_argument("--manifest", type=Path, required=True); item.add_argument("--controller-decisions", type=Path, required=True)
    item.add_argument("--tile-side", type=int, default=640); item.add_argument("--coverage", choices=["complete", "single"], default="complete"); item.add_argument("--out", type=Path, required=True); item.add_argument("--audit-out", type=Path, required=True); item.set_defaults(func=build_rois)
    item = sub.add_parser("annotate")
    item.add_argument("--mapped", type=Path, required=True); item.add_argument("--rois", type=Path, required=True); item.add_argument("--out", type=Path, required=True); item.set_defaults(func=annotate)
    item = sub.add_parser("build-terminal")
    item.add_argument("--base-tasks", type=Path, required=True); item.add_argument("--manifest", type=Path, required=True); item.add_argument("--controller-decisions", type=Path, required=True); item.add_argument("--mapped", type=Path, required=True); item.add_argument("--coverage", choices=["complete", "single"], default="complete")
    item.add_argument("--assimilation", choices=["typed", "untyped"], default="typed")
    item.add_argument("--out", type=Path, required=True); item.add_argument("--active-out", type=Path, required=True); item.add_argument("--state-out", type=Path, required=True); item.add_argument("--audit-out", type=Path, required=True); item.set_defaults(func=build_terminal)
    item = sub.add_parser("combine")
    item.add_argument("--base", type=Path, required=True); item.add_argument("--candidate", type=Path, required=True); item.add_argument("--states", type=Path, required=True); item.add_argument("--out", type=Path, required=True); item.add_argument("--audit-out", type=Path, required=True); item.set_defaults(func=combine)
    item = sub.add_parser("closure-guard")
    item.add_argument("--base", type=Path, required=True); item.add_argument("--full", type=Path, required=True); item.add_argument("--manifest", type=Path, required=True); item.add_argument("--out", type=Path, required=True); item.add_argument("--audit-out", type=Path, required=True); item.set_defaults(func=closure_guard)
    item = sub.add_parser("closure-admission")
    item.add_argument("--manifest", type=Path, required=True); item.add_argument("--decisions", type=Path, required=True); item.add_argument("--mapped", type=Path, required=True); item.add_argument("--states", type=Path, required=True); item.add_argument("--terminal", type=Path, required=True); item.add_argument("--out-dir", type=Path, required=True); item.add_argument("--audit-out", type=Path, required=True); item.set_defaults(func=closure_admission)
    item = sub.add_parser("executable-geometry-state")
    item.add_argument("--base", type=Path, required=True); item.add_argument("--full", type=Path, required=True); item.add_argument("--manifest", type=Path, required=True); item.add_argument("--states", type=Path, required=True); item.add_argument("--dedup-iou", type=float, default=0.8); item.add_argument("--out", type=Path, required=True); item.add_argument("--audit-out", type=Path, required=True); item.set_defaults(func=executable_geometry_state)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    raise SystemExit(args.func(args))
