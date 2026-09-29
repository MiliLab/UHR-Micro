#!/usr/bin/env python3
"""Task-name-free multistage runtime for evidence-contract strong-path perception.

This module implements the shared global-context, support-planning,
native-resolution observation, source-registration, and registered-synthesis
trajectory.  It does not read benchmark task names, answers, annotations, or
metrics.  Ablation-specific feedback transitions are implemented separately
on top of this shared trajectory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from string import Template
from typing import Any, Iterable


HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[4]
HIST_SCRIPT = HERE.parent
PROMPTS = json.loads((HERE.parents[1] / "prompts.json").read_text(encoding="utf-8"))
if str(HIST_SCRIPT) not in sys.path:
    sys.path.insert(0, str(HIST_SCRIPT))

from common import (  # noqa: E402
    ROI_SIDE,
    abs_box_to_full1000,
    box_intersects,
    box_iou,
    clamp_box_abs,
    expand_box_to_roi,
    extract_numeric_lists,
    format_list,
    full1000_point_to_abs,
    parse_points,
    point_to_roi_abs,
)


REGION_OF = re.compile(r"\s+(?:in|near|within)\s+the\s+region\s+of\s*\[[^\]]+\]", re.IGNORECASE)
REMOTE_IMAGE = re.compile(r"\s+in\s+this\s+remote\s+sensing\s+image", re.IGNORECASE)
NUMBER_OF = re.compile(r"number\s+of\s+(.+?)\s+instances", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def keyed(rows: Iterable[dict[str, Any]], key: str = "sample_id") -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        ident = str(row.get(key) or "")
        if not ident or ident in output:
            raise ValueError(f"missing or duplicate {key}: {ident}")
        output[ident] = row
    return output


def strong_template(stage: str, key: str, **values: object) -> str:
    try:
        text = PROMPTS["public_observers"][stage][key]
    except KeyError as exc:
        raise FileNotFoundError(f"missing prompt template: public_observers/{stage}/{key}") from exc
    return Template(text).safe_substitute(
        {name: str(value) for name, value in values.items()}
    ).strip()


def native_prompt(row: dict[str, Any]) -> str:
    schema = str(row["evidence_state"]["answer_schema"])
    contract = {
        "OPTION": "Return one final field whose literal label is `Final answer:` and whose value is exactly one valid option letter. Do not repeat the options or add an explanation.",
        "SCALAR": "Return one final field whose literal label is `Final answer:` and whose value is exactly one non-negative integer. Do not add an explanation.",
        "SINGLE_GEOMETRY": "Use full-image 1000-base relative coordinates. Return one final field whose literal label is `Final answer:` and whose value is one four-coordinate bounding box.",
        "MASK_SUPPORT": "Use full-image 1000-base relative coordinates. Return one final field whose literal label is `Final answer:` and whose value is one four-coordinate bounding box.",
        "GEOMETRY_SET": "Use full-image 1000-base relative coordinates. Return one final field whose literal label is `Final answer:` and whose value is a comma-separated list of target boxes; use null only when no target is present.",
    }[schema]
    return f"""Inspect the original full remote-sensing image and answer the task.
Coordinates in the question and answer use the full-image 1000-base relative coordinate system whenever coordinates are required.

Task:
{row['question_1000']}

Output contract:
{contract}"""


def native_prompt_no_topology(row: dict[str, Any]) -> str:
    return f"""Inspect the original complete ultra-high-resolution remote-sensing image and form a global answer hypothesis.

Question:
{row['question_1000']}

Preserve complete-scene scope and global geometry. Do not invent tiny details that are not visibly resolved.

{row['final_format_hint']}
Do not repeat the question or candidate options in the final field."""


def planner_prompt(row: dict[str, Any]) -> str:
    return strong_template(
        "stage1_points",
        str(row["stage1_template_key"]),
        H=int(row["original_height_px"]),
        W=int(row["original_width_px"]),
        question=row["question_stage1"],
    )


def planner_prompt_no_topology(row: dict[str, Any]) -> str:
    return f"""You are viewing the complete remote-sensing image. Select native-resolution observations that may help answer the public question; do not answer it.

Question without final-answer formatting:
{row['question_stage1']}

All points use the complete source image's 0-1000 coordinate system. Return 1 to 4 useful, spatially separated points. Return only `[[x,y],...]`; do not output boxes, an answer, confidence, or an explanation."""


def stage_task_id(row: dict[str, Any], stage: str, disable_topology: bool) -> str:
    namespace = "generic" if disable_topology else "topology"
    return f"{row['qa_slug']}_{namespace}_{stage}"


def prepare(args: argparse.Namespace) -> int:
    manifest = read_jsonl(args.manifest)
    native: list[dict[str, Any]] = []
    planners: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source in manifest:
        row = copy.deepcopy(source)
        row.update(
            {
                "task_id": stage_task_id(source, "global", args.disable_topology),
                "stage_name": "complete_image_context_no_topology" if args.disable_topology else "complete_image_context",
                "model_image_path": source["image_abs_path"],
                "model_input_prompt": native_prompt_no_topology(source) if args.disable_topology else native_prompt(source),
                "evidence_topology_enabled": not args.disable_topology,
            }
        )
        native.append(row)
        stage1_needed = not bool(source.get("seed_boxes_abs_from_question")) if args.disable_topology else bool(source["stage1_needed"])
        if stage1_needed:
            row = copy.deepcopy(source)
            row.update(
                {
                    "task_id": stage_task_id(source, "plan", args.disable_topology),
                    "stage_name": "generic_question_conditioned_support_plan" if args.disable_topology else "question_conditioned_support_plan",
                    "model_image_path": source["image_abs_path"],
                    "model_input_prompt": planner_prompt_no_topology(source) if args.disable_topology else planner_prompt(source),
                    "evidence_topology_enabled": not args.disable_topology,
                }
            )
            planners.append(row)
        else:
            row = copy.deepcopy(source)
            row.update(
                {
                    "task_id": stage_task_id(source, "plan", args.disable_topology),
                    "stage_name": "public_coordinate_support",
                    "stage1_skipped": True,
                    "stage1_points_full1000": [],
                    "stage1_parse_ok": True,
                    "stage1_parse_error": None,
                    "model_response": None,
                    "evidence_topology_enabled": not args.disable_topology,
                }
            )
            skipped.append(row)
    native.sort(key=lambda row: -int(row["original_width_px"]) * int(row["original_height_px"]))
    planners.sort(key=lambda row: -int(row["original_width_px"]) * int(row["original_height_px"]))
    write_jsonl(args.out_dir / "native_tasks.jsonl", native)
    write_jsonl(args.out_dir / "planner_tasks.jsonl", planners)
    write_jsonl(args.out_dir / "planner_skipped.jsonl", skipped)
    for row in native + planners:
        prompt = str(row["model_input_prompt"])
        if row["sample_id"] in prompt or str(row["qa_relpath"]) in prompt:
            raise PermissionError("opaque runtime identity interpolated into model prompt")
    write_json(
        args.out_dir / "prepare_audit.json",
        {
            "rows": len(manifest),
            "native_tasks": len(native),
            "planner_tasks": len(planners),
            "public_support_skips": len(skipped),
            "task_key_used": False,
            "ground_truth_or_metric_used": False,
            "test_used": False,
            "evidence_topology_enabled": not args.disable_topology,
        },
    )
    return 0


def complete_planner(args: argparse.Namespace) -> int:
    manifest = keyed(read_jsonl(args.manifest))
    results = keyed(read_jsonl(args.results), "task_id") if args.results.is_file() else {}
    skipped = keyed(read_jsonl(args.skipped), "task_id")
    output: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for sid, source in manifest.items():
        ident = stage_task_id(source, "plan", args.disable_topology)
        if ident in skipped:
            row = copy.deepcopy(skipped[ident])
            statuses["public_coordinate_support"] += 1
        else:
            if ident not in results or results[ident].get("error"):
                raise RuntimeError(f"missing or failed planner result: {sid}")
            row = copy.deepcopy(results[ident])
            point_limit = 4 if args.disable_topology else int(source["max_stage1_points"])
            points = parse_points(str(row.get("model_response") or ""), point_limit)
            row["stage1_points_full1000"] = points
            row["stage1_parse_ok"] = bool(points)
            row["stage1_parse_error"] = None if points else "no_points_model_output"
            row["stage1_skipped"] = False
            statuses["model_points" if points else "deterministic_fallback_required"] += 1
        output.append(row)
    write_jsonl(args.out, output)
    write_json(args.out.with_name("planner_complete_audit.json"), {"rows": len(output), "statuses": dict(statuses), "infrastructure_errors": 0, "evidence_topology_enabled": not args.disable_topology})
    return 0


def grid_rois(width: int, height: int, count: int) -> list[list[int]]:
    centers = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.25), (0.5, 0.75), (0.25, 0.5)]
    output: list[list[int]] = []
    for rx, ry in centers:
        roi = point_to_roi_abs(rx * width, ry * height, width, height)
        if roi not in output:
            output.append(roi)
        if len(output) >= count:
            break
    return output


def cover_box(box: list[int], width: int, height: int, max_rois: int = 6) -> list[list[int]]:
    x1, y1, x2, y2 = clamp_box_abs(box, width, height)
    centers = [((x1 + x2) / 2, (y1 + y2) / 2)]
    if x2 - x1 > ROI_SIDE:
        centers.extend([(x1 + ROI_SIDE / 2, (y1 + y2) / 2), (x2 - ROI_SIDE / 2, (y1 + y2) / 2)])
    if y2 - y1 > ROI_SIDE:
        centers.extend([((x1 + x2) / 2, y1 + ROI_SIDE / 2), ((x1 + x2) / 2, y2 - ROI_SIDE / 2)])
    centers.extend([(x1 + ROI_SIDE / 2, y1 + ROI_SIDE / 2), (x2 - ROI_SIDE / 2, y2 - ROI_SIDE / 2)])
    output: list[list[int]] = []
    for cx, cy in centers:
        roi = point_to_roi_abs(cx, cy, width, height)
        if roi not in output:
            output.append(roi)
        if len(output) >= max_rois:
            break
    return output


def minimum_rois(row: dict[str, Any]) -> int:
    state = row["evidence_state"]
    if state["support_extent"] == "FULL_SCENE" or state["coverage_requirement"] == "COMPLETE_SCENE":
        return 4
    if state["answer_schema"] == "GEOMETRY_SET" and int(state.get("explicit_region_count") or 0) == 0:
        return 4
    if state["support_extent"] == "MULTIPLE_SUPPORTS":
        return 2
    if state["answer_schema"] == "MASK_SUPPORT":
        return 2
    return 1


def outward_two(rois: list[list[int]], points: list[list[float]], width: int, height: int) -> list[list[int]]:
    if len(rois) != 2 or len(points) < 2 or not box_intersects(rois[0], rois[1]):
        return rois
    p1, p2 = full1000_point_to_abs(points[0], width, height), full1000_point_to_abs(points[1], width, height)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dx, dy, norm = 1.0, 0.0, 1.0
    for step in range(64, ROI_SIDE * 3 + 1, 64):
        r1 = point_to_roi_abs(p1[0] - dx / norm * step, p1[1] - dy / norm * step, width, height)
        r2 = point_to_roi_abs(p2[0] + dx / norm * step, p2[1] + dy / norm * step, width, height)
        if not box_intersects(r1, r2):
            return [r1, r2]
    return [rois[0]]


def build_rois(args: argparse.Namespace) -> int:
    output: list[dict[str, Any]] = []
    for row in read_jsonl(args.planner_complete):
        width, height = int(row["original_width_px"]), int(row["original_height_px"])
        raw: list[list[int]] = []
        source = "PUBLIC_COORDINATE_SUPPORT" if args.disable_topology and row.get("seed_boxes_abs_from_question") else str(row["support_source"])
        boxes = row.get("seed_boxes_abs_from_question") or []
        if source == "PUBLIC_COORDINATE_SUPPORT" and boxes:
            for box in boxes:
                if args.disable_topology:
                    raw.extend(expand_box_to_roi([int(v) for v in box[:4]], width, height))
                elif row["evidence_state"]["evidence_composition"] == "NUMERIC_DIFFERENCE" or row["evidence_state"]["answer_schema"] == "SCALAR":
                    raw.extend(expand_box_to_roi([int(v) for v in box[:4]], width, height))
                elif row["evidence_state"]["answer_schema"] == "GEOMETRY_SET":
                    raw.extend(cover_box([int(v) for v in box[:4]], width, height))
                else:
                    raw.extend(expand_box_to_roi([int(v) for v in box[:4]], width, height))
        else:
            for point in row.get("stage1_points_full1000") or []:
                cx, cy = full1000_point_to_abs(point, width, height)
                raw.append(point_to_roi_abs(cx, cy, width, height))
        target = 2 if args.disable_topology else minimum_rois(row)
        if not raw:
            raw = grid_rois(width, height, target)
            source = "DETERMINISTIC_COVERAGE_FALLBACK"
        elif len(raw) < target:
            raw.extend(box for box in grid_rois(width, height, target + 4) if box not in raw)
            raw = raw[:target]
        raw = outward_two(raw, row.get("stage1_points_full1000") or [], width, height)
        kept: list[list[int]] = []
        for box in raw:
            box = clamp_box_abs(box, width, height)
            if all(box_iou(box, prior) <= float(args.nms_iou) for prior in kept):
                kept.append(box)
        for index, box in enumerate(kept, start=1):
            output.append(
                {
                    **row,
                    "roi_id": f"{row['qa_slug']}_topo_roi{index:02d}",
                    "roi_index": index,
                    "roi_abs_xyxy": box,
                    "roi_full1000_xyxy": abs_box_to_full1000(box, width, height),
                    "roi_width_px": box[2] - box[0],
                    "roi_height_px": box[3] - box[1],
                    "roi_source": source,
                }
            )
    output.sort(key=lambda row: -int(row["roi_width_px"]) * int(row["roi_height_px"]))
    write_jsonl(args.out, output)
    write_json(
        args.out.with_name("roi_summary.json"),
        {
            "qas": len({row["sample_id"] for row in output}),
            "rois": len(output),
            "mean_rois": len(output) / max(1, len({row["sample_id"] for row in output})),
            "fallback_qas": len({row["sample_id"] for row in output if row["roi_source"] == "DETERMINISTIC_COVERAGE_FALLBACK"}),
            "task_key_used": False,
            "evidence_topology_enabled": not args.disable_topology,
        },
    )
    return 0


def intersection(a: list[int], b: list[int]) -> list[int] | None:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def local_box(box: list[int], roi: list[int]) -> list[float]:
    rw, rh = max(1, roi[2] - roi[0]), max(1, roi[3] - roi[1])
    return [
        round(max(0.0, min(1000.0, (box[0] - roi[0]) / rw * 1000)), 1),
        round(max(0.0, min(1000.0, (box[1] - roi[1]) / rh * 1000)), 1),
        round(max(0.0, min(1000.0, (box[2] - roi[0]) / rw * 1000)), 1),
        round(max(0.0, min(1000.0, (box[3] - roi[1]) / rh * 1000)), 1),
    ]


def pair_label(row: dict[str, Any]) -> str:
    roi = [int(v) for v in row["roi_abs_xyxy"]]
    boxes = [[int(v) for v in box[:4]] for box in row.get("seed_boxes_abs_from_question") or [] if len(box) >= 4]
    if len(boxes) >= 2:
        rcx, rcy = (roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2
        scores = []
        for index, box in enumerate(boxes[:2]):
            inter = intersection(roi, box)
            area = 0 if inter is None else (inter[2] - inter[0]) * (inter[3] - inter[1])
            bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            scores.append((area * 1_000_000 - (rcx - bcx) ** 2 - (rcy - bcy) ** 2, index))
        return "A" if max(scores)[1] == 0 else "B"
    return "A" if int(row["roi_index"]) == 1 else "B"


def local_question(row: dict[str, Any], intersections: list[list[int]], disable_topology: bool = False) -> str:
    question = str(row.get("question_stage1") or row.get("question_roi") or row.get("question_original") or "").strip()
    state = row["evidence_state"]
    roi = [int(v) for v in row["roi_abs_xyxy"]]
    if not disable_topology and state["evidence_composition"] == "NUMERIC_DIFFERENCE":
        match = NUMBER_OF.search(question)
        target = re.sub(r"\s+", " ", match.group(1)).strip() if match else "requested target"
        label = pair_label(row)
        if intersections:
            supports = [local_box(box, roi) for box in intersections]
            return f"Count {target} instances for region {label} only. Only include objects inside the ROI-local 1000-base region(s) {supports} of this crop. Output only one Arabic numeral."
        return f"Count {target} instances for region {label} only, using only the visible content inside this ROI crop. Output only one Arabic numeral."
    explicit_region = int(state.get("explicit_region_count") or 0) > 0 and state["answer_schema"] in {"SCALAR", "GEOMETRY_SET"}
    if explicit_region:
        rewritten = REGION_OF.sub("", question).strip()
        rewritten = REMOTE_IMAGE.sub("", rewritten).strip()
        rewritten = re.sub(r"\s{2,}", " ", rewritten) or question
        if not intersections:
            return f"{rewritten} in this ROI crop."
        roi_area = max(1, (roi[2] - roi[0]) * (roi[3] - roi[1]))
        inter = intersections[0]
        inter_area = max(0, (inter[2] - inter[0]) * (inter[3] - inter[1]))
        suffix = "in this ROI crop." if inter_area / float(roi_area) >= 0.98 else f"inside the ROI-local 1000-base region {format_list(local_box(inter, roi))} of this crop."
        return f"{rewritten} {suffix}"
    return str(row.get("question_roi") or row.get("question_original") or "").strip()


def roi_context(row: dict[str, Any], intersections: list[list[int]], disable_topology: bool = False) -> str:
    if disable_topology:
        return f"Observation R{row['roi_index']} covers source-image 0-1000 box {format_list(row['roi_full1000_xyxy'])}."
    parts = [f"Current ROI in full-image 1000-base coordinates: {row.get('roi_full1000_xyxy')}."]
    if intersections:
        local_intersections = [local_box(inter, [int(value) for value in row["roi_abs_xyxy"]]) for inter in intersections]
        state = row["evidence_state"]
        if state["evidence_composition"] == "NUMERIC_DIFFERENCE":
            parts.append(f"This ROI is assigned to cross-region side {pair_label(row)}. Only use ROI-local 1000-base intersection boxes: {local_intersections}.")
        elif state["answer_schema"] == "GEOMETRY_SET":
            parts.append(f"The original requested region intersects this ROI at these ROI-local 1000-base boxes: {local_intersections}. Do not use full-image coordinates in your answer.")
        else:
            parts.append("For region-based tasks, only count or detect inside the visible intersection between the original question region and this ROI. " f"Those intersection boxes are ROI-local 1000-base: {local_intersections}. Do not use full-image coordinates in your answer.")
    return " ".join(parts)


def observer_prompt(row: dict[str, Any], question: str, context: str, disable_topology: bool = False) -> str:
    if not disable_topology:
        return strong_template(
            "stage3_roi",
            str(row["stage3_template_key"]),
            question=question,
            roi_context=context,
        )
    observer = row["execution_plan"]["local_observer"]
    schema = row["evidence_state"]["answer_schema"]
    if observer == "OBSERVE_PAIRED_COUNTS" or schema == "SCALAR":
        contract = "Count only matching instances visibly supported inside this observation. Return one integer."
    elif schema == "OPTION":
        contract = "Use only visible evidence inside this observation. Return one option letter if sufficient; otherwise return null."
    elif schema in {"SINGLE_GEOMETRY", "MASK_SUPPORT"}:
        contract = "Return one tight target box in observation-local 0-1000 coordinates, or null if absent."
    else:
        contract = "Return every visible matching target as a box in observation-local 0-1000 coordinates, or null if none."
    emphasis = "Inspect question-relevant evidence visible in this observation." if disable_topology else {
        "OBSERVE_SCENE_CONFIGURATION": "Preserve spatial ordering and configuration; local evidence alone may be incomplete.",
        "OBSERVE_FINE_TARGET": "Inspect subtype-defining fine visual details rather than coarse category alone.",
        "OBSERVE_OBJECT_PART": "Localize only the requested physical part, not the whole object.",
        "OBSERVE_MARKED_RELATION": "Use the visible marked objects and their source-relative geometry.",
        "OBSERVE_EXPLICIT_SET_SUPPORT": "Check the complete stated support and do not include evidence outside it.",
        "OBSERVE_LANGUAGE_SET_SUPPORT": "Check all visible candidates belonging to the language-described support.",
    }.get(observer, "Inspect the question-conditioned target evidence at native detail.")
    return f"""You are viewing one native-resolution observation cropped from a larger ultra-high-resolution remote-sensing image.

Source-registration context:
{context}

Evidence query:
{question}

{emphasis}
{contract}
Never reinterpret source-image coordinates as observation-local coordinates. Do not infer objects outside the displayed observation, repeat alternatives, or add an explanation."""


def build_stage3(args: argparse.Namespace) -> int:
    output: list[dict[str, Any]] = []
    for source in read_jsonl(args.rois):
        row = copy.deepcopy(source)
        meta = args.crop_dir / str(row["roi_id"]) / "roi_meta.json"
        if meta.is_file():
            row = json.loads(meta.read_text(encoding="utf-8"))
        roi = [int(v) for v in row["roi_abs_xyxy"]]
        intersections = [inter for box in row.get("seed_boxes_abs_from_question") or [] if len(box) >= 4 if (inter := intersection(roi, [int(v) for v in box[:4]]))]
        question = local_question(row, intersections, args.disable_topology)
        context = roi_context(row, intersections, args.disable_topology)
        task = copy.deepcopy(row)
        # The ROI manifest retains the proposal-stage response for provenance.
        # Keep it under an explicit name and clear the response slot consumed by
        # the local-observation runner so a failed observation can never look
        # like a successful planner answer.
        planner_model_response = task.get("model_response")
        task.update(
            {
                "task_id": f"{row['roi_id']}_{'generic' if args.disable_topology else 'topology'}_observe",
                "stage_name": "native_local_observation_no_topology" if args.disable_topology else "native_local_observation",
                "question_roi_local": question,
                "source_support_label": pair_label(row) if row["evidence_state"]["evidence_composition"] == "NUMERIC_DIFFERENCE" else None,
                "model_image_path": str((args.crop_dir / str(row["roi_id"]) / "roi.png").resolve()),
                "model_input_prompt": observer_prompt(row, question, context, args.disable_topology),
                "planner_model_response": planner_model_response,
                "model_response": None,
                "evidence_topology_enabled": not args.disable_topology,
            }
        )
        output.append(task)
    output.sort(key=lambda row: -int(row["roi_width_px"]) * int(row["roi_height_px"]))
    write_jsonl(args.out, output)
    write_json(args.out.with_name("stage3_task_audit.json"), {"tasks": len(output), "qas": len({row["sample_id"] for row in output}), "task_key_used": False, "test_used": False, "evidence_topology_enabled": not args.disable_topology})
    return 0


def stage3_evidence_text(rows: list[dict[str, Any]], registration: bool) -> str:
    lines: list[str] = []
    candidates: list[str] = []
    for row in sorted(rows, key=lambda item: int(item.get("roi_index") or 0)):
        raw = str(row.get("model_response") or "").strip()
        mapped = row.get("stage3_mapped_response_full1000") if registration else None
        if registration:
            lines.append(f"ROI {row.get('roi_index')}: mapped_full1000={mapped}; raw_roi_answer={raw}")
            lists = row.get("stage3_mapped_coord_lists_full1000") or []
            if lists:
                for values in lists:
                    candidates.append(f"ROI {row.get('roi_index')}: {values}")
            elif mapped and str(mapped).strip().lower() not in {"null", "none", ""}:
                for values in extract_numeric_lists(str(mapped), lengths=(4, 8)):
                    candidates.append(f"ROI {row.get('roi_index')}: {values}")
        else:
            lines.append(f"Observation R{row.get('roi_index')}: unregistered_local={raw}")
            for values in extract_numeric_lists(raw, lengths=(4, 8)):
                candidates.append(f"R{row.get('roi_index')}: local_only={values}")
    if candidates:
        if registration:
            lines.insert(0, "Structured full-image 1000-base candidate boxes copied from ROI answers:")
            lines[1:1] = candidates
        else:
            lines.insert(0, "Structured geometry evidence: " + "; ".join(candidates))
    return "\n".join(lines) if lines else ("No ROI answers." if registration else "No usable local observation.")


def synthesis_prompt(
    row: dict[str, Any],
    evidence_text: str,
    feedback_text: str,
    native_text: str,
    registration: bool,
) -> str:
    if not registration:
        return synthesis_prompt_no_topology(row, evidence_text, native_text, registration)
    base = strong_template(
        "stage4_finalize",
        str(row["terminal_template_key"]),
        native_answer_text=native_text,
        question_original=row["question_original"],
        stage3_results_text=evidence_text,
    )
    if feedback_text:
        base += "\n\nAdditional contract-reopened evidence:\n" + feedback_text
    return base


def synthesis_prompt_no_topology(
    row: dict[str, Any],
    evidence_text: str,
    native_text: str,
    registration: bool,
) -> str:
    registration_text = (
        "Local evidence has been mapped into the complete source-image 0-1000 frame."
        if registration
        else "This ablation removes source registration: local coordinates have no source-frame ownership."
    )
    return f"""Synthesize the final answer from the complete-image hypothesis and the available native-resolution observations.

Original public question:
{row['question_original']}

{registration_text}

Complete-image hypothesis:
{native_text}

Local observation evidence:
{evidence_text}

{row['final_format_hint']}
You may reason briefly, but the last line must begin with the literal label `Final answer:`. Do not repeat candidate options, name a benchmark task, or include a worked example."""


def build_stage4(args: argparse.Namespace) -> int:
    mapped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.mapped):
        mapped[str(row["sample_id"])].append(row)
    native = keyed(read_jsonl(args.native), "sample_id")
    unregistered_contexts = keyed(read_jsonl(args.unregistered_contexts), "sample_id") if args.unregistered_contexts else {}
    if args.disable_registration and not unregistered_contexts:
        raise ValueError("--disable-registration requires --unregistered-contexts with overlay-size-matched plain images")
    feedback: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    enabled_kinds = {value.strip() for value in str(args.feedback_kinds or "").split(",") if value.strip()}
    if args.feedback and args.feedback.is_file():
        for item in read_jsonl(args.feedback):
            if not enabled_kinds or str(item.get("feedback_kind")) in enabled_kinds:
                feedback[str(item["sample_id"])].append(item)
    output: list[dict[str, Any]] = []
    for source in read_jsonl(args.overlays):
        sid = str(source["sample_id"])
        evidence = stage3_evidence_text(mapped.get(sid, []), not args.disable_registration)
        feedback_text = "\n".join(
            f"{item.get('feedback_kind')}: "
            + str(
                (item.get('feedback_mapped_response') or item.get('model_response'))
                if not args.disable_registration
                else (item.get('model_response') or 'null')
            )
            for item in feedback.get(sid, [])
        )
        # The validated strong trajectory exposes the native full-image answer
        # to synthesis only when the public contract calls for global
        # configuration review.  For all other contracts it remains a separate
        # observation and is not injected as a textual answer prior.
        native_text = (
            str(native[sid].get("model_response") or "No native full-image answer is available.")
            if str(source.get("special_operator")) == "GLOBAL_CONFIGURATION_REVIEW"
            else "No native full-image answer is available for this task."
        )
        row = copy.deepcopy(source)
        upstream_model_response = row.get("model_response")
        suffix = "generic" if args.disable_topology else "topology"
        if args.disable_registration:
            suffix += "_noreg"
        row.update(
            {
                "task_id": f"{source['qa_slug']}_{suffix}_synthesis",
                "stage_name": ("global_local_synthesis_no_topology" if args.disable_topology else "global_local_registered_synthesis") if not args.disable_registration else "global_local_unregistered_synthesis_ablation",
                "stage3_results_text": evidence,
                "native_answer_text": native_text,
                "source_registration_enabled": not args.disable_registration,
                "evidence_topology_enabled": not args.disable_topology,
                "model_image_path": unregistered_contexts[sid]["unregistered_context_image_path"] if args.disable_registration else source["overlay_image_path"],
                "context_size_matched_to_registered_overlay": bool(args.disable_registration),
                "feedback_evidence_text": feedback_text,
                "feedback_kinds_enabled": sorted(enabled_kinds),
                "model_input_prompt": synthesis_prompt_no_topology(source, evidence, native_text, not args.disable_registration) if args.disable_topology else synthesis_prompt(source, evidence, feedback_text, native_text, not args.disable_registration),
                "upstream_model_response": upstream_model_response,
                "model_response": None,
            }
        )
        output.append(row)
    output.sort(key=lambda row: -int(row.get("original_width_px") or 0) * int(row.get("original_height_px") or 0))
    write_jsonl(args.out, output)
    write_json(
        args.out.with_name(args.out.stem + "_audit.json"),
        {
            "tasks": len(output),
            "unique_sample_ids": len({row["sample_id"] for row in output}),
            "source_registration_enabled": not args.disable_registration,
            "evidence_topology_enabled": not args.disable_topology,
            "feedback_file": str(args.feedback) if args.feedback else None,
            "feedback_kinds_enabled": sorted(enabled_kinds),
            "task_key_used": False,
            "ground_truth_or_metric_used": False,
            "test_used": False,
            "unregistered_context_manifest": str(args.unregistered_contexts) if args.unregistered_contexts else None,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--out-dir", type=Path, required=True)
    item.add_argument("--disable-topology", action="store_true")
    item.set_defaults(func=prepare)
    item = sub.add_parser("complete-planner")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--results", type=Path, required=True)
    item.add_argument("--skipped", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--disable-topology", action="store_true")
    item.set_defaults(func=complete_planner)
    item = sub.add_parser("build-rois")
    item.add_argument("--planner-complete", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--nms-iou", type=float, default=0.2)
    item.add_argument("--disable-topology", action="store_true")
    item.set_defaults(func=build_rois)
    item = sub.add_parser("build-stage3")
    item.add_argument("--rois", type=Path, required=True)
    item.add_argument("--crop-dir", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--disable-topology", action="store_true")
    item.set_defaults(func=build_stage3)
    item = sub.add_parser("build-stage4")
    item.add_argument("--overlays", type=Path, required=True)
    item.add_argument("--mapped", type=Path, required=True)
    item.add_argument("--native", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--disable-registration", action="store_true")
    item.add_argument("--unregistered-contexts", type=Path)
    item.add_argument("--disable-topology", action="store_true")
    item.add_argument("--feedback", type=Path)
    item.add_argument("--feedback-kinds", default="")
    item.set_defaults(func=build_stage4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
