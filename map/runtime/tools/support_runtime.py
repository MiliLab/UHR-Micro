#!/usr/bin/env python3
"""Unified public support acquisition and coordinate-correct local observers."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from string import Template
from typing import Any

from io_utils import keyed, read_jsonl, write_json, write_jsonl


METHOD = Path(__file__).resolve().parents[1]
PROMPTS = json.loads((METHOD / "prompts.json").read_text(encoding="utf-8"))
if str(METHOD) not in sys.path:
    sys.path.insert(0, str(METHOD))

from view_geometry import ViewGeometry, ViewGeometryError  # noqa: E402
from prepare_public import (  # noqa: E402
    BOUNDED_ONE_LIST_CLAUSE_RE,
    BOUNDED_TWO_CORNER_CLAUSE_RE,
    TWO_CORNER_REGION_RE,
    strip_answer_instruction,
)


ROI_SIDE = 1024
MIN_PLANNER_POINT_DISTANCE_1000 = 150.0
TARGET_DETAIL_MIN_SIDE_PX = ROI_SIDE // 8
TARGET_DETAIL_EXPAND_RATIO = 0.25
POINT_RE = re.compile(
    r"\[\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\]"
)
LIST_RE = re.compile(r"\[([^\[\]]+)\]")
AXIS_THRESHOLD_RE = re.compile(
    r"center\s+([xy])-coordinate\s+in\s+the\s+original\s+image\s+is\s+at\s+or\s+"
    r"(left|right|above|below)(?:\s+of)?\s+(-?(?:\d+(?:\.\d*)?|\.\d+))\s+pixels?",
    re.IGNORECASE,
)
SOURCE_ABSOLUTE_LIST_RE = re.compile(
    r"\[\s*-?(?:\d+(?:\.\d*)?|\.\d+)\s*,", re.IGNORECASE
)
PUBLIC_TARGET_LOCATOR_RE = re.compile(
    r"object\s+indicated\s+by\s+(?:the\s+)?(?:"
    r"(?:(?:axis[- ]aligned|oriented)\s+)?bounding\s+box"
    r"(?:\s+with\s+vertex\s+coordinates)?|"
    r"center\s+point(?:\s+with\s+vertex\s+coordinates)?"
    r")\s*\[[^\[\]]+\]",
    re.IGNORECASE,
)


def render(values: list[float]) -> str:
    def one(value: float) -> str:
        return (
            str(int(round(value)))
            if abs(value - round(value)) < 1e-7
            else f"{value:.4f}".rstrip("0").rstrip(".")
        )

    return "[" + ",".join(one(float(value)) for value in values) + "]"


def template(stage: str, key: str, row: dict[str, Any]) -> str:
    try:
        text = PROMPTS[stage][key]
    except KeyError as exc:
        raise FileNotFoundError(f"missing prompt template: {stage}/{key}") from exc
    return Template(text).substitute(
        H=int(row["original_height_px"]),
        W=int(row["original_width_px"]),
        question=str(row["question_stage1"]),
        support_instruction=support_instruction(row),
    ).strip()


def support_instruction(row: dict[str, Any]) -> str:
    """Describe support cardinality from the public contract, never task identity."""

    contract = row["evidence_contract"]
    maximum = int(row["max_stage1_points"])
    minimum = minimum_views(row)
    if contract["support_topology"] == "PAIRWISE_SUPPORT":
        return (
            "Return exactly two representative points in question-mention order: "
            "the first for the first referenced support or object, and the second "
            "for the second referenced support or object."
        )
    if not 1 <= minimum <= maximum:
        raise ValueError(
            f"public support cardinality is inconsistent: minimum={minimum}, "
            f"maximum={maximum}"
        )
    cardinality = (
        f"exactly {minimum}"
        if minimum == maximum
        else f"between {minimum} and {maximum}"
    )
    return (
        f"Return {cardinality} representative support points. "
        f"When returning multiple points, keep distinct points at least "
        f"{int(MIN_PLANNER_POINT_DISTANCE_1000)} units apart in the source-image "
        "0-1000 coordinate system and distribute them over the question-relevant "
        "evidence."
    )


def native_prompt(row: dict[str, Any]) -> str:
    return f"""Inspect the original complete ultra-high-resolution remote-sensing image and record a provisional complete-scene observation for a later global-local synthesis stage.

Public question:
{row['question_stage1']}

Any numeric spatial coordinate above has already been rewritten to the complete source image's 0-1000 system, even if the original wording still says pixels. Preserve complete-scene scope and global geometry. Do not invent tiny details that are not visibly resolved. This is evidence, not the terminal answer; a later composer will combine it with native-resolution local observations.

The public question's terminal answer-format instruction has been removed for this evidence stage. {row['provisional_format_hint']}
Prefix the typed result with `Provisional evidence:`. Do not repeat the question or candidate options and do not provide a worked answer example."""


def prepare(args: argparse.Namespace) -> int:
    native, planner, skipped = [], [], []
    for source in read_jsonl(args.manifest):
        row = copy.deepcopy(source)
        row.update(
            {
                "task_id": f"{source['qa_slug']}_clean_global",
                "stage_name": "complete_scene_observation",
                "model_image_path": source["image_abs_path"],
                "model_input_prompt": native_prompt(source),
                "model_response": None,
            }
        )
        native.append(row)
        plan = copy.deepcopy(source)
        plan["task_id"] = f"{source['qa_slug']}_clean_plan"
        if source["stage1_needed"]:
            plan.update(
                {
                    "stage_name": "public_support_proposal",
                    "model_image_path": source["image_abs_path"],
                    "model_input_prompt": template(
                        "stage1_points", str(source["stage1_template_key"]), source
                    ),
                    "model_response": None,
                }
            )
            planner.append(plan)
        else:
            plan.update(
                {
                    "stage_name": "public_support_proposal_skipped",
                    "stage1_skipped": True,
                    "stage1_points_full1000": [],
                    "stage1_parse_ok": True,
                    "stage1_parse_error": None,
                    "model_response": None,
                }
            )
            skipped.append(plan)
    native.sort(
        key=lambda row: -int(row["original_width_px"])
        * int(row["original_height_px"])
    )
    planner.sort(
        key=lambda row: -int(row["original_width_px"])
        * int(row["original_height_px"])
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "native_tasks.jsonl", native)
    write_jsonl(args.out_dir / "planner_tasks.jsonl", planner)
    write_jsonl(args.out_dir / "planner_skipped.jsonl", skipped)
    write_json(
        args.out_dir / "prepare_audit.json",
        {
            "schema_version": "map-support-prepare-audit-1",
            "rows": len(native),
            "native_tasks": len(native),
            "planner_tasks": len(planner),
            "public_support_skips": len(skipped),
            "task_identity_used": False,
            "test_used": False,
            "passed": len(native) == len(planner) + len(skipped),
        },
    )
    return 0


def strict_points(
    text: str,
    limit: int,
    require_pair: bool,
    minimum: int | None = None,
) -> list[list[float]]:
    raw = str(text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(value, list) or not value:
        return []
    points: list[list[float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            return []
        try:
            point = [float(item[0]), float(item[1])]
        except (TypeError, ValueError):
            return []
        if not all(
            math.isfinite(coordinate) and 0.0 <= coordinate <= 1000.0
            for coordinate in point
        ):
            return []
        # A repeated valid proposal is not a reason to discard every usable
        # point.  Deduplicate first, then enforce the public maximum on unique
        # supports.  Exact pairwise proposals remain a two-slot contract.
        if point not in points:
            points.append(point)
    required = 2 if require_pair else 1 if minimum is None else int(minimum)
    if required < 1 or required > limit:
        raise ValueError(
            f"invalid public point cardinality: minimum={required}, maximum={limit}"
        )
    if len(points) > limit:
        return []
    if require_pair and len(points) != 2:
        return []
    # A non-pairwise partial plan is useful evidence. Preserve its valid points;
    # ``build_rois`` deterministically supplements only the missing minimum.
    if not require_pair and not points:
        return []
    return points


def complete_planner(args: argparse.Namespace) -> int:
    manifest = keyed(read_jsonl(args.manifest))
    results = keyed(read_jsonl(args.results), "task_id")
    skipped = keyed(read_jsonl(args.skipped), "task_id")
    output = []
    statuses: Counter[str] = Counter()
    for sid, source in manifest.items():
        task_id = f"{source['qa_slug']}_clean_plan"
        if task_id in skipped:
            row = copy.deepcopy(skipped[task_id])
            statuses["public_support"] += 1
        else:
            result = results.get(task_id)
            if result is None or result.get("error"):
                raise RuntimeError(f"missing infrastructure-success planner output: {sid}")
            row = copy.deepcopy(result)
            pairwise = source["evidence_contract"]["support_topology"] == "PAIRWISE_SUPPORT"
            points = strict_points(
                str(row.get("model_response") or ""),
                int(source["max_stage1_points"]),
                pairwise,
                minimum_views(source),
            )
            row.update(
                {
                    "stage1_skipped": False,
                    "stage1_points_full1000": points,
                    "stage1_parse_ok": bool(points),
                    "stage1_parse_error": None if points else "no_strict_valid_points",
                }
            )
            statuses["strict_model_points" if points else "fallback_required"] += 1
        output.append(row)
    write_jsonl(args.out, output)
    write_json(
        args.audit,
        {
            "schema_version": "map-planner-completion-audit-1",
            "rows": len(output),
            "statuses": dict(sorted(statuses.items())),
            "coordinate_clamping_used": False,
            "coordinate_guessing_used": False,
            "task_identity_used": False,
            "test_used": False,
            "passed": len(output) > 0,
        },
    )
    return 0


def source1000_box(box: list[int], width: int, height: int) -> list[float]:
    return [
        round(box[0] / width * 1000.0, 4),
        round(box[1] / height * 1000.0, 4),
        round(box[2] / width * 1000.0, 4),
        round(box[3] / height * 1000.0, 4),
    ]


def center_roi(x: float, y: float, width: int, height: int) -> list[int]:
    side_w, side_h = min(ROI_SIDE, width), min(ROI_SIDE, height)
    x0 = int(round(x - side_w / 2.0))
    y0 = int(round(y - side_h / 2.0))
    x0 = max(0, min(width - side_w, x0))
    y0 = max(0, min(height - side_h, y0))
    return [x0, y0, x0 + side_w, y0 + side_h]


def point_roi(point: list[float], width: int, height: int) -> list[int]:
    if len(point) != 2 or not all(0.0 <= value <= 1000.0 for value in point):
        raise ValueError(f"invalid strict source1000 point: {point}")
    return center_roi(point[0] / 1000.0 * width, point[1] / 1000.0 * height, width, height)


def box_aabb(values: list[int]) -> list[int]:
    if len(values) == 4:
        return values[:]
    if len(values) == 8:
        return [min(values[0::2]), min(values[1::2]), max(values[0::2]), max(values[1::2])]
    raise ValueError(f"unsupported public support geometry: {values}")


def validate_source_box(values: list[int], width: int, height: int) -> list[int]:
    box = box_aabb(values)
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError(f"invalid public source box: {values}")
    return box


def support_context_roi(values: list[int], width: int, height: int) -> list[int]:
    """Return the historical MAP 1024 crop centred on public support."""

    x0, y0, x1, y1 = validate_source_box(values, width, height)
    return center_roi((x0 + x1) / 2.0, (y0 + y1) / 2.0, width, height)


def target_detail_enabled(row: dict[str, Any], values: list[int]) -> bool:
    """Return whether public semantics request historical detail magnification.

    ``TARGET_DETAIL`` is compiled from a public fine-detail requirement. A
    public target box is also required, so this operator can never synthesize
    a crop from hidden object geometry.
    """

    contract = row["evidence_contract"]
    if (
        contract.get("support_extent") != "SINGLE_TARGET"
        or contract.get("support_source") != "PUBLIC_TARGET_SUPPORT"
        or "TARGET_DETAIL" not in contract.get("operator_activations", [])
    ):
        return False
    validate_source_box(
        values,
        int(row["original_width_px"]),
        int(row["original_height_px"]),
    )
    return True


def target_detail_layout(
    values: list[int], width: int, height: int
) -> dict[str, list[int]]:
    """Reproduce the historical per-axis expand-25%, pad, resize-1024 layout.

    The virtual crop canvas is allowed to extend beyond the source image.  Its
    missing pixels remain black; ``export_views`` records the visible source
    rectangle inside that virtual canvas before the forced 1024-by-1024
    bilinear resize.
    """

    x0, y0, x1, y1 = validate_source_box(values, width, height)
    box_width = max(1.0, float(x1 - x0))
    box_height = max(1.0, float(y1 - y0))
    centre_x = (x0 + x1) * 0.5
    centre_y = (y0 + y1) * 0.5
    virtual_width = max(
        float(TARGET_DETAIL_MIN_SIDE_PX),
        box_width * (1.0 + 2.0 * TARGET_DETAIL_EXPAND_RATIO),
    )
    virtual_height = max(
        float(TARGET_DETAIL_MIN_SIDE_PX),
        box_height * (1.0 + 2.0 * TARGET_DETAIL_EXPAND_RATIO),
    )
    desired_x0 = int(round(centre_x - virtual_width * 0.5))
    desired_y0 = int(round(centre_y - virtual_height * 0.5))
    desired_x1 = int(round(centre_x + virtual_width * 0.5))
    desired_y1 = int(round(centre_y + virtual_height * 0.5))
    canvas_width = max(1, desired_x1 - desired_x0)
    canvas_height = max(1, desired_y1 - desired_y0)
    source_x0 = max(0, min(width, desired_x0))
    source_y0 = max(0, min(height, desired_y0))
    source_x1 = max(0, min(width, desired_x1))
    source_y1 = max(0, min(height, desired_y1))
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        raise RuntimeError("public target detail has no source-image intersection")
    paste_x = source_x0 - desired_x0
    paste_y = source_y0 - desired_y0
    return {
        "source_roi_abs_xyxy": [source_x0, source_y0, source_x1, source_y1],
        "virtual_canvas_size_px": [canvas_width, canvas_height],
        "source_paste_xyxy": [
            paste_x,
            paste_y,
            paste_x + source_x1 - source_x0,
            paste_y + source_y1 - source_y0,
        ],
    }


def public_support_candidates(
    row: dict[str, Any], values: list[int], width: int, height: int, label: str | None
) -> list[dict[str, Any]]:
    """Compile historical MAP acquisitions solely from the public contract."""

    support = validate_source_box(values, width, height)
    contract = row["evidence_contract"]
    source = str(contract.get("support_source") or "")
    extent = str(contract.get("support_extent") or "")
    activations = set(contract.get("operator_activations") or [])

    # QueryCoordZoom capability: the complete explicit bounded support is sent
    # at its original pixel dimensions.  The downstream ViewGeometry maps its
    # local canvas coordinates back to this exact source crop.
    if source == "PUBLIC_EXPLICIT_SUPPORT" and extent == "BOUNDED_SUPPORT":
        return [
            {
                "roi_abs_xyxy": support,
                "source_support_label": label,
                "roi_source": "PUBLIC_EXPLICIT_QUERY_NATIVE",
                "roi_view_kind": "QUERY_NATIVE_SUPPORT",
                "view_render_mode": "QUERY_NATIVE",
                "observer_generation_profile": "QUERY_NATIVE",
                "declared_support_abs_xyxy": support,
            }
        ]

    # Historical fine-detail capability: one public target, expanded by 25%
    # independently on each axis with a 128-pixel minimum, black edge padding,
    # and a forced 1024-square observation.
    if (
        source == "PUBLIC_TARGET_SUPPORT"
        and extent == "SINGLE_TARGET"
        and "TARGET_DETAIL" in activations
        and target_detail_enabled(row, support)
    ):
        layout = target_detail_layout(support, width, height)
        return [
            {
                "roi_abs_xyxy": layout["source_roi_abs_xyxy"],
                "source_support_label": label,
                "roi_source": "PUBLIC_TARGET_HISTORICAL_DETAIL",
                "roi_view_kind": "TARGET_DETAIL",
                "view_render_mode": "SCALE_NORMALIZED_DETAIL",
                "observer_generation_profile": "STANDARD",
                "declared_support_abs_xyxy": support,
                "target_detail_virtual_canvas_size_px": layout[
                    "virtual_canvas_size_px"
                ],
                "target_detail_source_paste_xyxy": layout[
                    "source_paste_xyxy"
                ],
            }
        ]

    # Ordinary explicit/target supports retain the original centred 1024 MAP
    # view; large support geometry is not replaced by a new ownership tiler.
    context = support_context_roi(support, width, height)
    return [
        {
            "roi_abs_xyxy": context,
            "source_support_label": label,
            "roi_source": "PUBLIC_SUPPORT_CENTER_NATIVE",
            "roi_view_kind": "PUBLIC_SUPPORT_NATIVE",
            "view_render_mode": "NATIVE_PAD",
            "observer_generation_profile": "STANDARD",
            "declared_support_abs_xyxy": support,
        }
    ]


def cover_box(
    values: list[int],
    width: int,
    height: int,
    *,
    max_rois: int | None = None,
) -> list[list[int]]:
    x0, y0, x1, y1 = box_aabb(values)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"invalid public source box: {values}")
    side_w, side_h = min(ROI_SIDE, width), min(ROI_SIDE, height)

    def starts(low: int, high: int, side: int, limit: int) -> list[int]:
        span = high - low
        if span <= side:
            return [max(0, min(limit - side, int(round((low + high - side) / 2.0))))]
        count = int(math.ceil(span / side))
        return [
            max(
                0,
                min(
                    limit - side,
                    int(round(low + index * (span - side) / (count - 1))),
                ),
            )
            for index in range(count)
        ]

    x_starts = starts(x0, x1, side_w, width)
    y_starts = starts(y0, y1, side_h, height)
    candidates = [
        [left, top, left + side_w, top + side_h]
        for top in y_starts
        for left in x_starts
    ]
    candidates = [
        roi for index, roi in enumerate(candidates) if roi not in candidates[:index]
    ]
    # When the fixed budget cannot cover the whole region, retain a
    # deterministic spatially distributed subset rather than a top-left prefix.
    ordered: list[list[int]] = []
    remaining = list(candidates)
    while remaining and (max_rois is None or len(ordered) < max_rois):
        if not ordered:
            target_x, target_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            chosen = min(
                remaining,
                key=lambda roi: (
                    ((roi[0] + roi[2]) / 2.0 - target_x) ** 2
                    + ((roi[1] + roi[3]) / 2.0 - target_y) ** 2,
                    roi[1],
                    roi[0],
                ),
            )
        else:
            chosen = max(
                remaining,
                key=lambda roi: (
                    min(
                        ((roi[0] + roi[2] - prior[0] - prior[2]) / 2.0) ** 2
                        + ((roi[1] + roi[3] - prior[1] - prior[3]) / 2.0) ** 2
                        for prior in ordered
                    ),
                    -roi[1],
                    -roi[0],
                ),
            )
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def grid_rois(width: int, height: int, count: int) -> list[list[int]]:
    if count < 1:
        return []
    # Reuse the same aspect-ratio-aware interval tiler as declared public
    # support.  A fixed 3x3 centre list can collapse to fewer than four unique
    # views on very wide or tall images; source geometry, not a task label,
    # determines this deterministic fallback lattice.
    return cover_box(
        [0, 0, width, height], width, height, max_rois=count
    )


def fallback_rois(
    width: int, height: int, target: int, *, pairwise: bool
) -> list[list[int]]:
    rois = grid_rois(width, height, target)
    if pairwise and len(rois) == 1:
        # One crop can contain both public supports on a small source, but two
        # typed observations preserve A/B provenance.
        return [rois[0], rois[0]]
    return rois


def fallback_target(row: dict[str, Any], width: int, height: int) -> int:
    required = minimum_views(row)
    if row["evidence_contract"]["support_topology"] == "PAIRWISE_SUPPORT":
        return required
    side_w, side_h = min(ROI_SIDE, width), min(ROI_SIDE, height)
    possible = int(math.ceil(width / side_w)) * int(math.ceil(height / side_h))
    return min(required, possible)


def construction_target(
    row: dict[str, Any],
    width: int,
    height: int,
    candidates: list[tuple[list[int], str | None, str]],
    *,
    public_boxes_present: bool,
) -> int:
    """Return the pre-cap acquisition target from support representation.

    A declared public region is already complete when its deterministic tiling
    is complete, even if that takes fewer than the generic four distributed
    model-proposed views.  Requiring four in that case invents redundant crops
    and contradicts the public geometry.  Exact duplicates share one view,
    while A/B-labelled pairwise provenance remains distinct.
    """

    unique = {(tuple(roi), label) for roi, label, _source in candidates}
    if public_boxes_present and not unique:
        raise ValueError("public support geometry produced no view")
    if unique:
        return len(unique)
    return fallback_target(row, width, height)


def box_iou(first: list[int], second: list[int]) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if intersection and union else 0.0


def minimum_views(row: dict[str, Any]) -> int:
    contract = row["evidence_contract"]
    if contract["support_topology"] == "PAIRWISE_SUPPORT":
        return 2
    if contract.get("support_extent") == "FULL_SCENE":
        return 4
    if (
        contract.get("answer_schema") == "GEOMETRY_SET"
        and contract.get("composition") == "AGGREGATION"
    ):
        return 4
    return 1


def fallback_candidate(
    roi: list[int], label: str | None = None
) -> dict[str, Any]:
    return {
        "roi_abs_xyxy": roi,
        "source_support_label": label,
        "roi_source": "DETERMINISTIC_PUBLIC_FALLBACK",
        "roi_view_kind": "FALLBACK_NATIVE",
        "view_render_mode": "NATIVE_PAD",
        "observer_generation_profile": "STANDARD",
    }


def build_rois(args: argparse.Namespace) -> int:
    output = []
    fallback = 0
    minimum_view_shortfalls: list[dict[str, Any]] = []
    view_kinds: Counter[str] = Counter()
    for row in read_jsonl(args.planner_complete):
        width, height = int(row["original_width_px"]), int(row["original_height_px"])
        pairwise = row["evidence_contract"]["support_topology"] == "PAIRWISE_SUPPORT"
        candidates: list[dict[str, Any]] = []
        boxes = [
            box_aabb([int(round(float(value))) for value in box])
            for box in row.get("seed_boxes_abs_from_question") or []
        ]
        if boxes:
            if pairwise and len(boxes) != 2:
                raise ValueError(
                    f"pairwise public support must expose exactly A/B: {row['sample_id']}"
                )
            for box_index, box in enumerate(boxes):
                label = ("A" if box_index == 0 else "B") if pairwise else None
                candidates.extend(
                    public_support_candidates(row, box, width, height, label)
                )
        else:
            points = row.get("stage1_points_full1000") or []
            for point_index, point in enumerate(points):
                label = ("A" if point_index == 0 else "B") if pairwise else None
                candidates.append(
                    {
                        "roi_abs_xyxy": point_roi(point, width, height),
                        "source_support_label": label,
                        "roi_source": "STRICT_MODEL_SUPPORT_POINT",
                        "roi_view_kind": "MODEL_SUPPORT_NATIVE",
                        "view_render_mode": "NATIVE_PAD",
                        "observer_generation_profile": "STANDARD",
                    }
                )
        model_proposed = not boxes
        if not candidates:
            fallback += 1
            target = fallback_target(row, width, height)
            fallback_views = fallback_rois(
                width, height, target, pairwise=pairwise
            )
            for index, roi in enumerate(fallback_views):
                label = ("A" if index == 0 else "B") if pairwise else None
                candidates.append(fallback_candidate(roi, label))
        kept: list[dict[str, Any]] = []
        for candidate in candidates:
            roi = candidate["roi_abs_xyxy"]
            label = candidate.get("source_support_label")
            source = candidate["roi_source"]
            if source != "STRICT_MODEL_SUPPORT_POINT":
                key = (
                    tuple(roi),
                    label,
                    candidate.get("roi_view_kind"),
                )
                if not any(
                    key
                    == (
                        tuple(prior["roi_abs_xyxy"]),
                        prior.get("source_support_label"),
                        prior.get("roi_view_kind"),
                    )
                    for prior in kept
                ):
                    kept.append(candidate)
                continue
            comparable = [
                prior
                for prior in kept
                if prior["roi_source"] == "STRICT_MODEL_SUPPORT_POINT"
                and (not pairwise or label is None or prior.get("source_support_label") == label)
            ]
            if all(
                box_iou(roi, prior["roi_abs_xyxy"]) <= float(args.nms_iou)
                for prior in comparable
            ):
                kept.append(candidate)

        # Model-proposed support retains the public-contract minimum. Explicit
        # geometry already defines its acquisition: one exact/query or target
        # crop per support, with two separately labelled views for a pair.
        required = minimum_views(row) if model_proposed else (2 if pairwise else 1)
        if model_proposed and len(kept) < required:
            fallback += 1
            existing = {tuple(item["roi_abs_xyxy"]) for item in kept}
            for roi in grid_rois(width, height, int(row["max_native_views"])):
                if len(kept) >= required:
                    break
                if tuple(roi) in existing:
                    continue
                if all(
                    box_iou(roi, prior["roi_abs_xyxy"]) <= float(args.nms_iou)
                    for prior in kept
                ):
                    kept.append(fallback_candidate(roi))
                    existing.add(tuple(roi))
        if len(kept) < required:
            minimum_view_shortfalls.append(
                {
                    "sample_id": row["sample_id"],
                    "required": required,
                    "available": len(kept),
                }
            )
        for index, candidate in enumerate(kept, start=1):
            roi = candidate["roi_abs_xyxy"]
            view_kinds[str(candidate["roi_view_kind"])] += 1
            output.append(
                {
                    **row,
                    **candidate,
                    "roi_id": f"{row['qa_slug']}_clean_roi{index:02d}",
                    "roi_index": index,
                    "roi_abs_xyxy": roi,
                    "roi_full1000_xyxy": source1000_box(roi, width, height),
                    "crop_full1000_xyxy": source1000_box(roi, width, height),
                    "roi_width_px": roi[2] - roi[0],
                    "roi_height_px": roi[3] - roi[1],
                }
            )
    output.sort(key=lambda row: -int(row["roi_width_px"]) * int(row["roi_height_px"]))
    write_jsonl(args.out, output)
    audit = {
            "schema_version": "map-roi-construction-audit-2",
            "samples": len({row["sample_id"] for row in output}),
            "rois": len(output),
            "fallback_samples": fallback,
            "nms_iou": float(args.nms_iou),
            "view_kind_counts": dict(sorted(view_kinds.items())),
            "planner_cardinality": "public_contract_minimum_to_public_maximum",
            "minimum_view_policy": (
                "PAIRWISE=2; MODEL_PROPOSED FULL_SCENE or "
                "GEOMETRY_SET+AGGREGATION=4; otherwise=1"
            ),
            "deterministic_grid_fallback_used": fallback > 0,
            "ownership_tiling_used": False,
            "query_native_is_exact_original_pixel_crop": True,
            "target_detail_is_historical_expand25_min128_resize1024": True,
            "model_coordinate_clamping_used": False,
            "task_identity_used": False,
            "minimum_view_shortfalls": minimum_view_shortfalls,
            "test_used": False,
            "passed": not minimum_view_shortfalls,
        }
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"ROI construction audit failed: {audit}")
    return 0


def intersect(first: list[float], second: list[float]) -> list[float] | None:
    box = [max(first[0], second[0]), max(first[1], second[1]), min(first[2], second[2]), min(first[3], second[3])]
    return box if box[2] > box[0] and box[3] > box[1] else None


def clip_polygon_to_rect(
    values: list[float], rect: list[float]
) -> list[list[float]]:
    """Clip an arbitrary public polygon to a source ROI without clamping vertices."""

    if len(values) < 6 or len(values) % 2:
        raise ValueError("public polygon requires at least three xy vertices")
    points = [
        [float(values[index]), float(values[index + 1])]
        for index in range(0, len(values), 2)
    ]
    x0, y0, x1, y1 = [float(value) for value in rect]

    def clip(
        polygon: list[list[float]],
        inside: Any,
        crossing: Any,
    ) -> list[list[float]]:
        if not polygon:
            return []
        output: list[list[float]] = []
        previous = polygon[-1]
        previous_inside = bool(inside(previous))
        for current in polygon:
            current_inside = bool(inside(current))
            if current_inside:
                if not previous_inside:
                    output.append(crossing(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(crossing(previous, current))
            previous, previous_inside = current, current_inside
        return output

    def at_x(first: list[float], second: list[float], boundary: float) -> list[float]:
        dx = second[0] - first[0]
        if abs(dx) < 1e-12:
            return [boundary, first[1]]
        ratio = (boundary - first[0]) / dx
        return [boundary, first[1] + ratio * (second[1] - first[1])]

    def at_y(first: list[float], second: list[float], boundary: float) -> list[float]:
        dy = second[1] - first[1]
        if abs(dy) < 1e-12:
            return [first[0], boundary]
        ratio = (boundary - first[1]) / dy
        return [first[0] + ratio * (second[0] - first[0]), boundary]

    points = clip(points, lambda p: p[0] >= x0, lambda a, b: at_x(a, b, x0))
    points = clip(points, lambda p: p[0] <= x1, lambda a, b: at_x(a, b, x1))
    points = clip(points, lambda p: p[1] >= y0, lambda a, b: at_y(a, b, y0))
    points = clip(points, lambda p: p[1] <= y1, lambda a, b: at_y(a, b, y1))
    deduplicated: list[list[float]] = []
    for point in points:
        if not deduplicated or any(
            abs(point[index] - deduplicated[-1][index]) > 1e-7
            for index in (0, 1)
        ):
            deduplicated.append(point)
    if len(deduplicated) > 1 and all(
        abs(deduplicated[0][index] - deduplicated[-1][index]) <= 1e-7
        for index in (0, 1)
    ):
        deduplicated.pop()
    return deduplicated if len(deduplicated) >= 3 else []


def localize_public_coordinates(row: dict[str, Any], view: ViewGeometry) -> tuple[str, list[dict[str, Any]]]:
    absolute = list(row.get("coord_lists_abs_from_question") or [])
    original_question = str(row["question_original"])
    numeric_entries: list[dict[str, Any]] = []
    absolute_index = 0
    for source_match in LIST_RE.finditer(original_question):
        pieces = [
            part.strip() for part in source_match.group(1).split(",") if part.strip()
        ]
        if len(pieces) < 2 or len(pieces) % 2:
            continue
        try:
            [float(piece) for piece in pieces]
        except ValueError:
            continue
        if absolute_index >= len(absolute):
            raise ValueError("public coordinate list binding underflow")
        numeric_entries.append(
            {
                "start": source_match.start(),
                "end": source_match.end(),
                "values": [float(value) for value in absolute[absolute_index]],
                "absolute_index": absolute_index,
            }
        )
        absolute_index += 1
    if absolute_index != len(absolute):
        raise ValueError(
            "public coordinate list binding overflow before localization: "
            f"used={absolute_index}, available={len(absolute)}"
        )

    # A public two-corner rectangle is one region.  Localize its intersection
    # with the view as a region, rather than independently declaring its two
    # original corners outside when the rectangle crosses this observation.
    pair_entry_indices: set[int] = set()
    pair_placeholders: dict[str, str] = {}

    def replace_two_corner_region(match: re.Match[str]) -> str:
        entries = [
            entry
            for entry in numeric_entries
            if int(entry["start"]) >= match.start()
            and int(entry["end"]) <= match.end()
        ]
        if len(entries) != 2 or any(len(entry["values"]) != 2 for entry in entries):
            raise ValueError("two-corner public rectangle binding is malformed")
        for entry in entries:
            pair_entry_indices.add(int(entry["absolute_index"]))
        first, second = entries[0]["values"], entries[1]["values"]
        box = [
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[0], second[0]),
            max(first[1], second[1]),
        ]
        visible = intersect(box, list(view.source_roi_xyxy_px.as_tuple()))
        if visible is None:
            rewritten = "rectangular area [outside this observation]"
        else:
            local_box = list(view.source_pixel_box_to_local1000(visible))
            rewritten = (
                "rectangular area in this observation's local 0-1000 system is "
                + render(local_box)
            )
        records.append(
            {
                "source_values": [*first, *second],
                "source_semantic": "TWO_CORNER_RECTANGLE",
                "source_box_abs": box,
                "visible_source_box_abs": visible,
                "rewritten": rewritten,
            }
        )
        token = f"__MAP_PUBLIC_RECTANGLE_{len(pair_placeholders)}__"
        pair_placeholders[token] = rewritten
        return token

    records = []
    question = TWO_CORNER_REGION_RE.sub(
        replace_two_corner_region, original_question
    )
    remaining_absolute = [
        values
        for entry_index, values in enumerate(absolute)
        if entry_index not in pair_entry_indices
    ]
    index = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal index
        pieces = [part.strip() for part in match.group(1).split(",") if part.strip()]
        if len(pieces) < 2 or len(pieces) % 2:
            return match.group(0)
        try:
            [float(piece) for piece in pieces]
        except ValueError:
            return match.group(0)
        if index >= len(remaining_absolute):
            raise ValueError("public coordinate list binding underflow")
        values = [float(value) for value in remaining_absolute[index]]
        index += 1
        rewritten: str
        try:
            if len(values) == 2:
                mapped = list(view.source_pixel_point_to_local1000(values))
                rewritten = render(mapped)
            elif len(values) == 4:
                box = values
                visible = intersect(box, list(view.source_roi_xyxy_px.as_tuple()))
                if visible is None:
                    rewritten = "[outside this observation]"
                else:
                    rewritten = render(list(view.source_pixel_box_to_local1000(visible)))
            else:
                clipped = clip_polygon_to_rect(
                    values, list(view.source_roi_xyxy_px.as_tuple())
                )
                if not clipped:
                    rewritten = "[outside this observation]"
                else:
                    mapped_values: list[float] = []
                    for point in clipped:
                        mapped_values.extend(
                            view.source_pixel_point_to_local1000(point)
                        )
                    rewritten = render(mapped_values)
        except ViewGeometryError:
            rewritten = "[outside this observation]"
        records.append({"source_values": values, "rewritten": rewritten})
        return rewritten

    question = LIST_RE.sub(replace, question)
    for token, rewritten in pair_placeholders.items():
        question = question.replace(token, rewritten)
    if index != len(remaining_absolute):
        raise ValueError(
            "public coordinate list binding overflow: "
            f"used={index}, available={len(remaining_absolute)}"
        )
    thresholds = list(row.get("axis_thresholds_abs_from_question") or [])
    threshold_index = 0

    def replace_threshold(match: re.Match[str]) -> str:
        nonlocal threshold_index
        if threshold_index >= len(thresholds):
            raise ValueError("public axis-threshold binding underflow")
        expected = thresholds[threshold_index]
        threshold_index += 1
        axis = match.group(1).lower()
        relation = match.group(2).lower()
        value = float(match.group(3))
        if (
            expected.get("axis") != axis
            or expected.get("relation") != relation
            or abs(float(expected.get("value_abs")) - value) > 1e-7
        ):
            raise ValueError("public axis-threshold semantic binding mismatch")
        roi = view.source_roi_xyxy_px
        low, high = (roi.x0, roi.x1) if axis == "x" else (roi.y0, roi.y1)
        less_equal = relation in {"left", "above"}
        all_visible = (less_equal and value >= high) or (
            not less_equal and value <= low
        )
        none_visible = (less_equal and value < low) or (
            not less_equal and value > high
        )
        if all_visible:
            rewritten = (
                f"center position automatically satisfies the public source-{axis} "
                "threshold everywhere in this observation"
            )
        elif none_visible:
            rewritten = (
                f"center position cannot satisfy the public source-{axis} threshold "
                "anywhere in this observation"
            )
        else:
            if axis == "x":
                point = [value, (roi.y0 + roi.y1) / 2.0]
                local_value = view.source_pixel_point_to_local1000(point)[0]
            else:
                point = [(roi.x0 + roi.x1) / 2.0, value]
                local_value = view.source_pixel_point_to_local1000(point)[1]
            rendered_value = render([local_value])[1:-1]
            relation_phrase = (
                f"{relation} of" if relation in {"left", "right"} else relation
            )
            rewritten = (
                f"center {axis}-coordinate in this observation's local 0-1000 "
                f"system is at or {relation_phrase} {rendered_value}"
            )
        records.append(
            {
                "axis_threshold_abs": expected,
                "rewritten": rewritten,
            }
        )
        return rewritten

    question = AXIS_THRESHOLD_RE.sub(replace_threshold, question)
    if threshold_index != len(thresholds):
        raise ValueError(
            "public axis-threshold binding overflow: "
            f"used={threshold_index}, available={len(thresholds)}"
        )
    # The shared local contract is the only output-format instruction visible
    # at this stage. Public choices and semantic question content are retained.
    return strip_answer_instruction(question), records


def local_contract(row: dict[str, Any]) -> str:
    schema = row["evidence_contract"]["answer_schema"]
    query_native = row.get("observer_generation_profile") == "QUERY_NATIVE"
    if schema == "SCALAR":
        return "Return exactly one non-negative integer and no explanation."
    if schema == "OPTION":
        return "Return exactly one public option letter if the view is sufficient; otherwise return exactly null."
    if schema == "SINGLE_GEOMETRY":
        coordinate_system = (
            "native-crop local 0-1000 geometry"
            if query_native
            else "observation-local 0-1000 geometry"
        )
        return (
            f"Return one explicit {coordinate_system}: either a "
            "tight AABB4 [x1,y1,x2,y2] or a tight convex QUAD8 in perimeter "
            "order; return exactly null when absent. Final public encoding is "
            "applied only after source registration."
        )
    if schema == "GEOMETRY_SET":
        coordinate_system = (
            "native-crop local 0-1000"
            if query_native
            else "observation-local 0-1000"
        )
        return (
            f"Return every visible matching target as a separate {coordinate_system} "
            "AABB4 or convex QUAD8; mixed explicit encodings are allowed. "
            "Return exactly null when none are visible. Final public encoding is "
            "applied only after source registration."
        )
    raise ValueError(schema)


def local_emphasis(row: dict[str, Any], view: ViewGeometry) -> str:
    contract = row["evidence_contract"]
    lines: list[str] = []
    kind = str(row.get("roi_view_kind") or "MODEL_SUPPORT_NATIVE")
    if kind == "QUERY_NATIVE_SUPPORT":
        lines.append(
            "This is the complete declared bounded support at its original pixel "
            "resolution, without resizing or padding. Its geometry coordinates use "
            "a local 0-1000 canvas over the complete crop."
        )
    elif kind == "TARGET_DETAIL":
        lines.append(
            "This is the public target expanded by 25 percent on each axis with "
            "a 128-pixel minimum, black edge padding where necessary, and a "
            "bilinear resize to 1024 by 1024. Apparent display size is not source "
            "extent; use the recorded view geometry for coordinates."
        )

    if contract["composition"] == "NUMERIC_DIFFERENCE":
        lines.append(
            f"This view belongs to public support {row.get('source_support_label')}. "
            "Count only that support; do not compute the final difference."
        )
    coverage = row.get("public_support_coverage") or {}
    if coverage.get("status") == "COMPLETE_WITHIN_BUDGET":
        lines.append(
            "Record only evidence assigned to this view; the selected view set jointly "
            "covers the declared public support at the required detail."
        )
    elif coverage.get("status") == "PARTIAL_FIXED_BUDGET":
        lines.append(
            "Record only evidence assigned to this fixed-budget observation. Some "
            "declared support lies outside the selected source-view footprints; "
            "unresolved detail in those portions must remain unknown."
        )
    else:
        lines.append(
            "Inspect only the question-conditioned visual evidence assigned to this "
            "observation; regions outside it remain unknown."
        )
    return "\n".join(lines)


LOCAL_TASK_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "task_id",
        "stage_name",
        "model_image_path",
        "model_input_prompt",
        "answer_serialization",
        "evidence_contract",
        "coordinate_protocol",
        "roi_id",
        "roi_index",
        "roi_abs_xyxy",
        "roi_full1000_xyxy",
        "roi_width_px",
        "roi_height_px",
        "roi_source",
        "source_support_label",
        "observer_generation_profile",
        "query_crop_origin_abs_px",
        "query_crop_size_px",
        "native_view_budget",
        "public_support_coverage",
        "view_geometry",
        "local_coordinate_protocol",
        "padding_semantics",
        "question_roi_local",
        "public_coordinate_rewrites",
        "roi_image_sha256",
        "roi_image_format",
        "source_image_sha256_verified",
        "render_spec_sha256",
    }
)


def local_task_payload(
    row: dict[str, Any],
    crop_dir: Path,
    prompt: str,
    question: str,
    rewrites: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project one ROI record onto the exact Stage-3 model-task boundary.

    Planner responses, request bindings, attempts, workers, latency, and parse
    state intentionally remain in internal provenance only.  They must never be
    copied into a subsequent model request object.
    """

    inherited = LOCAL_TASK_FIELDS - {
        "schema_version",
        "task_id",
        "stage_name",
        "model_image_path",
        "model_input_prompt",
        "question_roi_local",
        "public_coordinate_rewrites",
    }
    missing = sorted(key for key in inherited if key not in row)
    if missing:
        raise ValueError(f"ROI record lacks clean local-task fields: {missing}")
    task = {key: copy.deepcopy(row[key]) for key in inherited}
    task.update(
        {
            "schema_version": "map-local-observer-task-1",
            "task_id": f"{row['roi_id']}_clean_observe",
            "stage_name": "source_registered_native_observation",
            "model_image_path": str(
                (crop_dir / str(row["roi_id"]) / "roi.png").resolve()
            ),
            "model_input_prompt": prompt,
            "question_roi_local": question,
            "public_coordinate_rewrites": copy.deepcopy(rewrites),
        }
    )
    if set(task) != LOCAL_TASK_FIELDS:
        raise PermissionError("local model task field projection is not exact")
    return task


def query_native_question(question: str) -> str:
    """Remove exactly one public bounded-support clause for its exact crop.

    QUERY_NATIVE displays precisely the public search region, so repeating its
    source-image coordinates in the crop prompt is both unnecessary and
    coordinate-unsafe.  Only the two frozen public phrasings are accepted.
    Absence, ambiguity, or any residual numeric source list is rejected instead
    of silently sending a mixed coordinate protocol to the local observer.
    """

    candidates: list[tuple[re.Pattern[str], re.Match[str]]] = []
    for pattern in (
        BOUNDED_ONE_LIST_CLAUSE_RE,
        BOUNDED_TWO_CORNER_CLAUSE_RE,
    ):
        candidates.extend((pattern, match) for match in pattern.finditer(question))
    if len(candidates) != 1:
        raise ValueError(
            "QUERY_NATIVE requires exactly one recognized public bounded-support "
            f"clause, found {len(candidates)}"
        )
    pattern, expected_match = candidates[0]
    stripped, removed = pattern.subn("", question)
    if removed != 1 or expected_match.group(0) in stripped:
        raise ValueError(
            "QUERY_NATIVE public bounded-support clause was not removed exactly once"
        )
    stripped = re.sub(r"\s{2,}", " ", stripped).strip()
    if (
        BOUNDED_ONE_LIST_CLAUSE_RE.search(stripped)
        or BOUNDED_TWO_CORNER_CLAUSE_RE.search(stripped)
        or SOURCE_ABSOLUTE_LIST_RE.search(stripped)
    ):
        raise ValueError(
            "QUERY_NATIVE prompt retains a source-image absolute coordinate"
        )
    return stripped


def public_single_target_question(question: str) -> tuple[str, list[dict[str, str]]]:
    """Consume one public target locator that is already realized by the view."""

    matches = list(PUBLIC_TARGET_LOCATOR_RE.finditer(question))
    if len(matches) != 1:
        raise ValueError(
            "public single-target view requires exactly one target locator, "
            f"found {len(matches)}"
        )
    original = matches[0].group(0)
    rewritten, count = PUBLIC_TARGET_LOCATOR_RE.subn(
        "object shown in this image", question
    )
    if count != 1 or PUBLIC_TARGET_LOCATOR_RE.search(rewritten):
        raise ValueError("public target locator was not consumed exactly once")
    if SOURCE_ABSOLUTE_LIST_RE.search(rewritten):
        raise ValueError("public target prompt retains a source-image coordinate")
    return rewritten.strip(), [
        {
            "public_target_locator": original,
            "rewritten": "object shown in this image",
        }
    ]


def build_local(args: argparse.Namespace) -> int:
    tasks = []
    coordinate_rows = 0
    for source in read_jsonl(args.rois):
        metadata_path = args.crop_dir / str(source["roi_id"]) / "roi_meta.json"
        row = json.loads(metadata_path.read_text(encoding="utf-8"))
        view = ViewGeometry.from_metadata(dict(row["view_geometry"]))
        if row.get("observer_generation_profile") == "QUERY_NATIVE":
            crop_width, crop_height = [int(value) for value in row["query_crop_size_px"]]
            question = query_native_question(str(row["question_original"]))
            rewrites = []
            coordinate_preamble = (
                f"This is the exact public bounded-support crop at its original "
                f"pixel resolution: height {crop_height} pixels and width "
                f"{crop_width} pixels. Geometry in your response uses a normalized "
                "0-1000 canvas over this entire crop, with [0,0] at its top-left "
                "and [1000,1000] at its bottom-right; x increases rightward and y "
                "increases downward. "
                "The recorded source origin is used deterministically after the "
                "observation; do not output source-image coordinates."
            )
            prompt_style = "QUERY_NATIVE_COMPLETE_SUPPORT"
        elif (
            row["evidence_contract"].get("support_source")
            == "PUBLIC_TARGET_SUPPORT"
            and row["evidence_contract"].get("support_extent")
            == "SINGLE_TARGET"
            and row["evidence_contract"].get("answer_schema") == "OPTION"
        ):
            question, rewrites = public_single_target_question(
                str(row["question_original"])
            )
            coordinate_preamble = ""
            prompt_style = "PUBLIC_SINGLE_TARGET_OPTION"
        else:
            question, rewrites = localize_public_coordinates(row, view)
            coordinate_preamble = (
                "The visible source content occupies this observation-local 0-1000 "
                f"support: {render(list(view.source_pixel_box_to_local1000(list(view.source_roi_xyxy_px.as_tuple()))))}.\n"
                "Black padding, if present outside that support, is not source evidence. "
                "Any numeric public location in the question below has already been "
                "rewritten from source pixels into this observation's local 0-1000 "
                "coordinates. Do not reinterpret it as source-image coordinates."
            )
            prompt_style = "STANDARD_EVIDENCE"
        coordinate_rows += int(bool(rewrites))
        if prompt_style == "PUBLIC_SINGLE_TARGET_OPTION":
            prompt = f"""You are viewing a cropped remote-sensing image focused on one target object.

Answer the multiple-choice question using only the object shown in this image. Output only one valid option letter, with no explanation.

Question:
{question}"""
        elif prompt_style == "QUERY_NATIVE_COMPLETE_SUPPORT":
            prompt = f"""This is a cropped sub-region of a larger remote-sensing image. The saved crop has height {crop_height} pixels and width {crop_width} pixels, and it exactly covers the bounded support declared by the public question.

Whenever the answer involves coordinates or bounding boxes, use a normalized 0-1000 coordinate system over this entire crop: [0,0] is its top-left and [1000,1000] is its bottom-right. The runtime deterministically maps that local frame back to the source image.

Please answer the following task:

{question}

{local_contract(row)}"""
        else:
            prompt = f"""You are viewing one source-registered observation derived from a larger ultra-high-resolution remote-sensing image.

{coordinate_preamble}

Public evidence question:
{question}

Observation policy:
{local_emphasis(row, view)}
{local_contract(row)}
Do not infer objects outside the displayed source content and do not provide a worked answer example."""
        task = local_task_payload(row, args.crop_dir, prompt, question, rewrites)
        tasks.append(task)
    tasks.sort(key=lambda row: -int(row["roi_width_px"]) * int(row["roi_height_px"]))
    write_jsonl(args.out, tasks)
    write_json(
        args.audit,
        {
            "schema_version": "map-local-observer-prepare-audit-1",
            "tasks": len(tasks),
            "samples": len({row["sample_id"] for row in tasks}),
            "rows_with_public_coordinate_rewrite": coordinate_rows,
            "canvas_aware_local_coordinates": True,
            "source_coordinate_text_left_in_local_prompt": False,
            "exact_model_task_field_set": sorted(LOCAL_TASK_FIELDS),
            "planner_result_or_attempt_fields_forwarded": False,
            "upstream_model_response_forwarded": False,
            "task_identity_used": False,
            "test_used": False,
            "passed": (
                len({row["task_id"] for row in tasks}) == len(tasks)
                and all(set(row) == LOCAL_TASK_FIELDS for row in tasks)
            ),
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    item = commands.add_parser("prepare")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--out-dir", type=Path, required=True)
    item.set_defaults(func=prepare)
    item = commands.add_parser("complete-planner")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--results", type=Path, required=True)
    item.add_argument("--skipped", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=complete_planner)
    item = commands.add_parser("build-rois")
    item.add_argument("--planner-complete", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.add_argument("--nms-iou", type=float, default=0.2)
    item.set_defaults(func=build_rois)
    item = commands.add_parser("build-local")
    item.add_argument("--rois", type=Path, required=True)
    item.add_argument("--crop-dir", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=build_local)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
