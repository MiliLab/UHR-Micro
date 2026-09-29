#!/usr/bin/env python3
"""Apply the public-contract view budget after deterministic ROI construction."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from io_utils import keyed, read_jsonl, write_json, write_jsonl


def box_aabb(values: list[Any]) -> list[float]:
    """Return the public support's axis-aligned source-pixel envelope."""

    coordinates = [float(value) for value in values]
    if len(coordinates) == 4:
        box = coordinates
    elif len(coordinates) >= 8 and len(coordinates) % 2 == 0:
        box = [
            min(coordinates[0::2]),
            min(coordinates[1::2]),
            max(coordinates[0::2]),
            max(coordinates[1::2]),
        ]
    else:
        raise ValueError(f"unsupported public support geometry: {values}")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"degenerate public support geometry: {values}")
    return box


def intersection(first: list[float], second: list[float]) -> list[float] | None:
    box = [
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    ]
    return box if box[2] > box[0] and box[3] > box[1] else None


def rectangle_union_area(rectangles: list[list[float]]) -> float:
    """Exact union area for axis-aligned rectangles using an x sweep."""

    valid = [
        [float(value) for value in rectangle]
        for rectangle in rectangles
        if rectangle[2] > rectangle[0] and rectangle[3] > rectangle[1]
    ]
    if not valid:
        return 0.0
    xs = sorted({rectangle[0] for rectangle in valid} | {rectangle[2] for rectangle in valid})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (rectangle[1], rectangle[3])
            for rectangle in valid
            if rectangle[0] < right and rectangle[2] > left
        )
        covered_y = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered_y += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered_y += end - start
        area += (right - left) * covered_y
    return area


def native_detail_coverage_box(row: dict[str, Any]) -> list[float] | None:
    """Return the actual source footprint visible in one selected view."""

    values = row.get("roi_abs_xyxy")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("selected view lacks source ROI")
    return [float(value) for value in values]


def context_coverage_box(row: dict[str, Any]) -> list[float] | None:
    if row.get("roi_view_kind") != "QUERY_NATIVE_SUPPORT":
        return None
    values = row.get("roi_abs_xyxy")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("support-context view lacks source ROI")
    return [float(value) for value in values]


def public_support_coverage(
    source: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    budget: int,
    uncapped_views: int,
) -> dict[str, Any]:
    """Measure what the selected fixed-budget views actually cover.

    This is public-input geometry only.  It is descriptive state for honest
    downstream reasoning, not a task route and not an admission requirement
    that silently changes the common eight-view budget.
    """

    raw_supports = list(source.get("seed_boxes_abs_from_question") or [])
    if not raw_supports:
        return {
            "status": "NOT_MEASURABLE_MODEL_PROPOSED",
            "public_support_box_count": 0,
            "complete_support_box_count": 0,
            "incomplete_support_box_count": 0,
            "mean_coverage_ratio": None,
            "minimum_coverage_ratio": None,
            "complete_support_context_box_count": 0,
            "fixed_view_budget": budget,
            "selected_views": len(selected),
            "uncapped_views": uncapped_views,
            "support_records": [],
            "support_coordinate_system": "source_image_base1000_xyxy",
        }

    width = float(source["original_width_px"])
    height = float(source["original_height_px"])
    if width <= 0 or height <= 0:
        raise ValueError("invalid source dimensions for public support coverage")
    pairwise = source["evidence_contract"]["support_topology"] == "PAIRWISE_SUPPORT"
    records: list[dict[str, Any]] = []
    for support_index, raw_support in enumerate(raw_supports):
        support = box_aabb(raw_support)
        label = ("A" if support_index == 0 else "B") if pairwise else None
        eligible = [
            row
            for row in selected
            if not pairwise or row.get("source_support_label") == label
        ]
        native_intersections = [
            clipped
            for row in eligible
            if (coverage_box := native_detail_coverage_box(row)) is not None
            if (clipped := intersection(support, coverage_box))
            is not None
        ]
        context_intersections = [
            clipped
            for row in eligible
            if (coverage_box := context_coverage_box(row)) is not None
            if (clipped := intersection(support, coverage_box)) is not None
        ]
        support_area = (support[2] - support[0]) * (support[3] - support[1])
        ratio = min(1.0, rectangle_union_area(native_intersections) / support_area)
        context_ratio = min(
            1.0, rectangle_union_area(context_intersections) / support_area
        )
        records.append(
            {
                "support_index": support_index + 1,
                "support_label": label,
                "support_source_base1000_xyxy": [
                    round(support[0] / width * 1000.0, 10),
                    round(support[1] / height * 1000.0, 10),
                    round(support[2] / width * 1000.0, 10),
                    round(support[3] / height * 1000.0, 10),
                ],
                "coverage_ratio": round(ratio, 10),
                "native_detail_coverage_ratio": round(ratio, 10),
                "support_context_coverage_ratio": round(context_ratio, 10),
                "complete_support_context_available": context_ratio >= 1.0 - 1e-9,
                "complete": ratio >= 1.0 - 1e-9,
            }
        )

    complete = sum(bool(record["complete"]) for record in records)
    ratios = [float(record["coverage_ratio"]) for record in records]
    return {
        "status": (
            "COMPLETE_WITHIN_BUDGET"
            if complete == len(records)
            else "PARTIAL_FIXED_BUDGET"
        ),
        "public_support_box_count": len(records),
        "complete_support_box_count": complete,
        "incomplete_support_box_count": len(records) - complete,
        "mean_coverage_ratio": round(sum(ratios) / len(ratios), 10),
        "minimum_coverage_ratio": min(ratios),
        "complete_support_context_box_count": sum(
            bool(record["complete_support_context_available"]) for record in records
        ),
        "fixed_view_budget": budget,
        "selected_views": len(selected),
        "uncapped_views": uncapped_views,
        "support_records": records,
        "support_coordinate_system": "source_image_base1000_xyxy",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    manifest = keyed(read_jsonl(args.manifest))
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input):
        grouped[str(row["sample_id"])].append(row)
    if set(grouped) != set(manifest):
        raise ValueError("ROI construction did not cover the fixed public manifest")
    output: list[dict[str, Any]] = []
    dropped = 0
    per_sample: Counter[int] = Counter()
    coverage_statuses: Counter[str] = Counter()
    public_support_ratios: list[float] = []
    public_support_boxes = 0
    complete_support_boxes = 0
    incomplete_support_boxes = 0
    complete_support_context_boxes = 0
    pairwise_samples = 0
    pairwise_samples_complete = 0
    for sid, source in manifest.items():
        rows = sorted(grouped[sid], key=lambda row: int(row.get("roi_index") or 0))
        total_budget = int(source["max_native_views"])
        budget = total_budget
        if budget < 1:
            raise ValueError(
                f"public observation budget leaves no main native view for {sid}: "
                f"total={total_budget}"
            )
        if source["evidence_contract"]["support_topology"] == "PAIRWISE_SUPPORT":
            by_label = {
                label: [row for row in rows if row.get("source_support_label") == label]
                for label in ("A", "B")
            }
            if not all(by_label.values()):
                raise ValueError(f"pairwise public contract lacks A/B support: {sid}")
            selected = []
            index = 0
            while len(selected) < budget:
                added = False
                for label in ("A", "B"):
                    if index < len(by_label[label]) and len(selected) < budget:
                        selected.append(by_label[label][index])
                        added = True
                if not added:
                    break
                index += 1
            if not {row.get("source_support_label") for row in selected} >= {"A", "B"}:
                raise ValueError(f"pairwise budget removed one public support: {sid}")
        else:
            selected = rows[:budget]
        coverage = public_support_coverage(
            source,
            selected,
            budget=budget,
            uncapped_views=len(rows),
        )
        coverage_statuses[str(coverage["status"])] += 1
        public_support_boxes += int(coverage["public_support_box_count"])
        complete_support_boxes += int(coverage["complete_support_box_count"])
        incomplete_support_boxes += int(coverage["incomplete_support_box_count"])
        complete_support_context_boxes += int(
            coverage["complete_support_context_box_count"]
        )
        public_support_ratios.extend(
            float(record["coverage_ratio"])
            for record in coverage["support_records"]
        )
        if source["evidence_contract"]["support_topology"] == "PAIRWISE_SUPPORT":
            pairwise_samples += 1
            pairwise_samples_complete += int(
                coverage["status"] == "COMPLETE_WITHIN_BUDGET"
            )
        dropped += len(rows) - len(selected)
        per_sample[len(selected)] += 1
        for index, row in enumerate(selected, start=1):
            item = dict(row)
            item["roi_index"] = index
            item["native_view_budget"] = budget
            item["public_support_coverage"] = coverage
            output.append(item)
    write_jsonl(args.out, output)
    audit = {
        "schema_version": "map-public-view-budget-audit-1",
        "samples": len(grouped),
        "views": len(output),
        "dropped_over_budget": dropped,
        "maximum_views_observed": max(per_sample) if per_sample else 0,
        "single_native_observation_budget": True,
        "view_count_distribution": {
            str(key): value for key, value in sorted(per_sample.items())
        },
        "public_support_coverage": {
            "status_distribution": dict(sorted(coverage_statuses.items())),
            "support_boxes": public_support_boxes,
            "complete_support_boxes": complete_support_boxes,
            "incomplete_support_boxes": incomplete_support_boxes,
            "complete_support_context_boxes": complete_support_context_boxes,
            "mean_coverage_ratio": (
                round(sum(public_support_ratios) / len(public_support_ratios), 10)
                if public_support_ratios
                else None
            ),
            "minimum_coverage_ratio": (
                min(public_support_ratios) if public_support_ratios else None
            ),
            "pairwise_samples": pairwise_samples,
            "pairwise_samples_complete": pairwise_samples_complete,
            "partial_coverage_is_explicit_not_silently_complete": True,
            "coverage_ratio_measures_selected_source_view_footprints": True,
            "ownership_tiling_used": False,
            "query_native_context_is_exact_public_support": True,
        },
        "task_identity_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        audit["maximum_views_observed"] <= 8
        and public_support_boxes == complete_support_boxes + incomplete_support_boxes
        and sum(coverage_statuses.values()) == len(grouped)
    )
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"view budget failed: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
