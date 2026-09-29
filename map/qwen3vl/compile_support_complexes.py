#!/usr/bin/env python3
"""Compile dynamic source-support complexes without dataset-count assumptions."""

from __future__ import annotations

import argparse
import json
import sys
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


"""Compile public source-registered observations into actionable support complexes."""



FORBIDDEN_KEYS = {
    "annotation",
    "answer",
    "benchmark_task_key",
    "family",
    "ground_truth",
    "metric",
    "qa_path",
    "reference_answer",
    "score",
    "split",
    "task_key",
}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["sample_id"])].append(row)
    return output


def keyed(rows: Iterable[dict[str, Any]], key: str = "sample_id") -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        ident = str(row.get(key) or "")
        if not ident or ident in output:
            raise ValueError(f"missing or duplicate {key}: {ident}")
        output[ident] = row
    return output


def finite_box(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"invalid source box: {value!r}")
    box = [float(item) for item in value]
    if not all(math.isfinite(item) and 0 <= item <= 1000 for item in box):
        raise ValueError(f"non-finite/out-of-range source box: {value!r}")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"non-positive source box: {value!r}")
    return [round(item, 4) for item in box]


def base1000_to_abs(box: list[float], width: int, height: int) -> list[int]:
    x1 = max(0, min(width - 1, int(round(box[0] * width / 1000.0))))
    y1 = max(0, min(height - 1, int(round(box[1] * height / 1000.0))))
    x2 = max(x1 + 1, min(width, int(round(box[2] * width / 1000.0))))
    y2 = max(y1 + 1, min(height, int(round(box[3] * height / 1000.0))))
    return [x1, y1, x2, y2]


def abs_to_base1000(box: list[int], width: int, height: int) -> list[float]:
    return [
        round(box[0] * 1000.0 / width, 4),
        round(box[1] * 1000.0 / height, 4),
        round(box[2] * 1000.0 / width, 4),
        round(box[3] * 1000.0 / height, 4),
    ]


def centered_native_box(center: list[float], width: int, height: int, side: int = 640) -> list[int]:
    crop_w = min(side, width)
    crop_h = min(side, height)
    cx = center[0] * width / 1000.0
    cy = center[1] * height / 1000.0
    x1 = min(max(int(round(cx - crop_w / 2.0)), 0), width - crop_w)
    y1 = min(max(int(round(cy - crop_h / 2.0)), 0), height - crop_h)
    return [x1, y1, x1 + crop_w, y1 + crop_h]


def point_in_box(point: tuple[float, float], box: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def largest_uncovered_center(boxes: list[list[float]], grid: int = 31) -> list[float]:
    """Return a deterministic source-grid center maximizing distance from receipts."""
    best: tuple[float, float, float] | None = None
    for iy in range(grid):
        y = (iy + 0.5) * 1000.0 / grid
        for ix in range(grid):
            x = (ix + 0.5) * 1000.0 / grid
            if any(point_in_box((x, y), box) for box in boxes):
                continue
            if boxes:
                distance = min(
                    max(box[0] - x, 0.0, x - box[2]) ** 2
                    + max(box[1] - y, 0.0, y - box[3]) ** 2
                    for box in boxes
                )
            else:
                distance = 0.0
            candidate = (distance, -iy * grid - ix, x, y)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return [500.0, 500.0]
    return [round(best[2], 4), round(best[3], 4)]


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(nested_keys(item))
    return keys


def make_node(row: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(row["model_image_path"])).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    expected_hash = str(row.get("roi_image_sha256") or "")
    actual_hash = sha256_file(image_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(f"registered image hash mismatch: {image_path}")
    box = finite_box(row["roi_full1000_xyxy"])
    return {
        "node_id": f"R{int(row.get('roi_index') or 0)}",
        "source_box_1000": box,
        "native_image_path": str(image_path),
        "native_image_sha256": actual_hash,
        "visible_observation_prefix_500": str(
            row.get("stage3_mapped_response_full1000") or row.get("model_response") or ""
        ).strip()[:500],
    }


def compile_action(source: dict[str, Any], nodes: list[dict[str, Any]], overlay: dict[str, Any]) -> dict[str, Any]:
    state = source["evidence_state"]
    topology = str(state["evidence_topology"])
    composition = str(state["evidence_composition"])
    schema = str(state["answer_schema"])
    extent = str(state["support_extent"])
    width, height = int(source["original_width_px"]), int(source["original_height_px"])
    boxes = [node["source_box_1000"] for node in nodes]

    common = {
        "sample_id": source["sample_id"],
        "qa_slug": source["qa_slug"],
        "question_original": source["question_original"],
        "question_1000": source["question_1000"],
        "public_evidence_state": state,
        "source_coordinate_system": "source_image_base1000_xyxy",
        "requires_new_vlm_observation": True,
        "changes_historical_controller": False,
        "changes_existing_prediction": False,
        "source_image_path": str(Path(source["image_abs_path"]).resolve()),
        "source_image_sha256": str(source["image_sha256"]),
    }
    if topology == "SINGLE_SUPPORT":
        center = [
            (boxes[0][0] + boxes[0][2]) / 2.0,
            (boxes[0][1] + boxes[0][3]) / 2.0,
        ]
        crop_abs = centered_native_box(center, width, height)
        common.update({
            "action_kind": "NESTED_NATIVE_REINSPECTION",
            "support_node_ids": [nodes[0]["node_id"]],
            "source_crop_abs_xyxy": crop_abs,
            "source_crop_1000_xyxy": abs_to_base1000(crop_abs, width, height),
            "context_inputs": [nodes[0]["native_image_path"]],
            "context_input_sha256": [nodes[0]["native_image_sha256"]],
        })
        return common

    if topology == "PAIRWISE_SUPPORT":
        common.update({
            "action_kind": "NONLOCAL_JOINT_ENDPOINT_OBSERVATION",
            "support_node_ids": [node["node_id"] for node in nodes],
            "endpoint_source_boxes_1000": boxes,
            "observation_inputs": [overlay["overlay_image_path"]]
            + [node["native_image_path"] for node in nodes[:4]],
            "observation_input_sha256": [overlay["overlay_image_sha256"]]
            + [node["native_image_sha256"] for node in nodes[:4]],
            "observation_input_roles": ["GLOBAL_SOURCE_LAYOUT"]
            + [f"REGISTERED_ENDPOINT_{index + 1}" for index in range(min(4, len(nodes)))],
        })
        return common

    if topology == "SET" and schema == "SCALAR" and extent == "FULL_SCENE":
        center = largest_uncovered_center(boxes)
        crop_abs = centered_native_box(center, width, height)
        common.update({
            "action_kind": "UNOBSERVED_SCENE_RESIDUAL_OBSERVATION",
            "support_node_ids": [node["node_id"] for node in nodes],
            "residual_center_1000": center,
            "source_crop_abs_xyxy": crop_abs,
            "source_crop_1000_xyxy": abs_to_base1000(crop_abs, width, height),
            "context_inputs": [overlay["overlay_image_path"]],
            "context_input_sha256": [overlay["overlay_image_sha256"]],
        })
        return common

    raise ValueError(
        f"no excluded-state action for topology={topology}, composition={composition}, "
        f"schema={schema}, extent={extent}"
    )



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-manifest", type=Path, required=True)
    parser.add_argument("--mapped", type=Path, required=True)
    parser.add_argument("--overlays", type=Path, required=True)
    parser.add_argument("--controller-admitted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    open_rows = read_jsonl(args.open_manifest)
    mapped = grouped(read_jsonl(args.mapped))
    overlays = keyed(read_jsonl(args.overlays))
    admitted_ids = {str(row["sample_id"]) for row in read_jsonl(args.controller_admitted)}
    open_ids = {str(row["sample_id"]) for row in open_rows}
    if not admitted_ids <= open_ids:
        raise ValueError("controller-admitted states are not a subset of OPEN states")

    complexes: list[dict] = []
    actions: list[dict] = []
    topology_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    forbidden_hits: Counter[str] = Counter()
    for source in open_rows:
        sample_id = str(source["sample_id"])
        forbidden_hits.update(nested_keys(source) & FORBIDDEN_KEYS)
        nodes = [
            make_node(row)
            for row in sorted(mapped[sample_id], key=lambda item: int(item.get("roi_index") or 0))
        ]
        if not nodes:
            raise ValueError(f"OPEN state has no registered observations: {sample_id}")
        state = source["evidence_state"]
        topology = str(state["evidence_topology"])
        if topology == "PAIRWISE_SUPPORT" and len(nodes) < 2:
            raise ValueError(f"pairwise state has fewer than two supports: {sample_id}")
        topology_counts[topology] += 1
        complexes.append({
            "schema_version": "map-source-support-complex-1",
            "sample_id": sample_id,
            "qa_slug": source["qa_slug"],
            "public_evidence_state": state,
            "nodes": nodes,
            "support_hyperedge": {
                "topology": topology,
                "node_ids": [node["node_id"] for node in nodes],
                "source_coordinate_system": "source_image_base1000_xyxy",
            },
            "support_controller_admitted": sample_id in admitted_ids,
        })
        if sample_id not in admitted_ids:
            action = compile_action(source, nodes, overlays[sample_id])
            action.pop("changes_historical_controller", None)
            action["changes_support_controller"] = False
            actions.append(action)
            action_counts[str(action["action_kind"])] += 1

    output_forbidden = nested_keys(complexes) & FORBIDDEN_KEYS
    output_forbidden.update(nested_keys(actions) & FORBIDDEN_KEYS)
    expected_actions = len(open_rows) - len(admitted_ids)
    audit = {
        "schema_version": "map-source-support-compile-audit-1",
        "open_rows": len(open_rows),
        "controller_admitted_rows": len(admitted_ids),
        "controller_excluded_rows": expected_actions,
        "compiled_complex_rows": len(complexes),
        "compiled_action_rows": len(actions),
        "topology_counts": dict(sorted(topology_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "all_excluded_states_actionable": len(actions) == expected_actions,
        "all_actions_require_observation": all(row["requires_new_vlm_observation"] for row in actions),
        "input_forbidden_key_hits": dict(sorted(forbidden_hits.items())),
        "output_forbidden_keys": sorted(output_forbidden),
        "dataset_count_used_as_routing_rule": False,
        "ground_truth_annotation_task_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = all([
        len(open_rows) == len(open_ids),
        audit["all_excluded_states_actionable"],
        audit["all_actions_require_observation"],
        not forbidden_hits,
        not output_forbidden,
    ])
    write_jsonl(args.output_dir / "support_complexes.jsonl", complexes)
    write_jsonl(args.output_dir / "excluded_state_actions.jsonl", actions)
    write_json(args.output_dir / "SUPPORT_COMPLEX_COMPILE_AUDIT.json", audit)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

