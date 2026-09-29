#!/usr/bin/env python3
"""Apply source-registered geometry union with one frozen NMS rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox(geometry: list[float]) -> tuple[float, float, float, float]:
    if len(geometry) == 4:
        x1, y1, x2, y2 = geometry
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if len(geometry) == 8:
        xs = geometry[0::2]
        ys = geometry[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    raise ValueError(f"unsupported geometry length: {len(geometry)}")


def iou(left: list[float], right: list[float]) -> float:
    ax1, ay1, ax2, ay2 = bbox(left)
    bx1, by1, bx2, by2 = bbox(right)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def valid_geometry(geometry: list[Any], expected_length: int) -> bool:
    if len(geometry) != expected_length:
        return False
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in geometry):
        return False
    values = [float(value) for value in geometry]
    if any(value < 0 or value > 1000 for value in values):
        return False
    x1, y1, x2, y2 = bbox(values)
    return x2 > x1 and y2 > y1


def deduplicate(candidates: list[dict[str, Any]], threshold: float) -> tuple[list[list[float]], int]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            0 if row["source_kind"] == "RESIDUAL_NATIVE640" else 1,
            bbox(row["geometry"])[1],
            bbox(row["geometry"])[0],
            json.dumps(row["geometry"], separators=(",", ":")),
        ),
    )
    kept: list[dict[str, Any]] = []
    suppressed = 0
    for candidate in ordered:
        if any(iou(candidate["geometry"], prior["geometry"]) >= threshold for prior in kept):
            suppressed += 1
            continue
        kept.append(candidate)
    kept.sort(key=lambda row: (bbox(row["geometry"])[1], bbox(row["geometry"])[0]))
    return [row["geometry"] for row in kept], suppressed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier-baseline", type=Path, required=True)
    parser.add_argument("--terminal-base", type=Path, required=True)
    parser.add_argument("--terminal-tasks", type=Path, required=True)
    parser.add_argument("--mapped-observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.iou_threshold != 0.5:
        raise PermissionError("the preregistered geometry-algebra threshold is exactly 0.5")

    carrier = read_jsonl(args.carrier_baseline)
    base = {str(row["sample_id"]): row for row in read_jsonl(args.terminal_base)}
    admitted_ids = {str(row["sample_id"]) for row in read_jsonl(args.terminal_tasks)}
    mapped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.mapped_observations):
        mapped[str(row["sample_id"])].append(row)
    geometry_ids = {
        sample_id for sample_id in admitted_ids
        if base[sample_id]["evidence_contract"]["answer_schema"] == "GEOMETRY_SET"
    }

    outputs: list[dict[str, Any]] = []
    activated = 0
    fallbacks = 0
    input_candidates = 0
    output_candidates = 0
    suppressed_total = 0
    by_encoding: defaultdict[str, int] = defaultdict(int)
    for original in carrier:
        sample_id = str(original["diagnostics"]["sample_id"])
        if sample_id not in geometry_ids:
            outputs.append(original)
            continue
        runtime = base[sample_id]
        target_encoding = str(runtime["evidence_contract"]["terminal_geometry_encoding"])
        expected_length = 4 if target_encoding == "AABB4" else 8 if target_encoding == "QUAD8" else 0
        if not expected_length:
            raise ValueError(f"unsupported terminal geometry encoding: {target_encoding}")
        candidates: list[dict[str, Any]] = []
        for candidate in runtime["typed_evidence_state"]["typed_aggregate"]["output_ready_geometry_candidates"]:
            geometry = [float(value) for value in candidate["output_geometry"]]
            if valid_geometry(geometry, expected_length):
                candidates.append({"geometry": geometry, "source_kind": "PRIOR_REGISTERED"})
        for observation in mapped[sample_id]:
            for candidate in observation.get("stage3_registered_geometries_full1000") or []:
                geometry = [float(value) for value in candidate["output_geometry_source1000"]]
                if valid_geometry(geometry, expected_length):
                    candidates.append({"geometry": geometry, "source_kind": "RESIDUAL_NATIVE640"})
        if not candidates:
            outputs.append(original)
            fallbacks += 1
            continue
        geometries, suppressed = deduplicate(candidates, args.iou_threshold)
        response = "Final answer: " + json.dumps(geometries, ensure_ascii=False, separators=(",", ":"))
        row = dict(original)
        row["raw_response"] = response
        row["final_response"] = response
        row["prompt_id"] = "nonlocal-support-closure-v1:source-registered-geometry-algebra"
        row["prompt_sha256"] = hashlib.sha256(b"deterministic-source-geometry-union-nms-iou-0.5").hexdigest()
        row["logical_vlm_calls"] = max(0, int(row.get("logical_vlm_calls") or 0) - 1)
        row["trajectory"] = [
            event for event in list(row.get("trajectory") or [])
            if event.get("stage") != "support_residual_typed_update"
        ] + [{
            "stage": "source_registered_geometry_algebra",
            "operator": "RESIDUAL_FIRST_UNION_NMS",
            "iou_threshold": args.iou_threshold,
            "input_candidates": len(candidates),
            "output_candidates": len(geometries),
            "suppressed_candidates": suppressed,
            "logical_vlm_calls": 0,
        }]
        diagnostics = dict(row.get("diagnostics") or {})
        accounting = dict(diagnostics.get("call_accounting") or {})
        logical = dict(accounting.get("logical_calls_by_stage") or {})
        attempts = dict(accounting.get("attempt_numbers_by_stage") or {})
        logical.pop("support_residual_terminal", None)
        attempts.pop("support_residual_terminal", None)
        if accounting:
            accounting["logical_calls_by_stage"] = logical
            accounting["attempt_numbers_by_stage"] = attempts
            accounting["logical_vlm_calls"] = row["logical_vlm_calls"]
            accounting["known_physical_vlm_attempts"] = max(0, int(accounting.get("known_physical_vlm_attempts") or 0) - 1)
            accounting["physical_vlm_attempts"] = max(0, int(accounting.get("physical_vlm_attempts") or 0) - 1)
            diagnostics["call_accounting"] = accounting
        diagnostics["source_registered_geometry_algebra"] = {
            "activated": True,
            "target_encoding": target_encoding,
            "operator": "RESIDUAL_FIRST_UNION_NMS",
            "iou_threshold": args.iou_threshold,
            "input_candidates": len(candidates),
            "output_candidates": len(geometries),
            "suppressed_candidates": suppressed,
            "ground_truth_or_metric_used": False,
            "test_used": False,
        }
        row["diagnostics"] = diagnostics
        outputs.append(row)
        activated += 1
        input_candidates += len(candidates)
        output_candidates += len(geometries)
        suppressed_total += suppressed
        by_encoding[target_encoding] += 1

    write_jsonl(args.out, outputs)
    audit = {
        "schema_version": "source-registered-geometry-algebra-canonical-audit-1",
        "carrier_rows": len(carrier),
        "canonical_rows": len(outputs),
        "eligible_geometry_states": len(geometry_ids),
        "activated_rows": activated,
        "fallback_rows": fallbacks,
        "activation_by_terminal_encoding": dict(sorted(by_encoding.items())),
        "input_candidates": input_candidates,
        "output_candidates": output_candidates,
        "suppressed_candidates": suppressed_total,
        "iou_threshold": args.iou_threshold,
        "logical_calls_saved": activated,
        "unique_qa_relpaths": len({row["qa_relpath"] for row in outputs}),
        "ground_truth_annotation_task_metric_used": False,
        "gpu_used": False,
        "test_used": False,
        "canonical_sha256": sha256_file(args.out),
    }
    audit["passed"] = all([
        len(carrier) == len(outputs),
        activated + fallbacks == len(geometry_ids),
        audit["unique_qa_relpaths"] == len(outputs),
    ])
    write_json(args.audit, audit)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
