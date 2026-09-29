#!/usr/bin/env python3
"""Build observer tasks whose prompts use the recorded display coordinate frame."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve()
TOOLS = HERE.parents[1] / "runtime" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import strong_path_runtime as strong  # noqa: E402
from io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from variant_contract import selected_variant  # noqa: E402


def displayed_local_box(
    box: list[int], roi: list[int], *, padded: bool
) -> list[float]:
    width = 1024 if padded else max(1, roi[2] - roi[0])
    height = 1024 if padded else max(1, roi[3] - roi[1])
    return [
        round(max(0.0, min(1000.0, (box[0] - roi[0]) / width * 1000)), 1),
        round(max(0.0, min(1000.0, (box[1] - roi[1]) / height * 1000)), 1),
        round(max(0.0, min(1000.0, (box[2] - roi[0]) / width * 1000)), 1),
        round(max(0.0, min(1000.0, (box[3] - roi[1]) / height * 1000)), 1),
    ]


def geometry_note(row: dict[str, Any], padded: bool) -> str:
    width = int(row["roi_width_px"])
    height = int(row["roi_height_px"])
    if not padded:
        return (
            f"Observation geometry: the complete model image is the exact native "
            f"{width}x{height} source crop with no resize and no padding. "
            "Observation-local [0,0] and [1000,1000] are its top-left and bottom-right."
        )
    x1 = round(width / 1024 * 1000, 1)
    y1 = round(height / 1024 * 1000, 1)
    return (
        "Observation geometry: the complete model image is a 1024x1024 canvas. "
        f"Real source pixels occupy observation-local [0,0,{x1},{y1}]; the area "
        "to its right or below is black padding and is not source evidence. "
        "All observation-local coordinates are normalized over the complete canvas."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--disable-topology", action="store_true")
    args = parser.parse_args()
    variant_name, variant = selected_variant()
    padded = variant["render_mode"] == "NATIVE_PAD"
    original_local_box = strong.local_box
    strong.local_box = lambda box, roi: displayed_local_box(  # type: ignore[assignment]
        box, roi, padded=padded
    )
    temporary = args.out.with_suffix(args.out.suffix + ".upstream")
    try:
        strong.build_stage3(
            SimpleNamespace(
                rois=args.rois,
                crop_dir=args.crop_dir,
                out=temporary,
                disable_topology=bool(args.disable_topology),
            )
        )
    finally:
        strong.local_box = original_local_box
    rows = read_jsonl(temporary)
    temporary.unlink()
    output = []
    explicit_notes = 0
    for source in rows:
        row = dict(source)
        width = int(row["roi_width_px"])
        height = int(row["roi_height_px"])
        prompt = str(row["model_input_prompt"])
        if padded:
            expected_input = "1024x1024"
        else:
            expected_input = f"{width}x{height}"
            prompt = re.sub(
                r"one 1024x1024 ROI crop",
                f"one exact native {expected_input} ROI crop",
                prompt,
                count=1,
            )
        note = geometry_note(row, padded)
        prompt = f"{note}\n\n{prompt}"
        row["model_input_prompt"] = prompt
        row["render_ablation_variant"] = variant_name
        row["display_coordinate_protocol"] = (
            "rendered_canvas_base1000" if padded else "native_crop_base1000"
        )
        row["observer_generation_profile"] = "STANDARD"
        output.append(row)
        explicit_notes += prompt.startswith("Observation geometry:")
    write_jsonl(args.out, output)
    audit = {
        "schema_version": "support-render-observer-prompt-audit-1",
        "variant": variant_name,
        "rows": len(output),
        "explicit_geometry_notes": explicit_notes,
        "coordinate_protocol": (
            "rendered_canvas_base1000" if padded else "native_crop_base1000"
        ),
        "all_standard_generation_profile": all(
            row.get("observer_generation_profile") == "STANDARD" for row in output
        ),
        "test_used": False,
    }
    audit["passed"] = explicit_notes == len(output)
    write_json(args.out.with_name("variant_observer_task_audit.json"), audit)
    if not audit["passed"]:
        raise RuntimeError(f"variant observer construction failed: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

