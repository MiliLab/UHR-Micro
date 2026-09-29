#!/usr/bin/env python3
"""Compatibility CLI that renders one frozen support-view condition."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve()
TOOLS = HERE.parents[1] / "runtime" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import export_views  # noqa: E402
from io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from variant_contract import selected_variant  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--image-format", choices=("png",), default="png")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    variant_name, variant = selected_variant()
    rows = read_jsonl(args.roi_manifest)
    rendered_rows = []
    for source in rows:
        row = dict(source)
        row["view_render_mode"] = variant["render_mode"]
        # Keep the frozen STANDARD decoding budget even for exact native crops.
        row["observer_generation_profile"] = "STANDARD"
        row["render_ablation_variant"] = variant_name
        rendered_rows.append(row)
    rendered_manifest = args.out_dir.parent / "rois.rendered.jsonl"
    write_jsonl(rendered_manifest, rendered_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        results = list(
            pool.map(
                lambda row: export_views.export_one(
                    row, args.out_dir, bool(args.resume)
                ),
                rendered_rows,
            )
        )
    stale = Counter(
        reason for result in results for reason in result.get("stale_reasons") or []
    )
    audit = {
        "schema_version": "support-render-ablation-export-1",
        "variant": variant_name,
        "tile_side_px": variant["tile_side_px"],
        "render_mode": variant["render_mode"],
        "rows": len(rendered_rows),
        "unique_roi_ids": len({str(row["roi_id"]) for row in rendered_rows}),
        "written": sum(result["state"] == "written" for result in results),
        "rewritten_stale": sum(
            result["state"] == "rewritten" for result in results
        ),
        "reused": sum(result["state"] == "reused" for result in results),
        "stale_reason_counts": dict(sorted(stale.items())),
        "coordinate_protocol": (
            "native_crop_base1000"
            if variant["render_mode"] == "QUERY_NATIVE"
            else "rendered_canvas_base1000"
        ),
        "resize_used": False,
        "padding_used": variant["render_mode"] == "NATIVE_PAD",
        "test_used": False,
    }
    audit["passed"] = (
        audit["rows"] == audit["unique_roi_ids"]
        and audit["rows"]
        == audit["written"] + audit["rewritten_stale"] + audit["reused"]
    )
    write_json(args.out_dir / "export_summary.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"variant view export failed: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

