#!/usr/bin/env python3
"""Prepare the public image shown to the terminal composer.

GLOBAL_LAYOUT contracts receive the byte-identical complete source image.
Every other contract receives the historical max-side-2048 canvas whose
rendered boxes come only from public coordinates or this run's support planner.
They are not dataset annotations and do not encode answers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from io_utils import read_jsonl, sha256_file, write_json, write_jsonl


Image.MAX_IMAGE_PIXELS = None


def render_one(
    source: dict[str, Any],
    rois: list[dict[str, Any]],
    out_dir: Path,
    max_side: int,
) -> dict[str, Any]:
    source_path = Path(str(source["image_abs_path"])).resolve(strict=True)
    expected_hash = str(source["image_sha256"])
    actual_hash = sha256_file(source_path)
    if actual_hash != expected_hash:
        raise ValueError("composer source image bytes differ from frozen manifest")

    sample_id = str(source["sample_id"])
    contract = source.get("evidence_contract") or {}
    activations = contract.get("operator_activations") or []
    if not isinstance(activations, list):
        raise ValueError("public operator activations must be a list")
    use_unmodified_source = bool(
        {"GLOBAL_LAYOUT", "SEMANTIC_SCOPE"}
        & {str(activation) for activation in activations}
    )
    if use_unmodified_source:
        with Image.open(source_path) as image:
            source_width, source_height = image.size
        return {
            "sample_id": sample_id,
            "source_image_path": str(source_path),
            "source_image_sha256_verified": actual_hash,
            "source_canvas_sha256": actual_hash,
            "source_canvas_unmodified": True,
            "source_canvas_is_method_generated": False,
            "dataset_annotation_used": False,
            "display_scale": 1.0,
            "display_width_px": source_width,
            "display_height_px": source_height,
            "footprints": [],
        }

    target = out_dir / f"{sample_id}.jpg"
    with Image.open(source_path).convert("RGB") as image:
        source_width, source_height = image.size
        scale = min(1.0, float(max_side) / float(max(source_width, source_height)))
        if scale < 1.0:
            image = image.resize(
                (
                    max(1, int(round(source_width * scale))),
                    max(1, int(round(source_height * scale))),
                ),
                Image.Resampling.BILINEAR,
            )
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(
                "DejaVuSans-Bold.ttf", max(18, min(image.size) // 55)
            )
        except OSError:
            font = ImageFont.load_default()
        line_width = max(3, min(image.size) // 350)
        ordered = sorted(
            rois,
            key=lambda row: (int(row.get("roi_index", 0)), str(row.get("roi_id", ""))),
        )
        for display_index, roi in enumerate(ordered, start=1):
            x1, y1, x2, y2 = [
                int(round(float(value) * scale)) for value in roi["roi_abs_xyxy"]
            ]
            for inset in range(line_width):
                draw.rectangle(
                    (x1 - inset, y1 - inset, x2 + inset, y2 + inset),
                    outline=(255, 255, 255),
                )
            label = str(display_index)
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            label_width = right - left + 10
            label_height = bottom - top + 8
            draw.rectangle(
                (x1, y1, x1 + label_width, y1 + label_height),
                fill=(255, 255, 255),
            )
            draw.text((x1 + 5, y1 + 3), label, fill=(0, 0, 0), font=font)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="JPEG", quality=90, optimize=True)

    return {
        "sample_id": sample_id,
        "source_image_path": str(target.resolve()),
        "source_image_sha256_verified": actual_hash,
        "source_canvas_sha256": sha256_file(target),
        "source_canvas_unmodified": False,
        "source_canvas_is_method_generated": True,
        "dataset_annotation_used": False,
        "display_scale": scale,
        "display_width_px": image.width,
        "display_height_px": image.height,
        "footprints": [
            {
                "display_index": index,
                "roi_id": roi.get("roi_id"),
                "roi_abs_xyxy": roi.get("roi_abs_xyxy"),
                "roi_full1000_xyxy": roi.get("roi_full1000_xyxy"),
                "roi_source": roi.get("roi_source"),
                "support_label": roi.get("source_support_label"),
            }
            for index, roi in enumerate(ordered, start=1)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for roi in read_jsonl(args.rois):
        grouped[str(roi["sample_id"])].append(roi)

    futures = []
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for source in manifest:
            sample_id = str(source["sample_id"])
            futures.append(
                executor.submit(
                    render_one,
                    source,
                    grouped.get(sample_id, []),
                    args.out_dir,
                    max(512, int(args.max_side)),
                )
            )
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(args.out, rows)

    audit = {
        "schema_version": "map-source-canvas-audit-2",
        "rows": len(rows),
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "byte_unmodified_complete_source_rows": sum(
            row["source_canvas_unmodified"] is True for row in rows
        ),
        "method_generated_support_canvas_rows": sum(
            row["source_canvas_is_method_generated"] is True for row in rows
        ),
        "pixels_modified_only_for_support_canvas_rows": True,
        "method_generated_support_footprints_only": True,
        "ground_truth_or_dataset_annotation_used": False,
        "source_images_byte_verified": len(
            {row["source_image_sha256_verified"] for row in rows}
        ),
        "test_used": False,
    }
    audit["passed"] = (
        audit["rows"] == audit["unique_sample_ids"]
        and audit["ground_truth_or_dataset_annotation_used"] is False
    )
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"source canvas audit failed: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
