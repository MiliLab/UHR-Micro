#!/usr/bin/env python3
"""Export ROI crop images and per-ROI metadata."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import EXP_ROOT, load_jsonl


def export_one(
    out_dir: Path,
    row: dict,
    resume: bool,
    image_format: str,
    jpeg_quality: int,
) -> tuple[bool, str, str]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    roi_id = str(row["roi_id"])
    sub = out_dir / roi_id
    image_name = "roi.jpg" if image_format == "jpeg" else "roi.png"
    image_path = sub / image_name
    meta = sub / "roi_meta.json"
    if resume and image_path.is_file() and meta.is_file() and image_path.stat().st_size > 0:
        return True, roi_id, "skipped"
    sub.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = [int(v) for v in row["roi_abs_xyxy"]]
    with Image.open(row["image_abs_path"]).convert("RGB") as im:
        crop = im.crop((x1, y1, x2, y2))
        if crop.size != (1024, 1024):
            # Only pad when the source image dimension itself is smaller than 1024.
            padded = Image.new("RGB", (1024, 1024), (0, 0, 0))
            padded.paste(crop, (0, 0))
            crop = padded
        if image_format == "jpeg":
            crop.save(
                image_path,
                format="JPEG",
                quality=jpeg_quality,
                subsampling="4:2:0",
                optimize=False,
            )
        else:
            crop.save(image_path, format="PNG", optimize=False)
    row = dict(row)
    row["roi_image_path"] = str(image_path.resolve())
    row["roi_image_format"] = image_format
    meta.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True, roi_id, ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Export ablation2 ROI crops")
    ap.add_argument("--roi-manifest", type=Path, default=EXP_ROOT / "results" / "full_roi_agent" / "rois" / "roi_manifest.jsonl")
    ap.add_argument("--out-dir", type=Path, default=EXP_ROOT / "results" / "full_roi_agent" / "roi_crops")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-format", choices=("jpeg", "png"), default="png")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    rows = load_jsonl(args.roi_manifest.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = skipped = fail = 0
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {
            ex.submit(
                export_one,
                out_dir,
                row,
                bool(args.resume),
                str(args.image_format),
                max(1, min(100, int(args.jpeg_quality))),
            ): row
            for row in rows
        }
        for fut in as_completed(futs):
            good, rid, msg = fut.result()
            if good:
                ok += 1
                skipped += 1 if msg == "skipped" else 0
            else:
                fail += 1
                failures.append({"roi_id": rid, "error": msg})
    summary = {
        "input": str(args.roi_manifest.resolve()),
        "out_dir": str(out_dir),
        "total": len(rows),
        "ok_or_skipped": ok,
        "skipped_resume": skipped,
        "fail": fail,
        "image_format": str(args.image_format),
        "jpeg_quality": int(args.jpeg_quality) if args.image_format == "jpeg" else None,
    }
    if failures:
        with (out_dir / "failures.jsonl").open("w", encoding="utf-8") as f:
            for item in failures:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
