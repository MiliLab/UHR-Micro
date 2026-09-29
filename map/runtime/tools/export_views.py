#!/usr/bin/env python3
"""Export native MAP views with content-addressed resume validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, sha256_file, write_json

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from view_geometry import ViewGeometry  # noqa: E402


CANVAS_SIDE = 1024
RENDERER_VERSION = "map-capability-preserving-view-render-4"
SEMANTIC_HASH_VERSION = "map-view-semantic-input-1"
BLACK_RGB = [0, 0, 0]
RENDER_MODES = frozenset(
    {"NATIVE_PAD", "QUERY_NATIVE", "SCALE_NORMALIZED_DETAIL"}
)

# These fields describe execution of the upstream planner, not the semantic
# input consumed by the local observer. Every other manifest field is bound.
NON_SEMANTIC_RUNTIME_FIELDS = frozenset(
    {
        "api_mode",
        "attempt",
        "attempt_history",
        "attempt_index",
        "completed_at_unix",
        "error",
        "generation_elapsed_seconds",
        "gpu_id",
        "latency_ms",
        "model_response",
        "request_contract_sha256",
        "request_sha256",
        "started_at_unix",
        "worker",
    }
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=4096)
def _versioned_file_sha256(
    path_text: str,
    size: int,
    modified_ns: int,
    changed_ns: int,
) -> str:
    del size, modified_ns, changed_ns
    return sha256_file(Path(path_text))


def source_image_sha256(path: Path) -> str:
    """Hash bytes, caching only a specific filesystem object version."""

    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return _versioned_file_sha256(
        str(resolved),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def semantic_input_sha256(
    row: dict[str, Any], *, verified_source_sha256: str
) -> str:
    semantic_row = {
        key: value
        for key, value in row.items()
        if key not in NON_SEMANTIC_RUNTIME_FIELDS
    }
    semantic_row["image_sha256"] = verified_source_sha256
    return canonical_sha256(
        {
            "semantic_hash_version": SEMANTIC_HASH_VERSION,
            "manifest_semantics": semantic_row,
        }
    )


def render_layout(
    row: dict[str, Any], source_width: int, source_height: int, mode: str
) -> dict[str, Any]:
    """Return the exact canvas/content geometry for one frozen render mode."""

    if mode not in RENDER_MODES:
        raise ValueError(f"unsupported view render mode: {mode}")
    if mode == "NATIVE_PAD":
        if source_width > CANVAS_SIDE or source_height > CANVAS_SIDE:
            raise ValueError(
                f"native ROI exceeds frozen canvas: {source_width}x{source_height}"
            )
        return {
            "canvas_size_px": [CANVAS_SIDE, CANVAS_SIDE],
            "content_rect_canvas_xyxy_px": [0, 0, source_width, source_height],
            "rendered_content_size_px": [source_width, source_height],
            "query_crop_origin_abs_px": None,
            "query_crop_size_px": None,
            "padding_semantics": "right_bottom_black_padding_is_not_source_evidence",
        }
    if mode == "QUERY_NATIVE":
        x0, y0, _x1, _y1 = [int(value) for value in row["roi_abs_xyxy"]]
        return {
            "canvas_size_px": [source_width, source_height],
            "content_rect_canvas_xyxy_px": [0, 0, source_width, source_height],
            "rendered_content_size_px": [source_width, source_height],
            "query_crop_origin_abs_px": [x0, y0],
            "query_crop_size_px": [source_width, source_height],
            "padding_semantics": "none_exact_original_pixel_crop",
        }

    virtual = row.get("target_detail_virtual_canvas_size_px")
    paste = row.get("target_detail_source_paste_xyxy")
    if not isinstance(virtual, list) or len(virtual) != 2:
        raise ValueError("target detail lacks virtual canvas size")
    if not isinstance(paste, list) or len(paste) != 4:
        raise ValueError("target detail lacks source paste rectangle")
    virtual_w, virtual_h = [int(value) for value in virtual]
    px0, py0, px1, py1 = [int(value) for value in paste]
    if (
        virtual_w < 1
        or virtual_h < 1
        or not (0 <= px0 < px1 <= virtual_w and 0 <= py0 < py1 <= virtual_h)
        or px1 - px0 != source_width
        or py1 - py0 != source_height
    ):
        raise ValueError("target detail virtual canvas/paste geometry is inconsistent")
    scale_x = CANVAS_SIDE / float(virtual_w)
    scale_y = CANVAS_SIDE / float(virtual_h)
    content_rect = [
        px0 * scale_x,
        py0 * scale_y,
        px1 * scale_x,
        py1 * scale_y,
    ]
    return {
        "canvas_size_px": [CANVAS_SIDE, CANVAS_SIDE],
        "content_rect_canvas_xyxy_px": content_rect,
        "rendered_content_size_px": [
            content_rect[2] - content_rect[0],
            content_rect[3] - content_rect[1],
        ],
        "query_crop_origin_abs_px": None,
        "query_crop_size_px": None,
        "padding_semantics": "historical_virtual_crop_black_edge_padding_before_resize",
        "target_detail_virtual_canvas_size_px": [virtual_w, virtual_h],
        "target_detail_source_paste_xyxy": [px0, py0, px1, py1],
    }


def render_spec(
    geometry: ViewGeometry,
    *,
    verified_source_sha256: str,
    render_mode: str,
    source_crop_size_px: tuple[int, int],
    layout: dict[str, Any],
) -> dict[str, Any]:
    paste_mode = {
        "NATIVE_PAD": "native_crop_top_left_without_resize",
        "QUERY_NATIVE": "exact_original_pixel_crop_without_resize_or_padding",
        "SCALE_NORMALIZED_DETAIL": (
            "historical_expand25_min128_black_pad_then_force_resize_1024_square"
        ),
    }[render_mode]
    return {
        "renderer_version": RENDERER_VERSION,
        "source_image_sha256": verified_source_sha256,
        "view_geometry": geometry.to_metadata(),
        "view_render_mode": render_mode,
        "source_crop_size_px": list(source_crop_size_px),
        "rendered_content_size_px": list(layout["rendered_content_size_px"]),
        "canvas_size_px": list(layout["canvas_size_px"]),
        "content_rect_canvas_xyxy_px": list(
            layout["content_rect_canvas_xyxy_px"]
        ),
        "canvas_fill_rgb": BLACK_RGB,
        "paste_mode": paste_mode,
        "resize_filter": (
            "PIL_BILINEAR" if render_mode == "SCALE_NORMALIZED_DETAIL" else None
        ),
        "image_mode": "RGB",
        "image_format": "PNG",
        "png_optimize": False,
    }


def _resume_mismatches(
    prior: dict[str, Any],
    metadata: dict[str, Any],
    image_path: Path,
) -> list[str]:
    fields = (
        "source_image_sha256_verified",
        "semantic_input_sha256",
        "render_spec",
        "render_spec_sha256",
        "view_geometry",
    )
    mismatches = [field for field in fields if prior.get(field) != metadata.get(field)]
    expected_crop_hash = prior.get("roi_image_sha256")
    if not isinstance(expected_crop_hash, str) or len(expected_crop_hash) != 64:
        mismatches.append("roi_image_sha256_missing")
    elif sha256_file(image_path) != expected_crop_hash:
        mismatches.append("roi_image_sha256")
    return mismatches


def _atomic_save_png(image: Any, image_path: Path) -> None:
    temporary = image_path.with_name(image_path.name + ".tmp")
    image.save(temporary, format="PNG", optimize=False)
    os.replace(temporary, image_path)


def _atomic_write_metadata(metadata: dict[str, Any], metadata_path: Path) -> None:
    temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, metadata_path)


def export_one(row: dict[str, Any], out_dir: Path, resume: bool) -> dict[str, Any]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    roi_id = str(row["roi_id"])
    target_dir = out_dir / roi_id
    image_path = target_dir / "roi.png"
    metadata_path = target_dir / "roi_meta.json"
    source_path = Path(str(row["image_abs_path"])).resolve(strict=True)
    x0, y0, x1, y1 = [int(value) for value in row["roi_abs_xyxy"]]
    source_w = int(row["original_width_px"])
    source_h = int(row["original_height_px"])
    content_w, content_h = x1 - x0, y1 - y0
    if not (0 <= x0 < x1 <= source_w and 0 <= y0 < y1 <= source_h):
        raise ValueError(f"invalid source ROI for {roi_id}: {[x0, y0, x1, y1]}")
    render_mode = str(row.get("view_render_mode") or "NATIVE_PAD")
    layout = render_layout(row, content_w, content_h, render_mode)
    with Image.open(source_path) as source_header:
        if source_header.size != (source_w, source_h):
            raise ValueError(
                f"source dimensions differ for {roi_id}: "
                f"declared={(source_w, source_h)}, actual={source_header.size}"
            )
    verified_source_hash = source_image_sha256(source_path)
    declared_source_hash = row.get("image_sha256")
    if declared_source_hash is not None and str(declared_source_hash) != verified_source_hash:
        raise ValueError(
            f"source image hash differs for {roi_id}: "
            f"declared={declared_source_hash}, actual={verified_source_hash}"
        )

    geometry = ViewGeometry.axis_aligned(
        source_size_px=(source_w, source_h),
        source_roi_xyxy_px=(x0, y0, x1, y1),
        canvas_size_px=layout["canvas_size_px"],
        content_rect_canvas_xyxy_px=layout["content_rect_canvas_xyxy_px"],
    )
    semantic_hash = semantic_input_sha256(
        row, verified_source_sha256=verified_source_hash
    )
    current_render_spec = render_spec(
        geometry,
        verified_source_sha256=verified_source_hash,
        render_mode=render_mode,
        source_crop_size_px=(content_w, content_h),
        layout=layout,
    )
    metadata = dict(row)
    metadata.update(
        {
            "roi_image_path": str(image_path.resolve()),
            "roi_image_format": "png",
            "view_geometry": geometry.to_metadata(),
            "local_coordinate_protocol": (
                "native_crop_base1000"
                if render_mode == "QUERY_NATIVE"
                else "rendered_canvas_base1000"
            ),
            "padding_semantics": layout["padding_semantics"],
            "view_render_mode": render_mode,
            "rendered_content_size_px": list(layout["rendered_content_size_px"]),
            "query_crop_origin_abs_px": layout["query_crop_origin_abs_px"],
            "query_crop_size_px": layout["query_crop_size_px"],
            "source_image_sha256_verified": verified_source_hash,
            "semantic_input_sha256": semantic_hash,
            "render_spec": current_render_spec,
            "render_spec_sha256": canonical_sha256(current_render_spec),
        }
    )

    stale_reasons: list[str] = []
    if (
        resume
        and image_path.is_file()
        and metadata_path.is_file()
        and image_path.stat().st_size > 0
    ):
        try:
            prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            stale_reasons = ["invalid_metadata_json"]
        else:
            stale_reasons = _resume_mismatches(prior, metadata, image_path)
            if not stale_reasons:
                return {
                    "roi_id": roi_id,
                    "state": "reused",
                    "source_image_sha256": verified_source_hash,
                    "semantic_input_sha256": semantic_hash,
                    "render_spec_sha256": metadata["render_spec_sha256"],
                    "roi_image_sha256": prior["roi_image_sha256"],
                }

    target_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source:
        crop = source.crop((x0, y0, x1, y1)).convert("RGB")
        if crop.size != (content_w, content_h):
            raise RuntimeError(f"unexpected crop dimensions for {roi_id}: {crop.size}")
        if render_mode == "QUERY_NATIVE":
            rendered = crop
        elif render_mode == "SCALE_NORMALIZED_DETAIL":
            virtual_w, virtual_h = layout["target_detail_virtual_canvas_size_px"]
            px0, py0, _px1, _py1 = layout["target_detail_source_paste_xyxy"]
            virtual_canvas = Image.new(
                "RGB", (virtual_w, virtual_h), tuple(BLACK_RGB)
            )
            virtual_canvas.paste(crop, (px0, py0))
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            rendered = virtual_canvas.resize(
                (CANVAS_SIDE, CANVAS_SIDE), resample=resampling
            )
        else:
            rendered = Image.new(
                "RGB", (CANVAS_SIDE, CANVAS_SIDE), tuple(BLACK_RGB)
            )
            rendered.paste(crop, (0, 0))
        _atomic_save_png(rendered, image_path)
    crop_hash = sha256_file(image_path)
    metadata["roi_image_sha256"] = crop_hash
    _atomic_write_metadata(metadata, metadata_path)
    return {
        "roi_id": roi_id,
        "state": "rewritten" if stale_reasons else "written",
        "stale_reasons": stale_reasons,
        "source_image_sha256": verified_source_hash,
        "semantic_input_sha256": semantic_hash,
        "render_spec_sha256": metadata["render_spec_sha256"],
        "roi_image_sha256": crop_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    rows = read_jsonl(args.roi_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        results = list(
            pool.map(
                lambda row: export_one(row, args.out_dir, bool(args.resume)), rows
            )
        )
    stale_counts = Counter(
        reason for item in results for reason in item.get("stale_reasons") or []
    )
    render_modes = Counter(
        str(row.get("view_render_mode") or "NATIVE_PAD") for row in rows
    )
    summary = {
        "schema_version": "map-view-export-audit-3",
        "rows": len(rows),
        "unique_roi_ids": len({row["roi_id"] for row in rows}),
        "written": sum(item["state"] == "written" for item in results),
        "rewritten_stale": sum(item["state"] == "rewritten" for item in results),
        "reused": sum(item["state"] == "reused" for item in results),
        "stale_reason_counts": dict(sorted(stale_counts.items())),
        "canvas_size_policy": (
            "STANDARD=1024_square; QUERY_NATIVE=exact_crop_dimensions"
        ),
        "render_mode_counts": dict(sorted(render_modes.items())),
        "query_native_resized_or_padded": False,
        "target_detail_resize_policy": (
            "historical_per_axis_expand25_min128_black_pad_bilinear_1024_square"
        ),
        "view_transform_recorded": True,
        "resume_binds_source_image_sha256": True,
        "resume_binds_semantic_input_sha256": True,
        "resume_binds_render_spec_sha256": True,
        "resume_binds_roi_image_sha256": True,
        "padding_coordinates_rejected_by_registration": True,
        "test_used": False,
    }
    summary["passed"] = (
        summary["rows"] == summary["unique_roi_ids"]
        and summary["rows"]
        == summary["written"] + summary["rewritten_stale"] + summary["reused"]
    )
    write_json(args.out_dir / "export_summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError(f"view export failed: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
