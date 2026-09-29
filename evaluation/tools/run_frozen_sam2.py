#!/usr/bin/env python3
"""Run the one frozen SAM2 bbox-to-mask protocol without loading GT answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

EVAL_ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_rle(mask: np.ndarray) -> str:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]
    return encoded.decode("utf-8") if isinstance(encoded, bytes) else str(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", type=Path, required=True)
    parser.add_argument("--split", choices=["Development", "Validation", "Test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test-after-freeze", action="store_true")
    parser.add_argument("--phase0-contract", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    phase0_path = args.phase0_contract or EVAL_ROOT / "configs/phase0_contract.lock.json"
    phase0 = read_json(phase0_path)
    contract = read_json(EVAL_ROOT / "configs/sam2_contract.json")
    parsed_rows = read_jsonl(args.parsed)
    parsed_by_qa: dict[str, dict[str, Any]] = {}
    for row in parsed_rows:
        qa = str(row.get("qa_relpath") or "")
        if not qa or qa in parsed_by_qa:
            raise ValueError(f"missing/duplicate qa_relpath: {qa}")
        parsed_by_qa[qa] = row
    manifest_path = args.manifest or EVAL_ROOT / "manifests" / f"{args.split}.jsonl"
    manifest = read_jsonl(manifest_path)
    seg_manifest = [row for row in manifest if row["task_key"] in {"Understanding/ReferSegmentation", "Understanding/ComponentSegmentation"}]

    sam2_root = Path(os.environ.get("MAP_SAM2_ROOT", contract["sam2_root_local"]))
    sys.path.insert(0, str(sam2_root))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(
        str(contract["hydra_config_name"]),
        str(sam2_root / contract["checkpoint_relpath_sam2_root"]),
        device=args.device,
    )
    predictor = SAM2ImagePredictor(model)
    use_cuda_autocast = str(args.device).startswith("cuda")
    tf32_enabled = bool(use_cuda_autocast and torch.cuda.get_device_properties(args.device).major >= 8)
    if tf32_enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dataset_root = args.dataset_root or EVAL_ROOT.parents[1] / "UHR-Bench-v4"
    output: list[dict[str, Any]] = []
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_autocast
    ):
        for manifest_row in seg_manifest:
            qa = str(manifest_row["qa_relpath"])
            parsed = parsed_by_qa.get(qa)
            source_pixel_boxes = parsed.get("source_pixel_boxes") if parsed else None
            bbox = source_pixel_boxes[0] if isinstance(source_pixel_boxes, list) and source_pixel_boxes else None
            bbox_sha = canonical_hash(bbox)
            base = {
                "schema_version": "uhr-bench-v4-sam2-mask-1",
                "run_id": str(parsed.get("run_id") if parsed else "missing"),
                "phase0_contract_sha256": phase0["phase0_contract_sha256"],
                "benchmark_manifest_sha256": phase0["benchmark_manifest_sha256"],
                "sam2_contract_sha256": contract["sam2_contract_sha256"],
                "qa_relpath": qa,
                "task_key": manifest_row["task_key"],
                "image_sha256": manifest_row["image_sha256"],
                "bbox_sha256": bbox_sha,
                "status": "bbox_invalid",
                "pred_mask_rle": None,
                "mask_size_hw": None,
                "diagnostics": {},
            }
            if not parsed or parsed.get("parse_status") != "ok" or parsed.get("protocol_status") != "ok" or not isinstance(bbox, list) or len(bbox) != 4:
                base["diagnostics"] = {"reason": "requires_one_valid_source_pixel_AABB"}
                output.append(base)
                continue
            try:
                image_path = (dataset_root / manifest_row["image_relpath"]).resolve(strict=True)
                dataset_root_resolved = dataset_root.resolve(strict=True)
                if not image_path.is_relative_to(dataset_root_resolved):
                    raise PermissionError("manifest image escaped frozen dataset root")
                actual_image_sha256 = sha256_file(image_path)
                if actual_image_sha256 != str(manifest_row["image_sha256"]):
                    base["status"] = "image_error"
                    base["diagnostics"] = {
                        "reason": "source_image_sha256_mismatch",
                        "actual_image_sha256": actual_image_sha256,
                    }
                    output.append(base)
                    continue
                image = np.asarray(Image.open(image_path).convert("RGB"))
                predictor.set_image(image)
                masks, scores, _ = predictor.predict(
                    point_coords=None, point_labels=None, box=np.asarray(bbox, dtype=np.float32)[None, :],
                    mask_input=None, multimask_output=False, return_logits=False,
                )
                selected = 0
                mask = masks[selected]
                base.update(status="success", pred_mask_rle=compact_rle(mask), mask_size_hw=[int(mask.shape[0]), int(mask.shape[1])])
                base["diagnostics"] = {
                    "selected_mask_index": selected,
                    "predicted_iou_score": float(scores[selected]),
                    "numerical_precision": "bfloat16" if use_cuda_autocast else "float32_nonformal_fallback",
                    "tf32_enabled": tf32_enabled,
                    "source_image_sha256_verified": actual_image_sha256,
                    "sam2_runtime_contract": sam2_verification,
                }
            except Exception as exc:
                base["status"] = "sam2_error"
                base["diagnostics"] = {"error_type": type(exc).__name__, "error": str(exc)}
            output.append(base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"rows": len(output), "success": sum(row["status"] == "success" for row in output), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
