from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .constants import CHOICE_TASKS, COUNT_TASKS, MULTI_BOX_TASKS, SEGMENTATION_TASKS, SINGLE_BOX_TASKS
from .geometry import Box, validate_box
from .io import read_jsonl, sha256_file
from .parsing import parse_boxes


@dataclass(frozen=True)
class GroundTruthSample:
    qa_relpath: str
    task_key: str
    image_relpath: str
    image_width_px: int
    image_height_px: int
    question: str
    answer_raw: str
    answer_value: Any
    qa_sha256: str
    image_sha256: str
    split_position: int


def _parse_polygon(text: str) -> list[int]:
    value = json.loads(text)
    if not isinstance(value, list) or len(value) < 6 or len(value) % 2:
        raise ValueError("refer segmentation GT must be one flat even-length polygon")
    out = [int(float(item)) for item in value]
    return out


def parse_gt_answer(task: str, text: str, width: int, height: int) -> Any:
    value = str(text).strip()
    if task in COUNT_TASKS:
        if not value.isdigit():
            raise ValueError(f"{task}: GT is not a nonnegative integer: {value!r}")
        return int(value)
    if task in CHOICE_TASKS:
        if value not in {"A", "B", "C", "D"}:
            raise ValueError(f"{task}: GT is not A-D: {value!r}")
        return value
    if task in SINGLE_BOX_TASKS - SEGMENTATION_TASKS or task in MULTI_BOX_TASKS:
        boxes, status, diagnostics = parse_boxes(value)
        if status != "ok" or not boxes:
            raise ValueError(f"{task}: invalid GT boxes: {diagnostics}")
        for box in boxes:
            ok, reason = validate_box(box, width, height, range_check=False)
            if not ok:
                raise ValueError(f"{task}: invalid GT box order/shape: {reason}")
        if task in SINGLE_BOX_TASKS and len(boxes) != 1:
            raise ValueError(f"{task}: expected exactly one GT box, got {len(boxes)}")
        return boxes
    if task == "Understanding/ReferSegmentation":
        return _parse_polygon(value)
    if task == "Understanding/ComponentSegmentation":
        if not value:
            raise ValueError("empty COCO RLE GT")
        return value
    raise ValueError(f"unsupported task: {task}")


def load_ground_truth(
    eval_root: Path,
    dataset_root: Path,
    split: str,
    *,
    verify_hashes: bool = True,
    manifest_dir: Path | None = None,
) -> list[GroundTruthSample]:
    manifest_path = (manifest_dir or (eval_root / "manifests")) / f"{split}.jsonl"
    rows = read_jsonl(manifest_path)
    samples: list[GroundTruthSample] = []
    seen: set[str] = set()
    for expected_position, row in enumerate(rows):
        qa_relpath = str(row["qa_relpath"])
        if qa_relpath in seen:
            raise ValueError(f"duplicate manifest qa_relpath: {qa_relpath}")
        seen.add(qa_relpath)
        if int(row["split_position"]) != expected_position:
            raise ValueError(f"noncanonical split_position at {qa_relpath}")
        qa_path = dataset_root / qa_relpath
        if verify_hashes and sha256_file(qa_path) != str(row["qa_sha256"]):
            raise ValueError(f"QA hash mismatch: {qa_relpath}")
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        task = str(row["task_key"])
        width, height = int(row["image_width_px"]), int(row["image_height_px"])
        answer = str(qa.get("Answer") or "")
        samples.append(GroundTruthSample(
            qa_relpath=qa_relpath,
            task_key=task,
            image_relpath=str(row["image_relpath"]),
            image_width_px=width,
            image_height_px=height,
            question=str(qa.get("Question") or ""),
            answer_raw=answer,
            answer_value=parse_gt_answer(task, answer, width, height),
            qa_sha256=str(row["qa_sha256"]),
            image_sha256=str(row["image_sha256"]),
            split_position=expected_position,
        ))
    return samples


def decode_coco_rle(counts: str, height: int, width: int) -> np.ndarray | None:
    try:
        from pycocotools import mask as mask_util
    except ImportError:
        return None
    for encoded in (str(counts).encode("utf-8"), str(counts)):
        try:
            mask = mask_util.decode({"size": [int(height), int(width)], "counts": encoded})
            if mask.ndim == 3:
                mask = mask.any(axis=2)
            return (mask > 0).astype(np.uint8)
        except Exception:
            continue
    return None


def gt_segmentation_mask(sample: GroundTruthSample) -> np.ndarray | None:
    if sample.task_key == "Understanding/ComponentSegmentation":
        return decode_coco_rle(str(sample.answer_value), sample.image_height_px, sample.image_width_px)
    if sample.task_key != "Understanding/ReferSegmentation":
        return None
    flat = list(sample.answer_value)
    points = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    image = Image.new("L", (sample.image_width_px, sample.image_height_px), 0)
    ImageDraw.Draw(image).polygon(points, outline=1, fill=1)
    return np.asarray(image, dtype=np.uint8)
