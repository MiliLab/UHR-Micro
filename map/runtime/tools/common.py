#!/usr/bin/env python3
"""Shared utilities for the Qwen3VL8B ablation2 ROI-agent experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from string import Template
from typing import Any


DEFAULT_MODEL_PATH = Path(os.environ.get("MAP_MODEL_PATH") or Path(__file__).resolve().parents[3] / "models/Qwen3-VL-main/Qwen3-VL-8B-Instruct")
DEFAULT_BENCH_ROOT = Path(os.environ.get("MAP_DATASET", ""))
DEFAULT_INDEX = DEFAULT_BENCH_ROOT / "Validation" / "index.json"
EXP_ROOT = Path(__file__).resolve().parents[1]

TASK_CHOICE = frozenset(
    {
        "Understanding/Classification",
        "Understanding/Fine-grainedRecognition",
        "Relationship/distributionUnderstanding",
        "Relationship/RelationshipRecognition",
        "Relationship/NearestFarthest",
    }
)
TASK_COUNT = frozenset(
    {
        "Count/ConditionalCount",
        "Count/RegionCount",
        "Count/GlobalCount",
        "Count/CrossRegionCompare",
    }
)
TASK_GROUND_SINGLE = frozenset({"Grounding/BasicGrounding", "Grounding/ComplexGrounding"})
TASK_GROUND_MULTI = frozenset(
    {
        "Grounding/GlobalDetection",
        "Grounding/RegionDetection",
        "Grounding/MultiConditionRetrieval",
    }
)
TASK_SEGMENTATION = frozenset(
    {
        "Understanding/ReferSegmentation",
        "Understanding/ComponentSegmentation",
    }
)
ALL_TASKS = TASK_CHOICE | TASK_COUNT | TASK_GROUND_SINGLE | TASK_GROUND_MULTI | TASK_SEGMENTATION
GLOBAL_POINT_TASKS = frozenset({"Count/GlobalCount", "Grounding/GlobalDetection"})
PROGRAM_ROI_TASKS = frozenset(
    {
        "Count/RegionCount",
        "Grounding/RegionDetection",
        "Understanding/Classification",
        "Understanding/Fine-grainedRecognition",
    }
)
POINT_MIN_DIST = 150.0
ROI_SIDE = 1024
COORD_DECIMALS = 1

_LIST_RE = re.compile(r"\[([^\[\]]+)\]")
_FINAL_RE = re.compile(r"Final answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Pair:
    image_relpath: str
    qa_relpath: str

    def task_key(self) -> str:
        parts = Path(self.qa_relpath).parts
        if len(parts) < 3 or parts[0] not in {"Validation", "Test"}:
            return "unknown/unknown"
        return f"{parts[1]}/{parts[2]}"


def qa_slug(qa_relpath: str) -> str:
    return hashlib.sha256(qa_relpath.encode("utf-8")).hexdigest()[:16]


def task_route(task_key: str) -> str:
    if task_key in TASK_CHOICE:
        return "choice"
    if task_key in TASK_COUNT:
        return "count"
    if task_key in TASK_GROUND_SINGLE:
        return "single_bbox"
    if task_key in TASK_GROUND_MULTI:
        return "multi_bbox"
    if task_key in TASK_SEGMENTATION:
        return "segmentation"
    return "unknown"


def task_prompt_key(task_key: str) -> str:
    mapping = {
        "Grounding/BasicGrounding": "basic_grounding",
        "Grounding/ComplexGrounding": "complex_grounding",
        "Grounding/GlobalDetection": "global_detection",
        "Grounding/RegionDetection": "region_detection",
        "Grounding/MultiConditionRetrieval": "multi_condition_retrieval",
        "Count/GlobalCount": "global_count",
        "Count/RegionCount": "region_count",
        "Count/ConditionalCount": "conditional_count",
        "Count/CrossRegionCompare": "cross_region_compare",
        "Understanding/Classification": "classification",
        "Understanding/Fine-grainedRecognition": "fine_grained_recognition",
        "Relationship/distributionUnderstanding": "relationship",
        "Relationship/RelationshipRecognition": "relationship",
        "Relationship/NearestFarthest": "relationship",
        "Understanding/ReferSegmentation": "segmentation",
        "Understanding/ComponentSegmentation": "segmentation",
    }
    return mapping.get(task_key, task_route(task_key))


def stage1_prompt_key(task_key: str) -> str:
    if task_key == "Grounding/GlobalDetection":
        return "global_detection"
    if task_key == "Count/GlobalCount":
        return "global_count"
    if task_key == "Grounding/MultiConditionRetrieval":
        return "multi_condition_retrieval"
    if task_key in TASK_SEGMENTATION:
        return "local_segmentation"
    if task_key in TASK_GROUND_SINGLE:
        return "local_grounding"
    if task_key in TASK_GROUND_MULTI:
        return "local_detection"
    if task_key in TASK_COUNT:
        return "local_count"
    if task_key in TASK_CHOICE:
        return "local_choice"
    return "local_grounding"


def final_prompt_key(task_key: str) -> str:
    route = task_route(task_key)
    if route == "segmentation":
        return "segmentation"
    if route == "multi_bbox":
        return "multi_bbox"
    if route == "single_bbox":
        return "single_bbox"
    if route == "count":
        return "count"
    if route == "choice":
        return "choice"
    return "single_bbox"


def load_pairs(index_path: Path) -> list[Pair]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    rows = data.get("pairs")
    if not isinstance(rows, list):
        raise ValueError(f"{index_path} missing list field: pairs")
    out: list[Pair] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        image = row.get("image")
        qa = row.get("qa")
        if isinstance(image, str) and isinstance(qa, str):
            out.append(Pair(image_relpath=image, qa_relpath=qa))
    return out


def answer_to_text(answer: Any) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def read_image_hw(path: Path) -> tuple[int, int]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        w, h = im.size
    return h, w


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def round1(v: float) -> float:
    return round(float(v), COORD_DECIMALS)


def abs_nums_to_full1000(nums: list[float], width: int, height: int) -> list[float]:
    out: list[float] = []
    for i, v in enumerate(nums):
        dim = float(width if i % 2 == 0 else height)
        out.append(round1(clamp(float(v) / max(dim, 1.0) * 1000.0, 0.0, 1000.0)))
    return out


def full1000_nums_to_abs(nums: list[float], width: int, height: int) -> list[int]:
    out: list[int] = []
    for i, v in enumerate(nums):
        dim = float(width if i % 2 == 0 else height)
        out.append(int(round(clamp(float(v), 0.0, 1000.0) / 1000.0 * dim)))
    return out


def abs_box_to_full1000(box: list[float] | tuple[float, ...], width: int, height: int) -> list[float]:
    return abs_nums_to_full1000([float(v) for v in box[:4]], width, height)


def full1000_box_to_abs(box: list[float] | tuple[float, ...], width: int, height: int) -> list[int]:
    vals = full1000_nums_to_abs([float(v) for v in box[:4]], width, height)
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(clamp(x1, 0, max(0, width - 1)))
    y1 = int(clamp(y1, 0, max(0, height - 1)))
    x2 = int(clamp(x2, 0, width))
    y2 = int(clamp(y2, 0, height))
    return [x1, y1, x2, y2]


def roi1000_nums_to_full1000(nums: list[float], roi_abs: list[int], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = roi_abs
    rw = max(1.0, float(x2 - x1))
    rh = max(1.0, float(y2 - y1))
    abs_vals: list[float] = []
    for i, v in enumerate(nums):
        if i % 2 == 0:
            abs_vals.append(float(x1) + float(v) / 1000.0 * rw)
        else:
            abs_vals.append(float(y1) + float(v) / 1000.0 * rh)
    return abs_nums_to_full1000(abs_vals, width, height)


def extract_numeric_lists(text: str, lengths: tuple[int, ...] = (2, 4, 8)) -> list[list[float]]:
    out: list[list[float]] = []
    for m in _LIST_RE.finditer(str(text or "")):
        raw = m.group(1)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) not in lengths:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if all(math.isfinite(v) for v in vals):
            out.append(vals)
    return out


def is_valid_abs_box(box: list[int] | tuple[int, ...], width: int, height: int) -> bool:
    if len(box) < 4:
        return False
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def is_valid_full1000_box(box: list[float] | tuple[float, ...]) -> bool:
    if len(box) < 4:
        return False
    vals = [float(v) for v in box[:4]]
    return all(0.0 <= v <= 1000.0 for v in vals) and vals[2] > vals[0] and vals[3] > vals[1]


def format_num(v: float) -> str:
    if abs(float(v) - round(float(v))) < 1e-6:
        return str(int(round(float(v))))
    return f"{float(v):.1f}"


def format_list(vals: list[float] | tuple[float, ...]) -> str:
    return "[" + ",".join(format_num(float(v)) for v in vals) + "]"


def convert_question_coords_to_full1000(question: str, width: int, height: int) -> tuple[str, list[list[float]], list[list[float]]]:
    original_lists: list[list[float]] = []
    converted_lists: list[list[float]] = []

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) not in (2, 4, 8):
            return match.group(0)
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return match.group(0)
        conv = abs_nums_to_full1000(vals, width, height)
        original_lists.append(vals)
        converted_lists.append(conv)
        return format_list(conv)

    return _LIST_RE.sub(repl, question), original_lists, converted_lists


def strip_final_instruction(question: str, task_key: str) -> str:
    q = str(question).strip()
    patterns = [
        r"\s+Return their locations as oriented bounding boxes, where each box is represented by 4 polygon points in the format \[[^\]]+\]\.?",
        r"\s+Return their locations as axis-aligned bounding boxes, where each box is represented by \[[^\]]+\]\.?",
        r"\s+return their locations as oriented bounding boxes, where each box is represented by 4 polygon points in the format \[[^\]]+\]\.?",
        r"\s+return their locations as axis-aligned bounding boxes, where each box is represented by \[[^\]]+\]\.?",
        r"\s+and return their locations as oriented bounding boxes, where each box is represented by 4 polygon points in the format \[[^\]]+\]\.?",
        r"\s+and return their locations as axis-aligned bounding boxes, where each box is represented by \[[^\]]+\]\.?",
        r"\s+Return its location as a bounding box represented by \[[^\]]+\]\.?",
        r"\s+Please output only the letter of the correct option \(A, B, C, or D\)\.?",
        r"\s+and return only one Arabic numeral\.?",
        r"\s+Please answer with one Arabic numeral\.?",
    ]
    for pat in patterns:
        q = re.sub(pat, "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+(and|to)\s*$", "", q, flags=re.IGNORECASE)
    if task_key in TASK_CHOICE:
        q = re.sub(r"\n?Choose one:\n(?:[A-D][\.:].*(?:\n|$))+", "", q, flags=re.IGNORECASE)
    return q.strip()


def has_explicit_region(task_key: str, coord_lists: list[list[float]]) -> bool:
    if not coord_lists:
        return False
    if task_key in PROGRAM_ROI_TASKS:
        return True
    return False


def seed_boxes_from_question(task_key: str, coord_lists_abs: list[list[float]], width: int, height: int) -> list[list[int]]:
    boxes: list[list[int]] = []
    if not coord_lists_abs:
        return boxes
    for vals in coord_lists_abs:
        if len(vals) == 2:
            cx, cy = vals
            boxes.append(point_to_roi_abs(cx, cy, width, height, ROI_SIDE))
        elif len(vals) == 4:
            x1, y1, x2, y2 = vals
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
        elif len(vals) == 8:
            xs = vals[0::2]
            ys = vals[1::2]
            boxes.append([int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))])
    return [clamp_box_abs(b, width, height) for b in boxes]


def point_to_roi_abs(cx: float, cy: float, width: int, height: int, side: int = ROI_SIDE) -> list[int]:
    if width <= side:
        x1, x2 = 0, width
    else:
        x1 = int(round(cx - side / 2))
        x1 = int(clamp(x1, 0, width - side))
        x2 = x1 + side
    if height <= side:
        y1, y2 = 0, height
    else:
        y1 = int(round(cy - side / 2))
        y1 = int(clamp(y1, 0, height - side))
        y2 = y1 + side
    return [x1, y1, x2, y2]


def full1000_point_to_abs(point: list[float] | tuple[float, float], width: int, height: int) -> tuple[float, float]:
    x = clamp(float(point[0]), 0.0, 1000.0) / 1000.0 * width
    y = clamp(float(point[1]), 0.0, 1000.0) / 1000.0 * height
    return x, y


def abs_point_to_full1000(point: list[float] | tuple[float, float], width: int, height: int) -> list[float]:
    x, y = point
    return [round1(float(x) / max(1, width) * 1000.0), round1(float(y) / max(1, height) * 1000.0)]


def clamp_box_abs(box: list[int] | tuple[int, ...], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(clamp(x1, 0, max(0, width - 1)))
    y1 = int(clamp(y1, 0, max(0, height - 1)))
    x2 = int(clamp(x2, x1 + 1, width))
    y2 = int(clamp(y2, y1 + 1, height))
    return [x1, y1, x2, y2]


def expand_box_to_roi(box: list[int], width: int, height: int, side: int = ROI_SIDE) -> list[list[int]]:
    x1, y1, x2, y2 = clamp_box_abs(box, width, height)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= side and bh <= side:
        return [point_to_roi_abs((x1 + x2) * 0.5, (y1 + y2) * 0.5, width, height, side)]
    rois: list[list[int]] = []
    if bw >= bh:
        centers = [(x1 + min(side / 2, bw / 2), (y1 + y2) * 0.5), (x2 - min(side / 2, bw / 2), (y1 + y2) * 0.5)]
    else:
        centers = [((x1 + x2) * 0.5, y1 + min(side / 2, bh / 2)), ((x1 + x2) * 0.5, y2 - min(side / 2, bh / 2))]
    for cx, cy in centers:
        roi = point_to_roi_abs(cx, cy, width, height, side)
        if roi not in rois:
            rois.append(roi)
    return rois[:2]


def box_intersects(a: list[int], b: list[int]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def box_iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    ba = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(aa + ba - inter)


def dedup_points(points: list[list[float]], min_dist: float = POINT_MIN_DIST) -> list[list[float]]:
    out: list[list[float]] = []
    for p in points:
        if len(p) < 2:
            continue
        x, y = clamp(float(p[0]), 0.0, 1000.0), clamp(float(p[1]), 0.0, 1000.0)
        if all(math.hypot(x - q[0], y - q[1]) >= min_dist for q in out):
            out.append([round1(x), round1(y)])
    return out


def parse_points(text: str, max_points: int) -> list[list[float]]:
    lists = extract_numeric_lists(text, lengths=(2, 4, 8))
    points: list[list[float]] = []
    for vals in lists:
        if len(vals) == 2:
            points.append(vals[:2])
        elif len(vals) == 4:
            x1, y1, x2, y2 = vals
            points.append([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
        elif len(vals) == 8:
            xs = vals[0::2]
            ys = vals[1::2]
            points.append([sum(xs) / len(xs), sum(ys) / len(ys)])
    return dedup_points(points)[:max_points]


def parse_final_answer(text: str) -> str | None:
    m = _FINAL_RE.search(str(text or ""))
    if not m:
        return None
    ans = m.group(1).strip()
    # Keep the first non-empty line when the model appends extra prose after the answer.
    for line in ans.splitlines():
        line = line.strip()
        if line:
            return line
    return ans


def template_values(row: dict[str, Any]) -> dict[str, Any]:
    vals = dict(row)
    vals.setdefault("H", row.get("original_height_px") or row.get("full_height_px") or row.get("roi_height_px") or 0)
    vals.setdefault("W", row.get("original_width_px") or row.get("full_width_px") or row.get("roi_width_px") or 0)
    vals.setdefault("H_full", row.get("original_height_px") or row.get("full_height_px") or 0)
    vals.setdefault("W_full", row.get("original_width_px") or row.get("full_width_px") or 0)
    vals.setdefault("question", row.get("question_1000") or row.get("question") or "")
    vals.setdefault("question_original", row.get("question_original") or row.get("question") or "")
    vals.setdefault("stage3_results_text", "")
    vals.setdefault("roi_context", "")
    vals.setdefault("final_format_hint", row.get("final_format_hint") or "")
    return vals


def render_prompt(prompt_path: Path, row: dict[str, Any], question_key: str = "question") -> str:
    tpl = prompt_path.read_text(encoding="utf-8").rstrip()
    vals = template_values(row)
    if question_key in row:
        vals["question"] = row.get(question_key) or ""
    return Template(tpl).safe_substitute(vals).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(path: Path, id_key: str) -> set[str]:
    out: set[str] = set()
    for row in load_jsonl(path):
        ident = str(row.get(id_key) or "")
        if ident and not row.get("error"):
            out.add(ident)
    return out


def _auto_pixels(min_pixels: int | None, max_pixels: int | None) -> tuple[int, int]:
    if min_pixels is None or max_pixels is None:
        patch_size = 16
        merge_base = 2
        pixels_per_token = patch_size * patch_size * merge_base * merge_base
        return min_pixels or pixels_per_token * 768, max_pixels or pixels_per_token * 12800
    return min_pixels, max_pixels


def _build_vllm_messages(image_path: str, prompt: str, min_pixels: int | None, max_pixels: int | None) -> list[dict[str, Any]]:
    min_px, max_px = _auto_pixels(min_pixels, max_pixels)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}", "min_pixels": min_px, "max_pixels": max_px},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _prepare_inputs_for_vllm(messages: list[dict[str, Any]], processor) -> dict[str, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:

        def process_vision_info(messages, **kwargs):  # type: ignore[no-redef]
            image_inputs = []
            for msg in messages:
                for content in msg.get("content", []):
                    if content.get("type") == "image":
                        image_inputs.append(content["image"].replace("file://", ""))
            return image_inputs, None, {}

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": prompt, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def qwen_worker(
    worker_idx: int,
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    model_path: str,
    config: dict[str, Any],
    log_dir: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    torch.cuda.set_device(0)
    err_log = Path(log_dir) / f"worker_{worker_idx}_gpu{gpu_id}_error.txt"
    err_log.parent.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        trust_remote_code=True,
        max_model_len=int(config["max_model_len"]),
        limit_mm_per_prompt={"image": 1},
        seed=42,
    )
    sampling_params = SamplingParams(
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        max_tokens=int(config["max_new_tokens"]),
        stop_token_ids=[],
    )
    while True:
        task = task_queue.get()
        if task is None:
            break
        t0 = time.perf_counter()
        try:
            messages = _build_vllm_messages(
                image_path=str(task["model_image_path"]),
                prompt=str(task["model_input_prompt"]),
                min_pixels=task.get("min_pixels"),
                max_pixels=task.get("max_pixels"),
            )
            outputs = llm.generate([_prepare_inputs_for_vllm(messages, processor)], sampling_params=sampling_params, use_tqdm=False)
            text = outputs[0].outputs[0].text.replace("<think>", "").replace("</think>", "").strip()
            row = dict(task)
            row.update(
                {
                    "model_response": text,
                    "api_mode": "qwen3vl_local_vllm_ablation2_roi_agent",
                    "worker": worker_idx,
                    "gpu_id": gpu_id,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                }
            )
            result_queue.put({"status": "success", "row": row})
        except Exception as exc:
            with err_log.open("a", encoding="utf-8") as ef:
                ef.write(f"task_id={task.get('task_id')}\n{type(exc).__name__}: {exc}\n{traceback.format_exc()}\n{'=' * 80}\n")
            row = dict(task)
            row.update(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "api_mode": "qwen3vl_local_vllm_ablation2_roi_agent",
                    "worker": worker_idx,
                    "gpu_id": gpu_id,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                }
            )
            result_queue.put({"status": "error", "row": row})
    del llm
    torch.cuda.empty_cache()


def run_qwen_tasks(
    tasks: list[dict[str, Any]],
    out_path: Path,
    id_key: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    num_gpus: int = 8,
    gpu_ids_arg: str = "",
    max_model_len: int = 128000,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    gpu_memory_utilization: float = 0.85,
    resume: bool = True,
) -> dict[str, Any]:
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = out_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    done = done_ids(out_path, id_key) if resume else set()
    todo = [t for t in tasks if str(t.get(id_key) or "") not in done]
    if gpu_ids_arg.strip():
        gpu_ids = [int(x.strip()) for x in gpu_ids_arg.split(",") if x.strip()]
    else:
        gpu_ids = list(range(min(int(num_gpus), int(torch.cuda.device_count()))))
    if not gpu_ids:
        raise RuntimeError("No GPU available for Qwen inference")
    gpu_ids = gpu_ids[: int(num_gpus)]
    config = {
        "max_model_len": int(max_model_len),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "gpu_memory_utilization": float(gpu_memory_utilization),
    }
    run_config = {
        "model_path": str(model_path.resolve()),
        "out": str(out_path.resolve()),
        "id_key": id_key,
        "total_tasks_loaded": len(tasks),
        "skipped_by_resume": len(tasks) - len(todo),
        "tasks_to_run": len(todo),
        "gpu_ids": gpu_ids,
        "config": config,
    }
    out_path.with_name(out_path.stem + "_run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)
    if not todo:
        summary = {"processed": 0, "ok": 0, "error": 0, "output": str(out_path), "skipped_by_resume": len(tasks)}
        out_path.with_name(out_path.stem + "_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    task_queue: Queue = Queue()
    result_queue: Queue = Queue()
    processes: list[Process] = []
    for worker_idx, gpu_id in enumerate(gpu_ids):
        p = Process(
            target=qwen_worker,
            args=(worker_idx, gpu_id, task_queue, result_queue, str(model_path.resolve()), config, str(logs_dir)),
        )
        p.start()
        processes.append(p)
        print(f"[main] worker={worker_idx} gpu={gpu_id} pid={p.pid}", flush=True)
    for task in todo:
        task_queue.put(task)
    for _ in processes:
        task_queue.put(None)
    mode = "a" if resume else "w"
    ok = err = received = 0
    t0 = time.perf_counter()
    last_print = time.time()
    with out_path.open(mode, encoding="utf-8") as fout:
        while received < len(todo):
            try:
                item = result_queue.get(timeout=2)
            except Exception:
                if time.time() - last_print >= 15:
                    print(f"[main] progress {received}/{len(todo)} ({received * 100.0 / len(todo):.1f}%)", flush=True)
                    last_print = time.time()
                continue
            fout.write(json.dumps(item["row"], ensure_ascii=False) + "\n")
            fout.flush()
            received += 1
            ok += 1 if item.get("status") == "success" else 0
            err += 1 if item.get("status") != "success" else 0
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)
    task_queue.close()
    result_queue.close()
    task_queue.join_thread()
    result_queue.cancel_join_thread()
    elapsed = time.perf_counter() - t0
    summary = {
        "processed": len(todo),
        "ok": ok,
        "error": err,
        "total_loaded": len(tasks),
        "skipped_by_resume": len(tasks) - len(todo),
        "elapsed_sec": round(elapsed, 2),
        "throughput_task_per_sec": round(len(todo) / elapsed, 4) if elapsed > 0 else None,
        "output": str(out_path),
        "logs_dir": str(logs_dir),
    }
    out_path.with_name(out_path.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def counter_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def merge_error_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for row in rows:
        if row.get("error"):
            c[str(row.get("error"))] += 1
    return dict(sorted(c.items()))
