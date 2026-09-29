from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import bbox_diagonal, centroid, polygon_iou, scale_similarity


def box_score_s_v2(pred: Sequence[float], gt: Sequence[float]) -> float:
    overlap = polygon_iou(pred, gt)
    pc, gc = centroid(pred), centroid(gt)
    distance = math.hypot(pc[0] - gc[0], pc[1] - gc[1])
    center_similarity = max(0.0, 1.0 - distance / (2.0 * bbox_diagonal(gt)))
    shape_scale_similarity = scale_similarity(pred, gt)
    return float(overlap + 0.5 * (1.0 - overlap) * center_similarity * shape_scale_similarity)


def multi_box_score_v2(
    pred: list[list[float]], gt: list[list[float]], *, gate: str,
) -> dict[str, Any]:
    if gate not in {"iou_0.3", "score_s_0.3"}:
        raise ValueError(f"unsupported multi-box gate: {gate}")
    if not pred or not gt:
        return {"score": 0.0, "valid_matches": 0, "matched_quality_sum": 0.0, "pairs": []}
    quality = np.zeros((len(gt), len(pred)), dtype=np.float64)
    overlap = np.zeros_like(quality)
    for gi, gt_box in enumerate(gt):
        for pi, pred_box in enumerate(pred):
            overlap[gi, pi] = polygon_iou(pred_box, gt_box)
            quality[gi, pi] = box_score_s_v2(pred_box, gt_box)
    valid = overlap >= 0.3 if gate == "iou_0.3" else quality >= 0.3
    # First maximize the number of valid pairs, then total Score-S quality.
    weights = valid.astype(np.float64) * (2.0 + quality)
    rows, cols = linear_sum_assignment(-weights)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if valid[int(r), int(c)]]
    total = float(sum(quality[r, c] for r, c in pairs))
    return {
        "score": float(2.0 * total / (len(gt) + len(pred))),
        "valid_matches": len(pairs),
        "matched_quality_sum": total,
        "pairs": pairs,
    }


def relative_count_score(pred: int | None, gt: int) -> float:
    if pred is None:
        return 0.0
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    return float(max(0.0, 1.0 - abs(pred - gt) / float(gt)))


def mask_score_s_v2(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    if gt.shape != pred.shape:
        return 0.0, 0.0
    gt_bool, pred_bool = gt.astype(bool), pred.astype(bool)
    inter = float(np.logical_and(gt_bool, pred_bool).sum())
    union = float(np.logical_or(gt_bool, pred_bool).sum())
    iou = inter / union if union > 0 else 0.0
    gy, gx = np.nonzero(gt_bool)
    py, px = np.nonzero(pred_bool)
    if not len(gx) or not len(px):
        return float(iou), float(iou)
    gt_centroid = (float(gx.mean()), float(gy.mean()))
    pred_centroid = (float(px.mean()), float(py.mean()))
    gt_diag = max(math.hypot(float(gx.max() - gx.min() + 1), float(gy.max() - gy.min() + 1)), 1.0)
    center_similarity = max(0.0, 1.0 - math.hypot(
        gt_centroid[0] - pred_centroid[0], gt_centroid[1] - pred_centroid[1]
    ) / (2.0 * gt_diag))
    gt_area, pred_area = float(len(gx)), float(len(px))
    area_similarity = min(gt_area, pred_area) / max(gt_area, pred_area)
    score = iou + 0.5 * (1.0 - iou) * center_similarity * area_similarity
    return float(score), float(iou)
