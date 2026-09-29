"""UHR-Bench v4 reference evaluator (Parser V2 + Metric V2)."""

from .metrics import box_score_s_v2, mask_score_s_v2, multi_box_score_v2

__all__ = ["box_score_s_v2", "mask_score_s_v2", "multi_box_score_v2"]
