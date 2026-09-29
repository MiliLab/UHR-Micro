from __future__ import annotations

from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parent
SPLITS = ("Development", "Validation", "Test")
MODEL_REGISTRY_ID = "qwen3_vl_8b"
INPUT_SCHEMA_VERSION = "uhr-bench-v4-inference-input-1"
PREDICTION_SCHEMA_VERSION = "uhr-bench-v4-prediction-1"
RUN_SCHEMA_VERSION = "uhr-bench-v4-run-1"

TASK_TO_PREDICTION_TYPE = {
    "Grounding/GlobalDetection": "multi_bbox",
    "Grounding/RegionDetection": "multi_bbox",
    "Grounding/BasicGrounding": "single_bbox",
    "Grounding/ComplexGrounding": "single_bbox",
    "Grounding/MultiConditionRetrieval": "multi_bbox",
    "Understanding/Classification": "choice",
    "Understanding/Fine-grainedRecognition": "choice",
    "Understanding/ReferSegmentation": "segmentation_bbox",
    "Understanding/ComponentSegmentation": "segmentation_bbox",
    "Count/GlobalCount": "count",
    "Count/RegionCount": "count",
    "Count/ConditionalCount": "count",
    "Count/CrossRegionCompare": "count",
    "Relationship/RelationshipRecognition": "choice",
    "Relationship/distributionUnderstanding": "choice",
    "Relationship/NearestFarthest": "choice",
}

# Only numeric list spans in these tasks are spatial input coordinates.  Other
# numerals (class names, counts, option labels, and format placeholders) must
# never be rewritten.
COORDINATE_INPUT_TASKS = frozenset(
    {
        "Count/RegionCount",
        "Count/CrossRegionCompare",
        "Grounding/RegionDetection",
        "Grounding/ComplexGrounding",
        "Understanding/Classification",
        "Understanding/ComponentSegmentation",
        "Understanding/Fine-grainedRecognition",
    }
)

FORBIDDEN_GT_KEYS = frozenset(
    {
        "answer",
        "answer_ref",
        "answer_reference",
        "gt",
        "gt_answer",
        "ground_truth",
        "reference_answer",
    }
)

