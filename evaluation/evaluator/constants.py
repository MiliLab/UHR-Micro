from __future__ import annotations

TASK_TYPES = {
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

SEGMENTATION_TASKS = {
    "Understanding/ReferSegmentation",
    "Understanding/ComponentSegmentation",
}
SINGLE_BOX_TASKS = {
    task for task, kind in TASK_TYPES.items() if kind in {"single_bbox", "segmentation_bbox"}
}
MULTI_BOX_TASKS = {task for task, kind in TASK_TYPES.items() if kind == "multi_bbox"}
COUNT_TASKS = {task for task, kind in TASK_TYPES.items() if kind == "count"}
CHOICE_TASKS = {task for task, kind in TASK_TYPES.items() if kind == "choice"}

RUNTIME_STATUSES = {
    "not_attempted",
    "success",
    "runtime_or_api_error",
    "tool_max_turns",
    "empty_response",
    "postprocess_failure",
}

CONCRETE_COORDINATE_PROTOCOLS = {
    "base1000",
    "absolute_model_input",
    "base100",
    "base1",
    "absolute_source",
}
