from __future__ import annotations

from typing import Any

from .constants import TASK_TO_PREDICTION_TYPE
from .io import sha256_text


COORDINATE_INSTRUCTION = (
    "Spatial coordinate protocol: base1000 relative to the full source image; "
    "the top-left is (0, 0) and the bottom-right is (1000, 1000). "
    "Any numeric spatial coordinates in the question already use this protocol. "
    "If the requested final answer contains spatial coordinates, use the same protocol."
)


def build_prompt(
    model_question: str,
    task_key: str,
    prompt_contract: dict[str, Any],
    has_input_coordinates: bool = False,
) -> tuple[str, str, str]:
    prediction_type = TASK_TO_PREDICTION_TYPE[task_key]
    suffix = str(prompt_contract[prediction_type]).strip()
    common = str(prompt_contract["common"]).strip()
    parts = [
        model_question.strip(),
        "Output contract:",
        common,
        suffix,
    ]
    if has_input_coordinates or prediction_type in {"single_bbox", "multi_bbox", "segmentation_bbox"}:
        parts.append(COORDINATE_INSTRUCTION)
    parts.append("Return only the requested final field, with no explanation or reasoning.")
    prompt = "\n\n".join(parts)
    prompt_id = f"qwen3-vl-base1000-v2:{prediction_type}"
    return prompt_id, prompt, sha256_text(prompt)
