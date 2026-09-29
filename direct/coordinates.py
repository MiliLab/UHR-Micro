from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .constants import COORDINATE_INPUT_TASKS


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
NUMERIC_LIST_RE = re.compile(
    rf"\[\s*{NUMBER}(?:\s*,\s*{NUMBER})+\s*\]"
)
NUMBER_RE = re.compile(NUMBER)


def _normalized(value: Decimal, extent: int, protocol: str) -> str:
    if protocol == "base1000":
        base, quantum = Decimal(1000), Decimal("0.01")
    elif protocol == "base100":
        # GeoChat is trained on a discrete 100 x 100 integer grid.
        base, quantum = Decimal(100), Decimal("1")
    elif protocol == "base1":
        # Four decimals keep question-coordinate quantization far below one
        # source pixel for ordinary UHR-Bench canvases.
        base, quantum = Decimal(1), Decimal("0.0001")
    else:
        raise ValueError(f"unsupported normalized coordinate protocol: {protocol}")
    normalized = (value * base / Decimal(extent)).quantize(
        quantum, rounding=ROUND_HALF_UP
    )
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def normalize_question_coordinates_to_protocol(
    question: str,
    task_key: str,
    width_px: int,
    height_px: int,
    protocol: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert recognized spatial lists to a declared full-source grid.

    Source values are retained verbatim for audit, while the model-facing
    values are clipped to the source canvas before normalization.  This keeps
    every coordinate inside the protocol's declared closed range.
    """
    if protocol not in {"base1000", "base100", "base1"}:
        raise ValueError(f"unsupported normalized coordinate protocol: {protocol}")
    matches = list(NUMERIC_LIST_RE.finditer(question))
    if matches and task_key not in COORDINATE_INPUT_TASKS:
        raise ValueError(
            f"unrouted numeric list in {task_key}; refusing broad numeric rewrite"
        )
    if task_key not in COORDINATE_INPUT_TASKS or not matches:
        return question, []

    pieces: list[str] = []
    transforms: list[dict[str, Any]] = []
    cursor = 0
    for match in matches:
        source_text = match.group(0)
        source_values = [Decimal(item) for item in NUMBER_RE.findall(source_text)]
        if len(source_values) < 2 or len(source_values) % 2:
            raise ValueError(f"invalid coordinate span: {source_text}")
        extents = [
            Decimal(width_px if index % 2 == 0 else height_px)
            for index in range(len(source_values))
        ]
        clipped_source_values = [
            min(max(value, Decimal(0)), extent)
            for value, extent in zip(source_values, extents)
        ]
        unclipped_model_values = [
            _normalized(value, int(extent), protocol)
            for value, extent in zip(source_values, extents)
        ]
        model_values = [
            _normalized(value, int(extent), protocol)
            for value, extent in zip(clipped_source_values, extents)
        ]
        clipped_indices = [
            index
            for index, (source, clipped) in enumerate(
                zip(source_values, clipped_source_values)
            )
            if source != clipped
        ]
        replacement = "[" + ", ".join(model_values) + "]"
        pieces.extend((question[cursor : match.start()], replacement))
        transforms.append(
            {
                "source_char_start": match.start(),
                "source_char_end": match.end(),
                "source_text": source_text,
                "source_values_absolute_px": [str(value) for value in source_values],
                "source_values_clipped_absolute_px": [
                    str(value) for value in clipped_source_values
                ],
                "model_text": replacement,
                f"model_values_{protocol}": model_values,
                f"unclipped_model_values_{protocol}": unclipped_model_values,
                "source_canvas": {"width_px": width_px, "height_px": height_px},
                "clipped": bool(clipped_indices),
                "clipped_indices": clipped_indices,
                "clip_policy": f"clamp_source_coordinates_to_canvas_before_{protocol}",
                "rounding": {
                    "base1000": "two_decimal_places_half_up_strip_trailing_zeros",
                    "base100": "integer_half_up",
                    "base1": "four_decimal_places_half_up_strip_trailing_zeros",
                }[protocol],
                "normalized_protocol": protocol,
            }
        )
        cursor = match.end()
    pieces.append(question[cursor:])
    return "".join(pieces), transforms


def normalize_question_coordinates(
    question: str,
    task_key: str,
    width_px: int,
    height_px: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Backward-compatible base1000 wrapper for the existing frozen adapters."""
    return normalize_question_coordinates_to_protocol(
        question, task_key, width_px, height_px, "base1000"
    )


def transform_question_coordinates_to_model_input(
    question: str,
    task_key: str,
    source_width_px: int,
    source_height_px: int,
    model_width_px: int,
    model_height_px: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Map source-pixel spatial lists to the resized Qwen model-input canvas."""
    matches = list(NUMERIC_LIST_RE.finditer(question))
    if matches and task_key not in COORDINATE_INPUT_TASKS:
        raise ValueError(f"unrouted numeric list in {task_key}; refusing broad numeric rewrite")
    if task_key not in COORDINATE_INPUT_TASKS or not matches:
        return question, []

    def mapped_text(value: Decimal, source_extent: int, model_extent: int) -> str:
        mapped = (value * Decimal(model_extent) / Decimal(source_extent)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        text = format(mapped, "f").rstrip("0").rstrip(".")
        return "0" if text in {"", "-0"} else text

    pieces: list[str] = []
    transforms: list[dict[str, Any]] = []
    cursor = 0
    for match in matches:
        source_text = match.group(0)
        source_values = [Decimal(item) for item in NUMBER_RE.findall(source_text)]
        if len(source_values) < 2 or len(source_values) % 2:
            raise ValueError(f"invalid coordinate span: {source_text}")
        source_extents = [
            Decimal(source_width_px if index % 2 == 0 else source_height_px)
            for index in range(len(source_values))
        ]
        model_extents = [
            model_width_px if index % 2 == 0 else model_height_px
            for index in range(len(source_values))
        ]
        clipped = [min(max(value, Decimal(0)), extent) for value, extent in zip(source_values, source_extents)]
        model_values = [
            mapped_text(value, int(source_extent), int(model_extent))
            for value, source_extent, model_extent in zip(clipped, source_extents, model_extents)
        ]
        unclipped_model_values = [
            mapped_text(value, int(source_extent), int(model_extent))
            for value, source_extent, model_extent in zip(source_values, source_extents, model_extents)
        ]
        clipped_indices = [i for i, (left, right) in enumerate(zip(source_values, clipped)) if left != right]
        replacement = "[" + ", ".join(model_values) + "]"
        pieces.extend((question[cursor:match.start()], replacement))
        transforms.append({
            "source_char_start": match.start(), "source_char_end": match.end(),
            "source_text": source_text,
            "source_values_absolute_px": [str(value) for value in source_values],
            "source_values_clipped_absolute_px": [str(value) for value in clipped],
            "model_text": replacement,
            "model_values_absolute_input_px": model_values,
            "unclipped_model_values_absolute_input_px": unclipped_model_values,
            "source_canvas": {"width_px": source_width_px, "height_px": source_height_px},
            "model_input_canvas": {"width_px": model_width_px, "height_px": model_height_px},
            "source_to_model_input": {
                "scale_x": model_width_px / source_width_px,
                "scale_y": model_height_px / source_height_px,
                "offset_x": 0.0, "offset_y": 0.0,
            },
            "clipped": bool(clipped_indices), "clipped_indices": clipped_indices,
            "clip_policy": "clamp_source_coordinates_to_canvas_before_model_input_scaling",
            "rounding": "two_decimal_places_half_up_strip_trailing_zeros",
        })
        cursor = match.end()
    pieces.append(question[cursor:])
    return "".join(pieces), transforms


def audit_transform_roundtrip(transform: dict[str, Any]) -> float:
    width = int(transform["source_canvas"]["width_px"])
    height = int(transform["source_canvas"]["height_px"])
    source = [
        Decimal(item)
        for item in transform.get(
            "source_values_clipped_absolute_px",
            transform["source_values_absolute_px"],
        )
    ]
    protocol = str(transform.get("normalized_protocol") or "base1000")
    base = {"base1000": Decimal(1000), "base100": Decimal(100), "base1": Decimal(1)}[protocol]
    model = [Decimal(item) for item in transform[f"model_values_{protocol}"]]
    errors = []
    for index, (src, normalized) in enumerate(zip(source, model)):
        extent = width if index % 2 == 0 else height
        reconstructed = normalized * Decimal(extent) / base
        errors.append(abs(float(reconstructed - src)))
    return max(errors, default=0.0)
