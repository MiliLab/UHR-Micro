from __future__ import annotations

import ast
import re
from typing import Any

from .geometry import Box

_WRAPPERS = (
    re.compile(r"^\s*(?:final\s+answer|answer)\s*:\s*(.*?)\s*$", re.I | re.S),
    re.compile(r"^\s*<(final|answer)>\s*(.*?)\s*</\1>\s*$", re.I | re.S),
)
_INTEGER = re.compile(r"^[0-9]+$")
_CHOICE = re.compile(r"^\(?\s*([ABCD])\s*\)?\.?$", re.I)


def unwrap_final(text: str | None) -> tuple[str | None, str]:
    if text is None or not str(text).strip():
        return None, "empty"
    value = str(text).strip()
    wrapper_hits = 0
    for pattern in _WRAPPERS:
        match = pattern.fullmatch(value)
        if match:
            wrapper_hits += 1
            value = match.group(match.lastindex or 1).strip()
    if wrapper_hits > 1 or re.search(r"(?:final\s+answer|answer)\s*:", value, re.I):
        return None, "ambiguous_wrapper"
    return value, "plain" if wrapper_hits == 0 else "explicit_wrapper"


def parse_count(text: str | None) -> tuple[int | None, str, dict[str, Any]]:
    value, wrapper = unwrap_final(text)
    if value is None:
        return None, "syntax_failure", {"reason": wrapper}
    if not _INTEGER.fullmatch(value):
        return None, "syntax_failure", {"reason": "expected_nonnegative_integer", "wrapper": wrapper}
    return int(value), "ok", {"wrapper": wrapper}


def parse_choice(text: str | None) -> tuple[str | None, str, dict[str, Any]]:
    value, wrapper = unwrap_final(text)
    if value is None:
        return None, "syntax_failure", {"reason": wrapper}
    match = _CHOICE.fullmatch(value)
    if not match:
        return None, "syntax_failure", {"reason": "expected_one_choice_A_to_D", "wrapper": wrapper}
    return match.group(1).upper(), "ok", {"wrapper": wrapper}


def _coerce_box(value: Any) -> Box | None:
    if not isinstance(value, (list, tuple)) or len(value) not in (4, 8):
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        return None
    return [float(v) for v in value]


def parse_boxes(text: str | None) -> tuple[list[Box], str, dict[str, Any]]:
    value, wrapper = unwrap_final(text)
    if value is None:
        return [], "syntax_failure", {"reason": wrapper}
    if not (value.startswith("[") and value.endswith("]")):
        return [], "syntax_failure", {"reason": "boxes_must_be_bracketed", "wrapper": wrapper}
    candidates = [value]
    if re.search(r"\]\s*,\s*\[", value) and not value.startswith("[["):
        candidates.insert(0, f"[{value}]")
    parsed: Any = None
    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
            break
        except (SyntaxError, ValueError):
            continue
    if parsed is None:
        return [], "syntax_failure", {"reason": "invalid_box_literal", "wrapper": wrapper}
    one = _coerce_box(parsed)
    if one is not None:
        return [one], "ok", {"wrapper": wrapper}
    if not isinstance(parsed, (list, tuple)):
        return [], "syntax_failure", {"reason": "expected_box_or_box_list", "wrapper": wrapper}
    boxes: list[Box] = []
    for item in parsed:
        box = _coerce_box(item)
        if box is None:
            return [], "syntax_failure", {"reason": "mixed_or_bad_box_list", "wrapper": wrapper}
        boxes.append(box)
    return boxes, "ok", {"wrapper": wrapper}
