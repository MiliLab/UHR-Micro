from __future__ import annotations

import ast
import re
from typing import Any

from .geometry import Box
from .parsing import parse_boxes as parse_boxes_v1
from .parsing import parse_choice as parse_choice_v1
from .parsing import parse_count as parse_count_v1


# A marker is explicit only when it says "final answer", "the answer is", or
# uses an answer label followed by ':'/'='.  Plain prose containing the word
# "answer" is deliberately excluded.
_EXPLICIT_ANSWER_MARKER = re.compile(
    r"(?ix)"
    r"(?:"
    r"(?:the\s+)?final\s+answer\s*(?:is\s*)?(?::|=)?"
    r"|the\s+answer\s+is\s*(?::|=)?"
    r"|answer\s*(?::|=)"
    r")\s*"
)
_TAG = re.compile(r"</?(?:think|final|answer)>", re.I)
_ANSWER_BLOCK = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.I | re.S)
_ANY_TAG = re.compile(r"<[^>]+>")
_COUNT_PAYLOAD = re.compile(r"^([0-9]+)\s*[.!]?$", re.S)
_COUNT_TOKEN = re.compile(r"(?<![\w.])([0-9]+)(?!\w|\.[0-9])")
_CHOICE_PAYLOAD = re.compile(
    r"^(?:option\s*)?\(?([ABCD])\)?(?=\s|[.:\-]|$)"
    r"(?:\s*[.:\-]\s*.*|\s+.*)?\s*$",
    re.I | re.S,
)
_SINGLE_LINE_CHOICE = re.compile(
    r"^\(?([ABCD])\)?\s*[.:\-]\s*\S.*$", re.I | re.S
)
_CHOICE_TOKEN = re.compile(r"(?<![A-Za-z0-9])([ABCD])(?![A-Za-z0-9])", re.I)


def _marker_suffixes(text: str) -> list[str]:
    return [text[match.end() :] for match in _EXPLICIT_ANSWER_MARKER.finditer(text)]


def _last_complete_answer_block(text: str) -> str | None:
    """Return the last complete native answer payload, never a tool payload."""
    blocks = list(_ANSWER_BLOCK.finditer(text))
    if not blocks:
        return None
    payload = blocks[-1].group(1)
    if re.search(r"<tool_call\b|\btool_call\b", payload, re.I):
        return None
    return _ANY_TAG.sub(" ", payload).strip()


def _first_nonempty_line(text: str) -> str:
    for raw_line in text.splitlines():
        line = _TAG.sub("", raw_line).strip()
        if line:
            return line
    return ""


def _balanced_list(text: str) -> str | None:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_explicit_boxes(literal: str) -> tuple[list[Box], str, dict[str, Any]]:
    """Parse a bare box list or the native [{"bbox_2d": ...}] wrapper."""
    boxes, status, diagnostics = parse_boxes_v1(literal)
    if status == "ok":
        return boxes, status, diagnostics
    try:
        payload = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return [], status, diagnostics
    records = [payload] if isinstance(payload, dict) else payload
    if not isinstance(records, (list, tuple)) or not records:
        return [], status, diagnostics
    extracted: list[Box] = []
    for record in records:
        if not isinstance(record, dict) or "bbox_2d" not in record:
            return [], status, diagnostics
        value = record["bbox_2d"]
        if not isinstance(value, (list, tuple)) or len(value) not in (4, 8):
            return [], status, diagnostics
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            return [], status, diagnostics
        extracted.append([float(item) for item in value])
    return extracted, "ok", {"wrapper": "explicit_bbox_2d_object_list"}


def parse_count(text: str | None) -> tuple[int | None, str, dict[str, Any]]:
    old_value, old_status, old_diag = parse_count_v1(text)
    if old_status == "ok":
        return old_value, old_status, {**old_diag, "parser_path": "strict_v1"}
    if text is None or not str(text).strip():
        return None, "syntax_failure", {"reason": "empty", "parser_path": "v2_fallback"}
    value = str(text)
    answer_payload = _last_complete_answer_block(value)
    if answer_payload is not None:
        tokens = _COUNT_TOKEN.findall(answer_payload)
        if len(tokens) == 1:
            return int(tokens[0]), "ok", {
                "wrapper": "complete_answer_tag",
                "parser_path": "v2_fallback",
                "selection_policy": "last_complete_answer_tag_unique_integer",
            }
    suffixes = _marker_suffixes(value)
    if not suffixes:
        return None, "syntax_failure", {
            "reason": "no_explicit_final_count", "parser_path": "v2_fallback"
        }
    payload = _first_nonempty_line(suffixes[-1])
    match = _COUNT_PAYLOAD.fullmatch(payload)
    if not match:
        return None, "syntax_failure", {
            "reason": "last_explicit_count_invalid",
            "parser_path": "v2_fallback",
            "explicit_candidate_count": len(suffixes),
        }
    return int(match.group(1)), "ok", {
        "wrapper": "last_explicit_answer",
        "parser_path": "v2_fallback",
        "explicit_candidate_count": len(suffixes),
        "selection_policy": "last_explicit_answer",
    }


def parse_choice(text: str | None) -> tuple[str | None, str, dict[str, Any]]:
    old_value, old_status, old_diag = parse_choice_v1(text)
    if old_status == "ok":
        return old_value, old_status, {**old_diag, "parser_path": "strict_v1"}
    if text is None or not str(text).strip():
        return None, "syntax_failure", {"reason": "empty", "parser_path": "v2_fallback"}
    value = str(text)
    answer_payload = _last_complete_answer_block(value)
    if answer_payload is not None:
        tokens = [token.upper() for token in _CHOICE_TOKEN.findall(answer_payload)]
        if len(tokens) == 1:
            return tokens[0], "ok", {
                "wrapper": "complete_answer_tag",
                "parser_path": "v2_fallback",
                "selection_policy": "last_complete_answer_tag_unique_choice",
            }
    suffixes = _marker_suffixes(value)
    if suffixes:
        payload = _first_nonempty_line(suffixes[-1])
        match = _CHOICE_PAYLOAD.fullmatch(payload)
        if match:
            return match.group(1).upper(), "ok", {
                "wrapper": "last_explicit_answer",
                "parser_path": "v2_fallback",
                "explicit_candidate_count": len(suffixes),
                "selection_policy": "last_explicit_answer",
            }
        return None, "syntax_failure", {
            "reason": "last_explicit_choice_invalid",
            "parser_path": "v2_fallback",
            "explicit_candidate_count": len(suffixes),
        }
    lines = [
        _TAG.sub("", line).strip()
        for line in value.splitlines()
        if _TAG.sub("", line).strip()
    ]
    if len(lines) == 1:
        match = _SINGLE_LINE_CHOICE.fullmatch(lines[0])
        if match:
            return match.group(1).upper(), "ok", {
                "wrapper": "single_line_option_with_label",
                "parser_path": "v2_fallback",
                "selection_policy": "whole_response_single_line_option",
            }
    return None, "syntax_failure", {
        "reason": "no_unique_explicit_final_choice", "parser_path": "v2_fallback"
    }


def parse_boxes(text: str | None) -> tuple[list[Box], str, dict[str, Any]]:
    old_boxes, old_status, old_diag = parse_boxes_v1(text)
    if old_status == "ok":
        return old_boxes, old_status, {**old_diag, "parser_path": "strict_v1"}
    if text is None or not str(text).strip():
        return [], "syntax_failure", {"reason": "empty", "parser_path": "v2_fallback"}
    value = str(text)
    answer_payload = _last_complete_answer_block(value)
    if answer_payload is not None:
        literal = _balanced_list(answer_payload)
        if literal is not None:
            boxes, status, diagnostics = _parse_explicit_boxes(literal)
            if status == "ok":
                return boxes, "ok", {
                    "wrapper": "complete_answer_tag_balanced_list",
                    "parser_path": "v2_fallback",
                    "selection_policy": "last_complete_answer_tag_first_complete_balanced_list",
                }
    suffixes = _marker_suffixes(value)
    if not suffixes:
        return [], "syntax_failure", {
            "reason": "no_explicit_final_boxes", "parser_path": "v2_fallback"
        }
    literal = _balanced_list(suffixes[-1])
    if literal is None:
        return [], "syntax_failure", {
            "reason": "last_explicit_boxes_unbalanced_or_absent",
            "parser_path": "v2_fallback",
            "explicit_candidate_count": len(suffixes),
        }
    boxes, status, diagnostics = _parse_explicit_boxes(literal)
    if status != "ok":
        return [], status, {
            **diagnostics,
            "reason": "last_explicit_boxes_invalid",
            "parser_path": "v2_fallback",
            "explicit_candidate_count": len(suffixes),
        }
    return boxes, "ok", {
        "wrapper": "last_explicit_answer_balanced_list",
        "parser_path": "v2_fallback",
        "explicit_candidate_count": len(suffixes),
        "selection_policy": "last_explicit_answer_first_complete_balanced_list",
    }
