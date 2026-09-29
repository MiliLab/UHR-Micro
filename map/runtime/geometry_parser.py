#!/usr/bin/env python3
"""Geometry parsing and deterministic encoding conversion for MAP.

Two different boundaries are intentionally kept separate:

* terminal answers are parsed against one declared public output encoding;
* internal visual observations may natively report either an ``AABB4`` or a
  ``QUAD8`` geometry, with the observed encoding recorded per geometry.

Both boundaries accept only one explicit ``Final answer:`` field or a response
whose complete contents are unambiguously geometry (or ``null``).  Numbers
embedded in prose are deliberately never harvested.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import re
from typing import Any, Sequence


GEOMETRY_ENCODINGS = frozenset({"AABB4", "QUAD8"})
FINAL_LABEL_RE = re.compile(r"(?im)^[ \t]*final[ \t]+answer[ \t]*:[ \t]*")
LIST_RE = re.compile(r"\[([^\[\]]*)\]")
NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
NULL_RE = re.compile(r"^(?:null|none)\.?$", re.IGNORECASE)
NO_OBJECT_RE = re.compile(
    r"^(?:there\s+(?:are|is)\s+(?:none|no\s+(?:matching\s+)?"
    r"(?:objects?|targets?|instances?)(?:\s+(?:visible|detected|found))?)|"
    r"no\s+(?:matching\s+)?(?:objects?|targets?|instances?)\s+"
    r"(?:visible|detected|found))\.?$",
    re.IGNORECASE,
)
WHOLE_CODE_FENCE_RE = re.compile(
    r"^```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```$",
    re.IGNORECASE | re.DOTALL,
)
GEOMETRY_SEQUENCE_RE = re.compile(
    r"^\s*G(?:(?:\s*[,;]\s*|\s+)G)*\s*\.?\s*$"
)
NESTED_GEOMETRY_SEQUENCE_RE = re.compile(
    r"^\s*\[\s*G(?:\s*[,;]\s*G)*\s*\]\s*\.?\s*$"
)


def normalize_geometry_encoding(encoding: str) -> str:
    """Return a canonical public encoding without aliases or inference."""

    normalized = str(encoding or "").strip()
    if normalized not in GEOMETRY_ENCODINGS:
        raise ValueError(f"unsupported geometry encoding: {encoding!r}")
    return normalized


def _geometry_lists(payload: str) -> tuple[list[list[float]], str | None]:
    """Parse a geometry-only payload and reject every non-grammar token."""

    matches = list(LIST_RE.finditer(payload))
    if not matches:
        return [], "no_geometry_list"
    syntax_parts: list[str] = []
    cursor = 0
    output: list[list[float]] = []
    for match in matches:
        syntax_parts.append(payload[cursor : match.start()])
        syntax_parts.append("G")
        cursor = match.end()
        pieces = [piece.strip() for piece in match.group(1).split(",")]
        if not pieces or any(not piece for piece in pieces):
            return [], "malformed_coordinate_list"
        if any(NUMBER_RE.fullmatch(piece) is None for piece in pieces):
            return [], "non_numeric_coordinate"
        values = [float(piece) for piece in pieces]
        if not all(math.isfinite(value) for value in values):
            return [], "non_finite_coordinate"
        output.append(values)
    syntax_parts.append(payload[cursor:])
    syntax = "".join(syntax_parts)
    if (
        GEOMETRY_SEQUENCE_RE.fullmatch(syntax) is None
        and NESTED_GEOMETRY_SEQUENCE_RE.fullmatch(syntax) is None
    ):
        return [], "geometry_embedded_in_explanation"
    return output, None


def explicit_payload(text: str) -> tuple[str | None, str]:
    """Select an explicit terminal field or a fully geometry-native answer."""

    raw = str(text or "").strip()
    if not raw:
        return None, "empty_response"
    labels = list(FINAL_LABEL_RE.finditer(raw))
    if len(labels) > 1:
        return None, "ambiguous_multiple_final_fields"
    if labels:
        payload = raw[labels[0].end() :].strip()
        if not payload:
            return None, "empty_final_field"
        return payload, "final_field"
    if NULL_RE.fullmatch(raw):
        return raw, "native_null"
    parsed, error = _geometry_lists(raw)
    if error is not None or not parsed:
        return None, error or "no_explicit_geometry"
    return raw, "native_geometry_only"


def observation_payload(text: str) -> tuple[str | None, str]:
    """Return one complete native observation payload without prose scanning."""

    raw = str(text or "").strip()
    if not raw:
        return None, "empty_response"
    labels = list(FINAL_LABEL_RE.finditer(raw))
    if len(labels) > 1:
        return None, "ambiguous_multiple_final_fields"
    if labels:
        payload = raw[labels[0].end() :].strip()
        mode = "final_field"
    else:
        payload = raw
        mode = "native_observation"
    if not payload:
        return None, "empty_final_field" if labels else "empty_response"
    fence = WHOLE_CODE_FENCE_RE.fullmatch(payload)
    if fence is not None:
        payload = fence.group(1).strip()
        mode += "_code_fence"
    if "```" in payload:
        return None, "incomplete_or_embedded_code_fence"
    if NULL_RE.fullmatch(payload) or NO_OBJECT_RE.fullmatch(payload):
        return payload, mode + "_null"
    return payload, mode


def _bbox_dict(value: Any) -> tuple[list[float] | None, str | None]:
    if not isinstance(value, dict):
        return None, "json_geometry_item_not_object"
    if set(value) - {"bbox_2d", "label"} or "bbox_2d" not in value:
        return None, "json_geometry_object_has_unsupported_fields"
    if "label" in value and not isinstance(value["label"], str):
        return None, "json_geometry_label_not_string"
    raw = value["bbox_2d"]
    if not isinstance(raw, list) or not raw:
        return None, "json_bbox_2d_not_numeric_list"
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
        return None, "json_bbox_2d_not_numeric_list"
    output = [float(item) for item in raw]
    if not all(math.isfinite(item) for item in output):
        return None, "non_finite_coordinate"
    return output, None


def _json_geometry_lists(payload: str) -> tuple[list[list[float]] | None, str | None]:
    """Parse a complete JSON-native geometry answer, never embedded JSON."""

    if not payload.startswith(("[", "{")):
        return None, None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None, "invalid_or_truncated_json"
    if isinstance(value, dict):
        geometry, error = _bbox_dict(value)
        return ([] if geometry is None else [geometry]), error
    if not isinstance(value, list) or not value:
        return [], "json_geometry_array_empty_or_invalid"
    if all(
        not isinstance(item, bool) and isinstance(item, (int, float))
        for item in value
    ):
        numeric = [float(item) for item in value]
        return [numeric], None if all(math.isfinite(item) for item in numeric) else "non_finite_coordinate"
    if all(isinstance(item, list) for item in value):
        output: list[list[float]] = []
        for item in value:
            if not item or any(
                isinstance(number, bool) or not isinstance(number, (int, float))
                for number in item
            ):
                return [], "json_geometry_nested_item_not_numeric_list"
            numeric = [float(number) for number in item]
            if not all(math.isfinite(number) for number in numeric):
                return [], "non_finite_coordinate"
            output.append(numeric)
        return output, None
    if all(isinstance(item, dict) for item in value):
        output = []
        for item in value:
            geometry, error = _bbox_dict(item)
            if error is not None or geometry is None:
                return [], error or "invalid_json_bbox_2d"
            output.append(geometry)
        return output, None
    return [], "mixed_or_ambiguous_json_geometry_array"


def _rejected_geometry(
    index: int,
    reason: str,
    *,
    geometry: Sequence[float] | None = None,
    source: str,
) -> dict[str, Any]:
    """Build one auditable rejection without inventing replacement values."""

    return {
        "candidate_index": int(index),
        "reason": str(reason),
        "geometry": (
            [float(value) for value in geometry]
            if geometry is not None
            else None
        ),
        "source": str(source),
    }


def _plain_observation_candidates(
    payload: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Parse a complete list-only response while isolating bad list items.

    The grammar is validated before any coordinate is considered.  This is
    what prevents a numeric list embedded in prose from becoming evidence.
    Once the complete response is known to be a list sequence, each explicit
    list is decoded independently so that one malformed member need not erase
    other well-formed members of a geometry set.
    """

    matches = list(LIST_RE.finditer(payload))
    if not matches:
        return [], [], "no_geometry_list"
    syntax_parts: list[str] = []
    cursor = 0
    for match in matches:
        syntax_parts.append(payload[cursor : match.start()])
        syntax_parts.append("G")
        cursor = match.end()
    syntax_parts.append(payload[cursor:])
    syntax = "".join(syntax_parts)
    if (
        GEOMETRY_SEQUENCE_RE.fullmatch(syntax) is None
        and NESTED_GEOMETRY_SEQUENCE_RE.fullmatch(syntax) is None
    ):
        return [], [], "geometry_embedded_in_explanation"

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        pieces = [piece.strip() for piece in match.group(1).split(",")]
        if not pieces or any(not piece for piece in pieces):
            rejected.append(
                _rejected_geometry(
                    index,
                    "malformed_coordinate_list",
                    source="plain_coordinate_list",
                )
            )
            continue
        if any(NUMBER_RE.fullmatch(piece) is None for piece in pieces):
            rejected.append(
                _rejected_geometry(
                    index,
                    "non_numeric_coordinate",
                    source="plain_coordinate_list",
                )
            )
            continue
        values = [float(piece) for piece in pieces]
        if not all(math.isfinite(value) for value in values):
            rejected.append(
                _rejected_geometry(
                    index,
                    "non_finite_coordinate",
                    geometry=values,
                    source="plain_coordinate_list",
                )
            )
            continue
        candidates.append(
            {
                "candidate_index": index,
                "geometry": values,
                "source": "plain_coordinate_list",
            }
        )
    return candidates, rejected, None


def _json_observation_candidates(
    payload: str,
) -> tuple[
    list[dict[str, Any]] | None,
    list[dict[str, Any]],
    str | None,
]:
    """Decode only a complete JSON geometry value, item by item."""

    if not payload.startswith(("[", "{")):
        return None, [], None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return [], [], "invalid_or_truncated_json"

    if isinstance(value, dict):
        items: list[Any] = [value]
    elif isinstance(value, list):
        if not value:
            return [], [], "json_geometry_array_empty_or_invalid"
        if all(
            not isinstance(item, bool) and isinstance(item, (int, float))
            for item in value
        ):
            items = [value]
        else:
            items = list(value)
    else:
        return [], [], "json_geometry_root_not_array_or_object"

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        source = "json_numeric_list"
        geometry: list[float] | None = None
        error: str | None = None
        if isinstance(item, dict):
            source = "json_bbox_2d"
            geometry, error = _bbox_dict(item)
        elif isinstance(item, list):
            if not item or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                for number in item
            ):
                error = "json_geometry_item_not_numeric_list"
            else:
                geometry = [float(number) for number in item]
                if not all(math.isfinite(number) for number in geometry):
                    error = "non_finite_coordinate"
        else:
            error = "json_geometry_item_not_list_or_object"
        if error is not None or geometry is None:
            rejected.append(
                _rejected_geometry(
                    index,
                    error or "invalid_json_geometry_item",
                    geometry=geometry,
                    source=source,
                )
            )
            continue
        candidates.append(
            {
                "candidate_index": index,
                "geometry": geometry,
                "source": source,
            }
        )
    return candidates, rejected, None


def convex_quad(values: Sequence[float]) -> bool:
    """Require four distinct, ordered vertices of a strict convex polygon."""

    if len(values) != 8:
        return False
    points = [
        (float(values[index]), float(values[index + 1]))
        for index in range(0, 8, 2)
    ]
    if len(set(points)) != 4:
        return False
    crosses: list[float] = []
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        third = points[(index + 2) % 4]
        cross = (second[0] - first[0]) * (third[1] - second[1]) - (
            second[1] - first[1]
        ) * (third[0] - second[0])
        if math.isclose(cross, 0.0, rel_tol=0.0, abs_tol=1e-10):
            return False
        crosses.append(cross)
    if not (
        all(value > 0.0 for value in crosses)
        or all(value < 0.0 for value in crosses)
    ):
        return False
    doubled_area = sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )
    return not math.isclose(
        doubled_area, 0.0, rel_tol=0.0, abs_tol=1e-10
    )


def valid_in_coordinate_extent(
    values: Sequence[float],
    encoding: str,
    coordinate_extent: Sequence[float],
) -> bool:
    """Validate shape inside an explicit x/y extent without clamping."""

    normalized = normalize_geometry_encoding(encoding)
    try:
        max_x, max_y = (float(value) for value in coordinate_extent)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) and value > 0.0 for value in (max_x, max_y)):
        return False
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        return False
    if any(
        value < 0.0 or value > (max_x if index % 2 == 0 else max_y)
        for index, value in enumerate(numeric)
    ):
        return False
    if normalized == "AABB4":
        return (
            len(numeric) == 4
            and numeric[2] > numeric[0]
            and numeric[3] > numeric[1]
        )
    return len(numeric) == 8 and convex_quad(numeric)


def valid(values: Sequence[float], encoding: str) -> bool:
    """Validate shape and the public base1000 domain."""

    return valid_in_coordinate_extent(values, encoding, (1000.0, 1000.0))


def observed_geometry_encoding(
    values: Sequence[float],
    coordinate_extent: Sequence[float] = (1000.0, 1000.0),
) -> str | None:
    """Return the unambiguous native encoding implied by a valid value list."""

    if len(values) == 4 and valid_in_coordinate_extent(
        values, "AABB4", coordinate_extent
    ):
        return "AABB4"
    if len(values) == 8 and valid_in_coordinate_extent(
        values, "QUAD8", coordinate_extent
    ):
        return "QUAD8"
    return None


def _observation_geometry_rejection_reason(
    values: Sequence[float],
    *,
    accepted: set[str],
    coordinate_extent: Sequence[float],
) -> tuple[str | None, str | None]:
    """Return ``(encoding, error)`` under one fixed declared coordinate frame."""

    numeric = [float(value) for value in values]
    if len(numeric) not in (4, 8):
        return None, "unsupported_geometry_arity"
    if not all(math.isfinite(value) for value in numeric):
        return None, "non_finite_coordinate"
    max_x, max_y = (float(value) for value in coordinate_extent)
    if any(
        value < 0.0 or value > (max_x if index % 2 == 0 else max_y)
        for index, value in enumerate(numeric)
    ):
        return None, "coordinate_out_of_declared_extent"
    if len(numeric) == 4:
        encoding = "AABB4"
        if numeric[2] <= numeric[0] or numeric[3] <= numeric[1]:
            return None, "degenerate_aabb"
    else:
        encoding = "QUAD8"
        if not convex_quad(numeric):
            return None, "non_convex_or_degenerate_quad"
    if encoding not in accepted:
        return None, "observation_geometry_encoding_not_accepted"
    return encoding, None


def _add_rejection_metadata(
    result: dict[str, Any],
    rejected: Sequence[dict[str, Any]],
    *,
    candidate_count: int,
    valid_count: int,
) -> dict[str, Any]:
    """Attach stable counts and reasons to both success and failure records."""

    rejection_rows = [dict(row) for row in rejected]
    reason_counts = Counter(str(row["reason"]) for row in rejection_rows)
    result.update(
        {
            "candidate_geometry_count": int(candidate_count),
            "valid_geometry_count": int(valid_count),
            "rejected_geometry_count": len(rejection_rows),
            "invalid_geometry_count": len(rejection_rows),
            "rejected_geometries": rejection_rows,
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
            # Compatibility for downstream registration diagnostics.  Only
            # explicit numeric candidates appear here; no replacement or
            # clamped geometry is ever synthesized.
            "invalid": [
                list(row["geometry"])
                for row in rejection_rows
                if isinstance(row.get("geometry"), list)
            ],
        }
    )
    return result


def convert_geometry_encoding(
    values: Sequence[float], source_encoding: str, output_encoding: str
) -> list[float]:
    """Project one valid geometry to a declared output representation.

    Conversion is deterministic and parameter free.  AABB-to-QUAD emits the
    four rectangle corners in perimeter order.  QUAD-to-AABB emits the exact
    enclosing axis-aligned extent.  The input geometry is never overwritten;
    callers retain it together with this output-boundary projection.
    """

    source = normalize_geometry_encoding(source_encoding)
    target = normalize_geometry_encoding(output_encoding)
    numeric = [float(value) for value in values]
    if not valid(numeric, source):
        raise ValueError(f"invalid {source} geometry cannot be converted")
    if source == target:
        return numeric
    if source == "AABB4":
        x1, y1, x2, y2 = numeric
        converted = [x1, y1, x2, y1, x2, y2, x1, y2]
    else:
        xs = numeric[0::2]
        ys = numeric[1::2]
        converted = [min(xs), min(ys), max(xs), max(ys)]
    if not valid(converted, target):
        raise ValueError(f"deterministic {source}-to-{target} conversion failed")
    return converted


def _invalid(mode: str, error: str, *, encoding: str | None) -> dict[str, Any]:
    return {
        "valid": False,
        "geometries": [],
        "mode": mode,
        "error": error,
        "null": False,
        "geometry_encoding": encoding,
    }


def parse_geometry_response(
    text: str, encoding: str, *, allow_multiple: bool
) -> dict[str, Any]:
    """Parse geometry without clamping, aliases, length guesses, or prose scan."""

    try:
        normalized = normalize_geometry_encoding(encoding)
    except ValueError:
        return _invalid(
            "public_contract",
            "unsupported_public_geometry_encoding",
            encoding=None,
        )
    payload, mode = explicit_payload(text)
    if payload is None:
        return _invalid(mode, mode, encoding=normalized)
    if NULL_RE.fullmatch(payload.strip()):
        return {
            "valid": True,
            "geometries": [],
            "mode": mode,
            "error": None,
            "null": True,
            "geometry_encoding": normalized,
        }
    parsed, payload_error = _geometry_lists(payload)
    if payload_error is not None:
        return _invalid(mode, payload_error, encoding=normalized)
    if not allow_multiple and len(parsed) != 1:
        return _invalid(
            mode,
            "single_geometry_requires_exactly_one_list",
            encoding=normalized,
        )
    if not parsed:
        return _invalid(mode, "no_geometry_list", encoding=normalized)
    invalid = [values for values in parsed if not valid(values, normalized)]
    if invalid:
        result = _invalid(
            mode,
            "invalid_geometry_encoding_or_shape",
            encoding=normalized,
        )
        result["invalid"] = invalid
        return result
    return {
        "valid": True,
        "geometries": parsed,
        "geometry_records": [
            {"geometry": values, "geometry_encoding": normalized}
            for values in parsed
        ],
        "mode": mode,
        "error": None,
        "null": False,
        "geometry_encoding": normalized,
        "geometry_encodings": [normalized for _ in parsed],
    }


def parse_observation_geometry_response(
    text: str,
    *,
    allow_multiple: bool,
    accepted_encodings: Sequence[str] = ("AABB4", "QUAD8"),
    coordinate_extent: Sequence[float] = (1000.0, 1000.0),
) -> dict[str, Any]:
    """Parse an internal observation without forcing the terminal encoding.

    Encoding is inferred only from an explicit coordinate-list arity and then
    fully validated in the one coordinate extent declared by the caller.
    Four numbers are an AABB4 observation and eight numbers are a strict convex
    QUAD8 observation.  There is no value-based coordinate-protocol inference,
    clamping, or prose scan.

    A geometry set keeps every independently valid member of a complete
    structured response and records rejected members with exact reasons.  A
    non-null set fails only when no valid member remains.  A single-geometry
    interface is deliberately stricter: multiple valid members or a mixture
    of a valid member and a rejected member is a structural conflict.
    """

    try:
        accepted = {
            normalize_geometry_encoding(encoding) for encoding in accepted_encodings
        }
    except ValueError:
        return _invalid(
            "observation_contract",
            "unsupported_observation_geometry_encoding",
            encoding=None,
        )
    if not accepted:
        return _invalid(
            "observation_contract",
            "empty_observation_geometry_encoding_set",
            encoding=None,
        )
    try:
        extent = tuple(float(value) for value in coordinate_extent)
    except (TypeError, ValueError):
        return _invalid(
            "observation_contract",
            "invalid_observation_coordinate_extent",
            encoding=None,
        )
    if (
        len(extent) != 2
        or not all(math.isfinite(value) and value > 0.0 for value in extent)
    ):
        return _invalid(
            "observation_contract",
            "invalid_observation_coordinate_extent",
            encoding=None,
        )
    payload, mode = observation_payload(text)
    if payload is None:
        return _invalid(mode, mode, encoding=None)
    if NULL_RE.fullmatch(payload.strip()) or NO_OBJECT_RE.fullmatch(payload.strip()):
        return _add_rejection_metadata(
            {
                "valid": True,
                "geometries": [],
                "geometry_records": [],
                "mode": mode,
                "error": None,
                "null": True,
                "geometry_encoding": None,
                "geometry_encodings": [],
            },
            [],
            candidate_count=0,
            valid_count=0,
        )

    parsed_json, rejected, payload_error = _json_observation_candidates(payload)
    if parsed_json is None:
        candidates, rejected, payload_error = _plain_observation_candidates(
            payload
        )
    elif payload_error == "invalid_or_truncated_json":
        # ``[...], [...]`` is the documented native list-sequence grammar, not
        # a JSON root.  Accept that form only when the *entire* payload passes
        # the strict list-only grammar.  Otherwise retain the JSON error; this
        # deliberately prevents truncated JSON or JSON embedded in prose from
        # falling through to numeric harvesting.
        plain_candidates, plain_rejected, plain_error = (
            _plain_observation_candidates(payload)
        )
        if plain_error is None:
            candidates = plain_candidates
            rejected = plain_rejected
            payload_error = None
        else:
            candidates = parsed_json
    else:
        candidates = parsed_json
    if payload_error is not None:
        return _add_rejection_metadata(
            _invalid(mode, payload_error, encoding=None),
            rejected,
            candidate_count=len(candidates) + len(rejected),
            valid_count=0,
        )

    candidate_count = len(candidates) + len(rejected)
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        values = [float(value) for value in candidate["geometry"]]
        encoding, error = _observation_geometry_rejection_reason(
            values,
            accepted=accepted,
            coordinate_extent=extent,
        )
        if error is not None or encoding is None:
            rejected.append(
                _rejected_geometry(
                    int(candidate["candidate_index"]),
                    error or "invalid_observation_geometry",
                    geometry=values,
                    source=str(candidate["source"]),
                )
            )
            continue
        records.append(
            {
                "geometry": values,
                "geometry_encoding": encoding,
            }
        )

    if not allow_multiple and len(records) > 1:
        return _add_rejection_metadata(
            _invalid(
                mode,
                "single_geometry_multiple_valid_geometries",
                encoding=None,
            ),
            rejected,
            candidate_count=candidate_count,
            valid_count=len(records),
        )
    if not allow_multiple and records and rejected:
        return _add_rejection_metadata(
            _invalid(
                mode,
                "single_geometry_structure_conflict",
                encoding=None,
            ),
            rejected,
            candidate_count=candidate_count,
            valid_count=len(records),
        )
    if not records:
        return _add_rejection_metadata(
            _invalid(
                mode,
                "no_valid_observation_geometry",
                encoding=None,
            ),
            rejected,
            candidate_count=candidate_count,
            valid_count=0,
        )

    encodings = [str(record["geometry_encoding"]) for record in records]
    homogeneous = encodings[0] if len(set(encodings)) == 1 else "MIXED"
    return _add_rejection_metadata(
        {
            "valid": True,
            "geometries": [record["geometry"] for record in records],
            "geometry_records": records,
            "mode": mode,
            "error": None,
            "null": False,
            "geometry_encoding": homogeneous,
            "geometry_encodings": encodings,
        },
        rejected,
        candidate_count=candidate_count,
        valid_count=len(records),
    )


__all__ = [
    "GEOMETRY_ENCODINGS",
    "convex_quad",
    "convert_geometry_encoding",
    "explicit_payload",
    "normalize_geometry_encoding",
    "observed_geometry_encoding",
    "parse_observation_geometry_response",
    "parse_geometry_response",
    "valid",
    "valid_in_coordinate_extent",
]
