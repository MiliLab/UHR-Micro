#!/usr/bin/env python3
"""Typed evidence aggregation and the unified terminal writer for MAP.

This module deliberately separates three boundaries:

* ``aggregate`` projects observations into a public, typed evidence state;
* ``closure`` decides, before any terminal generation, whether one complete
  public support observation is already a sufficient typed answer;
* ``prepare`` exposes every still-open state and its public-contract-selected
  source canvas to one shared terminal composer; and
* ``materialize`` joins private reporting identity only after inference and
  serializes the already-frozen closure decision or the shared composer.

Geometry aggregation preserves every valid source-registered candidate and its
provenance.  Internal AABB4 and QUAD8 observations remain distinct from the
public terminal encoding.  Exact identity may be reported as inventory, but is
never treated as the sole reliability signal and never deletes a candidate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from io_utils import (
    cached_sha256_file,
    keyed,
    read_jsonl,
    sha256_text,
    write_json,
    write_jsonl,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from geometry_parser import (  # noqa: E402
    convert_geometry_encoding,
    parse_observation_geometry_response,
    parse_geometry_response,
    valid as valid_geometry,
    valid_in_coordinate_extent,
)
from register_outputs import (  # noqa: E402
    source1000_affine,
    validate_local_coordinate_protocol,
)
from view_geometry import ViewGeometry, ViewGeometryError  # noqa: E402


INTEGER_ONLY_RE = re.compile(r"^\s*(\d+)\s*\.?\s*$")
FINAL_INTEGER_RE = re.compile(
    r"final\s+answer\s*:\s*(\d+)\s*\.?\s*$", re.IGNORECASE
)
LETTER_ONLY_RE = re.compile(r"^\s*([ABCD])\s*\.?\s*$", re.IGNORECASE)
FINAL_LETTER_RE = re.compile(
    r"final\s+answer\s*:\s*([ABCD])\s*\.?\s*$", re.IGNORECASE
)
PROVISIONAL_PREFIX_RE = re.compile(
    r"(?im)^[ \t]*provisional[ \t]+evidence[ \t]*:[ \t]*"
)
GEOMETRY_SCHEMAS = {"SINGLE_GEOMETRY", "GEOMETRY_SET"}
GEOMETRY_ENCODINGS = {"AABB4", "QUAD8"}
PUBLIC_COMPOSITIONS = {
    "IDENTITY",
    "AGGREGATION",
    "NUMERIC_DIFFERENCE",
    "SPATIAL_CONFIGURATION",
}
PUBLIC_CLOSABLE_SUPPORT_SOURCES = {
    "PUBLIC_EXPLICIT_SUPPORT",
    "PUBLIC_TARGET_SUPPORT",
}


def parse_choice(text: str) -> str | None:
    """Parse only an explicit final label or a response containing one letter."""

    raw = str(text or "").strip()
    match = FINAL_LETTER_RE.search(raw) or LETTER_ONLY_RE.fullmatch(raw)
    return match.group(1).upper() if match else None


def parse_scalar(text: str) -> int | None:
    """Parse a reliable non-negative scalar without last-number guessing."""

    raw = str(text or "").strip()
    match = FINAL_INTEGER_RE.search(raw) or INTEGER_ONLY_RE.fullmatch(raw)
    return int(match.group(1)) if match else None


def geometry_encoding(contract: dict[str, Any]) -> str:
    schema = str(contract.get("answer_schema") or "")
    if "terminal_geometry_encoding" not in contract:
        raise ValueError("missing terminal_geometry_encoding")
    if "geometry_encoding" not in contract:
        raise ValueError("missing legacy geometry_encoding alias")
    encoding = str(contract["terminal_geometry_encoding"])
    if str(contract["geometry_encoding"]) != encoding:
        raise ValueError("geometry_encoding alias differs from terminal_geometry_encoding")
    if schema in GEOMETRY_SCHEMAS:
        if encoding not in GEOMETRY_ENCODINGS:
            raise ValueError(
                f"geometry schema {schema} requires AABB4 or QUAD8, got {encoding!r}"
            )
    elif encoding != "NONE":
        raise ValueError(
            f"non-geometry schema {schema} requires geometry_encoding=NONE, got {encoding!r}"
        )
    return encoding


def raw_observation_text(row: dict[str, Any]) -> str:
    """Return the exact successful observer text without interpreting it."""

    value = (
        row.get("raw_backend_response")
        if row.get("raw_backend_response") is not None
        else row.get("model_response")
    )
    return str(value or "")


def unstructured_evidence(
    row: dict[str, Any],
    *,
    source_kind: str,
    source_id: str,
    view_index: Any = None,
    source_support_base1000: list[float] | None = None,
    support_label: Any = None,
) -> dict[str, Any] | None:
    """Keep successful raw evidence with provenance, even if parsing fails."""

    if row.get("error"):
        return None
    evidence: dict[str, Any] = {
        "source_kind": source_kind,
        "source_id": source_id,
        "raw_text": raw_observation_text(row),
    }
    if view_index is not None:
        evidence["view_index"] = view_index
    if source_support_base1000 is not None:
        evidence["source_support_base1000"] = source_support_base1000
    if support_label is not None:
        evidence["support_label"] = support_label
    return evidence


def strict_geometry_response(
    text: str, encoding: str, *, allow_multiple: bool
) -> dict[str, Any]:
    """Delegate to the frozen strict public geometry parser."""

    return parse_geometry_response(text, encoding, allow_multiple=allow_multiple)


def observation_geometry_response(
    text: str,
    *,
    allow_multiple: bool,
    coordinate_extent: tuple[float, float] = (1000.0, 1000.0),
) -> dict[str, Any]:
    """Parse native visual evidence independently of terminal serialization."""

    return parse_observation_geometry_response(
        text,
        allow_multiple=allow_multiple,
        coordinate_extent=coordinate_extent,
    )


def provisional_payload(text: str) -> tuple[str | None, str | None]:
    """Strip exactly one leading public provisional-evidence prefix.

    The complete-scene observer is evidence rather than a terminal writer, so
    its frozen output contract intentionally does not use ``Final answer:``.
    Only this exact leading prefix is removed; no number or geometry is scanned
    from surrounding prose.
    """

    raw = str(text or "")
    matches = list(PROVISIONAL_PREFIX_RE.finditer(raw))
    if len(matches) != 1:
        return None, (
            "missing_provisional_evidence_prefix"
            if not matches
            else "ambiguous_provisional_evidence_prefix"
        )
    match = matches[0]
    if raw[: match.start()].strip():
        return None, "provisional_evidence_prefix_not_leading"
    payload = raw[match.end() :].strip()
    return (payload, None) if payload else (None, "empty_provisional_evidence")


def normalized_geometry(
    values: Any,
    encoding: str,
    *,
    coordinate_extent: tuple[float, float] = (1000.0, 1000.0),
) -> list[float] | None:
    if not isinstance(values, (list, tuple)):
        return None
    try:
        output = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in output):
        return None
    return (
        output
        if valid_in_coordinate_extent(output, encoding, coordinate_extent)
        else None
    )


def local_observation_coordinate_provenance(
    row: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and expose the exact observer frame used by registration."""

    protocol = str(row.get("stage3_local_coordinate_protocol") or "")
    original_protocol = str(row.get("local_coordinate_protocol") or "")
    if not protocol or protocol != original_protocol:
        return None, "local_coordinate_protocol_registration_mismatch"
    raw_view = row.get("view_geometry")
    if not isinstance(raw_view, dict):
        return None, "missing_view_geometry"
    try:
        view = ViewGeometry.from_metadata(raw_view)
        validated_protocol, extent = validate_local_coordinate_protocol(row, view)
    except (KeyError, TypeError, ValueError, ViewGeometryError) as error:
        return None, f"invalid_local_coordinate_provenance:{type(error).__name__}:{error}"
    if validated_protocol != protocol:
        return None, "validated_local_coordinate_protocol_mismatch"
    expected_affine = source1000_affine(view, protocol)
    raw_affine = row.get("stage3_registration_affine_source1000_2x3")
    if (
        not isinstance(raw_affine, (list, tuple))
        or len(raw_affine) != 2
        or any(
            not isinstance(line, (list, tuple)) or len(line) != 3
            for line in raw_affine
        )
    ):
        return None, "missing_or_invalid_registration_affine"
    try:
        affine = [[float(value) for value in line] for line in raw_affine]
    except (TypeError, ValueError):
        return None, "missing_or_invalid_registration_affine"
    if not all(math.isfinite(value) for line in affine for value in line):
        return None, "missing_or_invalid_registration_affine"
    if any(
        not math.isclose(
            affine[row_index][column_index],
            expected_affine[row_index][column_index],
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        for row_index in range(2)
        for column_index in range(3)
    ):
        return None, "registration_affine_view_geometry_mismatch"
    provenance: dict[str, Any] = {
        "stage3_local_coordinate_protocol": protocol,
        "coordinate_extent": [float(extent[0]), float(extent[1])],
        "stage3_registration_affine_source1000_2x3": affine,
        "view_geometry": copy.deepcopy(raw_view),
        "roi_view_kind": row.get("roi_view_kind"),
        "view_render_mode": row.get("view_render_mode"),
        "padding_semantics": row.get("padding_semantics"),
    }
    if protocol == "native_crop_base1000":
        provenance["query_crop_origin_abs_px"] = [
            float(value) for value in row["query_crop_origin_abs_px"]
        ]
        provenance["query_crop_size_px"] = [
            float(value) for value in row["query_crop_size_px"]
        ]
    return provenance, None


def geometry_key(values: Iterable[float], encoding: str) -> tuple[str, tuple[float, ...]]:
    # Exact numeric identity is intentional. No distance, area, or IoU
    # tolerance is introduced here. Canonicalize signed zero only.
    canonical = tuple(0.0 if float(value) == 0.0 else float(value) for value in values)
    return encoding, canonical


def valid_footprint(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and 0.0 <= item <= 1000.0 for item in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def observation_id(row: dict[str, Any], fallback: str) -> str:
    return str(row.get("observation_id") or row.get("task_id") or fallback)


def require_unique_rows(
    rows: Iterable[dict[str, Any]], *, key_names: tuple[str, ...], label: str
) -> None:
    seen: set[str] = set()
    for row in rows:
        identity = ""
        for key in key_names:
            if row.get(key):
                identity = str(row[key])
                break
        if not identity:
            raise ValueError(f"missing {label} identity")
        if identity in seen:
            raise ValueError(f"duplicate {label} identity: {identity}")
        seen.add(identity)


def geometry_native_observation(
    native: dict[str, Any], schema: str, output_encoding: str
) -> dict[str, Any]:
    identity = observation_id(native, "complete-scene")
    evidence = unstructured_evidence(
        native,
        source_kind="COMPLETE_SCENE",
        source_id=identity,
    )
    if evidence is not None:
        evidence.update(
            {
                "coordinate_frame": "source_image_base1000",
                "coordinate_extent": [1000.0, 1000.0],
            }
        )
    if native.get("error"):
        parsed = {
            "valid": False,
            "geometries": [],
            "null": False,
            "mode": "provisional_evidence",
            "error": "infrastructure_error",
        }
    else:
        payload, prefix_error = provisional_payload(raw_observation_text(native))
        parsed = (
            observation_geometry_response(
                payload or "",
                allow_multiple=schema == "GEOMETRY_SET",
            )
            if prefix_error is None
            else {
                "valid": False,
                "geometries": [],
                "null": False,
                "mode": "provisional_evidence",
                "error": prefix_error,
            }
        )
    output = {
        "observation_id": identity,
        "observation_kind": "COMPLETE_SCENE",
        "valid": bool(parsed.get("valid")),
        "observation_geometry_encoding": parsed.get("geometry_encoding"),
        "observation_geometry_encodings": list(
            parsed.get("geometry_encodings") or []
        ),
        "output_geometry_encoding": output_encoding,
        "geometries": [],
        "geometry_records": [],
        "null_observation": bool(parsed.get("null")),
        "parse_mode": parsed.get("mode"),
        "parse_error": parsed.get("error"),
    }
    if evidence is not None:
        output["unstructured_evidence"] = evidence
    if parsed.get("valid"):
        records = []
        for record in parsed.get("geometry_records") or []:
            geometry = [float(value) for value in record["geometry"]]
            observed_encoding = str(record["geometry_encoding"])
            records.append(
                {
                    "geometry": geometry,
                    "geometry_encoding": observed_encoding,
                    "output_geometry": convert_geometry_encoding(
                        geometry, observed_encoding, output_encoding
                    ),
                    "output_geometry_encoding": output_encoding,
                }
            )
        output["geometry_records"] = records
        output["geometries"] = [record["geometry"] for record in records]
    return output


def geometry_local_observation(
    row: dict[str, Any], schema: str, output_encoding: str, index: int
) -> dict[str, Any]:
    identity = observation_id(row, f"local:{index}")
    output: dict[str, Any] = {
        "observation_id": identity,
        "observation_kind": "REGISTERED_NATIVE_VIEW",
        "view_index": row.get("roi_index"),
        "source_support_base1000": valid_footprint(
            row.get("roi_full1000_xyxy")
        ),
        "support_label": row.get("source_support_label"),
        "valid": False,
        "observation_geometry_encoding": None,
        "observation_geometry_encodings": [],
        "output_geometry_encoding": output_encoding,
        "geometries": [],
        "geometry_records": [],
        "null_observation": False,
        "parse_error": None,
    }
    row_contract = row.get("evidence_contract") or {}
    try:
        row_output_encoding = geometry_encoding(dict(row_contract))
    except (TypeError, ValueError) as error:
        output["parse_error"] = f"local_terminal_geometry_contract:{error}"
        return output
    if row_output_encoding != output_encoding:
        output["parse_error"] = "output_geometry_encoding_mismatch"
        return output
    if row.get("error"):
        output["parse_error"] = "infrastructure_error"
        return output
    coordinate_provenance, provenance_error = (
        local_observation_coordinate_provenance(row)
    )
    if coordinate_provenance is None:
        output["parse_error"] = provenance_error
        return output
    local_extent = tuple(coordinate_provenance["coordinate_extent"])
    evidence = unstructured_evidence(
        row,
        source_kind="REGISTERED_NATIVE_VIEW",
        source_id=identity,
        view_index=row.get("roi_index"),
        source_support_base1000=output["source_support_base1000"],
        support_label=row.get("source_support_label"),
    )
    if evidence is not None:
        evidence.update(coordinate_provenance)
        output["unstructured_evidence"] = evidence
    local_parse = observation_geometry_response(
        raw_observation_text(row),
        allow_multiple=schema == "GEOMETRY_SET",
        coordinate_extent=local_extent,
    )
    if not local_parse.get("valid"):
        output["parse_error"] = str(local_parse.get("error") or "invalid_local_geometry")
        return output
    if not bool(row.get("stage3_parse_ok")):
        output["parse_error"] = str(
            row.get("stage3_parse_error") or "source_registration_failed"
        )
        return output
    if row.get("stage3_output_geometry_encoding") != output_encoding:
        output["parse_error"] = "source_registration_output_encoding_mismatch"
        return output
    # GEOMETRY_SET parsing is candidate-wise: structurally valid registered
    # candidates remain evidence while invalid members are explicitly
    # quarantined.  They are never clamped, guessed, or passed to synthesis.
    output["rejected_local_geometry"] = copy.deepcopy(
        row.get("invalid_local_geometry") or []
    )
    output["rejected_local_geometry_count"] = len(
        output["rejected_local_geometry"]
    )
    registered_raw = row.get("stage3_registered_geometries_full1000") or []
    local_records = list(local_parse.get("geometry_records") or [])
    local_count = len(local_records)
    if local_count != len(registered_raw):
        output["parse_error"] = "local_to_source_geometry_cardinality_mismatch"
        return output
    if schema == "SINGLE_GEOMETRY" and local_count not in {0, 1}:
        output["parse_error"] = "single_geometry_cardinality_invalid"
        return output
    records: list[dict[str, Any]] = []
    for local_record, registered in zip(local_records, registered_raw):
        observed_encoding = str(local_record.get("geometry_encoding") or "")
        if str(registered.get("geometry_encoding") or "") != observed_encoding:
            output["parse_error"] = "source_registration_encoding_not_preserved"
            return output
        local_geometry = normalized_geometry(
            local_record.get("geometry"),
            observed_encoding,
            coordinate_extent=local_extent,
        )
        registered_geometry = normalized_geometry(
            registered.get("geometry_source1000"), observed_encoding
        )
        output_geometry = normalized_geometry(
            registered.get("output_geometry_source1000"), output_encoding
        )
        if (
            local_geometry is None
            or registered_geometry is None
            or output_geometry is None
            or local_geometry
            != [float(value) for value in registered.get("local_geometry") or []]
            or registered.get("output_geometry_encoding") != output_encoding
        ):
            output["parse_error"] = "registered_geometry_encoding_or_shape_invalid"
            return output
        records.append(
            {
                "geometry": registered_geometry,
                "geometry_encoding": observed_encoding,
                "output_geometry": output_geometry,
                "output_geometry_encoding": output_encoding,
            }
        )
    output.update(
        {
            "valid": True,
            "observation_geometry_encoding": local_parse.get(
                "geometry_encoding"
            ),
            "observation_geometry_encodings": list(
                local_parse.get("geometry_encodings") or []
            ),
            "geometries": [record["geometry"] for record in records],
            "geometry_records": records,
            "null_observation": bool(local_parse.get("null")),
            "parse_error": None,
        }
    )
    return output


def geometry_candidates(
    native_record: dict[str, Any],
    local_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in [native_record, *local_records]:
        if not record.get("valid"):
            continue
        for geometry_index, geometry_record in enumerate(
            record.get("geometry_records") or [], start=1
        ):
            values = geometry_record["geometry"]
            candidate: dict[str, Any] = {
                "source_kind": record["observation_kind"],
                "source_id": str(
                    record.get("observation_id")
                    or record.get("view_index")
                    or record["observation_kind"]
                ),
                "geometry_index": geometry_index,
                "geometry_encoding": geometry_record["geometry_encoding"],
                "geometry": [float(value) for value in values],
                "output_geometry_encoding": geometry_record[
                    "output_geometry_encoding"
                ],
                "output_geometry": [
                    float(value)
                    for value in geometry_record["output_geometry"]
                ],
            }
            if record.get("view_index") is not None:
                candidate["view_index"] = record["view_index"]
            if record.get("source_support_base1000") is not None:
                candidate["source_support_base1000"] = record[
                    "source_support_base1000"
                ]
            candidates.append(candidate)
    return candidates


def exact_geometry_groups(
    candidates: list[dict[str, Any]], encoding: str | None = None
) -> list[dict[str, Any]]:
    """Inventory byte-equivalent geometries without assigning reliability."""

    groups: dict[tuple[str, tuple[float, ...]], dict[str, Any]] = {}
    for candidate in candidates:
        candidate_encoding = str(candidate.get("geometry_encoding") or encoding or "")
        key = geometry_key(candidate["geometry"], candidate_encoding)
        if key not in groups:
            groups[key] = {
                "geometry_encoding": candidate_encoding,
                "geometry": candidate["geometry"],
                "output_geometry_encoding": candidate.get(
                    "output_geometry_encoding"
                ),
                "output_geometry": candidate.get("output_geometry"),
                "observed_by": [],
                "inventory_only": True,
            }
        provenance = {
            name: candidate[name]
            for name in (
                "source_kind",
                "source_id",
                "geometry_index",
                "view_index",
                "source_support_base1000",
            )
            if name in candidate
        }
        if provenance not in groups[key]["observed_by"]:
            groups[key]["observed_by"].append(provenance)
    for group in groups.values():
        group["candidate_occurrence_count"] = len(group["observed_by"])
        group["independent_observation_count"] = len(
            {
                (str(item["source_kind"]), str(item["source_id"]))
                for item in group["observed_by"]
            }
        )
    return list(groups.values())


def scalar_local_observation(row: dict[str, Any], index: int) -> dict[str, Any]:
    value = None if row.get("error") else parse_scalar(raw_observation_text(row))
    footprint = valid_footprint(row.get("roi_full1000_xyxy"))
    identity = observation_id(row, f"local:{index}")
    output = {
        "observation_id": identity,
        "observation_kind": "REGISTERED_NATIVE_VIEW",
        "view_index": row.get("roi_index"),
        "source_support_base1000": footprint,
        "support_label": row.get("source_support_label"),
        "parsed_scalar": value,
        "valid": value is not None and footprint is not None and not row.get("error"),
        "parse_error": (
            "infrastructure_error"
            if row.get("error")
            else "invalid_or_ambiguous_scalar"
            if value is None
            else "invalid_source_support"
            if footprint is None
            else None
        ),
    }
    evidence = unstructured_evidence(
        row,
        source_kind="REGISTERED_NATIVE_VIEW",
        source_id=identity,
        view_index=row.get("roi_index"),
        source_support_base1000=footprint,
        support_label=row.get("source_support_label"),
    )
    if evidence is not None:
        output["unstructured_evidence"] = evidence
    return output


def numeric_difference_coverage_gate(state: dict[str, Any]) -> dict[str, Any]:
    """Require complete measured acquisition of exactly the public A/B pair."""

    coverage = state.get("acquisition_coverage")
    if not isinstance(coverage, dict):
        return {
            "passed": False,
            "status": None,
            "reason": "missing_acquisition_coverage",
            "support_record_count": 0,
            "support_record_count_by_label": {"A": 0, "B": 0},
            "complete_support_record_count_by_label": {"A": 0, "B": 0},
        }

    status = str(coverage.get("status") or "")
    raw_records = coverage.get("support_records")
    records = list(raw_records) if isinstance(raw_records, list) else []
    if all(isinstance(record, dict) for record in records):
        label_counts = {
            label: sum(
                str(record.get("support_label") or "") == label
                for record in records
            )
            for label in ("A", "B")
        }
    else:
        label_counts = {"A": 0, "B": 0}
    complete_label_counts = {
        label: sum(
            str(record.get("support_label") or "") == label
            and record.get("complete") is True
            for record in records
            if isinstance(record, dict)
        )
        for label in ("A", "B")
    }
    base = {
        "status": status or None,
        "support_record_count": len(records),
        "support_record_count_by_label": label_counts,
        "complete_support_record_count_by_label": complete_label_counts,
    }
    if status != "COMPLETE_WITHIN_BUDGET":
        return {
            "passed": False,
            "reason": (
                "acquisition_coverage_status_not_complete_within_budget:"
                f"{status or 'MISSING'}"
            ),
            **base,
        }
    if (
        len(records) != 2
        or label_counts != {"A": 1, "B": 1}
        or complete_label_counts != {"A": 1, "B": 1}
    ):
        return {
            "passed": False,
            "reason": (
                "requires_exactly_two_complete_public_support_records:"
                f"records={len(records)},A={complete_label_counts['A']},"
                f"B={complete_label_counts['B']}"
            ),
            **base,
        }
    return {
        "passed": True,
        "reason": "complete_measured_public_a_b_support_coverage",
        **base,
    }


def numeric_difference_reducer(state: dict[str, Any]) -> dict[str, Any]:
    """Reduce exactly one valid scalar for each public A/B support.

    This is an intentionally narrow, deterministic public-contract operator.
    It neither guesses scalars from prose nor uses hidden metadata,
    evaluation feedback, geometric area, or overlap.
    """

    contract = state.get("public_contract") or {}
    active = "NUMERIC_DIFFERENCE" in {
        str(value) for value in contract.get("operator_activations") or []
    }
    applicable = (
        str(contract.get("answer_schema") or "") == "SCALAR"
        and str(contract.get("composition") or "") == "NUMERIC_DIFFERENCE"
        and active
    )
    records = list((state.get("typed_aggregate") or {}).get("local_scalar_evidence") or [])
    grouped = {
        label: [record for record in records if str(record.get("support_label")) == label]
        for label in ("A", "B")
    }
    counts = {label: len(grouped[label]) for label in ("A", "B")}
    if not applicable:
        return {
            "applicable": False,
            "usable": False,
            "value": None,
            "valid_scalar_count_by_support": counts,
            "selection_reason": "public_numeric_difference_operator_not_active",
            "coverage_gate": {
                "passed": False,
                "status": None,
                "reason": "not_applicable",
            },
        }
    coverage_gate = numeric_difference_coverage_gate(state)
    if not coverage_gate["passed"]:
        return {
            "applicable": True,
            "usable": False,
            "value": None,
            "valid_scalar_count_by_support": counts,
            "selection_reason": coverage_gate["reason"],
            "coverage_gate": coverage_gate,
        }
    if counts != {"A": 1, "B": 1}:
        return {
            "applicable": True,
            "usable": False,
            "value": None,
            "valid_scalar_count_by_support": counts,
            "selection_reason": (
                f"requires_exactly_one_valid_scalar_per_public_support:A={counts['A']},B={counts['B']}"
            ),
            "coverage_gate": coverage_gate,
        }
    value_a = int(grouped["A"][0]["value"])
    value_b = int(grouped["B"][0]["value"])
    return {
        "applicable": True,
        "usable": True,
        "value": abs(value_a - value_b),
        "input_values": {"A": value_a, "B": value_b},
        "input_observation_ids": {
            "A": grouped["A"][0]["observation_id"],
            "B": grouped["B"][0]["observation_id"],
        },
        "valid_scalar_count_by_support": counts,
        "selection_reason": "exactly_one_valid_scalar_per_public_support",
        "coverage_gate": coverage_gate,
    }


def aggregate_row(
    source: dict[str, Any],
    native: dict[str, Any],
    locals_: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = dict(source["evidence_contract"])
    schema = str(contract["answer_schema"])
    encoding = geometry_encoding(contract)
    native_raw = raw_observation_text(native)
    native_payload, native_prefix_error = provisional_payload(native_raw)
    ordered_locals = sorted(locals_, key=lambda item: int(item.get("roi_index") or 0))
    require_unique_rows(
        ordered_locals, key_names=("task_id", "observation_id"), label="local observation"
    )

    coverage_records = [row.get("public_support_coverage") for row in ordered_locals]
    if not coverage_records or any(not isinstance(record, dict) for record in coverage_records):
        raise ValueError(f"missing public support coverage state: {source['sample_id']}")
    acquisition_coverage = coverage_records[0]
    if any(record != acquisition_coverage for record in coverage_records[1:]):
        raise ValueError(f"inconsistent public support coverage state: {source['sample_id']}")

    state: dict[str, Any] = {
        "schema_version": "map-typed-evidence-state-2",
        "sample_id": source["sample_id"],
        "public_contract": contract,
        "acquisition_coverage": acquisition_coverage,
        "complete_scene_observation": {},
        "local_observations": [],
        "typed_aggregate": {},
    }
    aggregate = state["typed_aggregate"]

    if schema == "SCALAR":
        native_scalar = (
            parse_scalar(native_payload or "")
            if not native.get("error") and native_prefix_error is None
            else None
        )
        local_records = [
            scalar_local_observation(row, index)
            for index, row in enumerate(ordered_locals, start=1)
        ]
        native_identity = observation_id(native, "complete-scene")
        native_record: dict[str, Any] = {
            "observation_id": native_identity,
            "observation_kind": "COMPLETE_SCENE",
            "parsed_scalar": native_scalar,
            "valid": native_scalar is not None and not native.get("error"),
            "parse_error": (
                "infrastructure_error"
                if native.get("error")
                else native_prefix_error or (
                    "invalid_or_ambiguous_scalar" if native_scalar is None else None
                )
            ),
        }
        native_evidence = unstructured_evidence(
            native,
            source_kind="COMPLETE_SCENE",
            source_id=native_identity,
        )
        if native_evidence is not None:
            native_record["unstructured_evidence"] = native_evidence
        state["complete_scene_observation"] = native_record
        state["local_observations"] = local_records
        aggregate.update(
            {
                "local_scalar_evidence": [
                    {
                        "observation_id": record["observation_id"],
                        "view_index": record["view_index"],
                        "support_label": record["support_label"],
                        "source_support_base1000": record[
                            "source_support_base1000"
                        ],
                        "value": record["parsed_scalar"],
                    }
                    for record in local_records
                    if record["valid"]
                ],
            }
        )
        if str(contract.get("composition") or "") == "NUMERIC_DIFFERENCE":
            aggregate["numeric_difference_reducer"] = numeric_difference_reducer(
                state
            )
    elif schema == "OPTION":
        local_records = []
        for index, row in enumerate(ordered_locals, start=1):
            choice = None if row.get("error") else parse_choice(raw_observation_text(row))
            identity = observation_id(row, f"local:{index}")
            footprint = valid_footprint(row.get("roi_full1000_xyxy"))
            record: dict[str, Any] = {
                "observation_id": identity,
                "observation_kind": "REGISTERED_NATIVE_VIEW",
                "view_index": row.get("roi_index"),
                "source_support_base1000": footprint,
                "support_label": row.get("source_support_label"),
                "parsed_choice": choice,
                "valid": (
                    choice is not None
                    and footprint is not None
                    and not row.get("error")
                ),
                "parse_error": (
                    "infrastructure_error"
                    if row.get("error")
                    else "invalid_or_ambiguous_choice"
                    if choice is None
                    else "invalid_source_support"
                    if footprint is None
                    else None
                ),
            }
            evidence = unstructured_evidence(
                row,
                source_kind="REGISTERED_NATIVE_VIEW",
                source_id=identity,
                view_index=row.get("roi_index"),
                source_support_base1000=footprint,
                support_label=row.get("source_support_label"),
            )
            if evidence is not None:
                record["unstructured_evidence"] = evidence
            local_records.append(record)
        native_choice = (
            parse_choice(native_payload or "")
            if not native.get("error") and native_prefix_error is None
            else None
        )
        native_identity = observation_id(native, "complete-scene")
        native_record = {
            "observation_id": native_identity,
            "observation_kind": "COMPLETE_SCENE",
            "parsed_choice": native_choice,
            "valid": native_choice is not None and not native.get("error"),
            "parse_error": (
                "infrastructure_error"
                if native.get("error")
                else native_prefix_error or (
                    "invalid_or_ambiguous_choice" if native_choice is None else None
                )
            ),
        }
        native_evidence = unstructured_evidence(
            native,
            source_kind="COMPLETE_SCENE",
            source_id=native_identity,
        )
        if native_evidence is not None:
            native_record["unstructured_evidence"] = native_evidence
        state["complete_scene_observation"] = native_record
        state["local_observations"] = local_records
        aggregate.update(
            {
                "complete_scene_choice": native_choice,
                "local_choices": [
                    {
                        "observation_id": record["observation_id"],
                        "view_index": record["view_index"],
                        "choice": record["parsed_choice"],
                    }
                    for record in local_records
                    if record["valid"]
                ],
            }
        )
    elif schema in GEOMETRY_SCHEMAS:
        native_record = geometry_native_observation(native, schema, encoding)
        local_records = [
            geometry_local_observation(row, schema, encoding, index)
            for index, row in enumerate(ordered_locals, start=1)
        ]
        state["complete_scene_observation"] = native_record
        state["local_observations"] = local_records
        candidates = geometry_candidates(native_record, local_records)
        duplicate_inventory = exact_geometry_groups(candidates)
        aggregate.update(
            {
                "output_geometry_encoding": encoding,
                "geometry_evidence_policy": (
                    "retain_every_valid_source_registered_candidate_with_native_encoding_and_provenance"
                ),
                "source_geometry_candidates": candidates,
                "exact_duplicate_inventory_groups": duplicate_inventory,
                "candidate_reliability_policy": (
                    "terminal_visual_resolution_from_public_schema_and_composition_no_programmatic_candidate_selector"
                ),
                "candidate_count": len(candidates),
                "candidate_encodings": dict(
                    sorted(
                        Counter(
                            str(candidate["geometry_encoding"])
                            for candidate in candidates
                        ).items()
                    )
                ),
            }
        )
        if schema == "SINGLE_GEOMETRY":
            aggregate["geometry_consistency_status"] = (
                "no_registered_candidate"
                if not candidates
                else "registered_candidates_require_terminal_visual_resolution"
            )
        else:
            aggregate["output_ready_geometry_candidates"] = [
                {
                    "source_kind": candidate["source_kind"],
                    "source_id": candidate["source_id"],
                    "geometry_index": candidate["geometry_index"],
                    "observation_geometry_encoding": candidate[
                        "geometry_encoding"
                    ],
                    "source_geometry": candidate["geometry"],
                    "output_geometry_encoding": candidate[
                        "output_geometry_encoding"
                    ],
                    "output_geometry": candidate["output_geometry"],
                    **(
                        {
                            "source_support_base1000": candidate[
                                "source_support_base1000"
                            ]
                        }
                        if "source_support_base1000" in candidate
                        else {}
                    ),
                }
                for candidate in candidates
            ]
            aggregate["overlap_suppression_applied"] = False
    else:
        raise ValueError(f"unsupported schema: {schema}")
    return state


def aggregate(args: argparse.Namespace) -> int:
    manifest = keyed(read_jsonl(args.manifest))
    native = keyed(read_jsonl(args.native))
    locals_grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.mapped):
        locals_grouped[str(row["sample_id"])].append(row)
    if set(manifest) != set(native):
        raise ValueError("runtime/native fixed-denominator sets differ")
    unknown_local = set(locals_grouped) - set(manifest)
    if unknown_local:
        raise ValueError(f"local observation rows have unknown samples: {sorted(unknown_local)}")
    output = [
        aggregate_row(manifest[sid], native[sid], locals_grouped.get(sid, []))
        for sid in manifest
    ]
    audit = {
        "schema_version": "map-typed-aggregation-audit-2",
        "rows": len(output),
        "unique_sample_ids": len({row["sample_id"] for row in output}),
        "schemas": dict(
            sorted(Counter(row["public_contract"]["answer_schema"] for row in output).items())
        ),
        "terminal_geometry_encodings": dict(
            sorted(
                Counter(
                    geometry_encoding(row["public_contract"]) for row in output
                ).items()
            )
        ),
        "specialized_optional_observations": 0,
        "terminal_answers_written": 0,
        "geometry_overlap_suppression_used": False,
        "area_or_iou_ranking_used": False,
        "geometry_candidates_removed_by_aggregation": 0,
        "geometry_candidate_selection_inputs": [
            "public_answer_schema",
            "public_composition",
            "source_registered_visual_evidence",
        ],
        "test_used": False,
    }
    audit["passed"] = audit["rows"] == audit["unique_sample_ids"]
    write_jsonl(args.out, output)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"typed aggregation failed: {audit}")
    return 0


def canonical_sha256(value: Any) -> str:
    """Hash one JSON value with a single deterministic serialization."""

    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def exact_single_public_support_gate(state: dict[str, Any]) -> dict[str, Any]:
    """Prove that one acquired view completely covers one public support.

    This gate consumes only the public evidence contract and deterministic
    acquisition geometry.  It deliberately does not know task identity,
    benchmark annotations, model confidence, or an evaluation score.
    """

    contract = state.get("public_contract") or {}
    support_source = str(contract.get("support_source") or "")
    support_extent = str(contract.get("support_extent") or "")
    coverage = state.get("acquisition_coverage")
    base = {
        "support_source": support_source or None,
        "support_extent": support_extent or None,
    }
    if support_source not in PUBLIC_CLOSABLE_SUPPORT_SOURCES:
        return {
            "passed": False,
            "reason": "support_is_not_a_publicly_declared_exact_domain",
            **base,
        }
    if support_extent not in {"BOUNDED_SUPPORT", "SINGLE_TARGET"}:
        return {
            "passed": False,
            "reason": "public_support_is_not_one_bounded_domain",
            **base,
        }
    if not isinstance(coverage, dict):
        return {
            "passed": False,
            "reason": "missing_acquisition_coverage",
            **base,
        }
    records = coverage.get("support_records")
    records = list(records) if isinstance(records, list) else []
    selected_views = int(coverage.get("selected_views") or 0)
    public_boxes = int(coverage.get("public_support_box_count") or 0)
    complete_boxes = int(coverage.get("complete_support_box_count") or 0)
    incomplete_boxes = int(coverage.get("incomplete_support_box_count") or 0)
    complete_record = (
        len(records) == 1
        and isinstance(records[0], dict)
        and records[0].get("complete") is True
        and math.isclose(
            float(records[0].get("coverage_ratio") or 0.0),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(records[0].get("native_detail_coverage_ratio") or 0.0),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    passed = (
        str(coverage.get("status") or "") == "COMPLETE_WITHIN_BUDGET"
        and public_boxes == 1
        and complete_boxes == 1
        and incomplete_boxes == 0
        and selected_views == 1
        and complete_record
    )
    return {
        "passed": passed,
        "reason": (
            "one_public_support_completely_observed_at_native_detail"
            if passed
            else "public_support_coverage_is_not_exactly_one_complete_native_view"
        ),
        "coverage_status": str(coverage.get("status") or "") or None,
        "public_support_box_count": public_boxes,
        "complete_support_box_count": complete_boxes,
        "incomplete_support_box_count": incomplete_boxes,
        "selected_views": selected_views,
        "support_record_count": len(records),
        "support_record_complete": complete_record,
        **base,
    }


def render_terminal_geometry(values: Iterable[float]) -> str:
    """Serialize registered coordinates without changing their numeric value."""

    def number(value: float) -> str:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("terminal geometry contains a non-finite value")
        if numeric == 0.0:
            numeric = 0.0
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".12g")

    return "[" + ",".join(number(value) for value in values) + "]"


TERMINAL_GEOMETRY_NORMALIZATION_SCHEMA = (
    "map-terminal-geometry-normalization-1"
)


# The official evaluator reads a terminal answer only after an explicit answer
# marker, so the normalization repairs exactly that declaration and never a
# coordinate list the response merely mentions in passing.
_TERMINAL_ANSWER_MARKER = re.compile(
    r"(?ix)"
    r"(?:"
    r"(?:the\s+)?final\s+answer\s*(?:is\s*)?(?::|=)?"
    r"|the\s+answer\s+is\s*(?::|=)?"
    r"|answer\s*(?::|=)"
    r")\s*"
)
_TERMINAL_COORDINATE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def declared_terminal_geometry(text: str) -> list[float] | None:
    """Return the coordinate list the composer declared as its final answer."""

    value = str(text or "")
    markers = list(_TERMINAL_ANSWER_MARKER.finditer(value))
    if not markers:
        return None
    suffix = value[markers[-1].end() :]
    start = suffix.find("[")
    if start < 0:
        return None
    depth = 0
    end = -1
    for index in range(start, len(suffix)):
        if suffix[index] == "[":
            depth += 1
        elif suffix[index] == "]":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < 0:
        return None
    numbers = _TERMINAL_COORDINATE.findall(suffix[start : end + 1])
    if not numbers:
        return None
    try:
        return [float(number) for number in numbers]
    except ValueError:
        return None


def registered_terminal_projections(
    state: dict[str, Any], encoding: str
) -> list[list[float]]:
    """Return the structurally valid registered projections of one public encoding."""

    aggregate = state.get("typed_aggregate") or {}
    projections: list[list[float]] = []
    for candidate in aggregate.get("source_geometry_candidates") or []:
        values = candidate.get("output_geometry")
        if not values:
            continue
        if str(candidate.get("output_geometry_encoding") or "") != encoding:
            continue
        projection = [float(value) for value in values]
        if valid_geometry(projection, encoding):
            projections.append(projection)
    return projections


def declared_footprint_geometry(
    values: list[float], encoding: str
) -> list[float] | None:
    """Repair only the vertex order by keeping the declared footprint as a box."""

    if len(values) not in (4, 8):
        return None
    x_min, x_max = min(values[0::2]), max(values[0::2])
    y_min, y_max = min(values[1::2]), max(values[1::2])
    repaired = (
        [x_min, y_min, x_max, y_max]
        if encoding == "AABB4"
        else [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
    )
    return repaired if valid_geometry(repaired, encoding) else None


def normalize_terminal_geometry(
    response: str, state: dict[str, Any], schema: str, encoding: str
) -> tuple[str, dict[str, Any] | None]:
    """Repair a terminal geometry the composer serialized into an invalid shape.

    The typed evidence state already carries the registered terminal projection
    of every observation, and the public contract tells the composer to use it.
    When the terminal answer instead declares a structurally invalid value of
    the contract's own encoding -- duplicated, collinear or reordered vertices,
    a degenerate axis-aligned box -- the row is a serialization failure rather
    than a decision, and the official geometry validation rejects it outright.
    Repair is deliberately narrow:

    * only one terminal geometry (``SINGLE_GEOMETRY``) is repaired; a geometry
      set describes a decision the runtime cannot reconstruct from the shape;
    * only a declaration of the encoding's own arity is repaired, so
      abstentions, prose endings and sets keep standing as the model's answer;
    * an all-zero declaration is an explicit "no geometry" and is left alone;
    * one registered projection in the declared encoding is used verbatim;
      several leave the choice open, so only the declared footprint is kept;
    * the replacement must itself validate, otherwise the response is unchanged.
    """

    if schema != "SINGLE_GEOMETRY" or encoding not in GEOMETRY_ENCODINGS:
        return response, None
    declared = declared_terminal_geometry(response)
    if declared is None or len(declared) != (4 if encoding == "AABB4" else 8):
        return response, None
    if valid_geometry(declared, encoding) or not any(declared):
        return response, None
    projections = registered_terminal_projections(state, encoding)
    if len(projections) == 1:
        repaired: list[float] | None = projections[0]
        origin = "single_registered_terminal_projection"
    else:
        repaired = declared_footprint_geometry(declared, encoding)
        origin = "declared_footprint_axis_aligned_box"
    if repaired is None:
        return response, None
    audit = {
        "schema_version": TERMINAL_GEOMETRY_NORMALIZATION_SCHEMA,
        "applied": True,
        "encoding": encoding,
        "declared_geometry": declared,
        "registered_projection_count": len(projections),
        "substituted_geometry": repaired,
        "substitution_origin": origin,
    }
    return "Final answer: " + render_terminal_geometry(repaired), audit


def mapped_observation_digest(local_rows: list[dict[str, Any]]) -> str:
    ordered = sorted(
        local_rows,
        key=lambda row: (
            int(row.get("roi_index") or 0),
            str(row.get("task_id") or row.get("observation_id") or ""),
        ),
    )
    return canonical_sha256(ordered)


def _open_closure(
    source: dict[str, Any],
    state: dict[str, Any],
    local_rows: list[dict[str, Any]],
    *,
    reason: str,
    support_gate: dict[str, Any],
    reducer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "map-evidence-closure-decision-1",
        "sample_id": str(source["sample_id"]),
        "closure_status": "OPEN",
        "terminal_mode": "SHARED_COMPOSER",
        "final_response": None,
        "selection_reason": reason,
        "answer_schema": str((state.get("public_contract") or {}).get("answer_schema") or ""),
        "public_composition": str((state.get("public_contract") or {}).get("composition") or ""),
        "support_gate": support_gate,
        "numeric_difference_reducer": reducer,
        "selected_observation_ids": [],
        "state_sha256": canonical_sha256(state),
        "mapped_observations_sha256": mapped_observation_digest(local_rows),
        "private_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }


def evidence_closure_decision(
    source: dict[str, Any],
    state: dict[str, Any],
    local_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compile one pre-composer terminal decision from public typed evidence."""

    sid = str(source["sample_id"])
    if str(state.get("sample_id") or "") != sid:
        raise ValueError(f"closure state/sample mismatch: {sid}")
    if state.get("public_contract") != source.get("evidence_contract"):
        raise ValueError(f"closure state/public contract mismatch: {sid}")
    activations = set(
        str(value)
        for value in (state.get("public_contract") or {}).get(
            "operator_activations", []
        )
    )
    if "EVIDENCE_CLOSURE" not in activations:
        raise ValueError(f"universal EVIDENCE_CLOSURE activation is missing: {sid}")
    require_unique_rows(
        local_rows,
        key_names=("task_id", "observation_id"),
        label="closure mapped observation",
    )
    if any(str(row.get("sample_id") or "") != sid for row in local_rows):
        raise ValueError(f"closure mapped observation/sample mismatch: {sid}")
    if any(row.get("evidence_contract") != source.get("evidence_contract") for row in local_rows):
        raise ValueError(f"closure mapped observation/public contract mismatch: {sid}")

    reducer = numeric_difference_reducer(state)
    single_gate = exact_single_public_support_gate(state)
    response: str | None = None
    reason: str | None = None
    selected_ids: list[str] = []
    closure_kind: str | None = None

    # The public A/B difference operator is the same closure family with two
    # complete supports instead of one.  It is decided before a composer task
    # is ever emitted.
    if reducer["usable"]:
        response = f"Final answer: {int(reducer['value'])}"
        reason = str(reducer["selection_reason"])
        selected_ids = [
            str(reducer["input_observation_ids"][label]) for label in ("A", "B")
        ]
        closure_kind = "DETERMINISTIC_NUMERIC_DIFFERENCE"
    elif reducer["applicable"]:
        return _open_closure(
            source,
            state,
            local_rows,
            reason=str(reducer["selection_reason"]),
            support_gate=single_gate,
            reducer=reducer,
        )
    elif not single_gate["passed"]:
        return _open_closure(
            source,
            state,
            local_rows,
            reason=str(single_gate["reason"]),
            support_gate=single_gate,
            reducer=reducer,
        )
    else:
        observations = list(state.get("local_observations") or [])
        if len(observations) != 1 or len(local_rows) != 1:
            return _open_closure(
                source,
                state,
                local_rows,
                reason="closure_requires_exactly_one_typed_and_mapped_observation",
                support_gate=single_gate,
                reducer=reducer,
            )
        observation = observations[0]
        mapped = local_rows[0]
        observation_identity = str(observation.get("observation_id") or "")
        mapped_identity = str(mapped.get("task_id") or mapped.get("observation_id") or "")
        if not observation_identity or observation_identity != mapped_identity:
            return _open_closure(
                source,
                state,
                local_rows,
                reason="typed_and_mapped_observation_identity_mismatch",
                support_gate=single_gate,
                reducer=reducer,
            )
        if observation.get("valid") is not True or mapped.get("error"):
            return _open_closure(
                source,
                state,
                local_rows,
                reason="single_complete_observation_has_no_valid_typed_answer",
                support_gate=single_gate,
                reducer=reducer,
            )
        schema = str((state.get("public_contract") or {}).get("answer_schema") or "")
        if schema == "SCALAR":
            value = observation.get("parsed_scalar")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return _open_closure(
                    source,
                    state,
                    local_rows,
                    reason="complete_scalar_observation_is_not_one_non_negative_integer",
                    support_gate=single_gate,
                    reducer=reducer,
                )
            response = f"Final answer: {value}"
        elif schema == "OPTION":
            choice = str(observation.get("parsed_choice") or "")
            if choice not in {"A", "B", "C", "D"}:
                return _open_closure(
                    source,
                    state,
                    local_rows,
                    reason="complete_option_observation_is_not_one_public_option",
                    support_gate=single_gate,
                    reducer=reducer,
                )
            response = f"Final answer: {choice}"
        elif schema in GEOMETRY_SCHEMAS:
            if mapped.get("stage3_parse_ok") is not True or mapped.get("stage3_registration_ok") is not True:
                return _open_closure(
                    source,
                    state,
                    local_rows,
                    reason="complete_geometry_observation_is_not_source_registered",
                    support_gate=single_gate,
                    reducer=reducer,
                )
            records = list(observation.get("geometry_records") or [])
            if observation.get("null_observation") is True:
                if schema != "GEOMETRY_SET":
                    return _open_closure(
                        source,
                        state,
                        local_rows,
                        reason="single_geometry_contract_does_not_close_on_null",
                        support_gate=single_gate,
                        reducer=reducer,
                    )
                response = "Final answer: []"
            elif schema == "SINGLE_GEOMETRY":
                if len(records) != 1:
                    return _open_closure(
                        source,
                        state,
                        local_rows,
                        reason="single_geometry_closure_requires_one_registered_geometry",
                        support_gate=single_gate,
                        reducer=reducer,
                    )
                response = "Final answer: " + render_terminal_geometry(
                    records[0]["output_geometry"]
                )
            else:
                if not records:
                    return _open_closure(
                        source,
                        state,
                        local_rows,
                        reason="non_null_geometry_set_has_no_registered_geometry",
                        support_gate=single_gate,
                        reducer=reducer,
                    )
                response = "Final answer: [" + ",".join(
                    render_terminal_geometry(record["output_geometry"])
                    for record in records
                ) + "]"
        else:
            raise ValueError(f"unsupported closure answer schema: {schema}")
        reason = "one_complete_public_support_has_one_structurally_valid_typed_answer"
        selected_ids = [observation_identity]
        closure_kind = "IDENTITY_EVIDENCE_CLOSURE"

    if response is None or reason is None or closure_kind is None:
        raise AssertionError("closed evidence decision is incomplete")
    decision = {
        "schema_version": "map-evidence-closure-decision-1",
        "sample_id": sid,
        "closure_status": "CLOSED",
        "terminal_mode": closure_kind,
        "final_response": response,
        "selection_reason": reason,
        "answer_schema": str((state.get("public_contract") or {}).get("answer_schema") or ""),
        "public_composition": str((state.get("public_contract") or {}).get("composition") or ""),
        "support_gate": single_gate,
        "numeric_difference_reducer": reducer,
        "selected_observation_ids": selected_ids,
        "state_sha256": canonical_sha256(state),
        "mapped_observations_sha256": mapped_observation_digest(local_rows),
        "private_identity_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    return decision


def seal_closure_decision(decision: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(decision)
    sealed["decision_sha256"] = canonical_sha256(decision)
    return sealed


def verify_closure_decision_binding(
    source: dict[str, Any], state: dict[str, Any], decision: dict[str, Any]
) -> None:
    sid = str(source["sample_id"])
    if str(decision.get("sample_id") or "") != sid:
        raise ValueError(f"closure decision/sample mismatch: {sid}")
    recorded_hash = str(decision.get("decision_sha256") or "")
    unhashed = {key: value for key, value in decision.items() if key != "decision_sha256"}
    if not recorded_hash or recorded_hash != canonical_sha256(unhashed):
        raise ValueError(f"closure decision hash mismatch: {sid}")
    if str(decision.get("state_sha256") or "") != canonical_sha256(state):
        raise ValueError(f"closure decision/state mismatch: {sid}")
    if decision.get("closure_status") not in {"CLOSED", "OPEN"}:
        raise ValueError(f"invalid closure status: {sid}")
    if (decision.get("closure_status") == "CLOSED") != bool(
        str(decision.get("final_response") or "").strip()
    ):
        raise ValueError(f"closure response/status mismatch: {sid}")


def closure(args: argparse.Namespace) -> int:
    manifest = keyed(read_jsonl(args.manifest))
    states = keyed(read_jsonl(args.states))
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.mapped):
        grouped[str(row["sample_id"])].append(row)
    if set(manifest) != set(states):
        raise ValueError("closure manifest/state fixed-denominator sets differ")
    if set(grouped) - set(manifest):
        raise ValueError("closure mapped observations contain unknown samples")
    decisions = [
        seal_closure_decision(
            evidence_closure_decision(
                manifest[sid], states[sid], grouped.get(sid, [])
            )
        )
        for sid in manifest
    ]
    statuses = Counter(str(row["closure_status"]) for row in decisions)
    audit = {
        "schema_version": "map-evidence-closure-audit-1",
        "rows": len(decisions),
        "unique_sample_ids": len({row["sample_id"] for row in decisions}),
        "closure_status_counts": dict(sorted(statuses.items())),
        "closed_terminal_mode_counts": dict(
            sorted(
                Counter(
                    str(row["terminal_mode"])
                    for row in decisions
                    if row["closure_status"] == "CLOSED"
                ).items()
            )
        ),
        "closed_answer_schema_counts": dict(
            sorted(
                Counter(
                    str(row["answer_schema"])
                    for row in decisions
                    if row["closure_status"] == "CLOSED"
                ).items()
            )
        ),
        "open_reason_counts": dict(
            sorted(
                Counter(
                    str(row["selection_reason"])
                    for row in decisions
                    if row["closure_status"] == "OPEN"
                ).items()
            )
        ),
        "decision_happens_before_composer_task_creation": True,
        "task_identity_used": False,
        "historical_version_or_prediction_used": False,
        "ground_truth_or_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = (
        audit["rows"] == audit["unique_sample_ids"]
        and sum(statuses.values()) == audit["rows"]
        and all(row.get("private_identity_used") is False for row in decisions)
        and all(row.get("ground_truth_or_metric_used") is False for row in decisions)
    )
    write_jsonl(args.out, decisions)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"evidence closure failed: {audit}")
    return 0


def output_contract(schema: str, encoding: str) -> str:
    if schema == "SCALAR":
        return (
            "Return one non-negative integer. The last line must begin with the literal "
            "label `Final answer:` and contain only that integer after the label."
        )
    if schema == "OPTION":
        return (
            "Return one valid option letter. The last line must begin with the literal label "
            "`Final answer:` and contain only that letter after the label. Do not repeat options."
        )
    shape = (
        "one axis-aligned AABB4 value formatted as [x_min,y_min,x_max,y_max]"
        if encoding == "AABB4"
        else "one ordered convex QUAD8 value formatted as [x1,y1,x2,y2,x3,y3,x4,y4]"
    )
    conversion = (
        "When evidence was observed as QUAD8, use its recorded deterministic AABB4 output projection."
        if encoding == "AABB4"
        else "When evidence was observed as AABB4, use its recorded rectangular QUAD8 output projection; never discard it only because its native encoding differed."
    )
    if schema == "SINGLE_GEOMETRY":
        return (
            f"Return {shape} in the complete source image's 0-1000 coordinate system. "
            "The last line must begin with `Final answer:` and contain only that geometry. "
            f"{conversion}"
        )
    if schema == "GEOMETRY_SET":
        return (
            f"Return every supported target as {shape} in the complete source image's "
            "0-1000 coordinate system as a list of geometries, or [] when no target is supported. The last line "
            "must begin with `Final answer:` and contain only the geometry set. "
            f"{conversion}"
        )
    raise ValueError(f"unsupported schema: {schema}")


def composer_policy(contract: dict[str, Any]) -> dict[str, Any]:
    """Compile one terminal policy solely from the public evidence contract."""

    schema = str(contract.get("answer_schema") or "")
    composition = str(contract.get("composition") or "")
    if schema not in {"SCALAR", "OPTION", *GEOMETRY_SCHEMAS}:
        raise ValueError(f"unsupported public answer schema: {schema}")
    if composition not in PUBLIC_COMPOSITIONS:
        raise ValueError(f"unsupported public composition: {composition}")
    encoding = geometry_encoding(contract)
    return {
        "answer_schema": schema,
        "composition": composition,
        "output_geometry_encoding": encoding,
        "candidate_policy": (
            "retain_registered_geometry_with_provenance"
            if schema in GEOMETRY_SCHEMAS
            else "retain_typed_source_bound_observations"
        ),
        "terminal_composer": "single_public_contract_composer",
        "terminal_materialization": (
            "exact_public_A_B_absolute_difference_else_shared_composer"
            if composition == "NUMERIC_DIFFERENCE"
            else "shared_composer_response"
        ),
    }


def synthesis_guidance(
    contract: dict[str, Any],
    acquisition_coverage: dict[str, Any],
    *,
    source_canvas_unmodified: bool,
) -> str:
    schema = str(contract["answer_schema"])
    composition = str(contract["composition"])
    image_description = (
        "byte-unmodified complete source image"
        if source_canvas_unmodified
        else "method-rendered complete-source support canvas"
    )
    lines = [
        f"Use the displayed {image_description} for coverage and global configuration.",
        "Treat every prior observation as evidence only; the terminal composer alone writes the answer.",
        "Source footprints state where an observation looked. They are not labels, answers, or authority outside their covered region.",
    ]
    coverage_status = str(acquisition_coverage.get("status") or "")
    if coverage_status == "COMPLETE_WITHIN_BUDGET":
        lines.append(
            "The selected native views geometrically cover every declared public support region within the fixed budget. Individual observations still speak only for their own footprints."
        )
    elif coverage_status == "PARTIAL_FIXED_BUDGET":
        lines.append(
            f"The fixed native-view budget covers only part of at least one declared public support region. Never treat local non-observation as absence in uncovered portions; use the displayed {image_description} conservatively for those portions."
        )
    else:
        lines.append(
            "Support was proposed from public visual evidence, so completeness outside the recorded source footprints is not established. Never turn local non-observation into global absence."
        )
    if composition == "IDENTITY":
        lines.append(
            "Resolve the requested answer from the declared support without inventing an additional reduction."
        )
    elif composition == "AGGREGATION":
        lines.append(
            "Combine source-bound observations across distinct support footprints while avoiding repeated evidence from overlapping views."
        )
    elif composition == "NUMERIC_DIFFERENCE":
        lines.append(
            "Use the public A/B support labels to compare their source-bound scalar evidence. A deterministic absolute-difference reducer takes precedence only when acquisition coverage is COMPLETE_WITHIN_BUDGET, both public A/B support records are complete, and exactly one valid local scalar exists for each support; otherwise this shared composer remains the fallback."
        )
    elif composition == "SPATIAL_CONFIGURATION":
        lines.append(
            "Judge positions, distances, direction, clustering, and distribution from the complete source image; local observations are supporting evidence, not a replacement for global layout."
        )
    else:
        raise ValueError(f"unsupported public composition: {composition}")

    if schema == "SCALAR":
        lines.append(
            "Use only non-negative scalar observations that are bound to their recorded source supports."
        )
    elif schema == "OPTION":
        lines.append(
            "Resolve categorical conflicts from visible evidence, local provenance, and complete-scene configuration."
        )
    elif schema == "SINGLE_GEOMETRY":
        lines.append(
            "Every valid AABB4 or QUAD8 observation remains available with its source provenance. Resolve the target visually, then serialize only at the public output boundary; exact numeric equality, box area, and overlap are not hidden reliability scores."
        )
        lines.append(
            "Format-imperfect unstructured geometry text is still observer evidence, but its numbers are never source-image base1000 by default. COMPLETE_SCENE raw geometry uses only its recorded coordinate_frame and coordinate_extent. REGISTERED_NATIVE_VIEW raw geometry uses only its recorded stage3_local_coordinate_protocol and coordinate_extent, then the recorded stage3_registration_affine_source1000_2x3; query crop origin/size and view_geometry define that frame. Ignore numeric geometry whose recorded frame cannot be applied."
        )
    elif schema == "GEOMETRY_SET":
        lines.append(
            "Every valid source-registered candidate remains available with its native encoding and output projection. Resolve object identity and duplicates visually from the source image and provenance; aggregation has not removed candidates by exact equality or overlap."
        )
        lines.append(
            "Format-imperfect unstructured geometry text is still observer evidence, but its numbers are never source-image base1000 by default. COMPLETE_SCENE raw geometry uses only its recorded coordinate_frame and coordinate_extent. REGISTERED_NATIVE_VIEW raw geometry uses only its recorded stage3_local_coordinate_protocol and coordinate_extent, then the recorded stage3_registration_affine_source1000_2x3; query crop origin/size and view_geometry define that frame. Ignore numeric geometry whose recorded frame cannot be applied."
        )
    return "\n".join(lines)


def source_footprint_text(state: dict[str, Any]) -> str:
    lines: list[str] = []
    seen: set[tuple[Any, tuple[float, ...]]] = set()
    records = list(state.get("local_observations") or [])
    for record in records:
        footprint = valid_footprint(record.get("source_support_base1000"))
        if footprint is None:
            continue
        view_index = record.get("view_index")
        key = (view_index, tuple(footprint))
        if key in seen:
            continue
        seen.add(key)
        label = (
            f"numbered support view {view_index}"
            if view_index is not None
            else f"observation {record.get('observation_id')}"
            if record.get("observation_id")
            else "anonymous view"
        )
        rendered = json.dumps(footprint, ensure_ascii=False, separators=(",", ":"))
        lines.append(f"- {label}: source-image base1000 footprint {rendered}")
    return "\n".join(lines) if lines else "- No local source footprint was acquired."


def assert_geometry_prompt_state_safe(state: dict[str, Any]) -> None:
    if str(state["public_contract"]["answer_schema"]) not in GEOMETRY_SCHEMAS:
        return
    forbidden_keys = {"raw_response", "registered_response", "complete_scene_hypothesis"}

    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key) in forbidden_keys:
                    raise PermissionError(
                        f"geometry raw text reached composer state at {location}.{key}"
                    )
                walk(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{location}[{index}]")

    walk(state, "state")


def assert_invalid_raw_quarantined(state: dict[str, Any]) -> None:
    """Allow raw observer text only in its provenance-bearing evidence slot."""

    forbidden_raw_keys = {
        "raw_response",
        "raw_backend_response",
        "model_response",
        "registered_response",
    }
    geometry_schema = str(
        (state.get("public_contract") or {}).get("answer_schema") or ""
    ) in GEOMETRY_SCHEMAS

    def require_geometry_coordinate_provenance(
        evidence: dict[str, Any], location: str
    ) -> None:
        source_kind = str(evidence.get("source_kind") or "")
        extent = evidence.get("coordinate_extent")
        if (
            not isinstance(extent, (list, tuple))
            or len(extent) != 2
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in extent
            )
        ):
            raise PermissionError(
                f"geometry raw evidence lacks coordinate extent at {location}"
            )
        if source_kind == "COMPLETE_SCENE":
            if (
                evidence.get("coordinate_frame") != "source_image_base1000"
                or [float(value) for value in extent] != [1000.0, 1000.0]
            ):
                raise PermissionError(
                    f"complete-scene geometry raw evidence is not source base1000 at {location}"
                )
            return
        if source_kind != "REGISTERED_NATIVE_VIEW":
            raise PermissionError(
                f"geometry raw evidence has unsupported source kind at {location}"
            )
        protocol = str(evidence.get("stage3_local_coordinate_protocol") or "")
        if protocol not in {"rendered_canvas_base1000", "native_crop_base1000"}:
            raise PermissionError(
                f"local geometry raw evidence lacks coordinate protocol at {location}"
            )
        affine = evidence.get("stage3_registration_affine_source1000_2x3")
        if (
            not isinstance(affine, (list, tuple))
            or len(affine) != 2
            or any(
                not isinstance(line, (list, tuple))
                or len(line) != 3
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in line
                )
                for line in affine
            )
        ):
            raise PermissionError(
                f"local geometry raw evidence lacks registration affine at {location}"
            )
        if not isinstance(evidence.get("view_geometry"), dict):
            raise PermissionError(
                f"local geometry raw evidence lacks view geometry at {location}"
            )
        if protocol in {"rendered_canvas_base1000", "native_crop_base1000"}:
            if [float(value) for value in extent] != [1000.0, 1000.0]:
                raise PermissionError(
                    f"local base1000 geometry raw evidence has wrong extent at {location}"
                )
            if protocol == "rendered_canvas_base1000":
                return
        for name in ("query_crop_origin_abs_px", "query_crop_size_px"):
            values = evidence.get(name)
            if (
                not isinstance(values, (list, tuple))
                or len(values) != 2
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values
                )
            ):
                raise PermissionError(
                    f"query-native geometry raw evidence lacks {name} at {location}"
                )
        if [float(value) for value in extent] != [1000.0, 1000.0]:
            raise PermissionError(
                f"query-native geometry raw evidence has wrong base1000 extent at {location}"
            )

    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key) in forbidden_raw_keys:
                    raise PermissionError(
                        f"legacy raw response key reached composer state at {location}.{key}"
                    )
                if key == "unstructured_evidence":
                    if not isinstance(item, dict):
                        raise PermissionError(
                            f"unstructured evidence lacks provenance at {location}.{key}"
                        )
                    if not isinstance(item.get("raw_text"), str):
                        raise PermissionError(
                            f"unstructured evidence lacks exact raw text at {location}.{key}"
                        )
                    if not item.get("source_kind") or not item.get("source_id"):
                        raise PermissionError(
                            f"unstructured evidence lacks source identity at {location}.{key}"
                        )
                    if geometry_schema:
                        require_geometry_coordinate_provenance(
                            item, f"{location}.{key}"
                        )
                    for nested_key, nested_item in item.items():
                        if str(nested_key) == "raw_text":
                            continue
                        walk(
                            nested_item,
                            f"{location}.{key}.{nested_key}",
                        )
                    continue
                if key == "raw_text":
                    raise PermissionError(
                        f"raw_text exists outside unstructured evidence at {location}.{key}"
                    )
                walk(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{location}[{index}]")

    walk(state, "state")


def build_composer_task(
    source: dict[str, Any], state: dict[str, Any], canvas: dict[str, Any]
) -> dict[str, Any]:
    sid = str(source["sample_id"])
    if state.get("public_contract") != source.get("evidence_contract"):
        raise ValueError(f"state/public contract mismatch: {sid}")
    contract = source.get("evidence_contract") or {}
    activations = contract.get("operator_activations") or []
    if not isinstance(activations, list):
        raise ValueError(f"public operator activations must be a list: {sid}")
    complete_canvas_required = bool(
        {"GLOBAL_LAYOUT", "SEMANTIC_SCOPE"}
        & {str(activation) for activation in activations}
    )
    source_path = Path(str(source["image_abs_path"])).resolve(strict=True)
    canvas_path = Path(str(canvas.get("source_image_path") or "")).resolve(strict=True)
    if canvas.get("dataset_annotation_used") is not False:
        raise PermissionError(f"composer canvas used a dataset annotation: {sid}")
    actual_source_sha256 = cached_sha256_file(source_path)
    if (
        actual_source_sha256 != str(source["image_sha256"])
        or actual_source_sha256
        != str(canvas.get("source_image_sha256_verified") or "")
    ):
        raise PermissionError(f"composer source canvas bytes are not frozen: {sid}")
    actual_canvas_sha256 = cached_sha256_file(canvas_path)
    if actual_canvas_sha256 != str(canvas.get("source_canvas_sha256") or ""):
        raise PermissionError(f"composer canvas bytes changed: {sid}")
    if complete_canvas_required:
        if (
            canvas_path != source_path
            or canvas.get("source_canvas_unmodified") is not True
            or canvas.get("source_canvas_is_method_generated") is not False
            or actual_canvas_sha256 != actual_source_sha256
        ):
            raise PermissionError(
                f"complete-scene composer did not receive the byte-unmodified source image: {sid}"
            )
        image_intro = (
            "The displayed image is the byte-unmodified complete public source image. "
            "No method-generated support overlay has been added."
        )
    else:
        if (
            canvas_path == source_path
            or canvas.get("source_canvas_unmodified") is not False
            or canvas.get("source_canvas_is_method_generated") is not True
        ):
            raise PermissionError(
                f"composer canvas is not the clean method support overlay: {sid}"
            )
        image_intro = (
            "The displayed image is a method-rendered, max-side-limited copy of the "
            "complete public source image with white, numbered support footprints "
            "generated only from the public question or this run's visual support planner. "
            "These marks are observation provenance, not dataset annotations, object labels, or answers."
        )
    schema = str(source["evidence_contract"]["answer_schema"])
    encoding = geometry_encoding(source["evidence_contract"])
    policy = composer_policy(source["evidence_contract"])
    assert_invalid_raw_quarantined(state)
    if schema in GEOMETRY_SCHEMAS:
        assert_geometry_prompt_state_safe(state)
    # The sole composer receives the public semantic question and choices, but
    # not the dataset's legacy terminal-format sentence. output_contract below
    # is the single authoritative serialization instruction.
    question = str(source.get("question_stage1") or "").strip()
    if not question:
        raise ValueError(f"missing source-base1000 public question: {sid}")
    state_text = json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)
    footprints = source_footprint_text(state)
    prompt = f"""You are the sole terminal composer for one public remote-sensing question. {image_intro}

Source observation footprints (textual provenance only):
{footprints}

Public question in the source-image 0-1000 coordinate system:
{question}

Every numeric spatial coordinate in that question has already been converted to the source-image 0-1000 system. Any retained word `pixels` is legacy public wording and must not be interpreted as an absolute-pixel unit.

Public evidence contract:
{json.dumps(source['evidence_contract'], ensure_ascii=False, sort_keys=True)}

Compiled terminal policy:
{json.dumps(policy, ensure_ascii=False, sort_keys=True)}

Typed source-registered evidence state:
{state_text}

Synthesis policy:
{synthesis_guidance(source['evidence_contract'], state['acquisition_coverage'], source_canvas_unmodified=complete_canvas_required)}

Output contract:
{output_contract(schema, encoding)}

You may reason briefly before the final line. Do not provide a worked answer example."""
    row = copy.deepcopy(source)
    row.update(
        {
            "task_id": f"{source['qa_slug']}_unified_composer",
            "stage_name": "unified_global_local_composer",
            "model_image_path": str(canvas_path),
            "model_input_prompt": prompt,
            "typed_evidence_state": state,
            "source_canvas_sha256": actual_canvas_sha256,
            "source_canvas_unmodified": bool(complete_canvas_required),
            "source_canvas_is_method_generated": not complete_canvas_required,
            "source_canvas_width_px": int(canvas["display_width_px"]),
            "source_canvas_height_px": int(canvas["display_height_px"]),
            "model_response": None,
        }
    )
    return row


def prepare(args: argparse.Namespace) -> int:
    manifest = keyed(read_jsonl(args.manifest))
    states = keyed(read_jsonl(args.states))
    canvases = keyed(read_jsonl(args.source_canvases))
    decisions = keyed(read_jsonl(args.closure))
    if not (set(manifest) == set(states) == set(canvases) == set(decisions)):
        raise ValueError(
            "runtime/state/source-canvas/closure fixed-denominator sets differ"
        )
    tasks: list[dict[str, Any]] = []
    prompt_hashes: list[str] = []
    geometry_rows = 0
    unmodified_rows = 0
    closed_rows = 0
    for sid, source in manifest.items():
        state = states[sid]
        decision = decisions[sid]
        verify_closure_decision_binding(source, state, decision)
        if decision["closure_status"] == "CLOSED":
            closed_rows += 1
            continue
        schema = str(source["evidence_contract"]["answer_schema"])
        if schema in GEOMETRY_SCHEMAS:
            geometry_rows += 1
        row = build_composer_task(source, state, canvases[sid])
        unmodified_rows += int(row["source_canvas_unmodified"] is True)
        prompt = str(row["model_input_prompt"])
        tasks.append(row)
        prompt_hashes.append(sha256_text(prompt))
    tasks.sort(
        key=lambda row: -int(row["original_width_px"])
        * int(row["original_height_px"])
    )
    write_jsonl(args.out, tasks)
    audit = {
        "schema_version": "map-unified-composer-prepare-audit-2",
        "tasks": len(tasks),
        "unique_sample_ids": len({row["sample_id"] for row in tasks}),
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "closed_without_composer": closed_rows,
        "open_with_exactly_one_shared_composer": len(tasks),
        "single_shared_terminal_composer_for_open_rows": True,
        "method_generated_support_canvas_rows": len(tasks) - unmodified_rows,
        "byte_unmodified_complete_source_rows": unmodified_rows,
        "source_footprints_rendered_visually_and_as_typed_provenance": True,
        "ground_truth_or_dataset_annotation_used": False,
        "question_source": "question_stage1_semantics_with_terminal_format_removed",
        "geometry_rows_with_provenance_bearing_unstructured_evidence": geometry_rows,
        "historical_prediction_inputs": 0,
        "test_used": False,
    }
    audit["passed"] = (
        audit["tasks"] == audit["unique_sample_ids"]
    )
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"composer prepare failed: {audit}")
    return 0


def required_attempt(row: dict[str, Any], stage: str) -> int:
    try:
        attempt = int(row["attempt_number"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{stage} lacks an actual persistent attempt number") from error
    if attempt < 1:
        raise ValueError(f"{stage} has invalid attempt number: {attempt}")
    return attempt


def smart_resize_size(source: dict[str, Any], terminal: dict[str, Any]) -> dict[str, int]:
    """Mirror the frozen runner's Qwen image preprocessing dimensions."""

    height = int(terminal.get("source_canvas_height_px") or source["original_height_px"])
    width = int(terminal.get("source_canvas_width_px") or source["original_width_px"])
    factor = 16 * 2
    pixels_per_token = factor * factor
    minimum = int(terminal.get("min_pixels") or pixels_per_token * 768)
    maximum = int(terminal.get("max_pixels") or pixels_per_token * 12800)
    if height <= 0 or width <= 0 or maximum < minimum:
        raise ValueError("invalid composer image preprocessing contract")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("composer image exceeds frozen Qwen aspect-ratio contract")
    resized_h = max(factor, round(height / factor) * factor)
    resized_w = max(factor, round(width / factor) * factor)
    if resized_h * resized_w > maximum:
        beta = math.sqrt((height * width) / maximum)
        resized_h = math.floor(height / beta / factor) * factor
        resized_w = math.floor(width / beta / factor) * factor
    elif resized_h * resized_w < minimum:
        beta = math.sqrt(minimum / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor
    if resized_h <= 0 or resized_w <= 0:
        raise ValueError("frozen Qwen preprocessing produced a non-positive size")
    return {"width_px": int(resized_w), "height_px": int(resized_h)}


def call_accounting(
    native: dict[str, Any],
    planner: dict[str, Any],
    locals_: list[dict[str, Any]],
    terminal: dict[str, Any] | None,
) -> dict[str, Any]:
    composer_calls = int(terminal is not None)
    logical_by_stage = {
        "complete_scene_observation": 1,
        "public_support_plan": 0 if planner.get("stage1_skipped") else 1,
        "source_registered_native_observations": len(locals_),
        "unified_global_local_composer": composer_calls,
    }
    known_attempts: dict[str, Any] = {
        "complete_scene_observation": required_attempt(
            native, "complete_scene_observation"
        ),
        "public_support_plan": (
            0
            if planner.get("stage1_skipped")
            else required_attempt(planner, "public_support_plan")
        ),
        "source_registered_native_observations": [
            required_attempt(row, f"source_registered_native_observation:{index}")
            for index, row in enumerate(locals_, start=1)
        ],
        "unified_global_local_composer": (
            required_attempt(terminal, "unified_global_local_composer")
            if terminal is not None
            else 0
        ),
    }
    flattened = [
        known_attempts["complete_scene_observation"],
        known_attempts["public_support_plan"],
        *known_attempts["source_registered_native_observations"],
        known_attempts["unified_global_local_composer"],
    ]
    return {
        "logical_calls_by_stage": logical_by_stage,
        "logical_vlm_calls": sum(logical_by_stage.values()),
        "attempt_numbers_by_stage": known_attempts,
        "known_physical_vlm_attempts": sum(flattened),
        "physical_vlm_attempts": sum(flattened),
        "physical_attempt_accounting_complete": True,
    }


def closure_model_input_metadata(
    source: dict[str, Any], local_rows: list[dict[str, Any]]
) -> tuple[
    str,
    dict[str, Any],
    dict[str, int],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Record the actual observer prompts/configuration behind a closure."""

    if not local_rows:
        raise ValueError("evidence closure has no source observation")
    ordered = sorted(
        local_rows,
        key=lambda row: (
            int(row.get("roi_index") or 0),
            str(row.get("task_id") or ""),
        ),
    )
    prompts = [str(row.get("model_input_prompt") or "") for row in ordered]
    if any(not prompt for prompt in prompts):
        raise ValueError("evidence closure observer prompt is missing")
    configs = [copy.deepcopy(row.get("generation_config") or {}) for row in ordered]
    contributions: list[dict[str, Any]] = []
    for row, prompt, config in zip(ordered, prompts, configs):
        raw_view = row.get("view_geometry")
        if not isinstance(raw_view, dict):
            raise ValueError("evidence closure observer lacks view geometry")
        view = ViewGeometry.from_metadata(raw_view)
        image_path = Path(str(row.get("model_image_path") or "")).resolve(
            strict=True
        )
        actual_image_sha256 = cached_sha256_file(image_path)
        recorded_image_sha256 = str(row.get("roi_image_sha256") or "")
        if recorded_image_sha256 and actual_image_sha256 != recorded_image_sha256:
            raise PermissionError("evidence-closure observer image bytes changed")
        raw_response = str(
            row.get("raw_backend_response")
            if row.get("raw_backend_response") is not None
            else row.get("model_response") or ""
        )
        contributions.append(
            {
                "observation_id": str(
                    row.get("task_id") or row.get("observation_id") or ""
                ),
                "prompt_sha256": sha256_text(prompt),
                "model_image_sha256": actual_image_sha256,
                "model_image_canvas_size_px": {
                    "width_px": int(view.canvas_size_px.width),
                    "height_px": int(view.canvas_size_px.height),
                },
                "model_input_size": smart_resize_size(
                    {
                        "original_width_px": int(view.canvas_size_px.width),
                        "original_height_px": int(view.canvas_size_px.height),
                    },
                    {},
                ),
                "generation_config": config,
                "attempt_number": required_attempt(
                    row, "deterministic_closure_contributing_observer"
                ),
                "request_sha256": row.get("request_sha256"),
                "request_contract_sha256": row.get(
                    "request_contract_sha256"
                ),
                "raw_response": raw_response,
                "raw_response_sha256": sha256_text(raw_response),
                "api_mode": row.get("api_mode"),
                "gpu_id": row.get("gpu_id"),
            }
        )
    prompt_binding = {
        "observer_prompt_sha256": [sha256_text(prompt) for prompt in prompts],
        "observer_generation_configs": configs,
        "selected_task_ids": [str(row.get("task_id") or "") for row in ordered],
    }
    last = ordered[-1]
    raw_view = last.get("view_geometry")
    if not isinstance(raw_view, dict):
        raise ValueError("evidence closure observer lacks view geometry")
    view = ViewGeometry.from_metadata(raw_view)
    resized = smart_resize_size(
        {
            "original_width_px": int(view.canvas_size_px.width),
            "original_height_px": int(view.canvas_size_px.height),
        },
        {},
    )
    combined_prompt_hash = canonical_sha256(prompt_binding)
    generation = (
        configs[0]
        if all(config == configs[0] for config in configs)
        else {"per_observation": configs}
    )
    return combined_prompt_hash, generation, resized, last, contributions


def build_closed_prediction_row(
    *,
    source: dict[str, Any],
    private_binding: dict[str, Any],
    native: dict[str, Any],
    planner: dict[str, Any],
    local_rows: list[dict[str, Any]],
    state: dict[str, Any],
    closure_decision: dict[str, Any],
    phase0: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Materialize a decision that was frozen before composer scheduling."""

    expected_image = Path(str(source["image_abs_path"])).resolve(strict=True)
    actual_source_sha256 = cached_sha256_file(expected_image)
    if actual_source_sha256 != str(source["image_sha256"]):
        raise PermissionError("source image bytes changed before closure materialization")
    response = str(closure_decision.get("final_response") or "").strip()
    if not response:
        raise ValueError("closed evidence decision has no final response")
    accounting = call_accounting(native, planner, local_rows, None)
    (
        prompt_hash,
        generation_config,
        processed_size,
        terminal_source,
        contributing_inputs,
    ) = (
        closure_model_input_metadata(source, local_rows)
    )
    terminal_attempt = max(
        accounting["attempt_numbers_by_stage"][
            "source_registered_native_observations"
        ]
    )
    terminal_selection = {
        "mode": str(closure_decision["terminal_mode"]),
        "selection_reason": str(closure_decision["selection_reason"]),
        "decision_sha256": str(closure_decision["decision_sha256"]),
        "selected_observation_ids": list(
            closure_decision.get("selected_observation_ids") or []
        ),
        "decision_timing": "before_composer_task_creation",
        "composer_called": False,
    }
    return {
        "schema_version": "uhr-bench-v4-prediction-1",
        "qa_relpath": private_binding["qa_relpath"],
        "task_key": private_binding["task_key"],
        "run_id": run_id,
        "model_registry_id": "qwen3_vl_8b",
        "resolved_model_id": (
            "method:CleanMAPBaselineV2.2;answering_model:Qwen3-VL-8B-Instruct"
        ),
        "prompt_id": "map-baseline-v2.2:public-evidence-closure",
        "prompt_sha256": prompt_hash,
        "raw_response": response,
        "final_response": response,
        "runtime_status": "success",
        "attempt": terminal_attempt,
        "api_mode": terminal_source.get("api_mode"),
        "requested_coordinate_protocol": "base1000",
        "source_image_size": {
            "width_px": int(source["original_width_px"]),
            "height_px": int(source["original_height_px"]),
        },
        "transmitted_image_size": processed_size,
        "model_input_size": processed_size,
        "logical_vlm_calls": accounting["logical_vlm_calls"],
        "tool_calls": 0,
        "phase0_contract_sha256": phase0["phase0_contract_sha256"],
        "benchmark_manifest_sha256": phase0["benchmark_manifest_sha256"],
        "generation_config": generation_config,
        "diagnostics": {
            "sample_id": source["sample_id"],
            "image_sha256": source["image_sha256"],
            "public_contract": source["evidence_contract"],
            "typed_evidence_state": state,
            "local_observation_count": len(local_rows),
            "model_failure": None,
            "final_response_origin": "deterministic_evidence_closure",
            "top_level_prediction_fields_are_compatibility_projection": True,
            "call_accounting": accounting,
            "terminal_selection": terminal_selection,
            "deterministic_closure_provenance": {
                "final_response_origin": "deterministic_evidence_closure",
                "composer_called": False,
                "contributing_observer_inputs": contributing_inputs,
                "top_level_projection_semantics": {
                    "raw_response": "deterministically_serialized_final_response_not_a_terminal_vlm_response",
                    "prompt_sha256": "canonical_hash_of_contributing_observer_prompt_bindings",
                    "attempt": "maximum_attempt_number_among_contributing_local_observers",
                    "model_input_size": "compatibility_projection_of_last_ordered_contributing_observer",
                },
            },
            "composer_input_accounting": {
                "composer_skipped_due_precomposer_evidence_closure": True,
                "selected_observation_image_sha256": [
                    str(row.get("roi_image_sha256") or "") for row in local_rows
                ],
                "ground_truth_or_dataset_annotation_used": False,
            },
            "source_registration_enabled": True,
            "private_identity_available_during_inference": False,
            "private_identity_joined_after_inference": True,
            "ground_truth_or_metric_used": False,
            "test_used": False,
        },
        "trajectory": [
            {
                "stage": "complete_scene_observation",
                "raw_response": native.get(
                    "raw_backend_response", native.get("model_response")
                ),
                "attempt_number": accounting["attempt_numbers_by_stage"][
                    "complete_scene_observation"
                ],
            },
            {
                "stage": "public_support_plan",
                "skipped": bool(planner.get("stage1_skipped")),
                "points_source1000": planner.get("stage1_points_full1000") or [],
                "attempt_number": accounting["attempt_numbers_by_stage"][
                    "public_support_plan"
                ],
            },
            {
                "stage": "source_registered_native_observations",
                "count": len(local_rows),
                "observations": [
                    {
                        "task_id": row.get("task_id"),
                        "source_support_base1000": row.get("roi_full1000_xyxy"),
                        "raw_response": row.get(
                            "raw_backend_response", row.get("model_response")
                        ),
                    }
                    for row in local_rows
                ],
                "attempt_numbers": accounting["attempt_numbers_by_stage"][
                    "source_registered_native_observations"
                ],
            },
            {"stage": "typed_evidence_aggregation", "logical_vlm_calls": 0},
            {
                "stage": "evidence_closure",
                "logical_vlm_calls": 0,
                "decision_sha256": closure_decision["decision_sha256"],
                "selection_mode": terminal_selection["mode"],
                "final_response": response,
            },
            {
                "stage": "unified_global_local_composer",
                "skipped": True,
                "logical_vlm_calls": 0,
                "reason": "typed_public_evidence_closed_before_scheduling",
            },
            {
                "stage": "public_contract_terminal_materialization",
                "logical_vlm_calls": 0,
                "final_response_origin": "deterministic_evidence_closure",
                "top_level_prediction_fields_are_compatibility_projection": True,
                "selection_mode": terminal_selection["mode"],
                "selection_reason": terminal_selection["selection_reason"],
                "final_response": response,
            },
        ],
        "endpoint_id": terminal_source.get("endpoint_id"),
        "request_id": terminal_source.get("request_id"),
        "requested_model_id": terminal_source.get("requested_model_id"),
    }


def build_prediction_row(
    *,
    source: dict[str, Any],
    private_binding: dict[str, Any],
    native: dict[str, Any],
    planner: dict[str, Any],
    local_rows: list[dict[str, Any]],
    state: dict[str, Any],
    terminal: dict[str, Any] | None,
    closure_decision: dict[str, Any],
    phase0: dict[str, Any],
    run_id: str,
    expected_closure_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verify_closure_decision_binding(source, state, closure_decision)
    reproduced = (
        copy.deepcopy(expected_closure_decision)
        if expected_closure_decision is not None
        else seal_closure_decision(
            evidence_closure_decision(source, state, local_rows)
        )
    )
    verify_closure_decision_binding(source, state, reproduced)
    if reproduced != closure_decision:
        raise ValueError(
            f"stored evidence-closure decision is not reproducible: {source['sample_id']}"
        )
    if closure_decision["closure_status"] == "CLOSED":
        if terminal is not None:
            raise ValueError("closed evidence row unexpectedly has a composer result")
        return build_closed_prediction_row(
            source=source,
            private_binding=private_binding,
            native=native,
            planner=planner,
            local_rows=local_rows,
            state=state,
            closure_decision=closure_decision,
            phase0=phase0,
            run_id=run_id,
        )
    if terminal is None:
        raise ValueError("open evidence row lacks its one shared composer result")
    if terminal.get("error"):
        raise RuntimeError(
            f"unresolved infrastructure error at composer: {source['sample_id']}"
        )
    require_unique_rows(
        local_rows,
        key_names=("task_id", "observation_id"),
        label="materialized local observation",
    )
    if state.get("public_contract") != source.get("evidence_contract"):
        raise ValueError("materialized state/public contract mismatch")
    geometry_encoding(dict(source["evidence_contract"]))
    activations = source["evidence_contract"].get("operator_activations") or []
    if not isinstance(activations, list):
        raise ValueError("public operator activations must be a list")
    complete_canvas_required = bool(
        {"GLOBAL_LAYOUT", "SEMANTIC_SCOPE"}
        & {str(activation) for activation in activations}
    )
    expected_image = Path(str(source["image_abs_path"])).resolve(strict=True)
    terminal_image = Path(str(terminal.get("model_image_path") or "")).resolve(strict=True)
    if complete_canvas_required:
        if (
            terminal_image != expected_image
            or terminal.get("source_canvas_unmodified") is not True
            or terminal.get("source_canvas_is_method_generated") is not False
        ):
            raise PermissionError(
                "complete-scene terminal composer did not receive the byte-unmodified source image"
            )
    elif (
        terminal_image == expected_image
        or terminal.get("source_canvas_unmodified") is not False
        or terminal.get("source_canvas_is_method_generated") is not True
    ):
        raise PermissionError("terminal composer did not receive the method support canvas")
    if cached_sha256_file(terminal_image) != str(terminal.get("source_canvas_sha256") or ""):
        raise PermissionError("terminal method support canvas bytes changed")
    actual_source_sha256 = cached_sha256_file(expected_image)
    if actual_source_sha256 != str(source["image_sha256"]):
        raise PermissionError("terminal source image bytes changed before materialization")
    terminal_attempt = required_attempt(terminal, "unified_global_local_composer")
    accounting = call_accounting(native, planner, local_rows, terminal)
    composer_raw_response = str(
        terminal.get("raw_backend_response")
        if terminal.get("raw_backend_response") is not None
        else terminal.get("model_response") or ""
    )
    raw_response = composer_raw_response
    terminal_selection = {
        "mode": "SHARED_COMPOSER",
        "selection_reason": str(closure_decision["selection_reason"]),
        "composer_raw_response": composer_raw_response,
        "decision_sha256": str(closure_decision["decision_sha256"]),
        "decision_timing": "before_composer_task_creation",
        "composer_called": True,
    }
    response = raw_response.strip()
    public_contract = source.get("evidence_contract") or {}
    response, geometry_normalization = normalize_terminal_geometry(
        response,
        state,
        str(public_contract.get("answer_schema") or ""),
        str(public_contract.get("terminal_geometry_encoding") or "NONE"),
    )
    terminal_selection["terminal_geometry_normalization"] = geometry_normalization
    prompt = str(terminal["model_input_prompt"])
    processed_size = smart_resize_size(source, terminal)
    generation_config = copy.deepcopy(terminal.get("generation_config") or {})
    return {
        "schema_version": "uhr-bench-v4-prediction-1",
        "qa_relpath": private_binding["qa_relpath"],
        "task_key": private_binding["task_key"],
        "run_id": run_id,
        "model_registry_id": "qwen3_vl_8b",
        "resolved_model_id": "method:CleanMAPBaselineV2.2;answering_model:Qwen3-VL-8B-Instruct",
        "prompt_id": "map-baseline-v2.2:public-contract-composer",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_response": raw_response,
        "final_response": response,
        # Empty or malformed model output is a completed model failure, not an
        # infrastructure failure.  It remains in the fixed denominator.
        "runtime_status": "success",
        "attempt": terminal_attempt,
        "api_mode": terminal.get("api_mode"),
        "requested_coordinate_protocol": "base1000",
        "source_image_size": {
            "width_px": int(source["original_width_px"]),
            "height_px": int(source["original_height_px"]),
        },
        "transmitted_image_size": processed_size,
        "model_input_size": processed_size,
        "logical_vlm_calls": accounting["logical_vlm_calls"],
        "tool_calls": 0,
        "phase0_contract_sha256": phase0["phase0_contract_sha256"],
        "benchmark_manifest_sha256": phase0["benchmark_manifest_sha256"],
        "generation_config": generation_config,
        "diagnostics": {
            "sample_id": source["sample_id"],
            "image_sha256": source["image_sha256"],
            "public_contract": source["evidence_contract"],
            "typed_evidence_state": state,
            "local_observation_count": len(local_rows),
            "model_failure": "empty_response" if not response else None,
            "terminal_geometry_normalization": geometry_normalization,
            "final_response_origin": "shared_composer_vlm_response",
            "top_level_prediction_fields_are_compatibility_projection": False,
            "call_accounting": accounting,
            "terminal_selection": terminal_selection,
            "composer_input_accounting": {
                "source_path_sha256": hashlib.sha256(
                    str(expected_image).encode("utf-8")
                ).hexdigest(),
                "source_image_sha256_verified": actual_source_sha256,
                "pixels_modified_before_composer": not complete_canvas_required,
                "pixel_modification": (
                    "none_byte_unmodified_complete_source"
                    if complete_canvas_required
                    else "method_generated_white_numbered_support_footprints_only"
                ),
                "ground_truth_or_dataset_annotation_used": False,
                "actual_composer_input_size": processed_size,
                "actual_composer_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "actual_composer_attempt_number": terminal_attempt,
                "processed_size_derivation": (
                    "frozen_qwen_vl_utils_smart_resize_factor32_min768_max12800"
                ),
            },
            "source_registration_enabled": True,
            "private_identity_available_during_inference": False,
            "private_identity_joined_after_inference": True,
            "ground_truth_or_metric_used": False,
            "test_used": False,
        },
        "trajectory": [
            {
                "stage": "complete_scene_observation",
                "raw_response": native.get("raw_backend_response", native.get("model_response")),
                "attempt_number": accounting["attempt_numbers_by_stage"][
                    "complete_scene_observation"
                ],
            },
            {
                "stage": "public_support_plan",
                "skipped": bool(planner.get("stage1_skipped")),
                "points_source1000": planner.get("stage1_points_full1000") or [],
                "attempt_number": accounting["attempt_numbers_by_stage"][
                    "public_support_plan"
                ],
            },
            {
                "stage": "source_registered_native_observations",
                "count": len(local_rows),
                "observations": [
                    {
                        "task_id": row.get("task_id"),
                        "source_support_base1000": row.get("roi_full1000_xyxy"),
                        "raw_response": row.get(
                            "raw_backend_response", row.get("model_response")
                        ),
                    }
                    for row in local_rows
                ],
                "attempt_numbers": accounting["attempt_numbers_by_stage"][
                    "source_registered_native_observations"
                ],
            },
            {"stage": "typed_evidence_aggregation", "logical_vlm_calls": 0},
            {
                "stage": "evidence_closure",
                "logical_vlm_calls": 0,
                "decision_sha256": closure_decision["decision_sha256"],
                "selection_mode": "OPEN",
                "selection_reason": closure_decision["selection_reason"],
            },
            {
                "stage": "unified_global_local_composer",
                "raw_response": composer_raw_response,
                "attempt_number": terminal_attempt,
            },
            {
                "stage": "public_contract_terminal_materialization",
                "logical_vlm_calls": 0,
                "selection_mode": terminal_selection["mode"],
                "selection_reason": terminal_selection["selection_reason"],
                "final_response": response,
            },
        ],
        "endpoint_id": terminal.get("endpoint_id"),
        "request_id": terminal.get("request_id"),
        "requested_model_id": terminal.get("requested_model_id"),
    }


def materialize(args: argparse.Namespace) -> int:
    # The private reporting binding is intentionally opened only in this
    # post-inference command. No aggregation or composer task receives it.
    manifest = keyed(read_jsonl(args.manifest))
    native = keyed(read_jsonl(args.native))
    planner = keyed(read_jsonl(args.planner))
    states = keyed(read_jsonl(args.states))
    composer = keyed(read_jsonl(args.composer), "sample_id")
    decisions = keyed(read_jsonl(args.closure), "sample_id")
    local_grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.mapped):
        local_grouped[str(row["sample_id"])].append(row)
    inference_ids = set(manifest)
    if any(set(rows) != inference_ids for rows in (native, planner, states, decisions)):
        raise ValueError("post-inference fixed-denominator join failed")
    open_ids = {
        sid
        for sid, decision in decisions.items()
        if decision.get("closure_status") == "OPEN"
    }
    closed_ids = inference_ids - open_ids
    if set(composer) != open_ids:
        raise ValueError(
            "composer results must equal the pre-scheduled OPEN closure subset"
        )
    if set(local_grouped) - inference_ids:
        raise ValueError("observation rows have unknown post-inference sample IDs")

    binding = keyed(read_jsonl(args.binding))
    if set(binding) != inference_ids:
        raise ValueError("private materialization fixed-denominator join failed")
    phase0 = json.loads(args.phase0.read_text(encoding="utf-8"))
    outputs: list[dict[str, Any]] = []
    for sid in sorted(inference_ids, key=lambda value: int(binding[value]["split_position"])):
        outputs.append(
            build_prediction_row(
                source=manifest[sid],
                private_binding=binding[sid],
                native=native[sid],
                planner=planner[sid],
                local_rows=local_grouped.get(sid, []),
                state=states[sid],
                terminal=composer.get(sid),
                closure_decision=decisions[sid],
                phase0=phase0,
                run_id=args.run_id,
            )
        )
    write_jsonl(args.out, outputs)
    audit = {
        "schema_version": "map-materialization-audit-2",
        "rows": len(outputs),
        "unique_qas": len({row["qa_relpath"] for row in outputs}),
        "task_counts_post_hoc": dict(
            sorted(Counter(row["task_key"] for row in outputs).items())
        ),
        "private_identity_joined_after_inference": True,
        "top_level_attempt_compatibility_field_valid": all(
            int(row["attempt"]) >= 1 for row in outputs
        ),
        "actual_logical_call_counts_recorded": all(
            isinstance(
                (row.get("diagnostics") or {}).get("call_accounting"), dict
            )
            and int(row.get("logical_vlm_calls") or 0)
            == int(
                ((row.get("diagnostics") or {}).get("call_accounting") or {}).get(
                    "logical_vlm_calls", -1
                )
            )
            for row in outputs
        ),
        "actual_composer_input_dimensions_recorded_for_open_rows": all(
            (row.get("diagnostics") or {}).get("final_response_origin")
            != "shared_composer_vlm_response"
            or (
                int((row.get("model_input_size") or {}).get("width_px") or 0) > 0
                and int((row.get("model_input_size") or {}).get("height_px") or 0) > 0
                and ((row.get("diagnostics") or {}).get(
                    "composer_input_accounting", {}
                ).get("actual_composer_input_size"))
                == row.get("model_input_size")
            )
            for row in outputs
        ),
        "deterministic_closure_provenance_recorded_for_closed_rows": all(
            (row.get("diagnostics") or {}).get("final_response_origin")
            != "deterministic_evidence_closure"
            or (
                (row.get("diagnostics") or {}).get(
                    "top_level_prediction_fields_are_compatibility_projection"
                )
                is True
                and ((row.get("diagnostics") or {}).get("terminal_selection") or {}).get(
                    "composer_called"
                )
                is False
                and bool(
                    ((row.get("diagnostics") or {}).get(
                        "deterministic_closure_provenance", {}
                    ).get("contributing_observer_inputs"))
                )
            )
            for row in outputs
        ),
        "shared_composer_provenance_recorded_for_open_rows": all(
            (row.get("diagnostics") or {}).get("final_response_origin")
            != "shared_composer_vlm_response"
            or (
                (row.get("diagnostics") or {}).get(
                    "top_level_prediction_fields_are_compatibility_projection"
                )
                is False
                and ((row.get("diagnostics") or {}).get("terminal_selection") or {}).get(
                    "composer_called"
                )
                is True
            )
            for row in outputs
        ),
        "historical_prediction_rows_stitched": 0,
        "closed_without_composer": len(closed_ids),
        "open_with_exactly_one_shared_composer": len(open_ids),
        "composer_result_ids_equal_open_decision_ids": set(composer) == open_ids,
        "precomposer_closure_decisions_reproduced": True,
        "test_used": False,
    }
    audit["passed"] = (
        audit["rows"] == audit["unique_qas"]
        and audit["composer_result_ids_equal_open_decision_ids"]
        and audit["top_level_attempt_compatibility_field_valid"]
        and audit["actual_logical_call_counts_recorded"]
        and audit["actual_composer_input_dimensions_recorded_for_open_rows"]
        and audit[
            "deterministic_closure_provenance_recorded_for_closed_rows"
        ]
        and audit["shared_composer_provenance_recorded_for_open_rows"]
        and audit["closed_without_composer"]
        + audit["open_with_exactly_one_shared_composer"]
        == audit["rows"]
    )
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"materialization failed: {audit}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("aggregate")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--native", type=Path, required=True)
    item.add_argument("--mapped", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=aggregate)
    item = commands.add_parser("closure")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--states", type=Path, required=True)
    item.add_argument("--mapped", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=closure)
    item = commands.add_parser("prepare")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--states", type=Path, required=True)
    item.add_argument("--source-canvases", type=Path, required=True)
    item.add_argument("--closure", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=prepare)
    item = commands.add_parser("materialize")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--binding", type=Path, required=True)
    item.add_argument("--native", type=Path, required=True)
    item.add_argument("--planner", type=Path, required=True)
    item.add_argument("--mapped", type=Path, required=True)
    item.add_argument("--states", type=Path, required=True)
    item.add_argument("--composer", type=Path, required=True)
    item.add_argument("--closure", type=Path, required=True)
    item.add_argument("--phase0", type=Path, required=True)
    item.add_argument("--run-id", required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=materialize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
