#!/usr/bin/env python3
"""Register native observation geometry into the source-image frame.

Internal observations may be AABB4 or QUAD8 independently of the public output
encoding.  Their native encoding and source affine are retained.  A separate,
deterministic output projection is computed for the terminal composer; it does
not overwrite the original registered evidence.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from io_utils import read_jsonl, write_json, write_jsonl

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from geometry_parser import (  # noqa: E402
    convert_geometry_encoding,
    normalize_geometry_encoding,
    parse_observation_geometry_response,
    valid,
)
from view_geometry import (  # noqa: E402
    PaddingCoordinateError,
    ViewGeometry,
    ViewGeometryError,
)


GEOMETRY_SERIALIZATIONS = frozenset(
    {"single_bbox", "multi_bbox", "segmentation"}
)
STANDARD_LOCAL_PROTOCOL = "rendered_canvas_base1000"
QUERY_NATIVE_LOCAL_PROTOCOL = "native_crop_base1000"


def render(values: Sequence[float]) -> str:
    """Render mapped values without integer coercion or four-decimal loss."""

    def number(value: float) -> str:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".12g")

    return "[" + ",".join(number(value) for value in values) + "]"


def map_values(
    view: ViewGeometry,
    values: Sequence[float],
    encoding: str,
    local_coordinate_protocol: str,
) -> list[float]:
    """Map exactly one geometry under its recorded observation encoding."""

    normalized = normalize_geometry_encoding(encoding)
    if local_coordinate_protocol in {
        STANDARD_LOCAL_PROTOCOL,
        QUERY_NATIVE_LOCAL_PROTOCOL,
    }:
        if normalized == "AABB4":
            return list(view.local1000_box_to_source1000(values))
        output: list[float] = []
        for index in range(0, 8, 2):
            x, y = view.local1000_point_to_source1000(values[index : index + 2])
            output.extend((x, y))
        return output
    raise ValueError(
        f"unsupported local_coordinate_protocol: {local_coordinate_protocol!r}"
    )


def source1000_affine(
    view: ViewGeometry, local_coordinate_protocol: str
) -> list[list[float]]:
    """Record the exact observer-coordinate to source1000 affine."""

    affine = view.canvas_to_source
    source_w = float(view.source_size_px.width)
    source_h = float(view.source_size_px.height)
    if local_coordinate_protocol in {
        STANDARD_LOCAL_PROTOCOL,
        QUERY_NATIVE_LOCAL_PROTOCOL,
    }:
        input_x_scale = float(view.canvas_size_px.width) / 1000.0
        input_y_scale = float(view.canvas_size_px.height) / 1000.0
    else:
        raise ValueError(
            f"unsupported local_coordinate_protocol: {local_coordinate_protocol!r}"
        )
    return [
        [
            affine.xx * input_x_scale * 1000.0 / source_w,
            affine.xy * input_y_scale * 1000.0 / source_w,
            affine.x_offset * 1000.0 / source_w,
        ],
        [
            affine.yx * input_x_scale * 1000.0 / source_h,
            affine.yy * input_y_scale * 1000.0 / source_h,
            affine.y_offset * 1000.0 / source_h,
        ],
    ]


def validate_local_coordinate_protocol(
    row: Mapping[str, Any], view: ViewGeometry
) -> tuple[str, tuple[float, float]]:
    """Validate the two public observer coordinate protocols without guessing."""

    protocol = str(row.get("local_coordinate_protocol") or "")
    if protocol == STANDARD_LOCAL_PROTOCOL:
        return protocol, (1000.0, 1000.0)
    if protocol != QUERY_NATIVE_LOCAL_PROTOCOL:
        raise ValueError(f"unsupported local_coordinate_protocol: {protocol!r}")
    try:
        origin = [float(value) for value in row["query_crop_origin_abs_px"]]
        size = [float(value) for value in row["query_crop_size_px"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid QUERY_NATIVE crop metadata") from error
    if len(origin) != 2 or len(size) != 2 or any(
        not math.isfinite(value) for value in [*origin, *size]
    ):
        raise ValueError("invalid QUERY_NATIVE crop metadata")
    if size[0] <= 0.0 or size[1] <= 0.0:
        raise ValueError("QUERY_NATIVE crop size must be positive")
    roi = view.source_roi_xyxy_px
    content = view.content_rect_canvas_xyxy_px
    canvas = view.canvas_size_px
    expected = (
        math.isclose(origin[0], roi.x0, abs_tol=1e-8),
        math.isclose(origin[1], roi.y0, abs_tol=1e-8),
        math.isclose(size[0], roi.width, abs_tol=1e-8),
        math.isclose(size[1], roi.height, abs_tol=1e-8),
        math.isclose(canvas.width, size[0], abs_tol=1e-8),
        math.isclose(canvas.height, size[1], abs_tol=1e-8),
        math.isclose(content.x0, 0.0, abs_tol=1e-8),
        math.isclose(content.y0, 0.0, abs_tol=1e-8),
        math.isclose(content.x1, size[0], abs_tol=1e-8),
        math.isclose(content.y1, size[1], abs_tol=1e-8),
    )
    if not all(expected):
        raise ValueError(
            "QUERY_NATIVE crop metadata/view geometry are not one byte-scale crop frame"
        )
    return protocol, (1000.0, 1000.0)


def encoding_label(records: Sequence[Mapping[str, Any]]) -> str | None:
    encodings = [str(record["geometry_encoding"]) for record in records]
    if not encodings:
        return None
    return encodings[0] if len(set(encodings)) == 1 else "MIXED"


def public_geometry_encoding(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Read terminal serialization explicitly and verify the legacy alias.

    ``geometry_encoding`` remains in the public contract for backward
    compatibility, but it is never authoritative here. A disagreement is a
    contract failure instead of an implicit choice between the two fields.
    """

    contract = row.get("evidence_contract")
    if not isinstance(contract, Mapping):
        return None, "missing_public_evidence_contract"
    if "terminal_geometry_encoding" not in contract:
        return None, "missing_terminal_geometry_encoding"
    if "geometry_encoding" not in contract:
        return None, "missing_geometry_encoding_alias"
    terminal = str(contract["terminal_geometry_encoding"])
    if str(contract["geometry_encoding"]) != terminal:
        return None, "geometry_encoding_alias_mismatch"
    if terminal == "NONE":
        return terminal, None
    try:
        return normalize_geometry_encoding(terminal), None
    except ValueError:
        return None, "unsupported_terminal_geometry_encoding"


def _base_output(row: dict[str, Any], text: str) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "stage3_mapped_response_full1000": text,
            "stage3_mapped_coord_lists_full1000": [],
            "stage3_geometry_encoding": None,
            "stage3_mapped_geometry_encoding": None,
            "stage3_observation_geometry_encodings": [],
            "stage3_output_geometry_encoding": None,
            "stage3_registered_geometries_full1000": [],
            "stage3_output_coord_lists_full1000": [],
            "stage3_output_response_full1000": None,
            "stage3_registration_affine_source1000_2x3": None,
            "stage3_local_coordinate_protocol": None,
            "stage3_parse_mode": None,
            "stage3_parse_ok": True,
            "stage3_parse_error": None,
            "stage3_registration_ok": True,
            "stage3_contract_error": False,
            "stage3_null": False,
            "invalid_local_geometry": [],
        }
    )
    return output


def _fail(
    output: dict[str, Any],
    error: str,
    *,
    contract_error: bool = False,
    invalid: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output.update(
        {
            "stage3_mapped_response_full1000": None,
            "stage3_mapped_coord_lists_full1000": [],
            "stage3_parse_ok": False,
            "stage3_parse_error": error,
            "stage3_registration_ok": False,
            "stage3_contract_error": bool(contract_error),
            "invalid_local_geometry": invalid or [],
        }
    )
    return output


def register(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("model_response") or "").strip()
    serialization = str(row.get("answer_serialization") or "")
    output = _base_output(row, text)
    if row.get("error"):
        return _fail(output, "infrastructure_error", contract_error=True)
    output_encoding, encoding_error = public_geometry_encoding(row)
    if encoding_error is not None or output_encoding is None:
        return _fail(
            output,
            encoding_error or "missing_terminal_geometry_encoding",
            contract_error=True,
        )
    if serialization not in GEOMETRY_SERIALIZATIONS:
        output["stage3_output_geometry_encoding"] = output_encoding
        return output

    if output_encoding not in {"AABB4", "QUAD8"}:
        return _fail(
            output,
            "geometry_serialization_requires_terminal_geometry_encoding",
            contract_error=True,
        )
    output["stage3_output_geometry_encoding"] = output_encoding

    try:
        view = ViewGeometry.from_metadata(dict(row["view_geometry"]))
        local_protocol, coordinate_extent = validate_local_coordinate_protocol(
            row, view
        )
    except (KeyError, TypeError, ValueError, ViewGeometryError) as error:
        return _fail(
            output,
            f"invalid_view_geometry_or_local_protocol:{type(error).__name__}:{error}",
            contract_error=True,
        )
    output["stage3_local_coordinate_protocol"] = local_protocol
    output["stage3_registration_affine_source1000_2x3"] = source1000_affine(
        view, local_protocol
    )

    parsed = parse_observation_geometry_response(
        text,
        allow_multiple=serialization == "multi_bbox",
        coordinate_extent=coordinate_extent,
    )
    output["stage3_parse_mode"] = parsed.get("mode")
    if not parsed["valid"]:
        invalid = [
            {
                "local_values": row.get("geometry"),
                "candidate_index": row.get("candidate_index"),
                "source": row.get("source"),
                "error": str(row.get("reason") or parsed["error"]),
            }
            for row in parsed.get("rejected_geometries") or []
        ]
        return _fail(output, str(parsed["error"]), invalid=invalid)
    output["invalid_local_geometry"] = [
        {
            "local_values": row.get("geometry"),
            "candidate_index": row.get("candidate_index"),
            "source": row.get("source"),
            "error": str(row.get("reason") or "rejected_observation_geometry"),
        }
        for row in parsed.get("rejected_geometries") or []
    ]
    if parsed.get("null"):
        output.update(
            {
                "stage3_mapped_response_full1000": "null",
                "stage3_output_response_full1000": "null",
                "stage3_null": True,
            }
        )
        return output

    local_records = list(parsed.get("geometry_records") or [])
    observation_encoding = encoding_label(local_records)
    observation_encodings = [
        str(record["geometry_encoding"]) for record in local_records
    ]
    output["stage3_geometry_encoding"] = observation_encoding
    output["stage3_mapped_geometry_encoding"] = observation_encoding
    output["stage3_observation_geometry_encodings"] = observation_encodings

    try:
        registered_records = []
        for record in local_records:
            local_values = [float(value) for value in record["geometry"]]
            native_encoding = str(record["geometry_encoding"])
            source_values = map_values(
                view, local_values, native_encoding, local_protocol
            )
            output_values = convert_geometry_encoding(
                source_values, native_encoding, output_encoding
            )
            registered_records.append(
                {
                    "local_geometry": local_values,
                    "geometry_encoding": native_encoding,
                    "geometry_source1000": source_values,
                    "output_geometry_encoding": output_encoding,
                    "output_geometry_source1000": output_values,
                }
            )
    except PaddingCoordinateError as error:
        return _fail(
            output,
            f"padding_coordinate:{error}",
            invalid=[
                {
                    "local_values": record["geometry"],
                    "geometry_encoding": record["geometry_encoding"],
                    "error": "padding_coordinate",
                }
                for record in local_records
            ],
        )
    except (TypeError, ValueError, ViewGeometryError) as error:
        return _fail(
            output,
            f"geometry_registration:{type(error).__name__}:{error}",
            invalid=[
                {
                    "local_values": record["geometry"],
                    "geometry_encoding": record["geometry_encoding"],
                    "error": f"{type(error).__name__}:{error}",
                }
                for record in local_records
            ],
        )

    if not all(
        valid(record["geometry_source1000"], record["geometry_encoding"])
        and valid(
            record["output_geometry_source1000"],
            record["output_geometry_encoding"],
        )
        for record in registered_records
    ):
        return _fail(
            output,
            "mapped_geometry_failed_recorded_encoding",
            contract_error=True,
            invalid=[
                {
                    "local_values": record["local_geometry"],
                    "geometry_encoding": record["geometry_encoding"],
                    "mapped_values": record["geometry_source1000"],
                    "output_values": record["output_geometry_source1000"],
                    "output_geometry_encoding": record[
                        "output_geometry_encoding"
                    ],
                }
                for record in registered_records
            ],
        )
    mapped = [record["geometry_source1000"] for record in registered_records]
    output_mapped = [
        record["output_geometry_source1000"] for record in registered_records
    ]
    output.update(
        {
            "stage3_mapped_coord_lists_full1000": mapped,
            "stage3_mapped_response_full1000": ",".join(
                render(values) for values in mapped
            ),
            "stage3_registered_geometries_full1000": registered_records,
            "stage3_output_coord_lists_full1000": output_mapped,
            "stage3_output_response_full1000": ",".join(
                render(values) for values in output_mapped
            ),
        }
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    rows = [register(row) for row in read_jsonl(args.results)]
    errors = Counter(
        str(row["stage3_parse_error"])
        for row in rows
        if row.get("stage3_parse_error")
    )
    geometry_rows = [
        row
        for row in rows
        if str(row.get("answer_serialization")) in GEOMETRY_SERIALIZATIONS
    ]
    audit = {
        "schema_version": "map-source-registration-audit-2",
        "rows": len(rows),
        "unique_observation_ids": len({str(row["task_id"]) for row in rows}),
        "unique_samples": len({str(row["sample_id"]) for row in rows}),
        "geometry_rows": len(geometry_rows),
        "public_output_geometry_encoding_counts": dict(
            sorted(
                Counter(
                    str(row.get("stage3_output_geometry_encoding"))
                    for row in geometry_rows
                ).items()
            )
        ),
        "observation_geometry_encoding_counts": dict(
            sorted(
                Counter(
                    encoding
                    for row in geometry_rows
                    for encoding in row.get(
                        "stage3_observation_geometry_encodings"
                    )
                ).items()
            )
        ),
        "mixed_observation_encoding_rows": sum(
            row.get("stage3_geometry_encoding") == "MIXED"
            for row in geometry_rows
        ),
        "alternate_encoding_observation_rows": sum(
            any(
                encoding != row.get("stage3_output_geometry_encoding")
                for encoding in row.get(
                    "stage3_observation_geometry_encodings"
                )
            )
            for row in geometry_rows
        ),
        "valid_registered_geometry_rows": sum(
            bool(row.get("stage3_mapped_coord_lists_full1000"))
            for row in geometry_rows
        ),
        "valid_null_geometry_rows": sum(
            bool(row.get("stage3_null")) for row in geometry_rows
        ),
        "invalid_geometry_candidates": sum(
            len(row.get("invalid_local_geometry") or []) for row in rows
        ),
        "contract_error_rows": sum(
            bool(row.get("stage3_contract_error")) for row in rows
        ),
        "infrastructure_error_rows": sum(
            row.get("stage3_parse_error") == "infrastructure_error"
            for row in rows
        ),
        "parse_error_counts": dict(sorted(errors.items())),
        "coordinate_guessing_used": False,
        "coordinate_clamping_used": False,
        "padding_coordinates_accepted": False,
        "observation_encoding_preserved_after_registration": all(
            all(
                record.get("geometry_encoding")
                in row.get("stage3_observation_geometry_encodings", [])
                and valid(
                    record.get("geometry_source1000") or [],
                    str(record.get("geometry_encoding") or ""),
                )
                for record in row.get(
                    "stage3_registered_geometries_full1000"
                )
            )
            for row in geometry_rows
        ),
        "deterministic_output_projection_valid": all(
            all(
                record.get("output_geometry_encoding")
                == row.get("stage3_output_geometry_encoding")
                and valid(
                    record.get("output_geometry_source1000") or [],
                    str(record.get("output_geometry_encoding") or ""),
                )
                for record in row.get(
                    "stage3_registered_geometries_full1000"
                )
            )
            for row in geometry_rows
        ),
        "source_affine_recorded_for_registered_rows": all(
            not row.get("stage3_registered_geometries_full1000")
            or row.get("stage3_registration_affine_source1000_2x3") is not None
            for row in geometry_rows
        ),
        "test_used": False,
    }
    audit["passed"] = (
        audit["rows"] == audit["unique_observation_ids"]
        and audit["contract_error_rows"] == 0
        and audit["infrastructure_error_rows"] == 0
        and audit["observation_encoding_preserved_after_registration"]
        and audit["deterministic_output_projection_valid"]
        and audit["source_affine_recorded_for_registered_rows"]
    )
    write_jsonl(args.out, rows)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"registration audit failed: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
