#!/usr/bin/env python3
"""Compatibility CLI for affine-aware observation registration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
TOOLS = HERE.parents[1] / "runtime" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import register_outputs  # noqa: E402
from io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [register_outputs.register(row) for row in read_jsonl(args.stage3_run)]
    write_jsonl(args.out, rows)
    geometry = [
        row
        for row in rows
        if str(row.get("answer_serialization"))
        in register_outputs.GEOMETRY_SERIALIZATIONS
    ]
    audit = {
        "schema_version": "support-render-affine-registration-audit-1",
        "rows": len(rows),
        "geometry_rows": len(geometry),
        "parse_or_registration_failures": sum(
            not bool(row.get("stage3_parse_ok")) for row in geometry
        ),
        "padding_coordinate_rejections": sum(
            str(row.get("stage3_parse_error") or "").startswith("padding_coordinate:")
            for row in geometry
        ),
        "coordinate_guessing_used": False,
        "coordinate_clamping_used": False,
        "affine_recorded": all(
            row.get("stage3_registration_affine_source1000_2x3") is not None
            for row in geometry
            if row.get("stage3_parse_ok") and not row.get("stage3_null")
        ),
        "test_used": False,
    }
    audit["passed"] = audit["affine_recorded"]
    write_json(args.out.with_name("registration_audit.json"), audit)
    if not audit["passed"]:
        raise RuntimeError(f"affine registration audit failed: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

