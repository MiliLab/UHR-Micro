#!/usr/bin/env python3
"""Materialize MAP pairwise scalar operands and public absolute difference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def _coerce_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        return int(value)
    return None


def parse_operands(text: str) -> tuple[int | None, int | None, str]:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    for match in re.finditer(r"\{[^{}]*\}", cleaned, flags=re.DOTALL):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or set(payload) != {"r1", "r2"}:
            continue
        r1 = _coerce_nonnegative_int(payload.get("r1"))
        r2 = _coerce_nonnegative_int(payload.get("r2"))
        if r1 is not None and r2 is not None:
            return r1, r2, "json_object"
    patterns = {
        "r1": r"(?:\br1\b|\bregion\s*a\b|\bendpoint\s*r1\b)\s*[:=]\s*(\d+)",
        "r2": r"(?:\br2\b|\bregion\s*b\b|\bendpoint\s*r2\b)\s*[:=]\s*(\d+)",
    }
    values: dict[str, int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            values[key] = int(match.group(1))
    if set(values) == {"r1", "r2"}:
        return values["r1"], values["r2"], "labeled_native_text"
    return None, None, "invalid_or_incomplete"




def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--operand-outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-replacements", type=int, default=0)
    args = parser.parse_args()

    baseline = read_jsonl(args.baseline)
    receipts = [row for path in args.operand_outputs for row in read_jsonl(path)]
    receipt_by_id = {str(row["sample_id"]): row for row in receipts}
    expected_replacements = args.expected_replacements or len(receipt_by_id)
    if len(receipt_by_id) != expected_replacements:
        raise ValueError("unexpected unique operand receipt count")

    output: list[dict] = []
    admitted = fallbacks = 0
    parse_modes: dict[str, int] = {}
    activated: set[str] = set()
    for original in baseline:
        sample_id = str(original["diagnostics"]["sample_id"])
        receipt = receipt_by_id.get(sample_id)
        if receipt is None:
            output.append(original)
            continue
        row = dict(original)
        r1, r2, mode = parse_operands(str(receipt["model_response"]))
        parse_modes[mode] = parse_modes.get(mode, 0) + 1
        if r1 is not None and r2 is not None:
            response = str(abs(r1 - r2))
            terminal_mode = "DETERMINISTIC_ABSOLUTE_DIFFERENCE"
            admitted += 1
        else:
            response = str(original["final_response"])
            terminal_mode = "PRE_UPDATE_SUPPORTED_STATE_FALLBACK"
            fallbacks += 1
        row["raw_response"] = response
        row["final_response"] = response
        row["prompt_id"] = "map-v1:pairwise-scalar-operands"
        row["prompt_sha256"] = hashlib.sha256(receipt["model_input_prompt"].encode("utf-8")).hexdigest()
        row["logical_vlm_calls"] = int(original.get("logical_vlm_calls") or 0) + 1
        row["trajectory"] = list(original.get("trajectory") or []) + [{
            "stage": "map_scalar_algebra",
            "operator": "ABSOLUTE_DIFFERENCE",
            "source_boxes_1000": receipt["endpoint_source_boxes_1000"],
            "operand_r1": r1,
            "operand_r2": r2,
            "receipt_parse_mode": mode,
            "terminal_mode": terminal_mode,
            "raw_response": receipt["model_response"],
        }]
        diagnostics = dict(row.get("diagnostics") or {})
        accounting = dict(diagnostics.get("call_accounting") or {})
        logical = dict(accounting.get("logical_calls_by_stage") or {})
        attempts = dict(accounting.get("attempt_numbers_by_stage") or {})
        logical["map_scalar_operands"] = 1
        attempts["map_scalar_operands"] = int(receipt.get("attempt_number") or 1)
        if accounting:
            accounting["logical_calls_by_stage"] = logical
            accounting["attempt_numbers_by_stage"] = attempts
            accounting["logical_vlm_calls"] = row["logical_vlm_calls"]
            accounting["known_physical_vlm_attempts"] = int(accounting.get("known_physical_vlm_attempts") or 0) + int(receipt.get("attempt_number") or 1)
            accounting["physical_vlm_attempts"] = int(accounting.get("physical_vlm_attempts") or 0) + int(receipt.get("attempt_number") or 1)
            diagnostics["call_accounting"] = accounting
        diagnostics["map_scalar_algebra"] = {
            "activated": True,
            "operator": "ABSOLUTE_DIFFERENCE",
            "receipt_parse_mode": mode,
            "terminal_mode": terminal_mode,
            "request_sha256": receipt["request_sha256"],
            "ground_truth_or_metric_used": False,
            "test_used": False,
        }
        row["diagnostics"] = diagnostics
        output.append(row)
        activated.add(sample_id)

    if activated != set(receipt_by_id):
        raise ValueError("operand/baseline identity mismatch")
    write_jsonl(args.out, output)
    audit = {
        "schema_version": "map-pairwise-scalar-canonical-audit-1",
        "baseline_rows": len(baseline),
        "canonical_rows": len(output),
        "receipt_rows": len(receipt_by_id),
        "deterministic_operator_admissions": admitted,
        "fallback_rows": fallbacks,
        "parse_mode_counts": dict(sorted(parse_modes.items())),
        "logical_call_delta_total": len(receipt_by_id),
        "unique_qa_relpaths": len({row["qa_relpath"] for row in output}),
        "ground_truth_annotation_task_metric_used": False,
        "test_used": False,
        "canonical_sha256": sha256_file(args.out),
    }
    audit["passed"] = all([
        len(baseline) == len(output),
        len(receipt_by_id) == expected_replacements,
        admitted + fallbacks == expected_replacements,
        audit["unique_qa_relpaths"] == len(output),
    ])
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
