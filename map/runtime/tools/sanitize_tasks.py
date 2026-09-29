#!/usr/bin/env python3
"""Apply an explicit allowlist to every model-stage task manifest."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, write_json, write_jsonl


ALLOWED_TOP_LEVEL = {
    "schema_version",
    "sample_id",
    "qa_slug",
    "image_abs_path",
    "image_sha256",
    "original_width_px",
    "original_height_px",
    "question_original",
    "question_1000",
    "question_stage1",
    "question_roi",
    "question_roi_local",
    "question_rewritten",
    "seed_boxes_abs_from_question",
    "coord_lists_abs_from_question",
    "axis_thresholds_abs_from_question",
    "coordinate_protocol",
    "evidence_contract",
    "evidence_state",
    "execution_plan",
    "answer_serialization",
    "stage1_needed",
    "support_source",
    "max_stage1_points",
    "max_native_views",
    "provisional_format_hint",
    "stage1_template_key",
    "generation_contract",
    "task_id",
    "stage_name",
    "model_image_path",
    "model_input_prompt",
    "model_response",
    "view_geometry",
    "public_coordinate_rewrites",
    "roi_id",
    "roi_index",
    "roi_abs_xyxy",
    "roi_full1000_xyxy",
    "roi_width_px",
    "roi_height_px",
    "roi_source",
    "source_support_label",
    "native_view_budget",
    "public_support_coverage",
    "roi_image_path",
    "roi_image_format",
    "local_coordinate_protocol",
    "padding_semantics",
    "source_image_sha256",
    "source_image_sha256_verified",
    "render_spec_sha256",
    "render_spec",
    "semantic_input_sha256",
    "crop_image_sha256",
    "roi_image_sha256",
    "typed_evidence_state",
    "source_footprints",
    "source_canvas_unmodified",
    "source_canvas_sha256",
    "source_canvas_is_method_generated",
    "source_canvas_width_px",
    "source_canvas_height_px",
    "observer_generation_profile",
    "query_crop_origin_abs_px",
    "query_crop_size_px",
}
FORBIDDEN_PROMPT = re.compile(
    r"\b(?:v6|v7|v9|v10|v11|v12)\b|"
    r"Final\s+answer\s*:\s*(?:[ABCD]|\d|\[)|"
    r"ground\s*truth|validation\s+score|test\s+score|"
    r"annotation\s+color|colou?red\s+box\s+means",
    re.IGNORECASE,
)
FORBIDDEN_NESTED_FIELDS = frozenset(
    {
        "task_key",
        "task_family",
        "split",
        "answer",
        "annotation",
        "metric",
        "historical_version",
        "task_route",
    }
)
TYPED_STATE_START = "Typed source-registered evidence state:\n"
TYPED_STATE_END = "\n\nSynthesis policy:\n"
PROVENANCE_RAW_MASK = "<PROVENANCE_BOUND_MODEL_RAW_TEXT>"


def forbidden_prompt_match(prompt: str) -> re.Match[str] | None:
    """Strictly scan static text; no provenance exception is applied here."""

    return FORBIDDEN_PROMPT.search(str(prompt or ""))


def masked_provenance_state(state: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Mask only direct ``unstructured_evidence.raw_text`` values.

    This is a structural operation over the typed state, not a text replace.
    A raw string is exempt from the concrete-answer regex only when its direct
    parent is a provenance-bearing ``unstructured_evidence`` object.
    """

    masked_count = 0

    def walk(value: Any, *, inside_unstructured: bool = False) -> Any:
        nonlocal masked_count
        if isinstance(value, dict):
            output: dict[str, Any] = {}
            for key, item in value.items():
                name = str(key)
                if name == "unstructured_evidence":
                    if not isinstance(item, dict):
                        raise PermissionError(
                            "typed evidence contains non-object unstructured_evidence"
                        )
                    if not isinstance(item.get("raw_text"), str):
                        raise PermissionError(
                            "unstructured_evidence lacks exact raw_text"
                        )
                    if not item.get("source_kind") or not item.get("source_id"):
                        raise PermissionError(
                            "unstructured_evidence lacks provenance source identity"
                        )
                    output[name] = walk(item, inside_unstructured=True)
                elif name == "raw_text" and inside_unstructured:
                    if not isinstance(item, str):
                        raise PermissionError(
                            "unstructured_evidence raw_text is not a string"
                        )
                    output[name] = PROVENANCE_RAW_MASK
                    masked_count += 1
                elif name == "raw_text":
                    raise PermissionError(
                        "raw_text exists outside provenance-bound unstructured_evidence"
                    )
                else:
                    output[name] = walk(item, inside_unstructured=False)
            return output
        if isinstance(value, list):
            return [walk(item, inside_unstructured=False) for item in value]
        return copy.deepcopy(value)

    return walk(state), masked_count


def prompt_with_provenance_raw_masked(
    prompt: str, typed_state: dict[str, Any] | None
) -> tuple[str, int]:
    """Mask structurally verified raw-model spans in a composer prompt only."""

    text = str(prompt or "")
    if typed_state is None:
        if TYPED_STATE_START in text or TYPED_STATE_END in text:
            raise PermissionError(
                "composer typed-state section exists without typed_evidence_state"
            )
        return text, 0
    if not isinstance(typed_state, dict):
        raise PermissionError("typed_evidence_state must be an object")
    if text.count(TYPED_STATE_START) != 1 or text.count(TYPED_STATE_END) != 1:
        raise PermissionError("composer prompt lacks one exact typed-state span")
    start = text.index(TYPED_STATE_START) + len(TYPED_STATE_START)
    end = text.index(TYPED_STATE_END, start)
    expected_state = json.dumps(
        typed_state, ensure_ascii=False, sort_keys=True, indent=2
    )
    if text[start:end] != expected_state:
        raise PermissionError(
            "composer prompt typed-state span differs from typed_evidence_state"
        )
    masked_state, masked_count = masked_provenance_state(typed_state)
    masked_text = json.dumps(
        masked_state, ensure_ascii=False, sort_keys=True, indent=2
    )
    return text[:start] + masked_text + text[end:], masked_count


def validate_model_facing_task(row: dict[str, Any]) -> dict[str, Any]:
    """Run the single production model-boundary validation for one task."""

    unknown = sorted(set(row) - ALLOWED_TOP_LEVEL)
    if unknown:
        raise PermissionError(
            f"unknown fields reached explicit model-stage allowlist: {unknown}"
        )
    clean = {key: row[key] for key in row if key in ALLOWED_TOP_LEVEL}
    for required in ("sample_id", "task_id", "model_image_path", "model_input_prompt"):
        if not clean.get(required):
            raise ValueError(f"model-stage task lacks required field: {required}")
    hits = sorted(nested_keys(clean) & FORBIDDEN_NESTED_FIELDS)
    if hits:
        raise PermissionError(f"private nested fields reached model-facing task: {hits}")
    masked_prompt, masked_count = prompt_with_provenance_raw_masked(
        str(clean.get("model_input_prompt") or ""),
        clean.get("typed_evidence_state"),
    )
    match = forbidden_prompt_match(masked_prompt)
    match_location = "model_input_prompt" if match else None
    if match is None:
        for name in (
            "question_original",
            "question_1000",
            "question_stage1",
            "question_roi",
            "question_roi_local",
            "question_rewritten",
        ):
            if question_match := forbidden_prompt_match(str(clean.get(name) or "")):
                match = question_match
                match_location = name
                break
    return {
        "clean": clean,
        "forbidden_prompt_match": match.group(0) if match else None,
        "forbidden_prompt_match_location": match_location,
        "provenance_raw_text_fields_masked": masked_count,
    }


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        output = {str(key).lower() for key in value}
        for item in value.values():
            output.update(nested_keys(item))
        return output
    if isinstance(value, list):
        output: set[str] = set()
        for item in value:
            output.update(nested_keys(item))
        return output
    return set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    source = read_jsonl(args.input)
    output: list[dict[str, Any]] = []
    prompt_hits: list[dict[str, Any]] = []
    provenance_raw_text_fields_masked = 0
    for row in source:
        boundary = validate_model_facing_task(row)
        clean = boundary["clean"]
        provenance_raw_text_fields_masked += int(
            boundary["provenance_raw_text_fields_masked"]
        )
        if boundary["forbidden_prompt_match"]:
            prompt_hits.append(
                {
                    "task_id": clean.get("task_id"),
                    "match": boundary["forbidden_prompt_match"],
                    "location": boundary["forbidden_prompt_match_location"],
                }
            )
        output.append(clean)
    audit = {
        "schema_version": "map-model-facing-boundary-audit-1",
        "rows": len(output),
        "unique_task_ids": len({str(row.get("task_id")) for row in output}),
        "allowlisted_top_level_fields": sorted(ALLOWED_TOP_LEVEL),
        "unknown_fields_rejected": True,
        "forbidden_prompt_hits": prompt_hits,
        "provenance_raw_text_fields_structurally_masked": (
            provenance_raw_text_fields_masked
        ),
        "provenance_raw_text_exception_scope": (
            "typed_evidence_state.unstructured_evidence.raw_text_only"
        ),
        "opaque_sample_ids_only": all(
            bool(re.fullmatch(r"[0-9a-f]{64}", str(row.get("sample_id") or "")))
            for row in output
        ),
        "test_used": False,
    }
    audit["passed"] = (
        audit["rows"] == audit["unique_task_ids"]
        and audit["opaque_sample_ids_only"]
        and not prompt_hits
    )
    write_jsonl(args.out, output)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"model-facing boundary failed: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
