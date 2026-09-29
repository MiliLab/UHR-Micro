#!/usr/bin/env python3
"""Compile the public evidence contract into one MAP v2.2 stage graph."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, write_json, write_jsonl


SERIALIZATION = {
    "SCALAR": "count",
    "OPTION": "choice",
    "SINGLE_GEOMETRY": "single_bbox",
    "GEOMETRY_SET": "multi_bbox",
}

GEOMETRY_SCHEMAS = frozenset({"SINGLE_GEOMETRY", "GEOMETRY_SET"})

OBSERVER_GENERATION_CONFIGS: dict[str, dict[str, float | int]] = {
    "STANDARD": {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
    },
    "QUERY_NATIVE": {
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
    },
}

OBSERVER_LOCAL_OUTPUT_PROTOCOLS: dict[str, str] = {
    "STANDARD": "rendered_canvas_base1000",
    "QUERY_NATIVE": "native_crop_base1000",
}


def observer_generation_profile(contract: dict[str, Any]) -> str:
    """Compile the local-observer generation profile from public support only.

    One bounded public region is a different observable interface from a
    model-proposed support: its native crop is already declared by the question
    and may need the longer Query-Native response budget. Public A/B supports
    remain independent standard observations, and public target boxes keep
    their scale-normalized fine-detail view with the standard generation
    profile. No benchmark task, historical version, or score is consulted.
    """

    return (
        "QUERY_NATIVE"
        if (
            str(contract.get("support_source") or "")
            == "PUBLIC_EXPLICIT_SUPPORT"
            and str(contract.get("support_extent") or "") == "BOUNDED_SUPPORT"
        )
        else "STANDARD"
    )


def observer_local_observation_output(observer_profile: str) -> str:
    """Return the declared observer coordinate protocol for one profile."""

    try:
        return OBSERVER_LOCAL_OUTPUT_PROTOCOLS[observer_profile]
    except KeyError as error:
        raise ValueError(
            f"unsupported observer generation profile: {observer_profile}"
        ) from error


def template_profile(contract: dict[str, Any]) -> dict[str, str]:
    if str(contract["answer_schema"]) not in SERIALIZATION:
        raise ValueError(f"unsupported public answer schema: {contract['answer_schema']}")
    return {"stage1_template_key": "support_proposal"}


def provisional_hint(schema: str, observation_encoding: str) -> str:
    base = {
        "SCALAR": "The final field must contain one non-negative integer and no worked answer example.",
        "OPTION": "The final field must contain one valid option letter and must not repeat candidate options.",
        "SINGLE_GEOMETRY": "Return one provisional source-image base1000 geometry.",
        "GEOMETRY_SET": "Return all provisionally visible source-image base1000 geometries, or null when none are supported.",
    }[schema]
    if schema in GEOMETRY_SCHEMAS:
        if observation_encoding != "AABB4_OR_QUAD8":
            raise ValueError(
                f"unsupported model observation geometry encoding: {observation_encoding}"
            )
        return (
            f"{base} For each visible object, record one unambiguous AABB4 "
            "[x1,y1,x2,y2] or QUAD8 "
            "[x1,y1,x2,y2,x3,y3,x4,y4] observation. The terminal composer "
            "will perform the declared final-format conversion."
        )
    return base.replace("final field", "provisional field")


def initial_evidence_state(contract: dict[str, Any]) -> dict[str, Any]:
    """Expose only public physical state for the unified MAP graph."""

    schema = str(contract["answer_schema"])
    support_source = str(contract["support_source"])
    composition = str(contract["composition"])
    resolution = str(contract["resolution_requirement"])
    extent = str(contract["support_extent"])
    if support_source in {"PUBLIC_EXPLICIT_SUPPORT", "PUBLIC_TARGET_SUPPORT"}:
        initial = "USE_PUBLIC_SUPPORT"
    elif extent == "FULL_SCENE":
        initial = "PROPOSE_DISTRIBUTED_SUPPORT"
    else:
        initial = "PROPOSE_QUESTION_SUPPORT"
    return {
        "answer_schema": schema,
        "geometry_encoding": contract["geometry_encoding"],
        "observation_geometry_encoding": contract[
            "observation_geometry_encoding"
        ],
        "terminal_geometry_encoding": contract["terminal_geometry_encoding"],
        "support_extent": contract["support_extent"],
        "evidence_topology": contract["support_topology"],
        "resolution_requirement": resolution,
        "coverage_requirement": contract["coverage_requirement"],
        "evidence_composition": composition,
        "initial_operator": initial,
        "explicit_region_count": contract["explicit_support_count"],
    }


def execution_plan(contract: dict[str, Any]) -> dict[str, Any]:
    activations = list(contract["operator_activations"])
    return {
        "local_observer": "OBSERVE_NATIVE_SUPPORT",
        "source_registration": True,
        "operator_activations": activations,
        "inference_operator_activations": [
            name for name in activations if name != "MASK_ADAPTER"
        ],
        "post_inference_adapter_eligibility": [
            name for name in activations if name == "MASK_ADAPTER"
        ],
        "mask_adapter_changes_inference": False,
        "typed_evidence_aggregation": True,
        "single_terminal_composer": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    output: list[dict[str, Any]] = []
    stage1_counts: Counter[str] = Counter()
    observation_counts: Counter[str] = Counter()
    observer_profile_counts: Counter[str] = Counter()
    local_output_protocol_counts: Counter[str] = Counter()
    activation_counts: Counter[str] = Counter()
    for public in read_jsonl(args.public):
        contract = dict(public["evidence_contract"])
        schema = str(contract["answer_schema"])
        profile = template_profile(contract)
        observer_profile = observer_generation_profile(contract)
        local_output_protocol = observer_local_observation_output(observer_profile)
        for operator in contract["observation_operators"]:
            observation_counts[str(operator)] += 1
        for activation in contract["operator_activations"]:
            activation_counts[str(activation)] += 1
        stage1_counts[profile["stage1_template_key"]] += 1
        observer_profile_counts[observer_profile] += 1
        local_output_protocol_counts[local_output_protocol] += 1
        stage1_needed = contract["support_source"] in {
            "MODEL_PROPOSED_SUPPORT",
            "PUBLIC_SEMANTIC_SUPPORT",
        }
        output.append(
            {
                "schema_version": "map-runtime-contract-2.2",
                "sample_id": public["sample_id"],
                "qa_slug": str(public["sample_id"])[:16],
                "image_abs_path": public["image_abs_path"],
                "image_sha256": public["image_sha256"],
                "original_width_px": public["source_width_px"],
                "original_height_px": public["source_height_px"],
                "question_original": public["question_original"],
                "question_1000": public["question_source1000"],
                "question_stage1": public["question_support_planning"],
                "question_roi": public["question_source1000"],
                "seed_boxes_abs_from_question": public["explicit_support_boxes_abs"],
                "coord_lists_abs_from_question": public[
                    "coordinates_abs_from_public_question"
                ],
                "axis_thresholds_abs_from_question": public[
                    "axis_thresholds_abs_from_public_question"
                ],
                "coordinate_protocol": {
                    "original_public_question": public["input_coordinate_protocol"],
                    "model_facing_complete_question": public[
                        "model_facing_question_coordinate_protocol"
                    ],
                    "local_observation_output": local_output_protocol,
                    "terminal_output": "base1000",
                    "observation_geometry_encoding": contract[
                        "observation_geometry_encoding"
                    ],
                    "terminal_geometry_encoding": contract[
                        "terminal_geometry_encoding"
                    ],
                },
                "evidence_contract": contract,
                "evidence_state": initial_evidence_state(contract),
                "execution_plan": execution_plan(contract),
                "answer_serialization": SERIALIZATION[schema],
                "stage1_needed": stage1_needed,
                "support_source": (
                    "PUBLIC_SEMANTIC_SCOPE_WITH_MODEL_POINT_DETAIL"
                    if contract["support_source"] == "PUBLIC_SEMANTIC_SUPPORT"
                    else "MODEL_POINT_SUPPORT"
                    if stage1_needed
                    else "PUBLIC_COORDINATE_SUPPORT"
                ),
                "max_stage1_points": int(contract["max_support_points"]),
                "max_native_views": int(contract["max_native_views"]),
                "observer_generation_profile": observer_profile,
                "provisional_format_hint": provisional_hint(
                    schema, str(contract["observation_geometry_encoding"])
                ),
                **profile,
                "generation_contract": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "maximum_context_tokens": 128000,
                },
            }
        )

    forbidden = {
        "task_key",
        "task_family",
        "split",
        "answer",
        "annotation",
        "metric",
        "dataset_version",
        "historical_version",
        "task_route",
    }
    leaked = sorted(
        {
            key
            for row in output
            for key in row
            if key.lower() in forbidden
        }
    )
    observer_protocol_mismatches = [
        {
            "sample_id": row["sample_id"],
            "observer_generation_profile": row["observer_generation_profile"],
            "recorded_local_observation_output": row["coordinate_protocol"][
                "local_observation_output"
            ],
            "expected_local_observation_output": observer_local_observation_output(
                row["observer_generation_profile"]
            ),
        }
        for row in output
        if row["coordinate_protocol"]["local_observation_output"]
        != observer_local_observation_output(row["observer_generation_profile"])
    ]
    audit = {
        "schema_version": "map-runtime-contract-audit-2.2",
        "rows": len(output),
        "unique_sample_ids": len({row["sample_id"] for row in output}),
        "stage1_template_counts": dict(sorted(stage1_counts.items())),
        "observer_generation_profile_counts": dict(
            sorted(observer_profile_counts.items())
        ),
        "observer_generation_profiles": OBSERVER_GENERATION_CONFIGS,
        "observer_profile_compiled_from": (
            "public_support_source_and_public_support_extent"
        ),
        "observer_local_observation_output_by_profile": (
            OBSERVER_LOCAL_OUTPUT_PROTOCOLS
        ),
        "local_observation_output_counts": dict(
            sorted(local_output_protocol_counts.items())
        ),
        "observer_local_output_protocol_mismatch_rows": len(
            observer_protocol_mismatches
        ),
        "observer_local_output_protocol_mismatches": observer_protocol_mismatches,
        "geometry_encoding_counts": dict(
            sorted(Counter(row["evidence_contract"]["geometry_encoding"] for row in output).items())
        ),
        "observation_geometry_encoding_counts": dict(
            sorted(
                Counter(
                    row["evidence_contract"]["observation_geometry_encoding"]
                    for row in output
                ).items()
            )
        ),
        "terminal_geometry_encoding_counts": dict(
            sorted(
                Counter(
                    row["evidence_contract"]["terminal_geometry_encoding"]
                    for row in output
                ).items()
            )
        ),
        "observation_operator_counts": dict(sorted(observation_counts.items())),
        "operator_activation_counts": dict(sorted(activation_counts.items())),
        "mask_adapter_changes_inference": False,
        "private_fields": leaked,
        "one_stage_graph_for_all_rows": True,
        "historical_terminal_routes": 0,
        "completed_prediction_inputs": 0,
        "test_used": False,
    }
    audit["passed"] = (
        len(output) == audit["unique_sample_ids"]
        and not leaked
        and set(observer_profile_counts) == set(OBSERVER_GENERATION_CONFIGS)
        and not observer_protocol_mismatches
        and sum(local_output_protocol_counts.values()) == len(output)
    )
    write_jsonl(args.out, output)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"runtime contract gate failed: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
