#!/usr/bin/env python3
"""Prepare all executable pairwise scalar operands from the public contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


"""Prepare one-variable pairwise operand-algebra Validation tasks."""



MODEL = Path(os.environ.get("MAP_MODEL_PATH", ""))
ONLY_SENTENCE = (
    "Image 1 is the complete source layout. It contains only method-generated "
    "white observation footprints."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prompt_for(row: dict[str, Any]) -> str:
    state = row["public_evidence_state"]
    boxes = row["endpoint_source_boxes_1000"]
    endpoint_lines = "\n".join(
        f"- Image {index + 2}: endpoint {node_id}, source box {box}"
        for index, (node_id, box) in enumerate(zip(row["support_node_ids"], boxes))
    )
    return f"""Extract the typed operands of one nonlocal source-coordinate evidence unit for an ultra-high-resolution remote-sensing question.

{ONLY_SENTENCE} The footprints show where native observations were acquired; they are not labels or answers.
{endpoint_lines}

Images 2 and 3 are the separate, unresized native-resolution endpoint observations named above. Their coordinates use the complete source image's 0-1000 frame. Use Image 1 only for source ownership and global configuration, and use Images 2 and 3 for endpoint detail. Do not merge identities across source boxes and do not infer evidence from the footprint graphics themselves.

Every numeric spatial coordinate in the public question below has already been converted to the complete source image's 0-1000 system. Any retained word `pixels` is legacy public wording and does not change that coordinate system.

Public evidence contract:
{json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}

Public question:
{row['question_1000']}

Report the non-negative scalar observed for endpoint R1 and the non-negative scalar observed for endpoint R2. Do not compute or return the requested difference. Return one JSON object containing exactly the keys `r1` and `r2`; each value must be one non-negative Arabic integer, or null when that endpoint is not sufficiently supported by its observation. Return no explanation and no other fields.
Do not repeat the question, evidence contract, or endpoint descriptions."""


def prepare(args: argparse.Namespace) -> int:
    candidates = [
        row for row in read_jsonl(args.materialized_actions)
        if row["action_kind"] == "NONLOCAL_JOINT_ENDPOINT_OBSERVATION"
        and row["public_evidence_state"]["evidence_composition"] == "NUMERIC_DIFFERENCE"
        and row["public_evidence_state"]["resolution_requirement"] == "OBJECT"
    ]
    excluded_ids = {
        str(row["sample_id"])
        for path in args.exclude_outputs
        for row in read_jsonl(path)
    }
    ordered = sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"nonlocal-support-closure-v1:{row['sample_id']}".encode("utf-8")
        ).hexdigest(),
    )
    selected = [row for row in ordered if str(row["sample_id"]) not in excluded_ids][: args.sample_count]
    reference = {
        str(row["sample_id"]): row
        for path in args.reference_tasks
        for row in read_jsonl(path)
    }

    contract_payload = {
        "schema_version": "pairwise-operand-algebra-contract-1",
        "model_path": str(args.model.resolve()),
        "max_model_len": 128000,
        "max_new_tokens": 96,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "maximum_images_per_prompt": 3,
        "image_rendering": "same_three_images_as_direct_pairwise_terminal",
        "selection": {
            "ordering": "same_sha256_order_as_existing_numeric_difference_pilot",
            "excluded_completed_sample_ids": len(excluded_ids),
        },
        "ground_truth_annotation_metric_used": False,
        "test_used": False,
    }
    contract_hash = sha256_json(contract_payload)
    contract = dict(contract_payload)
    contract["request_contract_sha256"] = contract_hash

    tasks: list[dict[str, Any]] = []
    identity_checks = 0
    for row in selected:
        paths = [str(Path(path).resolve()) for path in row["materialized_observation_inputs"]]
        hashes = [sha256_file(Path(path)) for path in paths]
        if hashes != row["materialized_observation_input_sha256"]:
            raise ValueError(f"materialized input hash drift: {row['sample_id']}")
        old = reference.get(str(row["sample_id"]))
        if old is None:
            raise ValueError(f"selected row absent from direct-terminal reference: {row['sample_id']}")
        if old["model_image_sha256"] != hashes or old["endpoint_source_boxes_1000"] != row["endpoint_source_boxes_1000"]:
            raise ValueError(f"image/endpoint control drift: {row['sample_id']}")
        identity_checks += 1
        prompt = prompt_for(row)
        task = {
            "schema_version": "pairwise-operand-algebra-task-1",
            "task_id": f"{row['qa_slug']}_pairwise_operand_algebra",
            "sample_id": row["sample_id"],
            "model_image_paths": paths,
            "model_image_sha256": hashes,
            "model_input_prompt": prompt,
            "public_evidence_state": row["public_evidence_state"],
            "endpoint_source_boxes_1000": row["endpoint_source_boxes_1000"],
            "observer_generation_profile": "PAIRWISE_OPERAND_RECEIPT",
            "request_contract_sha256": contract_hash,
            "attempt_number": 1,
        }
        task["request_sha256"] = sha256_json({
            "task_id": task["task_id"],
            "image_sha256": hashes,
            "prompt": prompt,
            "observer_generation_profile": task["observer_generation_profile"],
            "request_contract_sha256": contract_hash,
        })
        tasks.append(task)

    audit = {
        "schema_version": "pairwise-operand-algebra-prepare-audit-1",
        "available_rows": len(candidates),
        "selected_rows": len(tasks),
        "excluded_completed_rows": len(excluded_ids),
        "same_image_and_endpoint_checks": identity_checks,
        "exact_only_sentence_retained": all(ONLY_SENTENCE in row["model_input_prompt"] for row in tasks),
        "images_per_task": sorted({len(row["model_image_paths"]) for row in tasks}),
        "selection_reads_answers_or_scores": False,
        "direct_answer_requested": False,
        "gpu_used": False,
        "test_used": False,
    }
    audit["passed"] = all([
        len(tasks) == args.sample_count,
        identity_checks == args.sample_count,
        not ({str(row["sample_id"]) for row in tasks} & excluded_ids),
        audit["exact_only_sentence_retained"],
        audit["images_per_task"] == [3],
    ])
    write_json(args.output_dir / "REQUEST_CONTRACT.json", contract)
    write_jsonl(args.output_dir / "pilot_tasks.jsonl", tasks)
    write_json(args.output_dir / "PILOT_PREPARE_AUDIT.json", audit)
    return 0 if audit["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-actions", type=Path, required=True)
    parser.add_argument("--reference-tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-outputs", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--model", type=Path, default=MODEL)
    return parser




def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()

    candidates = sorted(
        (
            row for row in read_jsonl(args.materialized_actions)
            if row["action_kind"] == "NONLOCAL_JOINT_ENDPOINT_OBSERVATION"
            and row["public_evidence_state"]["answer_schema"] == "SCALAR"
            and row["public_evidence_state"]["evidence_composition"] == "NUMERIC_DIFFERENCE"
            and row["public_evidence_state"]["resolution_requirement"] == "OBJECT"
        ),
        key=lambda row: str(row["sample_id"]),
    )
    contract_payload = {
        "schema_version": "map-pairwise-scalar-receipt-contract-1",
        "model_path": str(args.model.resolve()),
        "max_model_len": 128000,
        "max_new_tokens": 96,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "maximum_images_per_prompt": 3,
        "image_rendering": "separate_source-layout-and-native-endpoints",
        "selection": "all executable public SCALAR plus NUMERIC_DIFFERENCE plus PAIRWISE_SUPPORT states",
        "dataset_count_used_as_routing_rule": False,
        "ground_truth_annotation_task_metric_used": False,
        "test_used": False,
    }
    contract_hash = sha256_json(contract_payload)
    contract = dict(contract_payload)
    contract["request_contract_sha256"] = contract_hash

    tasks: list[dict] = []
    for row in candidates:
        paths = [str(Path(path).resolve()) for path in row["materialized_observation_inputs"]]
        hashes = [sha256_file(Path(path)) for path in paths]
        if hashes != row["materialized_observation_input_sha256"]:
            raise ValueError(f"materialized input hash drift: {row['sample_id']}")
        if len(paths) != 3:
            raise ValueError(f"pairwise scalar state does not have exactly three images: {row['sample_id']}")
        prompt = prompt_for(row)
        task = {
            "schema_version": "map-pairwise-scalar-receipt-task-1",
            "task_id": f"{row['qa_slug']}_map_pairwise_scalar",
            "sample_id": row["sample_id"],
            "model_image_paths": paths,
            "model_image_sha256": hashes,
            "model_input_prompt": prompt,
            "public_evidence_state": row["public_evidence_state"],
            "endpoint_source_boxes_1000": row["endpoint_source_boxes_1000"],
            "observer_generation_profile": "PAIRWISE_OPERAND_RECEIPT",
            "request_contract_sha256": contract_hash,
            "attempt_number": 1,
        }
        task["request_sha256"] = sha256_json({
            "task_id": task["task_id"],
            "image_sha256": hashes,
            "prompt": prompt,
            "observer_generation_profile": task["observer_generation_profile"],
            "request_contract_sha256": contract_hash,
        })
        tasks.append(task)

    audit = {
        "schema_version": "map-pairwise-scalar-prepare-audit-1",
        "selected_rows": len(tasks),
        "unique_sample_ids": len({str(row["sample_id"]) for row in tasks}),
        "all_public_executable_candidates_selected": len(tasks) == len(candidates),
        "exact_only_word_retained": all("contains only method-generated white observation footprints" in row["model_input_prompt"] for row in tasks),
        "images_per_task": sorted({len(row["model_image_paths"]) for row in tasks}),
        "dataset_count_used_as_routing_rule": False,
        "ground_truth_annotation_task_metric_used": False,
        "test_used": False,
    }
    audit["passed"] = audit["unique_sample_ids"] == len(tasks) and audit["all_public_executable_candidates_selected"] and audit["exact_only_word_retained"] and (audit["images_per_task"] == [3] or not tasks)
    write_json(args.output_dir / "REQUEST_CONTRACT.json", contract)
    write_jsonl(args.output_dir / "operand_tasks.jsonl", tasks)
    write_json(args.output_dir / "OPERAND_PREPARE_AUDIT.json", audit)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
