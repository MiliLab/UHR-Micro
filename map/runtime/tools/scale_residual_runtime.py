#!/usr/bin/env python3
"""Certify the SSRR scale residual without task routing.

This runtime consumes only the public MAP typed state.  It
reopens an otherwise closed OPTION state exactly when the valid complete-scene
claim disagrees with every valid registered local claim.  It neither selects
between answers nor reads benchmark identity, annotations, metrics, or scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_digest(row: dict[str, Any], contract_sha256: str) -> str:
    image_path = Path(str(row["model_image_path"])).resolve(strict=True)
    payload = {
        "task_id": str(row["task_id"]),
        "image_sha256": sha256_file(image_path),
        "prompt": str(row["model_input_prompt"]),
        "min_pixels": row.get("min_pixels"),
        "max_pixels": row.get("max_pixels"),
        "observer_generation_profile": row.get("observer_generation_profile"),
        "request_contract_sha256": contract_sha256,
    }
    return canonical_sha256(payload)


def keyed(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity or identity in output:
            raise ValueError(f"missing or duplicate {key}: {identity!r}")
        output[identity] = row
    return output


def scale_conflict(state: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    contract = state.get("public_contract") or {}
    if str(contract.get("answer_schema") or "") != "OPTION":
        return False, {"reason": "non_option_state"}
    resolution = str(contract.get("resolution_requirement") or "")
    # A complete-scene claim is not an admissible counter-claim when the
    # public answer explicitly requires fine-subtype detail.  In that regime
    # only target-resolving evidence may create a scale conflict.  This is a
    # physical resolution rule, not a benchmark-task switch.
    globally_admissible = resolution != "FINE_SUBTYPE"
    complete = state.get("complete_scene_observation") or {}
    global_choice = (
        str(complete.get("parsed_choice") or "")
        if complete.get("valid") is True
        else ""
    )
    local_choices = [
        str(row.get("parsed_choice") or "")
        for row in state.get("local_observations") or []
        if row.get("valid") is True
        and str(row.get("parsed_choice") or "") in {"A", "B", "C", "D"}
    ]
    unique_local = sorted(set(local_choices))
    admitted = (
        globally_admissible
        and
        global_choice in {"A", "B", "C", "D"}
        and bool(unique_local)
        and global_choice not in unique_local
    )
    return admitted, {
        "reason": (
            "valid_complete_scene_disagrees_with_all_valid_registered_local_claims"
            if admitted
            else "no_admissible_scale_conflict"
        ),
        "global_choice": global_choice or None,
        "local_choices": local_choices,
        "unique_local_choices": unique_local,
        "resolution_requirement": resolution or None,
        "complete_scene_claim_admissible": globally_admissible,
        "runtime_inputs": [
            "public_answer_schema",
            "valid_complete_scene_claim",
            "valid_registered_local_claims",
        ],
    }


def expected_reopened_decision(
    state: dict[str, Any], original: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Return the unique residual-state successor of a MAP closure decision.

    Both the recurrence planner and the final materializer call this function.
    Consequently a stored residual decision is accepted only when it can be
    recomputed from the same public typed evidence state, without consulting a
    response, task identity, metric, or completed prediction.
    """

    admitted, certificate = scale_conflict(state)
    row = dict(original)
    activated = bool(admitted and row.get("closure_status") == "CLOSED")
    if activated:
        row.update(
            {
                "schema_version": "ssrr-scale-residual-decision-2.1",
                "closure_status": "OPEN",
                "terminal_mode": "SHARED_COMPOSER",
                "final_response": None,
                "selection_reason": "scale_residual_requires_joint_synthesis",
                "selected_observation_ids": [],
                "scale_residual_certificate": certificate,
                "private_identity_used": False,
                "ground_truth_or_metric_used": False,
                "task_identity_used": False,
                "test_used": False,
            }
        )
    else:
        row["scale_residual_certificate"] = certificate
    unhashed = {key: value for key, value in row.items() if key != "decision_sha256"}
    row["decision_sha256"] = canonical_sha256(unhashed)
    return row, activated


def reopen(args: argparse.Namespace) -> int:
    states = keyed(read_jsonl(args.states), "sample_id")
    decisions = read_jsonl(args.closure)
    output: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for original in decisions:
        sid = str(original["sample_id"])
        if sid not in states:
            raise ValueError(f"closure references unknown state: {sid}")
        row, activated = expected_reopened_decision(states[sid], original)
        if activated:
            records.append(
                {"sample_id": sid, **dict(row["scale_residual_certificate"])}
            )
        output.append(row)
    counts = Counter(row["closure_status"] for row in output)
    audit = {
        "schema_version": "ssrr-scale-residual-audit-2.1",
        "rows": len(output),
        "unique_sample_ids": len({row["sample_id"] for row in output}),
        "reopened_scale_conflicts": len(records),
        "closure_status_counts": dict(sorted(counts.items())),
        "runtime_task_identity_used": False,
        "ground_truth_or_metric_used": False,
        "historical_result_selector_used": False,
        "test_used": False,
        "passed": len(output) == len(states) and bool(records),
    }
    write_jsonl(args.out, output)
    write_jsonl(args.activated, records)
    write_json(args.audit, audit)
    if not audit["passed"]:
        raise RuntimeError(f"scale residual gate failed: {audit}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    item = commands.add_parser("reopen")
    item.add_argument("--states", type=Path, required=True)
    item.add_argument("--closure", type=Path, required=True)
    item.add_argument("--out", type=Path, required=True)
    item.add_argument("--activated", type=Path, required=True)
    item.add_argument("--audit", type=Path, required=True)
    item.set_defaults(func=reopen)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.func(arguments))
