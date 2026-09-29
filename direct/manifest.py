from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .constants import FORBIDDEN_GT_KEYS, INPUT_SCHEMA_VERSION
from .io import assert_no_forbidden_keys, iter_jsonl


REQUIRED_INPUT_KEYS = {
    "schema_version",
    "phase0_contract_sha256",
    "phase1_contract_sha256",
    "benchmark_manifest_sha256",
    "model_registry_id",
    "split",
    "split_position",
    "qa_relpath",
    "task_key",
    "image_relpath",
    "image_sha256",
    "source_image_size",
    "source_question",
    "source_question_sha256",
    "model_question",
    "model_question_sha256",
    "question_coordinate_protocol",
    "coordinate_transforms",
    "requested_coordinate_protocol",
    "prompt_id",
    "prompt_text",
    "prompt_sha256",
}
OPTIONAL_INPUT_KEYS = {"expected_model_input_size"}


def validate_input_row(row: dict[str, Any], expected_position: int | None = None) -> None:
    if not REQUIRED_INPUT_KEYS.issubset(row) or not set(row).issubset(REQUIRED_INPUT_KEYS | OPTIONAL_INPUT_KEYS):
        missing = sorted(REQUIRED_INPUT_KEYS - set(row))
        extra = sorted(set(row) - REQUIRED_INPUT_KEYS - OPTIONAL_INPUT_KEYS)
        raise ValueError(f"inference input schema mismatch missing={missing} extra={extra}")
    assert_no_forbidden_keys(row, FORBIDDEN_GT_KEYS)
    if row["schema_version"] != INPUT_SCHEMA_VERSION:
        raise ValueError("wrong inference input schema version")
    if expected_position is not None and row["split_position"] != expected_position:
        raise ValueError(f"split position mismatch at {expected_position}")
    protocol = row["requested_coordinate_protocol"]
    if protocol not in {"base1000", "base100", "base1", "absolute_model_input", "absolute_source"}:
        raise ValueError(f"unsupported local-VLM protocol: {row['qa_relpath']}")
    expected_size = row.get("expected_model_input_size")
    if expected_size is not None:
        if set(expected_size) != {"width_px", "height_px"} or min(int(expected_size["width_px"]), int(expected_size["height_px"])) <= 0:
            raise ValueError(f"invalid expected model-input size: {row['qa_relpath']}")
    for transform in row["coordinate_transforms"]:
        values = (
            transform.get(f"model_values_{protocol}") if protocol in {"base1000", "base100", "base1"}
            else transform.get("source_values_clipped_absolute_px") if protocol == "absolute_source"
            else transform.get("model_values_absolute_input_px")
        )
        if not isinstance(values, list) or not values:
            raise ValueError(f"missing transformed coordinates: {row['qa_relpath']}")
        try:
            out_of_range = any(float(value) < 0.0 or float(value) > 1000.0 for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid transformed coordinates: {row['qa_relpath']}") from exc
        if protocol in {"base1000", "base100", "base1"}:
            base = {"base1000": 1000.0, "base100": 100.0, "base1": 1.0}[protocol]
            if any(float(value) < 0.0 or float(value) > base for value in values):
                raise ValueError(f"model-facing {protocol} coordinate out of range: {row['qa_relpath']}")
        if protocol == "absolute_model_input":
            canvas = transform.get("model_input_canvas")
            if expected_size is None or canvas != expected_size:
                raise ValueError(f"absolute input transform canvas mismatch: {row['qa_relpath']}")
            extents = [canvas["width_px"] if index % 2 == 0 else canvas["height_px"] for index in range(len(values))]
            if any(float(value) < 0.0 or float(value) > float(extent) for value, extent in zip(values, extents)):
                raise ValueError(f"model-facing absolute coordinate out of range: {row['qa_relpath']}")


def load_input_manifest(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    seen: set[str] = set()
    for row in rows:
        # Subset manifests preserve the canonical split_position rather than
        # renumbering samples. Full-manifest order is audited separately.
        validate_input_row(row)
        qa = row["qa_relpath"]
        if qa in seen:
            raise ValueError(f"duplicate qa_relpath: {qa}")
        seen.add(qa)
    return rows


def resolve_image(image_root: Path, image_relpath: str) -> Path:
    rel = Path(image_relpath)
    if not rel.parts or rel.parts[0] != "Datasource":
        raise ValueError(f"image must be rooted under Datasource: {image_relpath}")
    relative_under_image_root = Path(*rel.parts[1:])
    root = image_root.resolve()
    path = (root / relative_under_image_root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(path)
    return path
