#!/usr/bin/env python3
"""Materialize and verify the physical image inputs for Stage A actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


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


def verify_file(path_text: str, expected_hash: str, cache: dict[str, str]) -> str:
    path = Path(path_text).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = cache.get(str(path))
    if actual is None:
        actual = sha256_file(path)
        cache[str(path)] = actual
    if actual != expected_hash:
        raise ValueError(f"image hash mismatch: {path}")
    return actual


def materialize(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.actions)
    outputs: list[dict[str, Any] | None] = [None] * len(rows)
    counts: Counter[str] = Counter()
    hash_cache: dict[str, str] = {}
    materialized_crops = 0
    crop_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        output = dict(row)
        kind = str(row["action_kind"])
        counts[kind] += 1
        source_path = Path(row["source_image_path"]).resolve()
        verify_file(str(source_path), row["source_image_sha256"], hash_cache)

        if "source_crop_abs_xyxy" in row:
            crop_groups.setdefault(str(source_path), []).append((index, output))
            continue
        inputs = list(row["observation_inputs"])
        expected = list(row["observation_input_sha256"])
        if len(inputs) != len(expected) or len(inputs) < 3:
            raise ValueError(f"invalid joint observation bundle for {row['sample_id']}")
        actual = [verify_file(path, digest, hash_cache) for path, digest in zip(inputs, expected)]
        output["materialized_observation_inputs"] = [str(Path(path).resolve()) for path in inputs]
        output["materialized_observation_input_roles"] = row["observation_input_roles"]
        output["materialized_observation_input_sha256"] = actual
        output["render_transform"] = "NONE_SEPARATE_NATIVE_IMAGES"
        output["action_input_ready"] = True
        outputs[index] = output

    for source_text, group in crop_groups.items():
        source_path = Path(source_text)
        with Image.open(source_path) as image:
            image.load()
            width, height = image.size
            for index, output in group:
                x1, y1, x2, y2 = [int(value) for value in output["source_crop_abs_xyxy"]]
                width, height = image.size
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"crop out of source bounds for {output['sample_id']}")
                crop = image.crop((x1, y1, x2, y2))
                expected_size = (x2 - x1, y2 - y1)
                if crop.size != expected_size:
                    raise ValueError(f"crop size mismatch for {output['sample_id']}")
                output_path = args.output_dir / "images" / output["sample_id"] / "native_residual.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(output_path, format="PNG")
                output["materialized_observation_inputs"] = [str(output_path.resolve())]
                output["materialized_observation_input_roles"] = ["NEW_NATIVE_SOURCE_CROP"]
                output["materialized_observation_input_sha256"] = [sha256_file(output_path)]
                output["render_transform"] = "NONE_NATIVE_CROP_NO_RESIZE_NO_PADDING"
                output["materialized_size_px"] = [x2 - x1, y2 - y1]
                for path, digest in zip(
                    output.get("context_inputs") or [], output.get("context_input_sha256") or []
                ):
                    verify_file(path, digest, hash_cache)
                output["action_input_ready"] = True
                outputs[index] = output
                materialized_crops += 1

    if any(row is None for row in outputs):
        raise RuntimeError("one or more actions were not materialized")
    ready_outputs = [row for row in outputs if row is not None]

    audit = {
        "schema_version": "support-complex-stage-a-materialization-audit-1",
        "input_actions": len(rows),
        "materialized_actions": len(ready_outputs),
        "action_counts": dict(sorted(counts.items())),
        "new_native_crops": materialized_crops,
        "joint_multi_image_bundles": counts["NONLOCAL_JOINT_ENDPOINT_OBSERVATION"],
        "unique_verified_source_or_context_images": len(hash_cache),
        "all_actions_ready": len(ready_outputs) == len(rows) and all(row["action_input_ready"] for row in ready_outputs),
        "resize_used": False,
        "padding_used": False,
        "vlm_calls": 0,
        "gpu_used": False,
        "test_used": False,
    }
    audit["passed"] = all([
        len(ready_outputs) == len(rows),
        sum(counts.values()) == len(rows),
        counts["NONLOCAL_JOINT_ENDPOINT_OBSERVATION"] == audit["joint_multi_image_bundles"],
        audit["all_actions_ready"],
    ])
    write_jsonl(args.output_dir / "materialized_actions.jsonl", ready_outputs)
    write_json(args.output_dir / "STAGE_A_MATERIALIZATION_AUDIT.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"Stage A materialization gate failed: {audit}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(materialize(build_parser().parse_args()))
