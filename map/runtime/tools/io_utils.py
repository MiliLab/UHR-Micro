#!/usr/bin/env python3
"""Small deterministic I/O helpers for MAP Baseline v2.2."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable


_IMAGE_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def _archive_truncated_tail(path: Path, fragment: bytes) -> Path:
    digest = hashlib.sha256(fragment).hexdigest()[:16]
    archive = path.parent / "archive" / "truncated_jsonl_tails"
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f"{path.name}.{int(time.time())}.{digest}.fragment"
    target.write_bytes(fragment)
    return target


def recover_truncated_jsonl_tail(path: Path) -> Path | None:
    """Recover only one invalid, non-newline-terminated final JSONL fragment.

    A malformed complete line or any corruption before the final physical line
    is always fatal.  The removed fragment is archived beside the ledger before
    an atomic replacement, so recovery is narrow and auditable.
    """

    if not path.is_file() or path.stat().st_size == 0:
        return None
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            is_final = index == len(lines) - 1
            is_truncated = is_final and not raw.endswith((b"\n", b"\r"))
            if not is_truncated:
                raise ValueError(
                    f"JSONL corruption is not a recoverable truncated tail: {path}:{index + 1}"
                ) from error
            archive = _archive_truncated_tail(path, raw)
            valid_prefix = b"".join(lines[:index])
            temporary = path.with_suffix(path.suffix + ".recover.tmp")
            temporary.write_bytes(valid_prefix)
            os.replace(temporary, path)
            return archive
    return None


def read_jsonl(
    path: Path, *, recover_truncated_tail: bool = False
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    if recover_truncated_tail:
        recover_truncated_jsonl_tail(path)
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
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_sha256_file(path: Path) -> str:
    """Hash actual bytes, caching only an exact resolved path/stat identity."""

    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    digest = _IMAGE_DIGEST_CACHE.get(key)
    if digest is None:
        digest = sha256_file(resolved)
        _IMAGE_DIGEST_CACHE[key] = digest
    return digest


def request_digest(row: dict[str, Any], request_contract_sha256: str) -> str:
    """Hash exactly the image, prompt, and deterministic request contract.

    This supports checkpoint resume *within this run*.  It deliberately has no
    prior-run input and cannot import a historical prediction.
    """

    image_path = Path(str(row["model_image_path"])).resolve(strict=True)
    source_path_raw = row.get("image_abs_path")
    source_path = (
        Path(str(source_path_raw)).resolve(strict=True) if source_path_raw else None
    )
    image_digest = cached_sha256_file(image_path)
    if (
        row.get("image_sha256")
        and source_path is not None
        and image_path == source_path
        and image_digest != str(row["image_sha256"])
    ):
        raise ValueError("model image bytes differ from the declared source image hash")
    declared_model_hash = row.get("roi_image_sha256") or row.get("crop_image_sha256")
    if declared_model_hash and image_digest != str(declared_model_hash):
        raise ValueError("model crop bytes differ from the declared crop image hash")
    payload = {
        "task_id": str(row["task_id"]),
        "image_sha256": image_digest,
        "prompt": str(row["model_input_prompt"]),
        "min_pixels": row.get("min_pixels"),
        "max_pixels": row.get("max_pixels"),
        "observer_generation_profile": row.get(
            "observer_generation_profile"
        ),
        "request_contract_sha256": request_contract_sha256,
    }
    return sha256_bytes(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def keyed(
    rows: Iterable[dict[str, Any]], key: str = "sample_id"
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in output:
            raise ValueError(f"missing or duplicate {key}: {value!r}")
        output[value] = row
    return output


__all__ = [
    "cached_sha256_file",
    "keyed",
    "read_jsonl",
    "recover_truncated_jsonl_tail",
    "request_digest",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "write_json",
    "write_jsonl",
]
