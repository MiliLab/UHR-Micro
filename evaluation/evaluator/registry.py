from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


def load_model_registry(eval_root: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    registry = read_json(eval_root / "configs/model_registry.json")
    models = list(registry["models"])
    addendum_path = eval_root / "configs/model_registry.addendum.lock.json"
    if addendum_path.is_file():
        addendum = read_json(addendum_path)
        models.extend(addendum["models"])
    ids = [str(row["model_registry_id"]) for row in models]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate model_registry_id after addendum merge")
    by_id = {str(row["model_registry_id"]): dict(row) for row in models}
    override_path = eval_root / "configs/model_protocol_overrides.lock.json"
    if override_path.is_file():
        overrides = read_json(override_path)
        seen: set[str] = set()
        for override in overrides["overrides"]:
            model_id = str(override["model_registry_id"])
            if model_id in seen or model_id not in by_id:
                raise ValueError(f"duplicate or unknown protocol override: {model_id}")
            seen.add(model_id)
            by_id[model_id]["coordinate_protocol"] = str(override["coordinate_protocol"])
    return by_id, None
