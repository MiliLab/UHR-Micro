#!/usr/bin/env python3
"""Frozen render-condition definitions shared by the ablation tools."""

from __future__ import annotations

import os
from typing import Any


VARIANTS: dict[str, dict[str, Any]] = {
    "pad640": {"tile_side_px": 640, "render_mode": "NATIVE_PAD"},
    "native640": {"tile_side_px": 640, "render_mode": "QUERY_NATIVE"},
    "native512": {"tile_side_px": 512, "render_mode": "QUERY_NATIVE"},
    "pad512": {"tile_side_px": 512, "render_mode": "NATIVE_PAD"},
}


def selected_variant() -> tuple[str, dict[str, Any]]:
    name = os.environ.get("UHR_SUPPORT_RENDER_VARIANT", "").strip()
    if name not in VARIANTS:
        raise ValueError(
            f"UHR_SUPPORT_RENDER_VARIANT must be one of {sorted(VARIANTS)}, got {name!r}"
        )
    return name, dict(VARIANTS[name])

