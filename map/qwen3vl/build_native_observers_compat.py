#!/usr/bin/env python3
"""Expose the frozen native640 observer builder through the strong-path CLI."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
TARGET = HERE.parent / "build_variant_observers.py"


if __name__ == "__main__":
    os.environ["UHR_SUPPORT_RENDER_VARIANT"] = "native640"
    if len(sys.argv) > 1 and sys.argv[1] == "build-stage3":
        del sys.argv[1]
    runpy.run_path(str(TARGET), run_name="__main__")

