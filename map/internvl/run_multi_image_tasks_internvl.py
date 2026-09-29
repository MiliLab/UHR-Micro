#!/usr/bin/env python3
"""InternVL3.5-8B multi-image worker for MAP scalar operand receipts.

CLI-compatible with nonlocal_support_closure_v1/tools/run_multi_image_tasks.py
so that the frozen run_multi_image_parallel.py orchestrator can drive it:
  --tasks --contract --out --attempt-ledger --summary --gpu
  --max-infrastructure-attempts

Task rows carry model_image_paths (>=3 source-native joint bundle images) and
model_input_prompt. Generation: dynamic tiling per image (max_num=12),
num_patches_list, model.chat, do_sample=False.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ADAPTER_TOOLS = Path(__file__).resolve().parents[1] / "runtime" / "tools"
INTERNVL_REF_DIR = Path(os.environ.get("MAP_INTERNVL_REF_DIR", ""))
DEFAULT_MODEL = Path(os.environ.get("MAP_INTERNVL_MODEL_PATH") or Path(__file__).resolve().parents[3] / "models/InternVL3_5-8B")
MAX_NUM_TILES = 12

sys.path.insert(0, str(ADAPTER_TOOLS))
sys.path.insert(0, str(INTERNVL_REF_DIR))

from io_utils import read_jsonl, sha256_file  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=5)
    parser.add_argument("--global-view-max-side", type=int, default=0)
    parser.add_argument("--max-num-tiles", type=int, default=MAX_NUM_TILES)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # The frozen orchestrator forwards only a fixed arg set; preprocessing
    # profile overrides travel via environment variables (set by the
    # supervisor, inherited through the orchestrator subprocess).
    if not args.max_num_tiles or args.max_num_tiles == MAX_NUM_TILES:
        args.max_num_tiles = int(os.environ.get("UHR_INTERNVL_MAX_TILES", MAX_NUM_TILES))
    if not args.global_view_max_side:
        args.global_view_max_side = int(os.environ.get("UHR_INTERNVL_GLOBAL_VIEW_MAX_SIDE", 0))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    # Backbone-transfer: the frozen operand contract carries the Qwen model
    # path; the transfer backend is InternVL3.5-8B. The actual model is
    # recorded in the shard summary for the audit trail.
    model_path = str(DEFAULT_MODEL)
    print(f"model_override: contract declares {contract.get('model_path')}, using {model_path}", flush=True)
    max_new_tokens = int(contract.get("max_new_tokens", 4096))
    temperature = float(contract.get("temperature", 0.0))
    rows = read_jsonl(args.tasks)

    import torch
    from PIL import Image
    from internvl_grounding import load_image

    Image.MAX_IMAGE_PIXELS = None
    torch.cuda.set_device(0)
    from transformers import AutoModel, AutoTokenizer

    model = (
        AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        .eval()
        .to("cuda")
    )
    # Infra fix (manual-review resolution 2026-08-31): sdpa instead of eager
    # when flash-attn is absent; no scientific element changes.
    if getattr(model.language_model.config, "_attn_implementation", "eager") == "eager":
        model.language_model.config._attn_implementation = "sdpa"
        model.language_model._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    outputs: list[dict[str, Any]] = []
    for attempt in range(1, int(args.max_infrastructure_attempts) + 1):
        pending = [
            row for row in rows
            if not any(o["task_id"] == row["task_id"] and not o.get("error") for o in outputs)
        ]
        if not pending:
            break
        for row in pending:
            task_id = str(row["task_id"])
            started = time.perf_counter()
            try:
                images = [str(p) for p in row["model_image_paths"]]
                from PIL import Image as PILImage
                from internvl_grounding import dynamic_preprocess, build_transform
                gvms = args.global_view_max_side
                pixel_list = []
                for p in images:
                    if gvms:
                        im = PILImage.open(p).convert("RGB")
                        if max(im.size) > gvms:
                            scale = gvms / max(im.size)
                            im = im.resize((round(im.size[0] * scale), round(im.size[1] * scale)), PILImage.LANCZOS)
                        transform = build_transform(448)
                        tiles = dynamic_preprocess(im, image_size=448, use_thumbnail=True, max_num=args.max_num_tiles)
                        pixel_list.append(torch.stack([transform(t) for t in tiles]).to(torch.bfloat16).to("cuda"))
                    else:
                        pixel_list.append(load_image(p, max_num=args.max_num_tiles).to(torch.bfloat16).to("cuda"))
                pixel_values = torch.cat(pixel_list, dim=0)
                num_patches_list = [pv.size(0) for pv in pixel_list]
                generation_config = {"max_new_tokens": max_new_tokens}
                if temperature > 0:
                    generation_config["do_sample"] = True
                    generation_config["temperature"] = temperature
                else:
                    generation_config["do_sample"] = False
                prompt = str(row["model_input_prompt"])
                # The frozen operand prompt is Qwen-style (images passed as
                # message parts, no <image> markers). InternVL chat needs one
                # marker per image; markers are replaced by vision tokens and
                # are invisible to the model - prompt text semantics unchanged.
                if "<image>" not in prompt:
                    prompt = "<image>\n" * len(images) + prompt
                response = model.chat(
                    tokenizer, pixel_values, prompt,
                    generation_config, num_patches_list=num_patches_list,
                    return_history=False, verbose=False,
                )
                if isinstance(response, tuple):
                    response = response[0]
                out = dict(row)
                out.update(
                    {
                        "raw_backend_response": str(response),
                        "model_response": str(response),
                        "api_mode": "map_internvl3_5_8b_transformers_local_multi_image",
                        "gpu_id": args.gpu,
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "completed_at_unix": int(time.time()),
                    }
                )
                append_jsonl(args.attempt_ledger, {"status": "success", "task_id": task_id, "attempt": attempt})
                outputs.append(out)
            except Exception as error:
                with (args.out.parent / "errors.log").open("a", encoding="utf-8") as stream:
                    stream.write(f"task_id={task_id}\n{type(error).__name__}: {error}\n{traceback.format_exc()}\n{'=' * 80}\n")
                append_jsonl(args.attempt_ledger, {"status": "error", "task_id": task_id, "attempt": attempt, "error": str(error)})
                if attempt == int(args.max_infrastructure_attempts):
                    out = dict(row)
                    out.update(
                        {
                            "error": f"{type(error).__name__}: {error}",
                            "error_type": "infrastructure",
                            "gpu_id": args.gpu,
                            "completed_at_unix": int(time.time()),
                        }
                    )
                    outputs.append(out)
    write_jsonl(args.out, outputs)
    success = sum(1 for o in outputs if not o.get("error"))
    args.summary.write_text(
        json.dumps(
            {
                "schema_version": "map-internvl-multi-image-summary-1",
                "task_rows": len(rows),
                "success": success,
                "errors": len(outputs) - success,
                "model_path": model_path,
                "gpu": args.gpu,
            },
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return 0 if success == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
