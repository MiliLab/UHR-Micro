from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .base import AdapterExecutionError, AdapterResult


INTERNVL_SYSTEM_MESSAGE = (
    "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的"
    "多模态大语言模型。"
)


def build_internvl_query(prompt_text: str) -> str:
    """Build the checkpoint-native internvl2_5 single-turn prompt exactly."""
    return (
        f"<|im_start|>system\n{INTERNVL_SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n<image>\n{prompt_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def target_ratios(min_num: int, max_num: int) -> list[tuple[int, int]]:
    ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    return sorted(ratios, key=lambda value: value[0] * value[1])


def tile_metadata(
    width: int,
    height: int,
    *,
    image_size: int = 448,
    min_num: int = 1,
    max_num: int = 12,
    use_thumbnail: bool = True,
) -> dict[str, Any]:
    """Resolve the official InternVL dynamic grid without decoding the image."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    aspect_ratio = width / height
    area = width * height
    best_diff = float("inf")
    best = (1, 1)
    for ratio in target_ratios(min_num, max_num):
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff = diff
            best = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    content = best[0] * best[1]
    thumbnail = bool(use_thumbnail and content != 1)
    total = content + int(thumbnail)
    return {
        "tile_size": {"width_px": image_size, "height_px": image_size},
        "grid": {"columns": best[0], "rows": best[1]},
        "grid_canvas": {"width_px": image_size * best[0], "height_px": image_size * best[1]},
        "content_patch_count": content,
        "thumbnail_included": thumbnail,
        "total_patch_count": total,
    }


class InternVLAdapter:
    """Native vLLM adapter for frozen InternVL3.5 UHR-Bench-v4 inference."""

    def __init__(self, model_path: Path, config: dict[str, Any], physical_gpu_ids: list[str]):
        self.model_path = model_path
        self.config = config
        self.physical_gpu_ids = [str(value) for value in physical_gpu_ids]
        self.tokenizer = None
        self.llm = None
        self.sampling_params = None
        self.image_seq_length = int(config["image_preprocess"]["image_seq_length"])

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from PIL import Image
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM, SamplingParams

        Image.MAX_IMAGE_PIXELS = None
        tensor_parallel_size = int(self.config["tensor_parallel_size"])
        if tensor_parallel_size != len(self.physical_gpu_ids):
            raise ValueError(
                f"tensor_parallel_size={tensor_parallel_size} but worker sees "
                f"{len(self.physical_gpu_ids)} physical GPUs"
            )

        checkpoint = AutoConfig.from_pretrained(
            str(self.model_path), trust_remote_code=True, local_files_only=True
        )
        expected = self.config["image_preprocess"]
        observed = {
            "force_image_size": int(checkpoint.force_image_size),
            "min_dynamic_patch": int(checkpoint.min_dynamic_patch),
            "max_dynamic_patch": int(checkpoint.max_dynamic_patch),
            "use_thumbnail": bool(checkpoint.use_thumbnail),
            "downsample_ratio": float(checkpoint.downsample_ratio),
            "patch_size": int(checkpoint.vision_config.patch_size),
            "template": str(checkpoint.template),
        }
        required = {
            "force_image_size": int(expected["tile_size_px"]),
            "min_dynamic_patch": int(expected["min_dynamic_patch"]),
            "max_dynamic_patch": int(expected["max_dynamic_patch"]),
            "use_thumbnail": bool(expected["use_thumbnail"]),
            "downsample_ratio": float(expected["downsample_ratio"]),
            "patch_size": int(expected["patch_size"]),
            "template": "internvl2_5",
        }
        if observed != required:
            raise ValueError(f"checkpoint preprocessing contract mismatch: {observed} != {required}")
        computed_seq = int(
            (required["force_image_size"] // required["patch_size"]) ** 2
            * required["downsample_ratio"] ** 2
        )
        if computed_seq != self.image_seq_length:
            raise ValueError("configured image sequence length differs from checkpoint")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        if int(self.tokenizer.model_max_length) != int(self.config["max_model_len"]):
            raise ValueError("tokenizer model_max_length differs from frozen adapter")

        self.llm = LLM(
            model=str(self.model_path),
            tensor_parallel_size=tensor_parallel_size,
            dtype=self.config["dtype"],
            gpu_memory_utilization=float(self.config["gpu_memory_utilization"]),
            max_model_len=int(self.config["max_model_len"]),
            limit_mm_per_prompt={"image": 1},
            seed=int(self.config["seed"]),
            trust_remote_code=True,
            tokenizer_mode=str(self.config["tokenizer_mode"]),
            max_num_seqs=int(self.config["max_num_seqs"]),
            mm_encoder_tp_mode=str(self.config["mm_encoder_tp_mode"]),
        )
        generation = self.config["generation"]
        self.sampling_params = SamplingParams(
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
            max_tokens=int(generation["max_new_tokens"]),
            seed=int(self.config["seed"]),
            stop_token_ids=[int(generation["eos_token_id"])],
        )

    def _prepare(self, task: dict[str, Any]):
        if self.tokenizer is None:
            raise RuntimeError("adapter not loaded")
        from PIL import Image

        image = Image.open(task["image_abs_path"]).convert("RGB")
        width, height = image.size
        expected = task["source_image_size"]
        if (width, height) != (int(expected["width_px"]), int(expected["height_px"])):
            raise ValueError(
                f"source image dimensions differ from manifest: {(width, height)} != "
                f"{(expected['width_px'], expected['height_px'])}"
            )
        preprocess = self.config["image_preprocess"]
        tiles = tile_metadata(
            width,
            height,
            image_size=int(preprocess["tile_size_px"]),
            min_num=int(preprocess["min_dynamic_patch"]),
            max_num=int(preprocess["max_dynamic_patch"]),
            use_thumbnail=bool(preprocess["use_thumbnail"]),
        )
        query = build_internvl_query(task["prompt_text"])
        text_tokens_with_literal = len(self.tokenizer(query, add_special_tokens=False).input_ids)
        visual_tokens = int(tiles["total_patch_count"]) * self.image_seq_length
        # vLLM's InternVL replacement retains an effective three-token image
        # wrapper around the context features, exactly offsetting the three
        # slow-tokenizer IDs of the literal ``<image>`` marker.
        estimated_prompt_tokens = text_tokens_with_literal + visual_tokens
        if estimated_prompt_tokens + int(self.config["generation"]["max_new_tokens"]) > int(
            self.config["max_model_len"]
        ):
            raise ValueError("prompt plus frozen output allowance exceeds max_model_len")
        return (
            {"prompt": query, "multi_modal_data": {"image": image}},
            {"width_px": width, "height_px": height},
            tiles,
            text_tokens_with_literal,
            visual_tokens,
            estimated_prompt_tokens,
            hashlib.sha256(query.encode("utf-8")).hexdigest(),
        )

    def generate(self, task: dict[str, Any]) -> AdapterResult:
        if self.llm is None or self.sampling_params is None:
            raise RuntimeError("adapter not loaded")
        started = time.perf_counter()
        try:
            (
                vllm_input,
                source_size,
                tiles,
                text_tokens,
                visual_tokens,
                estimated_prompt_tokens,
                native_query_sha256,
            ) = self._prepare(task)
        except Exception as exc:
            raise AdapterExecutionError(
                f"preprocess failed: {type(exc).__name__}: {exc}", logical_vlm_calls=0
            ) from exc
        try:
            outputs = self.llm.generate([vllm_input], sampling_params=self.sampling_params, use_tqdm=False)
            request_output = outputs[0]
            candidate = request_output.outputs[0]
        except Exception as exc:
            raise AdapterExecutionError(
                f"generation failed: {type(exc).__name__}: {exc}", logical_vlm_calls=1
            ) from exc
        raw = candidate.text
        prompt_token_ids = list(getattr(request_output, "prompt_token_ids", []) or [])
        generated_ids = list(getattr(candidate, "token_ids", []) or [])
        diagnostics = {
            "backend": "vllm",
            "adapter": "InternVLAdapter",
            "physical_gpu_ids": self.physical_gpu_ids,
            "physical_gpu_id": self.physical_gpu_ids[0] if len(self.physical_gpu_ids) == 1 else None,
            "tensor_parallel_size": int(self.config["tensor_parallel_size"]),
            "mm_encoder_tp_mode": self.config["mm_encoder_tp_mode"],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "finish_reason": getattr(candidate, "finish_reason", None),
            "stop_reason": getattr(candidate, "stop_reason", None),
            "prompt_token_count": len(prompt_token_ids) or estimated_prompt_tokens,
            "estimated_prompt_token_count": estimated_prompt_tokens,
            "text_tokens_with_literal_image_marker": text_tokens,
            "visual_token_count": visual_tokens,
            "generated_token_count": len(generated_ids),
            "generated_token_ids_sha256": hashlib.sha256(
                json.dumps(generated_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "native_query_sha256": native_query_sha256,
            "native_template": "internvl2_5",
            "native_system_message_sha256": hashlib.sha256(
                INTERNVL_SYSTEM_MESSAGE.encode("utf-8")
            ).hexdigest(),
            "tile_preprocess": tiles,
            "image_sequence_length_per_patch": self.image_seq_length,
            "semantic_postprocessing": False,
        }
        return AdapterResult(
            raw_response=raw,
            final_response=raw.strip() if raw is not None else None,
            runtime_status="success",
            transmitted_image_size=source_size,
            model_input_size=None,
            logical_vlm_calls=1,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
