from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .base import AdapterExecutionError, AdapterResult


class Qwen3VLAdapter:
    """Qwen3-VL vLLM adapter supporting both TP1 and multi-GPU TP."""

    def __init__(self, model_path: Path, config: dict[str, Any], physical_gpu_ids: list[str]):
        self.model_path = model_path
        self.config = config
        self.physical_gpu_ids = [str(value) for value in physical_gpu_ids]
        self.processor = None
        self.llm = None
        self.sampling_params = None

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from PIL import Image
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        Image.MAX_IMAGE_PIXELS = None
        tensor_parallel_size = int(self.config["tensor_parallel_size"])
        if tensor_parallel_size != len(self.physical_gpu_ids):
            raise ValueError(
                f"tensor_parallel_size={tensor_parallel_size} but worker sees "
                f"{len(self.physical_gpu_ids)} physical GPUs"
            )
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False
        )
        self.llm = LLM(
            model=str(self.model_path),
            tensor_parallel_size=tensor_parallel_size,
            dtype=self.config["dtype"],
            gpu_memory_utilization=float(self.config["gpu_memory_utilization"]),
            max_model_len=int(self.config["max_model_len"]),
            limit_mm_per_prompt={"image": 1},
            seed=int(self.config["seed"]),
            trust_remote_code=False,
            max_num_seqs=1,
        )
        self.sampling_params = SamplingParams(
            temperature=float(self.config["generation"]["temperature"]),
            top_p=float(self.config["generation"]["top_p"]),
            top_k=int(self.config["generation"]["top_k"]),
            max_tokens=int(self.config["generation"]["max_new_tokens"]),
            seed=int(self.config["seed"]),
            stop_token_ids=[],
        )

    def _prepare(self, task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        if self.processor is None:
            raise RuntimeError("adapter not loaded")
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{task['image_abs_path']}",
                    "min_pixels": int(self.config["image_preprocess"]["min_pixels"]),
                    "max_pixels": int(self.config["image_preprocess"]["max_pixels"]),
                },
                {"type": "text", "text": task["prompt_text"]},
            ],
        }]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=int(self.config["image_preprocess"]["patch_size"]),
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            raise ValueError("video input is forbidden for UHR-Bench")
        if not isinstance(image_inputs, list) or len(image_inputs) != 1:
            raise ValueError("exactly one processed image is required")
        size = getattr(image_inputs[0], "size", None)
        if not isinstance(size, tuple) or len(size) != 2:
            raise TypeError("qwen_vl_utils did not return a sized image")
        processed_size = {"width_px": int(size[0]), "height_px": int(size[1])}
        return ({
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs},
            "mm_processor_kwargs": video_kwargs,
        }, processed_size)

    def generate(self, task: dict[str, Any]) -> AdapterResult:
        if self.llm is None or self.sampling_params is None:
            raise RuntimeError("adapter not loaded")
        started = time.perf_counter()
        try:
            vllm_input, processed_size = self._prepare(task)
        except Exception as exc:
            raise AdapterExecutionError(
                f"preprocess failed: {type(exc).__name__}: {exc}", logical_vlm_calls=0
            ) from exc
        try:
            outputs = self.llm.generate([vllm_input], sampling_params=self.sampling_params, use_tqdm=False)
            candidate = outputs[0].outputs[0]
        except Exception as exc:
            raise AdapterExecutionError(
                f"generation failed: {type(exc).__name__}: {exc}", logical_vlm_calls=1
            ) from exc
        raw = candidate.text
        status = "success" if raw and raw.strip() else "empty_response"
        diagnostics = {
            "backend": "vllm",
            "adapter": "Qwen3VLAdapter",
            "physical_gpu_ids": self.physical_gpu_ids,
            "physical_gpu_id": self.physical_gpu_ids[0] if len(self.physical_gpu_ids) == 1 else None,
            "tensor_parallel_size": int(self.config["tensor_parallel_size"]),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "finish_reason": getattr(candidate, "finish_reason", None),
            "stop_reason": getattr(candidate, "stop_reason", None),
            "generated_token_count": len(getattr(candidate, "token_ids", []) or []),
            "semantic_postprocessing": False,
        }
        return AdapterResult(
            raw_response=raw,
            final_response=raw.strip() if raw is not None else None,
            runtime_status=status,
            transmitted_image_size=processed_size,
            model_input_size=processed_size,
            logical_vlm_calls=1,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        self.llm = None
        self.processor = None
        self.sampling_params = None
