from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AdapterResult:
    raw_response: str | None
    final_response: str | None
    runtime_status: str
    transmitted_image_size: dict[str, int] | None
    model_input_size: dict[str, int] | None
    logical_vlm_calls: int
    tool_calls: int = 0
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    def load(self) -> None: ...

    def generate(self, task: dict[str, Any]) -> AdapterResult: ...

    def close(self) -> None: ...


class AdapterExecutionError(RuntimeError):
    def __init__(self, message: str, logical_vlm_calls: int):
        super().__init__(message)
        self.logical_vlm_calls = logical_vlm_calls
