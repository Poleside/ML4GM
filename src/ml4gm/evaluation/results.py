from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FoldResult:
    fold: str
    n_train: int
    n_test: int
    r2: float
    rmse: float
    mae: float


@dataclass(frozen=True)
class RunResult:
    run_name: str
    model: str
    validation: str
    seed: int
    folds: list[FoldResult]
    status: str = "completed"
    resolved_config: dict[str, Any] = field(default_factory=dict)
    software: dict[str, str] = field(default_factory=dict)
    input_manifest: dict[str, Any] = field(default_factory=dict)
    data_summary: dict[str, Any] = field(default_factory=dict)
    model_parameters: dict[str, Any] = field(default_factory=dict)
    validation_details: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=lambda: {"run_record": "result.json"})
    error: dict[str, str] | None = None

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        return path
