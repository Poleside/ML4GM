from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


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

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, allow_nan=False), encoding="utf-8")
        return path
