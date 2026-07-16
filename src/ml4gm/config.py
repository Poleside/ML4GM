from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_MODELS = {
    "random_forest",
    "lightgbm",
    "mlp",
    "seasonal_lstm",
    "temporal_lstm",
}
SUPPORTED_VALIDATION = {"loyo", "spatial", "block"}


class ConfigError(ValueError):
    """Raised when an ML4GM configuration is invalid."""


@dataclass(frozen=True)
class DataConfig:
    input: Path
    target: str = "dhdt"
    glacier_id: str = "rgiid"
    year: str = "year"
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationConfig:
    strategy: str
    folds: int = 5


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig
    model: ModelConfig
    validation: ValidationConfig
    random_seed: int = 42
    run_name: str = "ml4gm-run"

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in configuration file: {path}") from exc
        try:
            model_name = str(raw["model"]["name"])
            strategy = str(raw["validation"]["strategy"])
            input_path = Path(raw["data"]["input"])
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"Missing required configuration field: {exc}") from exc
        if model_name not in SUPPORTED_MODELS:
            raise ConfigError(f"Unsupported model: {model_name}")
        if strategy not in SUPPORTED_VALIDATION:
            raise ConfigError(f"Unsupported validation strategy: {strategy}")
        data_raw = raw["data"]
        return cls(
            data=DataConfig(
                input=input_path,
                target=str(data_raw.get("target", "dhdt")),
                glacier_id=str(data_raw.get("glacier_id", "rgiid")),
                year=str(data_raw.get("year", "year")),
                output_dir=Path(data_raw.get("output_dir", "outputs")),
            ),
            model=ModelConfig(model_name, dict(raw["model"].get("parameters", {}))),
            validation=ValidationConfig(
                strategy,
                int(raw["validation"].get("folds", 5)),
            ),
            random_seed=int(raw.get("random_seed", 42)),
            run_name=str(raw.get("run_name", "ml4gm-run")),
        )
