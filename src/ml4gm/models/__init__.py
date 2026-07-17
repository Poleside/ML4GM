from typing import Any

from ml4gm.models.base import ModelAdapter
from ml4gm.models.random_forest import RandomForestAdapter


def create_model(name: str, parameters: dict[str, Any], seed: int) -> ModelAdapter:
    if name == "random_forest":
        return RandomForestAdapter(parameters, seed)
    if name == "lightgbm":
        from ml4gm.models.lightgbm import LightGBMAdapter

        return LightGBMAdapter(parameters, seed)
    if name == "mlp":
        from ml4gm.models.mlp import MLPAdapter

        return MLPAdapter(parameters, seed)
    if name == "seasonal_lstm":
        from ml4gm.models.seasonal_lstm import SeasonalLSTMAdapter

        return SeasonalLSTMAdapter(parameters, seed)
    if name == "temporal_lstm":
        from ml4gm.models.temporal_lstm import TemporalLSTMAdapter

        return TemporalLSTMAdapter(parameters, seed)
    raise ValueError(f"Unsupported or unavailable model: {name}")


__all__ = ["ModelAdapter", "create_model"]
