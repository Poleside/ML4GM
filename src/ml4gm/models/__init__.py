from typing import Any

from ml4gm.models.base import ModelAdapter
from ml4gm.models.random_forest import RandomForestAdapter


def create_model(name: str, parameters: dict[str, Any], seed: int) -> ModelAdapter:
    if name == "random_forest":
        return RandomForestAdapter(parameters, seed)
    if name == "lightgbm":
        from ml4gm.models.lightgbm import LightGBMAdapter

        return LightGBMAdapter(parameters, seed)
    raise ValueError(f"Unsupported or unavailable model: {name}")


__all__ = ["ModelAdapter", "create_model"]
