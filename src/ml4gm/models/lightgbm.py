from pathlib import Path
from typing import Any

import joblib
from numpy.typing import NDArray


class LightGBMAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM support requires: pip install 'ml4gm[lightgbm]'"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"LightGBM native runtime failed to load: {exc}") from exc

        effective_parameters = dict(parameters)
        effective_parameters["random_state"] = seed
        effective_parameters.setdefault("verbosity", -1)
        self.estimator = LGBMRegressor(**effective_parameters)

    def fit(self, X: NDArray, y: NDArray) -> "LightGBMAdapter":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        return self.estimator.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
