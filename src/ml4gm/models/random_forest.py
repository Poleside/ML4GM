from pathlib import Path
from typing import Any

import joblib
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor


class RandomForestAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        self.estimator = RandomForestRegressor(random_state=seed, **parameters)

    def fit(self, X: NDArray, y: NDArray) -> "RandomForestAdapter":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        return self.estimator.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
