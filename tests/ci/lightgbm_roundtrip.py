"""Exercise the installed LightGBM adapter without optional-dependency skips."""

from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from lightgbm import LGBMRegressor

from ml4gm.models import create_model


def main() -> None:
    assert LGBMRegressor.__module__.startswith("lightgbm")
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = create_model("lightgbm", {"n_estimators": 5, "n_jobs": 1}, seed=42)
    predictions = model.fit(X, y).predict(X)
    assert predictions.shape == (4,)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        model.save(path)
        restored = joblib.load(path)
        np.testing.assert_allclose(restored.predict(X), predictions)


if __name__ == "__main__":
    main()
