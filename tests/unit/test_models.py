from pathlib import Path

import joblib
import numpy as np
import pytest

from ml4gm.models import create_model


def test_random_forest_adapter_round_trip(tmp_path: Path) -> None:
    model = create_model("random_forest", {"n_estimators": 5, "n_jobs": 1}, seed=42)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == (4,)

    path = tmp_path / "nested" / "model.joblib"
    model.save(path)
    assert path.exists()
    np.testing.assert_allclose(joblib.load(path).predict(X), predictions)


def test_create_model_rejects_unavailable_model() -> None:
    with pytest.raises(ValueError, match="Unsupported or unavailable model: lightgbm"):
        create_model("lightgbm", {}, seed=42)
