import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from ml4gm.models import create_model


class FakeLGBMRegressor:
    def __init__(self, **parameters: object) -> None:
        self.parameters = parameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FakeLGBMRegressor":
        self.value = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value)


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


def test_lightgbm_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_lightgbm(name: str, *args: object, **kwargs: object) -> object:
        if name == "lightgbm":
            raise ImportError("LightGBM unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_lightgbm)
    with pytest.raises(RuntimeError, match=r"ml4gm\[lightgbm\]"):
        create_model("lightgbm", {}, seed=42)


def test_lightgbm_native_library_error_has_dependency_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_with_broken_lightgbm(name: str, *args: object, **kwargs: object) -> object:
        if name == "lightgbm":
            raise OSError("libomp.dylib is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_lightgbm)
    with pytest.raises(RuntimeError, match=r"ml4gm\[lightgbm\]"):
        create_model("lightgbm", {}, seed=42)


def test_lightgbm_adapter_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(LGBMRegressor=FakeLGBMRegressor),
    )
    model = create_model("lightgbm", {"n_estimators": 5, "n_jobs": 1}, seed=42)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == (4,)

    path = tmp_path / "nested" / "model.joblib"
    model.save(path)
    assert path.exists()
    np.testing.assert_allclose(joblib.load(path).predict(X), predictions)


def test_installed_lightgbm_adapter_round_trip(tmp_path: Path) -> None:
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("LightGBM is not installed")
    try:
        from lightgbm import LGBMRegressor  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"LightGBM native runtime unavailable: {exc}")

    model = create_model("lightgbm", {"n_estimators": 5, "n_jobs": 1}, seed=42)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    predictions = model.fit(X, y).predict(X)

    path = tmp_path / "model.joblib"
    model.save(path)
    np.testing.assert_allclose(joblib.load(path).predict(X), predictions)
