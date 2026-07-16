import builtins
import importlib
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

from ml4gm.models import create_model

torch_available = importlib.util.find_spec("torch") is not None


def test_torch_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    require_torch = importlib.import_module("ml4gm.models.torch_utils").require_torch
    original_import = builtins.__import__

    def import_without_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("PyTorch unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    with pytest.raises(RuntimeError, match=r"pip install 'ml4gm\[torch\]'"):
        require_torch()


@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
def test_set_torch_seed_is_deterministic() -> None:
    torch_utils = importlib.import_module("ml4gm.models.torch_utils")
    require_torch = torch_utils.require_torch
    set_torch_seed = torch_utils.set_torch_seed
    torch = require_torch()

    set_torch_seed(42)
    first = (random.random(), np.random.random(), torch.rand(2))
    set_torch_seed(42)
    second = (random.random(), np.random.random(), torch.rand(2))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
def test_mlp_forward_and_fit() -> None:
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = X.sum(axis=1)
    model = create_model(
        "mlp",
        {"hidden_sizes": [8, 4], "epochs": 2, "batch_size": 4},
        seed=42,
    )

    model.fit(X, y)

    assert model.predict(X).shape == (8,)


@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
def test_mlp_training_is_deterministic() -> None:
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = X.sum(axis=1)
    parameters = {
        "hidden_sizes": [8, 4],
        "dropout": 0.2,
        "epochs": 3,
        "batch_size": 4,
    }

    first = create_model("mlp", parameters, seed=42).fit(X, y).predict(X)
    second = create_model("mlp", parameters, seed=42).fit(X, y).predict(X)

    np.testing.assert_array_equal(first, second)


@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"hidden_sizes": []}, "hidden_sizes"),
        ({"hidden_sizes": [8, 0]}, "hidden_sizes"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_mlp_rejects_invalid_parameters(
    parameters: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_model("mlp", parameters, seed=42)


@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
def test_mlp_rejects_predict_and_save_before_fit(tmp_path: Path) -> None:
    model = create_model("mlp", {}, seed=42)

    with pytest.raises(ValueError, match="fitted before predict"):
        model.predict(np.ones((2, 3)))
    with pytest.raises(ValueError, match="fitted before save"):
        model.save(tmp_path / "model.pt")


@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.array([]), np.array([]), "non-empty two-dimensional"),
        (np.ones((2, 3, 1)), np.ones(2), "non-empty two-dimensional"),
        (np.ones((2, 3)), np.ones(3), "same number of rows"),
        (np.ones((2, 3)), np.ones((2, 2)), "one-dimensional"),
    ],
)
def test_mlp_fit_validates_input_shape(
    X: np.ndarray, y: np.ndarray, message: str
) -> None:
    model = create_model("mlp", {"epochs": 1}, seed=42)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
def test_mlp_predict_validates_input_shape_and_feature_count() -> None:
    model = create_model("mlp", {"epochs": 1}, seed=42)
    model.fit(np.ones((4, 3)), np.ones(4))

    with pytest.raises(ValueError, match="two-dimensional"):
        model.predict(np.ones(3))
    with pytest.raises(ValueError, match="3 features"):
        model.predict(np.ones((2, 4)))
