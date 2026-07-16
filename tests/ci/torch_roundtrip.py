"""Exercise the installed PyTorch adapter without optional-dependency skips."""

import importlib
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ml4gm.models import create_model


def main() -> None:
    warnings.filterwarnings("error", message="Failed to initialize NumPy.*")
    torch = importlib.import_module("torch")
    major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
    assert (major, minor) >= (2, 4)
    numpy_major, numpy_minor = (int(part) for part in np.__version__.split(".")[:2])
    assert (numpy_major, numpy_minor) == (1, 26)
    print(f"Verified PyTorch {torch.__version__} with NumPy {np.__version__}")

    X = np.arange(24, dtype=float).reshape(8, 3)
    y = X.sum(axis=1)
    bridge_source = X.astype(np.float32)
    bridge_tensor = torch.from_numpy(bridge_source)
    bridge_result = bridge_tensor.numpy()
    np.testing.assert_array_equal(bridge_result, bridge_source)

    model = create_model(
        "mlp",
        {"hidden_sizes": [8, 4], "epochs": 2, "batch_size": 4},
        seed=42,
    )
    predictions = model.fit(X, y).predict(X)
    assert predictions.shape == (8,)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        model.save(path)
        checkpoint = torch.load(path)
        assert checkpoint["input_features"] == 3
        assert checkpoint["seed"] == 42


if __name__ == "__main__":
    main()
