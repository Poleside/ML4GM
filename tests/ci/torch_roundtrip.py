"""Exercise the installed PyTorch adapter without optional-dependency skips."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ml4gm.models import create_model


def main() -> None:
    major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
    assert (major, minor) >= (2, 4)
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = X.sum(axis=1)
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
