from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml4gm.models.torch_utils import require_torch, set_torch_seed


class MLPAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        self.parameters = dict(parameters)
        self.seed = seed
        self.hidden_sizes = [
            int(value) for value in parameters.get("hidden_sizes", [128, 64])
        ]
        self.dropout = float(parameters.get("dropout", 0.2))
        self.learning_rate = float(parameters.get("learning_rate", 1e-3))
        self.epochs = int(parameters.get("epochs", 100))
        self.batch_size = int(parameters.get("batch_size", 256))
        self._validate_parameters()
        self._torch = require_torch()
        self._model: Any | None = None
        self._input_features: int | None = None

    def _validate_parameters(self) -> None:
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("MLP hidden_sizes must contain positive integers")
        if not 0 <= self.dropout < 1:
            raise ValueError("MLP dropout must be at least 0 and less than 1")
        if self.learning_rate <= 0:
            raise ValueError("MLP learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("MLP epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("MLP batch_size must be positive")

    def _build(self, input_features: int) -> Any:
        nn = self._torch.nn
        layers: list[Any] = []
        previous = input_features
        for hidden in self.hidden_sizes:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(self.dropout)])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)

    def fit(self, X: NDArray, y: NDArray) -> MLPAdapter:
        values = np.asarray(X, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0:
            raise ValueError("MLP X must be a non-empty two-dimensional array")
        if targets.ndim != 1:
            raise ValueError("MLP y must be a one-dimensional array")
        if len(values) != len(targets):
            raise ValueError("MLP X and y must contain the same number of rows")

        set_torch_seed(self.seed)
        self._input_features = values.shape[1]
        self._model = self._build(self._input_features)
        dataset = self._torch.utils.data.TensorDataset(
            self._torch.tensor(values.tolist(), dtype=self._torch.float32),
            self._torch.tensor(targets.reshape(-1, 1).tolist(), dtype=self._torch.float32),
        )
        generator = self._torch.Generator().manual_seed(self.seed)
        loader = self._torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = self._torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
        )
        loss_fn = self._torch.nn.MSELoss()
        self._model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: NDArray) -> NDArray:
        if self._model is None or self._input_features is None:
            raise ValueError("MLP must be fitted before predict")
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("MLP X must be a two-dimensional array")
        if values.shape[1] != self._input_features:
            raise ValueError(f"MLP X must contain {self._input_features} features")
        self._model.eval()
        with self._torch.no_grad():
            tensor = self._torch.tensor(values.tolist(), dtype=self._torch.float32)
            return np.asarray(self._model(tensor).tolist(), dtype=np.float32).reshape(-1)

    def save(self, path: Path) -> None:
        if self._model is None or self._input_features is None:
            raise ValueError("MLP must be fitted before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "state_dict": self._model.state_dict(),
                "input_features": self._input_features,
                "parameters": self.parameters,
                "seed": self.seed,
            },
            path,
        )
