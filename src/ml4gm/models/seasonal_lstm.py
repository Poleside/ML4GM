from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml4gm.models.lstm_utils import finite_number, positive_integer
from ml4gm.models.torch_utils import require_torch, set_torch_seed


class SeasonalLSTMAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        model = "Seasonal-LSTM"
        self.seed = seed
        self.hidden = positive_integer(parameters.get("hidden", 128), model, "hidden")
        self.num_layers = positive_integer(parameters.get("num_layers", 1), model, "num_layers")
        self.dropout = finite_number(parameters.get("dropout", 0.2), model, "dropout")
        self.learning_rate = finite_number(
            parameters.get("learning_rate", 1e-3), model, "learning_rate"
        )
        self.epochs = positive_integer(parameters.get("epochs", 100), model, "epochs")
        self.batch_size = positive_integer(parameters.get("batch_size", 256), model, "batch_size")
        if not 0 <= self.dropout < 1:
            raise ValueError("Seasonal-LSTM dropout must be at least 0 and less than 1")
        if self.learning_rate <= 0:
            raise ValueError("Seasonal-LSTM learning_rate must be positive")
        self.parameters = {
            "hidden": self.hidden,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        self._torch = require_torch()
        self._device = self._torch.device("cpu")
        self._model: Any | None = None
        self._dimensions: tuple[int, int] | None = None

    def _build(self, sequence_features: int, static_features: int) -> Any:
        torch = self._torch
        outer = self

        class Network(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    sequence_features,
                    outer.hidden,
                    num_layers=outer.num_layers,
                    dropout=outer.dropout if outer.num_layers > 1 else 0.0,
                    batch_first=True,
                )
                head_hidden = max(1, outer.hidden // 2)
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(outer.hidden + static_features, head_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(outer.dropout),
                    torch.nn.Linear(head_hidden, 1),
                )

            def forward(self, sequence: Any, static: Any) -> Any:
                _, (hidden, _) = self.lstm(sequence)
                return self.head(torch.cat([hidden[-1], static], dim=1))

        return Network().to(self._device)

    def fit(self, X: NDArray, y: NDArray) -> SeasonalLSTMAdapter:
        del X, y
        raise ValueError("Seasonal-LSTM requires fit_inputs(sequence, static, y)")

    def predict(self, X: NDArray) -> NDArray:
        del X
        raise ValueError("Seasonal-LSTM requires predict_inputs(sequence, static)")

    def _validate_inputs(
        self, sequence: NDArray, static: NDArray, *, fitted: bool
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        values = np.asarray(sequence, dtype=np.float32)
        static_values = np.asarray(static, dtype=np.float32)
        if values.ndim != 3 or len(values) == 0 or values.shape[1] != 12 or values.shape[2] == 0:
            raise ValueError("Seasonal sequence must have shape (rows, 12, features)")
        if static_values.ndim != 2 or len(static_values) != len(values):
            raise ValueError("Seasonal static features must align with sequence rows")
        if not np.isfinite(values).all() or not np.isfinite(static_values).all():
            raise ValueError("Seasonal inputs must contain only finite values")
        if fitted and self._dimensions is not None:
            if values.shape[2] != self._dimensions[0]:
                raise ValueError(
                    f"Seasonal sequence must contain {self._dimensions[0]} sequence features"
                )
            if static_values.shape[1] != self._dimensions[1]:
                raise ValueError(
                    f"Seasonal static input must contain {self._dimensions[1]} features"
                )
        return values, static_values

    def fit_inputs(self, sequence: NDArray, static: NDArray, y: NDArray) -> SeasonalLSTMAdapter:
        values, static_values = self._validate_inputs(sequence, static, fitted=False)
        targets = np.asarray(y, dtype=np.float32)
        if targets.ndim != 1 or len(targets) != len(values):
            raise ValueError("Seasonal y must be one-dimensional and align with rows")
        if not np.isfinite(targets).all():
            raise ValueError("Seasonal y must contain only finite values")
        set_torch_seed(self.seed)
        self._dimensions = (values.shape[2], static_values.shape[1])
        self._model = self._build(*self._dimensions)
        dataset = self._torch.utils.data.TensorDataset(
            self._torch.tensor(values.tolist(), dtype=self._torch.float32, device=self._device),
            self._torch.tensor(
                static_values.tolist(), dtype=self._torch.float32, device=self._device
            ),
            self._torch.tensor(
                targets.reshape(-1, 1).tolist(), dtype=self._torch.float32, device=self._device
            ),
        )
        loader = self._torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._torch.Generator().manual_seed(self.seed),
        )
        optimizer = self._torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = self._torch.nn.MSELoss()
        self._model.train()
        for _ in range(self.epochs):
            for batch_sequence, batch_static, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(batch_sequence, batch_static), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_inputs(self, sequence: NDArray, static: NDArray) -> NDArray:
        if self._model is None:
            raise ValueError("Seasonal-LSTM must be fitted before predict")
        values, static_values = self._validate_inputs(sequence, static, fitted=True)
        self._model.eval()
        with self._torch.no_grad():
            result = self._model(
                self._torch.tensor(values.tolist(), dtype=self._torch.float32, device=self._device),
                self._torch.tensor(
                    static_values.tolist(), dtype=self._torch.float32, device=self._device
                ),
            )
        return np.asarray(result.tolist(), dtype=np.float32).reshape(-1)

    def save(self, path: Path) -> None:
        if self._model is None or self._dimensions is None:
            raise ValueError("Seasonal-LSTM must be fitted before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "state_dict": self._model.state_dict(),
                "dimensions": self._dimensions,
                "parameters": self.parameters,
                "seed": self.seed,
            },
            path,
        )
