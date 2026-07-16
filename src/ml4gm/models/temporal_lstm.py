from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml4gm.models.lstm_utils import finite_number, positive_integer
from ml4gm.models.torch_utils import require_torch, set_torch_seed


class TemporalLSTMAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        model = "Temporal-LSTM"
        self.seed = seed
        self.lookback = positive_integer(parameters.get("lookback", 3), model, "lookback")
        self.hidden = positive_integer(parameters.get("hidden", 128), model, "hidden")
        self.num_layers = positive_integer(parameters.get("num_layers", 1), model, "num_layers")
        self.dropout = finite_number(parameters.get("dropout", 0.2), model, "dropout")
        self.learning_rate = finite_number(
            parameters.get("learning_rate", 1e-3), model, "learning_rate"
        )
        self.epochs = positive_integer(parameters.get("epochs", 100), model, "epochs")
        self.batch_size = positive_integer(parameters.get("batch_size", 256), model, "batch_size")
        if not 0 <= self.dropout < 1:
            raise ValueError("Temporal-LSTM dropout must be at least 0 and less than 1")
        if self.learning_rate <= 0:
            raise ValueError("Temporal-LSTM learning_rate must be positive")
        self.parameters = {
            "lookback": self.lookback,
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
        self._input_features: int | None = None

    def _build(self, input_features: int) -> Any:
        torch = self._torch
        outer = self

        class Network(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_features,
                    outer.hidden,
                    num_layers=outer.num_layers,
                    dropout=outer.dropout if outer.num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.output = torch.nn.Linear(outer.hidden, 1)

            def forward(self, sequence: Any) -> Any:
                _, (hidden, _) = self.lstm(sequence)
                return self.output(hidden[-1])

        return Network().to(self._device)

    def fit(self, X: NDArray, y: NDArray) -> TemporalLSTMAdapter:
        del X, y
        raise ValueError("Temporal-LSTM requires fit_inputs(sequence, y)")

    def predict(self, X: NDArray) -> NDArray:
        del X
        raise ValueError("Temporal-LSTM requires predict_inputs(sequence)")

    def _validate_sequence(self, sequence: NDArray, *, fitted: bool) -> NDArray[np.float32]:
        values = np.asarray(sequence, dtype=np.float32)
        if (
            values.ndim != 3
            or len(values) == 0
            or values.shape[1] != self.lookback
            or values.shape[2] == 0
        ):
            raise ValueError(f"Temporal sequence must have shape (rows, {self.lookback}, features)")
        if not np.isfinite(values).all():
            raise ValueError("Temporal sequence must contain only finite values")
        if fitted and self._input_features is not None and values.shape[2] != self._input_features:
            raise ValueError(f"Temporal sequence must contain {self._input_features} features")
        return values

    def fit_inputs(self, sequence: NDArray, y: NDArray) -> TemporalLSTMAdapter:
        values = self._validate_sequence(sequence, fitted=False)
        targets = np.asarray(y, dtype=np.float32)
        if targets.ndim != 1 or len(targets) != len(values):
            raise ValueError("Temporal y must be one-dimensional and align with rows")
        if not np.isfinite(targets).all():
            raise ValueError("Temporal y must contain only finite values")
        set_torch_seed(self.seed)
        self._input_features = values.shape[2]
        self._model = self._build(self._input_features)
        dataset = self._torch.utils.data.TensorDataset(
            self._torch.tensor(values.tolist(), dtype=self._torch.float32, device=self._device),
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
            for batch_sequence, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(batch_sequence), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_inputs(self, sequence: NDArray) -> NDArray:
        if self._model is None:
            raise ValueError("Temporal-LSTM must be fitted before predict")
        values = self._validate_sequence(sequence, fitted=True)
        self._model.eval()
        with self._torch.no_grad():
            result = self._model(
                self._torch.tensor(values.tolist(), dtype=self._torch.float32, device=self._device)
            )
        return np.asarray(result.tolist(), dtype=np.float32).reshape(-1)

    def save(self, path: Path) -> None:
        if self._model is None or self._input_features is None:
            raise ValueError("Temporal-LSTM must be fitted before save")
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
