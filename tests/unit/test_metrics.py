import numpy as np
import pytest

from ml4gm.evaluation.metrics import regression_metrics


def test_regression_metrics() -> None:
    metrics = regression_metrics(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    assert metrics == {"r2": -1.0, "rmse": 2**-0.5, "mae": 0.5}


def test_regression_metrics_rejects_non_finite_results() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        regression_metrics(np.array([1.0]), np.array([1.0]))
