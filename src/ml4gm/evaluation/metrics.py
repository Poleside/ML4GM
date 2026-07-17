import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        metrics = {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    non_finite = [name for name, value in metrics.items() if not np.isfinite(value)]
    if non_finite:
        raise ValueError(f"Regression metrics contain non-finite values: {non_finite}")
    return metrics
