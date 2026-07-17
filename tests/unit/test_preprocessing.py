import numpy as np
import pandas as pd
import pytest

from ml4gm.data.preprocessing import prepare_fold


def test_scaler_is_fitted_on_training_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A", "A", "B"],
            "year": [2000, 2001, 2000],
            "dhdt": [0.0, 1.0, 2.0],
            "x": [0.0, 2.0, 100.0],
        }
    )
    fold = prepare_fold(frame, np.array([0, 1]), np.array([2]), "dhdt", "rgiid", "year")
    assert np.isclose(fold.X_train.mean(), 0.0)
    assert fold.X_test[0, 0] > 50.0


def test_rejects_feature_missing_from_all_training_rows() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A", "A", "B"],
            "year": [2000, 2001, 2000],
            "dhdt": [0.0, 1.0, 2.0],
            "x": [1.0, 2.0, 3.0],
            "snowfall": [np.nan, np.nan, 5.0],
        }
    )

    with pytest.raises(ValueError, match="snowfall"):
        prepare_fold(frame, np.array([0, 1]), np.array([2]), "dhdt", "rgiid", "year")
