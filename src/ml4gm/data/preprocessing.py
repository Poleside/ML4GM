from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PreparedFold:
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.float64]
    y_test: NDArray[np.float64]
    feature_names: list[str]
    transformer: Pipeline


def prepare_fold(
    frame: pd.DataFrame,
    train_idx: NDArray[np.int_],
    test_idx: NDArray[np.int_],
    target: str,
    glacier_id: str,
    year: str,
) -> PreparedFold:
    features = [column for column in frame.columns if column not in {target, glacier_id, year}]
    if not features:
        raise ValueError("No model features remain after excluding identifiers and target")

    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]
    transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    X_train = transformer.fit_transform(train[features]).astype(float)
    X_test = transformer.transform(test[features]).astype(float)
    return PreparedFold(
        X_train,
        X_test,
        train[target].to_numpy(dtype=float),
        test[target].to_numpy(dtype=float),
        features,
        transformer,
    )
