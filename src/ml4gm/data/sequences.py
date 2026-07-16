from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class SequenceData:
    sequence: NDArray[np.float64]
    static: NDArray[np.float64]
    target: NDArray[np.float64]
    glacier_ids: NDArray
    years: NDArray[np.int_]


def _require_columns(frame: pd.DataFrame, columns: list[str], kind: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {kind} columns: {', '.join(missing[:5])}")


def _finite(values: NDArray, description: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{description} must contain only finite values")


def _validate_feature_names(
    features: list[str],
    *,
    target: str,
    glacier_id: str,
    year: str,
) -> None:
    forbidden = {target, glacier_id, year}
    leaking = [feature for feature in features if feature in forbidden]
    if leaking:
        labels = []
        if target in leaking:
            labels.append("target")
        if glacier_id in leaking:
            labels.append("glacier identifier")
        if year in leaking:
            labels.append("year")
        raise ValueError(f"Features must not include {' or '.join(labels)}")


def build_seasonal_sequences(
    frame: pd.DataFrame,
    monthly_variables: list[str],
    static_features: list[str],
    target: str = "dhdt",
    glacier_id: str = "rgiid",
    year: str = "year",
) -> SequenceData:
    if not monthly_variables:
        raise ValueError("monthly_variables must be non-empty")
    if not static_features:
        raise ValueError("static_features must be non-empty")
    _validate_feature_names(
        [*monthly_variables, *static_features],
        target=target,
        glacier_id=glacier_id,
        year=year,
    )
    monthly_columns = [
        [f"{month}_{variable}" for variable in monthly_variables] for month in range(1, 13)
    ]
    _require_columns(
        frame,
        [column for columns in monthly_columns for column in columns],
        "seasonal",
    )
    _require_columns(frame, [*static_features, target, glacier_id, year], "seasonal")
    sequence = np.stack(
        [frame[columns].to_numpy(dtype=float) for columns in monthly_columns],
        axis=1,
    )
    static = frame[static_features].to_numpy(dtype=float)
    targets = frame[target].to_numpy(dtype=float)
    years = frame[year].to_numpy(dtype=int)
    _finite(sequence, "Seasonal sequence")
    _finite(static, "Seasonal static features")
    _finite(targets, "Seasonal target")
    return SequenceData(
        sequence=sequence,
        static=static,
        target=targets,
        glacier_ids=frame[glacier_id].to_numpy(),
        years=years,
    )


def build_temporal_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    lookback: int,
    glacier_id: str,
    year: str,
    target: str,
) -> SequenceData:
    if isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback <= 0:
        raise ValueError("lookback must be a positive integer")
    if not feature_columns:
        raise ValueError("feature_columns must be non-empty")
    _validate_feature_names(feature_columns, target=target, glacier_id=glacier_id, year=year)
    _require_columns(frame, [*feature_columns, glacier_id, year, target], "temporal")
    feature_values = frame[feature_columns].to_numpy(dtype=float)
    target_values = frame[target].to_numpy(dtype=float)
    year_values = pd.to_numeric(frame[year], errors="raise").to_numpy(dtype=float)
    _finite(feature_values, "Temporal features")
    _finite(target_values, "Temporal target")
    _finite(year_values, "Temporal years")
    if not np.equal(year_values, np.floor(year_values)).all():
        raise ValueError("Temporal years must be integers")

    sequences: list[NDArray[np.float64]] = []
    targets: list[float] = []
    glaciers: list[object] = []
    years: list[int] = []
    ordered_frame = frame.sort_values([glacier_id, year], kind="stable")
    for glacier, group in ordered_frame.groupby(glacier_id, sort=False):
        ordered = group.reset_index(drop=True)
        if ordered[year].duplicated().any():
            raise ValueError("Temporal glacier-year rows must be unique")
        for end in range(int(lookback), len(ordered)):
            start = end - int(lookback)
            observed = ordered.loc[start:end, year].to_numpy(dtype=int)
            if not np.all(np.diff(observed) == 1):
                continue
            window = ordered.iloc[start:end][feature_columns].to_numpy(dtype=float)
            sequences.append(window)
            targets.append(float(ordered.iloc[end][target]))
            glaciers.append(glacier)
            years.append(int(ordered.iloc[end][year]))
    if not sequences:
        raise ValueError("No valid temporal sequences were produced")
    sequence = np.stack(sequences)
    return SequenceData(
        sequence=sequence,
        static=np.empty((len(sequence), 0), dtype=float),
        target=np.asarray(targets, dtype=float),
        glacier_ids=np.asarray(glaciers),
        years=np.asarray(years, dtype=int),
    )
