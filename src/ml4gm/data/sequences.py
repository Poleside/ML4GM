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
    context_glacier_ids: NDArray
    context_years: NDArray[np.int_]


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


def _validate_row_identity(
    frame: pd.DataFrame, glacier_id: str, year: str, kind: str
) -> NDArray[np.int_]:
    if frame.empty:
        raise ValueError(f"{kind} frame must be non-empty")
    years = pd.to_numeric(frame[year], errors="raise").to_numpy(dtype=float)
    _finite(years, f"{kind} years")
    if not np.equal(years, np.floor(years)).all():
        raise ValueError(f"{kind} years must be integers")
    if frame.duplicated([glacier_id, year]).any():
        raise ValueError(f"{kind} glacier-year rows must be unique")
    return years.astype(int)


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
    years = _validate_row_identity(frame, glacier_id, year, "Seasonal")
    sequence = np.stack(
        [frame[columns].to_numpy(dtype=float) for columns in monthly_columns],
        axis=1,
    )
    static = frame[static_features].to_numpy(dtype=float)
    targets = frame[target].to_numpy(dtype=float)
    _finite(sequence, "Seasonal sequence")
    _finite(static, "Seasonal static features")
    _finite(targets, "Seasonal target")
    return SequenceData(
        sequence=sequence,
        static=static,
        target=targets,
        glacier_ids=frame[glacier_id].to_numpy(),
        years=years,
        context_glacier_ids=np.empty((len(frame), 0), dtype=object),
        context_years=np.empty((len(frame), 0), dtype=int),
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
    _validate_row_identity(frame, glacier_id, year, "Temporal")
    _finite(feature_values, "Temporal features")
    _finite(target_values, "Temporal target")

    sequences: list[NDArray[np.float64]] = []
    targets: list[float] = []
    glaciers: list[object] = []
    years: list[int] = []
    context_glaciers: list[NDArray] = []
    context_years: list[NDArray[np.int_]] = []
    ordered_frame = frame.sort_values([glacier_id, year], kind="stable")
    for glacier, group in ordered_frame.groupby(glacier_id, sort=False):
        ordered = group.reset_index(drop=True)
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
            context_glaciers.append(np.full(int(lookback), glacier, dtype=object))
            context_years.append(ordered.iloc[start:end][year].to_numpy(dtype=int))
    if not sequences:
        raise ValueError("No valid temporal sequences were produced")
    sequence = np.stack(sequences)
    return SequenceData(
        sequence=sequence,
        static=np.empty((len(sequence), 0), dtype=float),
        target=np.asarray(targets, dtype=float),
        glacier_ids=np.asarray(glaciers),
        years=np.asarray(years, dtype=int),
        context_glacier_ids=np.stack(context_glaciers),
        context_years=np.stack(context_years),
    )
