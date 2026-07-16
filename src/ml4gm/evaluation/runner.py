from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml4gm.config import RunConfig
from ml4gm.data.preprocessing import prepare_fold
from ml4gm.data.schema import validate_annual_table
from ml4gm.data.sequences import (
    SequenceData,
    build_seasonal_sequences,
    build_temporal_sequences,
)
from ml4gm.evaluation.metrics import regression_metrics
from ml4gm.evaluation.results import FoldResult, RunResult
from ml4gm.models import create_model
from ml4gm.validation import Split, block_splits, loyo_splits, spatial_splits


def _splits(config: RunConfig, frame: pd.DataFrame) -> list[Split]:
    if config.validation.strategy == "loyo":
        return loyo_splits(frame[config.data.year].to_numpy())
    if config.validation.strategy == "spatial":
        return spatial_splits(
            frame[config.data.glacier_id].to_numpy(),
            config.validation.folds,
            config.random_seed,
        )
    return block_splits(
        frame[config.data.glacier_id].to_numpy(),
        frame[config.data.year].to_numpy(),
        config.validation.folds,
    )


def _sequence_splits(
    config: RunConfig,
    glacier_ids: np.ndarray,
    years: np.ndarray,
    source_years: np.ndarray | None = None,
) -> list[Split]:
    if config.validation.strategy == "loyo":
        return loyo_splits(years)
    if config.validation.strategy == "spatial":
        return spatial_splits(glacier_ids, config.validation.folds, config.random_seed)
    return block_splits(
        glacier_ids,
        years,
        config.validation.folds,
        year_universe=source_years,
    )


def _scale_sequence(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_shape = train.shape
    test_shape = test.shape
    train_scaled = scaler.fit_transform(train.reshape(-1, train_shape[-1])).reshape(train_shape)
    test_scaled = scaler.transform(test.reshape(-1, test_shape[-1])).reshape(test_shape)
    return train_scaled, test_scaled


def _filter_temporal_train_indices(data: SequenceData, split: Split) -> np.ndarray:
    forbidden_years = set(split.held_out_years)
    retained = [
        index
        for index in split.train
        if forbidden_years.isdisjoint(data.context_years[index].tolist())
    ]
    if not retained:
        raise ValueError(
            f"{split.fold} has no temporal training windows after removing "
            f"{split.strategy} held-out years from training context"
        )
    return np.asarray(retained, dtype=int)


def _run_seasonal(config: RunConfig, frame: pd.DataFrame) -> RunResult:
    parameters = dict(config.model.parameters)
    monthly_variables = list(parameters.pop("monthly_variables", ["t2m", "tp"]))
    static_features = list(parameters.pop("static_features", ["Area", "Zmed"]))
    data = build_seasonal_sequences(
        frame,
        monthly_variables,
        static_features,
        config.data.target,
        config.data.glacier_id,
        config.data.year,
    )
    results: list[FoldResult] = []
    for split in _sequence_splits(
        config,
        data.glacier_ids,
        data.years,
        frame[config.data.year].to_numpy(),
    ):
        sequence_train, sequence_test = _scale_sequence(
            data.sequence[split.train], data.sequence[split.test]
        )
        scaler = StandardScaler()
        static_train = scaler.fit_transform(data.static[split.train])
        static_test = scaler.transform(data.static[split.test])
        model = create_model("seasonal_lstm", parameters, config.random_seed)
        model.fit_inputs(sequence_train, static_train, data.target[split.train])
        metrics = regression_metrics(
            data.target[split.test],
            model.predict_inputs(sequence_test, static_test),
        )
        results.append(
            FoldResult(
                split.fold,
                len(split.train),
                len(split.test),
                metrics["r2"],
                metrics["rmse"],
                metrics["mae"],
            )
        )
    return RunResult(
        config.run_name,
        config.model.name,
        config.validation.strategy,
        config.random_seed,
        results,
    )


def _run_temporal(config: RunConfig, frame: pd.DataFrame) -> RunResult:
    parameters = dict(config.model.parameters)
    default_features = [
        column
        for column in frame.columns
        if column not in {config.data.target, config.data.glacier_id, config.data.year}
    ]
    feature_columns = list(parameters.pop("feature_columns", default_features))
    lookback_value = parameters.get("lookback", 3)
    data = build_temporal_sequences(
        frame,
        feature_columns,
        lookback_value,
        config.data.glacier_id,
        config.data.year,
        config.data.target,
    )
    results: list[FoldResult] = []
    for split in _sequence_splits(
        config,
        data.glacier_ids,
        data.years,
        frame[config.data.year].to_numpy(),
    ):
        train = _filter_temporal_train_indices(data, split)
        sequence_train, sequence_test = _scale_sequence(
            data.sequence[train], data.sequence[split.test]
        )
        model = create_model("temporal_lstm", parameters, config.random_seed)
        model.fit_inputs(sequence_train, data.target[train])
        metrics = regression_metrics(data.target[split.test], model.predict_inputs(sequence_test))
        results.append(
            FoldResult(
                split.fold,
                len(train),
                len(split.test),
                metrics["r2"],
                metrics["rmse"],
                metrics["mae"],
            )
        )
    return RunResult(
        config.run_name,
        config.model.name,
        config.validation.strategy,
        config.random_seed,
        results,
    )


def run_evaluation(config: RunConfig) -> RunResult:
    if not config.data.input.exists():
        raise FileNotFoundError(
            f"Input data not found: {config.data.input}. "
            "Generate the sample or follow docs/full-data-setup.md."
        )
    frame = validate_annual_table(
        pd.read_csv(config.data.input),
        config.data.target,
        config.data.glacier_id,
        config.data.year,
    )
    if config.model.name == "seasonal_lstm":
        result = _run_seasonal(config, frame)
        result.write(config.data.output_dir / "result.json")
        return result
    if config.model.name == "temporal_lstm":
        result = _run_temporal(config, frame)
        result.write(config.data.output_dir / "result.json")
        return result
    fold_results: list[FoldResult] = []
    for split in _splits(config, frame):
        prepared = prepare_fold(
            frame,
            split.train,
            split.test,
            config.data.target,
            config.data.glacier_id,
            config.data.year,
        )
        model = create_model(config.model.name, config.model.parameters, config.random_seed)
        model.fit(prepared.X_train, prepared.y_train)
        metrics = regression_metrics(prepared.y_test, model.predict(prepared.X_test))
        fold_results.append(
            FoldResult(
                split.fold,
                len(split.train),
                len(split.test),
                metrics["r2"],
                metrics["rmse"],
                metrics["mae"],
            )
        )
    result = RunResult(
        config.run_name,
        config.model.name,
        config.validation.strategy,
        config.random_seed,
        fold_results,
    )
    result.write(config.data.output_dir / "result.json")
    return result
