from __future__ import annotations

import hashlib
import json
import platform
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml4gm import __version__
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


def _portable_path(path: Path) -> str:
    """Serialize configured paths without publishing host-specific directory trees."""
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return _portable_path(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _resolved_config(config: RunConfig) -> dict[str, Any]:
    return _json_value(asdict(config))


def _input_manifest(input_path: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()
    }
    candidates = (
        input_path.parent / "manifest.json",
        input_path.with_suffix(".manifest.json"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        identity["manifest_path"] = _portable_path(candidate)
        identity["manifest_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        try:
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            break
        if isinstance(manifest, dict) and isinstance(manifest.get("sha256"), str):
            identity["manifest_input_sha256"] = manifest["sha256"]
        break
    return identity


def _data_summary(config: RunConfig, frame: pd.DataFrame) -> dict[str, Any]:
    excluded = {config.data.target, config.data.glacier_id, config.data.year}
    features = [column for column in frame.columns if column not in excluded]
    years = sorted(int(value) for value in frame[config.data.year].unique())
    return {
        "rows": len(frame),
        "features": features,
        "feature_count": len(features),
        "glaciers": int(frame[config.data.glacier_id].nunique()),
        "year_coverage": {
            "minimum": years[0],
            "maximum": years[-1],
            "years": years,
        },
    }


def _effective_parameters(model: Any, config: RunConfig) -> dict[str, Any]:
    if hasattr(model, "parameters"):
        parameters = dict(model.parameters)
        parameters.setdefault("random_seed", config.random_seed)
        return _json_value(parameters)
    estimator = getattr(model, "estimator", None)
    if estimator is not None and hasattr(estimator, "get_params"):
        return _json_value(estimator.get_params(deep=False))
    parameters = dict(config.model.parameters)
    parameters.setdefault("random_seed", config.random_seed)
    return _json_value(parameters)


def _record(
    config: RunConfig,
    *,
    status: str,
    frame: pd.DataFrame | None = None,
    folds: list[FoldResult] | None = None,
    model_parameters: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> RunResult:
    input_identity: dict[str, Any] = {}
    if config.data.input.is_file():
        input_identity = _input_manifest(config.data.input)
    return RunResult(
        run_name=config.run_name,
        model=config.model.name,
        validation=config.validation.strategy,
        seed=config.random_seed,
        folds=folds or [],
        status=status,
        resolved_config=_resolved_config(config),
        software={"ml4gm": __version__, "python": platform.python_version()},
        input_manifest=input_identity,
        data_summary=_data_summary(config, frame) if frame is not None else {},
        model_parameters=model_parameters or _json_value(dict(config.model.parameters)),
        validation_details={
            "strategy": config.validation.strategy,
            "folds": config.validation.folds,
        },
        artifacts={"run_record": "result.json"},
        error=(
            {
                "type": type(error).__name__,
                "message": _portable_error_message(config, error),
            }
            if error is not None
            else None
        ),
    )


def _portable_error_message(config: RunConfig, error: BaseException) -> str:
    message = str(error)
    for path in (config.data.input, config.data.output_dir, Path.cwd(), Path.home()):
        if path.is_absolute():
            replacement = "~" if path == Path.home() else _portable_path(path)
            message = message.replace(str(path), replacement)
    return message


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


def _run_seasonal(
    config: RunConfig, frame: pd.DataFrame
) -> tuple[list[FoldResult], dict[str, Any]]:
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
    effective_parameters: dict[str, Any] = {}
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
        effective_parameters = _effective_parameters(model, config)
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
    return results, effective_parameters


def _run_temporal(
    config: RunConfig, frame: pd.DataFrame
) -> tuple[list[FoldResult], dict[str, Any]]:
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
    effective_parameters: dict[str, Any] = {}
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
        effective_parameters = _effective_parameters(model, config)
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
    return results, effective_parameters


def run_evaluation(config: RunConfig) -> RunResult:
    frame: pd.DataFrame | None = None
    try:
        if not config.data.input.exists():
            raise FileNotFoundError(
                f"Input data not found: {_portable_path(config.data.input)}. "
                "Generate the sample or follow docs/full-data-setup.md."
            )
        frame = validate_annual_table(
            pd.read_csv(config.data.input),
            config.data.target,
            config.data.glacier_id,
            config.data.year,
        )
        if config.model.name == "seasonal_lstm":
            fold_results, model_parameters = _run_seasonal(config, frame)
        elif config.model.name == "temporal_lstm":
            fold_results, model_parameters = _run_temporal(config, frame)
        else:
            fold_results = []
            model_parameters = {}
            for split in _splits(config, frame):
                prepared = prepare_fold(
                    frame,
                    split.train,
                    split.test,
                    config.data.target,
                    config.data.glacier_id,
                    config.data.year,
                )
                model = create_model(
                    config.model.name, config.model.parameters, config.random_seed
                )
                model_parameters = _effective_parameters(model, config)
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
        result = _record(
            config,
            status="completed",
            frame=frame,
            folds=fold_results,
            model_parameters=model_parameters,
        )
        result.write(config.data.output_dir / "result.json")
        return result
    except (OSError, RuntimeError, ValueError) as exc:
        failed = _record(
            config,
            status="failed",
            frame=frame,
            error=exc,
        )
        with suppress(OSError):
            failed.write(config.data.output_dir / "result.json")
        raise
