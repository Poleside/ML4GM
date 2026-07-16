from __future__ import annotations

import pandas as pd

from ml4gm.config import RunConfig
from ml4gm.data.preprocessing import prepare_fold
from ml4gm.data.schema import validate_annual_table
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
