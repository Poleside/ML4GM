from pathlib import Path

import pandas as pd
import pytest

from ml4gm.config import DataConfig, ModelConfig, RunConfig, ValidationConfig
from ml4gm.data.sequences import build_temporal_sequences
from ml4gm.evaluation.runner import run_evaluation
from ml4gm.validation import spatial_splits

pytest.importorskip("torch")


def _seasonal_frame() -> pd.DataFrame:
    rows = []
    for glacier_index, glacier in enumerate(("A", "B", "C", "D")):
        for year in range(2000, 2004):
            row = {
                "rgiid": glacier,
                "year": year,
                "dhdt": float(year - 2002 + glacier_index),
                "Area": float(5 + glacier_index),
                "Zmed": float(4500 + 10 * glacier_index),
            }
            for month in range(1, 13):
                row[f"{month}_t2m"] = float(month + year - 2000)
                row[f"{month}_tp"] = float(month * 10 + glacier_index)
            rows.append(row)
    return pd.DataFrame(rows)


def test_seasonal_runner_writes_result(tmp_path: Path) -> None:
    input_path = tmp_path / "seasonal.csv"
    _seasonal_frame().to_csv(input_path, index=False)
    config = RunConfig(
        data=DataConfig(input_path, output_dir=tmp_path / "seasonal-output"),
        model=ModelConfig(
            "seasonal_lstm",
            {
                "monthly_variables": ["t2m", "tp"],
                "static_features": ["Area", "Zmed"],
                "hidden": 4,
                "epochs": 1,
                "batch_size": 4,
            },
        ),
        validation=ValidationConfig("loyo", 2),
        random_seed=42,
        run_name="seasonal-test",
    )

    result = run_evaluation(config)

    assert len(result.folds) == 4
    assert (config.data.output_dir / "result.json").exists()


def test_temporal_spatial_groups_and_runner(tmp_path: Path) -> None:
    frame = _seasonal_frame()[["rgiid", "year", "dhdt", "Area", "Zmed"]]
    sequence_data = build_temporal_sequences(frame, ["Area", "Zmed"], 2, "rgiid", "year", "dhdt")
    for split in spatial_splits(sequence_data.glacier_ids, folds=2, seed=42):
        assert set(sequence_data.glacier_ids[split.train]).isdisjoint(
            set(sequence_data.glacier_ids[split.test])
        )
    input_path = tmp_path / "temporal.csv"
    frame.to_csv(input_path, index=False)
    config = RunConfig(
        data=DataConfig(input_path, output_dir=tmp_path / "temporal-output"),
        model=ModelConfig(
            "temporal_lstm",
            {
                "feature_columns": ["Area", "Zmed"],
                "lookback": 2,
                "hidden": 4,
                "epochs": 1,
                "batch_size": 4,
            },
        ),
        validation=ValidationConfig("spatial", 2),
        random_seed=42,
        run_name="temporal-test",
    )

    result = run_evaluation(config)

    assert len(result.folds) == 2
    assert (config.data.output_dir / "result.json").exists()
