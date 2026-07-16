import json
from pathlib import Path

import pandas as pd
import pytest

from ml4gm.config import DataConfig, ModelConfig, RunConfig, ValidationConfig
from ml4gm.data.sequences import build_temporal_sequences
from ml4gm.evaluation.runner import _filter_temporal_train_indices, run_evaluation
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
    record = json.loads((config.data.output_dir / "result.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["model"] == "seasonal_lstm"
    assert record["model_parameters"]["hidden"] == 4
    assert record["model_parameters"]["monthly_variables"] == ["t2m", "tp"]
    assert record["model_parameters"]["static_features"] == ["Area", "Zmed"]
    assert record["data_summary"]["rows"] == 16


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
    record = json.loads((config.data.output_dir / "result.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["model"] == "temporal_lstm"
    assert record["model_parameters"]["lookback"] == 2
    assert record["model_parameters"]["feature_columns"] == ["Area", "Zmed"]
    assert record["data_summary"]["rows"] == 16


def test_sequence_runner_records_default_input_design(tmp_path: Path) -> None:
    frame = _seasonal_frame()
    seasonal_input = tmp_path / "seasonal-default.csv"
    frame.to_csv(seasonal_input, index=False)
    seasonal_config = RunConfig(
        data=DataConfig(seasonal_input, output_dir=tmp_path / "seasonal-default"),
        model=ModelConfig(
            "seasonal_lstm",
            {"hidden": 4, "epochs": 1, "batch_size": 4},
        ),
        validation=ValidationConfig("loyo", 2),
    )

    run_evaluation(seasonal_config)
    seasonal_record = json.loads(
        (seasonal_config.data.output_dir / "result.json").read_text(encoding="utf-8")
    )

    assert seasonal_record["model_parameters"]["monthly_variables"] == ["t2m", "tp"]
    assert seasonal_record["model_parameters"]["static_features"] == ["Area", "Zmed"]

    temporal_input = tmp_path / "temporal-default.csv"
    frame[["rgiid", "year", "dhdt", "Area", "Zmed"]].to_csv(temporal_input, index=False)
    temporal_config = RunConfig(
        data=DataConfig(temporal_input, output_dir=tmp_path / "temporal-default"),
        model=ModelConfig(
            "temporal_lstm",
            {"hidden": 4, "epochs": 1, "batch_size": 4},
        ),
        validation=ValidationConfig("spatial", 2),
    )

    run_evaluation(temporal_config)
    temporal_record = json.loads(
        (temporal_config.data.output_dir / "result.json").read_text(encoding="utf-8")
    )

    assert temporal_record["model_parameters"]["feature_columns"] == ["Area", "Zmed"]
    assert temporal_record["model_parameters"]["lookback"] == 3


def test_temporal_loyo_removes_training_windows_containing_held_out_target_rows() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A"] * 6 + ["B"] * 6,
            "year": [2000, 2001, 2002, 2003, 2004, 2005] * 2,
            "dhdt": range(12),
            "x": [0.0, 1.0, 2002.0, 3.0, 4.0, 5.0] + [10.0, 11.0, 2002.0, 13.0, 14.0, 15.0],
        }
    )
    data = build_temporal_sequences(frame, ["x"], 2, "rgiid", "year", "dhdt")
    held_out = next(
        split
        for split in __import__("ml4gm.validation", fromlist=["loyo_splits"]).loyo_splits(
            data.years
        )
        if split.fold == "year-2002"
    )

    filtered = _filter_temporal_train_indices(data, held_out)

    assert data.years[filtered].tolist() == [2005, 2005]
    assert 2002 not in data.context_years[filtered]


def test_temporal_loyo_filters_held_out_year_from_unbalanced_glacier_context() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A"] * 6 + ["B"] * 6,
            "year": list(range(2000, 2006)) + list(range(2001, 2007)),
            "dhdt": range(12),
            "x": range(12),
        }
    )
    data = build_temporal_sequences(frame, ["x"], 2, "rgiid", "year", "dhdt")
    split = next(
        split
        for split in __import__("ml4gm.validation", fromlist=["loyo_splits"]).loyo_splits(
            data.years
        )
        if split.fold == "year-2002"
    )
    assert ("B", 2002) not in set(
        zip(
            data.glacier_ids[split.test].tolist(),
            data.years[split.test].tolist(),
            strict=True,
        )
    )

    filtered = _filter_temporal_train_indices(data, split)

    assert not any(
        glacier == "B" and target_year == 2003
        for glacier, target_year in zip(
            data.glacier_ids[filtered], data.years[filtered], strict=True
        )
    )
    assert 2002 not in data.context_years[filtered]


def test_temporal_block_removes_training_windows_containing_test_target_rows() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A"] * 8 + ["B"] * 8,
            "year": list(range(2000, 2008)) * 2,
            "dhdt": range(16),
            "x": range(16),
        }
    )
    data = build_temporal_sequences(frame, ["x"], 2, "rgiid", "year", "dhdt")
    split = next(
        split
        for split in __import__("ml4gm.validation", fromlist=["block_splits"]).block_splits(
            data.glacier_ids, data.years, folds=2
        )
        if split.fold == "block-0"
    )

    filtered = _filter_temporal_train_indices(data, split)
    test_targets = set(
        zip(
            data.glacier_ids[split.test].tolist(),
            data.years[split.test].tolist(),
            strict=True,
        )
    )
    train_context = {
        pair
        for index in filtered
        for pair in zip(
            data.context_glacier_ids[index].tolist(),
            data.context_years[index].tolist(),
            strict=True,
        )
    }

    assert train_context.isdisjoint(test_targets)


def test_temporal_block_filters_entire_held_out_year_group_from_context() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A"] * 6 + ["B"] * 5 + ["C"] * 6 + ["D"] * 6,
            "year": list(range(2000, 2006))
            + [2000, 2001, 2003, 2004, 2005]
            + list(range(2000, 2006)) * 2,
            "dhdt": range(23),
            "x": range(23),
        }
    )
    data = build_temporal_sequences(frame, ["x"], 1, "rgiid", "year", "dhdt")
    splits = __import__("ml4gm.validation", fromlist=["block_splits"]).block_splits(
        data.glacier_ids,
        data.years,
        folds=2,
        year_universe=frame["year"].to_numpy(),
    )
    split = next(item for item in splits if item.fold == "block-0")

    filtered = _filter_temporal_train_indices(data, split)

    assert set(data.context_years[filtered].reshape(-1)).isdisjoint(split.held_out_years)


def test_temporal_runner_scales_only_filtered_training_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A"] * 6 + ["B"] * 6,
            "year": [2000, 2001, 2002, 2003, 2004, 2005] * 2,
            "dhdt": range(12),
            "x": [0.0, 1.0, 2002.0, 3.0, 4.0, 5.0] + [10.0, 11.0, 2002.0, 13.0, 14.0, 15.0],
        }
    )
    input_path = tmp_path / "loyo.csv"
    frame.to_csv(input_path, index=False)
    seen_train: list[list[float]] = []

    def spy_scale(train: object, test: object) -> tuple[object, object]:
        seen_train.append(train.reshape(-1).tolist())
        return train, test

    class FakeModel:
        def fit_inputs(self, sequence: object, target: object) -> None:
            return None

        def predict_inputs(self, sequence: object) -> object:
            return __import__("numpy").zeros(len(sequence))

    runner = __import__("ml4gm.evaluation.runner", fromlist=["_scale_sequence"])
    monkeypatch.setattr(runner, "_scale_sequence", spy_scale)
    monkeypatch.setattr(runner, "create_model", lambda *args: FakeModel())
    config = RunConfig(
        data=DataConfig(input_path, output_dir=tmp_path / "output"),
        model=ModelConfig(
            "temporal_lstm",
            {"feature_columns": ["x"], "lookback": 2, "epochs": 1},
        ),
        validation=ValidationConfig("loyo", 2),
        random_seed=42,
        run_name="loyo-filter",
    )

    run_evaluation(config)

    held_out_2002_train = seen_train[0]
    assert 2002.0 not in held_out_2002_train
