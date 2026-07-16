import numpy as np
import pandas as pd
import pytest

from ml4gm.data.sequences import build_seasonal_sequences, build_temporal_sequences


def test_seasonal_builder_orders_months_and_preserves_metadata() -> None:
    row = {"rgiid": "A", "year": 2000, "dhdt": -0.2, "Area": 5.0}
    for month in range(1, 13):
        row[f"{month}_t2m"] = float(month)
        row[f"{month}_tp"] = float(month * 10)

    result = build_seasonal_sequences(pd.DataFrame([row]), ["t2m", "tp"], ["Area"])

    assert result.sequence.shape == (1, 12, 2)
    assert result.sequence[0, 0].tolist() == [1.0, 10.0]
    assert result.static.tolist() == [[5.0]]
    assert result.target.tolist() == [-0.2]
    assert result.glacier_ids.tolist() == ["A"]
    assert result.years.tolist() == [2000]


@pytest.mark.parametrize(
    ("monthly_variables", "static_features", "message"),
    [
        (["t2m", "dhdt"], ["Area"], "target"),
        (["t2m"], ["Area", "dhdt"], "target"),
        (["t2m"], ["Area", "year"], "year"),
        (["t2m"], ["Area", "rgiid"], "glacier"),
    ],
)
def test_seasonal_builder_rejects_leaking_features(
    monthly_variables: list[str], static_features: list[str], message: str
) -> None:
    row = {"rgiid": "A", "year": 2000, "dhdt": -0.2, "Area": 5.0}
    for month in range(1, 13):
        row[f"{month}_t2m"] = float(month)
        row[f"{month}_dhdt"] = -0.2

    with pytest.raises(ValueError, match=message):
        build_seasonal_sequences(pd.DataFrame([row]), monthly_variables, static_features)


def test_seasonal_builder_validates_columns_shapes_and_finite_values() -> None:
    row = {"rgiid": "A", "year": 2000, "dhdt": -0.2, "Area": 5.0}
    for month in range(1, 13):
        row[f"{month}_t2m"] = float(month)
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="Missing seasonal columns"):
        build_seasonal_sequences(frame.drop(columns=["12_t2m"]), ["t2m"], ["Area"])
    with pytest.raises(ValueError, match="monthly_variables"):
        build_seasonal_sequences(frame, [], ["Area"])
    with pytest.raises(ValueError, match="finite"):
        build_seasonal_sequences(frame.assign(**{"1_t2m": np.nan}), ["t2m"], ["Area"])


@pytest.mark.parametrize(
    ("frame_transform", "message"),
    [
        (lambda frame: frame.iloc[0:0], "non-empty"),
        (lambda frame: frame.assign(year=2000.5), "integers"),
        (
            lambda frame: pd.concat([frame, frame], ignore_index=True),
            "glacier-year.*unique",
        ),
    ],
)
def test_seasonal_builder_validates_row_identity(frame_transform: object, message: str) -> None:
    row = {"rgiid": "A", "year": 2000, "dhdt": -0.2, "Area": 5.0}
    for month in range(1, 13):
        row[f"{month}_t2m"] = float(month)
    frame = frame_transform(pd.DataFrame([row]))

    with pytest.raises(ValueError, match=message):
        build_seasonal_sequences(frame, ["t2m"], ["Area"])


def test_temporal_builder_never_crosses_glaciers() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A"] * 4 + ["B"] * 4,
            "year": [2000, 2001, 2002, 2003] * 2,
            "dhdt": range(8),
            "x": range(8),
        }
    )

    result = build_temporal_sequences(frame, ["x"], 2, "rgiid", "year", "dhdt")

    assert result.sequence.shape == (4, 2, 1)
    assert result.glacier_ids.tolist() == ["A", "A", "B", "B"]
    assert result.years.tolist() == [2002, 2003, 2002, 2003]
    assert result.context_glacier_ids.tolist() == [
        ["A", "A"],
        ["A", "A"],
        ["B", "B"],
        ["B", "B"],
    ]
    assert result.context_years.tolist() == [
        [2000, 2001],
        [2001, 2002],
        [2000, 2001],
        [2001, 2002],
    ]


def test_temporal_builder_skips_windows_across_year_gaps() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A"] * 5,
            "year": [2000, 2001, 2003, 2004, 2005],
            "dhdt": range(5),
            "x": range(5),
        }
    )

    result = build_temporal_sequences(frame, ["x"], 2, "rgiid", "year", "dhdt")

    assert result.sequence.tolist() == [[[2.0], [3.0]]]
    assert result.years.tolist() == [2005]


@pytest.mark.parametrize("feature", ["dhdt", "rgiid", "year"])
def test_temporal_builder_rejects_target_and_identifier_leakage(feature: str) -> None:
    frame = pd.DataFrame(
        {"rgiid": ["A"] * 3, "year": [2000, 2001, 2002], "dhdt": range(3), "x": range(3)}
    )

    with pytest.raises(ValueError, match="must not include"):
        build_temporal_sequences(frame, [feature], 2, "rgiid", "year", "dhdt")


def test_temporal_builder_validates_lookback_columns_and_finite_values() -> None:
    frame = pd.DataFrame(
        {"rgiid": ["A"] * 3, "year": [2000, 2001, 2002], "dhdt": range(3), "x": range(3)}
    )

    with pytest.raises(ValueError, match="positive integer"):
        build_temporal_sequences(frame, ["x"], True, "rgiid", "year", "dhdt")
    with pytest.raises(ValueError, match="Missing temporal columns"):
        build_temporal_sequences(frame, ["missing"], 2, "rgiid", "year", "dhdt")
    with pytest.raises(ValueError, match="finite"):
        build_temporal_sequences(
            frame.assign(x=[0.0, np.inf, 2.0]), ["x"], 2, "rgiid", "year", "dhdt"
        )
