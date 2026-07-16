import pandas as pd
import pytest

from ml4gm.data.schema import SchemaError, validate_annual_table


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rgiid": ["A", "A", "B", "B"],
            "year": [2000, 2001, 2000, 2001],
            "dhdt": [-0.2, -0.1, -0.3, -0.25],
            "Area": [4.0, 4.0, 8.0, 8.0],
        }
    )


def test_validate_annual_table_returns_sorted_copy() -> None:
    frame = valid_frame().iloc[::-1]

    result = validate_annual_table(frame, "dhdt", "rgiid", "year")

    assert list(result.columns[:3]) == ["rgiid", "year", "dhdt"]
    assert result.equals(result.sort_values(["rgiid", "year"]).reset_index(drop=True))
    assert frame.index.tolist() == [3, 2, 1, 0]


def test_reject_duplicate_glacier_year() -> None:
    frame = pd.concat([valid_frame(), valid_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(SchemaError, match="Duplicate glacier-year"):
        validate_annual_table(frame, "dhdt", "rgiid", "year")


def test_reject_missing_required_columns() -> None:
    with pytest.raises(SchemaError, match="Missing required columns: dhdt"):
        validate_annual_table(valid_frame().drop(columns="dhdt"))


def test_reject_table_without_features() -> None:
    frame = valid_frame().drop(columns="Area")

    with pytest.raises(SchemaError, match="At least one feature column"):
        validate_annual_table(frame)


def test_reject_non_finite_numeric_values() -> None:
    frame = valid_frame()
    frame.loc[0, "Area"] = float("inf")

    with pytest.raises(SchemaError, match="finite values"):
        validate_annual_table(frame)
