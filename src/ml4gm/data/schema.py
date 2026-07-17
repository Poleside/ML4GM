from __future__ import annotations

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """Raised when glacier data violate the ML4GM annual schema."""


def validate_annual_table(
    frame: pd.DataFrame,
    target: str = "dhdt",
    glacier_id: str = "rgiid",
    year: str = "year",
) -> pd.DataFrame:
    required = [glacier_id, year, target]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {', '.join(missing)}")

    result = frame.copy()
    if result[required].isna().any().any():
        raise SchemaError("Identifiers, years, and targets must not be missing")
    identifiers_are_valid = result[glacier_id].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    if not identifiers_are_valid.all():
        raise SchemaError("Glacier identifiers must be non-empty strings")

    normalized_years = pd.to_numeric(result[year], errors="raise")
    if not (normalized_years % 1 == 0).all():
        raise SchemaError("Years must be integers")
    result[year] = normalized_years.astype(int)
    if result.duplicated([glacier_id, year]).any():
        raise SchemaError("Duplicate glacier-year rows require explicit aggregation")

    result[target] = pd.to_numeric(result[target], errors="raise").astype(float)
    numeric_features = result.drop(columns=[glacier_id]).select_dtypes(include=[np.number])
    if not np.isfinite(numeric_features.to_numpy()).all():
        raise SchemaError("Numeric columns must contain only finite values")

    feature_columns = [
        column for column in result.columns if column not in {glacier_id, year, target}
    ]
    if not feature_columns:
        raise SchemaError("At least one feature column is required")

    ordered = [glacier_id, year, target, *feature_columns]
    return result[ordered].sort_values([glacier_id, year]).reset_index(drop=True)
