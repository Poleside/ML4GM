import numpy as np
from numpy.typing import NDArray

from ml4gm.validation.loyo import Split


def _group_map(values: NDArray, folds: int) -> dict[object, int]:
    unique = sorted(np.unique(values))
    if len(unique) < folds:
        raise ValueError(f"Block validation needs at least {folds} unique values")
    return {
        value: min(index * folds // len(unique), folds - 1) for index, value in enumerate(unique)
    }


def block_splits(glacier_ids: NDArray, years: NDArray, folds: int = 5) -> list[Split]:
    glaciers = np.asarray(glacier_ids)
    year_values = np.asarray(years)
    glacier_groups = _group_map(glaciers, folds)
    year_groups = _group_map(year_values, folds)
    result: list[Split] = []
    for fold in range(folds):
        test_mask = np.array(
            [
                glacier_groups[glacier] == fold and year_groups[year] == fold
                for glacier, year in zip(glaciers, year_values, strict=True)
            ]
        )
        train_mask = np.array(
            [
                glacier_groups[glacier] != fold and year_groups[year] != fold
                for glacier, year in zip(glaciers, year_values, strict=True)
            ]
        )
        train = np.flatnonzero(train_mask)
        test = np.flatnonzero(test_mask)
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"Block {fold} produced an empty fold")
        result.append(Split(f"block-{fold}", train, test))
    return result
