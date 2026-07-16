import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from ml4gm.validation.loyo import Split


def spatial_splits(glacier_ids: NDArray, folds: int = 5, seed: int = 42) -> list[Split]:
    del seed
    groups = np.asarray(glacier_ids)
    unique = np.unique(groups)
    if len(unique) < folds:
        raise ValueError(f"Spatial validation needs at least {folds} glaciers")
    splitter = GroupKFold(n_splits=folds)
    placeholder = np.zeros(len(groups))
    return [
        Split(
            f"spatial-{index}",
            train.astype(int),
            test.astype(int),
            strategy="spatial",
        )
        for index, (train, test) in enumerate(splitter.split(placeholder, groups=groups))
    ]
