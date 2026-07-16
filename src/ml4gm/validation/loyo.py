from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Split:
    fold: str
    train: NDArray[np.int_]
    test: NDArray[np.int_]


def loyo_splits(years: NDArray) -> list[Split]:
    values = np.asarray(years)
    splits: list[Split] = []
    for held_out in sorted(np.unique(values)):
        test = np.flatnonzero(values == held_out)
        train = np.flatnonzero(values != held_out)
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"Year {held_out} produced an empty fold")
        splits.append(Split(f"year-{held_out}", train, test))
    return splits
