import numpy as np
import pytest

from ml4gm.validation import block_splits, loyo_splits, spatial_splits


def test_loyo_isolates_each_year() -> None:
    years = np.array([2000, 2001, 2000, 2001])

    splits = loyo_splits(years)

    assert [split.fold for split in splits] == ["year-2000", "year-2001"]
    for split in splits:
        assert set(years[split.train]).isdisjoint(set(years[split.test]))


def test_loyo_rejects_a_single_year() -> None:
    with pytest.raises(ValueError, match="produced an empty fold"):
        loyo_splits(np.array([2000, 2000]))


def test_spatial_splits_isolate_glaciers() -> None:
    glaciers = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])

    splits = spatial_splits(glaciers, folds=2, seed=42)

    assert [split.fold for split in splits] == ["spatial-0", "spatial-1"]
    for split in splits:
        assert set(glaciers[split.train]).isdisjoint(set(glaciers[split.test]))


def test_spatial_splits_reject_insufficient_glaciers() -> None:
    with pytest.raises(ValueError, match="at least 3 glaciers"):
        spatial_splits(np.array(["A", "A", "B", "B"]), folds=3, seed=42)


def test_block_splits_isolate_glacier_and_year_groups() -> None:
    glaciers = np.repeat(["A", "B", "C", "D"], 4)
    years = np.tile([2000, 2001, 2002, 2003], 4)

    splits = block_splits(glaciers, years, folds=2)

    assert [split.fold for split in splits] == ["block-0", "block-1"]
    for split in splits:
        assert set(glaciers[split.train]).isdisjoint(set(glaciers[split.test]))
        assert set(years[split.train]).isdisjoint(set(years[split.test]))


def test_block_splits_reject_insufficient_unique_values() -> None:
    with pytest.raises(ValueError, match="at least 3 unique values"):
        block_splits(
            np.array(["A", "A", "B", "B"]),
            np.array([2000, 2001, 2000, 2001]),
            folds=3,
        )


def test_block_splits_reject_empty_intersection_fold() -> None:
    with pytest.raises(ValueError, match="Block 0 produced an empty fold"):
        block_splits(
            np.array(["A", "A", "B", "B"]),
            np.array([2001, 2001, 2000, 2000]),
            folds=2,
        )


def test_block_splits_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="zip"):
        block_splits(
            np.array(["A", "B", "C"]),
            np.array([2000, 2001]),
            folds=2,
        )
