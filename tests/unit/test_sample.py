from pathlib import Path

import pandas as pd

from ml4gm.data.sample import generate_sample


def test_generate_sample_is_deterministic(tmp_path: Path) -> None:
    first = generate_sample(tmp_path / "a.csv", glaciers=6, years=4, seed=42)
    second = generate_sample(tmp_path / "b.csv", glaciers=6, years=4, seed=42)
    left = pd.read_csv(first)
    right = pd.read_csv(second)
    pd.testing.assert_frame_equal(left, right)
    assert len(left) == 24
    assert left["rgiid"].nunique() == 6
