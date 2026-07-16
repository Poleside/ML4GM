import hashlib
import json
from pathlib import Path

import pandas as pd

from ml4gm.data import prepare_dataset


def test_prepare_sample_writes_sorted_data_and_manifest(tmp_path: Path) -> None:
    data_path, manifest_path = prepare_dataset(
        Path("data/sample/glacier_sample.csv"),
        tmp_path / "prepared.csv",
        tmp_path / "manifest.json",
        ["synthetic-ml4gm"],
    )

    prepared = pd.read_csv(data_path)
    assert prepared[["rgiid", "year"]].equals(
        prepared[["rgiid", "year"]].sort_values(["rgiid", "year"]).reset_index(drop=True)
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rows"] == 72
    assert manifest["sources"] == ["synthetic-ml4gm"]
    assert manifest["input_path"] == str(data_path)
    assert manifest["sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
