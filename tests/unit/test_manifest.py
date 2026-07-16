import json
from pathlib import Path

import pandas as pd

from ml4gm.data.manifest import DatasetManifest


def test_manifest_records_dataset_identity(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    frame = pd.DataFrame(
        {"rgiid": ["A", "A"], "year": [2000, 2001], "dhdt": [-0.2, -0.1], "x": [1, 2]}
    )
    frame.to_csv(csv, index=False)

    manifest = DatasetManifest.from_frame(
        frame,
        csv,
        ["synthetic"],
        "rgiid",
        "year",
        "dhdt",
    )

    assert manifest.rows == 2
    assert manifest.glaciers == 1
    assert manifest.years == [2000, 2001]
    assert manifest.features == ["x"]
    assert manifest.sources == ["synthetic"]
    assert len(manifest.sha256) == 64


def test_manifest_write_serializes_json(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    frame = pd.DataFrame({"rgiid": ["A"], "year": [2000], "dhdt": [-0.2], "x": [None]})
    frame.to_csv(csv, index=False)
    manifest = DatasetManifest.from_frame(
        frame,
        csv,
        ["synthetic"],
        "rgiid",
        "year",
        "dhdt",
    )
    output = tmp_path / "metadata" / "manifest.json"

    result = manifest.write(output)

    assert result == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["features"] == ["x"]
    assert payload["missing_values"]["x"] == 1
