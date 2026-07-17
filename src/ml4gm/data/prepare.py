from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml4gm.data.manifest import DatasetManifest
from ml4gm.data.schema import validate_annual_table


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    sources: list[str],
    target: str = "dhdt",
    glacier_id: str = "rgiid",
    year: str = "year",
) -> tuple[Path, Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_path}")

    frame = validate_annual_table(
        pd.read_csv(input_path),
        target=target,
        glacier_id=glacier_id,
        year=year,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    manifest = DatasetManifest.from_frame(
        frame,
        output_path,
        sources,
        glacier_id,
        year,
        target,
    )
    manifest.write(manifest_path)
    return output_path, manifest_path
