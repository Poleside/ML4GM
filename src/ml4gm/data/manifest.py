from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetManifest:
    input_path: str
    sha256: str
    sources: list[str]
    rows: int
    glaciers: int
    years: list[int]
    features: list[str]
    missing_values: dict[str, int]
    created_at: str

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        input_path: Path,
        sources: list[str],
        glacier_id: str,
        year: str,
        target: str,
    ) -> DatasetManifest:
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
        excluded = {glacier_id, year, target}
        return cls(
            input_path=str(input_path),
            sha256=digest,
            sources=sources,
            rows=len(frame),
            glaciers=int(frame[glacier_id].nunique()),
            years=sorted(int(value) for value in frame[year].unique()),
            features=[column for column in frame.columns if column not in excluded],
            missing_values={key: int(value) for key, value in frame.isna().sum().items()},
            created_at=datetime.now(UTC).isoformat(),
        )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path
