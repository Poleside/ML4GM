from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample(
    path: Path, glaciers: int = 12, years: int = 6, seed: int = 42
) -> Path:
    if glaciers < 2 or years < 2:
        raise ValueError("Sample data requires at least 2 glaciers and 2 years")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for glacier_index in range(glaciers):
        area = float(rng.uniform(2.0, 80.0))
        elevation = float(rng.uniform(3200.0, 6200.0))
        for offset in range(years):
            year = 2000 + offset
            temperature = float(rng.normal(-4.0 + 0.06 * offset, 1.2))
            precipitation = float(rng.uniform(200.0, 1200.0))
            dhdt = (
                -0.18
                - 0.035 * temperature
                + 0.00008 * precipitation
                + 0.00001 * (elevation - 4500.0)
                + float(rng.normal(0.0, 0.04))
            )
            rows.append(
                {
                    "rgiid": f"RGI60-13.{glacier_index:05d}",
                    "year": year,
                    "dhdt": dhdt,
                    "Area": area,
                    "Zmed": elevation,
                    "t2m": temperature,
                    "tp": precipitation,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
