# ML4GM Open-Source Transformation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform ML4GM from a machine-specific notebook repository into an Apache-2.0 licensed, installable, tested, leakage-aware glacier machine-learning toolkit and prepare truthful, repository-backed applications for both OpenAI open-source programs.

**Architecture:** Build a Python 3.11 `src/` package with configuration-driven data preparation, validation splits, model adapters, evaluation records, and an `argparse` CLI. Preserve existing notebooks individually under a legacy directory, make Random Forest the CPU quickstart vertical slice, then migrate optional LightGBM and PyTorch models behind stable interfaces. Separate code licensing from upstream scientific data terms and make every application claim traceable to a public repository artifact.

**Tech Stack:** Python 3.11, setuptools, NumPy, pandas, scikit-learn, PyYAML, joblib, optional LightGBM, optional PyTorch, pytest, Ruff, GitHub Actions.

## Global Constraints

- Code license is Apache License 2.0.
- Python 3.11 is the first supported runtime.
- Raw upstream datasets, large processed tables, trained weights, and experiment outputs must not be committed.
- Only synthetic data or a confirmed redistributable attributed subset may live under `data/sample/`.
- No implementation may contain `C:\ML4GM`, `/Users/...`, `/openbayes/...`, or another machine-specific root.
- Preprocessing learned from data must be fitted on training folds only.
- LOYO must isolate years, spatial GroupKFold must isolate glaciers, and block validation must isolate both glacier and year groups.
- Historical notebook output is not evidence of a newly reproduced benchmark.
- OpenAI models support maintenance only and must not generate glacier predictions.
- Existing files may be moved individually; no scripted, looped, globbed, piped, or batch deletion is permitted.
- Every task ends with tests and an independently reviewable commit.

---

## Target File Map

### Package and configuration

- `pyproject.toml`: package metadata, dependencies, entry point, Ruff and pytest configuration.
- `src/ml4gm/__init__.py`: public version.
- `src/ml4gm/config.py`: typed configuration loading and validation.
- `src/ml4gm/cli.py`: `data prepare`, `train`, `evaluate`, and `benchmark` commands.
- `configs/quickstart.yaml`: synthetic CPU demonstration.
- `configs/benchmark.yaml`: full-data benchmark template.

### Data

- `src/ml4gm/data/schema.py`: common annual glacier schema validation.
- `src/ml4gm/data/sources.py`: upstream source metadata models.
- `src/ml4gm/data/manifest.py`: checksums and prepared-dataset manifests.
- `src/ml4gm/data/preprocessing.py`: deterministic feature selection and fold-local preprocessing.
- `src/ml4gm/data/sample.py`: deterministic synthetic sample generator.
- `data/sample/glacier_sample.csv`: generated demonstration data.

### Validation and evaluation

- `src/ml4gm/validation/loyo.py`: year-isolated folds.
- `src/ml4gm/validation/spatial.py`: glacier-isolated folds.
- `src/ml4gm/validation/block.py`: glacier-and-year-isolated folds.
- `src/ml4gm/evaluation/metrics.py`: R2, RMSE, and MAE.
- `src/ml4gm/evaluation/results.py`: serializable run and fold records.
- `src/ml4gm/evaluation/runner.py`: training/evaluation orchestration.

### Models

- `src/ml4gm/models/base.py`: model protocol and registry.
- `src/ml4gm/models/random_forest.py`: required CPU model.
- `src/ml4gm/models/lightgbm.py`: optional LightGBM adapter.
- `src/ml4gm/models/mlp.py`: optional PyTorch tabular MLP.
- `src/ml4gm/models/seasonal_lstm.py`: optional monthly-sequence LSTM.
- `src/ml4gm/models/temporal_lstm.py`: optional multi-year LSTM.
- `src/ml4gm/models/torch_utils.py`: deterministic device, dataset, and training helpers.

### Tests

- `tests/unit/test_config.py`
- `tests/unit/test_schema.py`
- `tests/unit/test_manifest.py`
- `tests/unit/test_preprocessing.py`
- `tests/unit/test_validation.py`
- `tests/unit/test_metrics.py`
- `tests/unit/test_results.py`
- `tests/unit/test_models.py`
- `tests/unit/test_neural_models.py`
- `tests/integration/test_quickstart.py`
- `tests/integration/test_cli.py`
- `tests/repository/test_notebooks.py`
- `tests/repository/test_forbidden_paths.py`

### Governance, documentation, and applications

- `README.md`
- `LICENSE`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `MAINTAINERS.md`
- `CITATION.cff`
- `DATA_SOURCES.md`
- `ROADMAP.md`
- `CHANGELOG.md`
- `docs/data-schema.md`
- `docs/full-data-setup.md`
- `docs/benchmark-protocol.md`
- `docs/scientific-limitations.md`
- `docs/applications/codex-for-oss.md`
- `docs/applications/codex-open-source-fund.md`
- `.github/workflows/ci.yml`
- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/pull_request_template.md`

---

### Task 1: Establish Package Metadata, License, and Ignore Policy

**Files:**

- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `README.md`
- Create: `src/ml4gm/__init__.py`
- Modify: `.gitignore`
- Test: `tests/unit/test_package.py`

**Interfaces:**

- Produces: importable `ml4gm` package with `__version__ == "0.1.0"`.
- Produces: console entry point `ml4gm = ml4gm.cli:main`, implemented in Task 8.
- Produces: optional dependency groups `lightgbm`, `torch`, `notebooks`, and `dev`.

- [ ] **Step 1: Write the package metadata test**

```python
# tests/unit/test_package.py
from importlib.metadata import version

import ml4gm


def test_package_version_matches_metadata() -> None:
    assert ml4gm.__version__ == "0.1.0"
    assert version("ml4gm") == ml4gm.__version__
```

- [ ] **Step 2: Run the test and verify the package does not exist**

Run:

```bash
python3.11 -m pytest tests/unit/test_package.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ml4gm'`.

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ml4gm"
version = "0.1.0"
description = "Reproducible machine-learning benchmarks for glacier elevation change"
readme = "README.md"
requires-python = ">=3.11"
license = {file = "LICENSE"}
authors = [
  {name = "Poleside"},
  {name = "HectorGao"},
]
keywords = ["glacier", "climate", "machine-learning", "reproducibility"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "License :: OSI Approved :: Apache Software License",
  "Programming Language :: Python :: 3.11",
  "Topic :: Scientific/Engineering :: Atmospheric Science",
]
dependencies = [
  "joblib>=1.4,<2",
  "numpy>=1.26,<3",
  "pandas>=2.2,<3",
  "PyYAML>=6.0,<7",
  "scikit-learn>=1.5,<2",
]

[project.optional-dependencies]
lightgbm = ["lightgbm>=4.5,<5"]
torch = ["torch>=2.4,<3"]
notebooks = [
  "dask>=2024.7",
  "geopandas>=1.0",
  "jupyterlab>=4.2",
  "matplotlib>=3.9",
  "netCDF4>=1.7",
  "rasterio>=1.4",
  "scipy>=1.14",
  "seaborn>=0.13",
  "xarray>=2024.7",
]
dev = [
  "build>=1.2,<2",
  "pytest>=8.3,<9",
  "ruff>=0.6,<1",
]

[project.scripts]
ml4gm = "ml4gm.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- [ ] **Step 4: Create the package version**

```python
# src/ml4gm/__init__.py
"""ML4GM: reproducible glacier machine-learning benchmarks."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Create the minimum package README**

```markdown
# ML4GM

ML4GM is an early-stage toolkit for reproducible machine-learning benchmarks
of glacier elevation change. The supported CPU quickstart and full scientific
documentation are added in later tasks of this implementation plan.

The project is licensed under Apache License 2.0. Third-party scientific data
retain their own licenses and terms.
```

- [ ] **Step 6: Add the canonical Apache License 2.0**

Create `LICENSE` with the unmodified text published at:

```text
https://www.apache.org/licenses/LICENSE-2.0.txt
```

Verify the first line is exactly:

```text
Apache License
```

and the version line is exactly:

```text
Version 2.0, January 2004
```

- [ ] **Step 7: Replace the three-line ignore file with explicit project rules**

```gitignore
.DS_Store
.coverage
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/
.venv/
.worktrees/
.superpowers/
__pycache__/
*.egg-info/
build/
dist/
graphify-out/
models/
outputs/
data/raw/
data/processed/
data/**/*.nc
data/**/*.pkl
data/**/*.pt
data/**/*.joblib
!data/sample/
!data/sample/*.csv
```

- [ ] **Step 8: Install editable development dependencies and run the test**

Run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/unit/test_package.py -v
```

Expected: `1 passed`.

- [ ] **Step 9: Verify package build**

Run:

```bash
.venv/bin/python -m build
```

Expected: one wheel and one source distribution under `dist/`.

- [ ] **Step 10: Commit**

```bash
git add .gitignore LICENSE README.md pyproject.toml src/ml4gm/__init__.py tests/unit/test_package.py
git commit -m "build: package ML4GM as an Apache-2.0 project"
```

---

### Task 2: Add Typed Configuration and Deterministic Sample Data

**Files:**

- Create: `src/ml4gm/config.py`
- Create: `src/ml4gm/data/__init__.py`
- Create: `src/ml4gm/data/sample.py`
- Create: `configs/quickstart.yaml`
- Create: `configs/benchmark.yaml`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_sample.py`

**Interfaces:**

- Produces: `RunConfig.from_yaml(path: Path) -> RunConfig`.
- Produces: `generate_sample(path: Path, glaciers: int, years: int, seed: int) -> Path`.
- Config model names: `random_forest`, `lightgbm`, `mlp`, `seasonal_lstm`, `temporal_lstm`.
- Validation names: `loyo`, `spatial`, `block`.

- [ ] **Step 1: Write failing configuration tests**

```python
# tests/unit/test_config.py
from pathlib import Path

import pytest

from ml4gm.config import ConfigError, RunConfig


def test_load_quickstart_config() -> None:
    config = RunConfig.from_yaml(Path("configs/quickstart.yaml"))
    assert config.model.name == "random_forest"
    assert config.validation.strategy == "loyo"
    assert config.random_seed == 42


def test_reject_unknown_model(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "data:\\n  input: data.csv\\nmodel:\\n  name: magic\\n"
        "validation:\\n  strategy: loyo\\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Unsupported model"):
        RunConfig.from_yaml(path)
```

- [ ] **Step 2: Write the failing sample-data test**

```python
# tests/unit/test_sample.py
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
```

- [ ] **Step 3: Run tests and verify missing modules**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_config.py tests/unit/test_sample.py -v
```

Expected: collection fails because `ml4gm.config` and `ml4gm.data.sample` do not exist.

- [ ] **Step 4: Implement configuration dataclasses**

```python
# src/ml4gm/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_MODELS = {
    "random_forest",
    "lightgbm",
    "mlp",
    "seasonal_lstm",
    "temporal_lstm",
}
SUPPORTED_VALIDATION = {"loyo", "spatial", "block"}


class ConfigError(ValueError):
    """Raised when an ML4GM configuration is invalid."""


@dataclass(frozen=True)
class DataConfig:
    input: Path
    target: str = "dhdt"
    glacier_id: str = "rgiid"
    year: str = "year"
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationConfig:
    strategy: str
    folds: int = 5


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig
    model: ModelConfig
    validation: ValidationConfig
    random_seed: int = 42
    run_name: str = "ml4gm-run"

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            model_name = str(raw["model"]["name"])
            strategy = str(raw["validation"]["strategy"])
            input_path = Path(raw["data"]["input"])
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"Missing required configuration field: {exc}") from exc
        if model_name not in SUPPORTED_MODELS:
            raise ConfigError(f"Unsupported model: {model_name}")
        if strategy not in SUPPORTED_VALIDATION:
            raise ConfigError(f"Unsupported validation strategy: {strategy}")
        data_raw = raw["data"]
        return cls(
            data=DataConfig(
                input=input_path,
                target=str(data_raw.get("target", "dhdt")),
                glacier_id=str(data_raw.get("glacier_id", "rgiid")),
                year=str(data_raw.get("year", "year")),
                output_dir=Path(data_raw.get("output_dir", "outputs")),
            ),
            model=ModelConfig(model_name, dict(raw["model"].get("parameters", {}))),
            validation=ValidationConfig(
                strategy,
                int(raw["validation"].get("folds", 5)),
            ),
            random_seed=int(raw.get("random_seed", 42)),
            run_name=str(raw.get("run_name", "ml4gm-run")),
        )
```

- [ ] **Step 5: Implement deterministic synthetic data**

```python
# src/ml4gm/data/sample.py
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample(path: Path, glaciers: int = 12, years: int = 6, seed: int = 42) -> Path:
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
```

- [ ] **Step 6: Create configurations**

```yaml
# configs/quickstart.yaml
run_name: quickstart-rf-loyo
random_seed: 42
data:
  input: data/sample/glacier_sample.csv
  target: dhdt
  glacier_id: rgiid
  year: year
  output_dir: outputs/quickstart
model:
  name: random_forest
  parameters:
    n_estimators: 40
    max_depth: 6
    n_jobs: 1
validation:
  strategy: loyo
  folds: 5
```

```yaml
# configs/benchmark.yaml
run_name: full-ml4gm-benchmark
random_seed: 42
data:
  input: data/processed/merged_data.csv
  target: dhdt
  glacier_id: rgiid
  year: year
  output_dir: outputs/benchmark
model:
  name: random_forest
  parameters:
    n_estimators: 500
    max_depth: 16
    n_jobs: -1
validation:
  strategy: block
  folds: 5
```

- [ ] **Step 7: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_config.py tests/unit/test_sample.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Generate the committed sample**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; from ml4gm.data.sample import generate_sample; generate_sample(Path('data/sample/glacier_sample.csv'))"
```

Expected: `data/sample/glacier_sample.csv` contains 72 rows.

- [ ] **Step 9: Commit**

```bash
git add configs data/sample src/ml4gm/config.py src/ml4gm/data tests/unit/test_config.py tests/unit/test_sample.py
git commit -m "feat: add configuration and synthetic glacier sample"
```

---

### Task 3: Implement Schema Validation and Dataset Manifests

**Files:**

- Create: `src/ml4gm/data/schema.py`
- Create: `src/ml4gm/data/manifest.py`
- Create: `tests/unit/test_schema.py`
- Create: `tests/unit/test_manifest.py`

**Interfaces:**

- Produces: `validate_annual_table(frame, target, glacier_id, year) -> DataFrame`.
- Produces: `DatasetManifest.from_frame(...) -> DatasetManifest`.
- Produces: `DatasetManifest.write(path: Path) -> Path`.

- [ ] **Step 1: Write failing schema tests**

```python
# tests/unit/test_schema.py
import pandas as pd
import pytest

from ml4gm.data.schema import SchemaError, validate_annual_table


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rgiid": ["A", "A", "B", "B"],
            "year": [2000, 2001, 2000, 2001],
            "dhdt": [-0.2, -0.1, -0.3, -0.25],
            "Area": [4.0, 4.0, 8.0, 8.0],
        }
    )


def test_validate_annual_table_returns_sorted_copy() -> None:
    result = validate_annual_table(valid_frame(), "dhdt", "rgiid", "year")
    assert list(result.columns[:3]) == ["rgiid", "year", "dhdt"]
    assert result.equals(result.sort_values(["rgiid", "year"]).reset_index(drop=True))


def test_reject_duplicate_glacier_year() -> None:
    frame = pd.concat([valid_frame(), valid_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="Duplicate glacier-year"):
        validate_annual_table(frame, "dhdt", "rgiid", "year")
```

- [ ] **Step 2: Write failing manifest test**

```python
# tests/unit/test_manifest.py
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
    assert len(manifest.sha256) == 64
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_schema.py tests/unit/test_manifest.py -v
```

Expected: missing-module failures.

- [ ] **Step 4: Implement schema validation**

```python
# src/ml4gm/data/schema.py
from __future__ import annotations

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """Raised when glacier data violate the ML4GM annual schema."""


def validate_annual_table(
    frame: pd.DataFrame,
    target: str = "dhdt",
    glacier_id: str = "rgiid",
    year: str = "year",
) -> pd.DataFrame:
    required = [glacier_id, year, target]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {', '.join(missing)}")
    result = frame.copy()
    if result[required].isna().any().any():
        raise SchemaError("Identifiers, years, and targets must not be missing")
    if result.duplicated([glacier_id, year]).any():
        raise SchemaError("Duplicate glacier-year rows require explicit aggregation")
    result[year] = pd.to_numeric(result[year], errors="raise").astype(int)
    result[target] = pd.to_numeric(result[target], errors="raise").astype(float)
    numeric_features = result.drop(columns=[glacier_id]).select_dtypes(include=[np.number])
    if not np.isfinite(numeric_features.to_numpy()).all():
        raise SchemaError("Numeric columns must contain only finite values")
    feature_columns = [
        column for column in result.columns if column not in {glacier_id, year, target}
    ]
    if not feature_columns:
        raise SchemaError("At least one feature column is required")
    ordered = [glacier_id, year, target, *feature_columns]
    return result[ordered].sort_values([glacier_id, year]).reset_index(drop=True)
```

- [ ] **Step 5: Implement manifest serialization**

```python
# src/ml4gm/data/manifest.py
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
    ) -> "DatasetManifest":
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
```

- [ ] **Step 6: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_schema.py tests/unit/test_manifest.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/ml4gm/data/schema.py src/ml4gm/data/manifest.py tests/unit/test_schema.py tests/unit/test_manifest.py
git commit -m "feat: validate and identify glacier datasets"
```

---

### Task 4: Implement Leakage-Aware Validation Splits

**Files:**

- Create: `src/ml4gm/validation/__init__.py`
- Create: `src/ml4gm/validation/loyo.py`
- Create: `src/ml4gm/validation/spatial.py`
- Create: `src/ml4gm/validation/block.py`
- Create: `tests/unit/test_validation.py`

**Interfaces:**

- Produces: `Split(fold: str, train: ndarray, test: ndarray)`.
- Produces: `loyo_splits(years) -> list[Split]`.
- Produces: `spatial_splits(glacier_ids, folds, seed) -> list[Split]`.
- Produces: `block_splits(glacier_ids, years, folds) -> list[Split]`.

- [ ] **Step 1: Write failing invariant tests**

```python
# tests/unit/test_validation.py
import numpy as np

from ml4gm.validation import block_splits, loyo_splits, spatial_splits


def test_loyo_isolates_each_year() -> None:
    years = np.array([2000, 2001, 2000, 2001])
    for split in loyo_splits(years):
        assert set(years[split.train]).isdisjoint(set(years[split.test]))


def test_spatial_splits_isolate_glaciers() -> None:
    glaciers = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    for split in spatial_splits(glaciers, folds=2, seed=42):
        assert set(glaciers[split.train]).isdisjoint(set(glaciers[split.test]))


def test_block_splits_isolate_glacier_and_year_groups() -> None:
    glaciers = np.repeat(["A", "B", "C", "D"], 4)
    years = np.tile([2000, 2001, 2002, 2003], 4)
    for split in block_splits(glaciers, years, folds=2):
        assert set(glaciers[split.train]).isdisjoint(set(glaciers[split.test]))
        assert set(years[split.train]).isdisjoint(set(years[split.test]))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_validation.py -v
```

Expected: missing-module failure.

- [ ] **Step 3: Implement the common split record and LOYO**

```python
# src/ml4gm/validation/loyo.py
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
```

- [ ] **Step 4: Implement spatial splits**

```python
# src/ml4gm/validation/spatial.py
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
        Split(f"spatial-{index}", train.astype(int), test.astype(int))
        for index, (train, test) in enumerate(splitter.split(placeholder, groups=groups))
    ]
```

- [ ] **Step 5: Implement strict block splits**

```python
# src/ml4gm/validation/block.py
import numpy as np
from numpy.typing import NDArray

from ml4gm.validation.loyo import Split


def _group_map(values: NDArray, folds: int) -> dict[object, int]:
    unique = sorted(np.unique(values))
    if len(unique) < folds:
        raise ValueError(f"Block validation needs at least {folds} unique values")
    return {value: min(index * folds // len(unique), folds - 1) for index, value in enumerate(unique)}


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
```

- [ ] **Step 6: Export the public validation API**

```python
# src/ml4gm/validation/__init__.py
from ml4gm.validation.block import block_splits
from ml4gm.validation.loyo import Split, loyo_splits
from ml4gm.validation.spatial import spatial_splits

__all__ = ["Split", "block_splits", "loyo_splits", "spatial_splits"]
```

- [ ] **Step 7: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_validation.py -v
```

Expected: all tests pass and every leakage assertion holds.

- [ ] **Step 8: Commit**

```bash
git add src/ml4gm/validation tests/unit/test_validation.py
git commit -m "feat: enforce leakage-aware validation splits"
```

---

### Task 5: Implement Fold-Local Preprocessing, Metrics, and Result Records

**Files:**

- Create: `src/ml4gm/data/preprocessing.py`
- Create: `src/ml4gm/evaluation/__init__.py`
- Create: `src/ml4gm/evaluation/metrics.py`
- Create: `src/ml4gm/evaluation/results.py`
- Create: `tests/unit/test_preprocessing.py`
- Create: `tests/unit/test_metrics.py`
- Create: `tests/unit/test_results.py`

**Interfaces:**

- Produces: `prepare_fold(frame, train_idx, test_idx, ...) -> PreparedFold`.
- Produces: `regression_metrics(y_true, y_pred) -> dict[str, float]`.
- Produces: `RunResult.write(path) -> Path`.

- [ ] **Step 1: Write failing preprocessing test**

```python
# tests/unit/test_preprocessing.py
import numpy as np
import pandas as pd

from ml4gm.data.preprocessing import prepare_fold


def test_scaler_is_fitted_on_training_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A", "A", "B"],
            "year": [2000, 2001, 2000],
            "dhdt": [0.0, 1.0, 2.0],
            "x": [0.0, 2.0, 100.0],
        }
    )
    fold = prepare_fold(frame, np.array([0, 1]), np.array([2]), "dhdt", "rgiid", "year")
    assert np.isclose(fold.X_train.mean(), 0.0)
    assert fold.X_test[0, 0] > 50.0
```

- [ ] **Step 2: Write failing metrics and result tests**

```python
# tests/unit/test_metrics.py
import numpy as np

from ml4gm.evaluation.metrics import regression_metrics


def test_regression_metrics() -> None:
    metrics = regression_metrics(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    assert metrics == {"r2": -1.0, "rmse": 2**-0.5, "mae": 0.5}
```

```python
# tests/unit/test_results.py
import json
from pathlib import Path

from ml4gm.evaluation.results import FoldResult, RunResult


def test_run_result_writes_json(tmp_path: Path) -> None:
    result = RunResult(
        run_name="test",
        model="random_forest",
        validation="loyo",
        seed=42,
        folds=[FoldResult("year-2000", 8, 2, 0.4, 0.3, 0.2)],
    )
    path = result.write(tmp_path / "result.json")
    assert json.loads(path.read_text())["folds"][0]["fold"] == "year-2000"
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_preprocessing.py tests/unit/test_metrics.py tests/unit/test_results.py -v
```

Expected: missing-module failures.

- [ ] **Step 4: Implement fold-local preprocessing**

```python
# src/ml4gm/data/preprocessing.py
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PreparedFold:
    X_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_train: NDArray[np.float64]
    y_test: NDArray[np.float64]
    feature_names: list[str]
    transformer: Pipeline


def prepare_fold(
    frame: pd.DataFrame,
    train_idx: NDArray[np.int_],
    test_idx: NDArray[np.int_],
    target: str,
    glacier_id: str,
    year: str,
) -> PreparedFold:
    features = [column for column in frame.columns if column not in {target, glacier_id, year}]
    if not features:
        raise ValueError("No model features remain after excluding identifiers and target")
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]
    transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    X_train = transformer.fit_transform(train[features]).astype(float)
    X_test = transformer.transform(test[features]).astype(float)
    return PreparedFold(
        X_train,
        X_test,
        train[target].to_numpy(dtype=float),
        test[target].to_numpy(dtype=float),
        features,
        transformer,
    )
```

- [ ] **Step 5: Implement metrics**

```python
# src/ml4gm/evaluation/metrics.py
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
```

- [ ] **Step 6: Implement result records**

```python
# src/ml4gm/evaluation/results.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FoldResult:
    fold: str
    n_train: int
    n_test: int
    r2: float
    rmse: float
    mae: float


@dataclass(frozen=True)
class RunResult:
    run_name: str
    model: str
    validation: str
    seed: int
    folds: list[FoldResult]

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path
```

- [ ] **Step 7: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_preprocessing.py tests/unit/test_metrics.py tests/unit/test_results.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/ml4gm/data/preprocessing.py src/ml4gm/evaluation tests/unit/test_preprocessing.py tests/unit/test_metrics.py tests/unit/test_results.py
git commit -m "feat: add fold-local preprocessing and result records"
```

---

### Task 6: Implement the Model Protocol and Random Forest Vertical Slice

**Files:**

- Create: `src/ml4gm/models/__init__.py`
- Create: `src/ml4gm/models/base.py`
- Create: `src/ml4gm/models/random_forest.py`
- Create: `src/ml4gm/evaluation/runner.py`
- Create: `tests/unit/test_models.py`
- Create: `tests/integration/test_quickstart.py`

**Interfaces:**

- Produces: `ModelAdapter.fit`, `predict`, and `save`.
- Produces: `create_model(name, parameters, seed) -> ModelAdapter`.
- Produces: `run_evaluation(config: RunConfig) -> RunResult`.

- [ ] **Step 1: Write failing model and integration tests**

```python
# tests/unit/test_models.py
import numpy as np

from ml4gm.models import create_model


def test_random_forest_adapter_round_trip(tmp_path) -> None:
    model = create_model("random_forest", {"n_estimators": 5, "n_jobs": 1}, seed=42)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model.fit(X, y)
    assert model.predict(X).shape == (4,)
    model.save(tmp_path / "model.joblib")
    assert (tmp_path / "model.joblib").exists()
```

```python
# tests/integration/test_quickstart.py
from pathlib import Path

from ml4gm.config import RunConfig
from ml4gm.evaluation.runner import run_evaluation


def test_quickstart_produces_all_year_folds(tmp_path: Path) -> None:
    config = RunConfig.from_yaml(Path("configs/quickstart.yaml"))
    config = RunConfig(
        data=type(config.data)(
            input=config.data.input,
            target=config.data.target,
            glacier_id=config.data.glacier_id,
            year=config.data.year,
            output_dir=tmp_path,
        ),
        model=config.model,
        validation=config.validation,
        random_seed=config.random_seed,
        run_name=config.run_name,
    )
    result = run_evaluation(config)
    assert len(result.folds) == 6
    assert (tmp_path / "result.json").exists()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_models.py tests/integration/test_quickstart.py -v
```

Expected: missing model and runner modules.

- [ ] **Step 3: Implement the protocol and Random Forest adapter**

```python
# src/ml4gm/models/base.py
from pathlib import Path
from typing import Protocol

from numpy.typing import NDArray


class ModelAdapter(Protocol):
    def fit(self, X: NDArray, y: NDArray) -> "ModelAdapter": ...
    def predict(self, X: NDArray) -> NDArray: ...
    def save(self, path: Path) -> None: ...
```

```python
# src/ml4gm/models/random_forest.py
from pathlib import Path
from typing import Any

import joblib
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor


class RandomForestAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        self.estimator = RandomForestRegressor(random_state=seed, **parameters)

    def fit(self, X: NDArray, y: NDArray) -> "RandomForestAdapter":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        return self.estimator.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
```

- [ ] **Step 4: Implement the registry**

```python
# src/ml4gm/models/__init__.py
from typing import Any

from ml4gm.models.base import ModelAdapter
from ml4gm.models.random_forest import RandomForestAdapter


def create_model(name: str, parameters: dict[str, Any], seed: int) -> ModelAdapter:
    if name == "random_forest":
        return RandomForestAdapter(parameters, seed)
    raise ValueError(f"Unsupported or unavailable model: {name}")


__all__ = ["ModelAdapter", "create_model"]
```

- [ ] **Step 5: Implement the evaluation runner**

```python
# src/ml4gm/evaluation/runner.py
from __future__ import annotations

import pandas as pd

from ml4gm.config import RunConfig
from ml4gm.data.preprocessing import prepare_fold
from ml4gm.data.schema import validate_annual_table
from ml4gm.evaluation.metrics import regression_metrics
from ml4gm.evaluation.results import FoldResult, RunResult
from ml4gm.models import create_model
from ml4gm.validation import block_splits, loyo_splits, spatial_splits


def _splits(config: RunConfig, frame: pd.DataFrame):
    if config.validation.strategy == "loyo":
        return loyo_splits(frame[config.data.year].to_numpy())
    if config.validation.strategy == "spatial":
        return spatial_splits(
            frame[config.data.glacier_id].to_numpy(),
            config.validation.folds,
            config.random_seed,
        )
    return block_splits(
        frame[config.data.glacier_id].to_numpy(),
        frame[config.data.year].to_numpy(),
        config.validation.folds,
    )


def run_evaluation(config: RunConfig) -> RunResult:
    if not config.data.input.exists():
        raise FileNotFoundError(
            f"Input data not found: {config.data.input}. "
            "Generate the sample or follow docs/full-data-setup.md."
        )
    frame = validate_annual_table(
        pd.read_csv(config.data.input),
        config.data.target,
        config.data.glacier_id,
        config.data.year,
    )
    fold_results: list[FoldResult] = []
    for split in _splits(config, frame):
        prepared = prepare_fold(
            frame,
            split.train,
            split.test,
            config.data.target,
            config.data.glacier_id,
            config.data.year,
        )
        model = create_model(config.model.name, config.model.parameters, config.random_seed)
        model.fit(prepared.X_train, prepared.y_train)
        metrics = regression_metrics(prepared.y_test, model.predict(prepared.X_test))
        fold_results.append(
            FoldResult(
                split.fold,
                len(split.train),
                len(split.test),
                metrics["r2"],
                metrics["rmse"],
                metrics["mae"],
            )
        )
    result = RunResult(
        config.run_name,
        config.model.name,
        config.validation.strategy,
        config.random_seed,
        fold_results,
    )
    result.write(config.data.output_dir / "result.json")
    return result
```

- [ ] **Step 6: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_models.py tests/integration/test_quickstart.py -v
```

Expected: all tests pass and `result.json` is created in the temporary output directory.

- [ ] **Step 7: Commit**

```bash
git add src/ml4gm/models src/ml4gm/evaluation/runner.py tests/unit/test_models.py tests/integration/test_quickstart.py
git commit -m "feat: add the Random Forest benchmark vertical slice"
```

---

### Task 7: Add Dataset Preparation and Manifest Writing

**Files:**

- Create: `src/ml4gm/data/prepare.py`
- Create: `tests/integration/test_prepare.py`
- Modify: `src/ml4gm/data/__init__.py`

**Interfaces:**

- Produces: `prepare_dataset(input_path, output_path, manifest_path, ...) -> tuple[Path, Path]`.
- Consumes: `validate_annual_table` and `DatasetManifest`.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/test_prepare.py
import json
from pathlib import Path

from ml4gm.data.prepare import prepare_dataset


def test_prepare_sample_writes_sorted_data_and_manifest(tmp_path: Path) -> None:
    data_path, manifest_path = prepare_dataset(
        Path("data/sample/glacier_sample.csv"),
        tmp_path / "prepared.csv",
        tmp_path / "manifest.json",
        ["synthetic-ml4gm"],
    )
    assert data_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["rows"] == 72
    assert manifest["sources"] == ["synthetic-ml4gm"]
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_prepare.py -v
```

Expected: missing module.

- [ ] **Step 3: Implement preparation**

```python
# src/ml4gm/data/prepare.py
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
    frame = validate_annual_table(pd.read_csv(input_path), target, glacier_id, year)
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
```

- [ ] **Step 4: Run test**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_prepare.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/ml4gm/data/prepare.py tests/integration/test_prepare.py
git commit -m "feat: prepare traceable glacier datasets"
```

---

### Task 8: Implement the Command-Line Interface

**Files:**

- Create: `src/ml4gm/cli.py`
- Create: `tests/integration/test_cli.py`

**Interfaces:**

- Produces: `main(argv: list[str] | None = None) -> int`.
- `ml4gm data prepare --config ... --source synthetic-ml4gm`.
- `ml4gm train --config ...` and `ml4gm evaluate --config ...` run one configured evaluation.
- `ml4gm benchmark --config ...` runs the same stable pipeline and prints fold summary.

- [ ] **Step 1: Write failing CLI tests**

```python
# tests/integration/test_cli.py
from pathlib import Path

from ml4gm.cli import main


def test_cli_runs_quickstart(tmp_path: Path) -> None:
    exit_code = main(
        [
            "evaluate",
            "--config",
            "configs/quickstart.yaml",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "result.json").exists()


def test_cli_reports_missing_config(capsys) -> None:
    assert main(["evaluate", "--config", "missing.yaml"]) == 2
    assert "Configuration file not found" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_cli.py -v
```

Expected: missing module.

- [ ] **Step 3: Implement the CLI**

```python
# src/ml4gm/cli.py
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from ml4gm.config import ConfigError, RunConfig
from ml4gm.data.prepare import prepare_dataset
from ml4gm.evaluation.runner import run_evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml4gm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    data = subparsers.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    prepare = data_sub.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--source", action="append", required=True)
    prepare.add_argument("--output", default="data/processed/prepared.csv")
    prepare.add_argument("--manifest", default="data/processed/manifest.json")
    for name in ("train", "evaluate", "benchmark"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        return 2
    try:
        config = RunConfig.from_yaml(config_path)
        if args.command == "data":
            prepare_dataset(
                config.data.input,
                Path(args.output),
                Path(args.manifest),
                list(args.source),
                config.data.target,
                config.data.glacier_id,
                config.data.year,
            )
            return 0
        if args.output_dir:
            config = replace(
                config,
                data=replace(config.data, output_dir=Path(args.output_dir)),
            )
        result = run_evaluation(config)
        for fold in result.folds:
            print(
                f"{fold.fold}: R2={fold.r2:.4f} RMSE={fold.rmse:.4f} MAE={fold.mae:.4f}"
            )
        return 0
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"ML4GM error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests and smoke command**

Run:

```bash
.venv/bin/python -m pytest tests/integration/test_cli.py -v
.venv/bin/ml4gm evaluate --config configs/quickstart.yaml --output-dir /tmp/ml4gm-quickstart
```

Expected: tests pass and six year-fold metric lines print.

- [ ] **Step 5: Commit**

```bash
git add src/ml4gm/cli.py tests/integration/test_cli.py
git commit -m "feat: expose reproducible ML4GM commands"
```

---

### Task 9: Add Optional LightGBM Support

**Files:**

- Create: `src/ml4gm/models/lightgbm.py`
- Modify: `src/ml4gm/models/__init__.py`
- Modify: `tests/unit/test_models.py`

**Interfaces:**

- Produces: `LightGBMAdapter`.
- Missing dependency raises `RuntimeError` with install command `pip install 'ml4gm[lightgbm]'`.

- [ ] **Step 1: Add failing dependency and adapter tests**

```python
# append to tests/unit/test_models.py
import importlib.util
import pytest


def test_lightgbm_dependency_message() -> None:
    if importlib.util.find_spec("lightgbm") is not None:
        pytest.skip("LightGBM installed in this environment")
    with pytest.raises(RuntimeError, match=r"ml4gm\\[lightgbm\\]"):
        create_model("lightgbm", {}, seed=42)
```

- [ ] **Step 2: Run the targeted test**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_models.py::test_lightgbm_dependency_message -v
```

Expected: current registry raises the wrong error.

- [ ] **Step 3: Implement adapter**

```python
# src/ml4gm/models/lightgbm.py
from pathlib import Path
from typing import Any

import joblib
from numpy.typing import NDArray


class LightGBMAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM support requires: pip install 'ml4gm[lightgbm]'"
            ) from exc
        self.estimator = LGBMRegressor(random_state=seed, verbosity=-1, **parameters)

    def fit(self, X: NDArray, y: NDArray) -> "LightGBMAdapter":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        return self.estimator.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path)
```

Update the registry:

```python
if name == "lightgbm":
    from ml4gm.models.lightgbm import LightGBMAdapter

    return LightGBMAdapter(parameters, seed)
```

- [ ] **Step 4: Run core and optional tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_models.py -v
.venv/bin/python -m pip install -e ".[lightgbm]"
.venv/bin/python -m pytest tests/unit/test_models.py -v
```

Expected: dependency-message test skips after installation; add and run an adapter round-trip equivalent to the Random Forest test.

- [ ] **Step 5: Commit**

```bash
git add src/ml4gm/models/lightgbm.py src/ml4gm/models/__init__.py tests/unit/test_models.py
git commit -m "feat: add optional LightGBM benchmarks"
```

---

### Task 10: Add Deterministic PyTorch Utilities and MLP

**Files:**

- Create: `src/ml4gm/models/torch_utils.py`
- Create: `src/ml4gm/models/mlp.py`
- Modify: `src/ml4gm/models/__init__.py`
- Create: `tests/unit/test_neural_models.py`

**Interfaces:**

- Produces: `require_torch()`.
- Produces: `set_torch_seed(seed)`.
- Produces: `MLPAdapter(parameters, seed)`.
- Neural parameters accept `hidden_sizes`, `dropout`, `learning_rate`, `epochs`, and `batch_size`.

- [ ] **Step 1: Write failing MLP smoke test**

```python
# tests/unit/test_neural_models.py
import importlib.util

import numpy as np
import pytest

from ml4gm.models import create_model

torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
def test_mlp_forward_and_fit() -> None:
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = X.sum(axis=1)
    model = create_model(
        "mlp",
        {"hidden_sizes": [8, 4], "epochs": 2, "batch_size": 4},
        seed=42,
    )
    model.fit(X, y)
    assert model.predict(X).shape == (8,)
```

- [ ] **Step 2: Run test and verify unsupported model**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_neural_models.py -v
```

Expected: skip if PyTorch is absent or fail with unsupported model if present.

- [ ] **Step 3: Implement deterministic utilities**

```python
# src/ml4gm/models/torch_utils.py
from __future__ import annotations

import random

import numpy as np


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch support requires: pip install 'ml4gm[torch]'") from exc
    return torch


def set_torch_seed(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Implement the MLP adapter**

Implement `src/ml4gm/models/mlp.py` with:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml4gm.models.torch_utils import require_torch, set_torch_seed


class MLPAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        self.parameters = dict(parameters)
        self.seed = seed
        self.hidden_sizes = [int(value) for value in parameters.get("hidden_sizes", [128, 64])]
        self.dropout = float(parameters.get("dropout", 0.2))
        self.learning_rate = float(parameters.get("learning_rate", 1e-3))
        self.epochs = int(parameters.get("epochs", 100))
        self.batch_size = int(parameters.get("batch_size", 256))
        self._torch = require_torch()
        self._model = None
        self._input_features: int | None = None

    def _build(self, input_features: int):
        nn = self._torch.nn
        layers: list[Any] = []
        previous = input_features
        for hidden in self.hidden_sizes:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(self.dropout)])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)

    def fit(self, X: NDArray, y: NDArray) -> "MLPAdapter":
        values = np.asarray(X, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("MLP X must be a non-empty two-dimensional array")
        if len(values) != len(targets):
            raise ValueError("MLP X and y must contain the same number of rows")
        set_torch_seed(self.seed)
        self._input_features = values.shape[1]
        self._model = self._build(self._input_features)
        dataset = self._torch.utils.data.TensorDataset(
            self._torch.from_numpy(values),
            self._torch.from_numpy(targets),
        )
        generator = self._torch.Generator().manual_seed(self.seed)
        loader = self._torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = self._torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
        )
        loss_fn = self._torch.nn.MSELoss()
        self._model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: NDArray) -> NDArray:
        if self._model is None:
            raise ValueError("MLP must be fitted before predict")
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("MLP X must be a two-dimensional array")
        self._model.eval()
        with self._torch.no_grad():
            return self._model(self._torch.from_numpy(values)).numpy().reshape(-1)

    def save(self, path: Path) -> None:
        if self._model is None or self._input_features is None:
            raise ValueError("MLP must be fitted before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "state_dict": self._model.state_dict(),
                "input_features": self._input_features,
                "parameters": self.parameters,
                "seed": self.seed,
            },
            path,
        )
```

Register the adapter with:

```python
if name == "mlp":
    from ml4gm.models.mlp import MLPAdapter

    return MLPAdapter(parameters, seed)
```

- [ ] **Step 5: Register MLP and run tests**

Run:

```bash
.venv/bin/python -m pip install -e ".[torch]"
.venv/bin/python -m pytest tests/unit/test_neural_models.py -v
```

Expected: MLP smoke test passes.

- [ ] **Step 6: Commit**

```bash
git add src/ml4gm/models/torch_utils.py src/ml4gm/models/mlp.py src/ml4gm/models/__init__.py tests/unit/test_neural_models.py
git commit -m "feat: migrate the tabular MLP model"
```

---

### Task 11: Add Seasonal and Temporal Sequence Builders and LSTMs

**Files:**

- Create: `src/ml4gm/data/sequences.py`
- Create: `src/ml4gm/models/seasonal_lstm.py`
- Create: `src/ml4gm/models/temporal_lstm.py`
- Modify: `src/ml4gm/models/__init__.py`
- Create: `tests/unit/test_sequences.py`
- Modify: `tests/unit/test_neural_models.py`

**Interfaces:**

- Produces: `build_seasonal_sequences(frame, monthly_variables, static_features)`.
- Produces: `build_temporal_sequences(frame, feature_columns, lookback, glacier_id, year, target)`.
- Produces optional adapters for model registry.

- [ ] **Step 1: Write failing sequence tests**

```python
# tests/unit/test_sequences.py
import pandas as pd

from ml4gm.data.sequences import build_seasonal_sequences, build_temporal_sequences


def test_seasonal_builder_orders_months() -> None:
    row = {"rgiid": "A", "year": 2000, "dhdt": -0.2, "Area": 5.0}
    for month in range(1, 13):
        row[f"{month}_t2m"] = float(month)
        row[f"{month}_tp"] = float(month * 10)
    result = build_seasonal_sequences(pd.DataFrame([row]), ["t2m", "tp"], ["Area"])
    assert result.sequence.shape == (1, 12, 2)
    assert result.sequence[0, 0].tolist() == [1.0, 10.0]


def test_temporal_builder_never_crosses_glaciers() -> None:
    frame = pd.DataFrame(
        {
            "rgiid": ["A"] * 4 + ["B"] * 4,
            "year": [2000, 2001, 2002, 2003] * 2,
            "dhdt": range(8),
            "x": range(8),
        }
    )
    result = build_temporal_sequences(frame, ["x"], 2, "rgiid", "year", "dhdt")
    assert result.sequence.shape == (4, 2, 1)
    assert result.glacier_ids.tolist() == ["A", "A", "B", "B"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_sequences.py -v
```

Expected: missing module.

- [ ] **Step 3: Implement sequence records and builders**

```python
# src/ml4gm/data/sequences.py
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class SequenceData:
    sequence: NDArray[np.float64]
    static: NDArray[np.float64]
    target: NDArray[np.float64]
    glacier_ids: NDArray
    years: NDArray[np.int_]


def build_seasonal_sequences(
    frame: pd.DataFrame,
    monthly_variables: list[str],
    static_features: list[str],
    target: str = "dhdt",
    glacier_id: str = "rgiid",
    year: str = "year",
) -> SequenceData:
    monthly_columns = [
        [f"{month}_{variable}" for variable in monthly_variables] for month in range(1, 13)
    ]
    missing = [
        column
        for columns in monthly_columns
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing seasonal columns: {', '.join(missing[:5])}")
    sequence = np.stack(
        [frame[columns].to_numpy(dtype=float) for columns in monthly_columns],
        axis=1,
    )
    return SequenceData(
        sequence,
        frame[static_features].to_numpy(dtype=float),
        frame[target].to_numpy(dtype=float),
        frame[glacier_id].to_numpy(),
        frame[year].to_numpy(dtype=int),
    )


def build_temporal_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    lookback: int,
    glacier_id: str,
    year: str,
    target: str,
) -> SequenceData:
    sequences, targets, glaciers, years = [], [], [], []
    for glacier, group in frame.sort_values([glacier_id, year]).groupby(glacier_id):
        ordered = group.reset_index(drop=True)
        for end in range(lookback, len(ordered)):
            window = ordered.iloc[end - lookback : end]
            expected = list(range(int(window[year].iloc[0]), int(ordered[year].iloc[end]) + 1))
            observed = [*window[year].astype(int), int(ordered[year].iloc[end])]
            if observed != expected:
                continue
            sequences.append(window[feature_columns].to_numpy(dtype=float))
            targets.append(float(ordered[target].iloc[end]))
            glaciers.append(glacier)
            years.append(int(ordered[year].iloc[end]))
    if not sequences:
        raise ValueError("No valid temporal sequences were produced")
    array = np.stack(sequences)
    return SequenceData(
        array,
        np.empty((len(array), 0)),
        np.asarray(targets),
        np.asarray(glaciers),
        np.asarray(years),
    )
```

- [ ] **Step 4: Implement Seasonal-LSTM**

```python
# src/ml4gm/models/seasonal_lstm.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml4gm.models.torch_utils import require_torch, set_torch_seed


class SeasonalLSTMAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        self.parameters = dict(parameters)
        self.seed = seed
        self.hidden = int(parameters.get("hidden", 128))
        self.num_layers = int(parameters.get("num_layers", 1))
        self.dropout = float(parameters.get("dropout", 0.2))
        self.learning_rate = float(parameters.get("learning_rate", 1e-3))
        self.epochs = int(parameters.get("epochs", 100))
        self.batch_size = int(parameters.get("batch_size", 256))
        self._torch = require_torch()
        self._model = None
        self._dimensions: tuple[int, int] | None = None

    def _build(self, sequence_features: int, static_features: int):
        torch = self._torch

        class Network(torch.nn.Module):
            def __init__(self, outer) -> None:
                super().__init__()
                effective_dropout = outer.dropout if outer.num_layers > 1 else 0.0
                self.lstm = torch.nn.LSTM(
                    sequence_features,
                    outer.hidden,
                    num_layers=outer.num_layers,
                    dropout=effective_dropout,
                    batch_first=True,
                )
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(outer.hidden + static_features, outer.hidden // 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(outer.dropout),
                    torch.nn.Linear(outer.hidden // 2, 1),
                )

            def forward(self, sequence, static):
                _, (hidden, _) = self.lstm(sequence)
                combined = torch.cat([hidden[-1], static], dim=1)
                return self.head(combined)

        return Network(self)

    def fit(self, X: NDArray, y: NDArray):
        del X, y
        raise ValueError("Seasonal-LSTM requires fit_inputs(sequence, static, y)")

    def predict(self, X: NDArray):
        del X
        raise ValueError("Seasonal-LSTM requires predict_inputs(sequence, static)")

    def fit_inputs(
        self,
        sequence: NDArray,
        static: NDArray,
        y: NDArray,
    ) -> "SeasonalLSTMAdapter":
        sequence_values = np.asarray(sequence, dtype=np.float32)
        static_values = np.asarray(static, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if sequence_values.ndim != 3 or sequence_values.shape[1] != 12:
            raise ValueError("Seasonal sequence must have shape (rows, 12, features)")
        if static_values.ndim != 2 or len(static_values) != len(sequence_values):
            raise ValueError("Seasonal static features must align with sequence rows")
        set_torch_seed(self.seed)
        self._dimensions = (sequence_values.shape[2], static_values.shape[1])
        self._model = self._build(*self._dimensions)
        dataset = self._torch.utils.data.TensorDataset(
            self._torch.from_numpy(sequence_values),
            self._torch.from_numpy(static_values),
            self._torch.from_numpy(targets),
        )
        generator = self._torch.Generator().manual_seed(self.seed)
        loader = self._torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = self._torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = self._torch.nn.MSELoss()
        self._model.train()
        for _ in range(self.epochs):
            for batch_sequence, batch_static, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(batch_sequence, batch_static), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_inputs(self, sequence: NDArray, static: NDArray) -> NDArray:
        if self._model is None:
            raise ValueError("Seasonal-LSTM must be fitted before predict")
        sequence_values = np.asarray(sequence, dtype=np.float32)
        static_values = np.asarray(static, dtype=np.float32)
        self._model.eval()
        with self._torch.no_grad():
            return (
                self._model(
                    self._torch.from_numpy(sequence_values),
                    self._torch.from_numpy(static_values),
                )
                .numpy()
                .reshape(-1)
            )

    def save(self, path: Path) -> None:
        if self._model is None or self._dimensions is None:
            raise ValueError("Seasonal-LSTM must be fitted before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "state_dict": self._model.state_dict(),
                "dimensions": self._dimensions,
                "parameters": self.parameters,
                "seed": self.seed,
            },
            path,
        )
```

Append this test:

```python
@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
def test_seasonal_lstm_smoke() -> None:
    model = create_model(
        "seasonal_lstm",
        {"hidden": 4, "epochs": 2, "batch_size": 2},
        seed=42,
    )
    sequence = np.ones((2, 12, 2), dtype=float)
    static = np.ones((2, 3), dtype=float)
    model.fit_inputs(sequence, static, np.array([0.0, 1.0]))
    assert model.predict_inputs(sequence, static).shape == (2,)
```

- [ ] **Step 5: Implement Temporal-LSTM**

```python
# src/ml4gm/models/temporal_lstm.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml4gm.models.torch_utils import require_torch, set_torch_seed


class TemporalLSTMAdapter:
    def __init__(self, parameters: dict[str, Any], seed: int) -> None:
        self.parameters = dict(parameters)
        self.seed = seed
        self.lookback = int(parameters.get("lookback", 3))
        self.hidden = int(parameters.get("hidden", 128))
        self.num_layers = int(parameters.get("num_layers", 1))
        self.dropout = float(parameters.get("dropout", 0.2))
        self.learning_rate = float(parameters.get("learning_rate", 1e-3))
        self.epochs = int(parameters.get("epochs", 100))
        self.batch_size = int(parameters.get("batch_size", 256))
        self._torch = require_torch()
        self._model = None
        self._input_features: int | None = None

    def _build(self, input_features: int):
        torch = self._torch

        class Network(torch.nn.Module):
            def __init__(self, outer) -> None:
                super().__init__()
                effective_dropout = outer.dropout if outer.num_layers > 1 else 0.0
                self.lstm = torch.nn.LSTM(
                    input_features,
                    outer.hidden,
                    num_layers=outer.num_layers,
                    dropout=effective_dropout,
                    batch_first=True,
                )
                self.output = torch.nn.Linear(outer.hidden, 1)

            def forward(self, sequence):
                _, (hidden, _) = self.lstm(sequence)
                return self.output(hidden[-1])

        return Network(self)

    def fit(self, X: NDArray, y: NDArray):
        del X, y
        raise ValueError("Temporal-LSTM requires fit_inputs(sequence, y)")

    def predict(self, X: NDArray):
        del X
        raise ValueError("Temporal-LSTM requires predict_inputs(sequence)")

    def fit_inputs(self, sequence: NDArray, y: NDArray) -> "TemporalLSTMAdapter":
        values = np.asarray(sequence, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if values.ndim != 3 or values.shape[1] != self.lookback:
            raise ValueError(
                f"Temporal sequence must have shape (rows, {self.lookback}, features)"
            )
        set_torch_seed(self.seed)
        self._input_features = values.shape[2]
        self._model = self._build(self._input_features)
        dataset = self._torch.utils.data.TensorDataset(
            self._torch.from_numpy(values),
            self._torch.from_numpy(targets),
        )
        generator = self._torch.Generator().manual_seed(self.seed)
        loader = self._torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = self._torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = self._torch.nn.MSELoss()
        self._model.train()
        for _ in range(self.epochs):
            for batch_sequence, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(batch_sequence), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_inputs(self, sequence: NDArray) -> NDArray:
        if self._model is None:
            raise ValueError("Temporal-LSTM must be fitted before predict")
        values = np.asarray(sequence, dtype=np.float32)
        self._model.eval()
        with self._torch.no_grad():
            return self._model(self._torch.from_numpy(values)).numpy().reshape(-1)

    def save(self, path: Path) -> None:
        if self._model is None or self._input_features is None:
            raise ValueError("Temporal-LSTM must be fitted before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(
            {
                "state_dict": self._model.state_dict(),
                "input_features": self._input_features,
                "parameters": self.parameters,
                "seed": self.seed,
            },
            path,
        )
```

Append this test:

```python
@pytest.mark.skipif(not torch_available, reason="install ml4gm[torch]")
def test_temporal_lstm_smoke() -> None:
    model = create_model(
        "temporal_lstm",
        {"lookback": 2, "hidden": 4, "epochs": 2, "batch_size": 2},
        seed=42,
    )
    sequence = np.ones((4, 2, 3), dtype=float)
    model.fit_inputs(sequence, np.arange(4, dtype=float))
    assert model.predict_inputs(sequence).shape == (4,)
```

Register both adapters:

```python
if name == "seasonal_lstm":
    from ml4gm.models.seasonal_lstm import SeasonalLSTMAdapter

    return SeasonalLSTMAdapter(parameters, seed)
if name == "temporal_lstm":
    from ml4gm.models.temporal_lstm import TemporalLSTMAdapter

    return TemporalLSTMAdapter(parameters, seed)
```

- [ ] **Step 6: Route sequence models through leakage-safe sequence evaluation**

Add these imports to `src/ml4gm/evaluation/runner.py`:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

from ml4gm.data.sequences import build_seasonal_sequences, build_temporal_sequences
```

Add these helpers:

```python
def _sequence_splits(config: RunConfig, glacier_ids, years):
    if config.validation.strategy == "loyo":
        return loyo_splits(years)
    if config.validation.strategy == "spatial":
        return spatial_splits(glacier_ids, config.validation.folds, config.random_seed)
    return block_splits(glacier_ids, years, config.validation.folds)


def _scale_sequence(train: np.ndarray, test: np.ndarray):
    scaler = StandardScaler()
    train_shape = train.shape
    test_shape = test.shape
    train_scaled = scaler.fit_transform(train.reshape(-1, train_shape[-1])).reshape(train_shape)
    test_scaled = scaler.transform(test.reshape(-1, test_shape[-1])).reshape(test_shape)
    return train_scaled, test_scaled


def _run_seasonal(config: RunConfig, frame: pd.DataFrame) -> RunResult:
    parameters = dict(config.model.parameters)
    monthly_variables = list(parameters.pop("monthly_variables", ["t2m", "tp"]))
    static_features = list(parameters.pop("static_features", ["Area", "Zmed"]))
    data = build_seasonal_sequences(
        frame,
        monthly_variables,
        static_features,
        config.data.target,
        config.data.glacier_id,
        config.data.year,
    )
    results: list[FoldResult] = []
    for split in _sequence_splits(config, data.glacier_ids, data.years):
        sequence_train, sequence_test = _scale_sequence(
            data.sequence[split.train],
            data.sequence[split.test],
        )
        static_scaler = StandardScaler()
        static_train = static_scaler.fit_transform(data.static[split.train])
        static_test = static_scaler.transform(data.static[split.test])
        model = create_model("seasonal_lstm", parameters, config.random_seed)
        model.fit_inputs(sequence_train, static_train, data.target[split.train])
        metrics = regression_metrics(
            data.target[split.test],
            model.predict_inputs(sequence_test, static_test),
        )
        results.append(
            FoldResult(
                split.fold,
                len(split.train),
                len(split.test),
                metrics["r2"],
                metrics["rmse"],
                metrics["mae"],
            )
        )
    return RunResult(
        config.run_name,
        config.model.name,
        config.validation.strategy,
        config.random_seed,
        results,
    )


def _run_temporal(config: RunConfig, frame: pd.DataFrame) -> RunResult:
    parameters = dict(config.model.parameters)
    feature_columns = list(
        parameters.pop(
            "feature_columns",
            [
                column
                for column in frame.columns
                if column not in {
                    config.data.target,
                    config.data.glacier_id,
                    config.data.year,
                }
            ],
        )
    )
    lookback = int(parameters.get("lookback", 3))
    data = build_temporal_sequences(
        frame,
        feature_columns,
        lookback,
        config.data.glacier_id,
        config.data.year,
        config.data.target,
    )
    results: list[FoldResult] = []
    for split in _sequence_splits(config, data.glacier_ids, data.years):
        sequence_train, sequence_test = _scale_sequence(
            data.sequence[split.train],
            data.sequence[split.test],
        )
        model = create_model("temporal_lstm", parameters, config.random_seed)
        model.fit_inputs(sequence_train, data.target[split.train])
        metrics = regression_metrics(
            data.target[split.test],
            model.predict_inputs(sequence_test),
        )
        results.append(
            FoldResult(
                split.fold,
                len(split.train),
                len(split.test),
                metrics["r2"],
                metrics["rmse"],
                metrics["mae"],
            )
        )
    return RunResult(
        config.run_name,
        config.model.name,
        config.validation.strategy,
        config.random_seed,
        results,
    )
```

Immediately after schema validation in `run_evaluation`, add:

```python
if config.model.name == "seasonal_lstm":
    result = _run_seasonal(config, frame)
    result.write(config.data.output_dir / "result.json")
    return result
if config.model.name == "temporal_lstm":
    result = _run_temporal(config, frame)
    result.write(config.data.output_dir / "result.json")
    return result
```

Create:

```python
# tests/integration/test_sequence_models.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml4gm.config import DataConfig, ModelConfig, RunConfig, ValidationConfig
from ml4gm.data.sequences import build_temporal_sequences
from ml4gm.evaluation.runner import run_evaluation
from ml4gm.validation import spatial_splits

pytest.importorskip("torch")


def _seasonal_frame() -> pd.DataFrame:
    rows = []
    for glacier in ("A", "B", "C", "D"):
        for year in range(2000, 2004):
            row = {
                "rgiid": glacier,
                "year": year,
                "dhdt": float(year - 2002),
                "Area": 5.0,
                "Zmed": 4500.0,
            }
            for month in range(1, 13):
                row[f"{month}_t2m"] = float(month + year - 2000)
                row[f"{month}_tp"] = float(month * 10)
            rows.append(row)
    return pd.DataFrame(rows)


def test_seasonal_runner_writes_result(tmp_path: Path) -> None:
    input_path = tmp_path / "seasonal.csv"
    _seasonal_frame().to_csv(input_path, index=False)
    config = RunConfig(
        data=DataConfig(input_path, output_dir=tmp_path / "seasonal-output"),
        model=ModelConfig(
            "seasonal_lstm",
            {
                "monthly_variables": ["t2m", "tp"],
                "static_features": ["Area", "Zmed"],
                "hidden": 4,
                "epochs": 2,
                "batch_size": 4,
            },
        ),
        validation=ValidationConfig("loyo", 2),
        random_seed=42,
        run_name="seasonal-test",
    )
    result = run_evaluation(config)
    assert len(result.folds) == 4
    assert (config.data.output_dir / "result.json").exists()


def test_temporal_spatial_groups_and_runner(tmp_path: Path) -> None:
    frame = _seasonal_frame()[["rgiid", "year", "dhdt", "Area", "Zmed"]]
    sequence_data = build_temporal_sequences(
        frame,
        ["Area", "Zmed"],
        2,
        "rgiid",
        "year",
        "dhdt",
    )
    for split in spatial_splits(sequence_data.glacier_ids, folds=2, seed=42):
        assert set(sequence_data.glacier_ids[split.train]).isdisjoint(
            set(sequence_data.glacier_ids[split.test])
        )
    input_path = tmp_path / "temporal.csv"
    frame.to_csv(input_path, index=False)
    config = RunConfig(
        data=DataConfig(input_path, output_dir=tmp_path / "temporal-output"),
        model=ModelConfig(
            "temporal_lstm",
            {
                "feature_columns": ["Area", "Zmed"],
                "lookback": 2,
                "hidden": 4,
                "epochs": 2,
                "batch_size": 4,
            },
        ),
        validation=ValidationConfig("spatial", 2),
        random_seed=42,
        run_name="temporal-test",
    )
    result = run_evaluation(config)
    assert len(result.folds) == 2
    assert (config.data.output_dir / "result.json").exists()
```

- [ ] **Step 7: Register the two adapters and run tests**

Run:

```bash
.venv/bin/python -m pytest tests/unit/test_sequences.py tests/unit/test_neural_models.py -v
```

Expected: all sequence and neural smoke tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/ml4gm/data/sequences.py src/ml4gm/models/seasonal_lstm.py src/ml4gm/models/temporal_lstm.py src/ml4gm/models/__init__.py src/ml4gm/evaluation/runner.py tests/unit/test_sequences.py tests/unit/test_neural_models.py tests/integration/test_sequence_models.py
git commit -m "feat: migrate seasonal and temporal LSTM models"
```

---

### Task 12: Preserve Legacy Notebooks and Add Supported Tutorials

**Files:**

- Move individually: all existing `code/*.ipynb` to `notebooks/legacy/`.
- Move: `code/README_Notebooks.md` to `notebooks/legacy/README.md`.
- Move: `code/split_to_notebooks.py` to `notebooks/legacy/split_to_notebooks.py`.
- Create: `notebooks/tutorials/01_quickstart.ipynb`.
- Create: `notebooks/README.md`.
- Create: `tests/repository/test_notebooks.py`.
- Create: `tests/repository/test_forbidden_paths.py`.

**Interfaces:**

- Legacy notebooks remain byte-for-byte unchanged by the move.
- Tutorial imports `ml4gm` and runs the supported quickstart API.

- [ ] **Step 1: Write failing repository tests**

```python
# tests/repository/test_forbidden_paths.py
from pathlib import Path

FORBIDDEN = ("C:\\\\ML4GM", "/Users/", "/openbayes/")
SUPPORTED_ROOTS = [Path("src"), Path("configs"), Path("notebooks/tutorials"), Path("README.md")]


def test_supported_files_do_not_contain_machine_paths() -> None:
    offenders: list[str] = []
    for root in SUPPORTED_ROOTS:
        paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(value in text for value in FORBIDDEN):
                offenders.append(str(path))
    assert offenders == []
```

```python
# tests/repository/test_notebooks.py
import json
from pathlib import Path


def test_tutorial_notebooks_are_valid_json() -> None:
    notebooks = sorted(Path("notebooks/tutorials").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/repository -v
```

Expected: tutorial notebook assertion fails.

- [ ] **Step 3: Move each legacy file individually**

Create the destination directories:

```bash
mkdir -p notebooks/legacy
mkdir -p notebooks/tutorials
```

Run each command separately, reviewing `git status` after every move:

```bash
git mv code/01_dependencies_and_data.execsrc.ipynb notebooks/legacy/01_dependencies_and_data.execsrc.ipynb
git mv code/01_dependencies_and_data.ipynb notebooks/legacy/01_dependencies_and_data.ipynb
git mv code/01_dependencies_and_data.normalized.ipynb notebooks/legacy/01_dependencies_and_data.normalized.ipynb
git mv code/02_rf.execsrc.ipynb notebooks/legacy/02_rf.execsrc.ipynb
git mv code/02_rf.ipynb notebooks/legacy/02_rf.ipynb
git mv code/03_lgbm.ipynb notebooks/legacy/03_lgbm.ipynb
git mv code/04_MLP.ipynb notebooks/legacy/04_MLP.ipynb
git mv code/05_Seasonal-LSTM.ipynb notebooks/legacy/05_Seasonal-LSTM.ipynb
git mv code/06_Temporal_lstm.ipynb notebooks/legacy/06_Temporal_lstm.ipynb
git mv code/07_compare.ipynb notebooks/legacy/07_compare.ipynb
git mv code/ML_backup.ipynb notebooks/legacy/ML_backup.ipynb
git mv code/readWW.ipynb notebooks/legacy/readWW.ipynb
git mv code/README_Notebooks.md notebooks/legacy/README.md
git mv code/split_to_notebooks.py notebooks/legacy/split_to_notebooks.py
```

Do not use a loop, wildcard, script, or batch deletion.

- [ ] **Step 4: Create the supported tutorial**

Create `notebooks/tutorials/01_quickstart.ipynb` with four cells:

1. Markdown: title and synthetic-data warning.
2. Code: load `RunConfig`.
3. Code: run `run_evaluation`.
4. Code: display a DataFrame built from `result.folds`.

The code cells must contain:

```python
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ml4gm.config import RunConfig
from ml4gm.evaluation.runner import run_evaluation
```

```python
config = RunConfig.from_yaml(Path("../../configs/quickstart.yaml"))
result = run_evaluation(config)
```

```python
pd.DataFrame([asdict(fold) for fold in result.folds])
```

- [ ] **Step 5: Document supported versus legacy notebooks**

`notebooks/README.md` must state:

- tutorials use the supported package API;
- legacy notebooks preserve historical research and embedded outputs;
- legacy notebooks contain machine-specific paths and are not the quickstart;
- users should follow the root README and configuration files.

- [ ] **Step 6: Run repository tests**

Run:

```bash
.venv/bin/python -m pytest tests/repository -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add notebooks tests/repository
git commit -m "docs: preserve legacy notebooks and add a supported tutorial"
```

---

### Task 13: Add Data Sources, Scientific Documentation, and Governance

**Files:**

- Create: `README.md`
- Create: `DATA_SOURCES.md`
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SECURITY.md`
- Create: `MAINTAINERS.md`
- Create: `CITATION.cff`
- Create: `ROADMAP.md`
- Create: `CHANGELOG.md`
- Create: `docs/data-schema.md`
- Create: `docs/full-data-setup.md`
- Create: `docs/benchmark-protocol.md`
- Create: `docs/scientific-limitations.md`
- Create: `tests/repository/test_documentation.py`

**Interfaces:**

- README quickstart commands are executable.
- Every upstream source has a stable citation and redistribution statement.
- Maintainer roles match public GitHub permissions and user-approved facts.

- [ ] **Step 1: Write failing documentation tests**

```python
# tests/repository/test_documentation.py
from pathlib import Path


def test_required_open_source_documents_exist() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "MAINTAINERS.md",
        "CITATION.cff",
        "DATA_SOURCES.md",
        "ROADMAP.md",
        "CHANGELOG.md",
    ]
    assert [path for path in required if not Path(path).exists()] == []


def test_readme_has_runnable_quickstart() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "pip install -e \".[dev]\"" in text
    assert "ml4gm evaluate --config configs/quickstart.yaml" in text
    assert "synthetic" in text.lower()
```

- [ ] **Step 2: Run tests and verify missing files**

Run:

```bash
.venv/bin/python -m pytest tests/repository/test_documentation.py -v
```

Expected: required-document test fails.

- [ ] **Step 3: Write the root README**

Use these exact top-level sections:

```markdown
# ML4GM
## Why ML4GM exists
## Scientific scope
## Project status
## Quickstart
## Full scientific data
## Models
## Validation protocols
## Results and reproducibility
## Data licensing
## Contributing
## Citation
## Security
## Scientific limitations
```

Quickstart must contain:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ml4gm evaluate --config configs/quickstart.yaml
```

State explicitly that the bundled data are synthetic and cannot support
scientific conclusions.

- [ ] **Step 4: Write `DATA_SOURCES.md`**

Create a table with these columns:

```text
Source | Version | Purpose | Official URL/DOI | Access | License/terms | Redistribution | Required citation
```

Include:

- RGI v6, DOI `10.7265/4m1f-gd79`;
- ERA5-Land, DOI `10.24381/cds.e2161bac`, CC-BY as displayed by CDS;
- Hugonnet et al. 2021, DOI `10.1038/s41586-021-03436-z`;
- a note that exact Hugonnet dataset redistribution terms must be verified from
  the official data repository before any real subset is committed.

- [ ] **Step 5: Write governance documents**

Required content:

- `CONTRIBUTING.md`: setup, tests, issue-first scientific changes, data policy,
  PR checklist.
- `CODE_OF_CONDUCT.md`: Contributor Covenant v2.1 and enforcement contact routed
  through repository maintainers without publishing a private address.
- `SECURITY.md`: supported version `0.1.x`, private GitHub security advisories,
  no unauthorized scanning.
- `MAINTAINERS.md`: Poleside as owner; HectorGao as core collaborator with write
  access; shared review, issue, CI, documentation, and release duties.
- `CITATION.cff`: version `0.1.0`, repository URL, Apache-2.0, both maintainers,
  and the project title.
- `ROADMAP.md`: v0.1.0 reproducible preview, v0.2.0 verified full benchmark,
  v0.3.0 contributor model plugins.
- `CHANGELOG.md`: Keep a Changelog format with an Unreleased section and
  `0.1.0` preparation entries.

- [ ] **Step 6: Write scientific documentation**

`docs/data-schema.md` must document types and units for `rgiid`, `year`, `dhdt`,
static features, and monthly naming such as `1_t2m`.

`docs/full-data-setup.md` must use placeholders only for paths supplied by the
user at runtime, never undocumented download URLs. It must link to
`DATA_SOURCES.md`.

`docs/benchmark-protocol.md` must define LOYO, spatial GroupKFold, and strict
block validation with the exact leakage invariants.

`docs/scientific-limitations.md` must state that:

- the project is research software;
- synthetic quickstart scores have no scientific meaning;
- historical notebook outputs have not been revalidated by the package;
- predictions are not operational hazard or water-resource guidance.

- [ ] **Step 7: Run documentation tests and inspect links**

Run:

```bash
.venv/bin/python -m pytest tests/repository/test_documentation.py -v
rg -n "TBD|TODO|C:\\\\ML4GM|/Users/|/openbayes/" README.md DATA_SOURCES.md CONTRIBUTING.md SECURITY.md MAINTAINERS.md docs
```

Expected: tests pass and `rg` returns no matches.

- [ ] **Step 8: Commit**

```bash
git add README.md DATA_SOURCES.md CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md MAINTAINERS.md CITATION.cff ROADMAP.md CHANGELOG.md docs/data-schema.md docs/full-data-setup.md docs/benchmark-protocol.md docs/scientific-limitations.md tests/repository/test_documentation.py
git commit -m "docs: publish ML4GM governance and scientific guidance"
```

---

### Task 14: Add CI, Issue Templates, and Pull-Request Controls

**Files:**

- Create: `.github/workflows/ci.yml`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/pull_request_template.md`
- Create: `tests/repository/test_github_metadata.py`

**Interfaces:**

- CI runs on pushes and pull requests.
- Required jobs cover Ruff, pytest, package build, and quickstart CLI.

- [ ] **Step 1: Write failing metadata test**

```python
# tests/repository/test_github_metadata.py
from pathlib import Path


def test_github_community_files_exist() -> None:
    required = [
        ".github/workflows/ci.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/pull_request_template.md",
    ]
    assert [path for path in required if not Path(path).exists()] == []
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/repository/test_github_metadata.py -v
```

Expected: failure listing all missing files.

- [ ] **Step 3: Create CI**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [master]
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: ruff format --check .
      - run: ruff check .
      - run: pytest
      - run: python -m build
      - run: ml4gm evaluate --config configs/quickstart.yaml --output-dir /tmp/ml4gm-ci
```

- [ ] **Step 4: Create structured issue templates**

Bug template fields:

- environment;
- command/configuration;
- expected behavior;
- actual behavior;
- minimal data description with a warning not to upload restricted data.

Feature template fields:

- scientific or maintainer problem;
- proposed behavior;
- validation requirements;
- data/license implications.

PR template checklist:

- tests added or updated;
- leakage risks considered;
- no machine paths;
- no restricted data;
- docs updated;
- quickstart and pytest pass.

- [ ] **Step 5: Run tests and local CI commands**

Run:

```bash
.venv/bin/python -m pytest tests/repository/test_github_metadata.py -v
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m pytest
.venv/bin/python -m build
```

Expected: all commands pass.

- [ ] **Step 6: Commit**

```bash
git add .github tests/repository/test_github_metadata.py
git commit -m "ci: verify ML4GM contributions and releases"
```

---

### Task 15: Prepare Repository-Backed Application Drafts

**Files:**

- Create: `docs/applications/codex-for-oss.md`
- Create: `docs/applications/codex-open-source-fund.md`
- Create: `tests/repository/test_application_claims.py`

**Interfaces:**

- Every repository metric is labelled with verification date.
- Codex for OSS answers respect 500-character limits where specified.
- Personal fields remain clearly marked `USER INPUT REQUIRED` until supplied by
  the user; these markers are permitted only in application drafts and are not
  implementation placeholders.

- [ ] **Step 1: Write failing application tests**

```python
# tests/repository/test_application_claims.py
from pathlib import Path


def _field(text: str, name: str) -> str:
    marker = f"## {name}\\n\\n"
    return text.split(marker, 1)[1].split("\\n## ", 1)[0].strip()


def test_codex_for_oss_limited_answers_fit_500_characters() -> None:
    text = Path("docs/applications/codex-for-oss.md").read_text(encoding="utf-8")
    for name in [
        "Why does this repository qualify?",
        "How will you use API credits for your project?",
        "Anything else we should know?",
    ]:
        assert len(_field(text, name)) <= 500
```

- [ ] **Step 2: Run test and verify missing draft**

Run:

```bash
.venv/bin/python -m pytest tests/repository/test_application_claims.py -v
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Write Codex for OSS draft**

Use these field headings exactly:

```markdown
# Codex for Open Source Application
## First name
## Last name
## Email
## GitHub username
## GitHub repository URL
## Maintainer role
## Why does this repository qualify?
## Interested in
## OpenAI Organization ID
## How will you use API credits for your project?
## Anything else we should know?
## Evidence checked
```

Use:

- GitHub username: `HectorGao`;
- repository: `https://github.com/Poleside/ML4GM`;
- role: `Core maintainer`;
- interests: `Codex Security` and `API credits for my project`.

The three 500-character narrative answers must use the approved, truthful
project positioning and must be re-counted by the test after every edit.

Use these initial answers:

```text
Why does this repository qualify?

ML4GM is an early-stage open scientific toolkit for benchmarking machine-learning models that reconstruct glacier elevation change across 8,101 High Mountain Asia glaciers using RGI, ERA5-Land, and published geodetic observations. It addresses reproducibility, data leakage, and spatial-temporal generalization in climate-impact research.
```

```text
How will you use API credits for your project?

We will use API credits for Codex-assisted maintenance: modularizing research notebooks, generating and reviewing tests, checking reproducibility and data-leakage risks, documenting upstream datasets and licenses, triaging issues, reviewing pull requests, preparing releases, and keeping scientific workflows auditable. OpenAI models will support maintenance, not produce glacier predictions.
```

```text
Anything else we should know?

The repository is being converted from a researcher-specific notebook workspace into a public, tested, configuration-driven toolkit with a small synthetic demo dataset, reproducible benchmarks, data provenance records, contributor documentation, CI, a security policy, and separate licensing for code and third-party scientific data.
```

- [ ] **Step 4: Write Open Source Fund draft**

Use these field headings exactly:

```markdown
# Codex Open Source Fund Application
## First name
## Last name
## Email address
## LinkedIn URL
## GitHub personal
## Anything else you would like us to know
## Project
## Brief description
## GitHub repository
## Other contributors and roles
## How would you use API credits?
## Evidence checked
```

Describe:

- Poleside as project owner;
- HectorGao as core collaborator and applicant;
- a public, leakage-aware glacier ML benchmark toolkit;
- API credits used for maintenance, review, testing, provenance, onboarding,
  release automation, and security;
- no OpenAI-generated glacier predictions.

Use these initial project answers:

```text
Project

ML4GM — Reproducible Machine Learning Benchmarks for Glacier Elevation Change
```

```text
Brief description

ML4GM is an open scientific Python toolkit for comparing Random Forest, LightGBM, MLP, Seasonal-LSTM, and Temporal-LSTM models for glacier elevation-change reconstruction. It provides traceable data preparation and leakage-aware temporal, spatial, and spatiotemporal validation for High Mountain Asia research.
```

```text
Other contributors and roles

Poleside is the project owner and scientific lead. HectorGao is a core collaborator with repository write access, responsible for open-source packaging, testing, CI, documentation, issue and pull-request maintenance, release preparation, and the reproducibility roadmap.
```

```text
How would you use API credits?

We would use Codex and OpenAI API credits to convert legacy research notebooks into reviewed modules, generate and audit tests, detect data leakage and machine-specific paths, maintain data provenance and license records, triage issues, review pull requests, improve contributor onboarding, and automate release and security workflows. OpenAI models would support software maintenance only; glacier predictions remain deterministic scientific models.
```

- [ ] **Step 5: Verify current public metrics**

Run on the submission date:

```bash
gh repo view Poleside/ML4GM --json stargazerCount,forkCount,issues,pullRequests,licenseInfo,defaultBranchRef,viewerPermission
```

Record the date and exact values under `Evidence checked`. Do not convert weak
metrics into adoption claims.

- [ ] **Step 6: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/repository/test_application_claims.py -v
```

Expected: character-limit test passes.

- [ ] **Step 7: Commit**

```bash
git add docs/applications tests/repository/test_application_claims.py
git commit -m "docs: draft OpenAI open-source applications"
```

---

### Task 16: Run the Completion Audit and Prepare v0.1.0

**Files:**

- Modify: `CHANGELOG.md`
- Create: `docs/release-checklist.md`
- Create: `docs/completion-audit.md`

**Interfaces:**

- Completion audit maps every design success criterion to current evidence.
- No release tag is created until the public branch and CI are verified.

- [ ] **Step 1: Write the release checklist**

The checklist must include:

- clean Python 3.11 environment;
- editable install;
- Ruff format and lint;
- complete pytest suite;
- wheel and sdist build;
- quickstart CLI;
- package import;
- forbidden-path scan;
- no restricted data in Git;
- license and governance files;
- application character limits;
- GitHub CI status;
- public repository visibility;
- maintainer permission evidence.

- [ ] **Step 2: Run all local verification commands**

Run:

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m pytest
.venv/bin/python -m build
.venv/bin/ml4gm evaluate --config configs/quickstart.yaml --output-dir /tmp/ml4gm-release
.venv/bin/python -c "import ml4gm; assert ml4gm.__version__ == '0.1.0'"
git grep -n -E 'C:\\\\ML4GM|/Users/|/openbayes/' -- ':!notebooks/legacy/**' ':!docs/superpowers/**'
git status --short
```

Expected:

- all Python commands exit zero;
- forbidden-path search returns no supported-file matches;
- only intentionally untracked or ignored local artifacts appear.

- [ ] **Step 3: Audit tracked data**

Run:

```bash
git ls-files data
git ls-files '*.nc' '*.pkl' '*.pt' '*.joblib'
```

Expected:

- only `data/sample/glacier_sample.csv` is tracked under `data/`;
- no large scientific data or trained artifacts are tracked.

- [ ] **Step 4: Write `docs/completion-audit.md`**

Create a table:

```text
Requirement | Evidence | Status | Remaining action
```

Include all ten success criteria from the design specification. Mark a
criterion complete only when the referenced command, file, CI result, or public
repository state proves it.

- [ ] **Step 5: Update changelog**

Move completed Unreleased entries into:

```markdown
## [0.1.0] - 2026-07-17
```

If implementation completes on another date, use that actual date.

- [ ] **Step 6: Commit release preparation**

```bash
git add CHANGELOG.md docs/release-checklist.md docs/completion-audit.md
git commit -m "chore: prepare the ML4GM 0.1.0 release"
```

- [ ] **Step 7: Push and verify CI only after explicit repository review**

Run:

```bash
git push origin master
gh run list --workflow CI --limit 1
gh run watch
```

Expected: public `master` contains all work and the CI workflow succeeds.

- [ ] **Step 8: Recheck both application drafts against the public repository**

Confirm:

- README, license, CI, maintainer roles, and data documentation are public;
- metrics and permission claims are current;
- personal fields supplied by the user are correct;
- no narrative claims exceed the form limits.

- [ ] **Step 9: Open and fill both forms**

Use the public application pages:

```text
https://openai.com/form/codex-for-oss/
https://openai.com/form/codex-open-source-fund/
```

Populate fields from the reviewed drafts. Do not submit until the user confirms
the final personal details and explicitly authorizes submission.
