# ML4GM

**Reproducible Machine Learning Benchmarks for Glacier Elevation Change**

ML4GM is an early-stage research toolkit for comparing models that reconstruct
annual glacier surface-elevation change (`dhdt`) while enforcing spatial and
temporal leakage controls.

## Why ML4GM exists

Research notebooks are useful for exploration, but machine-specific paths,
implicit preprocessing, and inconsistent validation make scientific comparisons
hard to audit. ML4GM turns the repository's research workflows into a tested
Python package with configuration-driven experiments, deterministic splits, and
machine-readable results.

## Scientific scope

The project focuses on glacier elevation change in High Mountain Asia. It joins
annual glacier targets with topographic and climate predictors, then compares
regression models across held-out years, glaciers, or glacier-year blocks. The
package is an evaluation framework, not a pretrained glacier product.

## Project status

Version 0.1.0 is a reproducible preview. The Random Forest quickstart is
supported on CPU; LightGBM and PyTorch models are optional. Full-data benchmark
claims will be published only after upstream data provenance and redistribution
terms have been verified and the benchmark has been rerun through the package.
Legacy notebooks are preserved as historical research artifacts, not validated
release results.

## Quickstart

Python 3.11 is the supported interpreter. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ml4gm evaluate --config configs/quickstart.yaml
```

The command evaluates a small Random Forest with leave-one-year-out validation
and writes `outputs/quickstart/result.json`. The bundled sample is **synthetic**:
it demonstrates the schema and workflow, but its scores cannot support scientific conclusions.

## Full scientific data

ML4GM does not bundle the upstream scientific datasets. Researchers must obtain
them from their official providers, accept any applicable terms, and retain
their citations. See [DATA_SOURCES.md](DATA_SOURCES.md) for the source registry
and [docs/full-data-setup.md](docs/full-data-setup.md) for the preparation
workflow.

## Models

- Random Forest: supported CPU baseline.
- LightGBM: optional gradient-boosting adapter; install with `.[lightgbm]`.
- Multilayer perceptron: optional tabular PyTorch model; install with `.[torch]`.
- Seasonal LSTM: optional monthly climate-sequence model.
- Temporal LSTM: optional multi-year glacier-history model.

Model dependencies are intentionally optional so the CPU quickstart stays small.

## Validation protocols

ML4GM implements leave-one-year-out (LOYO), glacier-grouped spatial folds, and
strict glacier-and-year block validation. Preprocessing is fitted on each
training fold only. The exact leakage invariants are documented in
[docs/benchmark-protocol.md](docs/benchmark-protocol.md).

## Results and reproducibility

Each evaluation writes an authoritative `result.json` run record. It contains
the resolved portable configuration, ML4GM and Python versions, the exact input
SHA-256 (plus a neighbouring prepared-data manifest when present), data
coverage, effective model parameters, validation details, fold sizes, R²,
RMSE, MAE, and explicit artifact paths. Expected data, dependency, and model
runtime failures also write a record with `status: failed` before the error is
returned to the caller. Configuration files and prepared-data manifests should
still be retained with published results. Synthetic quickstart metrics and
embedded legacy-notebook outputs are not scientific benchmark evidence.

## Data licensing

Project-authored code is licensed under the
[Apache License 2.0](LICENSE). Third-party datasets do not inherit that license.
Their access, attribution, and redistribution conditions remain with their
providers; consult [DATA_SOURCES.md](DATA_SOURCES.md) before adding any data.

## Contributing

Scientific changes should begin with an issue describing the hypothesis,
validation impact, and data provenance. Read [CONTRIBUTING.md](CONTRIBUTING.md)
and the [roadmap](ROADMAP.md) before opening a pull request.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Cite each
upstream dataset separately using the provider citation in
[DATA_SOURCES.md](DATA_SOURCES.md).

## Security

Please report vulnerabilities privately through GitHub Security Advisories as
described in [SECURITY.md](SECURITY.md). Do not disclose a suspected
vulnerability in a public issue.

## Scientific limitations

ML4GM is research software and is not operational guidance for hazards or water
resources. Review [docs/scientific-limitations.md](docs/scientific-limitations.md)
before interpreting or publishing results.
