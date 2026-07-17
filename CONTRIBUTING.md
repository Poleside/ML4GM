# Contributing to ML4GM

Thank you for improving reproducible glacier machine learning. By participating,
you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Use Python 3.11 from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install `.[lightgbm]`, `.[torch]`, or `.[notebooks]` only when your change needs
that optional capability.

## Tests and quality checks

Run the smallest relevant test first, then the full suite:

```bash
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

New behavior and bug fixes require a failing regression test before the
implementation. Keep tests deterministic and CPU-bounded.

## Scientific changes start with an issue

Before changing targets, features, preprocessing, validation, metrics, or model
semantics, open an issue that states:

- the scientific question and expected benefit;
- affected data sources and units;
- leakage risks and the proposed validation design;
- compute and dependency impact;
- evidence that would support or reject the change.

Do not present historical notebook numbers as package-reproduced results.

## Data policy

Do not commit raw upstream datasets, large derived tables, trained weights, or
credentials. Review [DATA_SOURCES.md](DATA_SOURCES.md), document provenance and
checksums, and obtain maintainer confirmation of redistribution rights before
proposing any real-data fixture. Synthetic fixtures must be unmistakably
labelled and must not mimic a claim of observed scientific values.

## Pull request checklist

- [ ] The issue and intended scientific behavior are linked.
- [ ] Tests failed for the intended reason before the implementation and now pass.
- [ ] Full tests, lint, and formatting checks pass.
- [ ] Documentation, units, configuration, and changelog are updated.
- [ ] No machine-specific paths, secrets, generated outputs, or unapproved data are included.
- [ ] Data citations, licences, and redistribution status are recorded.
- [ ] Validation keeps test glaciers and/or years out of every fitted transform and model.

Maintainers may request scientific review in addition to code review.
