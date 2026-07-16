# Task 7 Report: Dataset Preparation and Manifest Writing

## Status

Complete.

## Implementation

- Added `prepare_dataset` in `src/ml4gm/data/prepare.py`.
- Validates and deterministically sorts annual glacier data through
  `validate_annual_table`.
- Writes the prepared CSV before constructing the manifest.
- Passes the prepared output path to `DatasetManifest.from_frame`, so the
  manifest SHA-256 identifies the written prepared artifact rather than the raw
  input.
- Publicly exports `prepare_dataset` from `ml4gm.data`.

## TDD Evidence

### RED

Command:

```text
.venv/bin/python -m pytest tests/integration/test_prepare.py -v
```

Result: collection failed with the expected `ImportError` because
`prepare_dataset` was not yet exported from `ml4gm.data`.

### GREEN

Command:

```text
.venv/bin/python -m pytest tests/integration/test_prepare.py -v
```

Result: `1 passed`.

The integration test verifies sorting, 72 rows, source provenance, prepared
output identity, and the SHA-256 of the written prepared CSV.

## Verification

```text
.venv/bin/python -m pytest -v
32 passed in 1.22s

.venv/bin/python -m ruff check src/ml4gm/data/prepare.py src/ml4gm/data/__init__.py tests/integration/test_prepare.py
All checks passed!

git diff --check
passed with no output
```

## Self-review

- Scope is limited to dataset preparation, its integration test, and public
  export.
- No CLI work or unrelated Ruff cleanup was included.
- Parent directories are created for both prepared data and manifest output.
- Missing input paths fail early with a path-specific `FileNotFoundError`.
- No concerns remain within Task 7 scope.
