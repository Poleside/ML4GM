## Summary

Describe the scientific or maintenance change and its user-visible effect.

## Validation

List the commands, datasets, splits, and metrics used to validate the change.

## Checklist

- [ ] Tests were added or updated.
- [ ] Glacier, year, spatial, and temporal leakage risks were considered.
- [ ] No machine-specific absolute paths were added.
- [ ] No restricted data, credentials, or data without redistribution rights were added.
- [ ] Data rights, provenance, and license implications were documented.
- [ ] User-facing documentation and the changelog were updated when needed.
- [ ] `ruff format --check .`, `ruff check .`, and `pytest` pass.
- [ ] The quickstart CLI completes successfully.
