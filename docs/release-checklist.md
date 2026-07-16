# ML4GM 0.1.0 Release Checklist

This is a release gate, not a statement that 0.1.0 has been released. Evidence
was refreshed on 2026-07-17 from commit `1d304ca` plus the Task 16 working tree.
Re-run every command against the exact revision proposed for release.

## Local reproducibility

- [x] **clean Python 3.11 environment** — created
  `/tmp/ml4gm-release-venv` with Python 3.11.15.
- [x] **editable install** — `python -m pip install -e .` completed in that
  clean environment using only the declared base dependencies.
- [x] **Ruff format and lint** — `.venv/bin/ruff format --check .` and
  `.venv/bin/ruff check .` exit zero after applying the one formatting change
  reported by the first check.
- [x] **complete pytest suite** — 164 tests passed and one optional LightGBM
  test skipped because the local x86_64 Python cannot load the installed arm64
  `libomp`. The public CI matrix must run the real LightGBM round trip.
- [x] **wheel and sdist build** — `.venv/bin/python -m build` produced
  `ml4gm-0.1.0-py3-none-any.whl` and `ml4gm-0.1.0.tar.gz`.
- [x] **quickstart CLI** — both the development environment and clean
  environment completed `ml4gm evaluate --config configs/quickstart.yaml`.
- [x] **package import** — both environments verified
  `ml4gm.__version__ == "0.1.0"`.

## Repository safety and rights

- [x] **forbidden-path scan** — the supported package, configurations, root
  documentation, and non-application documentation contain no match for any of
  the three machine-specific roots defined by
  `tests/repository/test_forbidden_paths.py`.
- [x] The full-tree scan is classified rather than silently ignored:
  `compare/` contains historical machine-specific paths;
  `notebooks/legacy/` contains preserved hosted-notebook output;
  `tests/repository/` contains the literal denial-list test fixtures;
  `docs/superpowers/` contains specification examples. None is a supported
  runtime path.
- [x] Current-tree **no restricted data in Git** check:
  `git ls-files data` returns only `data/sample/glacier_sample.csv`, which is
  labelled synthetic, and `git ls-files '*.nc' '*.pkl' '*.pt' '*.joblib'`
  returns nothing.
- [ ] Historical-data decision — Git history contains
  `code/model_result1.npz` from commit `e3c9184`. It holds eight arrays of
  32,404 target/prediction values and is no longer in the tracked tree. Before
  release, Poleside must confirm that retaining this already-public derived
  result in history is permitted, or approve a coordinated history rewrite.
- [x] Current-tree heuristic secret scan found no common API token, private-key,
  password-assignment, or API-key-assignment pattern outside the separated
  legacy/specification paths.
- [ ] Run the configured public dependency and secret/security checks. Local
  heuristic scans are not a substitute for public CI or GitHub security
  features.

## Project and application metadata

- [x] **license and governance files** — `LICENSE`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `MAINTAINERS.md`, `CITATION.cff`,
  `DATA_SOURCES.md`, and `ROADMAP.md` exist locally.
- [x] **application character limits** — the three limited Codex for Open
  Source answers are 425, 437, and 409 characters, within the 500-character
  limit; repository tests enforce the limit.
- [ ] Resolve every `USER INPUT REQUIRED` field in both application drafts:
  applicant name, email, OpenAI Organization ID, and LinkedIn URL where used.
- [ ] Recheck all application wording after the branch is public. The current
  drafts explicitly say that the preparation branch is not yet public.

## Public release gates

- [ ] **GitHub CI status** — the public repository currently has no workflow
  named `CI`; push or merge the reviewed branch, then require every job in
  `.github/workflows/ci.yml` to pass on the public revision.
- [x] **public repository visibility** — `gh repo view` reports
  `Poleside/ML4GM` is public.
- [ ] Public release contents — the public `master` commit is `2d4bf00`; it
  does not contain this preparation branch and GitHub reports no detected
  license.
- [x] **maintainer permission evidence** — GitHub reports `HectorGao` has
  `WRITE` viewer permission, consistent with `MAINTAINERS.md`.
- [ ] Private vulnerability reporting — the GitHub endpoint reports
  `"enabled": false`; Poleside should enable it, or the maintainers must retain
  and approve the fallback channel documented in `SECURITY.md`.
- [ ] Maintainer review and explicit release approval.
- [ ] Create and push `v0.1.0` only after every preceding release gate is
  closed. Until then, keep the changelog under `[Unreleased]`.
- [ ] Re-open both forms, fill the approved personal details, obtain explicit
  user authorization, and only then submit.

## Commands to refresh

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/python -m pytest
.venv/bin/python -m build
.venv/bin/ml4gm evaluate --config configs/quickstart.yaml --output-dir /tmp/ml4gm-release
.venv/bin/python -c "import ml4gm; assert ml4gm.__version__ == '0.1.0'"
.venv/bin/python -m pytest tests/repository/test_forbidden_paths.py
git ls-files data
git ls-files '*.nc' '*.pkl' '*.pt' '*.joblib'
git status --short
gh repo view Poleside/ML4GM --json visibility,licenseInfo,defaultBranchRef,viewerPermission
gh run list --repo Poleside/ML4GM --workflow CI --limit 1
gh api repos/Poleside/ML4GM/private-vulnerability-reporting
```

The supported forbidden-path command is expected to return exit status 1 with
no output because Git found no matches. Record that result as success; do not
replace it with a full-tree assertion that mistakes denial-list fixtures or
preserved legacy evidence for supported runtime code.
