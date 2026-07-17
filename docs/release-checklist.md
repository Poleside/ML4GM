# ML4GM 0.1.0 Release Checklist

This is a release gate, not a statement that 0.1.0 has been released. Evidence
was refreshed on 2026-07-17 against the complete reviewed working tree.
Re-run every command against the exact committed revision proposed for release.

## Local reproducibility

- [x] **clean Python 3.11 environment** — created
  `/tmp/ml4gm-release-venv` with Python 3.11.15.
- [x] **editable install** — `python -m pip install -e .` completed in that
  clean environment using only the declared base dependencies.
- [x] **Ruff format and lint** — `.venv/bin/ruff format --check .` reported 44
  files formatted and `.venv/bin/ruff check .` exited zero.
- [x] **complete pytest suite** — 177 tests passed and one optional LightGBM
  test skipped because the local x86_64 Python cannot load the installed arm64
  `libomp`. The public CI matrix must run the real LightGBM round trip.
- [x] **wheel and sdist build** — `.venv/bin/python -m build` produced
  `ml4gm-0.1.0-py3-none-any.whl` and `ml4gm-0.1.0.tar.gz`.
- [x] **quickstart CLI** — both the development environment and clean
  environment completed `ml4gm evaluate --config configs/quickstart.yaml`.
- [x] **package import** — both environments verified
  `ml4gm.__version__ == "0.1.0"`.
- [x] **authoritative run record** — integration and unit tests verify that
  result records contain the resolved configuration, runtime/package versions,
  coverage and metrics, immutable input SHA-256, and only a matching manifest
  parsed and hashed from one byte snapshot. Failure records preserve provenance
  when possible without masking the original exception.

## Repository safety and rights

- [x] **forbidden-path scan** — the supported package, configurations, root
  documentation, and non-application documentation contain no match for any of
  the three machine-specific roots defined by
  `tests/repository/test_forbidden_paths.py`.
- [x] The Task 16 brief's original scan was run exactly. It returned exit
  status 0 and matches in `compare/` historical reference scripts and
  `tests/repository/` denial-list literals. This is a classified inventory,
  not a supported-code assertion. The command already excludes
  `notebooks/legacy/` and `docs/superpowers/`. On the audit-only follow-up
  worktree, it additionally matches the two command literals below; that
  self-reference does not change the source-revision classification.
- [x] A second, explicit supported-scope scan returned exit status 1 with no
  output. That no-match result is the supported-code assertion.
- [x] Current-tree **no restricted data in Git** check:
  `git ls-files data` returns only `data/sample/glacier_sample.csv`, which is
  labelled synthetic, and `git ls-files '*.nc' '*.pkl' '*.pt' '*.joblib'`
  returns nothing.
- [ ] Historical-data decision — Git history contains
  `code/model_result1.npz` from commit `e3c9184`. It holds eight arrays of
  32,404 target/prediction values and is no longer in the tracked tree. The
  object is already in the public `origin/master` ancestry, so this finding
  does not block publishing or reviewing the preparation branch. It does block
  the `v0.1.0` tag/release until Poleside confirms the rights decision. Do not
  rewrite public history without explicit owner authorization and coordination.
- [x] Current-tree heuristic secret scan found no common API token, private-key,
  password-assignment, or API-key-assignment pattern outside the separated
  legacy/specification paths.
- [x] Dependency and secret-scanning gates are configured in
  `.github/workflows/ci.yml`: `dependency-audit` installs the project and runs
  blocking `pip-audit`, while `secret-scan` checks the complete Git history
  with the official Gitleaks action. Repository tests verify their commands,
  permissions, and checkout depth.
- [ ] Run the configured dependency and secret-scanning gates on the public
  revision. Local configuration and heuristic scans are not substitutes for a
  successful public CI run or GitHub security features.

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
  named `CI`. After maintainer review, push the feature branch and open a draft
  pull request to trigger CI. Require every job in `.github/workflows/ci.yml`
  to pass before merging to `master`.
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
- [ ] Maintainer review before pushing the feature branch/opening a draft PR;
  separate explicit approval is required before merge and release.
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
# Original Task 16 inventory: expected exit 0 with classified legacy/test matches.
git grep -n -E 'C:\\ML4GM|/Users/|/openbayes/' -- ':!notebooks/legacy/**' ':!docs/superpowers/**'
# Supported assertion: expected exit 1 and no output.
git grep -n -E 'C:\\ML4GM|/Users/|/openbayes/' -- src configs README.md DATA_SOURCES.md CONTRIBUTING.md SECURITY.md MAINTAINERS.md docs ':!docs/superpowers/**' ':!docs/applications/**' ':!docs/release-checklist.md' ':!docs/completion-audit.md'
.venv/bin/python -m pytest tests/repository/test_forbidden_paths.py
git ls-files data
git ls-files '*.nc' '*.pkl' '*.pt' '*.joblib'
git status --short
gh repo view Poleside/ML4GM --json visibility,licenseInfo,defaultBranchRef,viewerPermission
gh run list --repo Poleside/ML4GM --workflow CI --limit 1
gh api repos/Poleside/ML4GM/private-vulnerability-reporting
```

Record both results. The original command's exit status 0 is correct because it
inventories preserved reference scripts and test literals. The supported-scope
command's exit status 1 with no output is correct because Git found no match.
Do not conflate these two different claims.

After maintainer review, pushing this feature branch and opening a draft PR is
allowed and is how public CI evidence should be collected. Do not merge to
`master`, create or push `v0.1.0`, publish a release, add a dated 0.1.0
changelog heading, or submit either form until the corresponding gates close.
