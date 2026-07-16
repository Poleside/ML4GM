# ML4GM Open-Source Transformation Completion Audit

Audit date: 2026-07-17

This audit maps all ten success criteria in the approved design to current
evidence. “Complete locally” does not mean “published” or “released.” Public
branch and CI are not yet verified, so the transformation and applications
remain pending.

The package, workflow, application-draft, and governance source under test is
revision `514d7f2`. The follow-up prompted by the Task 16 review changes only
this audit, the release checklist, and their repository test. Fresh verification
therefore covers the resulting audit-only worktree while keeping the underlying
source revision explicit.

| Criterion | Requirement | Evidence | Status | Remaining action |
| --- | --- | --- | --- | --- |
| 1 | Repository is publicly licensed under Apache-2.0. | Local `LICENSE` and `pyproject.toml` declare Apache-2.0. GitHub reports the repository is public but its public `master` is still `2d4bf00` with `licenseInfo: null`. | **Pending public publication** | Review and publish the preparation branch; verify GitHub detects Apache-2.0. |
| 2 | A new user can install from a clean Python 3.11 environment. | A fresh Python 3.11.15 venv at `/tmp/ml4gm-release-venv` completed `pip install -e .`, imported version 0.1.0, and ran the CLI. | **Complete locally** | Repeat in public CI and from the reviewed release revision. |
| 3 | CPU quickstart completes without private paths or unavailable raw data. | The quickstart completed in both local environments using the tracked synthetic CSV. Supported-code forbidden-path scan had no matches. | **Complete locally** | Confirm the same command passes in public CI. |
| 4 | CI validates linting, tests, packaging, dependencies, and secrets. | `.github/workflows/ci.yml` defines quality, package, LightGBM, PyTorch, blocking dependency-audit, and full-history secret-scan jobs. Repository metadata tests verify the security jobs, but `gh run list --workflow CI` reports no public workflow because the branch is not published. | **Pending public CI** | Push or merge after review, then require every public CI job, including both security gates, to pass. |
| 5 | Leakage invariants are covered by automated tests. | The 167 passing tests include LOYO, spatial, block, fold-local preprocessing, and sequence-context isolation tests. | **Complete locally** | Reconfirm on the exact public revision. |
| 6 | Upstream provenance and redistribution boundaries are documented. | `DATA_SOURCES.md`, `docs/full-data-setup.md`, and the synthetic-data statements document acquisition and redistribution boundaries. The historical `code/model_result1.npz` is already present in the public `origin/master` ancestry. | **Complete locally; release decision pending** | It does not block publishing/reviewing this preparation branch, but Poleside must decide its rights status before the v0.1.0 tag/release. Do not rewrite public history without explicit owner authorization. |
| 7 | Legacy notebooks are preserved and separated from supported APIs. | Historical notebooks are under `notebooks/legacy/`; supported execution is under `src/ml4gm/` and `notebooks/tutorials/`. Repository tests verify this separation. | **Complete locally** | Preserve the separation in the published revision. |
| 8 | Governance, security, contribution, citation, and maintenance documents exist. | `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff`, and `MAINTAINERS.md` exist and repository tests pass. Private vulnerability reporting currently returns `"enabled": false`, with a documented maintainer-profile fallback. | **Complete locally; security enhancement pending** | Poleside should enable private vulnerability reporting or explicitly approve the fallback before release. |
| 9 | Public repository state supports every factual claim in both applications. | Draft claims are repository-tested and limited answers are under 500 characters. The drafts themselves disclose that the preparation branch is not public; public GitHub currently has no detected license or CI. | **Pending public evidence** | Publish, pass CI, refresh metrics/license/permissions, and re-review every claim. |
| 10 | Both forms are filled with approved personal details and submitted or ready for final user action. | Both drafts exist, but applicant name, email, OpenAI Organization ID, and LinkedIn fields still contain `USER INPUT REQUIRED`; no form was modified or submitted in Task 16. | **Pending user input and authorization** | Supply and approve personal fields, recheck the live forms, then explicitly authorize filling/submission. |

## Local verification evidence

- Ruff: 43 files formatted; lint passed.
- Tests: 167 passed, one LightGBM runtime test skipped due to the local
  x86_64-Python/arm64-`libomp` mismatch.
- Packaging: wheel and sdist built successfully.
- Quickstart: six LOYO folds completed in both the development and clean Python
  environments.
- Tracked data: only `data/sample/glacier_sample.csv`; no tracked `.nc`, `.pkl`,
  `.pt`, or `.joblib` files.
- Current-tree large-file review: the largest tracked files are preserved
  legacy notebooks. No model weights or raw scientific dataset is in the
  current tree.
- Historical-object review: commit `e3c9184` added the now-removed
  `code/model_result1.npz` blob containing target and prediction arrays. Its
  data-rights status is not proven by local inspection. Because it is already
  in the public upstream ancestry, this is a v0.1.0 release gate rather than a
  preparation-branch publication blocker.
- Original forbidden-path inventory: exit status 0, with `compare/` legacy
  reference paths and `tests/repository/` denial-list literals on source
  revision `514d7f2`. The audit-only follow-up also self-matches the command
  text recorded in the release checklist.
- Supported-scope forbidden-path assertion: exit status 1 with no output.
- Secret review: heuristic current-tree and history pattern searches found no
  common token, private-key, password, or API-key assignment pattern. This is
  supporting evidence, not a guarantee or a replacement for public scanning.
  Blocking `pip-audit` and full-history Gitleaks jobs are now configured with
  minimal permissions; their first public run remains pending publication.

## Release decision

Do not create `v0.1.0`, add a dated 0.1.0 changelog heading, push this branch,
or submit either form based on this local audit alone. The remaining actions
require repository-owner review, public CI evidence, personal details, and
explicit user authorization.
