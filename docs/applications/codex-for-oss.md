# Codex for Open Source Application

Copy only the answer beneath each heading into the matching form field. Refresh
the repository snapshot immediately before submission because the public
repository does not yet contain this open-source preparation branch.

## First name

USER INPUT REQUIRED

## Last name

USER INPUT REQUIRED

## Email

USER INPUT REQUIRED

## GitHub username

HectorGao

## GitHub repository URL

https://github.com/Poleside/ML4GM

## Maintainer role

Core maintainer

## Why does this repository qualify?

ML4GM is an early-stage open scientific toolkit for reproducible, leakage-aware machine-learning evaluation of glacier elevation change in High Mountain Asia. Its preserved research outputs cover 8,101 glaciers and 162,020 glacier-year rows from 2000–2019. The project is important because spatial and temporal leakage can overstate model performance; ML4GM makes splits, preprocessing, provenance, and limitations auditable.

## Interested in

Codex Security

API credits for my project

## OpenAI Organization ID

USER INPUT REQUIRED

## How will you use API credits for your project?

We will use API credits for Codex-assisted maintenance: converting legacy notebooks into reviewed modules, generating and auditing tests, checking reproducibility and data-leakage risks, documenting upstream data rights, triaging issues, reviewing pull requests, improving contributor onboarding, and automating releases and security workflows. OpenAI models will support software maintenance; they will not generate glacier predictions.

## Anything else we should know?

ML4GM is being converted from a researcher-specific notebook workspace into a public, tested, configuration-driven Python toolkit. The repository now has a synthetic quickstart, reproducible model adapters, provenance records, contributor and security policies, CI, and Apache-2.0 code licensing. Third-party scientific data remain separately governed and are not redistributed until their terms are verified.

## Evidence checked

- Form checked 2026-07-17:
  <https://openai.com/form/codex-for-oss/>. The live form requires first name,
  last name, ChatGPT-account email, public GitHub username and repository, a
  primary/core-maintainer choice, OpenAI Organization ID, and the three fields
  reproduced above. Each of the three narrative fields has a 500-character
  maximum.
- Verified 2026-07-17 public submission-preparation snapshot: 1 star, 0 forks,
  0 open issues, 0 open pull requests, 0 releases, default branch `master`,
  viewer permission `WRITE`, GitHub-detected license: none. These values are a
  point-in-time snapshot, not evidence of adoption.
- Snapshot commands:
  `gh repo view Poleside/ML4GM --json stargazerCount,forkCount,issues,pullRequests,licenseInfo,defaultBranchRef,viewerPermission,latestRelease`;
  `gh api repos/Poleside/ML4GM`;
  `gh api 'repos/Poleside/ML4GM/issues?state=open&per_page=100'`;
  `gh api 'repos/Poleside/ML4GM/pulls?state=open&per_page=100'`; and
  `gh api repos/Poleside/ML4GM/releases`.
- The public repository snapshot reports no detected license because this
  Apache-2.0 transformation is still on an unpushed preparation branch. Refresh
  the metrics and confirm the public license before submitting.
- Repository authority is documented in `MAINTAINERS.md`: Poleside is project
  lead/owner; HectorGao is a core collaborator with write access. The live
  GitHub query independently returned viewer permission `WRITE`.
- The scale statement is historical, not a new package benchmark. Preserved
  legacy notebook outputs in `notebooks/legacy/readWW.ipynb`,
  `notebooks/legacy/01_dependencies_and_data.ipynb`, and
  `notebooks/legacy/02_rf.ipynb` report 8,101 glaciers, 2000–2019, and 162,020
  glacier-year rows. As `README.md` states, they are not validated release results.
- Current supported scope was checked against `README.md`,
  `docs/benchmark-protocol.md`, `DATA_SOURCES.md`, `SECURITY.md`, `ROADMAP.md`,
  and `pyproject.toml`. No usage or adoption claim is made.
