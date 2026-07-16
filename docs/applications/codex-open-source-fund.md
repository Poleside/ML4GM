# Codex Open Source Fund Application

Copy only the answer beneath each heading into the matching form field. The
headings below map the repository plan's short labels to the live questions
checked on 2026-07-17:

- `Anything else you would like us to know` maps to “Is there anything else
  you’d like us to know?”
- `Project` maps to “Which open source project are you representing?”
- `Other contributors and roles` maps to the live question asking for other
  people and their roles.

The live page does not display character limits for these answers.

## First name

USER INPUT REQUIRED

## Last name

USER INPUT REQUIRED

## Email address

USER INPUT REQUIRED

## LinkedIn URL

USER INPUT REQUIRED

## GitHub personal

https://github.com/HectorGao

## Anything else you would like us to know

I am applying as ML4GM's core collaborator, with repository write access and
authorization from project owner Poleside to carry out the Apache-2.0
open-source transformation. The project is early-stage research software; this
application does not claim established usage or a validated full-data release.

## Project

ML4GM — Reproducible Machine Learning Benchmarks for Glacier Elevation Change

## Brief description

ML4GM is an open scientific Python toolkit for comparing Random Forest,
LightGBM, MLP, Seasonal-LSTM, and Temporal-LSTM models for glacier
elevation-change reconstruction. It provides traceable data preparation and
leakage-aware temporal, spatial, and spatiotemporal validation for High Mountain
Asia research.

## GitHub repository

https://github.com/Poleside/ML4GM

## Other contributors and roles

Poleside is the project owner and scientific lead. HectorGao is the applicant
and core collaborator with repository write access, contributing open-source
packaging, testing, CI, documentation, issue and pull-request maintenance,
release preparation, and the reproducibility roadmap.

## How would you use API credits?

We would use Codex and OpenAI API credits to convert legacy research notebooks
into reviewed modules, generate and audit tests, detect data leakage and
machine-specific paths, maintain data provenance and license records, triage
issues, review pull requests, improve contributor onboarding, and automate
release and security workflows. OpenAI models would support software maintenance
only; glacier predictions remain deterministic scientific models.

## Evidence checked

- Form checked 2026-07-17:
  <https://openai.com/form/codex-open-source-fund/>. The live form asks for
  first name, last name, email address, optional LinkedIn URL, personal GitHub,
  optional additional information, project name, brief description, repository,
  other people and roles, and use of API credits. It does not display character
  limits.
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
- Roles and responsibilities were checked against `MAINTAINERS.md`. Project
  scope and maintenance needs were checked against `README.md`,
  `docs/benchmark-protocol.md`, `DATA_SOURCES.md`, `SECURITY.md`, `ROADMAP.md`,
  and `pyproject.toml`.
- Preserved legacy notebook outputs document a historical 8,101-glacier,
  162,020-row, 2000–2019 research dataset, but those outputs are not validated
  release results and are not presented as a Fund adoption claim.
