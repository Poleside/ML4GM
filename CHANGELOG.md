# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Preparing 0.1.0

- Continuous-integration, issue, and pull-request community infrastructure.
- Installable `ml4gm` package and configuration-driven command-line interface.
- Synthetic CPU quickstart and deterministic dataset manifests.
- Random Forest, optional LightGBM, MLP, seasonal LSTM, and temporal LSTM models.
- LOYO, spatial GroupKFold, and strict block validation.
- Supported tutorial with preserved legacy research notebooks.
- Apache-2.0 licensing, governance, data-source registry, and scientific guidance.

### Release gating

Version 0.1.0 remains unreleased. Do not add a release date or create the
`v0.1.0` tag until the preparation branch is public, every GitHub Actions job
has passed on that public revision, the historical prediction artifact
documented in `docs/completion-audit.md` has received a data-rights decision,
and the maintainers have approved the release.
