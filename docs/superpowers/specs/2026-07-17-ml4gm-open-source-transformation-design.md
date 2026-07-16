# ML4GM Open-Source Transformation Design

**Date:** 2026-07-17

**Status:** Approved design

**Repository:** `Poleside/ML4GM`

**Maintainers:** Poleside, HectorGao

**Code license:** Apache License 2.0

## 1. Purpose

ML4GM will become a reproducible open scientific toolkit for benchmarking
machine-learning methods that reconstruct glacier elevation change (`dhdt`) in
High Mountain Asia.

The current repository contains research notebooks and comparison scripts for:

- Random Forest;
- LightGBM and legacy XGBoost comparisons;
- multilayer perceptrons;
- seasonal LSTM models using monthly climate sequences;
- temporal LSTM models using multi-year glacier histories;
- leave-one-year-out, spatial, and combined spatiotemporal validation.

The transformed project will preserve this scientific scope while replacing
machine-specific paths, duplicated notebook implementations, incomplete
dependencies, and implicit data assumptions with documented, testable, and
configuration-driven software.

## 2. Product Positioning

The public project name and description will be:

> **ML4GM — Reproducible Machine Learning Benchmarks for Glacier Elevation
> Change**

Its primary value is not a single trained model. It is a shared evaluation
framework for determining how glacier models generalize:

1. across years;
2. across glaciers;
3. across unseen glacier-and-year combinations.

The project will initially describe itself as an early-stage research software
preview. It will not claim broad adoption, production readiness, or validated
operational forecasting.

## 3. Users and Use Cases

### 3.1 Primary users

- glacier and climate researchers comparing regression methods;
- scientific software contributors improving reproducibility;
- students learning leakage-aware spatial and temporal validation;
- maintainers reviewing changes to data and model pipelines.

### 3.2 Supported use cases

- run a CPU-friendly demonstration from a clean environment;
- prepare an externally downloaded glacier dataset into the ML4GM schema;
- train one supported model from a configuration file;
- evaluate a model with LOYO, spatial GroupKFold, or block validation;
- compare metrics through a consistent results format;
- reproduce a documented benchmark when the full upstream data are available.

### 3.3 Non-goals for version 0.1.0

- distributing upstream datasets without confirmed redistribution permission;
- presenting historical notebook output as a newly reproduced benchmark;
- operational glacier, flood, water-resource, or hazard forecasting;
- using OpenAI models to generate glacier predictions;
- hosting a web application or managed training service;
- reproducing every exploratory cell from the legacy notebooks.

## 4. Repository Architecture

The target repository structure is:

```text
ML4GM/
├── src/ml4gm/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── sources.py
│   │   ├── manifest.py
│   │   └── preprocessing.py
│   ├── validation/
│   │   ├── loyo.py
│   │   ├── spatial.py
│   │   └── block.py
│   ├── models/
│   │   ├── base.py
│   │   ├── random_forest.py
│   │   ├── lightgbm.py
│   │   ├── mlp.py
│   │   ├── seasonal_lstm.py
│   │   └── temporal_lstm.py
│   └── evaluation/
│       ├── metrics.py
│       ├── results.py
│       └── runner.py
├── configs/
│   ├── quickstart.yaml
│   └── benchmark.yaml
├── notebooks/
│   ├── tutorials/
│   └── legacy/
├── data/
│   └── sample/
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
├── .github/
└── pyproject.toml
```

Each module will have one defined responsibility. Notebooks will call the
package API rather than retain the only implementation of a workflow.

Legacy notebooks will be preserved under `notebooks/legacy/` so the research
history remains available. They will be clearly marked as legacy artifacts and
will not be presented as the recommended interface.

## 5. Data Flow

The supported pipeline is:

```text
upstream data acquired by the user
    -> source and license metadata validation
    -> ML4GM tabular schema validation
    -> deterministic preprocessing
    -> leakage-aware split generation
    -> model training
    -> metric and artifact collection
    -> comparison report
```

The command-line interface will expose four primary command groups:

```text
ml4gm data prepare
ml4gm train
ml4gm evaluate
ml4gm benchmark
```

Every command will accept a configuration file. Paths will be relative or
explicit command-line values; no `C:\ML4GM`, user home directory, or hosted
notebook path will be embedded in the implementation.

## 6. Configuration and Experiment Records

Configuration will define:

- input and output locations;
- target and identifier columns;
- selected feature groups;
- model type and hyperparameters;
- validation strategy;
- random seed;
- compute limits;
- run name.

Each run will produce a machine-readable record containing:

- resolved configuration;
- package and Python versions;
- input manifest identifier;
- row and feature counts;
- glacier and year coverage;
- validation strategy;
- model parameters;
- R2, RMSE, and MAE;
- generated artifact paths;
- completion or failure status.

This record is the authoritative description of a run. Embedded notebook output
is historical context only.

## 7. Data Governance

### 7.1 Code and data licensing

Project-authored code will use Apache License 2.0 with authorization from the
repository owner.

Third-party scientific data will not inherit the code license. The repository
will include `DATA_SOURCES.md` with, for each source:

- source name and provider;
- dataset version;
- DOI or stable landing page;
- access instructions;
- license or terms;
- required attribution;
- redistribution status;
- subset used by ML4GM.

The initial source registry will cover:

- Randolph Glacier Inventory;
- ERA5 and ERA5-Land;
- Hugonnet et al. glacier elevation or mass-change products.

### 7.2 Repository data policy

Raw upstream datasets, large processed tables, trained model weights, and run
outputs will be ignored by Git by default.

The quickstart will use either:

1. a small real subset whose redistribution permission has been confirmed and
   whose attribution is included; or
2. synthetic data that matches the schema but is explicitly labelled as
   unsuitable for scientific conclusions.

No ambiguous third-party data will be committed merely to make the example run.

### 7.3 Data manifest

Prepared datasets will have a manifest containing:

- source dataset names and versions;
- source file checksums;
- processing configuration;
- output checksum;
- row count;
- glacier count;
- year range;
- feature names;
- missing-value summary;
- creation timestamp.

## 8. Data Schema and Validation

The common annual glacier table will require:

- `rgiid`: glacier identifier;
- `year`: integer target year;
- `dhdt`: elevation-change target in metres per year;
- documented static topographic features;
- documented monthly or annual climate features used by a selected model.

Schema validation will reject:

- missing identifiers, years, or targets;
- duplicate glacier-year rows unless an explicit aggregation rule is chosen;
- non-numeric target values;
- invalid or empty year ranges;
- feature sets that do not satisfy the selected model;
- non-finite values after preprocessing.

Errors will state the failing field, the observed condition, and the required
correction.

## 9. Validation Strategies and Leakage Protection

### 9.1 Leave-one-year-out

Each fold will reserve one complete target year for evaluation. Tests will
verify that the held-out year does not appear in training rows.

### 9.2 Spatial GroupKFold

Glacier identifiers will define groups. Tests will verify that no glacier ID is
shared between training and evaluation rows in a fold.

### 9.3 Spatiotemporal block validation

Glaciers and years will each be assigned to groups. Evaluation blocks will be
isolated by excluding the corresponding glacier group and year group from the
training data. Tests will verify both invariants.

### 9.4 Preprocessing isolation

Scalers, imputers, and learned preprocessing state will be fitted on training
rows only and applied to evaluation rows afterward.

## 10. Model Interfaces

Every supported model adapter will implement a small common interface:

```python
fit(X_train, y_train) -> model
predict(X) -> ndarray
save(path) -> None
```

Model-specific sequence construction will occur before the adapter boundary and
will preserve glacier and target-year identifiers for validation.

Random Forest will be the required quickstart model because it can run reliably
on CPU with a small dataset. LightGBM, MLP, Seasonal-LSTM, and Temporal-LSTM
will be optional benchmark components with dependency groups where appropriate.

Legacy XGBoost scripts will be retained as reference material unless their
behavior is explicitly migrated and tested.

## 11. Error Handling

Expected user-facing failures include:

- missing upstream data;
- missing dependency for an optional model;
- invalid configuration;
- invalid schema;
- insufficient years or glaciers for a selected split;
- empty training or evaluation fold;
- output path conflicts;
- unsupported model name.

The CLI will return a non-zero status and a concise corrective message. It will
not suppress warnings globally or continue with partially valid data.

## 12. Testing and Continuous Integration

Python 3.11 will be the first supported runtime.

### 12.1 Unit tests

Unit tests will cover:

- configuration loading and validation;
- schema validation;
- deterministic preprocessing;
- dataset manifests and checksums;
- R2, RMSE, and MAE;
- LOYO split invariants;
- spatial split invariants;
- block split invariants;
- result serialization.

### 12.2 Integration tests

A small demonstration dataset will run through:

```text
prepare -> train Random Forest -> evaluate -> write result
```

The integration test will verify reproducibility for a fixed seed and validate
the result schema rather than require an unrealistically precise scientific
score.

### 12.3 Optional-model smoke tests

Deep-learning models will receive small-batch construction and forward-pass
tests. Full hyperparameter searches and 20-fold benchmarks will not run in
ordinary CI.

### 12.4 Static and repository checks

CI will run:

- Ruff formatting and linting;
- pytest;
- package build;
- notebook JSON and import/path checks;
- secret scanning;
- dependency vulnerability scanning where the selected GitHub workflow permits.

## 13. Open-Source Governance

The repository will include:

- `README.md`;
- `LICENSE`;
- `CONTRIBUTING.md`;
- `CODE_OF_CONDUCT.md`;
- `SECURITY.md`;
- `MAINTAINERS.md`;
- `CITATION.cff`;
- `DATA_SOURCES.md`;
- issue templates;
- pull-request template;
- release and versioning guidance;
- a public roadmap.

`MAINTAINERS.md` will describe Poleside as the project owner and HectorGao as a
core collaborator with repository write access, along with their actual
responsibilities.

The first release will be `v0.1.0`, described as a reproducible research
software preview.

## 14. Documentation

The root README will include:

- the scientific question;
- supported models and validation strategies;
- project maturity;
- a CPU quickstart;
- full-data setup;
- data licensing boundaries;
- result interpretation;
- contribution and citation links.

Detailed documentation will cover:

- data schema;
- upstream data acquisition;
- configuration;
- benchmark protocols;
- adding a model;
- adding a data source;
- scientific limitations;
- release maintenance.

## 15. Codex and OpenAI Usage

OpenAI services will support project maintenance, not scientific target
generation.

Planned uses of API credits include:

- modularizing legacy notebooks;
- generating and reviewing tests;
- detecting path, dependency, and data-leakage regressions;
- reviewing pull requests;
- triaging issues;
- maintaining data-source and license documentation;
- drafting changelogs and release notes;
- improving contributor onboarding;
- running reproducibility and security review workflows.

All generated changes will remain subject to maintainer review and CI.

## 16. Application Strategy

### 16.1 Codex for Open Source

The application will:

- identify HectorGao as a core collaborator with write access;
- describe the real maintenance responsibilities created by the transformation;
- explain the climate-science and reproducibility value;
- state the repository's early-stage adoption honestly;
- request Codex Security only for authorized repository analysis;
- request API credits for maintenance workflows.

The application will not invent stars, downloads, users, citations, or
maintenance history.

### 16.2 Codex Open Source Fund

The proposal will focus on converting a dataset-constrained research artifact
into a reusable public scientific toolkit. The requested credits will support
maintainer automation, reproducibility checks, documentation, review, and
release work.

The application will distinguish OpenAI-assisted software maintenance from the
glacier prediction algorithms themselves.

### 16.3 Submission timing

Both applications should be submitted after:

- the public README and license are visible;
- the quickstart passes from a clean environment;
- CI passes on the public branch;
- the maintainer and data-source documentation are published;
- the application statements match the current repository state.

## 17. Migration Strategy

The work will be delivered in independently verifiable stages:

1. establish project metadata, licensing, governance, and packaging;
2. add the common schema, configuration, manifests, and sample data;
3. implement validation and evaluation primitives;
4. migrate the Random Forest quickstart;
5. migrate optional models behind the common interfaces;
6. convert notebooks into tutorials and preserve legacy artifacts;
7. add CI, documentation, and release metadata;
8. verify the clean-environment quickstart;
9. prepare and fill both application forms.

No bulk deletion will be used. Existing files will be moved or retired
individually and only when their replacement and historical preservation are
clear.

## 18. Success Criteria

The transformation is complete when:

1. the repository is publicly licensed under Apache-2.0;
2. a new user can install the package from a clean Python 3.11 environment;
3. the CPU quickstart completes without private paths or unavailable raw data;
4. CI validates linting, tests, and packaging;
5. leakage invariants are covered by automated tests;
6. upstream data provenance and redistribution boundaries are documented;
7. legacy notebooks are preserved and clearly separated from supported APIs;
8. governance, security, contribution, citation, and maintenance documents
   exist;
9. the public repository state supports every factual claim in both
   applications;
10. both forms are filled with user-approved personal details and submitted or
    left ready for the user's final submission action.

## 19. Risks and Mitigations

### Data redistribution uncertainty

Mitigation: distribute only synthetic data until permissions are verified, and
document official acquisition paths.

### Historical result reproducibility

Mitigation: label embedded outputs as historical and publish only results
produced by the new pipeline as verified benchmarks.

### Excessive refactoring

Mitigation: migrate the Random Forest vertical slice first, then add optional
models through stable interfaces.

### Expensive model evaluation

Mitigation: separate quick CI smoke tests from full scientific benchmarks.

### Weak current adoption signals

Mitigation: make no inflated claims; emphasize scientific importance,
reproducibility, maintenance responsibilities, and a concrete roadmap.

### AI use undermining scientific trust

Mitigation: limit OpenAI usage to software maintenance and keep scientific
predictions deterministic, reviewable, and independently testable.
