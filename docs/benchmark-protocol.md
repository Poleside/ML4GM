# Benchmark protocol

This protocol defines comparisons supported by ML4GM. A benchmark is
reproducible only when its data manifest, configuration, software revision,
environment, random seed, and fold-level results are retained.

## Shared leakage invariants

For every fold:

1. Choose train and test row indices before fitting preprocessing or a model.
2. Fit imputation, scaling, feature selection, and learned transforms on
   training rows only.
3. Fit the model on training rows only.
4. Apply the frozen training transforms to test rows.
5. Compute R², RMSE, and MAE only from held-out predictions.
6. Never use test targets for feature construction, tuning, stopping, or model
   selection.
7. Any nested tuning split must be contained entirely within the outer training
   rows.

Report every fold, not only an average, and record failures rather than silently
dropping difficult folds.

## Leave-one-year-out (LOYO)

Each unique target year forms one test fold. Training contains every row whose
year differs from the held-out year.

**Invariant:** no row from the held-out target year may enter training,
preprocessing fit, tuning, or temporal context. Temporal sequence construction
must not expose the held-out target year or later information through history
features.

LOYO measures generalization to an unseen year across glaciers represented in
other years; it is not a test of unseen-glacier generalization.

## Spatial GroupKFold

Assign all rows for a glacier identifier to one GroupKFold group. Each fold
holds out complete glacier groups while permitting training on other glaciers
from overlapping years.

**Invariant:** the sets of glacier identifiers in training and test are
disjoint. All rows, sequences, preprocessing statistics, and tuning data for a
held-out glacier remain outside training.

Spatial GroupKFold measures unseen-glacier generalization, not future-year
generalization.

## Strict block validation

Partition glacier identifiers and years into deterministic groups. For block
`k`, test rows have both glacier-group `k` and year-group `k`; training rows
must have neither glacier-group `k` nor year-group `k`. Rows sharing only one
held-out dimension are excluded from both sets.

**Invariant:** training and test share no held-out glacier group and no held-out
year group. A temporal sequence for a test sample must not borrow context from
excluded or test rows.

Strict blocks test joint transfer to unseen glacier-and-year combinations.
Because excluded cross-block rows reduce training data, report exact fold sizes.

## Regression coverage

The temporal leakage invariants are executable requirements, not documentation
alone. The integration suite includes
[`test_temporal_loyo_filters_held_out_year_from_unbalanced_glacier_context`](../tests/integration/test_sequence_models.py)
and
[`test_temporal_block_filters_entire_held_out_year_group_from_context`](../tests/integration/test_sequence_models.py).
These regressions verify that held-out year groups cannot re-enter model fitting
through temporal context windows.

## Comparison requirements

Use identical outer splits, input features, target units, and metric
implementations for models being compared. Disclose model-specific preprocessing
and compute budgets. Synthetic quickstart scores are software checks only and
must not appear as scientific benchmark results.
