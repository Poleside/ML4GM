# Scientific limitations

ML4GM is early-stage research software. Its interfaces and results require
independent scientific review before use in peer-reviewed analysis.

- The bundled quickstart data are synthetic. Their values and scores have no
  scientific meaning and cannot validate a glacier model.
- Historical notebook outputs have not been revalidated by the packaged
  pipeline. They preserve research history, not release benchmark evidence.
- Upstream observations and reanalyses carry measurement, sampling, spatial
  aggregation, temporal alignment, and coverage uncertainty.
- RGI geometry represents a versioned glacier inventory rather than continuous
  glacier-boundary evolution.
- Reanalysis variables may not resolve local topography or precipitation
  gradients at glacier scale.
- Elevation-change targets and derived features may share upstream processing
  assumptions; leakage-safe splitting does not remove source uncertainty.
- Results depend on feature definitions, units, missing-data handling,
  hyperparameters, compute budgets, and random seeds.
- LOYO, spatial, and strict block validation answer different generalization
  questions and should not be compared as interchangeable scores.

ML4GM predictions are not operational hazard warnings, flood forecasts,
navigation advice, infrastructure design inputs, or water-resource management
guidance. Decisions affecting people, ecosystems, or assets require appropriate
observations, domain models, uncertainty analysis, expert review, and local
governance.
