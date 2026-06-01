# Changelog

All notable changes to ChimeraBoost are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
### Changed
- **Classifier `min_child_weight` is now size-adaptive by default** (`None` → auto:
  full veto ~1 below ~500 training rows, fading to 0 above ~2000). The old flat
  `mcw=1` silently capped oblivious classification tree depth (~4.9 of 6),
  under-fitting larger data; the new default lifts binary Brier broadly (18W/0L on
  the Grinsztajn suite, +1.6pp, reaching the speed/accuracy Pareto frontier) while
  the size ramp protects small datasets (validated on an independent OpenML set).
  Root-caused by matching a stripped-down CatBoost: the gap was our min-leaf veto,
  not the oblivious tree structure. Regression is unaffected (a no-op in [0,1]
  post empty-child-exemption); explicit `min_child_weight` values are still honored.

### Added
- **Input validation** across both estimators: clear, actionable errors instead
  of cryptic numpy/numba tracebacks for predict-before-fit (`NotFittedError`),
  feature-count mismatch at predict time, and 1-D / empty / mismatched-length /
  complex / sparse / non-finite inputs and `y=None`.
- `n_features_in_` and (for DataFrame input) `feature_names_in_` attributes.
- A column-vector `y` of shape `(n, 1)` is now raveled with a
  `DataConversionWarning`; a continuous target passed to the classifier raises.
- **scikit-learn `check_estimator` compliance** for both estimators, with a
  single documented deviation: `sample_weight` reweights the loss but is not
  bit-exactly equivalent to integer row repetition. See the README
  "scikit-learn compatibility" section for the full list of deviations
  (NaN-in-X accepted as missing, dense-only, `cat_features`/`eval_set` kwargs).

## [0.9.1] - 2026-06-01
### Changed
- Tidied the README and benchmark tables; moved the "near-solved excluded from
  RMSE" note into a proper footnote and added the blended-strength Pareto image.
- Corrected the CatBoost speed claim to ~5x (geomean on the 59-dataset
  Grinsztajn 2022 benchmark); the old ~30x was from the categorical-heavy
  OpenML suite.

## [0.9.0] - 2026-06-01
### Fixed
- **Oblivious depth cap:** empty (pure) children are now exempt from the
  `min_child_weight` veto, so `depth` is a real lever again. Regression RMSE
  rose from 95.7% to 98.0% of best on the Grinsztajn suite (now beats sklearn),
  with a broad 26W/6L per-dataset sign test, and fits got faster.
### Changed
- Classifier defaults: `ordered_boosting=False`, `leaf_estimation_iterations=3`.
- Regressor default: `ordered_boosting=False`.
- Benchmarks: blended-strength Pareto, near-solved RMSE guard, `/bench` command.

## [0.8.0]
### Added
- First-class bagging (`n_ensembles`) and the Brier benchmark metric.
