# Core concepts

## Numeric binning

Numeric predictors are quantized once and trees search histogrammed gradients
and Hessians. Large fits sample at most 200,000 rows for border construction
unless `bin_sample_count=None` is supplied. Weighted fits use weighted
quantiles.

## Categorical predictors

Categorical columns must be declared through `cat_features`. CatBoost-mode
fits use ordered target statistics; leaf-wise and hybrid modes use K-fold
target statistics and compatible raw category-code features. Explicit
`ordinal_features` declarations replace only those columns with target-free
rank codes in the numeric binner.

## Tree modes

- `catboost`: symmetric trees and the supported ordered-boosting path.
- `lightgbm`: best-first, leaf-wise DarkoFit trees.
- `hybrid`: experimental shared prefix followed by leaf-wise expansion.
- `auto`: validation-backed selection among concrete modes.

Ordered boosting defaults on for classification and off for scalar regression.
MAE and Quantile reject explicit ordered leaf updates because they recompute
leaf values from residual statistics.

## Validation

An explicit `eval_set=(X_validation, y_validation)` takes precedence over
automatic splitting. Entity data should pass `groups=` and
`validation_strategy="group"`. Regression uses a group-shuffled holdout;
classification uses a stratified group split.

## Diagnostics

Resolved parameters, validation policy, selected mode, learning rate, stop
reason, binning, and warnings are stored in `model_.auto_params_` and fitted
training metadata. `diagnostic_warnings` controls runtime emission, not record
creation.

## Input compatibility

NaN is the missing-value representation. Complex values, masked arrays,
sparse matrices, undeclared nonnumeric columns, and infinity are rejected by
default. Trusted pipelines may use
`sklearn.config_context(assume_finite=True)` to bypass feature infinity checks.
Zero-row prediction batches return shape-correct empty outputs. Duplicate
categorical declarations are rejected.

## Serialization

`save_model` writes a versioned `.npz` archive that loads with
`allow_pickle=False`. Wrapper state, preprocessing, trees, distributional
metadata, ordinal declarations, and supported SHAP backgrounds round-trip.
