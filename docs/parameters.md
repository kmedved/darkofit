# Parameters

DarkoFit deliberately keeps product defaults conservative. The most useful
controls are below; fitted resolutions are recorded in
`model.model_.auto_params_`.

## Training

| Parameter | Default | Purpose |
|---|---:|---|
| `iterations` | `1000` | Maximum boosting rounds. |
| `learning_rate` | `None` | Automatic fitted-rate rule; pass a positive float to freeze it. |
| `depth` | `None` | Symmetric-tree depth or non-oblivious path-depth cap. |
| `l2_leaf_reg` | `"auto"` | Mode-aware leaf regularization. |
| `max_bins` | `254` | Numeric bin budget. |
| `thread_count` | `None` | Numba worker count; `None` uses the runtime limit. |
| `random_state` | `None` | Reproducible model and sampling seed. |

## Validation and refit

| Parameter | Default | Purpose |
|---|---:|---|
| `early_stopping` | `False` | Enable automatic or explicit validation stopping. |
| `early_stopping_rounds` | `None` | Automatic patience from fitted LR, or an explicit integer. |
| `validation_fraction` | `0.1` | Automatic holdout fraction; `"auto"` is supported. |
| `validation_strategy` | `"random"` | Random, weighted-stratified regression, or group-disjoint selection. |
| `use_best_model` | `True` | Retain the best validation prefix. |
| `refit` | `False` | Refit the selected policy on all rows. |

Pass `groups=` to `fit` for entity-disjoint validation. Use
`get_refit_params()` to export the fitted concrete policy for a manual refit.

## Structure

`tree_mode="catboost"` is the symmetric-tree default. `"lightgbm"` selects
DarkoFit's leaf-wise builder; it is not Microsoft LightGBM model
compatibility. `"hybrid"` and `"auto"` remain explicit experimental paths.

`linear_leaves=True` enables local linear leaf models for eligible scalar RMSE
oblivious-tree fits. It is default-off. `ordinal_features={column: order}`
explicitly maps declared ordered categories into the numeric binner; unknown
non-missing values fail closed.

## Sampling and categoricals

Uniform sampling is the default. GOSS and MVS are explicit alternatives.
Categorical predictors use target-statistic preprocessing. `ts_permutations`
controls repeated ordered target-stat permutations and defaults to one.

## Distributional

Set `loss` to `Gaussian`, `LogNormal`, `StudentT`, `Poisson`, or
`NegativeBinomial` with `tree_mode="lightgbm"`. See
[Distributional regression](uncertainty.md).

## Deprecated for 1.0

The current deprecation cycle warns on selected depthwise, low-level
histogram/leaf controls, learning-rate probes, Bayesian bootstrap,
weighted-GOSS, and global linear-residual controls. See the
[Changelog](https://github.com/kmedved/darkofit/blob/main/CHANGELOG.md) for
exact migration guidance.
