# Parameters

DarkoFit deliberately keeps product defaults conservative. The most useful
controls are below; fitted resolutions are recorded in
`model.model_.auto_params_`.

## Training

| Parameter | Default | Purpose |
|---|---:|---|
| `preset` | `None` | Opt-in product profile; `"accuracy"` applies the frozen A10 managed fields. |
| `iterations` | `1000` | Maximum boosting rounds. |
| `learning_rate` | `None` | Automatic fitted-rate rule; pass a positive float to freeze it. |
| `depth` | `None` | Mode default: 6 for CatBoost/classifier depthwise trees, 2 for depthwise RMSE, and unlimited (`-1`) for LightGBM/hybrid; pass `"auto"` for the separate effective-sample-size rule. |
| `l2_leaf_reg` | `"auto"` | Mode-aware leaf regularization. |
| `max_bins` | `254` | Numeric bin budget. |
| `thread_count` | `None` | Use the Numba runtime maximum, except LightGBM/hybrid fits at 50,000 rows or fewer are capped at 2. The fitted count is recorded and reused without changing the caller's ambient mask. |
| `random_state` | `None` | No fixed seed; pass an integer for reproducible model and sampling randomness. |

`None` is parameter-specific, not a universal alias for `"auto"`.
`learning_rate=None` selects the fitted automatic rate, while `depth=None`
uses the mode defaults above. Structure parameters that advertise `"auto"`
use that literal token for their data-dependent rule. Other numeric parameters
reject `None` unless their row explicitly documents it.

## Validation and refit

| Parameter | Default | Purpose |
|---|---:|---|
| `early_stopping` | `False` | Enable automatic or explicit validation stopping. |
| `early_stopping_rounds` | `None` | Automatic patience from fitted LR, or an explicit integer. |
| `validation_fraction` | `0.1` | Automatic holdout fraction; `"auto"` is supported. |
| `validation_strategy` | `"random"` | Random, weighted-stratified regression, or group-disjoint selection. |
| `use_best_model` | `True` | Retain the best validation prefix. |
| `refit` | `False` | Refit the selected policy on all rows. |
| `selection_rounds` | `None` | Cap each `tree_mode="auto"` audition, then refit the selected mode at the full budget. |

Pass `groups=` to `fit` for entity-disjoint validation. Use
`get_refit_params()` to export the fitted concrete policy for a manual refit.

`preset="accuracy"` manages `iterations=10000`, `tree_mode="auto"`,
`l2_leaf_reg=3`, `max_bins=128`, `learning_rate=0.1`,
`ts_permutations=1`, `linear_residual=False`, `early_stopping=True`, and
`use_best_model=True` during fit. The preset deliberately overrides
constructor values for those managed fields; explicit parameters outside that
list remain user overrides. The constructor values are restored after fitting;
the resolved profile is stored in fitted metadata, and `get_refit_params()`
returns its concrete selected mode with `preset=None`.

`selection_rounds` currently applies only to `tree_mode="auto"`. The capped
auditions choose a mode; DarkoFit then starts a fresh full-budget fit of that
mode. If a shared wall-clock deadline has expired before that refit starts,
the selected capped audition is retained and fitted metadata records
`final_refit_status="skipped_deadline"`. Leaving it at `None` preserves the
historical selection path.

## Ensembles

| Parameter | Default | Purpose |
|---|---:|---|
| `n_ensembles` | `1` | Number of independently bootstrapped members, from 1 through 256. Values above one opt into ensemble mode. |
| `ensemble_bootstrap` | `"rows"` | Bootstrap rows, or complete entities with `"groups"` and `groups=` in `fit`. |
| `ensemble_shared_preprocessing` | `True` | Reuse one target-free numeric preprocessor when safe. Categorical and ordinal fits fall back to member-local preprocessing. |

Each member uses its out-of-bag rows as an explicit early-stopping set.
Regression predictions are member means; classification probabilities are
soft-vote means. `shap_values()` averages member contributions and expected
values. Group bootstraps keep sampled and OOB groups disjoint. Supplying
`groups=` in ensemble mode requires `ensemble_bootstrap="groups"`; row
bootstraps reject groups rather than silently splitting entities.

Ensemble archives remain pickle-free: `save_model()` stores independently
loadable member NPZ payloads inside one validated outer NPZ. Explicit
`eval_set`, callbacks, automatic ordinal discovery, and distributional heads
are not supported in ensemble mode. `refit=True` is also rejected because it
would replace the bootstrap training rows that define each member. Declare
ordinal orders explicitly; categorical target-statistic preprocessing is
always fitted separately inside each member to avoid target leakage. Staged
prediction yields the common prefix shared by every member.

## Structure

`tree_mode="catboost"` is the symmetric-tree default. `"lightgbm"` selects
DarkoFit's leaf-wise builder; it is not Microsoft LightGBM model
compatibility. `"hybrid"` and `"auto"` remain explicit experimental paths.

`linear_leaves=True` enables local linear leaf models for eligible scalar RMSE
oblivious-tree fits. It is default-off. `ordinal_features={column: order}`
explicitly maps declared ordered categories into the numeric binner; unknown
non-missing values fail closed.

See [Feature recipes](recipes.md) for measured benefits and failure boundaries.

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
