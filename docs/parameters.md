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
| `n_ensembles` | `1` | Member count: 1–256 in legacy bootstrap mode; exactly 8 for v3. |
| `ensemble_bootstrap` | `"rows"` | Sample rows, or complete entities with `"groups"` and `groups=` in `fit`. |
| `ensemble_shared_preprocessing` | `True` | Reuse one target-free numeric preprocessor when safe. Categorical and ordinal fits fall back to member-local preprocessing. |
| `ensemble_mode` | `"bootstrap"` | Keep legacy bootstrap behavior, or select the fixed public `"v3"` recipe. |
| `ensemble_member_learning_rate` | `"policy"` | In v3, use recipe value `0.15`; pass `None` or a positive finite number to override members only. |
| `ensemble_member_colsample` | `"policy"` | In v3, use recipe value `0.85`; pass a finite number in `(0, 1]` to override members only. |
| `ensemble_parallelism` | `"auto"` | In v3, use static activation for large member fits on the measured 14-core macOS-arm64 envelope. Use `"sequential"` to roll back or `"parallel"` to force process workers. |

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

`ensemble_mode="v3"` is a separate explicit recipe. It requires
`n_ensembles=8`, samples 80% of rows or groups without replacement, and uses
the exact complement as each member's OOB validation set. Eight is the only
evaluated count, not a claim that it is universally optimal. Top-level
`learning_rate` and `colsample` remain base-model parameters; only the two
dedicated member parameters override the v3 policy. They must remain
`"policy"` in legacy bootstrap mode. V3 archives use safe-NPZ format 4 with
the resolved policy and sample/OOB provenance; legacy bootstrap archives stay
on format 1.

V3's default `ensemble_parallelism="auto"` uses a deterministic pre-fit score:
80%-sampled rows × input features × planned iterations × output width
(one for regression/binary, class count for multiclass). It engages seven
two-thread member workers at `80,000,000` work units or above only on the
measured 14-core macOS-arm64 envelope. Other shapes and machines stay
sequential. The resolved route, reason, CPU topology, and score are stored in
fitted metadata. `"sequential"` is the documented rollback; `"parallel"` is
an explicit research/escape-hatch override. Parallel fits were behavior-exact
on the four-case development grid and used about 2.28 GiB peak process-tree
RSS at worst; this is engineering characterization, not a portability claim.

## Structure

`tree_mode="catboost"` is the symmetric-tree default. `"lightgbm"` selects
DarkoFit's leaf-wise builder; it is not Microsoft LightGBM model
compatibility. `"hybrid"` and `"auto"` remain explicit experimental paths.

`linear_leaves="auto"` is the regressor default for scalar RMSE
oblivious-tree fits. Eligible fits audition constant and local-linear leaves
on a deterministic validation split and select linear only when the paired
per-row MSE gain is positive and at least two standard errors above zero.
Groups and sample weights are honored by the audition. Small or unsupported
fits fall back exactly and record why in `automatic_linear_selector_`.
`linear_leaves=False` bypasses the audition and is the rollback;
`linear_leaves=True` forces the eligible local-linear lane.

`ordinal_features={column: order}` explicitly maps declared ordered
categories into the numeric binner; unknown non-missing values fail closed.

`oblivious_kernel` is a bounded observability and escape-hatch option for the
eligible scalar CatBoost lane. `"fused"` and `"unfused"` force the two
behavior-equivalent histogram/split implementations; unsupported explicit
configurations fail loudly. The default `"auto"` records its deterministic
resolution in `oblivious_kernel_dispatch_` and fitted `auto_params_`. Within the
measured macOS-arm64 envelope, automatic dispatch uses the promoted static
`scan_work` threshold `1048576`. This is a deterministic product policy with
an explicit override and rollback surface, not a cross-hardware speed claim;
other hardware and unsupported shapes keep the established fused path.

See [Feature recipes](recipes.md) for measured benefits and failure boundaries.

## Sampling and categoricals

Uniform sampling is the default. GOSS and MVS are explicit alternatives.
Categorical predictors use target-statistic preprocessing. `ts_permutations`
controls repeated ordered target-stat permutations and defaults to one.

`categorical_crosses=True` opts an eligible `DarkoRegressor` into an automatic
held-out audition of group-centered numeric-by-category features. It currently
supports single-model scalar-RMSE fits with `tree_mode="catboost"`. The
audition fits a control and an augmented candidate, then fits the winner from
scratch and records the decision in
`group_centered_categorical_crosses_`. Data with no categorical columns, no
numeric columns, or too few selection rows falls back to the ordinary engine
exactly and records the reason. Incompatible requested modes fail loudly
instead of silently disabling the opt-in. The default `False` path does no
audition and preserves the established engine.

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
