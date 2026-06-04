# Parameter reference

Every constructor parameter of `ChimeraBoostRegressor` and `ChimeraBoostClassifier`,
grouped by what it controls. Where the two estimators differ, both defaults are
shown. You rarely need to touch most of these — the defaults are the benchmarked,
Pareto-frontier configuration — so the **Tune?** column flags the few worth reaching for.

!!! tip "The short list"
    For most problems the only knobs you'll touch are **`depth`** (raise to 8–10 for
    interaction-heavy regression), **`n_ensembles`** (variance reduction), and
    **`random_state`** (reproducibility). Everything else has a defensible default.

## Core boosting

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `iterations` | `2000` | rarely | Max boosting rounds (trees). With early stopping on, this is a ceiling, not a target — the best iteration is selected automatically. |
| `learning_rate` | `None` (auto) | rarely | Shrinkage per tree. `None` resolves to `0.1` with early stopping. Lower = more trees, often slightly better; raise to train faster. |
| `depth` | `6` | **yes** | Tree depth. A depth-`d` oblivious tree is `d` splits. Default is conservative to protect small data; **raise to 8–10 for large, interaction-heavy regression** (see [recipes](recipes.md#interaction-heavy-regression)). |
| `l2_leaf_reg` | `1.0` | rarely | L2 penalty on leaf values (Newton denominator). Larger = smoother leaves. |
| `min_child_weight` | `1.0` (reg) / `None`→auto (clf) | rarely | Minimum hessian mass each side of a split. The classifier's `None` resolves to a size-adaptive value (full veto below ~500 rows, off above ~2000) — a key small-data guard. |
| `leaf_estimation_iterations` | `1` (reg) / `3` (clf) | rarely | Extra Newton refinement steps per leaf. More steps sharpen the leaf value for logloss; little effect for RMSE. |

## Binning

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `max_bins` | `128` | no | Histogram bins per numeric feature. Tested extensively — raising it overfits noise and slows builds without generalizing. Left as a knob, but don't reach for it. |

## Row & column sampling

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `subsample` | `1.0` | optional | Row subsampling fraction per tree. Below 1, uses Minimum Variance Sampling (gradient-weighted, unbiased) rather than uniform. |
| `colsample` | `1.0` | optional | Fraction of columns eligible per tree (feature subsampling). |

## Categorical features

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `cat_smoothing` | `1.0` | rarely | Prior strength for ordered target statistics (higher = more shrinkage toward the global mean for rare categories). |
| `cat_n_permutations` | `4` | no | Independent orderings averaged in the ordered target encoder (√K variance reduction, CatBoost-style). |
| `cat_combinations` | `False` | optional | Add all pairwise category×category features. Helps mostly-categorical data; can crowd out numerics on mixed data. |

See [recipes → categoricals](recipes.md#categorical-features). Which columns are
categorical is passed to `fit(..., cat_features=[...])`, not the constructor.

## Loss (regressor only)

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `loss` | `"RMSE"` | task | `"RMSE"`, `"MAE"` (median), or `"Quantile"`. |
| `alpha` | `0.5` | task | Quantile level for `loss="Quantile"` (e.g. `0.9` for the 90th percentile). |

The classifier picks its loss automatically: binary logloss for 2 classes, softmax for 3+.

## Leaf models

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `linear_leaves` | `False` (reg) / `None`→auto (clf) | optional | Fit a small ridge **linear model** per leaf over the numeric split features, instead of a constant — adds local slope where step leaves underfit. Auto-on for **binary** classification (a validated Brier win); opt-in for regression. Falls back to constant below ~1000 rows. Not available for MAE/Quantile or multiclass. |
| `linear_lambda` | `1.0` | optional | Ridge penalty on the per-leaf slopes (larger = closer to constant leaves). |
| `hs_lambda` | `0.0` | optional | Hierarchical-shrinkage strength: when > 0, leaf values are recursively shrunk toward their ancestors (deep/low-mass leaves hardest). A cheap post-pass with no inference cost. |

## Ordered boosting

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `ordered_boosting` | `False` | no | Leave-one-out leaf training step (curbs the self-reinforcement of plain boosting). Off by default — interacts with `leaf_estimation_iterations` (mutually exclusive in the booster). |

## Early stopping

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `early_stopping` | `True` | rarely | Carve an internal validation split, stop when it plateaus, keep the best iteration. Set `False` to build a fixed `iterations` trees. |
| `early_stopping_rounds` | `None`→`50` | rarely | Patience (rounds without improvement) when early stopping is active. |
| `validation_fraction` | `0.2` | rarely | Fraction held out for the automatic validation split (stratified for classifiers). Ignored when an explicit `eval_set` is given to `fit`. |

See [recipes → early stopping](recipes.md#early-stopping) for `eval_set` and `groups`.

## Bagging

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `n_ensembles` | `None` | **yes** | `None`/`1` = single model. `≥2` trains that many members on bootstrap resamples and averages them (regressors average predictions; classifiers soft-vote calibrated probabilities). Cuts variance. |
| `ensemble_n_jobs` | `1` | optional | Processes used to fit members in parallel (`-1` = all cores). numba threads are split among workers when `thread_count` is `None`. |

## System

| Parameter | Default | Tune? | What it does |
|---|---|:--:|---|
| `thread_count` | `None` | optional | numba threads. `None`/`-1` = all detected cores. Affects determinism of floating-point reductions. |
| `random_state` | `None` | **yes** | Seed for reproducibility (deterministic given a fixed `thread_count`). |
| `verbose` | `False` | optional | Print per-round train/validation metrics. |

## `fit()` arguments

These are passed to `fit`, not the constructor:

| Argument | What it does |
|---|---|
| `cat_features` | List of column indices to treat as categorical. |
| `eval_set` | `(X_val, y_val)` explicit validation set — overrides the internal split. |
| `groups` | Group labels; keeps each group entirely in train or validation when auto-splitting. |
| `sample_weight` | Per-sample training weights (normalized to mean 1 internally). |

## Fitted attributes

Available after `fit`:

| Attribute | Meaning |
|---|---|
| `feature_importances_` | Split-gain importance per original feature, summing to 1. |
| `best_iteration_` | Number of trees kept (after early stopping). |
| `classes_` *(classifier)* | Original label values, in column order of `predict_proba`. |
| `temperature_` *(classifier)* | Fitted calibration temperature (>1 = scores were over-confident). |
| `expected_value_` | SHAP baseline; set after calling `shap_values` (see [SHAP](shap.md)). |
| `estimators_` | The fitted members when `n_ensembles > 1`, else `None`. |
| `model_` | The underlying booster when not bagged. |
