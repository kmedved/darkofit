# chimeraboost
Full-featured gradient boosting library with a Python/numba backend, inspired by CatBoost.

<img width="500" height="500" alt="ChatGPT Image May 26, 2026, 05_12_17 PM" src="https://github.com/user-attachments/assets/ee98a4e2-9fa7-4ef1-9e64-e398f398966c" />

* **What?**
    * GBDT library that only depends on numpy, numba, and scikit-learn
    * Accuracy-competitive with CatBoost and LightGBM in the benchmark suite,
      with speed depending strongly on dataset size, tree mode, and whether the
      Numba kernels are warm
    * Supports sample weights and automatic early stopping

* **Why?**
    * I want to be able to modify my GBDT library at will
    * I know Python and I don't know C

* **How?**

```
# pip install chimeraboost

from chimeraboost import ChimeraBoostClassifier
clf = ChimeraBoostClassifier(early_stopping=True)
clf.fit(X, y, sample_weight=w)
```

Models can be saved without pickle:

```
clf.save_model("model.npz")
clf2 = ChimeraBoostClassifier.load_model("model.npz")
```

For wrappers fitted with `refit=True`, scalar refit metadata is preserved on
load, while the fold-selection model itself is intentionally not persisted;
loaded wrappers expose `selection_model_persisted_=False`.

After fitting, `model_.auto_params_` records the resolved training context:
the actual learning rate, effective sample size, feature counts, tree sizing,
regularization, binning policy, early-stopping policy, sampling policy, and
threading choice used by that fit. It also records stochastic-regularization
settings for row sampling, Bayesian bootstrap, split-score noise, validation
split policy, learning-rate probe policy, target statistics, and any resolved
opt-in auto structure parameters.
Fit diagnostics include
learning-rate clipping, low effective-sample-size warnings, weighted-binning
activation, observed bin counts, feature expansion, and best-prefix policy.
Runtime diagnostic warnings are throttled by default with
`diagnostic_warnings="once"`; set `diagnostic_warnings="always"` while
debugging or `"never"` for quiet benchmark runs. The full diagnostic records are
kept in `auto_params_` regardless of runtime warning policy.

```
clf.model_.auto_params_["learning_rate"]
clf.model_.auto_params_["binning"]
clf.model_.auto_params_["diagnostics"]
```

When `learning_rate=None` or `"auto"`, ChimeraBoost uses a transparent
CatBoost-form rule fitted as `log(lr) ~ log(n) + log(iterations)`, with Kish
effective sample size replacing raw row count when `sample_weight` is supplied.
For materially weighted RMSE fits in CatBoost/oblivious-tree mode, the selector
applies an LR-only ChimeraBoost correction; unweighted and all-ones-weight fits
keep the raw CatBoost-form value.
The selector also applies a small bounded shrinkage when preprocessing expands
the model feature count heavily relative to effective sample size; this
high-dimensional adjustment is capped at a 15% LR reduction and recorded under
`auto_params_["learning_rate"]`.
LightGBM-mode unweighted fits apply an additional provisional dampening factor
because the CatBoost coefficients were fitted on symmetric-tree regularization,
not ChimeraBoost's leaf-wise stack.
Set `auto_learning_rate_probe=True` on the sklearn wrappers to run an opt-in
short validation probe around the final-budget resolved automatic learning
rate; the selected explicit rate, candidate scores, full-budget base rate, and
short-budget diagnostic rate are recorded under
`auto_params_["learning_rate_probe"]`. This is disabled by default.
The default boosting budget is `iterations=1000`, and the default numeric bin
budget is `max_bins=254`.

Tree builders are selectable:

```
ChimeraBoostClassifier(tree_mode="catboost")  # symmetric/oblivious default
ChimeraBoostClassifier(tree_mode="lightgbm")  # leaf-wise, non-oblivious
ChimeraBoostClassifier(tree_mode="hybrid")    # experimental shared-prefix then leaf-wise
ChimeraBoostClassifier(tree_mode="auto")      # validation-selected tree mode
ChimeraBoostClassifier(tree_mode="depthwise") # experimental level-wise
```

Tree modes:

* `tree_mode="catboost"` builds symmetric / oblivious trees and supports
  ordered boosting.
* `tree_mode="lightgbm"` builds ChimeraBoost's LightGBM-like histogram trees:
  non-oblivious, leaf-wise, best-first CART-style trees. This is not model or
  prediction compatibility with Microsoft LightGBM.
* `tree_mode="hybrid"` uses an experimental non-oblivious tree with a shallow
  shared symmetric prefix followed by best-first leaf-wise expansion. It stores
  and predicts as a normal ChimeraBoost non-oblivious tree.
* `tree_mode="auto"` is opt-in validation selection across `"catboost"`,
  `"lightgbm"`, and `"hybrid"`. Pair it with `refit=True` to train the selected
  concrete tree mode on all rows after selection.
* `tree_mode="depthwise"` (also accepted as `"levelwise"`) uses the
  experimental level-wise non-oblivious builder. Current benchmark notes show it
  can reduce rounds on some medium numeric tasks. For RMSE regression with
  omitted `depth`, this mode defaults to a shallow depth of 2; explicit depths
  and classification defaults are unchanged.

In LightGBM and hybrid modes, `num_leaves` is the main tree-size control and
`depth` is a maximum path-depth cap. `ordered_boosting` defaults to off for
these modes; setting `ordered_boosting=True` with either mode raises a
`ValueError`.
Categorical features still use ChimeraBoost's target-stat preprocessing, not
native LightGBM category-partition splits. CatBoost/depthwise modes use ordered
target statistics; LightGBM and hybrid modes use K-fold target statistics and
include raw category-code features for compatible RMSE/logloss-style scalar and
multiclass categorical fits. MAE and Quantile fits keep target-stat features
only.

For compatible LightGBM-mode multiclass fits,
`multiclass_tree_strategy="auto"` uses a shared vector-valued tree per boosting
round. Pass `multiclass_tree_strategy="per_class"` to force the older one-tree-
per-class route for comparisons.

Distributional regression is available with `loss="Gaussian"`:

```
reg = ChimeraBoostRegressor(loss="Gaussian", tree_mode="lightgbm")
reg.fit(X, y)
mu = reg.predict(X_test)
mu, sigma = reg.predict_dist(X_test)
lo, hi = reg.predict_interval(X_test, alpha=0.1)
draws = reg.sample(X_test, n_samples=100, random_state=0)
```

This fits a shared two-output leaf-wise tree per boosting round: one head for
the mean and one for log standard deviation. It is v1-lightweight by design:
use `tree_mode="lightgbm"` (aliases such as `"leafwise"` are accepted), with
uniform row subsampling (`subsample < 1`) and column subsampling (`colsample <
1`) supported. GOSS/MVS sampling, Bayesian bootstrap, ordered boosting, and
float32 histograms are still rejected for this loss. `predict()` returns the
mean so existing regression scoring keeps working. `min_child_weight` is
evaluated on the summed two-head Hessian mass, while `min_child_samples` keeps
its usual row-count meaning.

Validation and early-stopping use Gaussian NLL by default. Pass
`eval_metric="crps"` to select the best validation prefix by closed-form
Gaussian CRPS instead:

```
reg = ChimeraBoostRegressor(
    loss="Gaussian",
    tree_mode="lightgbm",
    early_stopping=True,
    eval_metric="crps",
)
```

For small data, early stopping is especially important for Gaussian fits:
training too long can make the log-standard-deviation head overfit residuals
and produce intervals that are too narrow. Use an explicit `eval_set` or
`early_stopping=True` when interval calibration matters.

`sigma_calibration="scalar"` is an opt-in validation-set calibration for
Gaussian models. It fits the NLL-optimal global scale
`sqrt(weighted_mean(((y - mu) / sigma) ** 2))` on the selected validation
prefix, persists the scale through save/load, and applies it to
`predict_dist`, `predict_interval`, and `sample`. It does not change
`predict()` or `predict_raw()`. With `refit=True`, the scale is frozen from the
selection-phase validation model and then applied to the full-data refit. The
calibration fold is the same fold used for early stopping, so the scale has a
small selection bias; the benchmark below found that negligible, but fits with
an effective calibration fold below 200 rows record a diagnostic warning
because the scale estimate can be noisy.

Distributional benchmark, mean over three seeds on the synthetic
heteroscedastic gate:

| model | NLL 100k | NLL 500k | CRPS 500k | cov90 500k | fit s 500k |
| --- | ---: | ---: | ---: | ---: | ---: |
| ChimeraBoost Gaussian, early-stopped + calibrated | **0.990** | **0.983** | **0.390** | 0.900 | 10.5 |
| NGBoost Normal | 1.014 | 1.008 | 0.396 | 0.905 | 125.9 |
| CatBoost `RMSEWithUncertainty` | 1.058 | 1.056 | 0.410 | 0.909 | 0.8 |
| LightGBM twin-model variance hack | 1.644 | 1.630 | 0.419 | 0.619 | 3.5 |

Command and per-seed rows live in
[BENCHMARK_NOTES.md](BENCHMARK_NOTES.md) and
[benchmarks/distributional_summary.md](benchmarks/distributional_summary.md).
Before using predicted `sigma` as downstream observation noise, validate on
real data with natural heteroscedasticity; the promotion table above is
synthetic.

Not implemented for Gaussian distributional regression in v1: CatBoost-style
per-parameter scalar trees, GOSS/MVS distributional sampling, Bayesian
bootstrap, additional heads such as Poisson/NegativeBinomial/StudentT or shared
multi-quantile, and a public custom vector-loss protocol.

Row sampling is selectable with `sampling="uniform"` (default),
`sampling="goss"` plus `top_rate` / `other_rate`,
`sampling="weighted_goss"` for a sample-weight-aware GOSS variant, or
experimental `sampling="mvs"` plus `subsample` / `mvs_reg`. MVS ranks rows from
the current gradient/Hessian magnitude and inverse-probability scales sampled
rows so the histograms stay on the same expected scale.

CatBoost-like stochastic regularization is opt-in:

```
ChimeraBoostClassifier(
    bootstrap_type="bayesian",
    bagging_temperature=0.5,
    sampling="mvs",
    subsample=0.7,
    random_strength=0.5,
)
```

`bootstrap_type="bayesian"` draws per-tree exponential row weights and
normalizes them to mean one. `random_strength` adds deterministic, seed-based
noise only while ordering split candidates; stored split gains and feature
importances remain based on true unnoised gain. All three mechanisms are off
by default, preserving deterministic default fits.

Large-fit preprocessing samples up to 200,000 rows when learning numeric bin
borders, similar to LightGBM's `bin_construct_sample_cnt`. Set
`bin_sample_count=None` to recover exact full-data border learning.
When `sample_weight` is supplied, those numeric borders use weighted quantiles;
`sample_weight=None` and all-ones weights preserve the unweighted border path.

Early-stopping selection can optionally be followed by a full-data refit:

```
clf = ChimeraBoostClassifier(
    early_stopping=True,
    refit=True,
    refit_strategy="exact",
)
clf.fit(X, y)
```

When an explicit or automatic validation set is present, ChimeraBoost keeps the
best validation prefix by default (`use_best_model=True`), matching CatBoost's
package-default behavior. Set `use_best_model=False` when an eval set is only
for monitoring and you want a patience-stopped model to keep all trees through
the stop point. `early_stopping_min_delta` controls patience resets; best-model
selection still uses the true validation argmin.

When `early_stopping=True` and `early_stopping_rounds` is left unset, patience
is resolved from the fitted learning rate as `ceil(5 / lr)`, clipped to
`20..200`. This keeps small learning rates from stopping too early while
preserving explicit numeric patience when supplied.
`early_stopping_min_delta=None` preserves the legacy `1e-9` improvement
tolerance. Pass a nonnegative number for an explicit tolerance or
`early_stopping_min_delta="auto"` to scale the tolerance from the baseline
validation loss.

Automatic validation splitting keeps `validation_fraction=0.1` by default.
Pass `validation_fraction="auto"` to resolve a held-out fraction from Kish
effective sample size, and for regression pass
`validation_strategy="weighted_stratified"` to stratify the validation split by
weighted target quantiles. The splitter caps the number of target strata to the
train/validation capacity and falls back to random splitting when stratification
is not feasible. Weighted target stratification is supported for ungrouped
regression splits; classification and grouped splits use their own explicit
class/group-aware split policies and reject this regression-only strategy.

`l2_leaf_reg` defaults to `"auto"` on the sklearn estimators. The resolver is
conservative: CatBoost-mode fits usually resolve near the historical `3.0`
default, while LightGBM-mode fits use that mode's lower regularization base.
Other structure defaults remain opt-in: `depth="auto"`, `num_leaves="auto"`,
`min_child_samples="auto"`, `min_child_weight="auto"`, and
`cat_smoothing="auto"`. Resolved values and the rule source are recorded in
`auto_params_["auto_structure"]`.

`get_refit_params()` returns the frozen parameters for a manual full-data refit:
it disables early stopping, uses the selected round count, and freezes the
resolved learning rate and resolved auto-structure/categorical-smoothing
values. Strategies `"sqrt"` and `"linear"` scale the selected round count by
the automatic validation split ratio; `"scaled"` aliases `"linear"`.

Training loss is evaluated every round by default for diagnostics. Set
`eval_train_loss=False` to skip that pass when you only care about the fitted
model or validation-set early stopping; validation loss and early stopping are
unchanged.

`histogram_parallelism="row"` enables an experimental row-parallel histogram
builder. The default `"auto"` keeps the measured-best feature-parallel path on
the current benchmark machine; use the row-parallel lane only when profiling
shows it helps your hardware.

Not implemented in `tree_mode="lightgbm"` or `tree_mode="hybrid"`: native
LightGBM categorical splits, DART, GPU training, sparse optimization, monotone
constraints, ranking, custom objectives, custom eval metrics, or LightGBM model
import/export.

Benchmark notes and fair comparison recipes live in
[BENCHMARK_NOTES.md](BENCHMARK_NOTES.md).

Optional Optuna-powered stepwise tuning is available through
`chimeraboost.tuning`:

```
from chimeraboost import ChimeraBoostClassifier
from chimeraboost.tuning import ChimeraBoostSearchCV

search = ChimeraBoostSearchCV(
    ChimeraBoostClassifier(iterations=1000),
    strategy="auto",
    tree_modes=("catboost", "lightgbm"),
    cv=5,
    n_trials=20,
    n_workers=4,
    storage="journal:///tmp/chimeraboost-study.log",
    random_state=0,
)
search.fit(X, y, cat_features=[0], groups=groups, sample_weight=w)
model = search.best_estimator_
```

Install the optional dependency with `pip install chimeraboost[tuning]`. The
tuner owns the CV folds, passes explicit validation sets to the wrappers,
scores with validation weights, and controls per-trial `thread_count` so
multi-process tuning does not oversubscribe Numba threads. `n_trials` is a
global budget across all phases, tree-mode lanes, and worker processes. With
`strategy="auto"`, small budgets use an OptGBM-like joint conditional search
around deterministic CatBoost/LightGBM probes, while larger budgets use the
laned stepwise phases. For example, `n_trials=20` runs two deterministic
tree-mode probes plus 18 compact joint trials; `n_trials=1000` reserves the
same two probes plus a 198-trial joint warm-up before allocating the rest to
structure, sampling/regularization, learning-rate/round, and binning phases.
After probes and warm-up, lane phases bias roughly 70% of each phase budget
toward the currently best tree-mode lane while still reserving budget for the
other lane.
Use `strategy="joint"` or `strategy="stepwise"` to force either mode. Set
`timeout=<seconds>` to stop by wall-clock time, including with
`n_trials=None`; set `early_stop_patience=<completed_trials>` or pass a custom
`study_stopper` callback for study-level stopping. Final refits preserve the
winning trial's model semantics by default: fold-local automatic learning
rates are not frozen and median fold round counts are not applied unless
`refit_learning_rate` or `refit_rounds` explicitly request that behavior.
Parallel search uses separate worker processes sharing Optuna storage; each
worker calls Optuna with `n_jobs=1` so Optuna thread-level parallelism does not
race with Chimeraboost's Numba thread pool.
The default tuning phases leave `random_strength=0.0` because split-score noise
currently uses a slower Python split-scoring path; include the explicit
`"split_noise"` phase when you want to tune that regularizer.
For `loss="Gaussian"`, the tuner resolves to the LightGBM/leaf-wise lane,
uses Gaussian NLL as the default objective, and keeps the
sampling/regularization phase inside the supported uniform row-sampling and
column-sampling surface.

* **To Do:**
    * Update multi-class classification loss scheme
