# DarkoFit

Fast, flexible machine learning for tabular data with a Python/Numba backend.
The package is currently centered on a full-featured gradient-boosting engine
inspired by CatBoost.

<img width="500" height="500" alt="ChatGPT Image May 26, 2026, 05_12_17 PM" src="https://github.com/user-attachments/assets/ee98a4e2-9fa7-4ef1-9e64-e398f398966c" />

* **What?**
    * Tabular machine-learning package that currently centers on GBDTs
    * Only depends on NumPy, Numba, and scikit-learn
    * Benchmark evidence is tracked against CatBoost, LightGBM, and
      ChimeraBoost. Current out-of-box regression defaults are close to
      ChimeraBoost but trail CatBoost overall; accuracy and speed vary
      materially by dataset. A frozen explicit accuracy profile reached
      ChimeraBoost development parity on the spent 13-dataset panel but still
      trailed CatBoost and awaits unseen confirmation
      ([benchmark notes](BENCHMARK_NOTES.md))
    * Supports sample weights and automatic early stopping

* **Why?**
    * I want to be able to modify my GBDT library at will
    * I know Python and I don't know C

* **How?**

```
# pip install darkofit

from darkofit import DarkoClassifier
clf = DarkoClassifier(early_stopping=True)
clf.fit(X, y, sample_weight=w)
```

Models can be saved without pickle:

```
clf.save_model("model.npz")
clf2 = DarkoClassifier.load_model("model.npz")
```

Pandas, Polars, and PyArrow-style named tables are accepted without making
those libraries mandatory dependencies. Named inputs may pass categorical
columns by name (for example, `cat_features=["team", "position"]`), and
prediction enforces the fit-time column names and order. Duplicate
`cat_features` entries are rejected. NaN is treated as missing; masked arrays,
complex values, infinity, sparse matrices, and undeclared nonnumeric columns
raise clear errors by default. Trusted legacy pipelines may skip feature
infinity checks consistently at fit, explicit evaluation, and prediction with
`sklearn.config_context(assume_finite=True)`; unchecked infinity in numeric
features uses the missing-value bin. Zero-row prediction batches are accepted
and return shape-correct empty outputs, while fit and evaluation data must
remain non-empty.

Exact interventional TreeSHAP is available for scalar oblivious-tree
regressors and binary classifiers. Contributions are reported in the original
input-feature space; classifier values explain raw log-odds. The call sets
`expected_value_`, so each row satisfies the additive identity against the
corresponding raw prediction:

```
import numpy as np

phi = clf.shap_values(X_test, X_background=X_reference)
raw_margin = clf.model_.predict_raw(X_test)
np.allclose(phi.sum(axis=1) + clf.expected_value_, raw_margin)
```

Constant and local-linear leaves are supported, and the deterministic fitted
background is preserved by safe `.npz` serialization. Multiclass,
distributional, global-linear-residual, and non-oblivious models currently
raise `NotImplementedError` instead of returning partial explanations. The
[basketball confirmation](benchmarks/basketball_tree_shap_result.md) matched
ChimeraBoost 0.15.0 attributions exactly and ran at comparable speed on the
frozen sports-data gate.

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

When `learning_rate=None` or `"auto"`, DarkoFit uses a transparent
CatBoost-form rule fitted as `log(lr) ~ log(n) + log(iterations)`, with Kish
effective sample size replacing raw row count when `sample_weight` is supplied.
For materially weighted RMSE fits in CatBoost/oblivious-tree mode, the selector
applies an LR-only DarkoFit correction; unweighted and all-ones-weight fits
keep the raw CatBoost-form value.
The selector also applies a small bounded shrinkage when preprocessing expands
the model feature count heavily relative to effective sample size; this
high-dimensional adjustment is capped at a 15% LR reduction and recorded under
`auto_params_["learning_rate"]`.
LightGBM-mode unweighted fits apply an additional provisional dampening factor
because the CatBoost coefficients were fitted on symmetric-tree regularization,
not DarkoFit's leaf-wise stack.
Set `auto_learning_rate_probe=True` on the sklearn wrappers to run an opt-in
short validation probe around the final-budget resolved automatic learning
rate; the selected explicit rate, candidate scores, full-budget base rate, and
short-budget diagnostic rate are recorded under
`auto_params_["learning_rate_probe"]`. This is disabled by default.
The default boosting budget is `iterations=1000`, and the default numeric bin
budget is `max_bins=254`.

Tree builders are selectable:

```
DarkoClassifier(tree_mode="catboost")  # symmetric/oblivious default
DarkoClassifier(tree_mode="lightgbm")  # leaf-wise, non-oblivious
DarkoClassifier(tree_mode="hybrid")    # experimental shared-prefix then leaf-wise
DarkoClassifier(tree_mode="auto")      # validation-selected tree mode
DarkoClassifier(tree_mode="depthwise") # experimental level-wise
```

Tree modes:

* `tree_mode="catboost"` builds symmetric / oblivious trees and supports
  ordered boosting.
* `tree_mode="lightgbm"` builds DarkoFit's LightGBM-like histogram trees:
  non-oblivious, leaf-wise, best-first CART-style trees. This is not model or
  prediction compatibility with Microsoft LightGBM.
* `tree_mode="hybrid"` uses an experimental non-oblivious tree with a shallow
  shared symmetric prefix followed by best-first leaf-wise expansion. It stores
  and predicts as a normal DarkoFit non-oblivious tree.
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

`ordered_boosting="auto"` (the default) is task-aware in CatBoost/depthwise
modes: it enables the ordered leave-one-out leaf update for classification
and disables it for scalar regression. Categorical regression still uses
ordered target-statistic preprocessing to prevent target leakage; applying
the additional ordered leaf update creates a train/inference gap and hurt
both numeric and categorical real-data guardrails. `MAE` and `Quantile` recompute
leaf values from residual statistics, so the ordered update never applies
to them: `"auto"` resolves off and explicit `ordered_boosting=True` raises
a `ValueError` instead of being silently ignored. Explicit values are
otherwise honored, and the resolved policy is recorded under
`auto_params_["tree"]["ordered_boosting_rule"]`.
Categorical features still use DarkoFit's target-stat preprocessing, not
native LightGBM category-partition splits. CatBoost/depthwise modes use ordered
target statistics; LightGBM and hybrid modes use K-fold target statistics and
include raw category-code features for compatible RMSE/logloss-style scalar and
multiclass categorical fits. MAE and Quantile fits keep target-stat features
only.

For compatible LightGBM-mode multiclass fits,
`multiclass_tree_strategy="auto"` uses a shared vector-valued tree per boosting
round. Pass `multiclass_tree_strategy="per_class"` to force the older one-tree-
per-class route for comparisons.

Distributional regression is available with shared vector-valued leaf-wise
trees. Continuous heads include `loss="Gaussian"`, `loss="LogNormal"`, and
`loss="StudentT"`; count heads include `loss="Poisson"` and
`loss="NegativeBinomial"`:

```
reg = DarkoRegressor(loss="Gaussian", tree_mode="lightgbm")
reg.fit(X, y)
mu = reg.predict(X_test)
mu, sigma = reg.predict_dist(X_test)
lo, hi = reg.predict_interval(X_test, alpha=0.1)
draws = reg.sample(X_test, n_samples=100, random_state=0)
```

Gaussian and LogNormal return two parameters, StudentT returns
`(mu, scale, nu)`, Poisson returns `(lambda,)`, and NegativeBinomial returns
`(mu, alpha)` where `Var = mu + alpha * mu**2`. `predict()` returns the
predictive mean for every head, and `predict_variance()` returns the variance
that downstream filters should consume.

Distributional fits use `tree_mode="lightgbm"` (aliases such as `"leafwise"`
are accepted), with uniform row subsampling (`subsample < 1`) and column
subsampling (`colsample < 1`) supported. GOSS/MVS sampling, Bayesian bootstrap,
ordered boosting, and float32 histograms are still rejected. `min_child_weight`
is evaluated on the summed multi-head Hessian mass, while `min_child_samples`
keeps its usual row-count meaning.
Gaussian, LogNormal, and StudentT standardize their canonical continuous target
internally and transform public distribution parameters back to the original
scale, so raw-unit targets such as prices or volumes do not change the Newton
regularization regime.

Validation and early-stopping use each distribution's NLL by default. For
Gaussian, pass `eval_metric="crps"` to select the best validation prefix by
closed-form Gaussian CRPS instead:

```
reg = DarkoRegressor(
    loss="Gaussian",
    tree_mode="lightgbm",
    early_stopping=True,
    eval_metric="crps",
)
```

When a tuning run uses `eval_metric="crps"`, early stopping and best-prefix
selection use CRPS, but trial ranking still uses the configured scorer; the
default Gaussian scorer remains NLL unless `scoring=` is set explicitly.

For small data, early stopping is especially important for Gaussian fits:
training too long can make the log-standard-deviation head overfit residuals
and produce intervals that are too narrow. Use an explicit `eval_set` or
`early_stopping=True` when interval calibration matters.

`dist_calibration="scalar"` is an opt-in validation-set calibration. For
Gaussian/LogNormal it fits the NLL-optimal global scale
`sqrt(weighted_mean(((y - mu) / sigma) ** 2))` on the selected validation
prefix; for StudentT it fits the scale by validation t-NLL; for Poisson it
fits the closed-form mean multiplier; and for NegativeBinomial it fits the
mean or dispersion multiplier by validation NB-NLL.
`dist_calibration="affine"` fits the continuous-head map
`scale' = exp(a + b * log(scale))`,
which can fix compressed scale ranges where low-scale bins are conservative
and high-scale bins under-cover. `dist_calibration="per_metric_affine"` fits
the same map per group, defaulting to the `metric_code` column name or using
`dist_calibration_feature=<column index/name>`, with the global affine map as
the fallback for unseen or small validation groups. The deprecated
`sigma_calibration` alias is still accepted for Gaussian for one release.
Calibration applies to
`predict_dist`, `predict_variance`, `predict_interval`, `sample`, and to
`predict()` when the calibrated parameter changes the predictive mean.
`predict_raw()` remains the uncalibrated fitted score surface. With
`refit=True`, the calibration is frozen from the selection-phase validation
model and then applied to the full-data refit.

`DarkoRegressor` also has opt-in linear residual boosting via
`linear_residual=True`. Before fitting trees, the wrapper fits a weighted ridge
trend on selected numeric raw input columns, trains the booster on residuals,
and adds the deterministic trend back at public prediction time:

```
reg = DarkoRegressor(
    loss="Gaussian",
    tree_mode="lightgbm",
    linear_residual=True,
    linear_residual_alpha=1.0,
)
reg.fit(X, y)
mu, sigma = reg.predict_dist(X_test)
```

`linear_residual_features="auto"` uses all usable non-categorical numeric raw
columns, while an explicit list of column indices or names can be supplied.
The v1 additive-location protocol supports `RMSE`, `MAE`, `Quantile`,
`Gaussian`, and `StudentT`; it intentionally rejects `LogNormal`, `Poisson`,
and `NegativeBinomial` until those heads have distribution-specific offset
protocols. For distributional fits, the trend shifts only the location
parameter and intervals/samples; `predict_variance()` remains the residual
distribution variance and does not include ridge coefficient uncertainty.
Diagnostics are stored under `model_.auto_params_["linear_residual"]`, and the
plain-array model archive preserves the trend without pickle.

Experimental per-leaf linear models are available for controlled research via
`linear_leaves=True`:

```
reg = DarkoRegressor(
    loss="RMSE",
    tree_mode="catboost",
    linear_leaves=True,
    linear_lambda=1.0,
)
```

The option is deliberately off by default. It fits a ridge-regularized local
linear model over each oblivious tree's numeric split features while leaving
DarkoFit's split search unchanged. The initial implementation requires scalar
RMSE, CatBoost/oblivious trees, plain (non-ordered) leaf updates, at least 1,000
training rows, and at least one numeric feature; ineligible small or
all-categorical fits record an exact constant-leaf fallback. Packed prediction,
safe `.npz` persistence, and fitted diagnostics under
`model_.auto_params_["linear_leaves"]` are supported. This remains an explicit
experimental mechanism: validation-selected use must pass the noisy-data and
cold-player basketball gates before any automatic policy is considered.
The first frozen basketball screen failed mean, leave-one-fold-out, team, and
cold-player quality gates, so neither direct use nor that validation selector
is an automatic default; see
[`benchmarks/basketball_linear_leaves_result.md`](benchmarks/basketball_linear_leaves_result.md).

Distributional benchmark, mean over three seeds on the synthetic
heteroscedastic gate. The calibrated DarkoFit row was refreshed after the
0.7.0 target-standardization change; external baselines are retained from the
same public benchmark matrix because they are not affected by DarkoFit's
internal target transform.

| model | NLL 100k | NLL 500k | CRPS 500k | cov90 500k | fit s 500k |
| --- | ---: | ---: | ---: | ---: | ---: |
| DarkoFit Gaussian, early-stopped + calibrated | **0.990** | **0.983** | **0.389** | 0.899 | 15.4 |
| NGBoost Normal | 1.014 | 1.008 | 0.396 | 0.905 | 125.9 |
| CatBoost `RMSEWithUncertainty` | 1.058 | 1.056 | 0.410 | 0.909 | 0.8 |
| LightGBM twin-model variance hack | 1.644 | 1.630 | 0.419 | 0.619 | 3.5 |

Command and per-seed rows live in
[BENCHMARK_NOTES.md](BENCHMARK_NOTES.md) and
[benchmarks/distributional_summary.md](benchmarks/distributional_summary.md);
the post-standardization calibrated DarkoFit check is in
[benchmarks/distributional_standardization_check.md](benchmarks/distributional_standardization_check.md).
A WNBA DARKO real-data observation check is also recorded in
[benchmarks/wnba_realdata_distributional_summary.md](benchmarks/wnba_realdata_distributional_summary.md):
per-metric affine Gaussian improves the held-out 2024-2026 one-step scale
check (NLL 0.404, CRPS 0.391, coverage 0.901, pooled sigma-bin RMS
`1.002/0.934/1.002/1.035/0.989`). The companion shadow replay in
[benchmarks/wnba_kalman_replay_summary.md](benchmarks/wnba_kalman_replay_summary.md)
injects `predict_variance()` as row-level `R_t`. The best current shadow lane,
StudentT(30) with a validation-tuned incumbent blend, reaches statistical
parity with the incumbent `sigma2 / sample_weight` heuristic (NLL 0.113884 vs
0.113910, RMSE 0.864423 vs 0.864424) while improving overall innovation
calibration (NIS 0.994 vs 0.982). The tiny likelihood gap is inside the
paired-bootstrap noise band and the lane still does not clear the strict
2-of-3 season replacement gate, so it is not yet a production replacement for
DARKO observation noise. The WNBA/DARKO artifacts were generated before the
0.7.0 target-standardization change and should be rerun before production replay
or release claims that depend on those exact sigma values.

Not implemented for distributional regression in v1: CatBoost-style
per-parameter scalar trees, GOSS/MVS distributional sampling, Bayesian
bootstrap, heterodispersion for NegativeBinomial, shared multi-quantile, and a
public custom vector-loss protocol.

Row sampling is selectable with `sampling="uniform"` (default),
`sampling="goss"` plus `top_rate` / `other_rate`,
`sampling="weighted_goss"` for a sample-weight-aware GOSS variant, or
experimental `sampling="mvs"` plus `subsample` / `mvs_reg`. MVS ranks rows from
the current gradient/Hessian magnitude and inverse-probability scales sampled
rows so the histograms stay on the same expected scale.

CatBoost-like stochastic regularization is opt-in:

```
DarkoClassifier(
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
clf = DarkoClassifier(
    early_stopping=True,
    refit=True,
    refit_strategy="exact",
)
clf.fit(X, y)
```

When an explicit or automatic validation set is present, DarkoFit keeps the
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

Fit-time callbacks can observe each boosting prefix and stop between rounds.
`WallClockStopper` uses a monotonic clock and treats its deadline as soft: a
tree already being built is allowed to finish before the next check.

```python
from darkofit import DarkoRegressor, WallClockStopper

model = DarkoRegressor(iterations=10_000).fit(
    X,
    y,
    eval_set=(X_val, y_val),
    callbacks=WallClockStopper(300, safety_margin=5),
)
print(model.model_.training_metadata_)
```

The fitted metadata records requested, attempted, completed, and retained
rounds; the best prefix; and the stop reason. With `tree_mode="auto"`, the same
callback objects and monotonic deadline are shared across the CatBoost,
LightGBM, and hybrid candidates. Candidate metadata records each fit's rounds,
score, learning rate, stop reason, and deadline state; candidates not yet
started when the deadline expires are marked `skipped_deadline` without paying
another setup pass. Callbacks remain intentionally rejected with automatic
learning-rate probes or `refit=True`, whose additional-fit budget semantics are
not defined.

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
values. For distributional models with calibration, the exported refit params
clear `dist_calibration`/`sigma_calibration` because the frozen map is fitted
metadata, not something a validation-free refit can recompute. Strategies `"sqrt"` and
`"linear"` scale the selected round count by the automatic validation split
ratio; `"scaled"` aliases `"linear"`.

Per-round training-loss evaluation is off by default
(`eval_train_loss=False`) because it is diagnostic-only and costs one full
O(n) pass per round (about 15% of multiclass fit time); validation loss and
early stopping are unchanged. Set `eval_train_loss=True` to populate
`train_history_`, or use `verbose=True`, which forces it on for progress
logging.

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
`darkofit.tuning`:

```
from darkofit import DarkoClassifier
from darkofit.tuning import DarkoSearchCV

search = DarkoSearchCV(
    DarkoClassifier(iterations=1000),
    strategy="auto",
    tree_modes=("catboost", "lightgbm"),
    cv=5,
    n_trials=20,
    n_workers=4,
    storage="journal:///tmp/darkofit-study.log",
    random_state=0,
)
search.fit(X, y, cat_features=[0], groups=groups, sample_weight=w)
model = search.best_estimator_
```

Install the optional dependency with `pip install darkofit[tuning]`. The
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
`study_stopper` callback for study-level stopping. Final refits match the
early-stopped models CV actually scored: the defaults
`refit_rounds="median_best"` and `refit_learning_rate="fold_median"` cap the
refit at the median fold-best round count and freeze the median fold learning
rate, instead of rerunning the full nominal iteration budget with early
stopping disabled. Pass `refit_rounds="preserve"` and/or
`refit_learning_rate="preserve"` to recover the previous
full-budget/re-resolved-rate refit semantics.
When distributional calibration is reattached from validation folds, the
refit always uses the median fold-best round count so the transported
calibration is applied to a comparable boosting horizon.
Parallel search uses separate worker processes sharing Optuna storage; each
worker calls Optuna with `n_jobs=1` so Optuna thread-level parallelism does not
race with DarkoFit's Numba thread pool.
The default tuning phases leave `random_strength=0.0` because split-score noise
currently uses a slower Python split-scoring path; include the explicit
`"split_noise"` phase when you want to tune that regularizer.
For `loss="Gaussian"`, the tuner resolves to the LightGBM/leaf-wise lane,
uses Gaussian NLL as the default objective, and keeps the
sampling/regularization phase inside the supported uniform row-sampling and
column-sampling surface.

* **To Do:**
    * Update multi-class classification loss scheme
