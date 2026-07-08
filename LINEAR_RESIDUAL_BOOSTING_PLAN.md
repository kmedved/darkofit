# Linear Residual Boosting Workplan

**Status:** proposed implementation plan, based on two Oracle feedback rounds
and current source inspection.
**Question answered:** how to move the useful `lrboost` idea into
ChimeraBoost natively, without adding an external dependency or weakening the
non-pickle model archive contract.
**Primary implementation surface:** `ChimeraBoostRegressor`.

## 0. Recommendation

Add linear residual boosting as opt-in wrapper functionality on
`ChimeraBoostRegressor`, not as a new core tree objective and not as a separate
estimator class.

The wrapper should fit a small internal weighted ridge trend on selected raw
numeric features, train the existing booster on residual targets, and add the
trend back in public prediction methods.

Minimal v1:

- Supported scalar losses: `RMSE`, `MAE`, `Quantile`.
- Supported distributional heads: `Gaussian`, `StudentT`.
- Rejected until v2 offset support: `LogNormal`, `Poisson`,
  `NegativeBinomial`.
- Predictive uncertainty default: residual-distribution uncertainty only;
  fitted linear trend is deterministic.

Default behavior must remain unchanged when `linear_residual=False`.

This matches the current architecture: `sklearn_api.py` already coordinates
validation splitting, tree-mode selection, auto learning-rate probes,
distributional calibration, refit, save/load, and wrapper-only parameters.
The core boosters should continue to receive ordinary targets and remain
unaware that those targets were residualized.

### 0.1 Verified Constraints And Adjudications

The two feedback rounds agree on the main architecture and disagree on several
details. This plan adopts the following decisions.

| Topic | Adjudication | Why |
|---|---|---|
| Public surface | Params on `ChimeraBoostRegressor`, not a new class | The wrapper already owns validation splits, probes, calibration, refit, and wrapper state. |
| Helper shape | New `chimeraboost/linear_residual.py` with a `WeightedRidgeTrend` class plus small private functions | The second review's single `_fit_weighted_ridge` function is too thin for persistence, diagnostics, feature names, and load validation. A class keeps state explicit without using pickle. |
| Default feature selector | `linear_residual_features="auto"`; accept `None` as an alias | `"auto"` is self-documenting in `get_params()`. `None` remains convenient and compatible with the second review. |
| Weight normalization | Normalize positive ridge weights to sum to `n_positive`, and use an average-loss ridge objective, equivalently `alpha_eff = alpha * n_positive` in the weighted-SVD denominator | The core booster normalizes weights internally, but the ridge trend has its own regularization semantics. This choice makes `alpha` invariant to global weight scaling and unaffected by adding zero-weight rows. |
| Constant columns | Drop in `"auto"` mode; explicit selectors raise only if all explicitly selected columns are unusable | Auto mode should be forgiving; explicit mode should surface user mistakes without rejecting mixed useful/constant selections. |
| Prediction hook | Do not change `model_.predict_raw`; use wrapper-local helpers that add the trend only for public predictions | This preserves residual-space diagnostics and avoids pretending the core booster knows about the trend. A private raw-copy helper may be used internally, but only after `_check_predict_input`. |
| Distributional calibration | Fit calibration on residual validation targets; public methods add the trend after residual params are calibrated | This is mathematically equivalent for location heads and keeps the existing `_calibrated_params_from_raw` funnel intact. |
| LogNormal/Poisson/NB | Reject in v1 | Additive residuals break positivity/count domains. These heads need the v2 link-scale offset protocol. |
| Beta uncertainty | Residual-only variance in v1; optional covariance later | Existing tree parameters are treated as fixed at prediction time, so including only linear beta uncertainty by default would be inconsistent and easy to overinterpret. |

## 1. Design Invariants

1. **Wrapper-only v1.** Do not touch tree growth, split scoring, or loss
   gradients for additive residual support.
2. **No validation leakage.** The linear trend must be fit only on the rows
   used to train the residual booster in the current fit phase.
3. **No arbitrary external primary model in v1.** Arbitrary sklearn estimators
   require pickle or a broad serialization protocol, which conflicts with the
   current archive design.
4. **Distribution semantics must be explicit.** Additive residualization is
   correct for location-family targets; count and positive-support heads need
   a link-scale offset protocol instead.
5. **Sample weights are first-class.** The ridge solve and residual booster
   both consume training weights, and validation weights must never enter the
   ridge solve.
6. **Persistence remains plain arrays plus JSON.** Save/load should not
   refit, reselect features, or recompute ridge moments.
7. **Raw booster diagnostics stay residual-space.** Public wrapper methods add
   the trend; the core booster raw score remains the residual/tree surface.

## 2. Public API

Add these constructor params to `ChimeraBoostRegressor.__init__`:

```python
linear_residual=False
linear_residual_alpha=1.0
linear_residual_features="auto"
linear_residual_fit_intercept=True
linear_residual_standardize=True
```

Add all five names to `_SKLEARN_ONLY` so they are not forwarded to
`GradientBoosting` or `DistributionalBoosting`.

### 2.1 Parameter Semantics

`linear_residual`

- `False`: current behavior.
- `True`: fit a weighted ridge trend and train the booster on residuals.

`linear_residual_alpha`

- Nonnegative ridge penalty on slope coefficients in the average weighted
  squared-error objective.
- Intercept is never penalized.
- `0.0` means weighted least squares using the same SVD path with
  pseudoinverse filtering.
- Implementation detail: with positive weights normalized to sum to
  `n_positive`, the weighted-SVD denominator uses
  `singular_value**2 + linear_residual_alpha * n_positive`.
- Validate finite and `>= 0.0`.

`linear_residual_features`

- `"auto"` or `None`: all raw input columns that are not categorical and are
  numeric/coercible to float.
- Integer sequence: raw column indices.
- Boolean mask of length `n_features`.
- String sequence: pandas column names, only when `feature_names_in_` exists.

Reject in v1:

- Callables.
- Duplicate features.
- Out-of-range indices.
- String selectors when feature names are unavailable.
- Explicit selection of a categorical feature.

`linear_residual_fit_intercept`

- Fit an unpenalized intercept by default.
- `False` solves through the origin after imputation/optional
  standardization.

`linear_residual_standardize`

- Default `True`.
- Feature moments are computed on positive-weight training rows.
- Missing values are imputed before scaling.
- When `False`, document that `linear_residual_alpha` is raw-unit dependent.

### 2.2 Fitted Attributes

Always after fit:

```python
linear_residual_enabled_: bool
linear_residual_active_: bool
```

When enabled but inactive:

```python
linear_residual_inactive_reason_: str
```

When active:

```python
linear_residual_alpha_: float
linear_residual_fit_intercept_: bool
linear_residual_standardize_: bool

linear_residual_feature_indices_: np.ndarray  # int64, shape (p,)
linear_residual_feature_names_: np.ndarray | None
linear_residual_dropped_features_: list[dict]

linear_residual_intercept_: float
linear_residual_coef_: np.ndarray             # raw-input-scale, shape (p,)
linear_residual_transformed_coef_: np.ndarray # design-scale, shape (p,)
linear_residual_center_: np.ndarray           # shape (p,)
linear_residual_scale_: np.ndarray            # shape (p,), all > 0
linear_residual_impute_values_: np.ndarray    # shape (p,)

linear_residual_rank_: int
linear_residual_singular_values_: np.ndarray
linear_residual_weight_sum_: float
linear_residual_effective_n_: float
linear_residual_target_mean_: float
linear_residual_trend_train_mean_: float
linear_residual_residual_stats_: dict
```

When `refit=True`, also keep compact selection-phase context:

```python
selection_linear_residual_summary_: dict
```

Do not persist `selection_model_` beyond the existing wrapper behavior.

## 3. Internal Weighted Ridge Helper

Create `chimeraboost/linear_residual.py`. Keep the stateful class small, and
factor the actual SVD math into private helper functions for testability. Do
not use `chimeraboost/linear.py`: the more specific filename avoids confusion
with future GLM/count-link work.

Suggested public-to-package-private shape:

```python
class WeightedRidgeTrend:
    @classmethod
    def fit(
        cls,
        X,
        y,
        sample_weight=None,
        *,
        feature_indices,
        feature_names=None,
        alpha=1.0,
        fit_intercept=True,
        standardize=True,
        explicit_features=False,
    ):
        ...

    def predict(self, X):
        ...

    def to_header(self):
        ...

    def to_arrays(self):
        ...

    @classmethod
    def from_header_arrays(cls, header, arrays):
        ...
```

Contract:

- Inputs are dense and row-aligned with `y`.
- Helper never mutates `X` or `y`.
- `sample_weight=None` means all ones.
- Prediction returns float64 shape `(n_samples,)`.
- Helper owns feature extraction, imputation, standardization, SVD solve,
  diagnostics, and trend serialization.
- Helper imports only NumPy and local validation helpers if needed.
- The class is the persisted state object. A private `_fit_weighted_ridge`
  helper may return low-level arrays, but the wrapper should store and restore
  a `WeightedRidgeTrend` instance, not a loose tuple of arrays.

### 3.1 Feature Extraction

Use raw input columns, not ChimeraBoost-binned or target-stat features.
The residual booster still sees original `X` and the original `cat_features`.

For `"auto"`:

- Start from all non-categorical raw input columns.
- Try float64 coercion.
- Drop non-numeric columns and record `reason="non_numeric"`.
- Drop all-missing columns and record `reason="all_missing"`.
- Drop constant columns and record `reason="constant"`.
- If nothing usable remains, set `linear_residual_active_=False` and fit the
  original model path unchanged.

For explicit selectors:

- Reject categorical columns.
- Reject non-numeric columns.
- Reject all-missing selected columns.
- Constant selected columns may be dropped if at least one selected feature
  remains usable; otherwise raise `ValueError`.

Missing values:

- Treat `NaN`, `inf`, `-inf`, and pandas missing values as missing.
- Compute imputation values from positive-weight finite rows.
- Impute to weighted column means.
- If standardized, imputed values transform to exactly zero.

Constant columns:

- Compute weighted variance after imputation.
- Use a tolerance such as `eps * max(1.0, abs(center_j)) ** 2`.
- Store every dropped feature with index, optional name, and reason.

### 3.2 Weighted Ridge Formula

Let selected and imputed features be `Xsel`, with `n` rows and `p` columns.
Let nonnegative training weights be `w`.

Weight normalization:

```python
if sample_weight is None:
    w_raw = np.ones(n, dtype=np.float64)
else:
    w_raw = np.asarray(sample_weight, dtype=np.float64)

positive = w_raw > 0.0
w = w_raw[positive] * (np.sum(positive) / np.sum(w_raw[positive]))
```

All moments and solves use `positive` rows only. Weight scaling by a positive
constant must not change fitted coefficients. Adding zero-weight rows must not
change the trend fit or the effective ridge penalty.

This intentionally differs slightly from the core booster implementation,
which normalizes weights after the wrapper forwards them into `fit`. The trend
solver receives wrapper-level weights and defines its own average-loss ridge
objective so `linear_residual_alpha` has stable meaning.

Feature moments:

```text
center_j = sum_i w_i x_ij / sum_i w_i
scale_j  = sqrt(sum_i w_i (x_ij - center_j)^2 / sum_i w_i)
```

If `standardize=True`:

```text
z_ij = (impute(x_ij) - center_j) / scale_j
```

If `standardize=False`, use imputed raw values as `z_ij`, but still store
`center`, `scale`, and imputation state for diagnostics/load validation.

Intercept:

If `fit_intercept=True`, solve slopes against centered target:

```text
ybar = sum_i w_i y_i / sum_i w_i
target_i = y_i - ybar
```

If `fit_intercept=False`, solve against raw `y` and set raw intercept to
`0.0`.

SVD solve:

```text
A = sqrt(W) Z
r = sqrt(W) target
A = U diag(d) Vt
```

For `alpha > 0`:

```text
coef_design = V @ ((d / (d^2 + alpha * n_positive)) * (U.T @ r))
```

For `alpha == 0`:

```text
coef_design = V @ ((1 / d) * (U.T @ r))
```

only for singular values satisfying:

```text
d > rcond * d_max
rcond = max(n_positive, p) * eps
```

The `alpha * n_positive` term is equivalent to minimizing:

```text
(1 / n_positive) * sum_i w_i * (target_i - z_i beta)^2
    + alpha * ||beta||_2^2
```

with positive-row weights normalized to mean one.

Do not use normal equations for the main solve; they square the condition
number and make high-collinearity behavior worse.

Raw coefficients for standardized features:

```text
coef_raw_j = coef_design_j / scale_j
intercept_raw = ybar - sum_j coef_design_j * center_j / scale_j
```

If not standardized, `coef_raw = coef_design` and intercept conversion uses
the unstandardized design convention.

Zero-weight rows:

- Excluded from moments, rank, singular values, and residual diagnostics.
- Still accepted at prediction time.

## 4. Fit Data Flow

### 4.1 Placement In `ChimeraBoostRegressor.fit`

The trend is fitted after wrapper input validation and after any automatic
train/validation split, but before:

- `make_model`;
- tree-mode candidate fits;
- auto learning-rate probes;
- distributional calibration;
- final full-data refit.

High-level flow:

```python
X_input = X
X, cat_features, n_features = _coerce_fit_X(X, cat_features)
eval_set = _ensure_dense_eval_set(eval_set)
eval_set = _validate_eval_set_features(...)
y = validate_target_vector(...)
sample_weight = _validate_wrapper_sample_weight(...)

X_full, y_full = X, y
sample_weight_full = sample_weight

# Existing automatic split, if needed.
if automatic_validation_needed:
    train_idx, val_idx, realized_validation_policy = _make_eval_split(...)
    eval_set = (X[val_idx], y[val_idx])
    eval_sample_weight = sample_weight[val_idx] if sample_weight is not None else None
    X, y = X[train_idx], y[train_idx]
    sample_weight = sample_weight[train_idx] if sample_weight is not None else None

# New residualization.
trend = _fit_linear_residual_trend(X, y, sample_weight, cat_features, ...)
y_fit = y - trend.predict(X)
if eval_set is not None:
    X_val, y_val = eval_set
    eval_set_fit = (X_val, y_val - trend.predict(X_val))
else:
    eval_set_fit = None

# Existing booster paths receive y_fit/eval_set_fit, including calibration.
```

Implementation note: keep a separate `eval_set_original` only if needed for
metadata. The object passed to `_run_learning_rate_probe`,
`_fit_tree_mode_auto`, `model.fit`, and residual distribution calibration must
be `eval_set_fit`. Otherwise early stopping and calibration would evaluate
different objectives.

### 4.2 Explicit Eval Set

When user passes `eval_set`, fit the trend only on training `X, y`. Compute
validation residuals with that fitted trend:

```python
y_val_resid = y_val - trend.predict(X_val)
```

Validation weights are passed to booster evaluation/calibration but are never
used by the ridge solve.

### 4.3 Automatic Validation Split

When `early_stopping`, `tree_mode="auto"`, or another selection path creates
an automatic split:

1. Split first using existing `_make_eval_split`.
2. Fit trend on split-training rows only.
3. Residualize split-training and validation targets.
4. Train/probe/select on residual targets.

This is the critical leak-prevention rule.

### 4.4 Tree-Mode Auto

Fit one selection trend before the candidate loop. All tree-mode candidates
use the same residualized target and validation residuals.

Do not refit the ridge trend per candidate. The candidate comparison should
measure tree-mode fit quality, not different linear baselines.

### 4.5 Auto Learning-Rate Probe

Run the probe on the same residualized train/eval data as the final model.
Do not fit a separate trend inside each probe.

The current `_run_learning_rate_probe` already takes `X`, `y`, `eval_set`,
and fit kwargs; pass the residualized `y` and residualized `eval_set`.

### 4.6 Distributional Calibration

Distributional calibration should operate on the residual distribution.

For `Gaussian`/`StudentT`, the selection model sees `y_resid` and predicts
residual location/scale. Existing scale calibration remains valid because:

```text
y - (trend + mu_resid) = y_resid - mu_resid
```

Fit calibration on residual validation targets and residual predictions, then
add the trend back only in public methods. In the current source, calibration
later reads `X_cal, y_cal = eval_set`; after this feature lands, that variable
must refer to the residualized eval set for supported v1 heads.

### 4.7 Refit

When `refit=True` and selection is active:

1. Selection phase:
   - split;
   - fit selection trend on selection-training rows;
   - train selection booster on residuals;
   - fit calibration on residual validation data.
2. Full refit phase:
   - fit a new trend on `X_full, y_full, sample_weight_full`;
   - residualize full target;
   - refit booster with frozen selected rounds/lr/tree mode;
   - keep frozen selection calibration metadata.

Final public predictions use the full-data trend and full-data residual
booster.

Record a compact `selection_linear_residual_summary_`, but preserve existing
behavior that the full selection model is not persisted after load.

## 5. Supported Loss Semantics

### 5.1 V1 Support Table

| Loss/head | v1 action | Public prediction semantics |
|---|---|---|
| `RMSE` | allow | `trend + residual_mean` |
| `MAE` | allow | `trend + residual_median_like_prediction` |
| `Quantile` | allow | `trend + residual_quantile` |
| `Gaussian` | allow | `(trend + mu_resid, sigma_resid)` |
| `StudentT` | allow | `(trend + mu_resid, scale_resid, nu)` |
| `LogNormal` | reject | requires log-location offset protocol |
| `Poisson` | reject | requires log-mean offset protocol |
| `NegativeBinomial` | reject | requires log-mean offset protocol |

### 5.2 Scalar Losses

Train the scalar booster on:

```text
r_i = y_i - trend(x_i)
```

Public prediction:

```text
yhat(x) = trend(x) + residual_booster_prediction(x)
```

`staged_predict` yields trend plus each staged residual prediction.

### 5.3 Gaussian

Residual booster params:

```text
(mu_resid, sigma)
```

Public params:

```text
mu_total = trend + mu_resid
sigma_total = sigma
```

`predict_variance` returns `sigma^2`, after any scale calibration. The
deterministic trend does not change variance.

`predict_interval` shifts residual intervals by the trend.

`sample` draws residual samples and adds the row trend.

### 5.4 StudentT

Residual booster params:

```text
(mu_resid, scale, nu)
```

Public params:

```text
mu_total = trend + mu_resid
scale_total = scale
nu_total = nu
```

`predict_variance` returns:

```text
scale^2 * nu / (nu - 2)
```

after any scale calibration. The trend shifts location only.

### 5.5 LogNormal V2

Do not residualize original `y` for LogNormal. Additive original-space
residuals can be negative and break positive-support predictions, intervals,
and samples.

Correct design is a log-location offset:

```text
offset(x) ~= linear model for E[log(y) | x]
m_total(x) = offset(x) + m_tree(x)
```

Then:

```text
mean(y | x) = exp(m_total + s^2 / 2)
var(y | x) = (exp(s^2) - 1) * exp(2 * m_total + s^2)
```

Scale calibration still targets `s`.

### 5.6 Poisson V2

Do not fit additive residuals on counts.

Correct design is a log-mean offset:

```text
eta_total(x) = offset(x) + eta_tree(x)
lambda(x) = exp(eta_total(x))
```

Mean calibration by scalar multiplier `c` composes as:

```text
eta_calibrated = eta_total + log(c)
```

### 5.7 NegativeBinomial V2

Use the same log-mean offset as Poisson:

```text
eta_total(x) = offset(x) + eta_tree(x)
mu(x) = exp(eta_total(x))
var(y | x) = mu + alpha * mu^2
```

Mean calibration adds `log(c)` to the mean offset. Dispersion calibration
continues to target `alpha` separately.

## 6. V2 Offset Protocol

Do not implement this in the first v1 PR. It is the prerequisite for
`LogNormal`, `Poisson`, and `NegativeBinomial` support.

### 6.1 Core Idea

`DistributionalBoosting` should support an optional raw-score offset:

```python
offset_train_raw: np.ndarray | None  # shape (K, n_train)
offset_eval_raw: np.ndarray | None   # shape (K, n_eval)
```

The tree ensemble still learns `F_tree`, but losses consume:

```text
F_total = F_tree + offset_raw
```

### 6.2 Training

Initialization becomes offset-aware:

- Gaussian/StudentT: initialize location on `y - location_offset`.
- LogNormal: initialize `m` on `log(y) - log_location_offset`.
- Poisson:

```text
exp(c) = sum_i w_i y_i / sum_i w_i exp(offset_i)
```

- NegativeBinomial: same mean-offset init as Poisson, with dispersion handled
  by the existing NB state path.

Gradient/hessian:

```python
F_total = F_tree + offset_train_raw
loss.grad_hess_class_major_into(y, F_total, sample_weight, grad, hess)
```

Tree leaf values update `F_tree` only.

Eval:

```python
F_eval_total = F_eval_tree + offset_eval_raw
loss.eval_class_major(y_eval, F_eval_total, eval_sample_weight)
```

### 6.3 Prediction

Public distributional prediction should use total raw scores:

```python
raw_total = model.predict_raw(X) + offset_model.predict_raw_offset(X)
params = loss.params_from_raw(raw_total)
```

If a diagnostic/tree-only raw path is desired, add a separate name such as
`predict_raw_tree`, not by changing existing public semantics silently.

Calibration composes after offsets:

- Continuous heads: calibrate scale after total location is formed.
- Count heads: mean calibration multiplies mean or adds `log(c)` to total raw
  mean.
- NB dispersion calibration remains independent of the mean offset.

## 7. Prediction Method Changes

Add a private helper on the wrapper:

```python
def _linear_residual_trend(self, X):
    if not getattr(self, "linear_residual_active_", False):
        return None
    return self.linear_residual_trend_.predict(X)
```

Optionally add a second private helper for public prediction plumbing:

```python
def _raw_with_location_trend(self, X):
    raw = self.model_.predict_raw(X)
    trend = self._linear_residual_trend(X)
    if trend is None:
        return raw, None
    shifted = np.array(raw, copy=True)
    if shifted.ndim == 2:
        shifted[:, 0] += trend
    else:
        shifted += trend
    return shifted, trend
```

This helper is only an implementation convenience for location-shift public
methods. It must not be exposed as `model_.predict_raw`, must not mutate the
core raw array in place, and must not be used for unsupported positive/count
heads.

### 7.1 `predict`

Current distributional prediction already uses `loss.mean_from_raw` or
calibrated `mean_from_params`.

New behavior:

- Scalar losses: `raw + trend`.
- Gaussian/StudentT: residual mean plus trend.
- Disabled/inactive: current behavior.

### 7.2 `staged_predict`

For every stage:

- Scalar: `stage_raw + trend`.
- Gaussian/StudentT: `loss.mean_from_raw(stage_raw) + trend`, with calibration
  if applicable and if current code would apply it.

### 7.3 `predict_dist`

For Gaussian:

```python
mu, sigma = residual_params
return mu + trend, sigma
```

For StudentT:

```python
mu, scale, nu = residual_params
return mu + trend, scale, nu
```

Trend addition must happen after residual calibration has produced params, so
mean/scale calibration composition remains simple.

Equivalent implementation: make a copied raw array, add the trend to column 0,
then call `loss.params_from_raw` and existing calibration. This is acceptable
for Gaussian/StudentT because their first raw column is pure location. The
observable contract is still "calibrate residual params, then shift location".

### 7.4 `predict_variance`

v1 deterministic trend:

- Return the residual distribution variance after calibration.
- Do not add beta uncertainty by default.

### 7.5 `predict_interval`

For Gaussian/StudentT:

1. Get residual interval from raw/calibrated params.
2. Add `trend` to both lower and upper bounds.

If using `_raw_with_location_trend`, this can instead call
`interval_from_raw`/`interval_from_params` on copied shifted raw/params.
Do not apply the trend twice.

### 7.6 `sample`

For Gaussian/StudentT:

1. Draw residual samples using existing loss methods.
2. Add `trend[:, None]`.

If using copied shifted raw/params, sampling from the shifted location already
includes the trend. Do not add it a second time.

Confirm sample shape from current losses is `(n_rows, n_samples)`.

## 8. Uncertainty

### 8.1 V1 Policy

Default policy: **residual distribution only**.

The fitted linear trend is treated as deterministic, matching the rest of
ChimeraBoost where fitted tree parameters do not contribute parameter
uncertainty to `predict_variance`.

Record this explicitly:

```python
auto_params_["linear_residual"]["predictive_variance_policy"] = (
    "residual_distribution_only"
)
auto_params_["linear_residual"]["beta_uncertainty_included"] = False
```

### 8.2 V2 Optional Beta Uncertainty

Potential future API:

```python
linear_residual_uncertainty="none" | "ridge_parametric" | "ridge_sandwich"
```

Use an augmented design `B` that includes an intercept column when fitted.
Penalty matrix:

```text
P = diag(0, alpha * n_positive, alpha * n_positive, ...)
```

or no intercept row when intercept is not fitted.
Use the same positive-row mean-one weights and average-loss alpha convention
as the v1 ridge fit.

Let:

```text
M = B.T W B + P
theta_hat = inv(M) B.T W y
```

Parametric covariance, if weights are precision-like:

```text
Cov(theta_hat) = sigma2_hat * inv(M) B.T W B inv(M)
```

Sandwich covariance, if weights behave like importance/frequency weights:

```text
Cov(theta_hat) =
    inv(M) [B.T diag(w_i^2 e_i^2) B] inv(M)
```

Effective degrees of freedom:

```text
df_ridge = trace(B.T W B inv(M))
sigma2_hat = sum_i w_i e_i^2 / max(n_positive - df_ridge, 1)
```

Row trend variance:

```text
v_beta(x) = b(x).T Cov(theta_hat) b(x)
```

Gaussian total variance:

```text
v_total = sigma_calibrated^2 + v_beta
```

StudentT total variance:

```text
v_total = scale_calibrated^2 * nu / (nu - 2) + v_beta
```

Intervals/sampling:

- Gaussian: approximate with `sqrt(v_total)`.
- StudentT: draw residual StudentT plus independent Normal trend noise, or use
  Monte Carlo intervals. Do not claim the sum is still StudentT.
- Calibration: fit residual scale calibration first; add beta uncertainty only
  after calibration.

Do not implement this in v1.

## 9. Persistence

Current persistence uses a JSON header plus plain NumPy arrays in a compressed
archive. `save_booster` already accepts `wrapper_header` and `wrapper_arrays`.

### 9.1 Save Payload

Extend `ChimeraBoostRegressor.save_model` to pass wrapper arrays:

```python
save_booster(
    self.model_,
    path,
    wrapper_header={
        "wrapper_class": type(self).__name__,
        "params": self._wrapper_params_header(),
        "state": self._wrapper_state_header(),
    },
    wrapper_arrays=self._wrapper_arrays(),
)
```

Add `_wrapper_arrays()` on the wrapper. It should return `{}` unless an active
linear-residual trend needs array state. The current classifier wrapper already
uses `wrapper_arrays` for class labels; the regressor currently does not, so
this is a real save/load extension.

Header state additions:

```json
{
  "linear_residual_version": 1,
  "linear_residual_enabled": true,
  "linear_residual_active": true,
  "linear_residual_alpha": 1.0,
  "linear_residual_fit_intercept": true,
  "linear_residual_standardize": true,
  "linear_residual_intercept": 12.345,
  "linear_residual_rank": 4,
  "linear_residual_weight_sum": 120.0,
  "linear_residual_effective_n": 118.7,
  "linear_residual_prediction_mode": "additive_location",
  "linear_residual_supported_loss": "Gaussian",
  "linear_residual_feature_names": ["age", "minutes", "height"],
  "linear_residual_dropped_features": [
    {"index": 5, "name": "team", "reason": "categorical"}
  ]
}
```

Arrays:

```python
wrapper_arrays["linear_residual_feature_indices"] = int64[p]
wrapper_arrays["linear_residual_coef"] = float64[p]
wrapper_arrays["linear_residual_transformed_coef"] = float64[p]
wrapper_arrays["linear_residual_center"] = float64[p]
wrapper_arrays["linear_residual_scale"] = float64[p]
wrapper_arrays["linear_residual_impute_values"] = float64[p]
wrapper_arrays["linear_residual_singular_values"] = float64[k]
```

Inactive payload:

```json
{
  "linear_residual_enabled": true,
  "linear_residual_active": false,
  "linear_residual_inactive_reason": "no_usable_auto_features"
}
```

When disabled, it is acceptable to omit all linear-residual state or store
`enabled=false, active=false`; choose whichever keeps load logic simpler.

### 9.2 Load Validation

Add `_restore_linear_residual_state(state, wrapper_arrays)` or similar.
Also update `ChimeraBoostRegressor.load_model` to keep the third return value:

```python
booster, wrapper_header, wrapper_arrays = load_booster(
    path, return_wrapper_payload=True
)
...
est._restore_wrapper_state(wrapper_header.get("state", {}))
est._restore_linear_residual_state(
    wrapper_header.get("state", {}), wrapper_arrays
)
```

Do not hide array restoration inside `_restore_wrapper_state` unless that
method gains a `wrapper_arrays` argument; the current signature only receives
JSON state.

Rules:

- Existing wrapper-class mismatch behavior remains.
- `ChimeraBoostClassifier.load_model` must reject any active linear-residual
  state.
- If `linear_residual_active=True`, all required arrays must exist.
- Feature indices are 1-D integer, unique, nonnegative, and `< n_features_in_`.
- `coef`, `transformed_coef`, `center`, `scale`, and `impute_values` are
  finite 1-D float arrays of identical length.
- `scale > 0`.
- Stored feature names, if present, match `feature_names_in_[indices]` when
  feature names exist.
- Active v1 linear residual state rejects `LogNormal`, `Poisson`, and
  `NegativeBinomial`.
- Inactive state must not require trend arrays.

Prediction preservation:

- Do not reselect features.
- Do not recompute means/scales.
- Do not refit ridge.
- Use loaded arrays directly.
- Preserve column order exactly.

Round-trip tests should use exact equality where practical, following current
distributional save/load tests.

## 10. Metadata And Diagnostics

Attach to both:

```python
model_.auto_params_["linear_residual"]
model_.auto_params_["diagnostics"]["linear_residual"]
```

Suggested payload:

```python
{
    "enabled": True,
    "active": True,
    "fit_stage": "selection_train" or "full_refit",
    "alpha": 1.0,
    "fit_intercept": True,
    "standardize": True,
    "feature_selector": "auto",
    "n_selected_input_features": 8,
    "n_used_features": 6,
    "dropped_features": [...],
    "rank": 6,
    "condition_number": 123.4,
    "weight_sum": 1000.0,
    "effective_sample_size": 842.0,
    "target_weighted_mean": 0.12,
    "trend_weighted_mean": 0.12,
    "residual_weighted_mean": 0.0,
    "target_weighted_variance": 2.4,
    "residual_weighted_variance": 1.7,
    "weighted_r2_against_constant": 0.2917,
    "eval_weighted_r2_against_constant": 0.18,  # optional, diagnostic-only
    "predictive_variance_policy": "residual_distribution_only",
    "beta_uncertainty_included": False,
}
```

For `refit=True`, final model metadata should include:

```python
"selection_trend": {
    "active": True,
    "n_used_features": 6,
    "weighted_r2_against_constant": 0.27,
}
```

## 11. Implementation Milestones

### M0: Plan Artifact

Deliver this plan. No production code changes.

Acceptance:

- Workplan covers API, math, fit flow, distributions, uncertainty,
  persistence, tests, benchmarks, and phased rollout.

### M1: Ridge Helper And Feature Selection

Files:

- Add `chimeraboost/linear_residual.py`.
- Add focused tests to `tests/test_chimeraboost.py` or a new
  `tests/test_linear_residual.py`.

Tasks:

- Implement `WeightedRidgeTrend`.
- Implement selector normalization.
- Implement missing-value imputation.
- Implement standardization and SVD ridge solve.
- Implement serialization helpers.

Acceptance:

- Helper tests pass for closed-form sanity, ridge shrinkage, intercept,
  weight scaling, zero-weight rows, singular designs, missing values,
  constants, and pandas names.
- No-missing SVD helper output matches sklearn Ridge after applying the
  documented `alpha * n_positive` convention.

### M2: Wrapper Fit Integration

Files:

- `chimeraboost/sklearn_api.py`.

Tasks:

- Add constructor params.
- Add params to `_SKLEARN_ONLY`.
- Validate supported losses.
- Integrate residualization after validation splitting.
- Route residualized targets through normal model/probe/tree-mode paths.
- Implement refit trend behavior.
- Attach metadata.

Acceptance:

- Disabled path matches prior behavior.
- Explicit and automatic eval-set leak tests pass.
- `tree_mode="auto"` and `auto_learning_rate_probe=True` work on residualized
  data.
- `refit=True` uses full-data trend in final predictions.

### M3: Prediction And Distributional Semantics

Files:

- `chimeraboost/sklearn_api.py`.
- Possibly `tests/test_distributional.py`.

Tasks:

- Add trend to `predict`.
- Add trend to `staged_predict`.
- Shift `predict_dist` locations for Gaussian/StudentT.
- Shift intervals and samples.
- Leave `predict_variance` residual-only.
- Reject LogNormal/Poisson/NB with clear messages.

Acceptance:

- RMSE/MAE/Quantile prediction identity tests pass.
- Gaussian params are `(trend + mu_resid, sigma)`.
- StudentT params are `(trend + mu_resid, scale, nu)`.
- Variance unchanged by deterministic trend.
- Interval/sample shift tests pass.

### M4: Persistence

Files:

- `chimeraboost/sklearn_api.py`.
- `chimeraboost/serialization.py` only if validation helpers are better placed
  there, but prefer wrapper-local restore validation if possible.

Tasks:

- Add wrapper array save path for regressor.
- Persist active and inactive trend state.
- Restore trend state on load.
- Validate arrays and reject malformed archives.

Acceptance:

- Active trend round-trips predictions exactly or with `rtol=0, atol=0` where
  current code can guarantee same dot order.
- Inactive trend round-trips.
- Corrupt array length/index/scale tests fail clearly.
- Cross-class loading errors stay correct.

### M5: Benchmarks And Documentation

Files:

- `README.md` or a short section in `BENCHMARK_NOTES.md`.
- `benchmarks/bench_linear_residual.py` or an extension to an existing
  benchmark harness.

Tasks:

- Add extrapolation synthetic benchmark.
- Add interpolation no-regression benchmark.
- Add distributional Gaussian/StudentT lanes.
- Document API and support matrix.

Acceptance:

- Extrapolation benchmark shows expected improvement.
- Interpolation benchmark documents neutral/slight-degradation cases.
- README states unsupported v2 heads clearly.

### M6: V2 Offset Protocol

Files:

- `chimeraboost/booster.py`.
- `chimeraboost/sklearn_api.py`.
- `chimeraboost/losses.py`.
- Tests in `tests/test_distributional.py`.

Tasks:

- Add offset arrays to `DistributionalBoosting.fit`.
- Make init/eval/grad total-raw aware.
- Implement wrapper offset model prediction.
- Enable LogNormal log-location offset.
- Enable Poisson/NB log-mean offsets.

Acceptance:

- LogNormal offset equivalence to Gaussian-on-log-y style tests.
- Poisson/NB offset initialization and mean calibration tests.
- Save/load preserves offset trend.

### M7: Optional Beta Uncertainty

Files:

- `chimeraboost/linear_residual.py`.
- `chimeraboost/sklearn_api.py`.
- Persistence tests.

Tasks:

- Add covariance computation modes.
- Serialize covariance and uncertainty metadata.
- Add variance/interval/sample behavior behind explicit opt-in.

Acceptance:

- Default remains residual-only.
- Opt-in Gaussian total variance includes `v_beta`.
- StudentT opt-in sampling/interval behavior uses Monte Carlo or clear
  approximation.

## 12. Tests

### 12.1 Helper Tests

- Closed-form weighted least squares sanity.
- Parity with `sklearn.linear_model.Ridge(solver="svd")` on no-missing,
  standardized, positive-weight cases after mapping the alpha convention
  correctly (`alpha_eff = alpha * n_positive` for mean-one positive weights).
  Do not require sklearn parity for NaN handling because sklearn Ridge rejects
  NaNs while this helper imputes them.
- Ridge shrinkage as alpha increases.
- Intercept not penalized.
- Weight scale invariance.
- Zero-weight extreme rows ignored.
- Singular and `p > n` designs finite.
- Missing values impute to weighted means.
- Constant columns drop or raise according to auto/explicit mode.
- Pandas name selectors resolve and predict-time name mismatch still raises.

### 12.2 Wrapper Tests

- Disabled mode fixed-seed predictions equal current model.
- `predict == trend + residual_raw` for scalar losses.
- `staged_predict` equals trend plus residual staged predictions.
- Explicit eval-set leak check.
- Automatic validation split leak check.
- Tree-mode auto uses one trend across candidates.
- Auto LR probe uses residualized targets.
- Refit uses full-data trend and selection calibration.
- Sample-weight scaling invariance.
- Zero-weight extreme rows do not affect trend.

### 12.3 Distribution Tests

Gaussian:

- Public `predict_dist` shifts only `mu`.
- `predict_variance` unchanged by trend.
- `predict_interval` bounds shifted by trend.
- `sample` shifted by trend.
- Calibration composes before location shift.

StudentT:

- Public `predict_dist` shifts only `mu`.
- `scale` and `nu` unchanged.
- Variance remains `scale^2 * nu / (nu - 2)`.
- Intervals/samples shifted by trend.

LogNormal/Poisson/NB:

- v1 `linear_residual=True` raises.
- Loading active v1 trend state for these heads raises.
- V2 tests wait for offset protocol.

### 12.4 Persistence Tests

- Active trend save/load.
- Inactive enabled trend save/load.
- Disabled save/load keeps old archives compatible.
- Missing wrapper arrays reject.
- Mismatched array lengths reject.
- Nonfinite coefficients reject.
- Nonpositive scale rejects.
- Feature index out of range rejects.
- Feature names mismatch rejects when names exist.
- Classifier loader rejects active trend archive.

### 12.5 Benchmarks

Extrapolation:

```python
x0_train ~ Uniform(-1, 1)
x0_test  ~ Uniform(1, 2)
x1, x2   ~ Normal
y = 8*x0 - 3*x1 + sin(6*x2) + noise
```

Compare:

- plain `ChimeraBoostRegressor(loss="RMSE")`;
- linear residual RMSE;
- plain Gaussian;
- linear residual Gaussian;
- optional StudentT if heavy-tail noise lane is included.

Report RMSE, MAE, NLL where applicable, coverage, and fit time.

Interpolation:

```python
x_train, x_test ~ same distribution
y = sin(8*x0) + x1*x2 + 0.2*x3 + noise
```

Report whether residualization degrades fit at fixed budget. Do not invent a
promotion threshold; compare robust holdout performance.

Use synthetic data and built-in sklearn datasets for required CI benchmarks.
`fetch_california_housing` is acceptable only as an optional local benchmark
when the dataset is already available, because it can require network access.

## 13. Documentation Updates

README additions:

- Short explanation of `linear_residual=True`.
- Support matrix for scalar/distributional heads.
- Warning that v1 predictive variance is residual-only.
- Example:

```python
reg = ChimeraBoostRegressor(
    loss="Gaussian",
    tree_mode="lightgbm",
    linear_residual=True,
    linear_residual_alpha=1.0,
)
reg.fit(X_train, y_train, eval_set=(X_val, y_val))
mu, sigma = reg.predict_dist(X_test)
```

Internal docs:

- Keep this workplan until M1-M5 are done.
- Once implemented, condense public-facing details into README and benchmark
  notes.

## 14. Do-Not-Do Traps

- Do not fit the linear trend before validation splitting.
- Do not residualize validation targets with a trend fit on train plus
  validation data.
- Do not refit the trend per tree-mode candidate.
- Do not refit the trend per LR-probe candidate.
- Do not implement v1 inside `GradientBoosting` or `DistributionalBoosting`.
- Do not persist sklearn `Ridge` or arbitrary primary models.
- Do not residualize original-space `y` for LogNormal.
- Do not residualize counts additively for Poisson/NB.
- Do not add beta uncertainty to `predict_variance` by default.
- Do not change raw booster prediction semantics without a separate explicit
  API decision.
- Do not expose a trend-shifted raw score as `model_.predict_raw`; any
  trend-shifted raw array must be a private wrapper-local copy.
- Do not apply the trend twice when mixing shifted raw helpers with
  interval/sample helpers.
- Do not claim `linear_residual_alpha` is unit-invariant when
  `linear_residual_standardize=False`.

## 15. Remaining Implementation Choices

The architecture decisions above are resolved. These lower-level choices can
be made during M1/M2 implementation:

1. **Exact equality after load:** if BLAS dot-product ordering varies across
   platforms, tests may need exact equality on stored trend arrays and
   `assert_allclose(..., rtol=0, atol=1e-12)` on final predictions. Prefer
   exact where local NumPy is stable.
2. **Private prediction helper shape:** either add trend directly in each
   public method or use `_raw_with_location_trend` internally. If the helper is
   used, tests must prove interval/sample methods do not double-shift.
3. **Inactive disabled-state persistence:** disabled models can omit
   linear-residual state entirely or store `enabled=false, active=false`. Pick
   the simpler loader branch and keep old archives loadable.

## 16. Suggested First PR Cut

The first implementation PR should stop at M1-M4 plus a tiny README note:

- `linear_residual.py`;
- wrapper params and residualized fit flow;
- scalar + Gaussian + StudentT public prediction semantics;
- save/load;
- focused tests;
- no benchmark claims beyond a small smoke benchmark artifact.

Benchmarks and README expansion should follow once correctness and persistence
are stable.
