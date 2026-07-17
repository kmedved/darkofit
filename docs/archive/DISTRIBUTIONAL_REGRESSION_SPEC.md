# Implementation Spec: Native Distributional Regression (`loss="Gaussian"`)

**Status:** implemented in the current working tree; optional external
comparison lanes still require installing NGBoost, CatBoost, and LightGBM
**Target:** DarkoFit `main` (line references as of commit `3029388` + current working tree; treat them as anchors, re-locate by symbol name if drifted)
**Audience:** implementing agent (Codex). This document is self-contained: every integration point, buffer layout, and formula needed is specified here. When this spec and the source disagree on a line number, trust the symbol name and the described behavior.

---

## 0. Revision notes from Oracle review

This revision incorporates three independent Oracle reviews of the first draft. Changes were made where the reviewers identified concrete implementation failures against current source:

- Loss kernels now explicitly handle `sample_weight is None` with `if/else`, skip zero-weight rows before any `sigma`/`z` arithmetic, and clip standardized residuals before squaring. This prevents valid zero-weight rows from poisoning histograms with `0 * inf = nan`, avoids Numba Optional typing pitfalls, and bounds the raw rho Newton step.
- `DistributionalBoosting.fit` now spells out the full fit-time preamble: validation, normalization of tree/sampling/dtype state, sample-weight normalization, auto-structure resolution, `l2_leaf_reg_`, early-stopping min-delta finalization, diagnostics, importance, cache reset, timing, and split-seed initialization. The earlier draft assumed too much from sibling boosters.
- Wrapper routing now preserves the existing dict-taking `make_model(model_kw)` contract used by tree-mode auto and the learning-rate probe, routes the refit model through the same factory, rejects `tree_mode="auto"` early for Gaussian, respects LightGBM aliases via normalized tree-mode checks, and restores Gaussian dispatch when loading wrapper archives.
- Serialization now names the required imports and the exact `kind == "multi"` output-width edit. Gaussian archives write `n_outputs`, not fake `n_classes`; multiclass continues to use `n_classes`.
- Performance language was narrowed: the training loop avoids avoidable per-round transposes/histogram allocations, but the builder still allocates per-tree workspace, and current flat prediction routing does not prefer flattened explicit-node leaf-wise trees.
- Test guidance changed from duplicate-row weight equivalence to invariants that hold under DarkoFit's mean-normalized weight convention.

One Oracle comment was not adopted as written: `DistributionalBoosting` does not need its own `_include_cat_codes` method for correctness because `_BaseBooster._include_cat_codes` exists in current source and returns true for lightgbm/hybrid. The spec still requires extending the RMSE-specific smoothing gates so Gaussian receives RMSE-style categorical treatment.

## 0.1 Post-implementation calibration revision

A follow-up rounds sweep corrected the initial benchmark diagnosis. The
n=800/20-round smoke over-coverage was an under-training snapshot: as rounds
increase, coverage crosses nominal and eventually collapses because the
log-σ head overfits shrinking train residuals. At 200k rows and 60 rounds the
original Gaussian lane was essentially calibrated; on small data, the
production risk is late-stage overconfidence, not intervals being too wide.

Changes adopted from that evidence:

- The benchmark adds a `darkofit_gaussian_es` lane with validation NLL early
  stopping, plus coverage binned by predicted σ. This distinguishes underfit,
  well-stopped, and σ-overfit regimes and shows whether miscalibration depends
  on predicted dispersion.
- The sklearn wrapper adds opt-in `sigma_calibration="scalar"` for Gaussian
  regressors. It requires an explicit validation set or an automatic
  early-stopping split, computes the closed-form NLL-optimal global scale
  `sqrt(weighted_mean(z²))` at the selected validation prefix after any
  best-model truncation, persists it through wrapper state, and applies it to
  `predict_dist`, `predict_interval`, and `sample`. It does **not** alter
  `predict_raw` or `predict`.
- The calibration fold is deliberately the same fold used for early stopping.
  This introduces a small selection bias because the fold also chose the best
  prefix, but the full benchmark found the effect negligible. Fits with a
  scalar calibration effective validation size below 200 record
  `small_sigma_calibration_fold` in `auto_params_["diagnostics"]["warnings"]`.
- With `refit=True`, the scalar scale is frozen from the selection-phase model
  because the full-data refit has no validation target left for calibration.
- OOF calibration, affine `a + b·log σ`, per-head LR/L2/gain knobs, and
  two-stage μ-then-joint training stay out of scope. They add fit cost or
  tuning surface without evidence yet; affine calibration should only be
  revisited if σ-binned coverage shows dispersion-dependent residual error
  that a scalar cannot fix.

## 0.2 As-built surface beyond the initial draft

The implementation intentionally surpassed the first version of this spec in
four places, and this document now treats them as part of v1:

- Uniform row subsampling and column subsampling are supported for Gaussian
  LightGBM-mode fits. Empty sampled draws keep one high-payload row, and
  sampled/masked depth-0 rounds retry up to the same capped guard used by the
  shared-vector path.
- `eval_metric="crps"` is public for Gaussian validation history,
  early-stopping patience, and best-prefix truncation.
- `DarkoStepwiseSearchCV` supports Gaussian regressors on the resolved
  LightGBM lane with a Gaussian-safe search space.
- `sigma_calibration="scalar"` is public, opt-in, validation-only calibration.

GOSS/MVS row sampling, Bayesian bootstrap, ordered boosting, non-LightGBM tree
modes, and float32 vector histograms remain rejected.

---

## 1. Summary

Add a heteroscedastic Gaussian regression head to DarkoFit: one model that jointly predicts a per-row mean **μ(x)** and standard deviation **σ(x)** by minimizing Gaussian negative log-likelihood (NLL), trained with **shared vector-valued trees** (two outputs per leaf) using the natural gradient (Fisher preconditioning). This is the same model family as NGBoost / CatBoost `RMSEWithUncertainty` / LightGBMLSS, but implemented natively on DarkoFit's existing multiclass shared-vector machinery.

User-facing result:

```python
from darkofit import DarkoRegressor

reg = DarkoRegressor(loss="Gaussian", tree_mode="lightgbm", early_stopping=True)
reg.fit(X, y, sample_weight=w)

mu = reg.predict(X_test)                      # point prediction = mean (unchanged API)
mu, sigma = reg.predict_dist(X_test)          # per-row mean and std
lo, hi = reg.predict_interval(X_test, alpha=0.1)   # central 90% interval
draws = reg.sample(X_test, n_samples=100, random_state=0)  # (n, 100) Monte Carlo draws

reg.save_model("model.npz")                   # round-trips predict_dist exactly
```

### Why this is cheap to build here

The hard parts already exist and are reused unmodified:

| Needed for distributional head | Already exists | Where |
|---|---|---|
| Vector-valued leaf-wise tree builder, K-generic | `build_leafwise_multiclass_tree` | tree.py:5864 |
| (K, n_feat, leaves, bins) histogram kernels + subtraction trick | `_build_multiclass_histograms_counts_into`, `_refill_multiclass_{right,left}_subtract_*` | tree.py:3079, 3142, 3179 |
| Split gain summed over K outputs, Newton leaf values per output | `_best_multiclass_splits_for_leaf_ids_counts`, `_multiclass_leaf_values_and_sums` | tree.py:3219, 2439 |
| Fast in-place train-F update (leaf gather, no re-traversal) | `add_multiclass_leaf_values_inplace` | tree.py:2485 |
| Vector-leaf prediction (serial + parallel) | `MultiNonObliviousTree.add_predict_class_major` | tree.py:3809 |
| Flat compiled ensemble with 2-D leaf values | `FlatNonObliviousEnsemble(vector_values=True)` | flat_model.py:318–384 |
| npz serialization of vector-leaf trees | `_pack_nonoblivious(..., vector_values=True)` (tree kind `"multi"`) | serialization.py:320–350, 687–698 |
| Weighted early stopping / best-model truncation / patience | `_BaseBooster` machinery, `_truncate_to_best_model` | booster.py:1003 |
| Auto learning rate w/ Kish ESS | `resolve_learning_rate_details` | auto_params.py |

The tree builder makes **zero classification assumptions**: `K, n_samples = grad.shape` (tree.py:5893), no `K>=3` checks, no K==2 special cases, no softmax anywhere in tree.py. Its current full signature (tree.py:5864 — note it has grown beyond older docs):

```python
def build_leafwise_multiclass_tree(
    X_binned, grad, hess, n_bins_per_feature, max_depth, l2, lr,
    min_gain=1e-8, feature_mask=None, min_child_weight=1.0,
    hist_buffers=None, return_training_state=False, X_hist_binned=None,
    X_route_binned=None,
    max_leaves=None, min_gain_to_split=None, min_child_samples=20,
    reuse_leaf_histograms=True,
    random_strength=0.0, split_seed=0, tree_iteration=0,
    grad_row_major=None, hess_row_major=None,
    leaf_dtype="int64",
)
```

If `grad_row_major`/`hess_row_major` are omitted, the builder allocates contiguous `(n, K)` transposes internally **every round** (tree.py:5896–5903) — the caller must supply preallocated buffers (§5.2). Classification is hardcoded only in `MulticlassBoosting` (label encoding, one-hot, `MultiSoftmax`) — which is why we add a new small booster class instead of touching it.

### Non-goals (v1, updated for v1.1 sampling)

- No CatBoost/oblivious/hybrid/depthwise tree modes for this loss (v1 = `tree_mode="lightgbm"` only, same restriction as `multiclass_tree_strategy="shared_vector"`).
- Uniform row subsampling (`sampling="uniform"`, `subsample < 1.0`) and column subsampling (`colsample < 1.0`) are supported as of the v1.1 pass because the Gaussian loop now mirrors the shared-vector retry logic for sampled `tree.depth == 0` rounds. GOSS/MVS row sampling, Bayesian bootstrap, and ordered boosting still raise clear errors.
- No other distribution heads (Poisson, NB, multi-quantile) — but the design leaves them one loss-class away (§12).

---

## 2. Math specification

### 2.1 Parametrization

Raw model output per row is a 2-vector, stored **class-major** to match the existing multiclass layout:

- `F[0, i] = mu_i` — the mean, in target units.
- `F[1, i] = rho_i = log(sigma_i)` — log standard deviation.

`K = 2` everywhere the multiclass machinery says "classes". Head index 0 is always μ, head index 1 is always ρ.

### 2.2 Loss

Per-row weighted NLL (weights follow the existing convention: multiply grad, hess, and eval contributions by `w_i`; see `_softmax_class_major_grad_hess_into` losses.py:265–292):

```
z_i    = (y_i − mu_i) / sigma_i,        sigma_i = exp(rho_i)
nll_i  = 0.5·log(2π) + rho_i + 0.5·z_i²
L      = Σ w_i · nll_i / Σ w_i
```

### 2.3 Gradients (∂nll/∂F)

```
g_mu_i  = ∂nll/∂mu  = −z_i / sigma_i  = (mu_i − y_i) / sigma_i²
g_rho_i = ∂nll/∂rho = 1 − z_i²
```

### 2.4 Hessians — natural gradient via Fisher preconditioning

Default and only v1 mode is `hessian_mode="natural"`: use the **Fisher information diagonal** in place of the observed Hessian:

```
h_mu_i  = 1 / sigma_i²        (this also equals the exact ∂²nll/∂mu²)
h_rho_i = 2.0                 (Fisher; the exact value 2·z_i² collapses to 0 at z=0 and destabilizes Newton steps)
```

Because leaf values are the Newton step `−G/(H + l2)` per output (tree.py:2455, `_multiclass_leaf_values_and_sums`), feeding Fisher diagonals makes each leaf update the **natural-gradient step** — the NGBoost trick, obtained for free from the existing second-order leaf formula. Consequences worth knowing:

- μ head: leaf value ≈ σ²-weighted mean of residuals scaled back — rows the model is confident about (small σ) dominate. This is the correct MLE behavior and self-stabilizes: `g_mu` can be huge when σ is small, but `h_mu` grows identically, so the Newton ratio stays bounded.
- ρ head: leaf value ≈ mean of `(z² − 1)/2`; with the v1 standardized-residual clip in §2.6, the raw update is explicitly bounded. Without that clip a single extreme outlier can push raw ρ far above the prediction clip and recover only slowly.

`GaussianNLL.__init__(hessian_mode="natural")` should accept and store the kwarg with `"natural"` as the only valid value in v1 (validate, raise on anything else) so the knob exists for experimentation without wrapper exposure.

The existing per-output guard `H[k,l] > 0` in `_multiclass_leaf_values_and_sums` (tree.py:2454) and the summed-hessian count gate `Σ_k hess[k,i] > 0` in the histogram kernels (tree.py:3103) are always satisfied here (h_mu > 0, h_rho = 2w > 0 for w > 0), so no kernel changes are needed.

### 2.5 Initialization (base score)

```
mu_0  = weighted_mean(y, w)
var_0 = weighted_mean((y − mu_0)², w)          # biased MLE variance, matches NLL objective
rho_0 = 0.5 · log(max(var_0, 1e-12))
rho_0 = clip(rho_0, −10.0, 10.0)               # constant-y and degenerate targets stay finite
```

Returns `np.array([mu_0, rho_0])`, shape `(2,)` — the analogue of `MultiSoftmax.init_class_major` (losses.py:379). F is initialized `np.tile(init_[:, None], (1, n))` exactly as booster.py:2113.

### 2.6 Numerical safety (inside the numba kernels)

Module-level constants in losses.py:

```python
_GAUSS_RHO_MIN = -15.0   # sigma >= 3.06e-7
_GAUSS_RHO_MAX = 15.0    # sigma <= 3.27e6
_GAUSS_Z_CLIP = 10.0     # bounds z^2 and the rho-head Newton ratio
```

Every kernel that exponentiates ρ first computes `r = min(max(F[1, i], _GAUSS_RHO_MIN), _GAUSS_RHO_MAX)`. This bounds `1/sigma²` at ~1.07e13 — large but finite in float64, and l2_leaf_reg plus the Fisher self-scaling keep μ leaf values sane.

Before any `z * z`, compute `z = (y[i] - F[0, i]) / sigma` and then clip the standardized residual used by the gradient/eval kernels:

```python
if z < -_GAUSS_Z_CLIP:
    z = -_GAUSS_Z_CLIP
elif z > _GAUSS_Z_CLIP:
    z = _GAUSS_Z_CLIP
```

This makes the optimized training objective a clipped-tail Gaussian NLL outside `|z| > 10`. That is a deliberate v1 robustness tradeoff: it prevents the ρ head from taking effectively unrecoverable giant positive raw steps on outliers, keeps eval/CRPS finite for extreme but valid finite targets, and still leaves the loss exactly Gaussian throughout the ordinary residual range. The finite-difference test should sample interior `|z| < 5` points for exact gradient checks and separately assert clipping behavior.

Zero-weight rows must be skipped before any `sigma`, `z`, `z*z`, NLL, or CRPS arithmetic. `_validate_sample_weight` allows individual zeros, and the vector histogram kernels still accumulate gradient/hessian payloads even when a row's summed hessian does not count toward `hc`; therefore zero-weight rows must write exact zero gradients/hessians and contribute nothing to eval reductions.

`predict_dist` applies the same clip when converting raw ρ to σ so train-time and predict-time σ agree.

### 2.7 CRPS (evaluation metric, closed form)

For reporting and benchmarks, and for CRPS-based validation selection when
`DarkoRegressor(loss="Gaussian", eval_metric="crps")` is requested:

```
z      = (y − mu) / sigma
Φ(z)   = 0.5 · (1 + erf(z / √2))
φ(z)   = exp(−z²/2) / √(2π)
crps_i = sigma · ( z·(2Φ(z) − 1) + 2φ(z) − 1/√π )
CRPS   = Σ w_i · crps_i / Σ w_i
```

`math.erf` is numba-supported; implement as an `@njit(cache=True, parallel=True)` kernel alongside the loss.

---

## 3. Architecture decision: new `DistributionalBoosting` class

**Do not modify `MulticlassBoosting`.** It hardcodes classification in seven places (label extraction `np.unique(y)` booster.py:1985, one-hot `_one_hot_class_major` booster.py:97, `MultiSoftmax(K)` booster.py:2003, label-based eval, `classes_` persistence, per_class fallback, strategy plumbing). Threading a regression loss through that would risk the most-used path in the library for no benefit.

Instead add `DistributionalBoosting(_BaseBooster)` in booster.py. Its `fit` is a *simpler* sibling of the shared-vector section of `MulticlassBoosting.fit` (booster.py:2157–2293): no one-hot, no per_class route, no GOSS/MVS/ordered branches, and only the minimal uniform-row/feature-mask sampling path added in v1.1. Expected size ≈ 200 lines, nearly all of it calls into existing helpers. It must expose the same attribute surface as `GradientBoosting` so the sklearn wrapper, refit machinery, and serialization treat it uniformly:

`loss_name`, `loss_kwargs`, `loss_`, `init_` (shape `(2,)`), `trees_` (list of `MultiNonObliviousTree`), `lr_`, `iterations_`, `best_iteration_`, `best_score_`, `train_history_`, `valid_history_`, `auto_params_`, `prep_`, `use_best_model_`, `early_stopping_rounds_`, `feature_importances_` (if `GradientBoosting` exposes it — mirror whatever `_rebuild_importance_from_trees` provides; vector-tree gains already sum over outputs, tree.py:3285), plus `n_outputs_ = 2`.

---

## 4. Component 1 — losses.py: `GaussianNLL`

### 4.1 Kernels (new, at module level near the softmax kernels)

Add `import math` to losses.py for CRPS (`math.erf`, `math.sqrt`).

```python
_GAUSS_RHO_MIN = -15.0
_GAUSS_RHO_MAX = 15.0
_GAUSS_Z_CLIP = 10.0
_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


@njit(cache=True, parallel=True)
def _gaussian_nll_grad_hess_into(y, F, sample_weight, grad_out, hess_out):
    # y: (n,) float64; F: (2, n) class-major raw scores (mu, rho)
    # grad_out/hess_out: (2, n) float64, written in place
    n = F.shape[1]
    for i in prange(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                grad_out[0, i] = 0.0
                grad_out[1, i] = 0.0
                hess_out[0, i] = 0.0
                hess_out[1, i] = 0.0
                continue
        r = F[1, i]
        if r < _GAUSS_RHO_MIN:
            r = _GAUSS_RHO_MIN
        elif r > _GAUSS_RHO_MAX:
            r = _GAUSS_RHO_MAX
        sigma = np.exp(r)
        inv_var = 1.0 / (sigma * sigma)
        z = (y[i] - F[0, i]) / sigma
        if z < -_GAUSS_Z_CLIP:
            z = -_GAUSS_Z_CLIP
        elif z > _GAUSS_Z_CLIP:
            z = _GAUSS_Z_CLIP
        grad_out[0, i] = w * (-(z / sigma))          # (mu - y) / sigma^2
        hess_out[0, i] = w * inv_var                 # Fisher == exact for mu
        grad_out[1, i] = w * (1.0 - z * z)
        hess_out[1, i] = w * 2.0                     # Fisher for rho
```

```python
@njit(cache=True, parallel=True)
def _gaussian_nll_eval(y, F, sample_weight):
    n = F.shape[1]
    total = 0.0
    weight_total = 0.0
    for i in prange(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                continue
        r = F[1, i]
        if r < _GAUSS_RHO_MIN:
            r = _GAUSS_RHO_MIN
        elif r > _GAUSS_RHO_MAX:
            r = _GAUSS_RHO_MAX
        sigma = np.exp(r)
        z = (y[i] - F[0, i]) / sigma
        if z < -_GAUSS_Z_CLIP:
            z = -_GAUSS_Z_CLIP
        elif z > _GAUSS_Z_CLIP:
            z = _GAUSS_Z_CLIP
        total += w * (_HALF_LOG_2PI + r + 0.5 * z * z)
        weight_total += w
    return total / weight_total
```

(Follow the accumulation style of `_softmax_class_major_eval_labels` losses.py:330 — if that kernel reduces with explicit thread-safe patterns rather than naive `prange` accumulation, copy its pattern exactly. If plain `prange` reduction is what the existing eval kernels do, match that.)

```python
@njit(cache=True, parallel=True)
def _gaussian_crps(y, F, sample_weight):
    # closed-form Gaussian CRPS, weighted mean; same weight/rho/z clip
    # discipline as above. Skip w <= 0 before any arithmetic.
    ...  # per §2.7, using math.erf
```

Note `sample_weight=None` dispatch: use explicit `if sample_weight is None: ... else: ...` blocks, not inline conditional expressions. Existing kernels compile separate Optional specializations, and the explicit block avoids Numba attempting to type `sample_weight[i]` when the argument is `None`.

### 4.2 Loss class

```python
class GaussianNLL:
    name = "Gaussian"
    is_classification = False
    adjusts_leaves = False
    constant_hessian = False
    n_outputs = 2

    def __init__(self, hessian_mode="natural"):
        if hessian_mode != "natural":
            raise ValueError("hessian_mode='natural' is the only supported mode")
        self.hessian_mode = hessian_mode

    def init_class_major(self, y, sample_weight=None):
        mu0 = float(np.average(y, weights=sample_weight))
        var0 = float(np.average((y - mu0) ** 2, weights=sample_weight))
        rho0 = 0.5 * np.log(max(var0, 1e-12))
        return np.array([mu0, float(np.clip(rho0, -10.0, 10.0))])

    def grad_hess_class_major_into(self, y, F, sample_weight, grad_out, hess_out):
        _gaussian_nll_grad_hess_into(y, F, sample_weight, grad_out, hess_out)

    def eval_class_major(self, y, F, sample_weight=None):
        return float(_gaussian_nll_eval(y, F, sample_weight))

    def crps_class_major(self, y, F, sample_weight=None):
        return float(_gaussian_crps(y, F, sample_weight))

    @staticmethod
    def mean_and_sigma(raw):
        # raw: (n, 2) sample-major output of predict_raw
        rho = np.clip(raw[:, 1], _GAUSS_RHO_MIN, _GAUSS_RHO_MAX)
        return raw[:, 0], np.exp(rho)

    def transform(self, raw):
        return raw  # identity on raw scores; point prediction handled by wrapper
```

**Interface note:** unlike `MultiSoftmax`, the target argument is the plain `(n,)` float target, not a one-hot `(K, n)` matrix. `DistributionalBoosting` is the only caller, so no polymorphism problem arises. Signatures deliberately mirror the class-major family so a future "vector loss protocol" can unify them (§12).

### 4.3 Registration

Do **not** add `GaussianNLL` to `LOSSES` (losses.py:415) — that dict routes into scalar `GradientBoosting` (booster.py:1680) and a 2-output loss would break it. Add a separate registry:

```python
VECTOR_LOSSES = {"Gaussian": GaussianNLL}
```

and export `GaussianNLL`, `VECTOR_LOSSES` from losses.py.

---

## 5. Component 2 — booster.py: `DistributionalBoosting`

Place after `MulticlassBoosting`. Model the code structure on `MulticlassBoosting.fit` (booster.py:1975–2418), keeping only the shared-vector lane, with `GradientBoosting.fit` (booster.py:~1660–1760) as the reference for regression target/eval-set handling. Every step below names the existing helper to call.

### 5.1 Constructor

```python
class DistributionalBoosting(_BaseBooster):
    def __init__(self, loss="Gaussian", loss_kwargs=None, **kwargs):
        super().__init__(**kwargs)          # full _BaseBooster signature, booster.py:225-239
        self.loss_name = loss
        self.loss_kwargs = dict(loss_kwargs or {})
```

### 5.2 `fit(X, y, sample_weight=None, eval_set=None, eval_sample_weight=None, cat_features=None, ...)`

Match `GradientBoosting.fit`'s exact signature (copy it — the sklearn wrapper forwards kwargs positionally/by name and both boosters must accept the same set).

**Step 0 — fit-time preamble and normalized state:** copy the same front matter as `GradientBoosting.fit` / `MulticlassBoosting.fit` before enforcing Gaussian constraints. Do not assume these fields already exist; they are refreshed on every fit after sklearn `set_params`:

```python
n_features = n_features_from_array_like(X)
cat_features = normalize_cat_features(cat_features, n_features)
X = array_like_to_numpy(X, object) if cat_features else array_like_to_numpy(X, np.float64)
n_samples = X.shape[0]
y = validate_target_vector(y, n_samples, dtype=np.float64)
_reject_eval_sample_weight_without_eval_set(eval_sample_weight, eval_set)
self.n_features_in_ = int(X.shape[1])

self.tree_mode_ = _normalize_tree_mode(self.tree_mode)
fit_random_state = normalize_random_state_seed(self.random_state)
self._fit_random_state_seed_ = fit_random_state
self.n_threads_ = _fit_thread_count(self.thread_count, self.tree_mode_, n_samples)
self.histogram_dtype_ = _normalize_histogram_dtype(self.histogram_dtype)
self.leaf_dtype_ = _normalize_leaf_dtype_name(self.leaf_dtype)
self._validate_sampling_config()
self.ordered_boosting_ = self._resolve_ordered_boosting()
w = _validate_sample_weight(sample_weight, n_samples)

try:
    self.loss_ = VECTOR_LOSSES[self.loss_name](**self.loss_kwargs)
except KeyError as exc:
    raise ValueError("loss='Gaussian' is the only distributional v1 loss") from exc

self._resolve_auto_structure_params(
    loss_name="Gaussian",
    n_samples=n_samples,
    sample_weight=w,
    X=X,
    cat_features=cat_features,
)
self.l2_leaf_reg_ = float(self.l2_leaf_reg)
```

Then raise `ValueError` with instructive messages when:

- `self.tree_mode_ != "lightgbm"` → `"loss='Gaussian' requires tree_mode='lightgbm' (shared vector trees); got '...'"`.
- `self.sampling_ in {"goss", "weighted_goss"}` or `self._mvs_active()` → GOSS and MVS sampling are not supported for Gaussian yet. Plain uniform subsampling via `subsample < 1.0` is allowed as of v1.1.
- `self._bayesian_bootstrap_active()` (booster.py:511).
- Do **not** reject `self.colsample < 1.0` in v1.1; pass the mask returned by `_feature_selection` into `build_leafwise_multiclass_tree(..., feature_mask=fmask, ...)`.
- `ordered_boosting` truthy/resolves on (`self.ordered_boosting_` semantics: resolve first the way `MulticlassBoosting` does, or simply reject `ordered_boosting=True`; `"auto"` must resolve to off).
- `histogram_dtype != "float64"` — the shared-vector lane is float64-only today; `MulticlassBoosting.fit` raises exactly this at booster.py:2244–2249. Copy that guard with a Gaussian-specific message (`"histogram_dtype='float32' is not supported for loss='Gaussian'; shared vector trees are float64-only"`). Do NOT silently ignore the setting. Normalize first, as that site does: `self.histogram_dtype_ = _normalize_histogram_dtype(self.histogram_dtype)`; also normalize `self.leaf_dtype_ = _normalize_leaf_dtype_name(self.leaf_dtype)` (booster.py:2244–2245) — `leaf_dtype_` is passed to the builder in Step 4.

`random_strength` is intentionally **allowed**. The multiclass builder accepts it and the random-split scoring path is K-generic; keep the existing deterministic split-seed initialization below.

These mirror the shared-vector gate at booster.py:2283–2295; reuse those helper predicates rather than reimplementing the conditions.

**Step 1 — target & prep:** validate `y` via `validate_target_vector` (as booster.py:1660/1975 do); build `self.prep_` and `X_binned` **exactly as `GradientBoosting.fit` does for an RMSE-loss lightgbm-mode fit** — including K-fold target statistics for categoricals and the raw-category-code companion features. Initialize timing before preprocessing, matching the scalar/multiclass boosters:

```python
timing = _new_timing(self.verbose_timing)
self.timing_ = timing
phase = _start_timing(timing)
X_binned = self._fit_transform_preprocessor(
    X, [y], cat_features, w,
    eval_set=eval_set,
    eval_sample_weight=eval_sample_weight,
)
X_route_binned = np.asfortranarray(X_binned)
X_hist_binned = X_route_binned if self.n_threads_ > 1 else X_binned
n_bins = self.prep_.n_bins_
# Validate and transform eval_set under the same phase, as the existing
# boosters do, producing Xv_binned/yv/wv when eval_set is present.
_add_timing(timing, "preprocess", phase)
```

Two audit points: booster.py:~1163 and ~1646 contain `getattr(self, "loss_name", None) == "RMSE"` checks that gate RMSE-style preprocessing behavior. Grep booster.py for `loss_name` and, at each site that means "loss is RMSE-like / compatible with raw category codes," extend the condition to include `"Gaussian"` (an explicit set literal `{"RMSE", "Gaussian"}` beats chained `==`). The μ head is an RMSE-style target, so RMSE treatment is correct.

Be surgical on those RMSE-style gates. Extend the LightGBM categorical smoothing/code sites at booster.py:647 and booster.py:1365. Do **not** broaden unrelated RMSE gates such as depthwise shallow defaults or CatBoost weighted-RMSE LR uplift; Gaussian rejects those modes.

**Step 2 — auto params:** call `self._resolve_fit_auto_params(loss_name="Gaussian", n_samples=n_samples, sample_weight=w, eval_set_present=..., p_model=X_binned.shape[1])` (booster.py:593), same as the multiclass call at booster.py:2310–2316. Requires the auto_params.py touchpoints in §6 to be done first.

**Step 3 — reusable buffers** (allocate the reusable grad/hess, row-major, and histogram buffers once before the round loop; the tree builder still allocates per-tree workspace):

```python
K = 2
self.n_outputs_ = K
self.init_ = self.loss_.init_class_major(y, w)                   # (2,)
self._record_scalar_target_stats(y, w)
F  = np.tile(self.init_[:, None], (1, n_samples))                # (2, n) float64, class-major
grad_buffer = np.empty_like(F)
hess_buffer = np.empty_like(F)
# Row-major shadow buffers for the histogram kernels — REQUIRED, see §1 note:
# without them the builder transposes grad/hess into fresh (n, 2) copies every
# round (tree.py:5896-5903). Mirrors booster.py:2384-2387.
grad_row_major = np.empty((n_samples, K), dtype=np.float64)
hess_row_major = np.empty((n_samples, K), dtype=np.float64)
hist_buffers = self._alloc_multiclass_hist_buffers(K, X_binned.shape[1], n_bins)  # booster.py:1343
self._importance = np.zeros(self.prep_.n_input_features_)
```

Eval set: Step 1 already validates and bins it via `self.prep_.transform`; after `self.init_` exists, initialize `Fv = np.tile(self.init_[:, None], (1, n_val))` (as the multiclass fit does).

Also compute the baseline loss before round 0 (mirror booster.py:1760–1764): `self.loss_.eval_class_major(yv, Fv, wv)` on the eval set if present else train — seeds the early-stopping best score the same way the other boosters do. Immediately after baseline computation call:

```python
self._finalize_early_stopping_min_delta(baseline_loss, "Gaussian")
self.auto_params_ = self._resolved_auto_params(
    n_samples=n_samples,
    n_raw_features=X.shape[1],
    X_binned=X_binned,
    n_bins=n_bins,
    sample_weight=w,
    eval_set_present=eval_set is not None,
    eval_n_samples=0 if yv is None else len(yv),
    eval_sample_weight=wv,
    rowpar_buffers=None,
    extra={"distributional": {"n_outputs": 2, "hessian_mode": self.loss_.hessian_mode}},
)
self._emit_auto_param_warnings()
self._reset_stochastic_diagnostics()
rng = np.random.default_rng(fit_random_state)
self._initialize_split_seed(rng, fit_random_state)
self.trees_ = []
self._flat_cache_ = None
self.train_history_, self.valid_history_ = [], []
eval_train = self.eval_train_loss or bool(self.verbose)
patience_score, patience_iter = np.inf, 0
best_prefix_score, best_prefix_iter = np.inf, 0
t0 = time.time()
```

**Step 4 — training loop (per round `m`):** the shared-vector loop (booster.py:~2440–2560, anchor: the `if use_shared_lightgbm_multiclass:` block containing the `build_leafwise_multiclass_tree` call at booster.py:2495) minus classification, plus the v1.1 uniform-sampling and feature-mask hooks. Match that call site argument-for-argument — in particular the row-major refills and `leaf_dtype`:

```python
phase = _start_timing(timing)
self.loss_.grad_hess_class_major_into(y, F, w, grad_buffer, hess_buffer)

fmask, findices = self._feature_selection(X_binned.shape[1], rng)
if self.subsample >= 1.0:
    row_indices_round = None
    grad_for_round = grad_buffer
    hess_for_round = hess_buffer
    self._record_sampling_diagnostic(None, n_samples)
else:
    mask = rng.random(n_samples) < self.subsample
    if not np.any(mask):
        # Avoid the sampled-empty draw becoming a false convergence signal.
        # Prefer the row with largest current gradient/Hessian payload.
        importance = (
            np.sum(np.abs(grad_buffer), axis=0)
            + np.sum(np.maximum(hess_buffer, 0.0), axis=0)
        )
        chosen = int(np.nanargmax(importance)) if np.any(importance > 0.0) else int(rng.integers(0, n_samples))
        mask[chosen] = True
    row_indices_round = np.flatnonzero(mask).astype(np.int64)
    self._record_sampling_diagnostic(row_indices_round, n_samples)
    row_mask = np.zeros(n_samples, dtype=bool)
    row_mask[row_indices_round] = True
    grad_for_round = np.where(row_mask[None, :], grad_buffer, 0.0)
    hess_for_round = np.where(row_mask[None, :], hess_buffer, 0.0)

# refill the preallocated row-major shadows (transpose assign, no allocation)
grad_row_major[:, :] = grad_for_round.T
hess_row_major[:, :] = hess_for_round.T
_add_timing(timing, "grad_hess", phase)

phase = _start_timing(timing)
tree, leaf, leaf_G, leaf_H = build_leafwise_multiclass_tree(
    X_binned, grad_for_round, hess_for_round, n_bins,
    self._max_tree_depth(), self.l2_leaf_reg_, self.lr_,
    feature_mask=fmask,
    min_child_weight=self.min_child_weight,
    hist_buffers=hist_buffers,
    return_training_state=True,
    X_hist_binned=X_hist_binned,
    X_route_binned=X_route_binned,
    max_leaves=self._max_tree_leaves(),
    min_child_samples=self.min_child_samples,
    min_gain_to_split=self.min_gain_to_split,
    random_strength=self.random_strength,
    split_seed=int(getattr(self, "_split_seed_", 0)),
    tree_iteration=m,
    grad_row_major=grad_row_major,
    hess_row_major=hess_row_major,
    leaf_dtype=self.leaf_dtype_,
)
_add_timing(timing, "tree_build", phase)
if tree.depth == 0:
    if (
        (row_indices_round is not None or fmask is not None)
        and m + 1 < self.iterations_
        and sampled_depth0_retries < _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES
    ):
        sampled_depth0_retries += 1
        continue
    break                                    # no split found after the retry policy, or no sampled path was active
sampled_depth0_retries = 0
phase = _start_timing(timing)
self.trees_.append(tree)
self._accumulate_importance(tree)
add_multiclass_leaf_values_inplace(leaf, tree.values, F)    # tree.py:2485 — leaf gather, no re-traversal
_add_timing(timing, "train_update", phase)

if eval_train:                               # honor self.eval_train_loss exactly as the multiclass loop does
    phase = _start_timing(timing)
    self.train_history_.append(self.loss_.eval_class_major(y, F, w))
    _add_timing(timing, "loss_eval", phase)

if Fv is not None:
    phase = _start_timing(timing)
    tree.add_predict_class_major(Xv_binned, Fv)             # in-place accumulate
    _add_timing(timing, "validation_predict", phase)
    phase = _start_timing(timing)
    val = self.loss_.eval_class_major(yv, Fv, wv)
    _add_timing(timing, "loss_eval", phase)
    self.valid_history_.append(val)
    # early stopping: copy the patience/min_delta/best-tracking block from the
    # multiclass shared-vector loop verbatim (directly after its val append)
```

**Step 5 — finalize:** mirror `MulticlassBoosting.fit`'s tail. Use the early-stopping variable from the copied shared-vector block:

```python
self.fit_time_ = time.time() - t0
self._truncate_to_best_model(best_prefix_iter, self.valid_history_)
self._refresh_stochastic_auto_params(n_samples)
self.best_iteration_ = len(self.trees_)
...
```

Do not use an undefined `best_iter` placeholder. `best_score_` selection follows the existing order: best validation score if `valid_history_`, otherwise final eval score if an eval set exists, otherwise last train history, otherwise baseline/train eval.

### 5.3 Prediction

```python
def predict_raw(self, X):
    # copy MulticlassBoosting.predict_raw (booster.py:2423-2439) with n_classes_ → 2
    # returns (n, 2) sample-major [mu_raw, rho_raw]

def predict_dist(self, X):
    raw = self.predict_raw(X)
    return GaussianNLL.mean_and_sigma(raw)   # (mu, sigma), each (n,)

def staged_predict_raw(self, X):
    # copy booster.py:2441-2452; every tree has add_predict_class_major so the per-class branch is dead — drop it
```

Flat ensemble: `_build_flat_ensemble` should return `build_flat_multiclass_ensemble(self.trees_, 2)`. `FlatNonObliviousEnsemble` already handles 2-D values and `add_predict_class_major` (flat_model.py:318–384), but current `flat_predict_preferred` does not prefer explicit-node non-oblivious flats, so `predict_raw` should keep the same “use flat only if preferred, otherwise per-tree loop” routing as `MulticlassBoosting.predict_raw`. Do not set or fake `n_classes_`; set `self.n_outputs_ = 2` and pass literal `2` into the flat builder.

---

## 6. Component 3 — auto_params.py touchpoints

Three surgical edits; grep for each symbol:

1. **`resolve_learning_rate_details` / `_LR_COEFS`:** `"Gaussian"` is not in the fitted coefficient table. Map it to the **RMSE** coefficient family explicitly (the μ head is an RMSE-shaped problem and dominates early LR sensitivity). The current resolver returns `"rule": AUTO_LR_RULE`; extend the details dict with an additional field such as `"loss_coefficient_source": "rmse_coefs_for_gaussian"` so `auto_params_["learning_rate"]` shows the fallback was deliberate. Today an unknown name silently falls back to RMSE coefficients (agent-verified, auto_params.py ~line 72) — make it explicit for `"Gaussian"`; leave unknown-name behavior alone.
2. **`LIGHTGBM_UNWEIGHTED_LR_MULTIPLIERS` normalization (booster.py:788–796):** `base_loss in {"MAE", "Quantile"}` → `"RMSE"` — extend to include `"Gaussian"` so the lightgbm-mode dampener resolves rather than hitting the `.get(..., 0.4)` default.
3. **`is_auto_learning_rate` / explicit-LR validation:** no change, but confirm the explicit-LR path accepts `learning_rate=0.05` for the new booster unchanged.

The auto-structure resolvers (`l2_leaf_reg="auto"`, `num_leaves="auto"`, `min_child_samples="auto"`, …) key on `tree_mode`, not loss — they work unchanged. One semantic to document (docstring + README): `min_child_weight` for this loss compares against **summed hessian mass across both heads** = `w·(1/σ² + 2)` (tree.py:3260–3266). The default 1.0 is effectively looser than for RMSE (where hess = w); this is acceptable for v1 — `min_child_samples` (default 20) is the binding constraint, as it is for multiclass.

---

## 7. Component 4 — sklearn_api.py: wrapper integration

### 7.1 Constructor

`DarkoRegressor.__init__` (sklearn_api.py:846): extend the `loss` docstring to `"RMSE" | "MAE" | "Quantile" | "Gaussian"`. Add `eval_metric=None` as the public distributional metric knob (`None`/`"nll"` selects Gaussian NLL, `"crps"` selects closed-form Gaussian CRPS for validation history, early-stopping patience, and best-prefix truncation). Add `sigma_calibration=None` as a sklearn-wrapper-only Gaussian knob; accept `None`/`False` for off and `True`/`"scalar"` for the scalar validation calibrator. `hessian_mode` is intentionally not exposed. Keep constructor sklearn-clone friendly; reject non-default `alpha` with `loss="Gaussian"` inside `fit`, not `__init__`; reject non-default `eval_metric` or non-off `sigma_calibration` for non-Gaussian losses.

### 7.2 fit() routing

**There are TWO core-model construction sites in the regressor's fit path, and both must route through one Gaussian-aware factory.** The initial fit constructs `GradientBoosting` at sklearn_api.py:1043, and the **refit path constructs a second, hardcoded `GradientBoosting` at sklearn_api.py:1116** (`refit_model = GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs, **refit_kw)`). Left as-is, `refit=True` with `loss="Gaussian"` would hit `LOSSES["Gaussian"]` inside `GradientBoosting.fit` and KeyError.

Before the `tree_mode_auto` branch, add Gaussian wrapper guards:

```python
if self.loss != "Gaussian" and self.eval_metric not in {None, "auto", "loss"}:
    raise ValueError("eval_metric is only configurable for loss='Gaussian'")
sigma_calibration_ = _normalize_sigma_calibration(self.sigma_calibration)
if self.loss != "Gaussian" and sigma_calibration_ is not None:
    raise ValueError("sigma_calibration is only supported for loss='Gaussian'")
if self.loss == "Gaussian":
    if self.alpha != 0.5:
        raise ValueError("alpha is only used with loss='Quantile'; leave alpha=0.5 for loss='Gaussian'")
    if _is_auto_tree_mode(self.tree_mode):
        raise ValueError("loss='Gaussian' requires tree_mode='lightgbm'; tree_mode='auto' is not supported in v1")
```

For non-auto values, do not compare raw strings. Either import and use the core `_normalize_tree_mode` helper or let `DistributionalBoosting.fit` perform the normalized guard. The wrapper may give an earlier error by checking `_normalize_tree_mode(model_kw.get("tree_mode", self.tree_mode)) != "lightgbm"`, which preserves public aliases such as `"leafwise"` / `"leaf_wise"`.

Keep the existing local factory's **dict-taking** contract. `_run_learning_rate_probe` and `_fit_tree_mode_auto` call `make_model(context_kwargs)` / `make_model(candidate_kwargs)`, not `make_model(**kwargs)`. Also preserve the preprocessing cache injection:

```python
def make_model(model_kw):
    if self.loss == "Gaussian":
        if _normalize_tree_mode(model_kw.get("tree_mode", self.tree_mode)) != "lightgbm":
            raise ValueError(
                "loss='Gaussian' requires tree_mode='lightgbm'; got "
                f"tree_mode={model_kw.get('tree_mode', self.tree_mode)!r}. Distributional regression uses "
                "shared vector-valued leaf-wise trees."
            )
        model = DistributionalBoosting(
            loss="Gaussian", loss_kwargs={}, **model_kw
        )
    else:
        model = GradientBoosting(
            loss=self.loss, loss_kwargs=loss_kwargs, **model_kw
        )
    if preprocessing_cache is not None:
        model._preprocessing_cache = preprocessing_cache
    return model

model = make_model(kw)                       # sklearn_api.py:1043 site
...
refit_model = make_model(refit_kw)           # sklearn_api.py:1116 site
```

This requires importing `DistributionalBoosting` (and `_normalize_tree_mode` if the wrapper performs the alias-aware early check) from `booster.py`. The classifier's twin refit site at sklearn_api.py:1546 stays untouched; `DarkoClassifier` has no `loss` constructor parameter today, so `DarkoClassifier(loss="Gaussian")` already fails with Python's unexpected-keyword `TypeError` unless a future API adds such a parameter.

Everything else downstream (eval-split creation via `_make_eval_split` sklearn_api.py:290 — regression path incl. `weighted_stratified`; early-stopping params) flows unchanged because `DistributionalBoosting` mirrors the `GradientBoosting` fit signature and attribute surface. Explicitly verify these three integrations in tests: `validation_fraction`/`eval_set`, `refit=True` + `get_refit_params()` (freezes `lr_` and selected rounds — both exist on the new class; assert the refit model is a `DistributionalBoosting`), and `sample_weight` + `eval_sample_weight`.

After any automatic validation split has been created, require a validation
set for scalar sigma calibration:

```python
if sigma_calibration_ is not None and eval_set is None:
    raise ValueError(
        "sigma_calibration='scalar' requires a validation set; pass "
        "eval_set or set early_stopping=True to create an automatic "
        "validation split"
    )
```

`sigma_calibration` must be added to `_SKLEARN_ONLY` so it is not forwarded to
`GradientBoosting` or `DistributionalBoosting`. Once the selection model has
finished and best-prefix truncation has run, compute:

```python
mu, sigma = selection_model.predict_dist(X_cal)
if eval_sample_weight is None:
    z2 = ((y_cal - mu) / np.maximum(sigma, 1e-12)) ** 2
    sigma_scale_ = sqrt(mean(z2))
else:
    positive = eval_sample_weight > 0
    z2 = (
        (y_cal[positive] - mu[positive])
        / np.maximum(sigma[positive], 1e-12)
    ) ** 2
    sigma_scale_ = sqrt(weighted_average(z2, eval_sample_weight[positive]))
```

Store `sigma_calibration_`, `sigma_scale_`, and `sigma_scale_source_ =
"selection_validation"` on the wrapper and mirror them into
`model_.auto_params_["sigma_calibration"]` and
`model_.auto_params_["diagnostics"]["sigma_calibration"]`, including raw
validation row count, positive-weight row count, effective validation row count,
and a `small_fold_warning` flag. If the effective calibration size is below
200, append `small_sigma_calibration_fold` to
`auto_params_["diagnostics"]["warnings"]` and emit it under the existing
`diagnostic_warnings` policy. For `refit=True`, do not recompute the scale on
full data; reattach the same frozen value to the refit model metadata.

### 7.3 LR probe

`_run_learning_rate_probe` (sklearn_api.py:662): the fallback chain `getattr(context_model, "loss_name", None) → context_model.loss_.name → "RMSE"` (sklearn_api.py:696–700) resolves correctly because `DistributionalBoosting.loss_name = "Gaussian"`. The probe's candidate scoring uses validation loss from `fit` → NLL by default, or CRPS when `eval_metric="crps"` is set. Add a test that `auto_learning_rate_probe=True` runs end-to-end with `loss="Gaussian"`.

### 7.4 predict / staged_predict / predict_dist / predict_interval / sample

```python
def predict(self, X):
    X = _check_predict_input(self, X)
    raw = self.model_.predict_raw(X)
    if self.loss == "Gaussian":
        return raw[:, 0]                     # mean; NLL-optimal point estimate under squared error
    return raw                                # existing scalar path unchanged (raw is (n,))

def staged_predict(self, X):
    # current implementation (sklearn_api.py:1149-1152) yields staged_predict_raw
    # directly — for Gaussian that would leak (n, 2) [mu, rho] arrays. Branch:
    X = _check_predict_input(self, X)
    if self.loss == "Gaussian":
        for raw in self.model_.staged_predict_raw(X):
            yield raw[:, 0].copy()
    else:
        yield from self.model_.staged_predict_raw(X)

def predict_dist(self, X):
    self._require_gaussian("predict_dist")
    X = _check_predict_input(self, X)
    mu, sigma = self.model_.predict_dist(X)   # raw MLE distribution
    return mu, sigma * getattr(self, "sigma_scale_", 1.0)

def predict_interval(self, X, alpha=0.1):
    self._require_gaussian("predict_interval")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    mu, sigma = self.predict_dist(X)
    from statistics import NormalDist          # no new direct dependency
    zq = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return mu - zq * sigma, mu + zq * sigma

def sample(self, X, n_samples=1, random_state=None):
    self._require_gaussian("sample")
    mu, sigma = self.predict_dist(X)
    rng = np.random.default_rng(random_state)
    return rng.normal(mu[:, None], sigma[:, None], size=(mu.shape[0], int(n_samples)))
```

`_require_gaussian(name)` raises `AttributeError`-style `ValueError`: `f"{name}() requires loss='Gaussian'; this model was fit with loss='{self.loss}'"`. Follow sklearn convention: these methods check `check_is_fitted` via the same `_check_predict_input` path as `predict`.

The scalar calibration scale is deliberately applied only through the public
distributional methods (`predict_dist`, `predict_interval`, `sample`) and not
inside `model_.predict_raw` / `model_.predict_dist`. Raw scores stay the fitted
MLE scores for diagnostics, staged raw prediction, and future calibration
experiments. `predict()` returns μ and is unchanged by sigma calibration.

### 7.5 predict() dispatch caveat

The scalar losses return `(n,)` from `predict_raw` while Gaussian returns `(n, 2)` — the branch above keys on `self.loss`, which is correct for a fitted wrapper but **also make the loaded-model path set `self.loss = "Gaussian"`** (§8) so a wrapper reconstructed from npz dispatches correctly. In `DarkoRegressor.load_model`, after `booster, wrapper_header, _ = load_booster(...)`, if `isinstance(booster, DistributionalBoosting)`, force `est.loss = booster.loss_name` after restoring wrapper params. If wrapper params exist and say a different loss, raise a model-archive error rather than letting wrapper dispatch diverge from the loaded booster.

Wrapper state must also persist `sigma_scale`, `sigma_calibration`, and
`sigma_scale_source` when calibration is active. A loaded calibrated Gaussian
wrapper should match pre-save `predict_dist` exactly; an uncalibrated or legacy
archive should default to scale `1.0` without writing a no-op `sigma_scale=1.0`
wrapper-state field.

### 7.6 Diagnostics

Record in `auto_params_` (already handled by booster §5.2); no wrapper change. `fit` diagnostics (LR clipping, low-ESS warnings) run in `_BaseBooster` and apply unchanged.

### 7.7 Exports

`darkofit/__init__.py`: no new top-level exports needed (`DistributionalBoosting` stays a core-level class like `GradientBoosting`; users go through the wrapper). Optionally export `GaussianNLL` for power users — skip in v1.

### 7.8 Tuner

`DarkoStepwiseSearchCV` (alias `DarkoSearchCV`) supports Gaussian regressors as of the v1.1 pass:

- Resolve Gaussian searches to the LightGBM/leaf-wise lane only. The public default `tree_modes=("catboost", "lightgbm")` should filter to `("lightgbm",)` instead of failing; explicit `tree_modes` with no LightGBM alias should raise an instructive `ValueError`.
- Default scoring for `loss="Gaussian"` is `neg_gaussian_nll`, computed from `predict_dist` with the same clipped Gaussian NLL surface as the loss kernels.
- The `sampling_regularization` phase must suggest only Gaussian-supported knobs: `sampling="uniform"`, `bootstrap_type="none"`, `subsample`, `colsample`, and `l2_leaf_reg`. Do not suggest GOSS/MVS/Bayesian options for Gaussian trials.
- Keep custom scorers working unchanged; the Gaussian default scorer is just the safe default when `scoring=None`.

---

## 8. Component 5 — serialization.py

### 8.1 Save

`save_booster` branches on model class (agent-verified layout: GradientBoosting header at serialization.py:646–648, MulticlassBoosting at 636–644). First add the required imports:

```python
from .losses import LOSSES, MultiSoftmax, VECTOR_LOSSES
# inside save_booster/load_booster local imports:
from .booster import GradientBoosting, MulticlassBoosting, DistributionalBoosting
```

Then add a third branch before the unsupported-class `else`:

```python
elif isinstance(booster, DistributionalBoosting):
    header["model_class"] = "DistributionalBoosting"
    header["init"] = [float(v) for v in booster.init_]        # 2 floats
    header["n_outputs"] = 2
    header["loss_name"] = booster.loss_name                    # "Gaussian"
    header["loss_kwargs"] = _jsonify(booster.loss_kwargs)
```

Tree packing needs **zero changes**: `trees_` is a list of `MultiNonObliviousTree`, `_tree_kind` returns `"multi"` (serialization.py:687), and the existing `_pack_nonoblivious(flat_trees, arrays, vector_values=True)` path (serialization.py:698, values stacked to `(total_leaves, 2)` at :338) applies. Verify `_tree_kind` doesn't peek at the booster class — if it inspects only tree objects (it does, per `type(first).__name__`), nothing to do.

### 8.2 Load

`load_booster` (serialization.py:735–928): add the `"DistributionalBoosting"` branch modeled on the MulticlassBoosting one (:774–785):

```python
booster = DistributionalBoosting(loss=header["loss_name"],
                                 loss_kwargs=header["loss_kwargs"], **params)
booster.init_ = np.array(header["init"], dtype=np.float64)
booster.n_outputs_ = int(header["n_outputs"])
booster.loss_ = VECTOR_LOSSES[header["loss_name"]](**header["loss_kwargs"])
```

**Tree unpacking — one required edit, not zero.** The `kind == "multi"` branch hard-keys the leaf-value width to the multiclass header field (serialization.py:1141–1144):

```python
elif kind == "multi":
    trees = _unpack_nonoblivious(
        data, MultiNonObliviousTree, n_bins,
        expected_value_width=header["n_classes"],
    )
```

A distributional model writes `n_outputs`, not `n_classes`, so this KeyErrors on load. Generalize the width lookup:

```python
elif kind == "multi":
    if "n_outputs" in header:
        value_width = int(header["n_outputs"])        # DistributionalBoosting
    else:
        value_width = int(header["n_classes"])        # MulticlassBoosting (unchanged)
    trees = _unpack_nonoblivious(
        data, MultiNonObliviousTree, n_bins,
        expected_value_width=value_width,
    )
```

Do **not** write a fake `n_classes` into the distributional header to dodge this — `n_classes` has classification semantics elsewhere in the loader (e.g. the per-class tree-count check at serialization.py:1154). Also audit `_validate_boosting_round_count` for any `n_classes` reads on the `"multi"` path.

The `*_per_class` reconstruction is multiclass-only and irrelevant here — the `"multi"` kind path above is the one distributional trees take. Do **not** bump format version for this feature. Current source already has `FORMAT_VERSION = 3` and `BASE_FORMAT_VERSION = 2`, and archive compatibility is keyed by `model_class` plus tolerated header fields; distributional support adds a new model class, not a new archive layout primitive.

Wrapper-level `save_model`/`load_model` delegate to these; ensure the reconstructed `DarkoRegressor` gets `loss="Gaussian"` set from the header so §7.5 dispatch works, and `predict_dist` on a loaded model equals the pre-save model to float64 exactness.

---

## 9. Performance specification

Design targets and why they hold:

1. **Per-round cost ≈ 1.7–2.2× scalar RMSE lightgbm-mode, not more.** The added work is exactly K=2 in the multiclass kernels: histogram fill writes 2 grad + 2 hess lanes vs 1+1 (plus the shared count lane both paths have); split scan sums 2 output gains per bin (tree.py:3270–3289). These kernels are the same compiled functions the multiclass path already runs — no new hot code. The gradient kernel adds one `exp` per row per round (§4.1), amortized noise against histogram bandwidth (tree.py's own comments identify memory bandwidth as the bound).
2. **No avoidable per-round grad/hist allocation.** `grad_buffer`/`hess_buffer` reused via `grad_hess_class_major_into`; the `(n, 2)` row-major shadows are allocated once and refilled by transpose-assign then passed as `grad_row_major=`/`hess_row_major=` (omitting them makes the builder allocate fresh transposes every round, tree.py:5896–5903); histogram buffers allocated once via `_alloc_multiclass_hist_buffers` (booster.py:1343) and reused via the `hist_buffers=` argument — this is already the multiclass pattern (booster.py:2384–2387, 2495–2513); copy it. The builder still allocates per-tree workspace arrays such as node arrays, row order/scratch, and leaf metadata, so do not claim literal zero allocation in the whole round loop. Buffer memory: `2 · 2 · n_features · max_leaves · max_bins · 8B` (hg+hh) + count lane; e.g. 100 features × 64 leaves × 254 bins ≈ 52 MB total — same order as a 2-class multiclass fit.
3. **Train-F update is the leaf-gather fast path.** `add_multiclass_leaf_values_inplace` (O(n·K) gather, tree.py:2485) — this path *already avoids* the known `tree.add_predict` re-traversal inefficiency flagged in the scalar MAE/Quantile path, so the distributional loop starts on the fast lane.
4. **Prediction:** `build_flat_multiclass_ensemble(self.trees_, 2)` can construct a `FlatNonObliviousEnsemble(vector_values=True)`, but current `flat_predict_preferred` deliberately does not prefer flattened explicit-node non-oblivious ensembles. Unless that routing policy is changed and benchmarked, `predict_raw` should honestly follow the existing multiclass dispatch: try `_flat_ensemble()`, use it only if `flat_predict_preferred(flat)` is true, otherwise loop over vector trees with `tree.add_predict_class_major`. `predict_dist` adds one vectorized `np.exp` over `(n,)`.
5. **JIT warm-up:** all tree kernels are shared with multiclass (identical signatures/dtypes ⇒ cached compilations reused). New compilation surface = 3 small loss kernels. Keep them `cache=True`.
6. **Class-major locality:** K=2 rows of F are `2 × n` contiguous; the class-major scatter concern from the perf review (large-K multiclass) is minimal at K=2.
7. **Do not** introduce float32 here; that's an orthogonal existing roadmap item. Match float64 conventions throughout.

Benchmark gate (must hold before merge, via `benchmarks/bench_distributional.py`, §11): on a 500k-row synthetic, `loss="Gaussian"` fit time ≤ 2.5× `loss="RMSE"` fit time at equal rounds/leaves, and ≥ 5× faster than NGBoost at equal round count (NGBoost fits sklearn trees per round; this should be a blowout — treat it as a sanity floor, not a target).

---

## 10. Test plan (tests/test_distributional.py, new file)

Unit — loss math:
1. **Finite-difference gradient check:** random interior `(y, F)` with `|z| < 5`, ε=1e-6 central differences on clipped NLL vs `_gaussian_nll_grad_hess_into` outputs for both heads, weighted and unweighted; rtol 1e-5. Also check `h_mu` equals FD second derivative w.r.t. μ in the unclipped interior.
2. **Init:** `init_class_major` equals numpy weighted mean / 0.5·log(weighted MLE var); constant-`y` target yields finite clipped ρ0 and a fit that does not produce NaN.
3. **Clipping and zero weights:** `F[1] = ±1e3` and extreme finite `y` produce finite grad/hess/eval; `sample_weight=0` rows write exactly zero grad/hess and are skipped by eval/CRPS before any arithmetic.
4. **CRPS:** closed form vs Monte-Carlo CRPS estimate on a small grid (1e5 draws, atol 5e-3); CRPS ≥ 0.

Training behavior:
5. **NLL decreases:** train NLL (with `eval_train_loss=True`) strictly non-increasing over the first 50 rounds on a synthetic set (allow tiny tolerance).
6. **Heteroscedastic recovery:** `n=20_000`, `y = sin(3x₀) + (0.3 + |x₁|)·ε`, `ε~N(0,1)`: after fit, `pearsonr(sigma_hat, 0.3+|x₁|) > 0.8` and central 90% interval coverage in `[0.86, 0.94]` on a held-out set. Seeded.
7. **Homoscedastic sanity:** constant true σ=2.0 → `median(sigma_hat) ∈ [1.8, 2.2]`.
8. **Weight invariants:** multiplying all nonzero sample weights by a constant gives identical predictions after `_validate_sample_weight` mean-normalization; zero-weight extreme rows do not produce NaN/inf and do not add nonzero gradient/hessian payloads. Do not assert row-duplication equivalence: duplication changes row count, min-child-sample legality, binning samples, and categorical target statistics.
9. **Point-prediction quality:** test-RMSE of `predict()` within 5% of a `loss="RMSE"` fit with identical params on dataset (6) — the μ head must not pay a meaningful accuracy tax.

API and wiring:
10. `predict(X) == predict_dist(X)[0]` exactly; `staged_predict` yields `(n,)` μ arrays (not `(n, 2)` raw) and its final element equals `predict`; interval symmetric about μ; `sample` mean/std → (μ, σ) as draws grow.
11. **Early stopping and calibration** on eval_set fires; `best_iteration_` set; `use_best_model` truncation shortens `trees_`; `sigma_calibration="scalar"` computes the validation `sqrt(weighted_mean(z²))` scale from the selected/truncated model, applies it to wrapper `predict_dist` but not core `model_.predict_dist`, rejects non-Gaussian losses and missing validation sets, and records metadata in `auto_params_`; `refit=True` runs end-to-end, freezes the selection-phase scale, and `isinstance(reg.model_, DistributionalBoosting)` holds after refit (regression test for the hardcoded-`GradientBoosting` refit site, sklearn_api.py:1116); `get_refit_params()` round-trip runs.
12. **Auto LR:** `learning_rate=None` resolves; `auto_params_["learning_rate"]` records the Gaussian→RMSE rule source; probe (`auto_learning_rate_probe=True`) runs.
13. **Guards raise / sampling fits / tuner runs:** Gaussian + `tree_mode="catboost"` (and `"auto"`, `"hybrid"`, `"depthwise"`); Gaussian + LightGBM aliases such as `"leafwise"` / `"leaf_wise"` fits successfully; + `sampling="goss"` and `sampling="mvs"` raise; + `subsample=0.5` fits and records stochastic diagnostics; + `colsample=0.8` fits; + sampled `tree.depth == 0` retries are capped; + `bootstrap_type="bayesian", bagging_temperature=0.5`; + `ordered_boosting=True`; + `histogram_dtype="float32"` (must raise, not silently fall back to float64); `predict_dist` on `loss="RMSE"` model; non-default `alpha` + Gaussian; `eval_metric="crps"` controls validation history; non-Gaussian `eval_metric="crps"` raises; tuner + Gaussian runs on the resolved LightGBM lane and uses `neg_gaussian_nll` by default. No classifier-specific Gaussian guard is needed unless the classifier API later adds a `loss` parameter.
14. **Serialization round-trip:** save/load → `predict_dist` allclose (rtol 0, atol 0 expected — same arrays), header contains `n_outputs=2` and no `n_classes`, loader derives leaf width from `n_outputs` (§8.2), loaded wrapper `predict` dispatches to μ, and scalar sigma calibration state survives wrapper save/load; a saved+loaded **multiclass** model still round-trips (guards the shared `kind == "multi"` width edit).
15. **Categoricals:** fit with `cat_features` on a mixed-frame; runs and predicts (exercises the RMSE-style prep gate extension of §5.2 step 1).
16. **Existing suite stays green:** no behavior change for any other loss (the only shared-code edits are the two `loss_name` set-literal extensions and auto_params mapping — assert `loss="RMSE"` fits byte-identical predictions before/after on a fixed seed, which the existing tests already effectively cover; run the full suite).

---

## 11. Benchmark script — benchmarks/bench_distributional.py

Follow the structure/CLI conventions of `benchmarks/bench_vs_lightgbm.py`. Datasets: the synthetic heteroscedastic generator from test (6) at 100k/500k rows, plus 2–3 OpenML regression sets already used by the existing bench harness. Contenders (each behind a soft import; skip with a printed notice if missing):

- DarkoFit `loss="Gaussian"` (this work)
- DarkoFit `loss="Gaussian"` with validation NLL early stopping
  (`darkofit_gaussian_es`)
- DarkoFit `loss="Gaussian"` with validation NLL early stopping plus
  `sigma_calibration="scalar"` (`darkofit_gaussian_es_calibrated`)
- NGBoost (`Normal` dist), equal rounds
- CatBoost `loss_function="RMSEWithUncertainty"`
- LightGBM twin-model baseline: model A = mean (L2); model B = L2 on `log((y−μ̂_oof)² + eps)` via out-of-fold μ̂ — the "practical hack" this feature replaces
- DarkoFit quantile pair (α=0.05, 0.95) — interval-only baseline

Metrics per contender: validation NLL, CRPS, 90% empirical coverage, mean interval width, σ-binned 90% coverage for every lane that exposes per-row σ, fit wall-time (kernels warmed; respect the existing benchmark-fairness note about timing encoding inside vs outside the timer — encode outside for all contenders). Emit a markdown table; add the result summary to BENCHMARK_NOTES.md. The early-stopped lane should accept a larger max-round budget than the fixed-round lane so the benchmark can separate under-training from late σ overfit.

---

## 12. Follow-ups explicitly out of scope (record in README "not implemented" list)

- **Per-parameter scalar trees** (NGBoost-style: one scalar tree per head per round) — would unlock `tree_mode="catboost"`/oblivious and ordered boosting for distributional fits via the per_class machinery (booster.py:2295–2405).
- **Remaining sampling relaxations:** GOSS/MVS and Bayesian bootstrap for Gaussian. Uniform subsample and colsample landed in v1.1 after adding the capped sampled-depth-zero retry that prevents an unlucky/no-split sample from being mistaken for global convergence.
- **More heads:** `Poisson`, `NegativeBinomial` (count sports stats), `StudentT` (heavy tails) — each is one `VECTOR_LOSSES` class; `StudentT`/NB are 2–3 output. Multi-quantile shared tree (all α in one model, monotone rearrangement at predict).
- **Public custom vector-loss protocol:** document the `n_outputs` / `init_class_major` / `grad_hess_class_major_into` / `eval_class_major` duck-type and accept user instances in `DistributionalBoosting(loss=<instance>)`. Deferred only because serialization of arbitrary user losses needs a story (`loss_name` round-trip).
- **More distributional API surface:** public custom vector losses and non-Gaussian `predict_dist` conventions.

Head-zoo deferral rationale: the tree and serialization layers can already carry
vector-valued leaves for any fixed `n_outputs`, but the public wrapper contract
is still Gaussian-shaped: `predict()` returns μ, `predict_dist()` returns
`(mu, sigma)`, `predict_interval()` assumes Normal quantiles, and `sample()`
draws from a Normal. Landing `Poisson`/`NegativeBinomial`/`StudentT` safely
therefore needs a small public distribution protocol first:

- every built-in vector loss declares `distribution_name`, `n_outputs`,
  `mean(raw)`, `params(raw)`, optional `interval(raw, alpha)`, optional
  `sample(raw, rng, n_samples)`, and its default validation metric;
- `DarkoRegressor.predict_dist` returns the natural parameter tuple for
  the fitted distribution, not always `(mu, sigma)`;
- save/load persists the built-in `loss_name` and reconstructed loss instance
  exactly as Gaussian does today; arbitrary user losses remain unsupported until
  custom-loss serialization is specified;
- tests cover target-domain validation (`Poisson`/NB nonnegative counts),
  finite-difference gradients, save/load, wrapper dispatch, and benchmark lanes
  before any head is advertised.

## 13. Suggested implementation order (each step lands green)

1. **losses.py:** kernels + `GaussianNLL` + `VECTOR_LOSSES` + tests (1)–(4). No integration yet.
2. **booster.py:** `DistributionalBoosting` fit/predict/staged + guards + auto_params touchpoints (§6) + tests (5)–(9), (13-core-level).
3. **sklearn_api.py / tuning:** routing, `predict_dist`/`predict_interval`/`sample`, CRPS validation metric, probe check, Gaussian tuner lane support + tests (10)–(13), (15).
4. **serialization.py:** save/load branch + test (14).
5. **Benchmark + docs:** §11 script, README section (usage block from §1, constraints, `min_child_weight` semantics note from §6), CHANGELOG entry, run full suite (16).

Estimated new code: ~120 lines losses.py, ~250 lines booster.py, ~90 lines sklearn_api.py, ~40 lines serialization.py, ~200 lines tests, ~150 lines benchmark.
