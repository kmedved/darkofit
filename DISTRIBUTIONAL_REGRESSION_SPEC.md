# Implementation Spec: Native Distributional Regression (`loss="Gaussian"`)

**Status:** proposed — not implemented
**Target:** ChimeraBoost `main` (line references as of commit `3029388` + current working tree; treat them as anchors, re-locate by symbol name if drifted)
**Audience:** implementing agent (Codex). This document is self-contained: every integration point, buffer layout, and formula needed is specified here. When this spec and the source disagree on a line number, trust the symbol name and the described behavior.

---

## 1. Summary

Add a heteroscedastic Gaussian regression head to ChimeraBoost: one model that jointly predicts a per-row mean **μ(x)** and standard deviation **σ(x)** by minimizing Gaussian negative log-likelihood (NLL), trained with **shared vector-valued trees** (two outputs per leaf) using the natural gradient (Fisher preconditioning). This is the same model family as NGBoost / CatBoost `RMSEWithUncertainty` / LightGBMLSS, but implemented natively on ChimeraBoost's existing multiclass shared-vector machinery.

User-facing result:

```python
from chimeraboost import ChimeraBoostRegressor

reg = ChimeraBoostRegressor(loss="Gaussian", tree_mode="lightgbm", early_stopping=True)
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

### Non-goals (v1)

- No CatBoost/oblivious/hybrid/depthwise tree modes for this loss (v1 = `tree_mode="lightgbm"` only, same restriction as `multiclass_tree_strategy="shared_vector"`).
- No GOSS/MVS/uniform-subsample row sampling, no Bayesian bootstrap, no `colsample < 1.0`, no ordered boosting (mirrors the existing shared-vector gate at booster.py:2018–2024). Raise clear errors; relaxations are listed in §12.
- No tuner (`ChimeraBoostSearchCV`) integration; the tuner is untouched and should reject `loss="Gaussian"` estimators cleanly (§7.8).
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
- ρ head: leaf value ≈ mean of `(z² − 1)/2` — bounded, smooth steps.

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
```

Every kernel that exponentiates ρ first computes `r = min(max(F[1, i], _GAUSS_RHO_MIN), _GAUSS_RHO_MAX)`. This bounds `1/sigma²` at ~1.07e13 — large but finite in float64, and l2_leaf_reg plus the Fisher self-scaling keep leaf values sane. Do **not** clip gradients; clip only the ρ used for `exp`.

`predict_dist` applies the same clip when converting raw ρ to σ so train-time and predict-time σ agree.

### 2.7 CRPS (evaluation metric, closed form)

For reporting and benchmarks (not used for early stopping in v1 — early stopping uses validation NLL):

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

Instead add `DistributionalBoosting(_BaseBooster)` in booster.py. Its `fit` is a *simpler* sibling of the shared-vector section of `MulticlassBoosting.fit` (booster.py:2157–2293): no one-hot, no per_class route, no sampling branches (all rejected up front), no ordered boosting. Expected size ≈ 200 lines, nearly all of it calls into existing helpers. It must expose the same attribute surface as `GradientBoosting` so the sklearn wrapper, refit machinery, and serialization treat it uniformly:

`loss_name`, `loss_kwargs`, `loss_`, `init_` (shape `(2,)`), `trees_` (list of `MultiNonObliviousTree`), `lr_`, `iterations_`, `best_iteration_`, `best_score_`, `train_history_`, `valid_history_`, `auto_params_`, `prep_`, `use_best_model_`, `early_stopping_rounds_`, `feature_importances_` (if `GradientBoosting` exposes it — mirror whatever `_rebuild_importance_from_trees` provides; vector-tree gains already sum over outputs, tree.py:3285), plus `n_outputs_ = 2`.

---

## 4. Component 1 — losses.py: `GaussianNLL`

### 4.1 Kernels (new, at module level near the softmax kernels)

```python
_GAUSS_RHO_MIN = -15.0
_GAUSS_RHO_MAX = 15.0
_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


@njit(cache=True, parallel=True)
def _gaussian_nll_grad_hess_into(y, F, sample_weight, grad_out, hess_out):
    # y: (n,) float64; F: (2, n) class-major raw scores (mu, rho)
    # grad_out/hess_out: (2, n) float64, written in place
    n = F.shape[1]
    for i in prange(n):
        w = 1.0 if sample_weight is None else sample_weight[i]
        r = F[1, i]
        if r < _GAUSS_RHO_MIN:
            r = _GAUSS_RHO_MIN
        elif r > _GAUSS_RHO_MAX:
            r = _GAUSS_RHO_MAX
        sigma = np.exp(r)
        inv_var = 1.0 / (sigma * sigma)
        z = (y[i] - F[0, i]) / sigma
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
        w = 1.0 if sample_weight is None else sample_weight[i]
        r = F[1, i]
        if r < _GAUSS_RHO_MIN:
            r = _GAUSS_RHO_MIN
        elif r > _GAUSS_RHO_MAX:
            r = _GAUSS_RHO_MAX
        sigma = np.exp(r)
        z = (y[i] - F[0, i]) / sigma
        total += w * (_HALF_LOG_2PI + r + 0.5 * z * z)
        weight_total += w
    return total / weight_total
```

(Follow the accumulation style of `_softmax_class_major_eval_labels` losses.py:330 — if that kernel reduces with explicit thread-safe patterns rather than naive `prange` accumulation, copy its pattern exactly. If plain `prange` reduction is what the existing eval kernels do, match that.)

```python
@njit(cache=True, parallel=True)
def _gaussian_crps(y, F, sample_weight):
    # closed-form Gaussian CRPS, weighted mean; same clip discipline as above
    ...  # per §2.7, using math.erf
```

Note `sample_weight=None` dispatch: the existing kernels are called with either `None` or an array — check how `_softmax_class_major_grad_hess_into` handles the None case (it is compiled for both via `if sample_weight is None` branches or an Optional type); replicate exactly.

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

**Step 0 — constraint validation (before any work):** raise `ValueError` with instructive messages when:

- `self.tree_mode` resolves to anything but `"lightgbm"` → `"loss='Gaussian' requires tree_mode='lightgbm' (shared vector trees); got '...'"`.
- `self._row_sampling_active()` (booster.py:517) → name the offending setting (`sampling`, `subsample`).
- `self._bayesian_bootstrap_active()` (booster.py:511).
- `self.colsample < 1.0`.
- `ordered_boosting` truthy/resolves on (`self.ordered_boosting_` semantics: resolve first the way `MulticlassBoosting` does, or simply reject `ordered_boosting=True`; `"auto"` must resolve to off).
- `random_strength != 0.0` — the multiclass builder accepts it, so *allow* it; do not reject. (Verify `build_leafwise_multiclass_tree`'s `random_strength` path compiles for K=2 — it should, it's K-generic.)
- `histogram_dtype != "float64"` — the shared-vector lane is float64-only today; `MulticlassBoosting.fit` raises exactly this at booster.py:2244–2249. Copy that guard with a Gaussian-specific message (`"histogram_dtype='float32' is not supported for loss='Gaussian'; shared vector trees are float64-only"`). Do NOT silently ignore the setting. Normalize first, as that site does: `self.histogram_dtype_ = _normalize_histogram_dtype(self.histogram_dtype)`; also normalize `self.leaf_dtype_ = _normalize_leaf_dtype_name(self.leaf_dtype)` (booster.py:2244–2245) — `leaf_dtype_` is passed to the builder in Step 4.

These mirror the shared-vector gate at booster.py:2283–2295; reuse those helper predicates rather than reimplementing the conditions.

**Step 1 — target & prep:** validate `y` via `validate_target_vector` (as booster.py:1660/1975 do); build `self.prep_` and `X_binned` **exactly as `GradientBoosting.fit` does for an RMSE-loss lightgbm-mode fit** — including K-fold target statistics for categoricals and the raw-category-code companion features. Two audit points: booster.py:~1163 and ~1646 contain `getattr(self, "loss_name", None) == "RMSE"` checks that gate RMSE-style preprocessing behavior. Grep booster.py for `loss_name` and, at each site that means "loss is RMSE-like / compatible with raw category codes," extend the condition to include `"Gaussian"` (an explicit set literal `{"RMSE", "Gaussian"}` beats chained `==`). The μ head is an RMSE-style target, so RMSE treatment is correct.

**Step 2 — auto params:** call `self._resolve_fit_auto_params(loss_name="Gaussian", n_samples=n, sample_weight=w, eval_set_present=..., p_model=X_binned.shape[1])` (booster.py:593), same as the multiclass call at booster.py:2310–2316. Requires the auto_params.py touchpoints in §6 to be done first.

**Step 3 — buffers** (all allocated once, before the round loop — the loop itself must be allocation-free):

```python
self.loss_ = VECTOR_LOSSES[self.loss_name](**self.loss_kwargs)   # KeyError → clear ValueError
K = 2
self.init_ = self.loss_.init_class_major(y, w)                   # (2,)
F  = np.tile(self.init_[:, None], (1, n_samples))                # (2, n) float64, class-major
grad_buffer = np.empty_like(F)
hess_buffer = np.empty_like(F)
# Row-major shadow buffers for the histogram kernels — REQUIRED, see §1 note:
# without them the builder transposes grad/hess into fresh (n, 2) copies every
# round (tree.py:5896-5903). Mirrors booster.py:2384-2387.
grad_row_major = np.empty((n_samples, K), dtype=np.float64)
hess_row_major = np.empty((n_samples, K), dtype=np.float64)
hist_buffers = self._alloc_multiclass_hist_buffers(K, X_binned.shape[1], n_bins)  # booster.py:1343
# Binned-matrix companions, exactly as booster.py:2305-2308:
X_route_binned = np.asfortranarray(X_binned)
X_hist_binned = X_route_binned if self.n_threads_ > 1 else X_binned
```

Eval set: bin via `self.prep_.transform`, init `Fv = np.tile(self.init_[:, None], (1, n_val))` (as the multiclass fit does).

Also compute the baseline loss before round 0 (mirror booster.py:1760–1764): `self.loss_.eval_class_major(yv, Fv, wv)` on the eval set if present else train — seeds the early-stopping best score the same way the other boosters do.

**Step 4 — training loop (per round `m`):** the shared-vector loop (booster.py:~2440–2560, anchor: the `if use_shared_lightgbm_multiclass:` block containing the `build_leafwise_multiclass_tree` call at booster.py:2495) minus classification and sampling. Match that call site argument-for-argument — in particular the row-major refills and `leaf_dtype`:

```python
self.loss_.grad_hess_class_major_into(y, F, w, grad_buffer, hess_buffer)

# refill the preallocated row-major shadows (transpose assign, no allocation)
grad_row_major[:, :] = grad_buffer.T
hess_row_major[:, :] = hess_buffer.T

tree, leaf, leaf_G, leaf_H = build_leafwise_multiclass_tree(
    X_binned, grad_buffer, hess_buffer, n_bins,
    self._max_tree_depth(), self.l2_leaf_reg_, self.lr_,
    feature_mask=None,                       # colsample==1.0 enforced
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
if tree.depth == 0:
    break                                    # no split found; safe here — no subsampling, so this is a true convergence signal
self.trees_.append(tree)
add_multiclass_leaf_values_inplace(leaf, tree.values, F)    # tree.py:2485 — leaf gather, no re-traversal

if eval_train:                               # honor self.eval_train_loss exactly as the multiclass loop does
    self.train_history_.append(self.loss_.eval_class_major(y, F, w))

if Fv is not None:
    tree.add_predict_class_major(Xv_binned, Fv)             # in-place accumulate
    val = self.loss_.eval_class_major(yv, Fv, wv)
    self.valid_history_.append(val)
    # early stopping: copy the patience/min_delta/best-tracking block from the
    # multiclass shared-vector loop verbatim (directly after its val append)
```

**Step 5 — finalize:** `self._truncate_to_best_model(best_iter, self.valid_history_)` (booster.py:1003); record `auto_params_` entries the way both existing fits do, plus `{"loss": "Gaussian", "n_outputs": 2, "hessian_mode": self.loss_.hessian_mode}`; set `best_iteration_`, `best_score_`, timing, diagnostics — mirror `MulticlassBoosting.fit`'s tail (booster.py:2405–2418).

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

Flat ensemble: reuse `self._flat_ensemble()` with `build_flat_multiclass_ensemble(self.trees_, 2)` exactly as booster.py:2421 — `FlatNonObliviousEnsemble` already handles 2-D values and `add_predict_class_major` (flat_model.py:318–384). If `_flat_ensemble` reads `self.n_classes_`, set `self.n_classes_ = None` and `self.n_outputs_ = 2`, and make the distributional class override whatever small hook fetches the output count (prefer a `_n_flat_outputs()` helper over aliasing `n_classes_` — do not fake a classification attribute).

---

## 6. Component 3 — auto_params.py touchpoints

Three surgical edits; grep for each symbol:

1. **`resolve_learning_rate_details` / `_LR_COEFS`:** `"Gaussian"` is not in the fitted coefficient table. Map it to the **RMSE** coefficient family explicitly (the μ head is an RMSE-shaped problem and dominates early LR sensitivity), and record `rule_source` so `auto_params_["learning_rate"]` shows the fallback was deliberate (e.g. `"catboost_form:rmse_coefs_for_gaussian"`). Today an unknown name silently falls back to RMSE coefficients (agent-verified, auto_params.py ~line 72) — make it explicit for `"Gaussian"`; leave unknown-name behavior alone.
2. **`LIGHTGBM_UNWEIGHTED_LR_MULTIPLIERS` normalization (booster.py:788–796):** `base_loss in {"MAE", "Quantile"}` → `"RMSE"` — extend to include `"Gaussian"` so the lightgbm-mode dampener resolves rather than hitting the `.get(..., 0.4)` default.
3. **`is_auto_learning_rate` / explicit-LR validation:** no change, but confirm the explicit-LR path accepts `learning_rate=0.05` for the new booster unchanged.

The auto-structure resolvers (`l2_leaf_reg="auto"`, `num_leaves="auto"`, `min_child_samples="auto"`, …) key on `tree_mode`, not loss — they work unchanged. One semantic to document (docstring + README): `min_child_weight` for this loss compares against **summed hessian mass across both heads** = `w·(1/σ² + 2)` (tree.py:3260–3266). The default 1.0 is effectively looser than for RMSE (where hess = w); this is acceptable for v1 — `min_child_samples` (default 20) is the binding constraint, as it is for multiclass.

---

## 7. Component 4 — sklearn_api.py: wrapper integration

### 7.1 Constructor

`ChimeraBoostRegressor.__init__` (sklearn_api.py:846): extend the `loss` docstring to `"RMSE" | "MAE" | "Quantile" | "Gaussian"`. No new constructor params (`hessian_mode` is intentionally not exposed; `alpha` is ignored for Gaussian — raise if user sets a non-default `alpha` with `loss="Gaussian"` to avoid silent confusion).

### 7.2 fit() routing

**There are TWO core-model construction sites in the regressor's fit path, and both must route through one Gaussian-aware factory.** The initial fit constructs `GradientBoosting` at sklearn_api.py:1043, and the **refit path constructs a second, hardcoded `GradientBoosting` at sklearn_api.py:1116** (`refit_model = GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs, **refit_kw)`). Left as-is, `refit=True` with `loss="Gaussian"` would hit `LOSSES["Gaussian"]` inside `GradientBoosting.fit` and KeyError. Define one local factory in `fit` and use it at both sites (and anywhere else a core model is built from `self.loss`, e.g. the LR-probe `make_model` if it constructs directly):

```python
def _make_core_model(**kw):
    if self.loss == "Gaussian":
        if self.tree_mode != "lightgbm":
            raise ValueError(
                "loss='Gaussian' requires tree_mode='lightgbm'; got "
                f"tree_mode={self.tree_mode!r}. Distributional regression uses "
                "shared vector-valued leaf-wise trees."
            )
        return DistributionalBoosting(loss="Gaussian", **kw)
    return GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs, **kw)

model = _make_core_model(**model_kw)          # sklearn_api.py:1043 site
...
refit_model = _make_core_model(**refit_kw)    # sklearn_api.py:1116 site
```

(The classifier's twin refit site at sklearn_api.py:1546 stays untouched — Gaussian is rejected there per §7.2's classifier guard.)

Everything else downstream (eval-split creation via `_make_eval_split` sklearn_api.py:290 — regression path incl. `weighted_stratified`; early-stopping params) flows unchanged because `DistributionalBoosting` mirrors the `GradientBoosting` fit signature and attribute surface. Explicitly verify these three integrations in tests: `validation_fraction`/`eval_set`, `refit=True` + `get_refit_params()` (freezes `lr_` and selected rounds — both exist on the new class; assert the refit model is a `DistributionalBoosting`), and `sample_weight` + `eval_sample_weight`.

**Wrapper guard:** `ChimeraBoostClassifier` must reject `loss="Gaussian"` if a user passes it (classifier constructs its own loss; add an explicit early `ValueError` if reachable).

### 7.3 LR probe

`_run_learning_rate_probe` (sklearn_api.py:662): the fallback chain `getattr(context_model, "loss_name", None) → context_model.loss_.name → "RMSE"` (sklearn_api.py:696–700) resolves correctly because `DistributionalBoosting.loss_name = "Gaussian"`. The probe's candidate scoring uses validation loss from `fit` → NLL, which is the right selection metric. No code change expected; add a test that `auto_learning_rate_probe=True` runs end-to-end with `loss="Gaussian"`.

### 7.4 predict / staged_predict / predict_dist / predict_interval / sample

```python
def predict(self, X):
    _check_predict_input(self, X)
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
            yield raw[:, 0]
    else:
        yield from self.model_.staged_predict_raw(X)

def predict_dist(self, X):
    self._require_gaussian("predict_dist")
    _check_predict_input(self, X)
    return self.model_.predict_dist(X)        # (mu, sigma)

def predict_interval(self, X, alpha=0.1):
    self._require_gaussian("predict_interval")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    mu, sigma = self.predict_dist(X)
    from scipy.special import ndtri            # scipy is a hard dependency of scikit-learn — no new install requirement
    zq = ndtri(1.0 - alpha / 2.0)
    return mu - zq * sigma, mu + zq * sigma

def sample(self, X, n_samples=1, random_state=None):
    self._require_gaussian("sample")
    mu, sigma = self.predict_dist(X)
    rng = np.random.default_rng(random_state)
    return rng.normal(mu[:, None], sigma[:, None], size=(mu.shape[0], int(n_samples)))
```

`_require_gaussian(name)` raises `AttributeError`-style `ValueError`: `f"{name}() requires loss='Gaussian'; this model was fit with loss='{self.loss}'"`. Follow sklearn convention: these methods check `check_is_fitted` via the same `_check_predict_input` path as `predict`.

### 7.5 predict() dispatch caveat

The scalar losses return `(n,)` from `predict_raw` while Gaussian returns `(n, 2)` — the branch above keys on `self.loss`, which is correct for a fitted wrapper but **also make the loaded-model path set `self.loss = "Gaussian"`** (§8) so a wrapper reconstructed from npz dispatches correctly.

### 7.6 Diagnostics

Record in `auto_params_` (already handled by booster §5.2); no wrapper change. `fit` diagnostics (LR clipping, low-ESS warnings) run in `_BaseBooster` and apply unchanged.

### 7.7 Exports

`chimeraboost/__init__.py`: no new top-level exports needed (`DistributionalBoosting` stays a core-level class like `GradientBoosting`; users go through the wrapper). Optionally export `GaussianNLL` for power users — skip in v1.

### 7.8 Tuner

`ChimeraBoostSearchCV` phase definitions assume scalar losses and tree-mode lanes. v1: add a guard in the tuner's estimator validation that raises `NotImplementedError("loss='Gaussian' is not supported by ChimeraBoostSearchCV yet")`. Grep tuning/search.py for where it reads `estimator.loss` or clones params; insert the check there.

---

## 8. Component 5 — serialization.py

### 8.1 Save

`save_booster` branches on model class (agent-verified layout: GradientBoosting header at serialization.py:646–648, MulticlassBoosting at 636–644). Add a third branch:

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

The `*_per_class` reconstruction is multiclass-only and irrelevant here — the `"multi"` kind path above is the one distributional trees take. Bump `FORMAT_VERSION` 2 → 3 **only if** loaders key strictly on version; if the loader tolerates extra header fields, keep version 2 and rely on `model_class` dispatch (prefer the latter; check how version is enforced at load and match the existing tolerance policy).

Wrapper-level `save_model`/`load_model` delegate to these; ensure the reconstructed `ChimeraBoostRegressor` gets `loss="Gaussian"` set from the header so §7.5 dispatch works, and `predict_dist` on a loaded model equals the pre-save model to float64 exactness.

---

## 9. Performance specification

Design targets and why they hold:

1. **Per-round cost ≈ 1.7–2.2× scalar RMSE lightgbm-mode, not more.** The added work is exactly K=2 in the multiclass kernels: histogram fill writes 2 grad + 2 hess lanes vs 1+1 (plus the shared count lane both paths have); split scan sums 2 output gains per bin (tree.py:3270–3289). These kernels are the same compiled functions the multiclass path already runs — no new hot code. The gradient kernel adds one `exp` per row per round (§4.1), amortized noise against histogram bandwidth (tree.py's own comments identify memory bandwidth as the bound).
2. **No allocation in the round loop.** `grad_buffer`/`hess_buffer` reused via `grad_hess_class_major_into`; the `(n, 2)` row-major shadows are allocated once and refilled by transpose-assign then passed as `grad_row_major=`/`hess_row_major=` (omitting them makes the builder allocate fresh transposes every round, tree.py:5896–5903); histogram buffers allocated once via `_alloc_multiclass_hist_buffers` (booster.py:1343) and reused via the `hist_buffers=` argument — this is already the multiclass pattern (booster.py:2384–2387, 2495–2513); copy it. Buffer memory: `2 · 2 · n_features · max_leaves · max_bins · 8B` (hg+hh) + count lane; e.g. 100 features × 64 leaves × 254 bins ≈ 52 MB total — same order as a 2-class multiclass fit.
3. **Train-F update is the leaf-gather fast path.** `add_multiclass_leaf_values_inplace` (O(n·K) gather, tree.py:2485) — this path *already avoids* the known `tree.add_predict` re-traversal inefficiency flagged in the scalar MAE/Quantile path, so the distributional loop starts on the fast lane.
4. **Prediction:** flat vector ensemble (`FlatNonObliviousEnsemble.add_predict_class_major`) with the existing `n ≥ 8192 → parallel` dispatch (tree.py:3813). `predict_dist` adds one vectorized `np.exp` over `(n,)`.
5. **JIT warm-up:** all tree kernels are shared with multiclass (identical signatures/dtypes ⇒ cached compilations reused). New compilation surface = 3 small loss kernels. Keep them `cache=True`.
6. **Class-major locality:** K=2 rows of F are `2 × n` contiguous; the class-major scatter concern from the perf review (large-K multiclass) is minimal at K=2.
7. **Do not** introduce float32 here; that's an orthogonal existing roadmap item. Match float64 conventions throughout.

Benchmark gate (must hold before merge, via `benchmarks/bench_distributional.py`, §11): on a 500k-row synthetic, `loss="Gaussian"` fit time ≤ 2.5× `loss="RMSE"` fit time at equal rounds/leaves, and ≥ 5× faster than NGBoost at equal round count (NGBoost fits sklearn trees per round; this should be a blowout — treat it as a sanity floor, not a target).

---

## 10. Test plan (tests/test_distributional.py, new file)

Unit — loss math:
1. **Finite-difference gradient check:** random `(y, F)`, ε=1e-6 central differences on `nll_i` vs `_gaussian_nll_grad_hess_into` outputs for both heads, weighted and unweighted; rtol 1e-5. Also check `h_mu` equals FD second derivative w.r.t. μ.
2. **Init:** `init_class_major` equals numpy weighted mean / 0.5·log(weighted MLE var); constant-`y` target yields finite clipped ρ0 and a fit that does not produce NaN.
3. **Clipping:** `F[1] = ±1e3` produces finite grad/hess/eval.
4. **CRPS:** closed form vs Monte-Carlo CRPS estimate on a small grid (1e5 draws, atol 5e-3); CRPS ≥ 0.

Training behavior:
5. **NLL decreases:** train NLL (with `eval_train_loss=True`) strictly non-increasing over the first 50 rounds on a synthetic set (allow tiny tolerance).
6. **Heteroscedastic recovery:** `n=20_000`, `y = sin(3x₀) + (0.3 + |x₁|)·ε`, `ε~N(0,1)`: after fit, `pearsonr(sigma_hat, 0.3+|x₁|) > 0.8` and central 90% interval coverage in `[0.86, 0.94]` on a held-out set. Seeded.
7. **Homoscedastic sanity:** constant true σ=2.0 → `median(sigma_hat) ∈ [1.8, 2.2]`.
8. **Weight equivalence:** `sample_weight=2.0` on a subset ≡ duplicating those rows (predict_dist allclose, rtol 1e-6) with fixed `random_state` and no early stopping.
9. **Point-prediction quality:** test-RMSE of `predict()` within 5% of a `loss="RMSE"` fit with identical params on dataset (6) — the μ head must not pay a meaningful accuracy tax.

API and wiring:
10. `predict(X) == predict_dist(X)[0]` exactly; `staged_predict` yields `(n,)` μ arrays (not `(n, 2)` raw) and its final element equals `predict`; interval symmetric about μ; `sample` mean/std → (μ, σ) as draws grow.
11. **Early stopping** on eval_set fires; `best_iteration_` set; `use_best_model` truncation shortens `trees_`; `refit=True` runs end-to-end and `isinstance(reg.model_, DistributionalBoosting)` holds after refit (regression test for the hardcoded-`GradientBoosting` refit site, sklearn_api.py:1116); `get_refit_params()` round-trip runs.
12. **Auto LR:** `learning_rate=None` resolves; `auto_params_["learning_rate"]` records the Gaussian→RMSE rule source; probe (`auto_learning_rate_probe=True`) runs.
13. **Guards raise:** Gaussian + `tree_mode="catboost"` (and `"auto"`, `"hybrid"`, `"depthwise"`); + `sampling="goss"`; + `subsample=0.5`; + `bootstrap_type="bayesian", bagging_temperature=0.5`; + `colsample=0.8`; + `ordered_boosting=True`; + `histogram_dtype="float32"` (must raise, not silently fall back to float64); classifier + Gaussian; `predict_dist` on `loss="RMSE"` model; non-default `alpha` + Gaussian; tuner + Gaussian → NotImplementedError.
14. **Serialization round-trip:** save/load → `predict_dist` allclose (rtol 0, atol 0 expected — same arrays), header contains `n_outputs=2` and no `n_classes`, loader derives leaf width from `n_outputs` (§8.2), loaded wrapper `predict` dispatches to μ; a saved+loaded **multiclass** model still round-trips (guards the shared `kind == "multi"` width edit).
15. **Categoricals:** fit with `cat_features` on a mixed-frame; runs and predicts (exercises the RMSE-style prep gate extension of §5.2 step 1).
16. **Existing suite stays green:** no behavior change for any other loss (the only shared-code edits are the two `loss_name` set-literal extensions and auto_params mapping — assert `loss="RMSE"` fits byte-identical predictions before/after on a fixed seed, which the existing tests already effectively cover; run the full suite).

---

## 11. Benchmark script — benchmarks/bench_distributional.py

Follow the structure/CLI conventions of `benchmarks/bench_vs_lightgbm.py`. Datasets: the synthetic heteroscedastic generator from test (6) at 100k/500k rows, plus 2–3 OpenML regression sets already used by the existing bench harness. Contenders (each behind a soft import; skip with a printed notice if missing):

- ChimeraBoost `loss="Gaussian"` (this work)
- NGBoost (`Normal` dist), equal rounds
- CatBoost `loss_function="RMSEWithUncertainty"`
- LightGBM twin-model baseline: model A = mean (L2); model B = L2 on `log((y−μ̂_oof)² + eps)` via out-of-fold μ̂ — the "practical hack" this feature replaces
- ChimeraBoost quantile pair (α=0.05, 0.95) — interval-only baseline

Metrics per contender: validation NLL, CRPS, 90% empirical coverage, mean interval width, fit wall-time (kernels warmed; respect the existing benchmark-fairness note about timing encoding inside vs outside the timer — encode outside for all contenders). Emit a markdown table; add the result summary to BENCHMARK_NOTES.md.

---

## 12. Follow-ups explicitly out of scope (record in README "not implemented" list)

- **Per-parameter scalar trees** (NGBoost-style: one scalar tree per head per round) — would unlock `tree_mode="catboost"`/oblivious and ordered boosting for distributional fits via the per_class machinery (booster.py:2295–2405).
- **Sampling relaxations:** uniform subsample + colsample first (feature_mask is already a builder param; the empty-draw `tree.depth==0 → break` bug noted in the July review must be fixed before enabling subsample here).
- **More heads:** `Poisson`, `NegativeBinomial` (count sports stats), `StudentT` (heavy tails) — each is one `VECTOR_LOSSES` class; `StudentT`/NB are 2–3 output. Multi-quantile shared tree (all α in one model, monotone rearrangement at predict).
- **Public custom vector-loss protocol:** document the `n_outputs` / `init_class_major` / `grad_hess_class_major_into` / `eval_class_major` duck-type and accept user instances in `DistributionalBoosting(loss=<instance>)`. Deferred only because serialization of arbitrary user losses needs a story (`loss_name` round-trip).
- **CRPS-based early stopping** (`eval_metric="crps"`).
- Tuner lanes for Gaussian (structure phase is loss-agnostic; only the objective metric changes).

## 13. Suggested implementation order (each step lands green)

1. **losses.py:** kernels + `GaussianNLL` + `VECTOR_LOSSES` + tests (1)–(4). No integration yet.
2. **booster.py:** `DistributionalBoosting` fit/predict/staged + guards + auto_params touchpoints (§6) + tests (5)–(9), (13-core-level).
3. **sklearn_api.py:** routing, `predict_dist`/`predict_interval`/`sample`, probe check, tuner guard + tests (10)–(13), (15).
4. **serialization.py:** save/load branch + test (14).
5. **Benchmark + docs:** §11 script, README section (usage block from §1, constraints, `min_child_weight` semantics note from §6), CHANGELOG entry, run full suite (16).

Estimated new code: ~120 lines losses.py, ~250 lines booster.py, ~90 lines sklearn_api.py, ~40 lines serialization.py, ~200 lines tests, ~150 lines benchmark.
