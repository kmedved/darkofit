# Implementation Spec: Distribution-Head Protocol + Student-t, Poisson, Negative Binomial, LogNormal

**Status:** implemented in this working tree for M0-M4 (protocol, LogNormal,
fixed-ν StudentT, Poisson, and NB global dispersion). M5 NB heterodispersion
remains gated/deferred by §4.2.
**Prerequisites already landed:** W0 metric consistency (`_GAUSS_EVAL_Z_GUARD = 1000.0`, unclipped CRPS), W1 scalar+affine sigma calibration, serialization `n_outputs == loss.n_outputs` check.
**Line anchors** are as of the current branch tip (`6a5e51c` + WNBA working tree); re-locate by symbol if drifted. When this spec and the source disagree on a line number, trust the symbol.
**Audience:** implementing agent (Codex). Self-contained: every formula, kernel pattern, integration point, and test is specified. The Gaussian head (`DISTRIBUTIONAL_REGRESSION_SPEC.md`) is the style/precision precedent; this spec extends it.

Adjudicated decisions carried in from the Oracle reviews (do not relitigate):

- **Student-t is fixed-ν, K=2** (`loss_kwargs={"nu": ...}`), not learned-ν K=3 — learning ν needs polygamma (absent from numba) and a third head for marginal benefit.
- **NB ships in two phases:** K=1 `global_dispersion=True` first (low risk, immediately useful), heterodispersion K=2 second and only behind an evidence gate.
- **SciPy usage policy:** kernels and core never import scipy. Wrapper-level quantile functions (`t.ppf`, `poisson.ppf`, `nbinom.ppf`) may import `scipy.stats` locally — scipy is a guaranteed transitive dependency of scikit-learn (pyproject depends on scikit-learn; sklearn hard-requires scipy). Gaussian keeps `statistics.NormalDist`.
- **Calibration vocabulary generalizes** (`sigma_calibration` → `dist_calibration`, deprecation alias kept for one release): continuous heads calibrate *scale*; Poisson calibrates *mean*; NB calibrates *mean* (+ optional dispersion).

---

## Part I — W7: the distribution protocol (prerequisite for every head)

### 1.1 Why

Before M0, the public surface was Gaussian-shaped in exactly these places
(historical source anchors retained for implementation context):

| Seam | Where | Before M0 |
|---|---|---|
| Core `predict_dist` | booster.py:3074–3076 | hardcodes `GaussianNLL.mean_and_sigma(raw)` |
| Wrapper point prediction | sklearn_api.py:1400 (`predict`), :1477 (`staged_predict`) | `raw[:, 0]` behind `self.loss == "Gaussian"` |
| Wrapper capability guard | sklearn_api.py:1643 `_require_gaussian` | name + message Gaussian-specific |
| Calibration application | `_predict_dist_checked` (sklearn_api.py:~1652) | assumes `(mu, sigma)` tuple, scales element 1 |
| Interval / sample | wrapper methods | assume Normal quantiles / `rng.normal` |
| Scorer default | tuning/scoring.py:51 | `loss == "Gaussian"` → `neg_gaussian_nll` |
| Tuner space/lanes | tuning/spaces.py:179, tuning/search.py (`gaussian = getattr(...) == "Gaussian"`) | string check |
| Eval-metric normalization | booster.py `_normalize_eval_metric` | hardcodes `{"nll", "crps"}` for Gaussian |
| Target validation | `DistributionalBoosting.fit` | none beyond float coercion (counts/positivity impossible to express) |

Every one of these becomes a protocol delegation. **The refactor must be behavior-preserving for Gaussian** — that is the M0 acceptance test (§1.8).

### 1.2 Protocol definition

Duck-type on every class registered in `VECTOR_LOSSES` (losses.py, after `GaussianNLL`). No ABC import — match the library's protocol-by-convention style, but document the contract in the losses.py module docstring.

Class attributes:

```python
name: str                     # registry key and serialized loss_name, e.g. "StudentT"
distribution_name: str        # human name for docs/metadata, e.g. "student_t"
n_outputs: int                # K — width of raw score vector per row
is_classification = False
adjusts_leaves = False
constant_hessian = False      # per existing loss attribute conventions
default_eval_metric: str      # e.g. "nll"
supported_eval_metrics: tuple # e.g. ("nll", "crps") — feeds _normalize_eval_metric
calibration_targets: tuple    # subset of ("scale", "mean", "dispersion"); () = none
interval_support: bool        # predict_interval implemented?
sample_support: bool          # sample() implemented? (all v1 heads: True)
```

Methods (signatures exact; `raw` is always the sample-major `(n, K)` output of `predict_raw`):

```python
def validate_target(self, y):
    """Raise ValueError on domain violations (counts, positivity). No-op default."""

def init_class_major(self, y, sample_weight=None) -> np.ndarray:   # (K,)
def grad_hess_class_major_into(self, y, F, sample_weight, grad_out, hess_out): ...
def eval_class_major(self, y, F, sample_weight=None) -> float:      # NLL, W0 guard policy
# crps_class_major only where closed-form exists (Gaussian). Listing "crps" in
# supported_eval_metrics without the method is a bug; _normalize_eval_metric checks.

def mean_from_raw(self, raw) -> np.ndarray:          # (n,) point prediction for predict()
def params_from_raw(self, raw) -> tuple:             # natural-parameter arrays, documented per head
def variance_from_raw(self, raw) -> np.ndarray:      # (n,) predictive variance — the Kalman consumer
def interval_from_raw(self, raw, alpha) -> tuple:    # (lo, hi); raise NotImplementedError if not interval_support
def sample_from_raw(self, raw, rng, n_samples) -> np.ndarray:  # (n, n_samples)
```

`variance_from_raw` is **required** on every head — it is the quantity the DARKO Kalman filter consumes, and for Student-t it is *not* `scale²` (see §2.6). Calibration hooks are wrapper-level (§1.5), not loss methods, because they need validation data.

### 1.3 Gaussian retrofit (no behavior change)

On `GaussianNLL`: add `distribution_name="gaussian"`, `default_eval_metric="nll"`, `supported_eval_metrics=("nll", "crps")`, `calibration_targets=("scale",)`, `interval_support=True`, `sample_support=True`, `validate_target` = no-op, and:

```python
def mean_from_raw(self, raw):
    return raw[:, 0].copy()

def params_from_raw(self, raw):
    return GaussianNLL.mean_and_sigma(raw)       # (mu, sigma); keep static alias

def variance_from_raw(self, raw):
    _, sigma = GaussianNLL.mean_and_sigma(raw)
    return sigma * sigma

def interval_from_raw(self, raw, alpha):
    from statistics import NormalDist
    mu, sigma = GaussianNLL.mean_and_sigma(raw)
    zq = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return mu - zq * sigma, mu + zq * sigma

def sample_from_raw(self, raw, rng, n_samples):
    mu, sigma = GaussianNLL.mean_and_sigma(raw)
    return rng.normal(mu[:, None], sigma[:, None], size=(mu.shape[0], int(n_samples)))
```

### 1.4 Core booster edits (booster.py)

1. `DistributionalBoosting.fit`: call `self.loss_.validate_target(y)` immediately after `validate_target_vector` (before preprocessing — domain errors must not cost a binning pass).
2. `predict_dist` (:3074–3076) → `return self.loss_.params_from_raw(raw)`. Add `predict_variance(X)` → `self.loss_.variance_from_raw(self.predict_raw(X))`.
3. `_normalize_eval_metric(eval_metric, loss_name)` → gains a `supported` argument (or resolve from `VECTOR_LOSSES[loss_name]`): accepted set = `{"loss", "auto", None} ∪ loss.supported_eval_metrics`; `"auto"`/None → `loss.default_eval_metric`. The Gaussian error message generalizes.
4. `_eval_metric_class_major` dispatch: `"crps"` → `loss_.crps_class_major` (unchanged); everything else → `eval_class_major`. Guard at fit time that the resolved metric is in `supported_eval_metrics`.
5. The fit loop, buffers, and histogram allocation are already K-generic (`K = int(self.loss_.n_outputs)`) — **no changes**. K=1 heads flow through `_alloc_multiclass_hist_buffers(1, ...)` and `build_leafwise_multiclass_tree` unchanged (grad/hess `(1, n)` class-major; leaf values `(n_leaves, 1)`). Accept the small vector-path overhead for K=1 in v1; a scalar-path route is a later optimization, not a correctness need.

### 1.5 Wrapper edits (sklearn_api.py)

1. Replace every `self.loss == "Gaussian"` branch (:1400, :1477, :1559, :1630, :1637) with `self.loss in VECTOR_LOSSES` (import `VECTOR_LOSSES` from `.losses`); keep the lightgbm-tree-mode guard and alias normalization exactly as-is, message updated to name the loss.
2. `predict` / `staged_predict`: `self.model_.loss_.mean_from_raw(raw)` (copy semantics now live in `mean_from_raw`).
3. `_require_gaussian(method)` → `_require_distributional(method, capability)`: raises unless `self.loss in VECTOR_LOSSES`; for `capability="interval"` additionally checks `loss.interval_support` and raises `NotImplementedError` with the head-specific reason (e.g. "Student-t intervals require scipy t quantiles — available; Poisson exact intervals land in v1.1").
4. `_predict_dist_checked` → delegate calibration by target:
   - `calibration_targets` containing `"scale"`: apply scalar `s` / affine `(a, b)` to the scale element of `params_from_raw` output (element index declared by the head as `scale_param_index = 1`).
   - `"mean"` (Poisson/NB): multiply the mean element by `c`.
   - Store which was applied in `auto_params_["dist_calibration"]`.
5. **Rename** `sigma_calibration` → `dist_calibration` (constructor param, `_SKLEARN_ONLY`, wrapper state keys, metadata). Back-compat: accept `sigma_calibration=` kwarg for one release with a `DeprecationWarning`, mapping onto `dist_calibration`; keep reading old wrapper-state keys on load (`sigma_scale` → generic `dist_scale`). Mode vocabulary per head:
   - Gaussian/StudentT/LogNormal: `"scalar"`, `"affine"` (scale-target; existing machinery).
   - Poisson: `"scalar"` = mean calibration, closed form `c = Σwy/Σwλ̂` (§3.6).
   - NB: `"scalar"` = mean calibration; `"dispersion"` = 1-D dispersion multiplier (§4.7).
   - Validation: requested mode must be legal for `calibration_targets`; else ValueError naming the head.
6. `predict_interval` / `sample` delegate to `interval_from_raw` / `sample_from_raw` after applying calibration to the raw-derived params (concretely: build a small `_calibrated_params(raw)` helper the interval/sample paths share, so calibration is applied exactly once and identically everywhere).

### 1.6 Tuning edits

- scoring.py: default scorer map keyed by loss name — `"Gaussian"`→`neg_gaussian_nll` (exists), add `"StudentT"`→`neg_student_t_nll`, `"Poisson"`→`neg_poisson_nll`, `"NegativeBinomial"`→`neg_negative_binomial_nll`, `"LogNormal"`→`neg_lognormal_nll`. Each `_neg_*` helper mirrors `_neg_gaussian_nll`'s shape: wrapper `predict_dist`/params → weighted mean NLL with the same W0 overflow-guard policy as the corresponding eval kernel. (These score *calibrated* params via the wrapper — intentional, same as Gaussian today.)
- search.py: `gaussian = getattr(self.estimator, "loss", None) == "Gaussian"` → `distributional = getattr(...) in VECTOR_LOSSES`; the lightgbm-lane restriction and error message generalize.
- spaces.py:179: the Gaussian sampling-space branch keys on `loss in VECTOR_LOSSES`. Add per-head extras: StudentT gets `trial.suggest_categorical(f"{prefix}_student_t_nu", [3.0, 4.0, 6.0, 10.0, 30.0])` routed through wrapper `dist_params`; NB-global gets nothing extra in v1 (dispersion is resolved, not tuned).
- Fold-calibration pooling (`_pooled_trial_sigma_calibration`): scalar scale pools by mass (exists); Poisson mean calibration pools **exactly** by summing per-fold `(Σwy, Σwλ̂)` — store both; NB mean and dispersion multipliers pool by validation-mass-weighted log scale; affine pools coefficients by validation mass.

### 1.7 Wrapper loss_kwargs plumbing

Today the wrapper hardcodes `loss_kwargs={}` for Gaussian. Heads with parameters need a route: add wrapper constructor param `dist_params=None` (dict), validated per head (`StudentT`: requires `nu > 2`, float; `NegativeBinomial`: optional `global_dispersion: bool = True`, optional fixed `r`; others: must be empty). `make_model` passes `loss_kwargs=dict(self.dist_params or {})`. Serialized via existing `loss_kwargs` header field. Do **not** overload `alpha`.

### 1.8 Serialization edits

- Add optional `header["loss_state"]` (jsonified dict) written when `getattr(booster.loss_, "state_", None)` is non-empty; restored onto the loss instance after construction on load. First consumer: NB global dispersion `{"r": float, "source": "..."}` (§4.4). Absent for other heads — loader must tolerate absence (old archives).
- The `n_outputs == loss.n_outputs` check (serialization.py:1105–1108) already lands generically. Add per-head round-trip tests.

### 1.9 M0 acceptance (protocol refactor, before any new head)

- Full suite green.
- **Gaussian regression pin:** fixed-seed Gaussian fit before/after refactor produces byte-identical `predict`, `predict_dist`, `predict_interval`, save/load payloads (excluding renamed state keys), and identical tuner trial scores on a 6-trial smoke study.
- `dist_calibration="scalar"` and legacy `sigma_calibration="scalar"` produce identical fits (alias test + DeprecationWarning assertion).

---

## Part II — the heads

Shared kernel style rules (all heads; copy the Gaussian kernels' discipline):

- Explicit `if sample_weight is None: w = 1.0 / else:` blocks (never inline ternary — numba Optional typing).
- `w <= 0.0` rows: write exact-zero grad/hess and `continue` **before any parameter arithmetic**; skip in eval reductions.
- Clip raw scores (η, ρ, κ) to head-specific `[MIN, MAX]` before `exp`.
- Grad kernels `@njit(cache=True, parallel=True)` with `prange`; eval kernels `@njit(cache=True)` serial `range` (matches Gaussian; sidesteps parfor reduction fragility).
- Eval NLL = **true** NLL with only an overflow guard per the W0 policy (guard values per head below). No |z|≤10-style robust clips in eval — that was the W0 bug class.
- Fisher/hessian must be strictly positive for every positive-weight row (split legality flows through summed hessians + count gate in the vector tree path).
- Weighted convention: multiply grad and hess by `w` (matches every existing kernel).

Per-head training-clip policy summary (decided here, rationale inline below):

| Head | Training grad clip | Eval overflow guard |
|---|---|---|
| Gaussian (exists) | z ∈ [−10, 10] straight-through | z ∈ [−1e3, 1e3] |
| StudentT | **none needed** — gradients intrinsically bounded | z ∈ [−1e150, 1e150] (log1p arg overflow only) |
| Poisson | η clip ±15 only; `g = λ − y` left exact | none beyond η clip (NLL linear in y·η) |
| NegBinomial | η, κ clips ±15; dispersion-hess floor/cap | none beyond clips |
| LogNormal | inherits Gaussian (on log y) | inherits Gaussian |

---

## 2. Student-t head (`loss="StudentT"`, K=2, fixed ν)

**Use case:** heavy-tailed continuous targets — the W5 tripwire outcome (per-bin RMS ≈ 1 but coverage short after affine calibration), and robust μ/scale for outlier-prone sports metrics. As ν→∞ this reproduces Gaussian; ν ∈ [3, 30] is the useful band.

### 2.1 Parametrization and constants

```python
_T_RHO_MIN, _T_RHO_MAX = -15.0, 15.0          # rho = log(scale), same rationale as Gaussian
_T_INIT_RHO_MIN, _T_INIT_RHO_MAX = -10.0, 10.0
_T_EVAL_Z_GUARD = 1e150                        # only so z*z cannot overflow float64
```

`F[0, i] = mu_i` (location), `F[1, i] = rho_i = log(scale_i)`. `nu` fixed from `loss_kwargs`; constructor:

```python
class StudentTNLL:
    name = "StudentT"
    distribution_name = "student_t"
    n_outputs = 2
    default_eval_metric = "nll"
    supported_eval_metrics = ("nll",)          # no closed-form CRPS in v1 (exists in literature; deferred)
    calibration_targets = ("scale",)
    interval_support = True                    # scipy t.ppf, wrapper-level
    sample_support = True

    def __init__(self, nu=6.0, hessian_mode="natural"):
        nu = float(nu)
        if not nu > 2.0:
            raise ValueError("StudentT requires nu > 2 (finite variance); got %r" % nu)
        if hessian_mode != "natural":
            raise ValueError("hessian_mode='natural' is the only supported mode")
        self.nu = nu
        self.hessian_mode = hessian_mode
```

ν ≤ 2 is rejected because `variance_from_raw` (the Kalman consumer) is `scale²·ν/(ν−2)`.

### 2.2 Loss, gradients, Fisher

With `z = (y − μ)/scale`:

```
nll = rho + 0.5·log(ν·π) + lgamma(ν/2) − lgamma((ν+1)/2) + 0.5·(ν+1)·log1p(z²/ν)
```

The three lgamma/log constants depend only on ν — **precompute once in `__init__`** (`self._nll_const = 0.5*math.log(nu*math.pi) + math.lgamma(nu/2.0) - math.lgamma((nu+1.0)/2.0)`) so kernels never call lgamma per row. Pass `nu` and `_nll_const` into the kernels as scalar arguments (numba-friendly; do not close over `self`).

Gradients (exact derivatives of the NLL above):

```
g_mu  = −(ν+1)·z / ((ν + z²)·scale)
g_rho = 1 − (ν+1)·z² / (ν + z²)
```

**No training clip is needed** — verify and test these bounds: `|g_mu| ≤ (ν+1)/(2·√ν·scale)` (maximum at |z|=√ν) and `g_rho ∈ (−ν, 1]`. The heavy-tail robustness that Gaussian gets from the z-clip, Student-t has natively (gradients *redescend* in z). This is the head's whole point; do not add a clip.

Fisher diagonal (natural gradient, exact standard results for the location-scale t):

```
h_mu  = w · ((ν+1)/(ν+3)) / scale²
h_rho = w · 2ν/(ν+3)
```

Both strictly positive for w > 0 ✓ (invariant holds). Note `h_rho` is constant-per-row like Gaussian's `2w`.

Kernel skeleton (grad; eval analogous with the serial/guard pattern):

```python
@njit(cache=True, parallel=True)
def _student_t_grad_hess_into(y, F, sample_weight, nu, grad_out, hess_out):
    n = F.shape[1]
    fisher_mu_coef = (nu + 1.0) / (nu + 3.0)
    fisher_rho = 2.0 * nu / (nu + 3.0)
    for i in prange(n):
        # weight block per shared rules (w<=0 -> zero all four, continue)
        r = F[1, i]
        if r < _T_RHO_MIN: r = _T_RHO_MIN
        elif r > _T_RHO_MAX: r = _T_RHO_MAX
        scale = np.exp(r)
        z = (y[i] - F[0, i]) / scale
        denom = nu + z * z
        grad_out[0, i] = w * (-((nu + 1.0) * z) / (denom * scale))
        hess_out[0, i] = w * fisher_mu_coef / (scale * scale)
        grad_out[1, i] = w * (1.0 - (nu + 1.0) * z * z / denom)
        hess_out[1, i] = w * fisher_rho
```

Eval kernel: same weight/clip discipline; guard `if z > _T_EVAL_Z_GUARD: z = _T_EVAL_Z_GUARD` (± symmetric) purely so `z*z` stays finite; `nll_i = r + nll_const + 0.5*(nu+1.0)*np.log1p(z*z/nu)` — note the tail grows only logarithmically, so no robustness clip is needed or wanted.

### 2.3 Initialization

Robust location/scale (weighted median + MAD), falling back to moments when MAD degenerates:

```python
def init_class_major(self, y, sample_weight=None):
    # filter w > 0 rows exactly as GaussianNLL.init_class_major does
    mu0 = _weighted_median(y, weights)                       # losses.py:23 helper exists
    mad = _weighted_median(np.abs(y - mu0), weights)
    scale0 = 1.4826 * mad * math.sqrt((self.nu - 2.0) / self.nu)   # MAD→std, std→t-scale
    if not scale0 > 0.0:                                     # constant/duplicate-heavy y
        var0 = float(np.average((y - mu0) ** 2, weights=weights))
        scale0 = math.sqrt(max(var0, 1e-12) * (self.nu - 2.0) / self.nu)
    rho0 = float(np.clip(math.log(max(scale0, 1e-12)), _T_INIT_RHO_MIN, _T_INIT_RHO_MAX))
    return np.array([mu0, rho0])
```

(The `(ν−2)/ν` factor converts a *standard-deviation* estimate to the t *scale* parameter, since `Var = scale²·ν/(ν−2)`. The MAD→σ constant 1.4826 is the Gaussian one — an approximation for t; fine for an init, note it in a comment.)

### 2.4 Protocol methods

```python
def mean_from_raw(self, raw):            # location = mean for nu > 1
    return raw[:, 0].copy()

def params_from_raw(self, raw):          # documented tuple: (mu, scale, nu_array)
    rho = np.clip(raw[:, 1], _T_RHO_MIN, _T_RHO_MAX)
    scale = np.exp(rho)
    return raw[:, 0].copy(), scale, np.full(raw.shape[0], self.nu)

def variance_from_raw(self, raw):        # THE Kalman quantity — not scale**2
    _, scale, _ = self.params_from_raw(raw)
    return scale * scale * (self.nu / (self.nu - 2.0))

def interval_from_raw(self, raw, alpha):
    from scipy.stats import t as _t      # wrapper-level import per scipy policy
    mu, scale, _ = self.params_from_raw(raw)
    q = float(_t.ppf(1.0 - alpha / 2.0, df=self.nu))
    return mu - q * scale, mu + q * scale

def sample_from_raw(self, raw, rng, n_samples):
    mu, scale, _ = self.params_from_raw(raw)
    draws = rng.standard_t(self.nu, size=(mu.shape[0], int(n_samples)))
    return mu[:, None] + scale[:, None] * draws
```

**Never call the second parameter "sigma"** anywhere user-facing — it is the t scale, not the standard deviation. `predict_dist` docstring must state `Var = scale²·ν/(ν−2)` and point to `predict_variance`.

### 2.5 Calibration

`dist_calibration="scalar"`: multiplier `s` on scale, fit by **1-D golden-section over log s** on the weighted validation t-NLL (no Gaussian closed form — the t-NLL scale MLE has no closed form). Reuse the golden-section scaffolding from `_fit_affine_sigma_calibration` (sklearn_api.py:232) with `b` frozen at 1 and the objective swapped to the t eval kernel. `"affine"`: same `(a, b)` machinery with the t-NLL objective; profile-`a` is no longer closed-form, so run the existing 2-level search (golden over `b`, inner golden over `a`) — bounded, deterministic, still cheap. Influence diagnostic carries over unchanged.

### 2.6 Booster/wrapper/tuner/serialization

- Guards identical to Gaussian (lightgbm-only, no GOSS/MVS/bayesian/ordered/float32-hist); uniform subsample + colsample allowed (K-generic loop unchanged).
- Auto-LR: add `"StudentT": ("RMSE", 1.0)` to `_FALLBACK_LOSS` (auto_params.py:37) and to the `LIGHTGBM_UNWEIGHTED_LR_MULTIPLIERS` normalization set (booster.py:989–992); record `loss_coefficient_source`.
- Serialization: `loss_name="StudentT"`, `loss_kwargs={"nu": ...}`, `n_outputs=2`; loader constructs via `VECTOR_LOSSES` (generic path from Part I); no `loss_state`.
- Tuner: `nu` categorical (§1.6); default scorer `neg_student_t_nll`.

### 2.7 Tests

1. FD gradient check on interior points for both heads, weighted+unweighted (rtol 1e-5); FD second derivative for μ matches `(ν+1)/(ν+3)/scale²` **only in expectation** — instead assert the *analytic gradient of the exact NLL*, and assert `h_mu` equals the stated Fisher formula directly.
2. Gradient-bound test: `|g_mu·scale| ≤ (ν+1)/(2√ν) + 1e-12` and `−ν ≤ g_rho ≤ 1` over a z-grid including ±1e6.
3. ν→30 sanity: on Gaussian synthetic data, StudentT(ν=30) NLL within 2% of Gaussian NLL and μ/σ̂-vs-scale·√(ν/(ν−2)) agree within 5%.
4. Heavy-tail win: `y = f(x) + t₃ noise` — StudentT(ν=3..6) beats Gaussian on held-out true t-NLL *and* Gaussian's σ̂ is inflated vs StudentT's `variance_from_raw` implied σ.
5. `nu=2.0` and `nu=-1` constructor rejection; `variance_from_raw` finiteness.
6. Zero-weight extreme rows; weight scale-invariance (mean-one normalization); save/load round-trip with `nu` preserved; interval uses t quantiles (wider than Normal at ν=3 — assert ratio > 1.05 at alpha=0.1); calibration scalar improves or preserves validation NLL.

### 2.8 Benchmark

Add a `darkofit_student_t` lane (ν grid) to `bench_distributional.py` on a heavy-tailed synthetic + the WNBA script behind a flag. Competitors: Gaussian lane (the relevant comparison), NGBoost `T` distribution if available.

---

## 3. Poisson head (`loss="Poisson"`, K=1)

**Use case:** count stats (boxscore events). First count head; also the template for NB.

### 3.1 Parametrization and constants

```python
_POIS_ETA_MIN, _POIS_ETA_MAX = -15.0, 15.0    # lambda in [3e-7, 3.3e6]
_POIS_HESS_FLOOR = 1e-6                        # lambda floor in the hessian only
```

`F[0, i] = eta_i = log(lambda_i)`.

### 3.2 Target validation

```python
def validate_target(self, y):
    if np.any(y < 0.0):
        raise ValueError("loss='Poisson' requires nonnegative targets")
    if np.any(np.abs(y - np.rint(y)) > 1e-8):
        raise ValueError(
            "loss='Poisson' requires integer counts; for continuous nonnegative "
            "targets use loss='LogNormal' (or a future Tweedie head)"
        )
```

(Exposure/offset support — `log(exposure)` base margin — is explicitly **out of scope v1**; note it in the docstring as the known limitation for per-minute rates. Model per-game counts directly, or wait for base-margin support.)

### 3.3 Loss, gradients, hessian

```
nll = lambda − y·eta + lgamma(y + 1)
g_eta = lambda − y
h_eta = max(lambda, _POIS_HESS_FLOOR)          # exact Hessian == Fisher == lambda
```

Include `lgamma(y+1)` in eval (comparability across models — Oracle A's point); precompute nothing (it's per-row but `math.lgamma` is numba-fast; or precompute `lgamma(y+1)` once per fit into a cached array passed to the eval kernel — do this, it's an (n,) array computed in `fit`, worth it since eval runs every round). The gradient is left exact (unbounded in y only through the data itself; λ is bounded by the η clip). The hessian floor prevents huge Newton steps in leaves where λ̂→0 but positive counts exist.

Newton-step sanity (document in a comment): leaf value = −ΣwG/(ΣwH+l2) = Σw(y−λ)/(Σwλ+l2) — the exact damped multiplicative update for a log-link Poisson; no additional clipping needed beyond the η clip at the next round's exp.

### 3.4 Initialization

```python
lambda0 = max(float(np.average(y, weights=w_pos)), 1e-12)   # w>0 filtered as usual
return np.array([math.log(lambda0)])                        # shape (1,)
```

### 3.5 Protocol methods

```python
def mean_from_raw(self, raw):
    eta = np.clip(raw[:, 0], _POIS_ETA_MIN, _POIS_ETA_MAX)
    return np.exp(eta)

def params_from_raw(self, raw):          # documented tuple: (lam,)
    return (self.mean_from_raw(raw),)

def variance_from_raw(self, raw):        # Poisson: Var = mean
    return self.mean_from_raw(raw)

def interval_from_raw(self, raw, alpha):
    from scipy.stats import poisson as _poisson
    lam = self.mean_from_raw(raw)
    return _poisson.ppf(alpha / 2.0, lam), _poisson.ppf(1.0 - alpha / 2.0, lam)
    # integer bounds; document that coverage is >= nominal (discreteness)

def sample_from_raw(self, raw, rng, n_samples):
    lam = self.mean_from_raw(raw)
    return rng.poisson(lam[:, None], size=(lam.shape[0], int(n_samples))).astype(np.float64)
```

`interval_support = True` (scipy ppf), `calibration_targets = ("mean",)`, `supported_eval_metrics = ("nll",)`.

### 3.6 Calibration — closed form and exactly poolable

`dist_calibration="scalar"` fits `λ' = c·λ̂` with the weighted-NLL-optimal closed form `c = Σ wᵢyᵢ / Σ wᵢλ̂ᵢ` on the validation fold. **Store the numerator and denominator** in the calibration metadata — SearchCV pools folds *exactly* by summing them (unlike Gaussian scales, no approximation). Applied to the mean element in `_predict_dist_checked`; `variance_from_raw` after calibration returns `c·λ̂` (calibrated mean = calibrated variance — document). `"affine"` is illegal for Poisson (ValueError).

### 3.7 K=1 mechanics note

`grad_hess_class_major_into` writes `(1, n)` buffers; the vector builder, histograms, leaf values, `add_predict_class_major`, flat ensemble, and serialization width checks all take K from the arrays — verified K-generic in the original recon (no `K>=2` assumptions in tree.py). One extra test pins this: a K=1 fit round-trips serialization with `n_outputs=1` and `expected_value_width=1`.

### 3.8 Tests

FD gradient/hessian on interior η; target validation (negative, non-integer, float-integer like `3.0` accepted); zero counts and large counts (y=0 rows drive λ→0: assert η clip + hess floor keep leaf values finite over 200 rounds); zero-weight rows; weight scale-invariance; Poisson-synthetic recovery (λ(x) known: `corr(λ̂, λ_true) > 0.9`, mean calibration `c ≈ 1 ± 0.05`); calibration pooling exactness across a 3-fold SearchCV smoke; save/load; sample/interval sanity (empirical coverage ≥ nominal).

### 3.9 Benchmark

Poisson-synthetic lane vs LightGBM `objective="poisson"` and CatBoost `loss_function="Poisson"` (point λ̂ quality + NLL; those libraries give mean-only, so NLL uses their λ̂ as the full distribution — a fair comparison since Poisson has no extra parameter).

---

## 4. Negative Binomial head (`loss="NegativeBinomial"`, NB2)

**Use case:** overdispersed counts — most real sports counts (Var > mean from opponent/context heterogeneity). Two phases; phase 1 ships alone.

### 4.1 Phase 1 — `global_dispersion=True` (K=1) — *default and first*

Mean head only: `F[0] = eta = log(mu)`, a **single shared** size parameter `r` (NB2: `Var = μ + μ²/r`). With `r` fixed per round, the mean-head math is exact:

```
nll = −lgamma(y+r) + lgamma(r) + lgamma(y+1) − r·log(r/(r+mu)) − y·log(mu/(r+mu))
g_eta = r·(mu − y)/(r + mu)
h_eta = w·max(r·mu/(r + mu), _NB_HESS_FLOOR)      # Fisher; strictly positive
```

`r` resolution (`loss_kwargs={"r": None}` default = auto):

1. **Init (method of moments):** `alpha0 = max((var0 − mu0)/mu0², 1e-4)`, `r0 = 1/alpha0` from weighted marginal moments. If `var0 ≤ mu0` (under-dispersed/equi-dispersed data), warn and set `r0 = 1e6` (≈ Poisson).
2. **Refresh schedule:** every 25 rounds *and* once after early stopping/truncation, re-solve `r` by 1-D golden section on the **training** weighted NLL holding all μ̂ fixed (`log r ∈ [log r0 − 6, log r0 + 6]`). This is a generalized version of the existing loss-refinement precedent (`adjusts_leaves` losses refit leaf values mid-training); it changes the loss surface only at refresh points, which boosting tolerates — but it must be **deterministic** and recorded: append each `(round, r)` to `loss.state_["r_path"]`, final value in `loss.state_["r"]`.
3. Explicit `loss_kwargs={"r": 12.5}` skips both — fixed dispersion, no refreshes.

The eval kernel takes `r` as a scalar argument. Precompute the `lgamma(y+1)` array once per fit (as Poisson); `lgamma(y+r)`/`lgamma(r)` must be per-row/per-refresh since `r` changes — `math.lgamma` inside the kernel is acceptable (serial eval kernel).

Protocol: `params_from_raw` → `(mu, alpha_array)` with `alpha = 1/r` broadcast (**alpha, not r** — `Var = μ + αμ²` is the user-facing convention, per both reviews); `variance_from_raw` → `mu + alpha·mu²` (the Kalman quantity); `sample_from_raw` → `rng.negative_binomial(n=r, p=r/(r+mu))`; `interval_from_raw` → `scipy.stats.nbinom.ppf` with the same discreteness note as Poisson. `calibration_targets = ("mean", "dispersion")`: mean scalar uses a 1-D golden-section search for a multiplier on `mu` against validation NB-NLL; `"dispersion"` uses the same search for a multiplier on `alpha`; SearchCV pools both positive multipliers by validation-mass-weighted log scale.

Serialization: `loss_kwargs={"global_dispersion": True, "r": <explicit-or-None>}`, `loss_state={"r": fitted, "r_path": [...]}` via the §1.8 mechanism. **This is the first consumer of `loss_state` — build §1.8 in the same PR.**

### 4.2 Phase 2 — heterodispersion (K=2) — *gated, not scheduled*

Evidence gate: on ≥ 2 real count datasets, phase-1 NB beats Poisson on weighted holdout NLL, **and** residual dispersion diagnostics (per-slice `E[(y−μ̂)²]` vs `μ̂ + α̂μ̂²`) show *structured* dispersion variation that a global `r` cannot fit. Do not build it on synthetic evidence alone.

Design (recorded now so the gate decision is cheap): `F = (log μ, kappa = log r)` with clips ±15. Mean head as phase 1. Dispersion head:

```
g_kappa = r·[digamma(r) − digamma(y+r) + log1p(mu/r) + (r+y)/(r+mu) − 1]
h_kappa = w·clip(g_kappa_unweighted², _NB_KAPPA_H_FLOOR=1e-3, _NB_KAPPA_H_CAP=25.0)
```

The squared-gradient hessian is a positive Fisher *approximation* (exact Fisher needs trigamma sums with no closed form); the floor/cap keep leaf Newton steps conditioned. Default the ρ-head-style LR multiplier to 0.5 on the κ head (mechanism: scale `tree.values[:, 1]` post-build — the W3.3 device). Numba digamma (no scipy in kernels) — standard recurrence + asymptotic series:

```python
@njit(cache=True)
def _digamma(x):
    # x > 0. Recurrence to x >= 8, then asymptotic expansion.
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    # psi(x) ~ ln x - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
    return result + np.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0)
    )
```

(Accuracy ~1e-12 for x ≥ 8 post-recurrence; add a unit test against `scipy.special.digamma` on a grid `[1e-3, 1e4]`.)

### 4.3 Tests (phase 1)

Target validation shared with Poisson; **Poisson-generated data must not prefer extreme dispersion** (fitted `r ≥ 100`, NB NLL within 1% of Poisson NLL — the review's guard test); NB-generated data (known α) recovers `alpha ∈ [0.7α_true, 1.4α_true]` and beats Poisson NLL by a margin; `r` refresh determinism (two identical fits → identical `r_path`); explicit-`r` skips refreshes; zero counts / large counts / zero weights; mean + dispersion calibration; save/load restores `loss_state["r"]` and `predict_dist` exactly.

---

## 5. LogNormal head (`loss="LogNormal"`, K=2) — *cheapest head; build first after M0 to shake out the protocol*

Gaussian machinery on `u = log y`, plus a Jacobian in eval and link-outs on the way back.

- `validate_target`: `y > 0` strictly (zeros → error message suggesting Poisson/NB for counts or `log1p` preprocessing at the user's discretion — do not silently shift).
- **Kernels:** compute `u = log(y)` **once in `fit`** into a cached array and reuse the *existing Gaussian kernels verbatim* on `u` (grad/hess: identical — the Jacobian `d u` term does not involve (m, ρ)). Eval: Gaussian eval on `u` **plus** the weighted mean of `log y` (also precomputable per fold) so reported NLL is the true density of `y`, comparable across heads. Implementation: `LogNormalNLL` wraps `GaussianNLL` internals; the loss stores `self._u_train`, `self._logy_mean_train` etc. — no: statelessness is cleaner. Instead pass `u` as the `y` argument from the booster… **Decision:** `DistributionalBoosting.fit` stays target-agnostic; `LogNormalNLL.init_class_major/grad_hess/eval` receive raw `y` and compute `np.log(y)` internally; the grad kernel takes `u` precomputed by the loss object via a small per-fit cache keyed on `id(y)` — too clever. **Final decision (keep it simple):** kernels take `y` and compute `log(y)` per row per round. One `np.log` per row per round is noise against histogram bandwidth (same argument as Gaussian's `exp`), and it keeps every loss stateless. The Jacobian term in eval likewise computes `log(y[i])` inline.
- Init: weighted mean/var of `log y` (Gaussian init on u), same ρ clips.
- Training clip: Gaussian's z-clip semantics on the log scale (reuse `_GAUSS_Z_CLIP`); eval guard `_GAUSS_EVAL_Z_GUARD` likewise.
- Protocol:
  - `params_from_raw` → `(m, s)` on the **log scale** — documented explicitly as `(log_mu, log_sigma)`-style params, because that is unambiguous.
  - `mean_from_raw` → `exp(m + s²/2)` (**the mean, not the median** — `predict()` must remain squared-error-meaningful; docstring states the median is `exp(m)`).
  - `variance_from_raw` → `(exp(s²) − 1)·exp(2m + s²)`.
  - `interval_from_raw` → `exp(m ± z_q·s)` via `NormalDist` (exact; no scipy needed).
  - `sample_from_raw` → `rng.lognormal(m[:, None], s[:, None], ...)`.
  - Guard: `s` from clipped ρ as Gaussian; additionally clip `m + s²/2 ≤ 700` before `exp` in `mean_from_raw`/`variance_from_raw` (overflow guard for absurd fits; warn if triggered).
- Calibration: `("scale",)` — the existing scalar/affine machinery applies to `s` verbatim (it is a Gaussian scale on log-space). Zero new calibration code.
- `supported_eval_metrics = ("nll",)` (log-space CRPS ≠ y-space CRPS; deferred).
- Tests: `y ≤ 0` rejection; equivalence pin — LogNormal fit on `y` produces identical trees to Gaussian fit on `log y` (same seed/params; compare `predict_raw`), with eval differing by exactly the weighted mean of `log y`; mean-vs-median correctness on a known lognormal synthetic; interval coverage; save/load.

## 6. Gamma head — deferred (recorded trigger only)

Build only if a real positive-continuous dataset shows LogNormal materially miscalibrated in per-bin RMS after affine calibration (multiplicative-error targets with sub-lognormal tails). Needs the §4.2 digamma machinery for a learned shape head; with fixed shape it adds little over LogNormal. No further spec until the trigger fires.

---

## 7. Milestones (each lands green; sizes are new-code estimates)

| M | Deliverable | Gate | Est. |
|---|---|---|---|
| M0 | Part I protocol + Gaussian retrofit + rename/alias + `loss_state` + tuner/scoring generalization | Gaussian byte-identical regression pin (§1.9); full suite green | ~350 loc + tests |
| M1 | LogNormal | Gaussian-on-log-y equivalence pin; protocol exercised end-to-end incl. calibration reuse | ~150 loc + tests |
| M2 | StudentT fixed-ν | §2.7 suite incl. heavy-tail win + ν→30 sanity; WNBA tripwire lane available | ~250 loc + tests |
| M3 | Poisson | §3.8 suite incl. exact calibration pooling; K=1 serialization pin | ~250 loc + tests |
| M4 | NB phase 1 (global dispersion) + `loss_state` consumer | §4.3 suite incl. Poisson-guard test | ~300 loc + tests |
| M5 | NB phase 2 (heterodispersion) | **gated** on §4.2 real-data evidence — not scheduled | — |

Order rationale: M1 before M2/M3 because LogNormal is the cheapest full traversal of the new protocol (catches protocol design errors at minimum cost); M2 next because the W5 tripwire may fire from WNBA per-metric diagnostics at any time; counts (M3/M4) follow for the boxscore models. Benchmarks: extend `bench_distributional.py` with `--head` lanes per milestone; every head gets a synthetic-recovery lane before any real-data claim.

Every milestone inherits the standing invariants: strictly positive per-head hessians for positive weights, zero-weight rows skipped before arithmetic, raw-param clips before every `exp`, W0 eval policy (true NLL + overflow guard only), weighted mean-one convention untouched, `tree_mode="lightgbm"` shared-vector path only.
