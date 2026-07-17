# DarkoFit Practical Wishlist

The biggest DarkoFit opportunities from this exercise are around distribution contracts and calibration—not more boosting algorithms. DarkoFit's Gaussian/Student-t fitting, sampling, calibration, and persistence already worked well.

That remains the strategic thesis of this plan. A coherent calibrated-distribution contract is a more differentiated and useful product layer than another marginal boosting-accuracy feature.

## Updated roadmap disposition

| Scope | Decision |
| --- | --- |
| Package roadmap | Distribution object and contract validator; post-fit calibration; one likelihood-context/raw-margin engine; exposure and trials; Binomial and global-concentration Beta-Binomial; thin capabilities; diagnostics/CRPS; pooled empirical residual wrapper. |
| Late, decision-gated | Copula joint outputs, only after the consumer chooses a wide target layout for correlated horizons. |
| Revisit on demand | `frequency_weight`, in-package clustered/bootstrap estimators, grouped empirical residuals, and distributional SHAP. |
| Research only | Any new clustered or season-block ensemble remains in `benchmarks/` until a materially new preregistered mechanism earns package consideration. |

## Highest-value additions

### 1. Post-fit calibration on a separate dataset

Add something like:

```python
model.fit_calibrator(
    X_cal,
    y_cal,
    method=None,  # uses dist_calibration; or pass "scale" / "affine"
    sample_weight=weights,
)
```

DarkoFit already calibrates distributions during fitting, but the forecasting workflow needs a clean sequence of train → model selection → later-season calibration. The calibrator should persist with the model and work with intervals, samples, CDFs, and variances.

### 2. CDF, survival-probability, and quantile predictions

```python
model.predict_cdf(X, value=threshold)
model.predict_probability_above(X, threshold)
model.predict_quantile(X, q=[0.10, 0.50, 0.90])
```

We currently need Monte Carlo draws to calculate `P(DPM >= X)`. Direct distribution evaluation would be faster, deterministic, and more accurate.

An even cleaner long-term interface would be:

```python
dist = model.predict_distribution(X)
dist.mean()
dist.variance()
dist.cdf(threshold)
dist.quantile(0.10)
dist.sample(1_000, random_state=42)
```

### 3. External base-prediction or offset support

```python
model.fit(
    X,
    y,
    base_prediction=glum_predictions,
)

model.predict_distribution(
    X_future,
    base_prediction=glum_future_predictions,
)
```

The POC uses Glum for the conditional mean and DarkoFit for the residual distribution. We had to manage that addition ourselves. DarkoFit's existing `linear_residual` is useful, but it does not let an external estimator such as Glum supply the mean.

This should support:

- Identity-scale location offsets for Gaussian and Student-t.
- Link-scale offsets for count and positive distributions.
- Clear persistence of the required offset protocol.

### 4. Public loss-capabilities introspection

```python
darkofit.get_loss_capabilities("StudentT")
```

This should report details such as:

- Valid target domain.
- Signed-value support.
- Available distribution operations.
- Exposure, trials, and offset support.
- Supported calibration methods and evaluation metrics.
- Compatible tree and residual modes.

The POC currently hard-codes that Gaussian and Student-t can model signed residuals. A capabilities API would let generic rate-metric code select valid heads without knowing DarkoFit internals.

## Needed for percentages and exposure-based rates

### 5. Aggregate Binomial loss

```python
model.fit(X, makes, trials=attempts)
```

This is the right basic likelihood for 3P%, FT%, and similar metrics. It should accept aggregate makes and attempts without expanding every attempt into a row.

### 6. Beta-Binomial loss

This would allow observed percentage variance to exceed ordinary Binomial variance. A useful parameterization would model:

- Expected success probability.
- Concentration or overdispersion.

It would be especially useful for player shooting projections, where latent skill heterogeneity and streakiness make a plain Binomial head too narrow.

### 7. Exposure and count-offset support

Poisson and Negative Binomial models should accept:

```python
model.fit(X, counts, exposure=minutes)
# or
model.fit(X, counts, offset=np.log(minutes))
```

Prediction and sampling should optionally accept future exposure. This unlocks rates such as events per possession or per minute while keeping the count-generating process statistically coherent.

### 8. Explicit weight semantics

Keep these concepts separate in the API:

- `sample_weight`: observation reliability or modeling importance.
- `frequency_weight`: repeated equivalent observations, if a concrete consumer eventually requires it.
- `exposure`: opportunities for a count.
- `trials`: denominator for Binomial observations.

Treating all four as generic weights can produce subtly incorrect likelihoods.

The committed implementation scope is `sample_weight`, `exposure`, and `trials`. `frequency_weight` is deferred because it has no current consumer and would also change histogram counts, `min_child_samples`, effective-sample-size rules, and subsampling semantics.

## Valuable for full multiyear projections

### 9. Joint, correlated distributional outputs

The production version will eventually want correlated draws across:

- Offensive and defensive DPM.
- Years 1–5.
- Potentially multiple related metrics.

A plausible first implementation is individually fitted marginal distributions plus a learned Gaussian or Student-t copula. The sampling contract could be:

```python
draws = model.sample_joint(X, n_samples=1_000)
# (rows, samples, targets)
```

This prevents implausible paths created by independently sampling every component and horizon.

### 10. Clustered bootstrap ensembles

A distributional head captures conditional outcome variance, but not all model uncertainty. An ensemble trained using player-clustered and season-block bootstraps could combine:

- Aleatoric uncertainty from each fitted distribution.
- Epistemic uncertainty across fitted models.

This matters more at longer horizons and for unusual player profiles.

This remains a benchmark-side research idea, not an in-package estimator proposal.

### 11. A common empirical-distribution wrapper

It would be useful to wrap any point estimator with held-out residual draws:

```python
model = EmpiricalResidualRegressor(
    estimator=some_booster,
)
```

It could implement the same `predict_distribution`, `cdf`, `interval`, and `sample` contract as native DarkoFit models. That would make fallbacks much easier to integrate.

The first package version should use one pooled out-of-fold residual distribution. Grouping and shrinkage remain a follow-up requiring evidence.

## Small, practical additions

- A `validate_distribution_api(X_probe)` check covering shapes, finite values, interval ordering, reproducible sampling, and save/load equality.
- PIT, coverage, threshold-Brier, and interval calibration diagnostics.
- CRPS support beyond Gaussian, especially Student-t and the proposed Binomial heads.
- Head-specific permutation importance as the interim explainability tool; distributional SHAP is deferred.

## Implementation proposals

### Existing foundation to reuse

These additions fit the current architecture without replacing the boosting engine:

| Concern | Current extension point | Proposed use |
| --- | --- | --- |
| Distribution heads | [`darkofit/losses.py`](https://github.com/kmedved/darkofit/blob/main/darkofit/losses.py) and `VECTOR_LOSSES` | Extend the existing duck-typed loss protocol with capabilities, CDFs, survival functions, and quantiles. |
| Distributional training | [`darkofit/booster.py`](https://github.com/kmedved/darkofit/blob/main/darkofit/booster.py) and `DistributionalBoosting` | Add one general per-row likelihood-context and raw-offset path. |
| Public API and calibration | [`darkofit/sklearn_api.py`](https://github.com/kmedved/darkofit/blob/main/darkofit/sklearn_api.py) and `DarkoRegressor` | Expose post-fit calibration, distribution objects, and explicit auxiliary fit inputs. |
| Safe persistence | [`darkofit/serialization.py`](https://github.com/kmedved/darkofit/blob/main/darkofit/serialization.py) | Preserve the plain-array, `allow_pickle=False` archive contract. |
| Existing offset design | [`LINEAR_RESIDUAL_BOOSTING_PLAN.md`](LINEAR_RESIDUAL_BOOSTING_PLAN.md) | Generalize the documented v2 raw-score offset design to caller-supplied base predictions, exposure, and offsets. |
| Distribution tests | [`tests/test_distributional.py`](https://github.com/kmedved/darkofit/blob/main/tests/test_distributional.py) | Extend the existing math, calibration, validation, and save/load coverage. |

The main design rule should be that every public distribution operation goes through one calibrated `PredictiveDistribution` object. That gives native heads, empirical residuals, mixtures, and copulas one contract instead of duplicating behavior across estimator methods.

### Proposal 1 — Post-fit calibration

#### Public API

```python
model.fit_calibrator(
    X_cal,
    y_cal,
    method=None,
    sample_weight=weights,
    calibration_feature=None,
)
```

The method should mutate and return the fitted estimator, following `fit()` conventions, but must not refit trees. `method=None` should resolve from the constructor's `dist_calibration`; if neither specifies a method, raise and ask the caller to choose one. This avoids two calibration defaults.

`base_prediction` and `offset` should be added to this method only when the raw-margin engine in Proposal 3 lands.

This also formalizes existing behavior: `DarkoStepwiseSearchCV._refit_best` already performs post-fit calibration by assigning private fitted attributes directly.

#### Implementation

- Extract the existing fit-time calibration block in `DarkoRegressor.fit` into a shared `_fit_dist_calibrator_on_data(...)` helper. Both `fit()` and `fit_calibrator()` should call it.
- Add one validated `_install_dist_calibration_state(...)` helper used by `fit_calibrator()` and pooled tuning calibration. Replace the direct fitted-attribute assignments in `DarkoStepwiseSearchCV._refit_best` with that shared path.
- Reuse the current scalar, affine, per-group affine, mean, and dispersion calibration helpers and fitted-state fields.
- Continue accepting `method="scale"` as an alias for the existing `"scalar"` mode.
- Compute every new calibration from the uncalibrated distribution, including any active linear residual and, after Proposal 3, the required external margin. Repeated calls must replace the old calibration rather than compound it.
- Build the new calibration state locally and install it only after all validation and fitting succeeds. A failed recalibration should leave the previous calibrator intact.
- Record provenance such as `source="post_fit"`, row count, positive-weight count, effective sample size, method, and fitted timestamp-free diagnostics in `auto_params_`.
- Persist through the existing wrapper state. Calibrating a loaded model, saving it again, and reloading it should work without an archive break.

#### Conformal scope

Ordinary split-conformal interval correction does not define a coherent CDF, variance, or sampling distribution. It should not be advertised as a full-distribution calibrator.

A later full-distribution option can use a monotone PIT/quantile warp:

```text
F_cal(y) = G(F_base(y))
Q_cal(q) = Q_base(G^-1(q))
```

The fitted empirical map `G` can be stored as weighted PIT knots. Intervals and sampling then derive from calibrated quantiles. This should initially be limited to continuous heads. If simple interval-only conformal correction is added sooner, its narrower capability must be explicit.

A PIT-warped distribution cannot be represented faithfully by the legacy parameter tuple. Under that calibrator, `predict_dist()` should raise and direct callers to `predict_distribution()` rather than return parameters that appear fully calibrated but are not.

#### Acceptance criteria

- Raw predictions, trees, and tree count are unchanged.
- All public distribution methods apply calibration exactly once.
- Zero-weight extreme observations do not affect the fitted calibrator.
- Invalid method/head combinations fail before state mutation.
- A second call replaces rather than compounds the first calibrator.
- Save/load preserves identical parameters, intervals, CDFs, variances, and seeded samples.

### Proposal 2 — CDF, survival, quantile, and distribution-object API

#### Public API

Add `darkofit/distributions.py` with an immutable `PredictiveDistribution` implementation:

```python
dist = model.predict_distribution(X)

dist.parameters()
dist.mean()
dist.variance()
dist.cdf(value)
dist.survival(value)
dist.probability_above(threshold)
dist.quantile(q)
dist.interval(alpha=0.1)
dist.sample(n_samples=1_000, random_state=42)
```

Estimator conveniences should delegate to the same object:

```python
model.predict_cdf(X, value=threshold)
model.predict_survival(X, value=threshold)
model.predict_probability_above(X, threshold)
model.predict_quantile(X, q=[0.10, 0.50, 0.90])
```

Keep `predict_dist()` as the backward-compatible tuple API.

#### Implementation

- Add `parameter_names`, `cdf_from_params`, `survival_from_params`, and `quantile_from_params` to every distribution head in `darkofit/losses.py`.
- Use a direct survival-function implementation instead of `1 - cdf` so extreme upper-tail probabilities remain accurate.
- Construct the object from already calibrated and location/offset-adjusted parameter arrays. The object should copy those arrays so it remains a snapshot if estimator state changes later.
- Define `cdf(value)` as `P(Y <= value)`.
- Define `probability_above(threshold)` explicitly as inclusive `P(Y >= threshold)`. For discrete heads this is `sf(ceil(threshold) - 1)`; continuous heads can use `sf(threshold)`.
- Declare SciPy directly in `pyproject.toml` in this milestone. DarkoFit already imports `scipy.special.gammaln` in `darkofit/tuning/scoring.py`, so the dependency should not remain accidentally transitive through scikit-learn.
- Land `validate_distribution_api(...)` with the object scaffold and use it as the gate for each head implementation. Deliberately broken fake distributions should prove that every contract failure is detected.

#### Shape contract

- Scalar threshold or quantile: `(n_rows,)`.
- One threshold per row: `(n_rows,)`.
- Quantile vector of length `m`: `(n_rows, m)`.
- Samples: `(n_rows, n_samples)`.
- Joint samples, when added later: `(n_rows, n_samples, n_targets)`.

#### Acceptance criteria

- Analytic values match trusted distribution implementations.
- Continuous `cdf(quantile(q))` agrees with `q` within numerical tolerance.
- Discrete quantiles satisfy the left-inverse inequalities.
- Intervals equal the corresponding lower and upper quantiles.
- Survival probabilities remain finite and accurate in extreme tails.
- Calibration, linear residuals, and future offsets affect every operation consistently.
- Same-seed samples and save/load outputs are identical.
- Every native head passes the public contract validator.

### Proposal 3 — External base predictions and offsets

This proposal and Proposal 7 should land as one booster change. External margins, count exposure, and trials transport all need the same validated per-row likelihood context; implementing them in separate passes would touch the same high-risk prediction rebuild paths twice.

#### Public API

Keep response-scale base predictions distinct from canonical link-scale offsets:

```python
model.fit(
    X,
    y,
    base_prediction=train_mean,       # Gaussian/Student-t identity location
    eval_base_prediction=valid_mean,
)

model.predict_distribution(
    X_future,
    base_prediction=future_mean,
)
```

```python
model.fit(
    X,
    counts,
    offset=train_log_rate,            # LogNormal/Poisson/NB raw link
    eval_offset=valid_log_rate,
)
```

`base_prediction` and `offset` should be mutually exclusive. Do not silently interpret the same array on different scales for different heads.

#### Implementation

- Generalize the raw-score offset design already specified in `LINEAR_RESIDUAL_BOOSTING_PLAN.md`.
- Extend `DistributionalBoosting.fit` with validated `base_margin` and `eval_base_margin` arrays. Losses consume `F_tree + base_margin`, while tree updates modify only `F_tree`.
- Thread total raw scores through initialization, gradient/Hessian evaluation, validation scoring, full validation-history prefix rescoring, Negative Binomial state refresh, the truncate-refresh fixed-point loop, flat-cache prediction, and ordinary prediction. Every reconstruction from `init_ + trees` must include the margin exactly once.
- Make initialization offset-aware:

  - Gaussian/Student-t: initialize on `y - location_offset`.
  - LogNormal: initialize on `log(y) - log_location_offset`.
  - Poisson/NB: initialize the rate intercept from weighted counts divided by weighted `exp(offset)`.

- Convert continuous offsets through the fitted target-standardization transform before adding them internally.
- Use residual/rate targets for categorical target statistics rather than leaking the unadjusted target into preprocessing.
- Initially reject simultaneous `linear_residual=True` and external base predictions so there is only one additive-location owner.

#### Persistence and ecosystem behavior

- Persist the required protocol—kind, scale, raw-head index, and whether future values are mandatory—but never persist Glum predictions or another external estimator.
- A loaded model fitted with an external margin must fail clearly if prediction, calibration, or evaluation omits the required future margin.
- Use a conditional archive-format bump for every model whose prediction contract requires a side array, regardless of head. An old loader that ignored a Gaussian/Student-t base prediction would silently emit residuals, which is as incorrect as ignoring count exposure.
- `DarkoSearchCV` can support these arrays by extending its existing per-fold fit-payload slicing. This is a mechanical follow-up and does not require sklearn metadata routing, but it may be deferred from the core margin PR.

#### Acceptance criteria

- Gaussian/Student-t results match the current manually residualized workflow.
- Identity base predictions shift location only, not variance.
- Adding `delta` to a log offset multiplies positive/count means by `exp(delta)`.
- Automatic validation and refit split auxiliary arrays with the same row indices as `X` and `y`.
- Calibration, CDFs, intervals, and samples include the margin exactly once.
- Missing, extra, non-finite, wrong-shaped, or conflicting inputs fail clearly.
- Save/load preserves the required-margin contract.
- With no likelihood context or margin supplied, fitted trees, raw predictions, public predictions, and frozen benchmark hashes are bit-identical to the current path.

### Proposal 4 — Thin loss-capabilities registry

Land this after the distribution surface, calibration methods, and likelihood-context inputs stabilize. Building it first would require a handwritten table for methods and attributes that do not exist yet—the drift risk this proposal is meant to avoid.

#### Public API

Add `darkofit/capabilities.py`:

```python
from darkofit import get_loss_capabilities, list_loss_capabilities

caps = get_loss_capabilities("StudentT")
```

Suggested result:

```python
{
    "schema_version": 1,
    "name": "StudentT",
    "family": "distributional_regression",
    "target_domain": "real",
    "supports_signed_targets": True,
    "parameter_names": ["location", "scale", "degrees_of_freedom"],
    "distribution_operations": [
        "mean", "variance", "cdf", "survival", "quantile", "interval", "sample"
    ],
    "auxiliary_fit_inputs": ["sample_weight", "base_prediction"],
    "calibration_methods": ["scale", "affine", "per_group_affine"],
    "eval_metrics": ["nll"],
    "tree_modes": ["lightgbm"],
    "residual_modes": ["linear_residual"],
}
```

#### Implementation

- Use a frozen `LossCapabilities` value object with a JSON-safe `to_dict()` method.
- Derive statistical fields from loss-class attributes. Do not create an unrelated handwritten table that can drift away from implementations.
- Keep v1 read-only. Do not refactor working calibration, metric, tree-mode, or residual guards through the registry merely to centralize them.
- Cover scalar losses as well as `VECTOR_LOSSES`, even if distribution-only fields are empty.
- Export the public functions from `darkofit/__init__.py`.

#### Acceptance criteria

- Registry keys exactly cover public losses.
- Every advertised operation has an implementation and a smoke test.
- Results are immutable, JSON serializable, and safe for callers to cache.
- Unknown loss names raise with the valid canonical names.
- No model-format change is needed for the registry itself.

### Shared prerequisite for proposals 3 and 5–7 — Likelihood context

Base margins, trials, exposure, and offsets should share one internal transport without becoming generic weights:

```python
def fit(
    self,
    X,
    y,
    *,
    trials=None,
    exposure=None,
    offset=None,
    base_prediction=None,
    sample_weight=None,
    eval_trials=None,
    eval_exposure=None,
    eval_offset=None,
    eval_base_prediction=None,
):
    ...
```

Add a validated internal `LikelihoodContext` and extend distribution-loss methods with an optional `context` argument. Existing calls with no context must retain their exact path and predictions.

Preprocessing also needs two distinct weight channels:

- Numeric binning weight: ordinary observation reliability.
- Categorical target-stat weight:

  - Binomial/Beta-Binomial: target `successes / trials`, mass proportional to `trials`.
  - Poisson/NB with exposure: target `counts / exposure`, mass proportional to `exposure`.

This prevents trials or minutes from being incorrectly treated as observation reliability while still giving rate target statistics the right information mass.

### Proposal 5 — Aggregate Binomial loss

#### Public contract

Keep the canonical sklearn `(X, y)` shape and one spelling:

```python
model.fit(X, makes, trials=attempts)

dist = model.predict_distribution(X_future, trials=future_attempts)
```

#### Loss implementation

Add `BinomialNLL` to `VECTOR_LOSSES` with one logit output. For successes `k`, trials `n`, and `p = sigmoid(eta)`:

```text
NLL = n * softplus(eta) - k * eta - log C(n, k)
grad = n * p - k
hess = n * p * (1 - p)
```

Initialize with the weighted pooled rate:

```text
p0 = sum(w * k) / sum(w * n)
```

The combinatorial term belongs in evaluation so reported NLL is the actual aggregate likelihood.

#### Prediction semantics

- `predict()` returns the latent expected rate `p`.
- `predict_dist()` returns `(p,)`.
- `predict_variance(X, trials=future_trials)` returns proportion variance `p(1-p)/n`.
- Intervals, CDFs, quantiles, and samples with future trials operate on the observed proportion `K/n`.
- Keep expected counts out of ordinary `predict()` so its units do not change based on an optional prediction argument. A later `predict_successes()` can return `n * p`.

#### Validation and acceptance

- Require finite integer successes and trials, `trials >= 1`, and `0 <= successes <= trials` in v1.
- `trials=1` must match binary Logloss gradients, Hessians, and likelihood.
- Aggregate gradients and Hessians must match explicitly expanded Bernoulli rows; with identical binning and row-count gates, the resulting trees must match.
- Analytic variance and Monte Carlo samples must agree.
- Scoring, tuning, calibration, and safe persistence must support the new head.

### Proposal 6 — Beta-Binomial loss

#### Staged implementation

Ship a global-concentration head first, following the existing global-dispersion Negative Binomial pattern:

```text
p = sigmoid(eta)
alpha = p * concentration
beta = (1 - p) * concentration
```

- Fit the mean as a one-output tree head.
- Accept fixed concentration through `dist_params`.
- Otherwise profile the exact weighted Beta-Binomial NLL over bounded log concentration, refresh periodically, and refresh once after final tree truncation.
- Persist learned concentration and its source through the existing `loss_state`.

#### Numerical work

- `math.lgamma` is sufficient for the likelihood and concentration profile.
- The mean gradient requires digamma. Add a tested Numba-compatible recurrence/asymptotic implementation rather than calling SciPy inside every boosting row.
- Use a positive Fisher/quasi-Hessian. Do not pass potentially negative observed curvature into tree building.
- Validate the digamma approximation and likelihood against SciPy and use finite differences to validate gradients.

The predictive proportion variance is:

```text
Var(K / n) = p(1 - p) * (concentration + n)
             / (n * (concentration + 1))
```

Sampling can draw a latent probability from Beta and then successes from Binomial. Intervals and quantiles can use `scipy.stats.betabinom`.

Prediction should use the same future-trials contract as the Binomial head.

#### Follow-up

A two-head version may model concentration as a function of features. It needs separately validated trigamma/Fisher behavior and should not be bundled into the global-concentration release.

#### Acceptance criteria

- Large concentration converges to Binomial behavior.
- Simulated overdispersion is recovered and improves held-out NLL over Binomial.
- Analytic and Monte Carlo variances agree.
- State refresh, early-stopping rescoring, truncation, and save/load remain deterministic.

### Proposal 7 — Exposure and count offsets

This is the count-specific public surface of Proposal 3's raw-margin engine, not a second booster project.

#### Public API

```python
model.fit(X, counts, exposure=minutes)
model.fit(X, counts, offset=np.log(minutes))

dist = model.predict_distribution(X_future, exposure=future_minutes)
rate = model.predict_rate(X_future)
```

`exposure` and `offset` are mutually exclusive. Exposure must be finite and strictly positive in v1.

#### Implementation

For Poisson and Negative Binomial:

```text
eta_total = eta_tree + log(exposure)
mu = exp(eta_total)
```

- Reuse the same raw-offset engine as proposal 3.
- Loss gradients, Hessians, evaluation, and NB dispersion refresh consume total raw scores.
- Tree leaf updates affect only the learned log-rate surface.
- Mean calibration composes after exposure; NB dispersion calibration remains separate.
- If training used exposure or an offset, count-valued prediction methods must require future context. `predict_rate()` is the explicit unit-exposure path.

#### Acceptance criteria

- `exposure=e` is exactly equivalent to `offset=log(e)`.
- Unit exposure reproduces legacy Poisson/NB predictions.
- Poisson means scale linearly with exposure.
- NB variance remains `mu + alpha * mu**2`.
- Validation, scoring, tuning, calibration, and sampling all use future exposure consistently.
- Persistence retains the required-context protocol but not row-level exposure values.

### Proposal 8 — Explicit auxiliary-data semantics

#### Committed contracts

- `sample_weight`: nonnegative relative reliability or modeling importance; retains the current global scale-invariance behavior.
- `exposure`: opportunity amount entering a count model's log mean.
- `trials`: denominator entering Binomial-family likelihood and predictive variance.
- `offset`: a caller-supplied canonical raw/link margin.

These inputs should remain separate in the public signature, likelihood context, diagnostics, and capabilities. Exposure and trials must never be multiplied into `sample_weight` or used as `min_child_samples` row counts.

#### Deferred contract

Do not add `frequency_weight` now. There is no named consumer in the current NBA workflow, and implementing true replication semantics would require coordinated changes to histogram counts, `min_child_samples`, effective-sample-size rules, automatic parameter selection, and subsampling. Document the distinction and reopen it only with a concrete dataset and acceptance protocol.

#### Acceptance criteria

- Scaling `sample_weight` globally does not change a fit.
- Exposure and trials affect their likelihoods without changing observation-reliability semantics.
- Automatic validation, refit, calibration, and tuning slice every auxiliary array with the same indices as the observations.
- `auto_params_` records which auxiliary inputs were supplied and summarizes their valid ranges without persisting row-level values.

### Proposal 9 — Joint correlated outputs

#### Scheduling gate

Decide the consumer data shape before scheduling library work. The proposed copula correlates target columns within a row. Correlating years 1–5 therefore requires a wide representation with one marginal per horizon, while the current long representation treats horizon as a feature and player-horizons as separate rows. That upstream modeling choice is a prerequisite, not an implementation detail of the copula.

#### Public API

Use a meta-estimator rather than changing the existing one-dimensional `DarkoRegressor` target contract:

```python
joint = JointDistributionRegressor(
    marginals={
        "offense_y1": DarkoRegressor(loss="StudentT"),
        "defense_y1": DarkoRegressor(loss="StudentT"),
    },
    copula="gaussian",
    cv=5,
    correlation_shrinkage="auto",
    random_state=42,
)

joint.fit(X, Y, groups=player_id, sample_weight=weights)
draws = joint.sample_joint(X_future, n_samples=1_000, random_state=42)
```

#### MVP

- Clone and fit one marginal estimator per target.
- Estimate dependence from group-safe out-of-fold PIT values or a dedicated copula-calibration set. Do not estimate it from in-sample residuals.
- Clamp PIT values away from zero and one, transform them to latent normal scores, estimate a weighted correlation matrix, shrink it toward identity, and repair it to positive semidefinite form.
- Restrict the first release to continuous marginals with exact CDFs and quantiles.
- Return a `JointPredictiveDistribution`; marginal access should reproduce the standalone marginal distributions exactly.
- Define scope clearly: dependence is across target columns within each row. Years 1–5 must be represented as output columns if they should be correlated. This does not correlate separate rows for the same player.

#### Persistence and acceptance

- Add a safe multi-model bundle with a versioned manifest, one existing DarkoFit archive per marginal, and plain arrays for names and correlation.
- Validate member count, names, shapes, correlation symmetry, unit diagonal, and positive-semidefinite status on load.
- Recover known synthetic correlations and reproduce sampled correlations within a preregistered tolerance.
- Preserve marginal CRPS while improving joint correlation or energy-score behavior over independent sampling.
- Save/load must preserve same-seed joint samples.

### Proposal 10 — Clustered bootstrap ensembles stay outside the package

#### Evidence boundary

The existing row-bootstrap OOB-5 mechanism is not an implementation candidate. Its stable confirmation explicitly closed that path because the frozen prediction-timing gate failed; see [`benchmarks/basketball_oob_ensemble_confirmation_result.md`](https://github.com/kmedved/darkofit/blob/main/benchmarks/basketball_oob_ensemble_confirmation_result.md).

Any clustered or season-block ensemble must be treated as a materially new mechanism with a new preregistered protocol, not as a rescue or rerun of OOB-5.

#### Research path

- Keep all clustered or season-block orchestration in `benchmarks/`, built from seeded calls to the existing estimator and `save_model()` APIs.
- Freeze a materially new protocol before running it. Support one resampling axis at a time; a two-way player-within-season bootstrap requires its own design.
- Materialize repeated cluster rows rather than introducing `frequency_weight` for this experiment.
- Use exact group-level OOB complements for validation and verify that every unit is wholly in-bag or OOB.
- Record deterministic member seeds, bootstrap hashes, fitted metadata, member outputs, and uncertainty decompositions.
- Compute mixture mean, member-averaged CDF, and aleatoric/epistemic variance in benchmark utilities.

Do not add `ClusteredBootstrapRegressor`, an ensemble constructor parameter, or ensemble bundle persistence to `darkofit/` under this plan. A general `MixtureDistribution` may join the Proposal 2 object family later if an independent package consumer needs it.

Passing a new benchmark would justify a fresh package proposal; it would not automatically promote an estimator API.

### Proposal 11 — Empirical residual wrapper

#### Public API

```python
model = EmpiricalResidualRegressor(
    estimator=some_point_estimator,
    cv=5,
    bias_correction=False,
    random_state=42,
)

model.fit(X, y, groups=player_id, sample_weight=weights)
dist = model.predict_distribution(X_future)
```

#### MVP

- Collect strictly out-of-fold residuals using cloned estimators, then refit one final point estimator on all training data. Alternatively accept a dedicated residual-calibration set.
- Never use in-sample residuals as the empirical uncertainty distribution.
- Store one sorted, weighted, pooled residual distribution.
- Define the empirical CDF as `P(residual <= r)` and quantile as its left inverse.
- Keep `predict()` identical to the wrapped estimator by default. With `bias_correction=False`, center the empirical residual support to weighted mean zero. An explicit opt-in may retain the observed residual mean and shift both the distribution mean and `predict()`.
- Limit v1 to point estimators to avoid double-counting uncertainty from already distributional models.

Grouping, effective-sample-size pooling, and shrinkage constants remain a follow-up after a real dataset demonstrates that pooled residuals are insufficient.

#### Persistence

Arbitrary sklearn estimators cannot be embedded safely in DarkoFit's non-pickle archive. Add a serializer-adapter registry, initially supporting `DarkoRegressor`. `save_model()` should raise for unsupported estimators rather than silently embedding pickle.

#### Acceptance criteria

- An overfitting estimator still produces non-degenerate OOF intervals.
- Weighted empirical CDFs, quantiles, variances, and seeded samples match hand calculations.
- Default `predict()` is exactly the wrapped estimator's prediction.
- Bias correction changes predictions only when explicitly enabled.
- DarkoFit-backed wrappers round-trip exactly; unsupported serializers fail clearly.
- A seeded skewed-residual gate shows calibrated coverage and better CRPS/Brier than a constant-Gaussian residual fallback.

### Proposals for the small, practical additions

#### Distribution contract validator

This lands in the first milestone with `PredictiveDistribution`, before later heads and wrappers rely on the contract.

Add:

```python
from darkofit.validation import validate_distribution_api

report = validate_distribution_api(
    model,
    X_probe,
    check_serialization=True,
    random_state=42,
)
```

The validator should return a structured report and optionally raise one aggregated `DistributionContractError`. It should check:

- shapes and finite values;
- nonnegative variance;
- interval ordering and nesting;
- CDF bounds and threshold monotonicity;
- quantile monotonicity;
- same-seed sample equality;
- continuous versus discrete `>`/`>=` probability semantics;
- deterministic and seeded save/load equality.

Add deliberately broken fake distributions to prove that each validation failure is detected.

#### Public diagnostics

Add a `distribution_diagnostics(...)` function that consumes a `PredictiveDistribution` and reports:

- Continuous PIT or seeded randomized PIT for discrete heads.
- PIT histogram and uniformity statistic.
- Coverage, width, and coverage error for requested interval levels.
- Threshold Brier scores with explicit event semantics.
- Optional group and predicted-scale slices.

Move reusable metric logic out of benchmark scripts rather than maintaining separate formulas in product code, tuning, and benchmarks.

#### CRPS

- Expose row-level `crps()` on distribution objects so post-fit calibration is respected.
- Keep the existing exact Gaussian formula.
- Add Student-t CRPS only after validating the closed form against numerical integration or Monte Carlo.
- Use finite-support CDF sums for Binomial and Beta-Binomial.
- Use an `O(m log m)` ensemble CRPS for empirical and mixture distributions.
- Advertise `"crps"` as an evaluation metric only after the corresponding public implementation and oracle tests exist.

#### Distributional explainability — deferred

Do not schedule distributional SHAP in this roadmap. The existing exact path is scalar-oblivious-only, while distributional heads require vector-valued non-oblivious trees. Closing that gap is a substantial kernel project without a named product decision it would change.

Use head-specific permutation importance as the interim diagnostic. Reopen raw-parameter SHAP only when a concrete consumer requires additive location/scale attributions and can fund a separate implementation and benchmark plan.

## Recommended delivery sequence

1. **One distribution surface and its test harness:** `PredictiveDistribution`, CDF/SF/quantile primitives, parameter names, the contract validator, and an explicit SciPy dependency.
2. **Calibration lifecycle:** shared calibration helper and public `fit_calibrator()`; port `DarkoStepwiseSearchCV` refit calibration onto it. Add PIT/quantile warping only after the parametric path is stable.
3. **One likelihood-context and raw-margin surgery:** external base margins, offsets, exposure, and trials transport in the same booster change. Every total-score rebuild must include context exactly once, and the no-context path must pass bit-identical frozen gates.
4. **Rate heads:** aggregate Binomial, followed by global-concentration Beta-Binomial.
5. **Thin discovery and scoring layer:** read-only capabilities registry, public diagnostics, and generalized CRPS after the operations they advertise exist.
6. **Slim empirical wrapper:** one pooled OOF residual distribution, no default bias shift, and safe persistence only for registered estimator serializers.
7. **Copula, conditionally:** schedule Gaussian-copula joint outputs only after the consumer explicitly chooses a wide target layout for correlated horizons.

Dropped from the committed package roadmap: `frequency_weight`, in-package clustered/bootstrap estimators, grouped empirical-residual shrinkage, and distributional SHAP. Each can return with a concrete consumer, evidence, and its own acceptance protocol.

The first two milestones immediately simplify the current POC. Milestone 3 is the key architectural step that makes arbitrary rate metrics statistically correct, but it also touches protected JIT and prediction paths. “No-context behavior is unchanged” is therefore a frozen compatibility gate: tests and benchmark hashes must prove it before the milestone can land.
