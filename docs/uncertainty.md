# Distributional regression

Distributional heads predict parameters rather than only a point estimate.
They use shared vector-valued leaf-wise trees.

```python
from darkofit import DarkoRegressor

model = DarkoRegressor(
    loss="Gaussian",
    tree_mode="lightgbm",
    early_stopping=True,
    random_state=4,
)
model.fit(X_train, y_train, eval_set=(X_validation, y_validation))

mean = model.predict(X_test)
mean, scale = model.predict_dist(X_test)
lower, upper = model.predict_interval(X_test, alpha=0.10)
variance = model.predict_variance(X_test)
draws = model.sample(X_test, n_samples=100, random_state=0)
```

## Heads

| Loss | Public parameters | Predictive mean |
|---|---|---|
| `Gaussian` | `(mu, sigma)` | `mu` |
| `LogNormal` | `(mu_log, sigma_log)` | lognormal mean |
| `StudentT` | `(mu, scale, nu)` | `mu` when defined |
| `Poisson` | `(lambda,)` | `lambda` |
| `NegativeBinomial` | `(mu, alpha)` | `mu` |

Continuous heads standardize their canonical target internally and transform
public parameters back to target units.

## Validation and calibration

NLL is the default validation metric. Gaussian also supports
`eval_metric="crps"`. Use an explicit validation set or early stopping when
interval calibration matters.

`dist_calibration` supports scalar, affine, and per-metric affine maps. A
fitted calibration applies consistently to distributions, variances,
intervals, samples, and any calibrated predictive mean. `predict_raw()` stays
on the uncalibrated fitted score surface.

Gaussian models additionally support opt-in split-conformal intervals:

```python
model = DarkoRegressor(
    loss="Gaussian",
    tree_mode="lightgbm",
    early_stopping=True,
    dist_calibration="affine",
    interval_calibration="conformal",
)
model.fit(X_train, y_train, eval_set=(X_validation, y_validation))

lower, upper = model.predict_interval(
    X_test,
    alpha=0.10,
    calibrate="conformal",
)
```

The conformal score is `abs(y - mu) / sigma`, evaluated on rows held out from
training, early stopping, model or learning-rate selection, and
`dist_calibration`. If the supplied validation set is needed for any of those
steps, half is deterministically reserved for conformal calibration. Otherwise
the entire validation set is the conformal holdout. The prediction interval
uses the finite-sample rank `ceil((n_cal + 1) * (1 - alpha))`. If that rank
would exceed the calibration sample count, prediction fails with guidance
instead of silently substituting a finite interval that lacks the requested
coverage guarantee.

This is deliberately explicit at prediction time. A model fitted with
`interval_calibration="conformal"` still returns its parametric interval unless
`calibrate="conformal"` is passed. Inspect
`model.interval_calibration_split_` and
`model.model_.auto_params_["interval_calibration"]` for provenance and sample
counts. Sample weights and `refit=True` are currently rejected because either
would require a different coverage argument. Marginal coverage is not a
conditional-coverage guarantee, and interval width must be reported alongside
coverage.

## Current limits

Distributional models require `tree_mode="lightgbm"`. GOSS/MVS,
Bayesian bootstrap, ordered boosting, float32 histograms, and TreeSHAP are not
supported for these vector-output heads.

See [Benchmark notes](https://github.com/kmedved/darkofit/blob/main/BENCHMARK_NOTES.md)
for the synthetic and WNBA evidence boundaries.
