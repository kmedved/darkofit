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

## Current limits

Distributional models require `tree_mode="lightgbm"`. GOSS/MVS,
Bayesian bootstrap, ordered boosting, float32 histograms, and TreeSHAP are not
supported for these vector-output heads.

See [Benchmark notes](https://github.com/kmedved/darkofit/blob/main/BENCHMARK_NOTES.md)
for the synthetic and WNBA evidence boundaries.
