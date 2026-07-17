# Basketball Gaussian scalar-calibration opportunity screen

## Question and scope

Does DarkoFit's existing opt-in Gaussian scalar distributional calibration
improve probabilistic accuracy and 80% interval calibration on noisy
basketball data without weakening held teams, genuinely cold players,
point-prediction behavior, runtime, or transient memory?

This is an external validation of an already-shipped explicit feature. It
changes no estimator API, model default, serialization format, fit path, or
prediction path. Basketball is the primary and fatal development boundary
because it is fast and directly represents the project's sports-data
priority. Failure stops this calibration mode before a broader distributional
panel or stronger documentation claim.

The screen isolates exactly one lever for each fitted Gaussian model:

- control parameters: the unchanged raw-model `(mu, sigma)`;
- candidate parameters: `(mu, sigma * s)`, where the existing product code
  fits one positive scalar `s` on an internal calibration split;
- control and candidate central interval: the Gaussian 80% interval from the
  corresponding parameter pair.

Both arms share the same fitted model, raw scores, mean, trees, preprocessing,
learning rate, tree mode, iteration horizon, random seed, and training rows.
Only the positive scale multiplier differs. The opportunity-matrix score is
`4 * 5 / 2 = 10`: high product value for DarkoFit's uncertainty moat, very
high mechanistic confidence because the feature already exists, and modest
screening effort. That exceeds the required score of 2.0.

## Frozen source, data, and task

- Pre-protocol DarkoFit source:
  `ba4b7f98004716a62e65d8bbb29a7074d3655313`.
- Frozen `darkofit/` package Git tree:
  `1a60b529c5f5d09920d81338406b491fb7275e3a`.
- Basketball CSV SHA-256:
  `43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2`.
- Creator training feature fingerprint:
  `05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b`.
- The target is the unchanged creator `MPG = MP / G` after the unchanged
  `MP > 500` filter and alphabetical team holdout.
- Target SHA-256 values, serialized as contiguous little-endian `float64`,
  are:
  - creator training:
    `7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf`;
  - overlap-exposed held teams:
    `c051a5ae966077792a2c28757ec4d06dc1660aef2cb4064cab10acac2216d1bf`;
  - cold-player subset:
    `cd16264232c0966c5823709c392b32638912f14596ff6ea4d5e8a6a2b5dd30e8`;
  - seen-player subset:
    `bca52624dbd022f53365fe319f1851d350253f2f9bd03a5360db56c7dad45d8b`.
- The existing 15 numeric features, unshuffled ten-fold creator split,
  alphabetical team holdout, and player masks remain unchanged. The creator
  split fingerprint is
  `7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea`.
- Creator training has 5,241 rows. The overlap-exposed held-team view has
  2,409 rows. Its cold-player subset has 585 rows whose exact source player
  identifier is absent from training; its mask SHA-256 is
  `e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19`.
- The frozen runtime is Python `3.12.13` on
  `macOS-26.5.2-arm64-arm-64bit`, Apple M5 Max, with 18 logical CPUs. Exact
  package versions are NumPy `2.4.6`, Numba `0.66.0`, llvmlite `0.48.0`,
  pandas `3.0.3`, scikit-learn `1.9.0`, SciPy `1.18.0`, joblib `1.5.3`, and
  threadpoolctl `3.6.0`.
- No TabArena, CTR23 development coordinate, or lockbox data is used.

The formal run must use clean committed `main`, equal to `origin/main`,
descended from the pre-protocol commit, with the exact frozen `darkofit/`
package tree and runtime stack above, 18 threads per fit, and random state 4.
The runner must hard-code and verify the final byte SHA-256 of this protocol,
its own normalized source hash, the package-tree hash, its executable support
file hashes, every frozen dependency version, platform, architecture, CPU
brand, and logical CPU count. A protocol, package, runtime, source, threshold,
or output-path change fails closed. Runner and runner-test commits outside the
package are permitted. Source state, dependency versions, data hashes, split
indices, prediction hashes, fitted metadata, calibration diagnostics, and
timings are recorded.

## Frozen fitting and calibration rule

For each external creator fold, split that fold's training rows with:

```text
ShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=4,
)
```

Apply the same rule to the complete creator training view before scoring the
held-team, seen-player, and cold-player views. The internal fit and
calibration indices must be disjoint, cover the external training indices
exactly, and be recorded and hashed.

Fit exactly one model on only the internal fit rows:

```text
DarkoRegressor(
    loss="Gaussian",
    tree_mode="lightgbm",
    dist_calibration="scalar",
    use_best_model=False,
    early_stopping=False,
    random_state=4,
    thread_count=18,
    diagnostic_warnings="never",
)
```

Pass the internal calibration rows as the explicit `eval_set`. Disabling
early stopping and best-model selection is mandatory: calibration rows fit
only the existing scalar scale and cannot choose a tree prefix, change
preprocessing, or affect tree construction. Do not refit.

Let the uncalibrated Gaussian parameters on the calibration rows be
`(mu_cal, sigma_cal) = fitted_core.params_from_raw(raw_cal)`. This
target-transform-aware conversion is mandatory. The existing product rule is:

```text
z = clip((y_cal - mu_cal) / max(sigma_cal, 1e-12), -1000, 1000)
s = sqrt(max(mean(z**2), 1e-12))
candidate_sigma = control_sigma * s
```

The runner must independently recompute this rule and require array-exact
agreement with the fitted `dist_scale_`. The scale must be finite, strictly
positive, and strictly inside `(1e-6, 1e6)`. Public `predict_dist`,
`predict_variance`, and `predict_interval(alpha=0.2)` must agree with an
independent reconstruction from the unchanged raw scores and fitted scale.
Control parameters and intervals are reconstructed from those same raw scores
with scale one. No weighting, affine map, group map, quantile offset,
early stopping, refit, cross-fold pooling, or fallback is allowed.

## Frozen metrics

For each evaluation view and arm, record:

- Gaussian negative log likelihood:
  `mean(log(sigma) + 0.5*((y-mu)/sigma)**2 + 0.5*log(2*pi))`;
- Gaussian CRPS using the exact closed form;
- central 80% interval coverage, absolute error from `0.8`, mean width, and
  crossing count;
- mean predicted scale and its SHA-256;
- point RMSE from `mu`;
- mean, lower-bound, upper-bound, variance, and raw-score SHA-256 values.

Candidate and control means, point RMSE, raw scores, and row order must be
array-exact within every fold and held-team-model view. Candidate lower and
upper bounds must equal the independent `(mu, sigma*s)` reconstruction
array-exactly. All scale values and widths must be finite and strictly
positive; crossings must be zero.

Creator-fold aggregate metrics are computed by pooling the ten external test
predictions, so every creator row appears exactly once. Fold metrics remain
separate. Because the ten folds have independently fitted scale multipliers,
the pooled candidate is a cross-fitted diagnostic and has no single global
scale. Held-team, seen-player, and cold-player metrics remain separate.

## Frozen runtime and memory check

Runtime uses the held-team model and all 2,409 held-team rows after one untimed
call per arm. Make a shallow control copy of the fitted estimator, set only the
copy's fitted `dist_calibration_` attribute to `None`, and assert that the two
estimators share the identical fitted `model_` object. Do not refit, deep-copy,
or mutate the candidate. Each timed call uses the shipped public
`predict_dist(X)` path:

```text
control = uncalibrated_control_copy.predict_dist(X)
candidate = scalar_calibrated_candidate.predict_dist(X)
```

This includes input validation, raw prediction, target-scale parameter
conversion, calibration dispatch, and the scalar multiplication exactly as
users invoke them. Before timing, require the control output to equal the
independent uncalibrated reconstruction array-exactly and the candidate output
to equal the independent `(mu, sigma*s)` reconstruction array-exactly.

Run 50 calls per arm in each of three reciprocal blocks:

```text
[(control, candidate), (candidate, control), (control, candidate)]
```

Every repeated output must be bit-identical within its arm. Record each block
time, per-call time, and the median per-call ratio. Each arm's maximum/minimum
block-time ratio must be at most `1.20`; the candidate/control median per-call
ratio must be at most `1.10`.

Separately measure five untimed-after-warmup public `predict_dist(X)` calls per
arm with `tracemalloc`. The candidate's maximum traced peak may not exceed the
control maximum by more than 256 KiB. This is an opportunity-screen bound, not
permission to skip a product-level profile if later changes are proposed.

## Fatal gates

The candidate advances only if all conditions hold:

1. every target, raw score, parameter, interval bound, scale, metric, timing,
   memory value, and fitted-model numeric value is finite and valid;
2. all fitted-scale checks and independent product-path reconstruction checks
   pass;
3. candidate and control means, point RMSE, and raw scores are array-exact
   within every individual creator fold and every held-team-model view;
4. pooled creator NLL is strictly lower, while pooled CRPS and absolute 80%
   coverage error are no worse;
5. candidate NLL is strictly lower on at least six of ten creator folds;
6. the worst creator-fold candidate/control NLL ratio is at most `1.02`;
7. held-team candidate NLL, CRPS, and absolute coverage error are each no
   worse;
8. cold-player candidate NLL, CRPS, and absolute coverage error are each no
   worse;
9. candidate/control mean-width ratios are at most `1.25` on pooled creator,
   held-team, and cold-player views;
10. every interval crossing count is zero;
11. all runtime stability and candidate/control overhead gates pass; and
12. the transient-memory gate passes.

The seen-player subset is reported but is not an additional gate because it
already dominates the held-team row count. Comparisons use unrounded
`float64` values. Ties pass except for the explicit pooled and fold
strict-improvement gates.

## Behavior proof and stop rules

- Ordering preserved: point predictions are unchanged; interval ordering is
  preserved because `s > 0`.
- Tie-breaking unchanged: no point-decision or score tie is modified.
- Floating point: fitted raw scores and means are unchanged; calibrated
  scales, variances, and interval bounds are new, separately hashed outputs.
- RNG seeds: unchanged; the same fitted model supplies both arms.
- Rollback: no product code is changed by this screen.

If any fatal gate fails:

- record `stop_distributional_scalar_calibration_at_basketball`;
- do not change a default, expand to affine or grouped calibration, or spend a
  broader panel;
- preserve the explicit API but document that its calibration-set objective
  did not generalize across the sports boundary; and
- do not rerun with altered split fractions, seeds, interval alpha, metrics,
  thresholds, or gates.

If every gate passes, the result authorizes only broader validation of the
existing explicit scalar mode on a preregistered distributional panel. It
does not authorize an automatic or universal default. A documentation
promotion still requires fresh non-basketball evidence.

Binary temperature scaling, quantile offsets, affine/grouped distributional
calibration, categorical combinations, and default policy changes are
separate mechanisms and are not authorized by this protocol.
