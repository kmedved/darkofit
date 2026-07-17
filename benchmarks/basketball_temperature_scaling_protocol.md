# Basketball binary temperature-scaling opportunity screen

## Question and scope

Does one validation-fitted positive temperature improve DarkoFit's binary
probability calibration on noisy basketball data without weakening held teams,
genuinely cold players, discrimination, class decisions, runtime, or transient
memory?

This is a benchmark-only opportunity screen. It changes no estimator API,
model default, serialization format, fit path, or prediction path. Basketball
is the primary and fatal development boundary because it is fast and directly
represents the project's sports-data priority. Failure stops the mechanism
before implementation or broader data.

The screen isolates exactly one lever. For each fitted classifier:

- control score: the model's unchanged raw logit `z`;
- control probability: `sigmoid(z)`;
- candidate score: `z / T`, where `T` is one positive scalar fitted only on
  an internal calibration split;
- candidate probability: `sigmoid(z / T)`.

Training rows, fitted trees, learning rate, tree mode, iteration horizon,
preprocessing, random seed, and every non-calibration parameter are identical.
A positive scalar division preserves score ordering, ties, the sign of every
logit, and therefore DarkoFit's strict `p > 0.5` binary class decision.

The opportunity matrix score is `3 * 4 / 2 = 6.0`: moderate product impact,
high mechanistic confidence, and low implementation effort. That exceeds the
required score of 2.0. No speed optimization is attempted; the runtime gate
only establishes that the added probability transform is acceptably cheap.

## Frozen source, data, and task

- Pre-protocol DarkoFit source:
  `ccf6d592d9788cf302cf68559f8723864a533c26`.
- Frozen `darkofit/` package Git tree:
  `1a60b529c5f5d09920d81338406b491fb7275e3a`.
- Basketball CSV SHA-256:
  `43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2`.
- Creator training feature fingerprint:
  `05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b`.
- The binary target is the already-defined creator transform
  `starter = int(GS / G >= 0.5)` after the unchanged `MP > 500` filter.
  `G` must be strictly positive on every retained row.
- Starter-label SHA-256 values, serialized as contiguous `uint8`, are:
  - creator training:
    `5c5215635fbf8597298be6b78bf84648fb7ef9da6b16ffb7cde95af2e52b0374`;
  - overlap-exposed held teams:
    `cfd520645de52579d2fe73cd3f62ba2e9ecc1f2a437c86cd10cea0199dfe0f46`;
  - cold-player subset:
    `79090c357e3f0e7cb454600276d26df54cedeb4398b8272c252d4e4fd626d668`;
  - seen-player subset:
    `58d6553479b7625902abcf7aff6f667701362d450157670f6f270128520a0b1f`.
- Creator training has 5,241 rows and 2,647 positives. Held teams have
  2,409 rows and 1,211 positives. The cold-player subset has 585 rows and 262
  positives. Every external train and test fold contains both classes.
- The existing 15 numeric features, unshuffled ten-fold creator split,
  alphabetical team holdout, and player masks remain unchanged. The creator
  split fingerprint is
  `7270350a38a687b6e783d18d9c02e5a45f2c7367faa4a6955e74c710f5e8daea`.
- The overlap-exposed held-team view has 2,409 rows. Its cold-player subset
  has 585 rows whose exact source player identifier is absent from training;
  its mask SHA-256 is
  `e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19`.
- The frozen runtime is Python `3.12.13` on
  `macOS-26.5.2-arm64-arm-64bit`, Apple M5 Max, with 18 logical CPUs. Exact
  package versions are NumPy `2.4.6`, Numba `0.66.0`, llvmlite `0.48.0`,
  pandas `3.0.3`, scikit-learn `1.9.0`, SciPy `1.18.0`, joblib `1.5.3`, and
  threadpoolctl `3.6.0`.
- No categorical-combination claim is made: the unchanged creator feature set
  is entirely numeric, so that separate mechanism cannot activate here.
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
indices, prediction hashes, fitted metadata, temperature diagnostics, and
timings are recorded.

## Frozen fitting and calibration rule

For each external creator fold, split that fold's training rows with:

```text
StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=4,
)
```

Apply the same rule to the complete creator training view before scoring the
held-team, seen-player, and cold-player views. The internal fit and calibration
indices must be disjoint, cover the external training indices exactly, contain
both classes, and be recorded and hashed.

Fit exactly one model on only the internal fit rows:

```text
DarkoClassifier(
    random_state=4,
    thread_count=18,
    diagnostic_warnings="never",
)
```

Do not pass `eval_set`. Thus `early_stopping=False`, `tree_mode="catboost"`,
and all other source-bound product defaults remain unchanged;
`use_best_model=True` is inactive; and calibration rows cannot select a tree
prefix or affect preprocessing or tree construction.

Let `z_cal` be the fitted model's raw binary logit on the internal calibration
rows. Fit the temperature in log space with SciPy
`optimize.minimize_scalar`:

```text
objective(log_T) =
    mean(logaddexp(0, z_cal / exp(log_T))
         - y_cal * z_cal / exp(log_T))

bounds = (log(0.05), log(20.0))
method = "bounded"
options = {"xatol": 1e-12, "maxiter": 500}
T = exp(result.x)
```

The optimizer must report success; its objective must be finite and no higher
than the unchanged `T=1` objective; and `T` must be strictly inside the bounds
by at least `1e-6` in log space. No weighting, clipping, bin-specific fitting,
class-specific fitting, player-specific fitting, cross-fold pooling, refit,
or manual fallback is allowed.

## Frozen metrics

For each evaluation view and arm, record:

- stable binary log loss computed from logits with `logaddexp`;
- Brier score;
- ten-bin equal-width expected calibration error. Assign row `i` to
  `b_i = min(floor(p_i * 10), 9)`. For each nonempty bin `b`, compute
  `gap_b = abs(mean(y_i | b_i=b) - mean(p_i | b_i=b))`; then compute
  `ECE = sum_b (n_b / n) * gap_b`. Empty bins contribute zero and are not
  otherwise averaged;
- accuracy under the strict `p > 0.5` rule;
- ROC AUC;
- logit and probability SHA-256 values.

Record raw and scaled score-order SHA-256 values using stable `argsort`.
Candidate and control class predictions, score ordering, and all score ties
must be exactly identical within each creator test fold. The held-team,
seen-player, and cold-player views use one common held-team model and
temperature, so those invariants must also hold both within each view and
across their shared held-team row order.

Creator-fold aggregate metrics are computed by pooling the ten external test
predictions, so every creator row appears exactly once. Fold metrics remain
separate. Because the ten folds have independently fitted temperatures, their
pooled scaled scores need not preserve cross-fold ordering; pooled creator ROC
AUC is reported only as a cross-fitted diagnostic and is not an invariance
gate. Held-team, seen-player, and cold-player metrics remain separate.

## Frozen runtime and memory check

Runtime uses the held-team model and all 2,409 held-team rows after one untimed
call per arm. Each timed call obtains the unchanged raw logit with the fitted
core's `predict_raw(X)` path. Control applies the same stable sigmoid used for
quality scoring; candidate divides that same raw logit by the fitted
temperature before applying the same stable sigmoid:

```text
z = fitted_core.predict_raw(X)
control = sigmoid(z)
candidate = sigmoid(z / T)
```

Run 50 calls per arm in each of three reciprocal blocks:

```text
[(control, candidate), (candidate, control), (control, candidate)]
```

Every repeated output must be bit-identical within its arm. Record each block
time, per-call time, and the median per-call ratio. Each arm's maximum/minimum
block-time ratio must be at most `1.20`; the candidate/control median per-call
ratio must be at most `1.10`.

Separately measure five untimed-after-warmup calls per arm with `tracemalloc`.
The candidate's maximum traced peak may not exceed the control maximum by more
than 256 KiB. This is an opportunity-screen bound, not permission to skip a
product-level memory profile if implementation is authorized.

## Fatal gates

The candidate advances only if all conditions hold:

1. every label, logit, probability, temperature, objective, metric, timing,
   memory value, and fitted-model numeric value is finite and valid;
2. all optimizer checks and temperature-bound checks pass;
3. candidate and control class predictions, score order, and score ties are
   exactly identical within every individual creator fold and within every
   held-team-model view; no cross-fold pooled-order invariant is imposed;
4. pooled creator log loss is strictly lower, while pooled Brier score and ECE
   are no worse;
5. candidate log loss is strictly lower on at least six of ten creator folds;
6. the worst creator-fold candidate/control log-loss ratio is at most `1.02`;
7. held-team candidate log loss, Brier score, and ECE are each no worse;
8. cold-player candidate log loss, Brier score, and ECE are each no worse;
9. all runtime stability and candidate/control overhead gates pass; and
10. the transient-memory gate passes.

The seen-player subset is reported but is not an additional gate because it
already dominates the held-team row count. Comparisons use unrounded float64
values. Ties pass except for the explicit pooled and fold strict-improvement
gates.

## Behavior proof and stop rules

- Ordering preserved: yes, division by finite positive `T` is monotonic.
- Tie-breaking unchanged: yes, equal logits remain equal after division.
- Floating point: model logits are unchanged; calibrated probabilities are a
  new, separately hashed output.
- RNG seeds: unchanged; the same fitted model supplies both arms.
- Rollback: no product code exists at this stage.

If any fatal gate fails:

- record `stop_before_product_implementation`;
- do not add a calibration parameter, automatic policy, serialization field,
  or prediction branch;
- do not rerun with altered targets, split fractions, seeds, bounds, metrics,
  thresholds, bins, or gates; and
- do not spend a broader panel.

If every gate passes, the result authorizes only a separately reviewed,
default-off product implementation. That implementation must preserve
bit-identical probabilities and serialized bytes when disabled; define
explicit and automatic validation sources; support staged prediction, refit,
weights, groups, binary class labels, serialization hardening, sklearn
compatibility, and fitted metadata; and pass another frozen basketball
confirmation before broader data. A universal or automatic default still
requires fresh non-basketball evidence.

Multiclass calibration, categorical combinations, quantile correction, and
distributional-interval calibration are separate mechanisms and are not
authorized by this protocol.
