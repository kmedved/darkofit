# Basketball split-conformal quantile opportunity screen

## Question and scope

Does a validation-residual constant shift improve DarkoFit's 10th- and
90th-percentile predictions on noisy basketball data without weakening held
teams or genuinely cold players?

This is a benchmark-only opportunity screen. It changes no estimator API,
model default, serialization format, or prediction path. Basketball is the
primary and fatal development boundary because it is fast and directly
represents the project's sports-data priority. Failure stops the mechanism
before implementation or broader data.

The screen isolates exactly one lever. For each fitted quantile model:

- control prediction: the model's unchanged raw prediction;
- candidate prediction: the same prediction plus one constant computed from
  an internal calibration split.

Learning rate, tree mode, iteration horizon, preprocessing, training rows,
random seed, and every fitted tree are therefore identical between the two
arms.

## Frozen source and data

- Pre-protocol DarkoFit source:
  `542e28177a4f9a8ab7fe734359e4d7647dce18d9`.
- Frozen `darkofit/` package Git tree:
  `1a60b529c5f5d09920d81338406b491fb7275e3a`.
- Basketball CSV SHA-256:
  `43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2`.
- Creator training fingerprints:
  - `X`: `05efa554a983942885b72b1b3fdcc97a10ecf4ccbb4b10843ee8b970129fe71b`;
  - `y`: `7c06b45d4649c392fdb0c3dc91a857650f2f3fc77341fe06ffdbee2b3a44aecf`.
- The existing `MP > 500` filter, 15 features, `MPG` target, unshuffled
  ten-fold creator split, alphabetical team holdout, and player masks remain
  unchanged.
- The overlap-exposed held-team view has 2,409 rows. Its cold-player subset
  has 585 rows whose exact source player identifier is absent from training;
  its mask SHA-256 is
  `e17887c9240fd95aee21d37d7e1e8a533c38ef87f4492acd28fb3aa496a3dd19`.
- The frozen runtime is Python `3.12.13` on
  `macOS-26.5.2-arm64-arm-64bit`, Apple M5 Max, with 18 logical CPUs. Exact
  package versions are NumPy `2.4.6`, Numba `0.66.0`, llvmlite `0.48.0`,
  pandas `3.0.3`, scikit-learn `1.9.0`, SciPy `1.18.0`, joblib `1.5.3`, and
  threadpoolctl `3.6.0`.
- No TabArena, CTR23 development coordinate, or lockbox data is used.

The run must use clean committed source descended from the pre-protocol
commit, with the exact frozen `darkofit/` package tree and runtime stack
above, 18 threads per fit, and random state 4. The formal runner must
hard-code and verify the final byte SHA-256 of this protocol, the package-tree
hash, every frozen dependency version, and the platform, architecture, CPU
brand, and logical CPU count; a protocol, package, runtime, or threshold
change fails closed. Runner and runner-test commits outside the package are
permitted. Source state, dependency versions, data hashes, split indices,
prediction hashes, fitted metadata, and offsets are recorded.

## Frozen fitting and calibration rule

For each external creator fold, split that fold's training rows with:

```text
ShuffleSplit(n_splits=1, test_size=0.10, random_state=4)
```

The same rule is applied to the complete creator training view before scoring
the held-team, seen-player, and cold-player views. The internal fit and
calibration indices must be disjoint, cover the external training indices
exactly, and be recorded and hashed.

Fit two models on only the internal fit rows:

```text
DarkoRegressor(
    loss="Quantile",
    alpha=<0.1 or 0.9>,
    random_state=4,
    thread_count=18,
    diagnostic_warnings="never",
)
```

All unlisted parameters retain the source-bound product defaults. In
particular, `early_stopping=False`, `tree_mode="catboost"`, and the normal
early-stopping-off automatic learning-rate policy remain unchanged. The
fit call must not receive `eval_set`; therefore the default
`use_best_model=True` is inactive, and the internal calibration rows cannot
select a fitted prefix or otherwise affect the model. They are never used to
build trees.

For quantile level `a`, let `q_cal` be the unchanged model prediction on the
internal calibration rows and let:

```text
residual = y_cal - q_cal
k = min(ceil((n_cal + 1) * a), n_cal)
offset = sort(residual)[k - 1]
candidate = control + offset
```

The runner must independently verify that every offset equals the frozen rank
definition. No interpolation, weighting, clipping, tail-specific tuning,
cross-fold pooling, refit, or per-player adjustment is allowed.

## Frozen metrics

For each alpha and evaluation view, record:

- marginal coverage `mean(y <= q)`;
- pinball loss;
- prediction SHA-256;
- offset and calibration residual rank.

For the paired 10%/90% interval, record:

- 80% interval coverage;
- mean interval width;
- crossing rate;
- summed lower-plus-upper pinball loss.

Creator-fold aggregate metrics are computed by pooling the ten external test
predictions, so every creator row appears exactly once. Fold-level metrics are
also retained. Held-team, seen-player, and cold-player metrics remain
separate.

## Fatal quality gates

The candidate advances only if all conditions hold:

1. every prediction, offset, metric, and fitted-model value is finite;
2. candidate quantile-crossing count is no greater than the unchanged
   control count on every evaluation view;
3. on pooled creator rows, pinball loss and absolute coverage error are no
   worse for each tail separately; 80% interval coverage error is no worse,
   and summed pinball loss is no higher;
4. candidate summed pinball loss is strictly lower on at least six of ten
   creator folds;
5. the worst creator-fold candidate/control summed-pinball ratio is at most
   `1.02`;
6. on the overlap-exposed held-team view, candidate pinball loss and absolute
   marginal-coverage error are no worse for each tail separately; candidate
   summed pinball and absolute 80% interval-coverage error are also no worse;
7. on the 585-row cold-player view, candidate pinball loss and absolute
   marginal-coverage error are no worse for each tail separately; candidate
   summed pinball and absolute 80% interval-coverage error are also no worse;
   and
8. candidate mean interval width is at most `1.25` times control on pooled
   creator rows, held teams, and cold players.

The seen-player subset is reported but is not an additional gate because it
already dominates the held-team row count. Ties pass except for the explicit
strict fold-win gate; comparisons use the unrounded float64 values stored in
the artifact.

## Stop and advance rules

If any fatal gate fails:

- record `stop_before_product_implementation`;
- do not add a calibration parameter, automatic policy, or serialization
  field;
- do not rerun with altered fractions, seeds, quantiles, thresholds, or
  player-specific offsets; and
- do not spend a broader panel.

If every gate passes, the result authorizes only a separately reviewed,
default-off product implementation that reproduces this exact offset when a
valid calibration set exists. That implementation must preserve unchanged
predictions when disabled, cover explicit and automatic validation sources,
staged prediction, refit semantics, serialization hardening, sklearn
compatibility, and weighted-data behavior before another basketball
confirmation. A universal or automatic default would still require broader
fresh evidence.

Temperature scaling for classification and conformal correction of
distributional intervals are separate mechanisms and are not authorized by
this protocol.
