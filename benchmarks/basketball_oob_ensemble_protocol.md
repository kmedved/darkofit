# Basketball OOB ensemble screen

## Decision being tested

This diagnostic asks whether a five-member DarkoFit bootstrap ensemble, with
each member early-stopped on its own out-of-bag (OOB) rows, is worth exposing
as an **opt-in accuracy feature**. It does not authorize a default change or
add an estimator parameter by itself.

The mechanism mirrors ChimeraBoost's documented bagging design, but this
screen is an independent DarkoFit implementation using the existing public
`DarkoRegressor` API. No ChimeraBoost source is copied into DarkoFit.

## Immutable basketball boundary

The data bytes, `MP > 500` filter, 15 features, target, seed 4, unshuffled
10-fold split, alphabetical team holdout, and cold-player subset are inherited
unchanged from the creator benchmark and `basketball_harness.py`.

The two arms are:

| Arm | Configuration |
|---|---|
| `default` | One current-default `DarkoRegressor(random_state=4)` |
| `oob_ensemble5` | Five bootstrap members; deterministic member and bootstrap seeds; each member uses `early_stopping=True`, automatic patience, its exact OOB complement as `eval_set`, `use_best_model=True`, and no refit; predictions are the arithmetic mean |

Every bootstrap has the same number of draws as its external training set.
The runner records bootstrap and OOB hashes, OOB row counts, fitted metadata,
member prediction hashes, and the averaged prediction. An empty OOB complement
is a hard failure in this benchmark rather than a fallback to contaminated
validation.

## Quality gates

The candidate advances to public-API implementation only if all five gates
pass:

1. mean 10-fold R² is not below the default;
2. at least 6 of 10 folds improve;
3. the mean R² delta remains nonnegative after omitting any one fold;
4. overlap-exposed team-holdout R² is not below the default; and
5. cold-player R² is not below the default.

The seen-player subset is reported but does not gate independently because it
is already represented in the full team holdout.

## Clean timing and complexity budget

Run three fresh-process reciprocal blocks:

1. `default`, candidate;
2. candidate, `default`;
3. `default`, candidate.

Each process gets the same full-machine thread allocation and warms one
complete first-fold fit and prediction outside the timer. Prediction and
fitted-behavior fingerprints must match exactly across repeats. Timing is
admissible only when each arm's maximum/minimum steady-time ratio is at most
1.20.

Because this is an explicitly requested five-model accuracy feature, parity
with the single model is not a sensible runtime requirement. The candidate
must nevertheless beat naive fivefold scaling:

- median 10-fold steady wall time must be at most 4.0× the default; and
- median summed prediction time must be at most 6.0× the default.

These are feature-specific gates frozen before the formal run. Passing means
only “implement and test an opt-in API.” It never means “make ensembles the
default.” A future public implementation must independently cover regression,
classification, weights, categoricals, groups, serialization, calibration,
distributional behavior, sklearn cloning, and process/thread budgeting before
release.

## Evidence discipline

The informal one-pass prototype used to decide whether a formal experiment
was worthwhile is development evidence only. The committed clean reciprocal
run is the decision artifact. No CTR23 lockbox coordinates are used.
