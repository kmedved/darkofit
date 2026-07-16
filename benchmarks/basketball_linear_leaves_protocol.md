# Basketball validation-selected linear-leaf protocol

## Decision being tested

This campaign asks whether hessian-weighted per-leaf linear models are a safe
candidate for closing DarkoFit's smooth-data quality gap without weakening its
small, noisy sports behavior. Basketball is the first fatal screen because it
is fast and directly represents the user's primary data regime.

The work is split into two independently reviewable stages:

1. an explicit, default-off `linear_leaves=True` core mechanism whose disabled
   and fallback paths must be bit-identical to today's constant leaves; and
2. a benchmark-only validation selector that chooses constant or linear
   leaves, then refits the winner on the complete external training fold using
   the unchanged current-default horizon and learning-rate policy.

Passing this campaign does not authorize a default or public automatic
selector. It only permits broader development validation.

## Provenance and optimization case

The small LU solver and hessian-weighted ridge formulation are adapted from
Apache-2.0 ChimeraBoost commits `425dfb3`, `832e7a2`, and `ce5ca0d`; attribution
is recorded in `NOTICE`. DarkoFit does not cherry-pick the donor commits. It
attaches coefficients after DarkoFit's existing split search and float64 leaf
refresh so split legality, tie-breaking, and the constant path remain native.

Existing basketball profiles put approximately 99% of DarkoFit fit time in
tree construction. The mechanism deliberately adds work there, so it must
earn that cost through quality and preserve a bounded prediction path.

| Opportunity | Impact | Confidence | Effort | Score |
|---|---:|---:|---:|---:|
| Linear leaves for the smooth-data gap | 5 | 4 | 5 | 4.0 |

The score is `impact × confidence / effort`; it clears the required 2.0 bar.

## Stage A: core behavior gates

Before any basketball run:

- `linear_leaves=False` and every ineligible fallback must reproduce the
  existing stable prediction goldens exactly;
- the readable linear-leaf oracle and NumPy solve comparison must pass;
- split features, thresholds, gains, leaf routing, and constant values must be
  unchanged when linear leaves are enabled—the feature may attach coefficients
  but may not change tree structure;
- leaves with fewer than twice their fitted coefficient count must fall back
  bit-identically to the constant Newton value;
- NaN bins contribute the standardized feature mean (zero), and all fitted
  coefficients and predictions must be finite;
- explicit linear leaves are initially eligible only for scalar RMSE,
  `tree_mode="catboost"`, at least 1,000 training rows, no ordered leaf update,
  and at least one numeric split feature; every other explicit request fails
  clearly or records a deterministic constant fallback;
- `.npz` serialization with `allow_pickle=False` must round-trip predictions
  bit-identically and reject malformed linear payloads; and
- prediction uses one packed forest traversal rather than one Python/Numba
  launch per tree.

Isomorphism proof for the disabled path:

- Ordering preserved: yes; no new code executes when linear leaves are off.
- Tie-breaking unchanged: yes; the existing tree builder is untouched.
- Floating point: identical on the constant path and constant fallbacks.
- RNG seeds: unchanged.
- Golden outputs: `prediction_goldens.py` plus strict golden tests must pass.

## Stage B: frozen basketball arms

The creator bytes, `MP > 500` filter, 15 features, target, seed 4, unshuffled
10 folds, alphabetical team holdout, and cold-player subset remain unchanged.

| Arm | Policy |
|---|---|
| `default` | Current `DarkoRegressor(random_state=4)` |
| `linear_leaf_select_refit` | Within each external training fold, create the existing deterministic 10% regression validation split. Fit constant and linear variants on the same selection rows with early stopping and identical non-leaf parameters. Choose the lower best validation RMSE; ties choose constant. Refit only the chosen leaf type on the complete external training fold with the unchanged default 1,000 rounds and full-data automatic learning rate. |

The selector records the split hashes, both validation curves/best scores,
chosen lane, selection tree counts, final tree count, resolved learning rates,
stop reasons, phase times, prediction hashes, linear-tree/leaf counts, model
bytes, and peak resident memory where available.

## Basketball gates

The selector advances only if all quality gates pass:

1. mean 10-fold R² is not below the default;
2. at least 6 of 10 folds improve;
3. every leave-one-fold-out mean delta is nonnegative;
4. overlap-exposed team-holdout R² is not below the default; and
5. cold-player R² is not below the default.

A clean canonical block runs first. If any fatal quality gate fails, the
campaign stops immediately and records the timing/resource confirmation as
not run; repeating a rejected behavioral result would spend machine time
without changing the decision. Only a quality-passing candidate proceeds to
three fresh-process reciprocal blocks, all of which must produce one behavior
fingerprint per arm. Each arm's steady max/min ratio must be at most 1.20.
Because the candidate deliberately fits two selectors plus one final model,
median fit wall time may be at most 3.5× default. Its packed median prediction
time may be at most 1.25× default, every corresponding serialized final-model
size ratio at most 3.0×, and median fresh-worker peak RSS at most 2.0× default.
A missing required measurement fails closed.

## Advance path

If basketball fails, stop and retain the explicit core only if it is useful,
fully supported, and default-off; otherwise revert it. If basketball passes,
run the selector on the 243 unused-but-spent CTR23 development coordinates,
especially `kin8nm`, `grid_stability`, and `space_ga`. Do not touch the
lockbox. A public automatic policy still requires a fresh preregistered panel,
classification/weighted/alternate-loss coverage, and the existing ≥80%
simulated lockbox-pass power gate.
