# GPBoost versus DarkoFit basketball characterization, v3

## Execution identity and prior invalid attempts

This final user-requested characterization supersedes two invalid raw-output
attempts recorded in `TESTING_LOG.md`. The v1 four-thread and v2 one-thread
runs both completed all workers but failed only because their runner demanded
byte-identical full prediction hashes across fresh processes. Neither attempt
published or summarized metrics. Synthetic cross-process checks were bitwise
stable for all four estimators, so the observed full-panel hash variation is
treated as an external-comparator measurement property, not a quality verdict.

V3 preserves the exact v2 single-thread arm/data/source envelope. It changes
only the pre-run repeatability contract: all three raw repeats are retained;
the runner requires identical split identity, constructor parameters, thread
environment, and fitted-tree metadata, while it records each repeat's
prediction fingerprint and reports all three score/cost measurements. It does
not choose a favorable repeat. A structural mismatch still fails closed.

## Data, arms, and scope

The cache is the already-spent player-disjoint basketball panel 2: complete
2014--2016 seasons crossed with `minutes_per_game`, `game_score`, and
`box_plus_minus`; ten frozen `GroupKFold(bref_id)` primary folds; and held
team/seen player/cold player views. Both implementations receive only the
same 15 numeric predictors. GPBoost's `GPModel`, player/group IDs, and other
extra inputs are excluded.

The two lanes are unchanged from v1/v2:

- public estimator defaults with only seed and one-thread controls;
- near-matched capacity with 1,000 maximum rounds, learning rate 0.1, depth
  6, L2 1, 128 bins, full sampling, no validation set, and fitted trees
  reported.

The DarkoFit source is a clean archive of
`b666d7d5c6583f6629adb8ae43795286c1260d43`; GPBoost is the installed
1.7.1.1 wheel. Three fresh workers per arm run in the fixed reciprocal order,
after one first-fold warmup outside timing.

## Reporting and non-claims

For every lane and primary/held-team/cold-player view, report all three
equal-lineage geometric `GPBoost RMSE / DarkoFit RMSE` ratios plus median,
minimum, and maximum. Likewise report all paired one-thread fit, prediction,
wall, and RSS ratios. These are Tier-E descriptive measurements only. They
authorize no product/default/release/Pareto/fresh-data claim, and they do not
evaluate GPBoost random effects.
