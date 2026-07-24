# GPBoost versus DarkoFit basketball characterization, v2

## Reason for this execution identity

This is a fresh execution identity for the same user-requested descriptive
comparison registered in `gpboost_basketball_v1_protocol.md`. The first,
four-thread execution completed all twelve fresh workers but its runner
refused to publish an artifact: behavior fingerprints differed across repeats
for every arm. The failure was discovered before metrics were summarized or
emitted; it is preserved in `TESTING_LOG.md` as an invalid attempt.

The arms, data, splits, source commit, seed, fit/predict methods, worker
counts, and output calculations are unchanged. The only change is a fixed
single-thread execution policy for both libraries and every native math
runtime. It gives the repetition check a deterministic envelope. Timing from
this run therefore characterizes one-thread performance only and must never
be blended with the invalid four-thread attempt or presented as a full-machine
throughput claim.

## Data and feature boundary

This remains a Tier-E descriptive characterization on the already-spent
`basketball_sports_panel_v2` cache: 2014--2016 seasons crossed with
`minutes_per_game`, `game_score`, and `box_plus_minus`; frozen ten-fold
player-disjoint primary folds; and all/seen/cold held-team views. Both arms
receive only the same 15 numeric feature columns. GPBoost receives no
`GPModel`, player ID, team ID, coordinates, or other grouping input.

## Arms and execution

The public-default and near-matched-tree-budget arms are exactly those in the
v1 protocol. In the matched lane, 1,000 is a maximum and retained trees are
reported. Every arm has random state 4, one native worker thread, one full
first-fold warmup outside timing, and three fresh-process repeats in rotating
order. The runner fails closed unless the prediction/fitted-metadata behavior
fingerprint is identical in all three repeats of each arm.

The DarkoFit implementation is a clean archive of
`b666d7d5c6583f6629adb8ae43795286c1260d43`; GPBoost is the installed PyPI
wheel 1.7.1.1. The create-only raw artifact records source/archive identity,
package versions, panel and split hashes, fitted-tree metadata, prediction
hashes, timing, RSS, raw repeats, and summary calculations.

## Reporting boundary

For each lane, report geometric means of `GPBoost RMSE / DarkoFit RMSE` for
primary, held-team, and cold-player views plus lineage win/loss/tie counts.
Report median paired GPBoost/Darko fit, prediction, wall, and RSS ratios as
one-thread measurements. These are observations, not pass/fail gates. The run
cannot authorize a DarkoFit default, product, release, Pareto, or fresh-data
claim.
