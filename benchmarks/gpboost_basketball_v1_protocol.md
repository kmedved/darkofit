# GPBoost versus DarkoFit basketball characterization, v1

## Question and evidence boundary

This is a user-requested, descriptive Tier-E comparison of GPBoost 1.7.1.1
and DarkoFit on the already-spent player-disjoint basketball panel 2. It
measures the libraries' public regression estimators on the same 15 numeric
predictors. It is not a DarkoFit product-policy gate, a release claim, or a
decision about a default.

The panel was previously spent by the row-OOB ensemble confirmation. This
comparison may characterize a third-party package but cannot be used to retune
either arm, reopen that campaign, or access fresh/lockbox data.

## Data and splits

The runner validates the frozen
`basketball_sports_panel_v2_manifest.json` and its cached processed panel
before execution. The data are the nine season-target lineages from complete
2014--2016 NBA seasons, using the panel's 15 predictors and three scalar
targets:

```text
minutes_per_game, game_score, box_plus_minus
```

Within each season, the middle third of teams is a held-team slice. The
remaining rows use the frozen ten-fold `GroupKFold(bref_id)` plan, so a player
does not appear in both sides of a primary fold. Held-team scoring is reported
separately for all, seen-player, and cold-player rows. Primary RMSE pools the
out-of-fold predictions within each lineage.

GPBoost receives no `GPModel`, player ID, team ID, coordinates, or other group
input. Its random-effect facility would supply a model input DarkoFit does not
receive, so it is deliberately outside this head-to-head feature-only
characterization.

## Arms

All arms use random state 4 and four worker threads. There are two lanes:

1. **Public defaults.** `DarkoRegressor(random_state=4, thread_count=4)` is
   compared with `GPBoostRegressor(random_state=4, n_jobs=4)`. Thread and seed
   controls are execution controls; the libraries otherwise retain their
   public defaults. In particular, GPBoost defaults to 100 boosting rounds,
   while DarkoFit defaults to 1,000 and resolves its own learning rate.
2. **Near-matched tree budget.** Both arms use a maximum of 1,000 rounds,
   learning rate 0.1, maximum depth 6, L2 penalty 1, 128 bins, full
   row/feature sampling, no validation set, and thus no validation-based early
   stopping. DarkoFit uses its CatBoost tree mode and minimum child samples 1.
   GPBoost uses 64 maximum leaves and minimum child samples 1. Actual retained
   tree counts are reported because a library may stop adding useful splits.
   The libraries' tree-growth and Hessian/minimum-child semantics remain
   different, so this is a capacity alignment, not prediction-exact engine
   parity.

## Execution and reporting

The immutable DarkoFit source is a clean archive of commit
`b666d7d5c6583f6629adb8ae43795286c1260d43`. The runner records that commit,
the archive content hash, GPBoost distribution version, runner/protocol
hashes, panel fingerprints, fitted-tree metadata, prediction hashes, timing,
and peak RSS.

Three fresh-worker blocks rotate arm order. Each worker makes one first-fold
fit outside the timing interval, then fits every primary fold and held-team
model. Imports, panel loading, and warmup are outside steady timing. The
runner requires behavior fingerprints to match across the three repeats for
each arm.

For each lane, the report gives the equal-lineage geometric mean of
`GPBoost RMSE / DarkoFit RMSE` for primary, held-team, and cold-player views,
along with lineage win/loss/tie counts. It reports median per-worker fit,
prediction, steady-wall, and RSS ratios in the same orientation. Ratios are
measurements only: no threshold is a pass/fail gate and neither lane can
authorize a DarkoFit change.
