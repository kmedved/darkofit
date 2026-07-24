# GPBoost regularization tuning on basketball data, v2 confirmation

## Status and purpose

This is the valid confirmation stage of the user-authorized external
comparator study registered in
[`gpboost_sports_tuning_v1_protocol.md`](gpboost_sports_tuning_v1_protocol.md).
It inherits the immutable v1 source-and-split registry and the immutable v1
development artifact. No tuning, selection, data partition, model policy, or
arm changes are allowed.

The first v1 confirmation worker failed before emitting a score or raw output:
a one-row all/seen/cold held-team slice has undefined R-squared and strict JSON
serialization rejected that `NaN`. No confirmation metric was inspected and no
artifact was written. V2 makes only this reporting correction: R-squared is
recorded as `null` for a slice with fewer than two rows. RMSE, the selection
metric and all confirmation comparisons, is unchanged. The failed v1
confirmation remains a failed pre-output control event; this protocol is
written before the valid confirmation is launched.

## Bound inputs and selected configuration

The raw Basketball Reference game-log source remains
`bbr_advanced_game_logs.csv`, SHA-256
`96e0efffb09e27f64cee395faa1783b025757c88efb74f5fb98cbd82c583d826`. The
immutable v1 registry binds the same 15 numeric predictors, 2008--2013
development rows and player-disjoint splits, and the untouched 2020
confirmation season. It has SHA-256
`6141142e69f1aca57df38573df2ad5d0167bdd1225149e7e31422176df6f96fe`.

The immutable v1 development study has SHA-256
`c4e6dc8ffd0e3a28fd48a1b1cd860cc71b484bdb52da8e3d9d541b6fd6fb2e76`. It
selected trial 36 after all 48 preregistered TPE trials, with these frozen
GPBoost parameters:

```text
learning_rate=0.016033593777058583  num_leaves=30   max_depth=3
min_child_samples=10                min_child_weight=0.24612149472196407
reg_lambda=1.644117697870773        reg_alpha=0.376256053156132
colsample_bytree=0.8213733831219489 subsample=0.7945928828064571
subsample_freq=1                    max_bin=255     n_estimators=2000
random_state=4                      n_jobs=4
```

`GPModel`, player IDs, team IDs, and all group-effect inputs are still
excluded. This is feature-only GPBoost vs DarkoFit characterization, not a
test of GPBoost's mixed-model use case.

## Confirmation execution and reporting

The 2020 confirmation remains three target lineages, player-disjoint ten-fold
primary evaluation, and the v1 middle-third held-team slice with all,
seen-player, and cold-player reports. Arms are unchanged: DarkoFit public
default, GPBoost public default, and frozen tuned GPBoost. The tuned arm uses
a deterministic player-disjoint inner split for early-stopping iteration and
refits the full outer train set at that selected iteration.

Three fresh-worker blocks rotate arm order. Each worker discards one first-
fold `minutes_per_game` warmup outside timing. All blocks are retained. The
artifact records RMSE, R-squared where defined, fit/predict/wall time, peak
RSS, tree metadata, and prediction hashes. It binds the archive's exact
DarkoFit Git revision and package fingerprint. No result selects another
configuration or changes either library.
