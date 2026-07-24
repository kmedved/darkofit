# GPBoost regularization tuning on basketball data, v1

## Purpose and boundary

This is a user-authorized external-comparator development study. It asks
whether a regularized, player-group-safe GPBoost configuration improves on
GPBoost's public default without repeating the over-capacity configuration
that lost on the previously spent 2014--2019 basketball panels.

The study may characterize GPBoost only. It cannot change DarkoFit, establish
Pareto dominance, or make a product/default/release claim. GPBoost's
`GPModel`, player IDs, team IDs, and other extra group inputs remain excluded;
both models receive the same 15 numeric predictors.

## Source and partition

The raw Basketball Reference game-log source is
`bbr_advanced_game_logs.csv`, SHA-256
`96e0efffb09e27f64cee395faa1783b025757c88efb74f5fb98cbd82c583d826`.
The existing formal sports panels spent 2014--2019. This study freezes:

- **development/tuning:** complete seasons 2008--2013, crossed with
  `minutes_per_game`, `game_score`, and `box_plus_minus` (18 lineages);
- **confirmation:** complete season 2020, crossed with the same three
  targets (three lineages).

The source/build manifest records row identities, processed bytes, and target
fingerprints before fitting. No confirmation score may be read until the
development artifact has selected and frozen one configuration.

Within a season, player-team-season rows require over 500 played minutes and
are built with the established 15-feature transformation. Development uses a
deterministic player-disjoint 80/20 group split in every lineage. Confirmation
uses player-disjoint ten-fold primary evaluation plus the middle-third held
team slice, reported as all/seen-player/cold-player views.

## Optuna search

The development study uses 48 sequential `TPESampler(seed=20260723)` trials,
four native threads, GPBoost `n_estimators=2000`, and 50-round early stopping
on the group-disjoint validation portion. Each trial is compared to the
GPBoost public default on the identical validation rows. Its objective is the
equal-lineage geometric mean of `trial RMSE / default RMSE` across the 18
development lineages; lower is better. The artifact records every cell ratio,
the worst lineage, all trial parameters, and the selected configuration.

The search space is:

```text
learning_rate       log-uniform [0.01, 0.15]
num_leaves          integer [7, 31]
max_depth           integer [3, 6]
min_child_samples   integer [10, 80]
min_child_weight    log-uniform [0.001, 10]
reg_lambda          log-uniform [0.1, 100]
reg_alpha           log-uniform [0.000001, 10]
colsample_bytree    uniform [0.60, 1.00]
subsample           uniform [0.60, 1.00], with subsample_freq=1
max_bin             categorical {63, 127, 255}
```

No hand grid, target-specific tuning, group effect, or post-selection tuning
is allowed. The selected configuration is the best finite objective as
reported by Optuna; its exact parameters and development artifact hash bind
confirmation.

## Confirmation

Confirmation compares three arms on 2020: DarkoFit public default, GPBoost
public default, and the frozen tuned GPBoost configuration. Each tuned outer
fold derives an early-stopping iteration from a deterministic player-disjoint
inner split, then refits on the complete outer training set at that iteration.
No 2020 result may affect parameters or the tuning study.

Three fresh-worker blocks rotate arm order. Every worker has one warmup
outside timing (the first `minutes_per_game` outer fold) and records
fit/predict/wall time, peak RSS, model metadata, and prediction hashes. The
warmup prediction is discarded rather than scored. The confirmation artifact
binds the clean DarkoFit source archive's exact Git revision and package
fingerprint. The report gives all three block-level GPBoost/DarkoFit and
tuned/default RMSE ratios; it does not discard a less favorable repeat.
