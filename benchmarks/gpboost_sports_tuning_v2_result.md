# Tuned GPBoost on fresh basketball data, v2 result

## Decision

The preregistered regularized GPBoost configuration materially improves the
public GPBoost default and is better than the DarkoFit public default on this
small, fresh 2020 basketball confirmation. It is a Tier-E external-comparator
characterization only: it authorizes no DarkoFit product/default/release claim
and says nothing about GPBoost's `GPModel` mixed-model surface.

The configuration was selected once from 48 group-disjoint Optuna trials on
the untouched 2008--2013 development partition. It was not adjusted after the
2020 result:

```text
learning_rate=0.016033593777058583  num_leaves=30   max_depth=3
min_child_samples=10                min_child_weight=0.24612149472196407
reg_lambda=1.644117697870773        reg_alpha=0.376256053156132
colsample_bytree=0.8213733831219489 subsample=0.7945928828064571
subsample_freq=1                    max_bin=255     n_estimators=2000
random_state=4                      n_jobs=4
```

## Quality

Values are geometric mean RMSE ratios, right arm / left arm; lower favors the
right arm. Every one of the three rotating fresh-worker blocks had the same
displayed ratios.

| Comparison | Player-disjoint primary | Held team | Cold player |
| --- | ---: | ---: | ---: |
| GPBoost default / DarkoFit default | 1.051918 | 0.977470 | 0.975940 |
| Tuned GPBoost / DarkoFit default | **0.962774** | **0.890823** | **0.888861** |
| Tuned GPBoost / GPBoost default | **0.915256** | **0.911356** | **0.910775** |

Thus tuning improved GPBoost by 8.47% primary, 8.86% held-team, and 8.92%
cold-player RMSE relative to its public default. Against DarkoFit, the tuned
configuration improved the same views by 3.72%, 10.92%, and 11.11%,
respectively. The three target-level primary ratios against DarkoFit were
0.981878 (`minutes_per_game`), 0.993275 (`game_score`), and 0.915051
(`box_plus_minus`).

## Four-thread cost telemetry

Tuned GPBoost / DarkoFit median ratios were 0.811465 total fit time, 0.268411
total prediction time, 0.810221 steady wall time, and 0.778300 peak RSS. The
tuned configuration costs about 9.9x its 100-tree GPBoost public default to
fit, but remains below DarkoFit in this limited 2020 envelope.

## Bound artifacts and limitation

- Protocol: [`gpboost_sports_tuning_v2_protocol.md`](gpboost_sports_tuning_v2_protocol.md)
- Frozen development record: [`gpboost_sports_tuning_v1_development_20260723.json`](gpboost_sports_tuning_v1_development_20260723.json)
- Raw confirmation: [`gpboost_sports_tuning_v2_confirmation_20260723.json`](gpboost_sports_tuning_v2_confirmation_20260723.json)

The source is the fixed Basketball Reference export; both libraries saw the
same 15 numeric predictors, without player/team IDs or GPBoost group effects.
The confirmation has only three 2020 target lineages, so it is not evidence of
general quality, full Pareto dominance, or production readiness.
