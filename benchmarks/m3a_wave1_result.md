# Wave 1 M3a result

## Outcome

Track B disposition: **close_preserve_current_opt_in**. The frozen group-safe survival rule did not pass.

No default change or cross-season generalization is authorized.

## Player-disjoint quality

| Pair | Primary geomean | Season-cluster p95 | Held-team | Cold-player |
|---|---:|---:|---:|---:|
| DarkoFit group8 / single | 1.025482 | 1.032391 | 1.016048 | 1.015661 |
| ChimeraBoost ensemble8 / single | 0.950230 | 0.962517 | 0.977973 | 0.977935 |

The interval resamples only three spent season clusters and is descriptive.

## Frozen DarkoFit survival checks

| Check | Value | Limit | Passed |
|---|---:|---:|:---:|
| integrity | True | required | yes |
| player_geomean | 1.025482 | 0.995000 | no |
| player_cluster_p95 | 1.032391 | 1.000000 | no |
| held_geomean | 1.016048 | 1.005000 | no |
| cold_geomean | 1.015661 | 1.005000 | no |
| worst_season | 1.036053 | 1.010000 | no |
| worst_player_cell | 1.061412 | 1.030000 | no |
| fit_cost | 4.770486 | 9.000000 | yes |
| predict_cost | 3.898647 | 9.000000 | yes |
| model_bytes | 3.929268 | 9.000000 | yes |
| peak_rss | 1.091085 | 4.000000 | yes |

## General medium-slice context

| Pair | Geomean RMSE ratio | Wins | Cells |
|---|---:|---:|---:|
| DarkoFit row8 / single | 1.019556 | 2 | 6 |
| ChimeraBoost ensemble8 / single | 0.947797 | 6 | 6 |

These row-sampling cells are descriptive and cannot rescue a failed group-safe sports result.

## Timing handling

Primary repeat series run: **no** — group-safe quality did not survive; repeats forbidden.

Diagnostic costs are single warmed observations. Aggregate RSS is a sampled worker-plus-child-process value.
