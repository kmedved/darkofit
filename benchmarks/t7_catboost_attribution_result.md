# T7 CatBoost mechanism attribution

**Decision: `freeze_t7_research_candidates`.**

| Contrast | Test ratio | Validation ratio | Worst task | W/L/T |
|---|---:|---:|---:|---:|
| `ordered_over_plain` | 1.171545 | 1.127883 | 1.598850 | 2/6/0 |
| `plain_over_default` | 1.000000 | 1.000000 | 1.000000 | 0/0/8 |
| `border_128_over_default` | 1.019929 | 1.000988 | 1.193205 | 3/2/3 |
| `leaf10_no_backtracking_over_default` | 1.019840 | 1.024908 | 1.236246 | 3/5/0 |
| `backtracking_over_no_backtracking` | 0.998229 | 0.992218 | 1.027992 | 3/5/0 |
| `ctr_complexity_2_over_default` | 1.007972 | 1.005857 | 1.060936 | 2/4/2 |
| `depth_4_over_default` | 1.181997 | 1.194660 | 1.773283 | 3/5/0 |
| `depth_8_over_default` | 0.964124 | 0.941758 | 1.408649 | 5/3/0 |
| `depth_by_n_p_over_default` | 0.962248 | 0.967478 | 1.000000 | 3/0/5 |

Frozen research candidates (maximum three): `depth_by_n_p`.

## Surviving candidate by dataset

| Candidate | Dataset | Test ratio | Validation ratio | Selected arm |
|---|---|---:|---:|---|
| `depth_by_n_p` | auction_verification | 1.000000 | 1.000000 | `default` |
| `depth_by_n_p` | video_transcoding | 0.745235 | 0.774124 | `depth_8` |
| `depth_by_n_p` | fps_benchmark | 1.000000 | 1.000000 | `default` |
| `depth_by_n_p` | cars | 0.987720 | 0.990935 | `depth_4` |
| `depth_by_n_p` | diamonds | 0.998549 | 1.000632 | `depth_8` |
| `depth_by_n_p` | bookprice_prediction | 1.000000 | 1.000000 | `default` |
| `depth_by_n_p` | ae_price_prediction | 1.000000 | 1.000000 | `default` |
| `depth_by_n_p` | munich-rent-index-1999 | 1.000000 | 1.000000 | `default` |

The fixed depth policy uses depth 4 below 100 inner-fit rows per feature,
depth 8 at or above 2,500, and CatBoost's default depth 6 otherwise. It
declines exactly to the default on the five middle-density datasets.

## DarkoFit anchor

Against CatBoost's product default, the immutable C2 DarkoFit control has an
equal-dataset RMSE ratio of
`1.029345`.

| CatBoost arm | DarkoFit / CatBoost RMSE |
|---|---:|
| `default` | 1.029345 |
| `depth_by_n_p` | 1.069729 |

This is a descriptive historical anchor, not a current-release confirmation
claim. The surviving CatBoost depth policy widens rather than closes that
historical competitive gap; porting the rule to DarkoFit would require a
separate implementation and outcome-unseen evaluation.

All results use spent development tasks. No confirmation panel or lockbox was
opened, and no default change is authorized.
