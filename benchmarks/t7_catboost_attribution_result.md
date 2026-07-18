# T7 CatBoost mechanism attribution

**Decision: `freeze_t7_research_candidates`.**

## Evidence bindings

- Frozen raw file SHA-256: `cf199793c5e3349ee4a8e3575870f9cacec2905e54b84fc4bcf2703a70cb518f`
- Frozen raw canonical SHA-256: `6673fe69c5e09d1e020252237c322e7795c14effdf375e9dd1c0db3ecc4772ee`
- Frozen protocol SHA-256: `18200d9bd8f6b43ec345be5755ce795f6284ae399a43eeae0144cd860718f460`
- Original run-time runner SHA-256: `be1178f8593d3ff52a19963812932b399fbfbc3fd1942b97ad663ee9fe728a49`
- Current hardened analyzer SHA-256: `4de5a7c36b4cfb8792d6a0517e80ef2c2747d4458c9fb78dbc9d71bb88947921`
- Current hardened runner SHA-256: `71d55bb9db9970754fe77f41cc595d5078921b637a1c53ff32d459f3889cc310`
- Frozen C2 split-helper SHA-256: `8da023ee1c6ab1311d0b8b152c8bcd82f80d6f323020efb4e86c71870caa8952`
- Current C2 split-helper SHA-256: `8da023ee1c6ab1311d0b8b152c8bcd82f80d6f323020efb4e86c71870caa8952`

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
| `ordered` | 0.878621 |
| `plain` | 1.029345 |
| `border_128` | 1.009231 |
| `leaf10_no_backtracking` | 1.009320 |
| `leaf10_any_improvement` | 1.011111 |
| `ctr_complexity_2` | 1.021204 |
| `depth_4` | 0.870853 |
| `depth_8` | 1.067648 |
| `depth_by_n_p` | 1.069729 |

This is a descriptive historical anchor, not a current-release confirmation
claim. All nine measured CatBoost arms and the assembled depth policy are
reported. The surviving CatBoost depth policy widens rather than closes the
historical competitive gap; porting the rule to DarkoFit would require a
separate implementation and outcome-unseen evaluation.

All results use spent development tasks. No confirmation panel or lockbox was
opened, and no default change is authorized.
