# Basketball sports automatic-policy confirmation, panel 2

## Decision

The five-member row-OOB ensemble **failed** the frozen Tier-D gate. Close it as a sports automatic policy without retuning on this now-spent panel.

Decision code: `close_oob_ensemble5_as_sports_automatic_policy`.

## Candidate versus control

| Measure | Result |
|---|---:|
| Equal-lineage RMSE ratio | 1.023115× |
| 95% bootstrap upper | 1.040115× |
| Remove-best-lineage ratio | 1.033121× |
| Worst lineage ratio | 1.064535× |
| Held-team aggregate ratio | 1.026277× |
| Cold-player aggregate ratio | 1.028733× |
| Median total-fit ratio | 2.918× |
| Median total-predict ratio | 2.677× |
| Median peak-RSS ratio | 1.056× |

## Same-machine context

| Arm | Primary RMSE | Cold-player RMSE | Median fit |
|---|---:|---:|---:|
| `catboost_1_2_10` | 1.865114 | 1.711959 | 59.095s |
| `darkofit_control` | 1.963257 | 1.807506 | 142.703s |
| `darkofit_sports_oob_ensemble5` | 2.008637 | 1.859440 | 416.344s |
| `chimeraboost_0_15_0` | 2.020070 | 1.840381 | 26.086s |

The nine target-season lineages receive equal weight. Primary folds are player-disjoint. External comparisons are descriptive and cannot rescue the candidate decision. Panel 2 is spent and may not be used for retuning.

## Evidence bindings

- Frozen raw artifact SHA-256: `787f7f34bf1e5207d231b01bc402c7a32174e24892b2118bb71d5ff4412517b3`.
- Original run-time analyzer SHA-256: `1752214e0451b44823557d03fd6e498fecdd8711ceacad95552c0e0dbd0d1e77`.
- Current hardened analyzer SHA-256: `382cd1e590335eb52dabe53ced8764d55c54323336c9024cea80714c2794a5e8`.

The original hash attests the analyzer that first published this result. The current hash attests the hardened analyzer that regenerates this derived report; the frozen raw artifact and campaign outcome were not changed or rerun.
