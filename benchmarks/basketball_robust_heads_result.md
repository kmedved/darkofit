# Basketball robust-head result

_Run 2026-07-17 from clean DarkoFit `f10b449` under the frozen
[`basketball_robust_heads_protocol.md`](basketball_robust_heads_protocol.md).
The complete create-only artifact is
[`basketball_robust_heads.json`](basketball_robust_heads.json), SHA-256
`f631a1346b55b407b68e83f15c9214e3a83c6d46e602d28cfd612ff3d1f819a4`._

## Decision

Close both robust heads as shaped. Neither advances to the sports confirmation
suite, and neither changes DarkoFit's RMSE default.

| Arm | Mean creator-fold R² | Delta vs RMSE | Fold wins | Held-team delta | Cold-player delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE control | 0.526750 | — | — | — | — |
| Student-t location | 0.518768 | −0.007982 | 3 / 10 | −0.010223 | +0.005348 |
| MAE | 0.518288 | −0.008461 | 2 / 10 | −0.002399 | +0.000697 |

Both robust heads modestly improved the 585-row cold-player subset, but that
effect did not generalize to the primary creator folds or the full held-team
view. Both failed the required `+0.002` mean gain, leave-one-fold-out
non-regression, and held-team non-regression gates.

## Execution

The fatal first block reproduced the RMSE control exactly at
`0.5267495183883605`. Because both candidates failed quality, the protocol
correctly skipped the two additional timing blocks. The single-block wall
times—descriptive only—were 10.08 seconds for RMSE, 58.76 seconds for
Student-t, and 13.87 seconds for MAE. No paired-ratio stability or runtime
claim is made.

No CTR23 dataset, confirmation coordinate, lockbox task, sample weight,
per-arm tuning, or new model code was used.
