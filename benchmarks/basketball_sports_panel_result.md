# Basketball multi-season sports confirmation result

## Decision

`random_strength=0.5` failed the frozen S4 confirmation gate. Close it without retuning on this now-spent panel; the global default remains `0.0`.

Decision code: `close_random_strength_0_5_without_s4_confirmation`.

## Candidate versus control

| Measure | Result |
|---|---:|
| Equal-cell mean R² delta | +0.000036 |
| Minimum leave-one-cell-out delta | -0.000702 |
| Held-team equal-cell R² delta | +0.002499 |
| Cold-player equal-cell R² delta | +0.001937 |
| Median fit-time ratio | 1.835× |
| Fit paired-ratio IQR/median | 0.019 |

## Same-machine comparison

| Arm | Equal-cell R² | Cold-player R² | Median total fit |
|---|---:|---:|---:|
| `catboost_1_2_10` | 0.803272 | 0.816725 | 37.998s |
| `darkofit_random_strength_0_5` | 0.786605 | 0.795666 | 176.722s |
| `darkofit_control` | 0.786569 | 0.793729 | 95.979s |
| `chimeraboost_0_15_0` | 0.761923 | 0.787290 | 16.203s |

Formal quality claims:

- Beats ChimeraBoost on S4: **True**.
- Beats CatBoost on S4: **False**.

The nine target-season cells receive equal weight. The result is specific to this preregistered basketball panel; it does not authorize a global default change. The panel is now spent and may not be used for retuning.

Raw artifact SHA-256: `de1a22ad42a98fe44136aba002806d6fbbe19139f7763eeb5e594a2bd8e42299`.
