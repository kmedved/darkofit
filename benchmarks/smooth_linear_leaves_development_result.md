# Smooth linear-leaf development result

_Run 2026-07-17 from clean `main` at `ff8d293`, with clean ChimeraBoost
0.15.0 at `851ab7f`, under the frozen
[`smooth_linear_leaves_development_protocol.md`](smooth_linear_leaves_development_protocol.md)._

## Decision

Advance `darko_linear_current` to benchmark-only selector design. Recommend
deprecating global `linear_residual` for DarkoFit 1.0, but do not delete it
yet.

No default or automatic policy is authorized. This was a 21-coordinate
development run on already spent CTR23 confirmation tasks; the lockbox
remained sealed.

## Primary results

| Contrast | Equal-task RMSE ratio | Difference | Dataset record | Split record |
|---|---:|---:|---:|---:|
| Current linear leaves / DarkoFit default | **0.9203×** | **7.97% better** | 3–0 | 21–0 |
| Matched-policy linear / DarkoFit default | 0.9261× | 7.39% better | 3–0 | 19–2 |
| Global linear residual / DarkoFit default | 0.9952× | 0.48% better | 2–1 | 13–8 |
| Current linear leaves / Chimera linear-only | **0.9637×** | **3.63% better** | 3–0 | 20–1 |
| Current linear leaves / Chimera product | 1.0162× | 1.62% worse | 1–2 | 5–16 |
| Chimera linear-only / Chimera product | 1.0546× | 5.46% worse | 0–3 | 2–17 |

The current-policy fixed-linear arm passed every development gate: its
equal-task ratio was below 0.98, it won all three datasets, no dataset
regressed, and it won every split. It also beat the matched 128-bin,
learning-rate-0.1 variant overall, so no smooth-specific numeric default is
needed at this stage.

## Per-dataset linear-leaf effect

| Dataset | Current linear / Darko default | Current linear / Chimera product |
|---|---:|---:|
| Grid stability | 0.9403× | 1.0205× |
| kin8nm | **0.8958×** | 1.0379× |
| space_ga | **0.9253×** | **0.9910×** |

DarkoFit's local-linear implementation is not the remaining problem: it beat
ChimeraBoost's explicit linear-only lane by 2.95–4.70% on every dataset. The
residual product gap comes from ChimeraBoost's separately selected 30
product/difference features, which improved its equal-task result by 5.46%.
That selector is a closed mechanism in this program and is not silently
reopened here.

## Linear residual disposition

Current local-linear leaves beat global linear residual on all three datasets
and 20 of 21 splits, by 7.52% equal-task RMSE. Global residual itself regressed
on space_ga and improved the three-task aggregate by only 0.48%. This meets
the preregistered criterion to prepare its 1.0 deprecation; removal still
requires the normal warning cycle and compatibility documentation.

## Evidence

- Raw artifact:
  [`smooth_linear_leaves_development.json`](smooth_linear_leaves_development.json),
  SHA-256
  `a4022fca9c80892b76a6572a9adb0932cf7068d1417491de70612c50b442a2db`.
- Protocol SHA-256:
  `3b1c1e031a5fba3483dbd06e20a67c18e31424698efb98f0acd603148284138f`.
- Runner SHA-256:
  `9e6241680490746dc865fd232f1be27d56fcef808f96a432f847cc12ec3484ba`.
- The artifact binds both clean repositories, the CTR23-v3 partition, task
  and data hashes, all 21 split-index hashes, fitted route metadata,
  predictions, same-machine timings, and peak RSS.
