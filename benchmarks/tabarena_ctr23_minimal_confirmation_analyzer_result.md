# Minimal CTR23 regression confirmation

external fixed-panel confirmation only; no lockbox, preset, or default-change authorization

Negative percentages favor the numerator. All quality estimates use paired outer-test RMSE ratios. Timing and memory-performance evidence is inadmissible by the frozen quality-only resource policy.

## Primary result and product guardrail

| Measure | Point | Registered interval statistic | Gate |
| --- | ---: | ---: | --- |
| A10 / ChimeraBoost | 0.942029 (-5.797%) | one-sided 95% upper 1.012091 | FAIL |
| A10 / product default | 0.866066 (-13.393%) | simultaneous max-regret 95% upper 1.046121 | FAIL |

The requested A10/M point estimate of at most 0.995 is report-only: **met**.

## Secondary CatBoost context (r0f0 only)

These contrasts are descriptive and have no advancement gate.

| Contrast | Point | Two-sided task-bootstrap 95% interval |
| --- | ---: | ---: |
| A10 / CatBoost | 1.055904 (+5.590%) | [0.951335, 1.186517] |
| ChimeraBoost / CatBoost | 1.117737 (+11.774%) | [0.972556, 1.414322] |
| product default / CatBoost | 1.215362 (+21.536%) | [1.033966, 1.518023] |

## A10 / ChimeraBoost by task

| Task | Dataset | Ratio | Change | Split W/L/T | Worst fold |
| ---: | --- | ---: | ---: | ---: | --- |
| 361236 | auction_verification | 0.767065 | -23.294% | 2/1/0 | f1 (1.127203) |
| 361251 | grid_stability | 1.023511 | +2.351% | 0/3/0 | f0 (1.044323) |
| 361252 | video_transcoding | 0.813095 | -18.691% | 3/0/0 | f2 (0.909034) |
| 361258 | kin8nm | 1.050955 | +5.096% | 0/3/0 | f0 (1.069694) |
| 361268 | fps_benchmark | 0.808571 | -19.143% | 2/1/0 | f0 (1.106377) |
| 361269 | health_insurance | 0.999591 | -0.041% | 2/1/0 | f0 (1.000099) |
| 361619 | student_performance_por | 1.004079 | +0.408% | 1/2/0 | f0 (1.030868) |
| 361622 | cars | 1.010596 | +1.060% | 1/2/0 | f0 (1.024583) |
| 361623 | space_ga | 1.061799 | +6.180% | 0/3/0 | f0 (1.067787) |

## Product-default task flags

- Task 361619 (`student_performance_por`): A10/default = 1.018036, above 1.01.

## Comparator stop-state qualification

210 ChimeraBoost/CatBoost child stop reasons were semantically unresolved by their official adapters. Direct callback instrumentation proves the time callback did not fire, and CatBoost's memory callback was ineligible. The unresolved label therefore distinguishes early stopping from iteration/no-split termination and does not weaken budget integrity.

## Terminal decision

Decision: **confirmation_not_established_clean_stop**.

stop regardless of outcome; do not add folds, tune, open the lockbox, or change a preset/default
