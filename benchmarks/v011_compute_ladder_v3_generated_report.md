# DarkoFit v0.11 release compute-ladder scoreboard

Status: **spent, descriptive release evidence; no policy advancement is authorized.**

All ratios are numerator / pinned ChimeraBoost v0.20 default unless the
table says matched profile; lower is better. Point estimates equally
weight the 13 fixed historical regression datasets after averaging each
dataset's three registered split log ratios. Intervals resample those
three coordinates within each fixed dataset; they do not imply 13
independent datasets.

## Public compute points

| Arm | Quality [95%] | Fit | Predict/call | Peak RSS | Peak-start RSS | Dataset W-L-T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 (default) | 1.0145x [1.0080, 1.0207] | 1.3796x | 2.2127x | 0.9739x | 0.7839x | 7-6-0 |
| DA (accuracy) | 1.0038x [0.9934, 1.0153] | 1.8642x | 2.6857x | 1.0117x | 1.1915x | 6-7-0 |
| D8 (ensemble) | 0.9996x [0.9937, 1.0054] | 3.4447x | 5.5602x | 1.0740x | 2.1758x | 10-3-0 |
| M0 (default) | 1.0000x [1.0000, 1.0000] | 1.0000x | 1.0000x | 1.0000x | 1.0000x | 0-0-13 |
| MA (accuracy) | 1.0159x [1.0085, 1.0229] | 1.4319x | 1.1088x | 1.1061x | 3.0214x | 5-8-0 |
| M8 (ensemble) | 0.9646x [0.9596, 0.9697] | 0.5644x | 15.1783x | 6.6888x | 8.4900x | 13-0-0 |

## Matched public profiles

| Contrast | Quality [95%] | Fit | Predict/call | Peak RSS | Dataset W-L-T |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0/M0 | 1.0145x [1.0080, 1.0207] | 1.3796x | 2.2127x | 0.9739x | 7-6-0 |
| DA/MA | 0.9881x [0.9747, 1.0035] | 1.3019x | 2.4221x | 0.9146x | 6-7-0 |
| D8/M8 | 1.0363x [1.0338, 1.0387] | 6.1036x | 0.3663x | 0.1606x | 2-11-0 |

## Stepwise frontier verdicts

| Axis | Comparable budgets | DarkoFit no worse | DarkoFit strictly better | Full-curve dominance |
| --- | ---: | ---: | ---: | ---: |
| fit_seconds | 3 | 0 | 0 | no |
| prediction_seconds_per_call | 4 | 1 | 1 | no |

The strict program target requires fit-frontier dominance, prediction-
frontier dominance, and no worse peak RSS at all three matched public
profiles. The verdict below is the predeclared point-estimate readout;
the paired intervals and complete coordinate rows remain part of the
evidence and prevent it from being read as a certificate.

**Strict Pareto victory: NO.**

This result covers regression on the fixed historical M2 task set. It
does not cover classification, CatBoost, fresh confirmation, lockbox
evidence, or TabArena placement.
