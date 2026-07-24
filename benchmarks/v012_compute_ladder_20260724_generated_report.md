# DarkoFit v0.12 release compute-ladder scoreboard

Status: **descriptive release scoreboard; not a tuning or shipping gate.**

All ratios are numerator / pinned ChimeraBoost v0.23 default unless the
table says matched profile; lower is better. Point estimates equally
weight the 13 fixed historical regression datasets after averaging each
dataset's three registered split log ratios. Intervals resample those
three coordinates within each fixed dataset; they do not imply 13
independent datasets.

## Public compute points

| Arm | Quality [95%] | Fit | Predict/call | Peak RSS | Peak-start RSS | Dataset W-L-T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 (default) | 1.0097x [1.0032, 1.0159] | 2.6016x | 3.2710x | 1.0220x | 1.3406x | 8-5-0 |
| DA (accuracy) | 1.0038x [0.9935, 1.0154] | 1.8260x | 3.6091x | 1.0116x | 1.1124x | 6-7-0 |
| D8 (ensemble) | 0.9996x [0.9937, 1.0055] | 2.0192x | 9.2919x | 2.6139x | 13.4379x | 10-3-0 |
| M0 (default) | 1.0000x [1.0000, 1.0000] | 1.0000x | 1.0000x | 1.0000x | 1.0000x | 0-0-13 |
| MA (accuracy) | 1.0159x [1.0082, 1.0228] | 1.4607x | 1.0789x | 1.1030x | 2.8962x | 5-8-0 |
| M8 (ensemble) | 0.9646x [0.9595, 0.9698] | 0.5655x | 5.0914x | 6.6886x | 8.1411x | 13-0-0 |

## Matched public profiles

| Contrast | Quality [95%] | Fit | Predict/call | Peak RSS | Dataset W-L-T |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0/M0 | 1.0097x [1.0032, 1.0159] | 2.6016x | 3.2710x | 1.0220x | 8-5-0 |
| DA/MA | 0.9881x [0.9746, 1.0037] | 1.2500x | 3.3451x | 0.9171x | 6-7-0 |
| D8/M8 | 1.0363x [1.0338, 1.0388] | 3.5708x | 1.8250x | 0.3908x | 2-11-0 |

## Stepwise frontier verdicts

| Axis | Comparable budgets | DarkoFit no worse | DarkoFit strictly better | Full-curve dominance |
| --- | ---: | ---: | ---: | ---: |
| fit_seconds | 2 | 0 | 0 | no |
| prediction_seconds_per_call | 4 | 0 | 0 | no |

The strict program target requires fit-frontier dominance, prediction-
frontier dominance, and no worse peak RSS at all three matched public
profiles. The verdict below is the fixed-protocol point-estimate readout;
the paired intervals and complete coordinate rows remain part of the
evidence and prevent it from being read as a certificate.

**Strict Pareto victory: NO.**

This result covers regression on the fixed historical M2 task set. It
does not cover classification, CatBoost, fresh confirmation, lockbox
evidence, or TabArena placement.

## Analysis rerun note

The initially published analyzer verified all raw worker artifacts but stopped before writing any metric output because its reused summarizer looked for the retired compatibility name `CONTRACT_ID`. This analyzer fix supplies that name through an isolated view. The completed 234-worker measurement was not rerun or changed; both analyzer hashes are retained in the summary and attestation.
