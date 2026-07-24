# DarkoFit v0.12 release compute-ladder result

**Status:** complete descriptive release evidence. This scoreboard is not a
tuning set or a shipping gate.

## Verdict

DarkoFit did **not** achieve strict Pareto dominance over ChimeraBoost's
current release. On this fixed historical regression slice, ChimeraBoost
v0.23.0 owns both measured quality-versus-fit-time and
quality-versus-prediction-time frontiers.

The result is not close at the product level. DarkoFit's default was 0.97%
worse in aggregate RMSE while taking 2.60x the fit time and 3.27x the
prediction time per call. DarkoFit's accuracy point was 1.19% better than
ChimeraBoost's depth-10 point on the aggregate estimate, but its interval
crossed parity and it was slower on both cost axes. At ensemble8,
ChimeraBoost was 3.63% better, fit 3.57x faster, and predicted 1.82x faster.
DarkoFit retained a large ensemble peak-RSS advantage.

This is the intended whole-curve interpretation: an isolated quality or
memory advantage cannot override losses elsewhere on the frontier.

## Scope and execution

- DarkoFit: public `v0.12.0`, commit
  `a9eb4dbbf8af0e6db42e9ace433e7a267c80fca7`.
- ChimeraBoost: public `v0.23.0`, commit
  `6667843b8970454b0f582ffd1ab2be033989c578`, verified as the latest
  upstream release before worker zero and again at result close.
- Data: the 13 fixed historical M2 regression datasets, with three registered
  `(repeat, fold)` coordinates per dataset. These are development data, not
  fresh confirmation or independent random datasets.
- Arms: direct public estimators at default, accuracy-oriented, and
  eight-member-ensemble compute points for each library.
- Resources: one 14-core Apple-silicon host, zero GPUs, fresh sequential
  worker processes, a common 14-thread ceiling, and an exclusive-machine
  preflight.
- Measurements: test RMSE, fit wall time, warmed repeated prediction time on
  the registered test batch, process-tree peak RSS, and fitted-policy
  metadata.
- Aggregation: equal-dataset geometric means after averaging each dataset's
  three coordinate log ratios. The fixed-seed intervals resample the three
  coordinates within every fixed dataset and do not imply 13 independent
  datasets.

All 234 workers completed once. The model measurement was not rerun.

## Public compute points

All ratios below use pinned ChimeraBoost default (`M0`) as denominator; lower
is better. `D0`, `DA`, and `D8` are DarkoFit default, accuracy, and ensemble8.
`MA` and `M8` are ChimeraBoost depth-10 and ensemble8.

| Arm | Quality [95%] | Fit | Predict/call | Peak RSS | Dataset W-L-T |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 | 1.0097x [1.0032, 1.0159] | 2.6016x | 3.2710x | 1.0220x | 8-5-0 |
| DA | 1.0038x [0.9935, 1.0154] | 1.8260x | 3.6091x | 1.0116x | 6-7-0 |
| D8 | 0.9996x [0.9937, 1.0055] | 2.0192x | 9.2919x | 2.6139x | 10-3-0 |
| M0 | 1.0000x [1.0000, 1.0000] | 1.0000x | 1.0000x | 1.0000x | 0-0-13 |
| MA | 1.0159x [1.0082, 1.0228] | 1.4607x | 1.0789x | 1.1030x | 5-8-0 |
| M8 | 0.9646x [0.9595, 0.9698] | 0.5655x | 5.0914x | 6.6886x | 13-0-0 |

## Matched public profiles

| Contrast | Quality [95%] | Fit | Predict/call | Peak RSS | Dataset W-L-T |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 / M0 | 1.0097x [1.0032, 1.0159] | 2.6016x | 3.2710x | 1.0220x | 8-5-0 |
| DA / MA | 0.9881x [0.9746, 1.0037] | 1.2500x | 3.3451x | 0.9171x | 6-7-0 |
| D8 / M8 | 1.0363x [1.0338, 1.0388] | 3.5708x | 1.8250x | 0.3908x | 2-11-0 |

The main product readout is:

- ChimeraBoost's ensemble improved aggregate RMSE by 3.54% over its default,
  won all 13 datasets, and fit in 0.565x its default's time.
- DarkoFit's ensemble was effectively tied with ChimeraBoost's default,
  improved on it on 10/13 datasets, and fit in 2.019x its time.
- B3 narrowed DarkoFit's matched ensemble-fit deficit from the historical
  v0.11/v0.20 readout of 6.10x to 3.57x here. Because the rival version also
  changed, that cross-release difference is descriptive rather than a clean
  causal estimate of B3.
- ChimeraBoost's ensemble-side prediction improvements reversed DarkoFit's
  historical matched-profile prediction lead: D8/M8 moved from 0.366x in the
  v0.11/v0.20 run to 1.825x here. Again, the version boundary changed.
- Peak RSS favored DarkoFit at accuracy and ensemble points, but D0 used 2.2%
  more peak RSS than M0. The strict matched-profile memory condition therefore
  also failed.

## Frontier decision

| Axis | Comparable budgets | DarkoFit no worse | DarkoFit strictly better | Full-curve dominance |
| --- | ---: | ---: | ---: | ---: |
| Fit time | 2 | 0 | 0 | no |
| Prediction time/call | 4 | 0 | 0 | no |

**Strict Pareto victory: no.**

## Integrity and analyzer repair

- 234/234 workers completed with return code zero; there were no extra stdout
  records and no partial or resumed worker set.
- Every worker entered and left fitting with an ambient Numba mask of 14.
- Process-tree RSS sampling reported zero errors and at least five samples for
  every worker.
- Every formal prediction interval exceeded the 0.5-second floor; the minimum
  was 0.561 seconds.
- Source worktrees were clean and exact-pin validated before every worker and
  at completion.
- All workers emitted Intel OpenMP's known `omp_set_nested` deprecation
  message. The 39 ChimeraBoost ensemble workers also captured the public
  bagged-member-default warning. Neither was a worker failure.

The first analyzer verified all 234 raw worker artifacts, then stopped before
writing any metric output because a reused historical summarizer requested the
retired name `CONTRACT_ID`. Commit `d4b9af4` supplied that compatibility name
through an isolated analyzer view. The unchanged raw measurement was analyzed
once with the repaired code. The attestation retains both analyzer hashes.

## Artifacts and hashes

- Protocol:
  [`v012_compute_ladder_protocol_20260724.md`](v012_compute_ladder_protocol_20260724.md),
  SHA-256 `5b885d0ab36c5c1a7545615f9aad3cae9c714758b67157f7e45359431f3d7dfc`.
- Runner SHA-256:
  `af78f19d7bf95ad1799874936ad3f5ee11f7ceca516729cad58b230011cf9a3d`.
- Planned analyzer SHA-256:
  `0cc073c3cb9493a6f6a32b2e2be85d942c318f56a4535ec2b8f720efa49cdbb2`.
- Executed analyzer SHA-256:
  `1d958ff2ba26567f3c71f8f7066664b9c3d8aaa5521e1dcbcd2cd8f0ed9d51e6`.
- Raw:
  [`v012_compute_ladder_20260724_raw.json`](v012_compute_ladder_20260724_raw.json),
  SHA-256 `404692f6f89d517bfeb470127267e3b18857a1c3e8bb12acf9cda6fcf9984809`.
- Manifest:
  [`v012_compute_ladder_20260724_manifest.json`](v012_compute_ladder_20260724_manifest.json),
  SHA-256 `3ddc1a8b1a2a0fab80afd0a5d6d7e7e895230d8a731a2e391c6b48965330f63b`.
- Summary:
  [`v012_compute_ladder_20260724_result.json`](v012_compute_ladder_20260724_result.json),
  SHA-256 `d2c1d814f549f7a5b12aef241a8670f20dedc489a413e4762a7c82b3ccc347f2`.
- Coordinate ratios:
  [`v012_compute_ladder_20260724_coordinate_ratios.csv`](v012_compute_ladder_20260724_coordinate_ratios.csv),
  SHA-256 `5d03c3c73ffb6d11fe8de8e64f2cec511b29bf4689d94ce976c7a17ea2b07b34`.
- Per-dataset table:
  [`v012_compute_ladder_20260724_per_dataset.csv`](v012_compute_ladder_20260724_per_dataset.csv),
  SHA-256 `4429bd1043ef11b7fbf755e822a61069489b50a177d92fc6565db40ee09e4e48`.
- Analyzer-generated report:
  [`v012_compute_ladder_20260724_generated_report.md`](v012_compute_ladder_20260724_generated_report.md),
  SHA-256 `5c40287f6a85f3ac535951711facb9478b1cb5c30ecd5a9ae3257263ece1009a`.
- Completion terminal:
  [`v012_compute_ladder_20260724_terminal.json`](v012_compute_ladder_20260724_terminal.json),
  SHA-256 `d54580ce7453ab4e8b572947722c5ab81c0f360d5c09e748985f0d395ef19037`.
- Analysis attestation:
  [`v012_compute_ladder_20260724_analysis_attestation.json`](v012_compute_ladder_20260724_analysis_attestation.json),
  SHA-256 `133ae800ac0a95479bc9e2940c8a92fec192fe8c524f9797a3051b9e171a6852`.

## Limitations and next action

This fixed historical slice covers regression only. It does not cover
classification, CatBoost, fresh confirmation, CTR23, the newest sports season,
or TabArena placement. Timing and RSS are scoped to one Apple-silicon host and
the registered test-batch shapes.

The scoreboard is complete. It makes the next priority unambiguous:
ChimeraBoost's ensemble quality and fit speed are the dominant product gaps,
while prediction is now also a measured loss at every comparable budget.
Q1 remains the already funded next speed-mechanism slot, but it should be
treated as one bounded engine improvement rather than an answer to the
ensemble-quality deficit.
