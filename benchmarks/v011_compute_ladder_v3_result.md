# DarkoFit v0.11 release compute-ladder result (v3)

**Status:** complete, spent descriptive release evidence. This record does not
authorize a policy, default, release, or benchmark claim beyond the fixed scope
below.

## Verdict

DarkoFit did **not** achieve the predeclared strict Pareto victory against the
latest ChimeraBoost release at execution time. ChimeraBoost v0.20.0 owns the
measured quality-versus-fit-time frontier, principally because its public
eight-member ensemble improved on its default quality while fitting faster.
DarkoFit retained the matched-profile peak-memory advantage and its ensemble
predicted substantially faster than ChimeraBoost's ensemble, but neither fact
overrides a loss elsewhere on the curve.

This is the intended whole-curve interpretation: a win at one compute point or
on one resource axis cannot excuse a loss at another.

## Frozen scope and execution

- DarkoFit: public `v0.11.0`, commit
  `0b820e332cec2c083b1dd89eef0fe306d69cfc0e`.
- ChimeraBoost: public `v0.20.0`, commit
  `7d48e053e5bd3c7aded1126871aeb0f1f6b84c46`, still the latest upstream
  release when worker zero started and when this result was closed.
- Data: the 13 fixed historical M2 regression datasets, with three registered
  `(repeat, fold)` coordinates per dataset. These are spent development data,
  not fresh confirmation or a random sample of independent datasets.
- Arms: direct public estimators at default, accuracy-oriented, and
  eight-member-ensemble compute points for each library. No AutoGluon outer
  bagging was used.
- Resources: one 14-core Apple-silicon host, zero GPUs, fresh sequential worker
  processes, common 14-thread ceiling, and an exclusive-machine preflight.
- Measurements: test RMSE, fit wall time, warmed repeated prediction time on
  the registered test batch, process-tree peak RSS, and fitted-policy metadata.
- Aggregation: equal-dataset geometric means after averaging each dataset's
  three coordinate log ratios; 10,000 fixed-seed bootstrap draws resampled the
  coordinates within every fixed dataset.

V1 stopped before any worker because its exclusivity check incorrectly treated
the launch shell as a conflicting benchmark process. V2 stopped before any
worker because the same check treated the `caffeinate` wrapper as a conflicting
sibling supervisor. Both identities were terminally closed without fitting.
V3 changed only the execution topology: a standalone `caffeinate` process and
an ordinary benchmark parent. The scientific protocol, pins, grid, and
decision rule stayed frozen. V3 completed all 234 workers once, with no resume
or favorable rerun.

## Public compute points

All ratios below use pinned ChimeraBoost default (`M0`) as the denominator;
lower is better. `D0`, `DA`, and `D8` are DarkoFit default, accuracy, and
ensemble8. `MA` and `M8` are ChimeraBoost depth-10 and ensemble8.

| Arm | Quality [95%] | Fit | Predict/call | Peak RSS | Dataset W-L-T |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 | 1.0145x [1.0080, 1.0207] | 1.3796x | 2.2127x | 0.9739x | 7-6-0 |
| DA | 1.0038x [0.9934, 1.0153] | 1.8642x | 2.6857x | 1.0117x | 6-7-0 |
| D8 | 0.9996x [0.9937, 1.0054] | 3.4447x | 5.5602x | 1.0740x | 10-3-0 |
| M0 | 1.0000x [1.0000, 1.0000] | 1.0000x | 1.0000x | 1.0000x | 0-0-13 |
| MA | 1.0159x [1.0085, 1.0229] | 1.4319x | 1.1088x | 1.1061x | 5-8-0 |
| M8 | 0.9646x [0.9596, 0.9697] | 0.5644x | 15.1783x | 6.6888x | 13-0-0 |

## Matched public profiles

| Contrast | Quality [95%] | Fit | Predict/call | Peak RSS | Dataset W-L-T |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 / M0 | 1.0145x [1.0080, 1.0207] | 1.3796x | 2.2127x | 0.9739x | 7-6-0 |
| DA / MA | 0.9881x [0.9747, 1.0035] | 1.3019x | 2.4221x | 0.9146x | 6-7-0 |
| D8 / M8 | 1.0363x [1.0338, 1.0387] | 6.1036x | 0.3663x | 0.1606x | 2-11-0 |

The important product-level readout is not a single aggregate score:

- ChimeraBoost `M8` was about 3.5% better than `M0` while taking 0.56x its
  fit time. It beat `M0` on all 13 fixed datasets.
- DarkoFit `D8` was essentially tied with `M0` on the aggregate point estimate,
  but was 3.63% worse than `M8` and took 6.10x `M8`'s fit time.
- DarkoFit's accuracy point beat ChimeraBoost's accuracy point by 1.19% on the
  aggregate estimate, with an interval that crossed parity, while taking 1.30x
  the fit time.
- DarkoFit retained lower matched-profile peak RSS at all three compute points.
  At ensemble8 it used 0.161x ChimeraBoost's peak RSS and 0.366x its prediction
  time per call. ChimeraBoost's ensemble therefore traded much higher memory
  and slower inference for clearly better quality and faster training.

## Predeclared frontier decision

| Axis | Comparable budgets | DarkoFit no worse | DarkoFit strictly better | Full-curve dominance |
| --- | ---: | ---: | ---: | ---: |
| Fit time | 3 | 0 | 0 | no |
| Prediction time/call | 4 | 1 | 1 | no |

The strict decision required fit-frontier dominance, prediction-frontier
dominance, and no worse matched-profile peak RSS. Peak RSS passed; both
frontier conditions failed. **Strict Pareto victory: no.**

## Post-run integrity audit

- 234/234 workers completed with return code zero; there were no extra stdout
  records and no partial or resumed worker set.
- Every worker entered and left fitting with an ambient Numba mask of 14.
  DarkoFit member kernels used their fitted thread count; ChimeraBoost's
  parallel ensemble members each used one thread as declared.
- Process-tree RSS sampling reported zero errors and at least five samples for
  every worker.
- Every prediction interval cleared the frozen 0.5-second floor. The minimum
  was 0.598 seconds; intervals were sized from three pilots, so some completed
  below the one-second target when the formal loop ran faster than its pilots.
- All 234 workers emitted Intel OpenMP's known `omp_set_nested` deprecation
  message. The 39 ChimeraBoost ensemble workers also emitted the public
  bagged-mode-default warning (captured twice by the library warning path).
  Neither message represented a worker or integrity failure.
- Source worktrees were clean and exact-pin validated before every worker and
  again at completion. The latest-release check still resolved to v0.20.0 when
  this result was closed.

## Artifacts and hashes

- Frozen contract:
  [`v011_compute_ladder_contract_v3_20260722.json`](v011_compute_ladder_contract_v3_20260722.json),
  SHA-256 `61e788f06b88eefcc2e3c08a38402bf93246e7334980a77061b46763650b581a`.
- Protocol SHA-256:
  `2b48ebe91ffe8586cad69c1abecafc14fc01dcb895c346c97d78a166c20a5e23`.
- Runner SHA-256:
  `db5b47af68fa0d74458c9d48d0c441caee8621cf1922542df2a27668118d14fb`.
- Analyzer SHA-256:
  `d65c84b16c1f43499687771ddb07e9f6dc23a5a1af09ba177f520733f05abf9b`.
- Raw:
  [`v011_compute_ladder_v3_raw.json`](v011_compute_ladder_v3_raw.json),
  SHA-256 `96f594da1a0ea885aa55d45636049d97b9b6e1a7f56d85679dfe879420636f79`.
- Manifest:
  [`v011_compute_ladder_v3_manifest.json`](v011_compute_ladder_v3_manifest.json),
  SHA-256 `01fbb053d1390c43758adc4f47da38e39b6beb53be26ed13548a5eb399d485d4`.
- Summary:
  [`v011_compute_ladder_v3_result.json`](v011_compute_ladder_v3_result.json),
  SHA-256 `28c904e4585d343d96366bf998edd39034795ab18f092a8765b0efe7049543d6`.
- Coordinate ratios:
  [`v011_compute_ladder_v3_coordinate_ratios.csv`](v011_compute_ladder_v3_coordinate_ratios.csv),
  SHA-256 `8887ba02c2ae2189907e4afc3064d3c262030432a776ccd9597985088f2d35df`.
- Per-dataset table:
  [`v011_compute_ladder_v3_per_dataset.csv`](v011_compute_ladder_v3_per_dataset.csv),
  SHA-256 `546592592a3a70720fa214245451374982f2e17f783341c07fbe03b97682dd10`.
- Analyzer-generated report:
  [`v011_compute_ladder_v3_generated_report.md`](v011_compute_ladder_v3_generated_report.md),
  SHA-256 `e23fc2d6b32a3cf22373227ce0a7bcd3604a4b9fb21383c40ac659af0362db11`.
- Completion terminal:
  [`v011_compute_ladder_v3_terminal.json`](v011_compute_ladder_v3_terminal.json),
  SHA-256 `212f15fc8a97c85dab57abcc8c6f6d9951272d3c4804825bdd8e3e6c8655045e`.
- Analysis attestation:
  [`v011_compute_ladder_v3_analysis_attestation.json`](v011_compute_ladder_v3_analysis_attestation.json),
  SHA-256 `cf8561717022d01e32ad9c35e94643b77b1e2c303e5bc2a7202376caaa308cbc`.

## Limitations and next action

This fixed historical regression slice does not include classification,
CatBoost, fresh confirmation, lockbox evidence, or TabArena placement. Timing
and RSS are scoped to one Apple-silicon host and the registered test-batch
shapes. The intervals describe coordinate sensitivity within the 13 fixed
datasets; they do not turn those datasets into independent random draws.

The scoreboard is terminal and spent. Its failure does not authorize tuning on
these outcomes. Per the binding plan, the next action is the Phase F historical
kill-rule audit, followed by the separately scoped automatic smooth-data
selector campaign.
