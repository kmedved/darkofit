# TabArena scalar-regression cap-horizon result

_Executed and analyzed on 2026-07-13 from clean DarkoFit commit `87aaf907`.
The frozen protocol is in
[`tabarena_regression_cap_horizon_protocol.md`](tabarena_regression_cap_horizon_protocol.md)._

## Decision

**Retain the 1,000-round horizon.** The 10,000-round arm was directionally
better, but it did not pass the predeclared promotion gate. It improved the
equal-dataset geometric-mean test RMSE by 0.453%, just short of the required
0.5%, while increasing equal-dataset training time by 12.88% and measured
inference time by 10.65%.

This is not a null result. The longer horizon materially improved
Physiochemical Protein and Superconductivity, and its hierarchical-bootstrap
upper bound stayed below parity. It nevertheless does not support increasing
the global default: 13.5% of short-arm children were capped at 1,000 and their
paired long-arm children executed beyond it, while five datasets were
cap-inactive and unchanged between arms. Airfoil improved by only 0.115%.
All eight cap-active datasets did improve, so a later adaptive or explicitly
opt-in horizon policy remains evidence-supported research. Subsequent mode,
target-statistic, representation, and linear-residual experiments keep 1,000
rounds fixed.

## Primary result

Ratios below one favor 10,000 rounds.

| Measure | 10,000 / 1,000 | Change |
| --- | ---: | ---: |
| Test RMSE, equal dataset | 0.995471 | -0.453% |
| Validation RMSE, equal dataset | 0.996232 | -0.377% |
| Training time, equal dataset | 1.128789 | +12.879% |
| Inference time, equal dataset | 1.106479 | +10.648% |
| Peak memory, equal dataset | 1.001786 | +0.179% |

- Hierarchical one-sided 95% upper bound: **0.999747**.
- Hierarchical two-sided 95% interval: **[0.988131, 0.999817]**.
- Dataset-level t-interval sensitivity: **[0.987969, 1.003030]**.
- Dataset wins/losses/ties: **8/0/5**; one-sided sign-test
  `p = 0.00390625`.
- Split wins/losses/ties: **63/2/157**.
- Worst split: Concrete `r6f0`, ratio **1.000058** (+0.0058%).

## Per-dataset test RMSE

| Dataset | 10,000 / 1,000 | Change | Repeat W/L/T |
| --- | ---: | ---: | ---: |
| Used Fiat 500 | 1.000000 | 0.000% | 0/0/10 |
| Food Delivery Time | 1.000000 | 0.000% | 0/0/3 |
| QSAR-TID-11 | 0.999285 | -0.072% | 3/0/0 |
| QSAR fish toxicity | 1.000000 | 0.000% | 0/0/10 |
| Airfoil self noise | 0.998851 | -0.115% | 8/1/1 |
| Concrete compressive strength | 0.999903 | -0.010% | 6/0/4 |
| Diamonds | 1.000000 | 0.000% | 0/0/3 |
| Healthcare insurance expenses | 1.000000 | 0.000% | 0/0/10 |
| Houses | 0.999157 | -0.084% | 3/0/0 |
| Miami housing | 0.999893 | -0.011% | 2/0/1 |
| Physiochemical Protein | 0.956000 | **-4.400%** | 3/0/0 |
| Superconductivity | 0.989070 | **-1.093%** | 3/0/0 |
| Wine Quality | 0.999890 | -0.011% | 2/0/1 |

No dataset met the frozen conditional-harm rule, and no dataset point estimate
regressed.

## Mechanism and integrity

- **444/444** outer jobs succeeded; no cache substitution, imputation,
  missing results, duplicates, or failures.
- **3,552/3,552** child-fit metadata records were complete and
  provenance-matched.
- The 1,000-round arm stopped 240 of 1,776 children at the iteration limit.
  Every paired long-arm child continued beyond 1,000 rounds, and all eight
  datasets containing capped children improved at the dataset level.
- The 10,000-round arm early-stopped all 1,776 children; its maximum completed
  round was 3,938. Neither arm hit the wall-clock deadline.
- All children resolved to learning rate 0.1, CatBoost tree mode, the boosting
  lane, 18 CPUs, and zero GPUs.
- The warmup used the same 18-thread allocation as every measured child.

## Frozen gate outcome

Four promotion requirements failed:

| Failed gate | Observed | Required |
| --- | ---: | ---: |
| Equal-dataset test RMSE | -0.453% | at least -0.500% |
| Dataset wins | 8 of 13 | at least 10 of 13 |
| Paired long-arm children over 1,000 rounds | 13.5% | at least 20.0% |
| Inference-time ratio | 1.1065 | at most 1.1000 |

All provenance, uncertainty, harm, validation, wall-clock, training-time, and
memory gates passed. Frozen gates are conjunctive, so the longer horizon does
not advance.

## Retained evidence

The repository retains the analyzer's machine-readable
[`summary`](tabarena_regression_cap_horizon_summary.json),
[`paired splits`](tabarena_regression_cap_horizon_paired_splits.csv),
[`per-repeat estimates`](tabarena_regression_cap_horizon_per_repeat.csv),
[`paired child metadata`](tabarena_regression_cap_horizon_paired_children.csv),
[`run manifest`](tabarena_regression_cap_horizon_run_manifest.json),
[`completion attestation`](tabarena_regression_cap_horizon_completion_attestation.json),
and [`warmup record`](tabarena_regression_cap_horizon_warmup_history.json).
The 8.8 MB safe analysis payload and 444 raw result pickles remain in the
hash-addressed local campaign directory; their hashes and sizes are bound by
the committed attestation.

## Provenance

- DarkoFit commit: `87aaf907829507a2e170ebc2628222d66ba6a30c`.
- Python: 3.12.13; AutoGluon: 1.5.1b20260712; TabArena source commit:
  `4cd1d2526874962daae048a6f2dcf34aa272f3fa`.
- Run manifest SHA-256:
  `bc9e2023c46627cbd63bb50a4aa21e0e2722bbae00484e74f749f638fe67531d`.
- Completion attestation SHA-256:
  `95a1fec286f6f91f819f52e15b7f8f4defc3f81740829e5f57e299a87140ecda`.
- Safe analysis payload SHA-256:
  `e5d4465c1f924b23910d52e3bc1f7d45724591dde18c1a530b1e512e07d550b5`.
- Analyzer summary SHA-256:
  `f92f896200e0beb02794957a29d5da5172f0ecd9c76a37e98267acdc3e3a0741`.
- Paired-split table SHA-256:
  `b42403e5fa7a6d4a7fdc9d5de7353be5672e0556f8422d11feac660352fdea38`.
- Paired-child table SHA-256:
  `a0ef84481438751f5abc855bdfb0e8aa97bd8c94e958ac292bcfb9e346b71e3f`.

The analyzer revalidated the complete attested campaign, executing source,
dependency lock, runtime, hardware, and result hashes immediately before
atomically publishing the decision files.
