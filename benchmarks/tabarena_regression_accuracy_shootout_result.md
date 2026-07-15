# TabArena regression accuracy-shootout result

_Executed and analyzed on 2026-07-15 from clean DarkoFit commit `54b5e99`.
The source-frozen development design is in
[`tabarena_regression_accuracy_shootout_protocol.md`](tabarena_regression_accuracy_shootout_protocol.md)._

## Decision

**Freeze the iteration-1 `A10` accuracy profile unchanged for external
confirmation.** It passed every preregistered development gate on the spent
13-dataset panel. No linear-residual iteration 2 is permitted by the decision
tree, and this result does not change the product default.

`A10` reached development parity with ChimeraBoost 0.14.1: equal-dataset test
RMSE was 2.438% lower. The magnitude is concentrated in Diamonds, however.
Without Diamonds the advantage is only 0.331%, the remaining datasets split
6-6, and one of the three repeat blocks reverses by 0.020%. The defensible
conclusion is therefore **parity with a slight development-panel edge**, not
established superiority.

CatBoost 1.2.10 remains ahead. A descriptive comparison from the authenticated
reused `C` columns puts `A10` 1.543% worse on test RMSE, with only two of 13
dataset wins. Every leave-one-dataset-out estimate still favors CatBoost.
`A10` is the strongest DarkoFit profile measured here, but it does not yet
beat every comparator.

This panel is spent development data: its tasks and outer coordinates had
already informed the candidate. Independent confirmation is required before
any external parity claim or product promotion.

## Scoreboard

Ratios were formed on each shared outer coordinate, reduced geometrically
within dataset, and then geometrically across 13 equally weighted datasets.
Negative percentages favor the numerator.

| Contrast | Test RMSE | Validation RMSE | Test dataset W/L/T | Test split W/L/T |
| --- | ---: | ---: | ---: | ---: |
| `A10 / ChimeraBoost` | **-2.438%** | -2.744% | 7/6/0 | 23/16/0 |
| `A10 / product default` | **-3.645%** | -3.881% | 9/4/0 | 30/9/0 |
| `A10 / B10` | **-2.780%** | -3.250% | 11/2/0 | 32/6/1 |
| `B10 / product default` | -0.889% | -0.652% | 9/4/0 | 26/13/0 |
| `A10 / CatBoost` (descriptive) | **+1.543%** | +1.476% | 2/11/0 | 9/30/0 |

The first four contrasts are source-frozen primary/attribution contrasts. The
CatBoost row is a post-gate descriptive recomputation from the exact reused
`C` observations already authenticated and exported in `paired_splits.csv`;
it did not influence the frozen decision.

Relative to CatBoost's equal-dataset test RMSE:

| Arm | Ratio to CatBoost | Gap |
| --- | ---: | ---: |
| CatBoost 1.2.10 | 1.000000 | baseline |
| `A10` | 1.015427 | +1.543% |
| ChimeraBoost 0.14.1 | 1.040800 | +4.080% |
| `B10` | 1.044468 | +4.447% |
| DarkoFit product default | 1.053834 | +5.383% |

Excluding Diamonds preserves that ordering: `A10` is 1.448% worse than
CatBoost, 0.331% better than ChimeraBoost, 0.907% better than `B10`, and
1.815% better than the product default.

## Frozen gates

| Gate | Observed | Limit | Result |
| --- | ---: | ---: | --- |
| Complete paired grid | 39 coordinates / 312 child pairs | exact | pass |
| Failed, imputed, deadline, or time-limit results | 0 | 0 | pass |
| Equal-dataset `A10 / M` | 0.975622 | <= 1.000 | pass |
| Worst dataset `A10 / P` | 1.008605 (Concrete) | <= 1.020 | pass |
| Worst leave-one-dataset-out `A10 / M` | 0.996685 (omit Diamonds) | <= 1.010 | pass |

All 13 leave-one-dataset-out `A10 / M` estimates pass, ranging from 0.972355
to 0.996685. Diamonds supplies 87.6% of the full mean-log advantage, so the
worst LODO result is the most informative robustness check.

## Per-dataset quality

Negative percentages favor `A10`. `C` is external CatBoost 1.2.10, not
DarkoFit's `tree_mode="catboost"`.

| Dataset | vs `B10` | vs product | vs ChimeraBoost | vs CatBoost |
| --- | ---: | ---: | ---: | ---: |
| Airfoil self noise | -5.14% | -7.52% | -1.53% | +8.56% |
| Used Fiat 500 | -0.55% | -0.90% | -0.95% | +0.30% |
| Concrete strength | +0.03% | +0.86% | -2.03% | +0.33% |
| Diamonds | -22.68% | -23.11% | -24.50% | +2.69% |
| Food delivery | -0.03% | +0.18% | +0.05% | +0.50% |
| Healthcare expenses | -0.53% | -0.60% | -0.75% | +0.35% |
| Houses | -0.24% | -0.79% | +1.57% | +2.31% |
| Miami housing | +0.25% | +0.74% | +1.30% | +3.45% |
| Physiochemical protein | -1.55% | -6.98% | +0.16% | -0.40% |
| QSAR-TID-11 | -1.29% | -2.52% | -1.60% | -0.51% |
| QSAR fish toxicity | -0.15% | +0.05% | +0.08% | +0.97% |
| Superconductivity | -0.53% | -2.63% | +0.09% | +0.67% |
| Wine quality | -1.02% | -1.23% | -0.30% | +1.18% |

The largest remaining CatBoost gap is Airfoil (+8.56%; worst split +13.65%).
Removing that single worst split still leaves `A10` 1.364% worse overall, and
all three repeat blocks favor CatBoost. The CatBoost conclusion is not an
Airfoil-only artifact.

## Selector and horizon diagnostics

The validation-local `A10` selector made 312 child decisions:

| Selected DarkoFit mode | Children | Share |
| --- | ---: | ---: |
| `catboost` | 142 | 45.5% |
| `hybrid` | 93 | 29.8% |
| `lightgbm` | 77 | 24.7% |

All 24 Diamonds children rejected DarkoFit's CatBoost-like mode (15 hybrid,
9 LightGBM), producing the 22.68% `A10 / B10` gain there. `A10` still trailed
external CatBoost on Diamonds by 2.69%, which identifies a DarkoFit mode gap
rather than CatBoost parity.

All 624 selected children stopped through ordinary early stopping with
50-round patience; no child hit a deadline or the 10,000-round cap. `A10`'s
median best iteration was 381 and its maximum was 3,582. Forty-eight selected
`A10` children had best iterations at or above 1,000, including all 24 protein
children, so the evidence does not support restoring a 1,000-round cap. All
children selected the boosting lane; linear residuals were off as frozen.

## Operational integrity

- **78/78** new outer jobs and **624/624** selected child fits completed.
- The exact grid contains 39 paired outer coordinates and 312 paired children.
- All 936 internal `A10` candidates fitted and produced finite validation
  scores.
- There were zero failures, missing or duplicate results, imputations, worker
  restarts, recovery events, OOMs, deadline hits, or time-limit stops.
- The two persistent production workers completed all 39 reciprocal waves.
- Preflight and production lifecycle/measured windows each recorded zero
  swap-out. Swap-in occurred and was allowed by `quality_only_swap_in`.
- Timing and memory-performance evidence is **inadmissible by policy**. Raw
  resource counters are retained only for operational auditing; this result
  makes no training, inference, or memory-performance claim.
- A second analyzer execution regenerated all five decision artifacts
  byte-for-byte. An independent recomputation matched every split ratio,
  geometric reduction, win count, worst case, LODO estimate, mode count, and
  gate.

## Next gate

The complete `A10` profile is now frozen unchanged: 10,000-round cap,
`tree_mode="auto"`, candidate order CatBoost/LightGBM/hybrid, validation
selection and tie-break, `l2_leaf_reg=3`, `max_bins=128`, `learning_rate=0.1`,
native representation, `ts_permutations=1`, and linear residuals off.

Before inspecting any eligible unseen task, commit the CTR23 contamination
registry, confirmation/lockbox assignment, exact outer-coordinate subset,
bootstrap hierarchy, and thresholds. Run frozen `A10` against ChimeraBoost and
CatBoost there. If confirmation passes, the first product step should be an
explicit scalar-regression accuracy preset. A default change additionally
requires fresh classification, weighted-RMSE, other-loss, and separately
admissible operational-resource gates.

CatBoost parity needs a new preregistered research track on fresh development
data. Airfoil-like numeric-string resolution, explicit declared ordinal maps,
and selector robustness are plausible leads, but none may be promoted from
this spent panel.

## Retained evidence

The repository retains the analyzer's machine-readable
[`summary`](tabarena_regression_accuracy_shootout_summary.json),
[`paired splits`](tabarena_regression_accuracy_shootout_paired_splits.csv),
[`per-dataset estimates`](tabarena_regression_accuracy_shootout_per_dataset.csv),
[`paired child metadata`](tabarena_regression_accuracy_shootout_paired_children.csv),
[`run manifest`](tabarena_regression_accuracy_shootout_run_manifest.json),
[`completion attestation`](tabarena_regression_accuracy_shootout_completion_attestation.json),
and [`warmup record`](tabarena_regression_accuracy_shootout_warmup_history.json).
The 4.0 MB safe analysis payload, detailed preflight/concurrency histories, and
raw result pickles remain in the hash-bound local campaign directory; the
committed attestation binds their hashes and sizes.

## Provenance

- DarkoFit commit: `54b5e99b2d06137c6cb8c5c9b5c7b66165520ebd`;
  Git tree: `a9bfc029d2a1c6fc0183b721985ba413e9db1151`.
- Python: 3.12.13; AutoGluon: 1.5.1b20260712; TabArena commit:
  `4cd1d2526874962daae048a6f2dcf34aa272f3fa`.
- Run manifest SHA-256:
  `59e4c86eef06b259151b8abc06d66df77e63c092ef7558a16d2be5319e8170a6`.
- Completion attestation SHA-256:
  `d19bf5f954565cba849e0d19d21a13e25f19bbf40cecfd5303568cb64698ac0f`.
- Safe analysis payload SHA-256:
  `8d4d24245b10a5af18779a5b73e4bf547f54e27a699f31134943a958c1a511bd`.
- Analyzer summary SHA-256:
  `368382dd425a76affebdbcd07e093d66430aea264ecf37161d7f743956733869`.
- Paired-split table SHA-256:
  `0fd2cb6685d9d92964ff319da039380fde02a57695493a01d2c351c4c214eaa1`.
- Per-dataset table SHA-256:
  `6e32449804b6f2dfb6eb136e012a3e62e446525327605453b1cc3bbff4f51cd2`.
- Paired-child table SHA-256:
  `fa67eb74c68560ffde5b691505d12a3bb9d3ee589d293905a9a9ab465da605a3`.
- Frozen protocol semantic digest:
  `ff1270ac7ae53bd18d694bc3ebf8bdc33738428467ef622e3f00741db2ffccd6`.
