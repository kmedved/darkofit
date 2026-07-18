# Panel 3 exact-policy power calibration

## Status and claim boundary

This is a frozen, development-only calibration campaign on already-spent
TabArena regression tasks. It estimates the complete fixed-panel effect
profiles of the two exact Panel 3 candidates against the exact current-default
control. It cannot confirm either candidate, authorize Panel 3, change a
default, support a product claim, or be combined with Panel 3 outcomes.

All tasks and coordinates below were used by the 13-task TabArena regression
program before this protocol was written. Every result is therefore
development evidence. The purpose of this lane is narrower: replace the
three-selected-lineage guarded-cross power input and the non-matching composite
input with an exact-policy census that includes engagements, declines, ties,
losses, and inapplicable size-gated coordinates.

## Fixed data boundary

Use the exact OpenML tasks and native feature views from the spent TabArena
regression panel:

| Order | Dataset | OpenML task |
| ---: | --- | ---: |
| 1 | `airfoil_self_noise` | 363612 |
| 2 | `Another-Dataset-on-used-Fiat-500` | 363615 |
| 3 | `concrete_compressive_strength` | 363625 |
| 4 | `diamonds` | 363631 |
| 5 | `Food_Delivery_Time` | 363672 |
| 6 | `healthcare_insurance_expenses` | 363675 |
| 7 | `houses` | 363678 |
| 8 | `miami_housing` | 363686 |
| 9 | `physiochemical_protein` | 363693 |
| 10 | `QSAR-TID-11` | 363697 |
| 11 | `QSAR_fish_toxicity` | 363698 |
| 12 | `superconductivity` | 363705 |
| 13 | `wine_quality` | 363708 |

For every task, run the three official OpenML coordinates `r0f0s0`,
`r1f1s0`, and `r2f2s0`: 39 outer coordinates in total. Load the task with
`include_row_id=False` and `include_ignore_attribute=False`. Preserve the
OpenML dataframe and categorical declarations. Targets must convert exactly
to finite `float64`; no imputation, task dropping, representation change, or
outcome-dependent preprocessing is permitted.

The runner records dataset identity, declared checksum, feature schema,
categorical columns, complete feature/target fingerprints, and exact train/test
index hashes. A missing coordinate, task drift, non-finite target, or failed
arm invalidates the campaign.

## Frozen arms

Each outer coordinate runs three arms, all with random state 4 and six threads:

1. `current_default`: the exact Panel 3 control,
   `DarkoRegressor(random_state=4, thread_count=6)`, fit on all outer-training
   rows.
2. `t5_composite_policy`: the exact Panel 3 T5 composite helper, including its
   2,000-row applicability gate, auto-mode audition, full-budget linear race,
   guarded top-six numeric diff/product features, candidate guard, best-prefix
   refit, and byte-exact current-default decline. The explicit ordinal map is
   empty on every task, matching the Panel 3 candidate contract.
3. `guarded_cross_features_policy`: the exact standalone Panel 3 guarded-cross
   helper: 2,000 rounds, learning rate 0.1, depth 6, L2 1, 128 bins, minimum
   child weight 1, CatBoost-style trees, a full-budget constant/linear race,
   top-six numeric split-gain diff/product candidates, a 0.95 validation-ratio
   engagement guard, and a selected-lane/best-prefix full-training refit.

The implementation is imported from
`run_panel3_confirmation.py`; this campaign must not copy or reinterpret those
fit paths. The exact candidate contract and runtime contract are source-bound.

## Execution and immutable artifacts

Before execution:

1. commit the protocol, freeze builder, runner, analyzer, exact helper sources,
   package sources, contracts, and all Panel 3 registry, execution, and
   calibration tests at source commit `H1`;
2. from a clean `H1`, create
   `benchmarks/panel3_cross_power_calibration_source_freeze.json`;
3. commit only that freeze artifact at `H2`; and
4. execute from a clean descendant whose `H1..HEAD` tracked diff contains only
   the freeze artifact.

The source freeze binds every executable source byte, the exact runtime
contract, both candidate contracts, the spent TabArena provenance artifacts,
the 13 tasks, 39 coordinates, three arms, clean Git source commit, and this
protocol. The runner rejects any drift.

Execution proceeds through 39 coordinate waves. Each wave launches three
isolated processes concurrently, one per arm, with six threads per process on
the 18-core host. A completed arm is spooled immediately, so a partner failure
does not discard finished work. Timing and memory are retained as operational
diagnostics only; concurrent contention is deliberate and no performance claim
is allowed.

Each of the 117 arm results is published once to a create-only spool record
under `.cache/panel3_cross_power_calibration_spool_v1/`. Resume may reuse a
spool record only after reopening it and verifying its source binding, content
hash, coordinate, arm, and result hash. Once all 117 records validate, the
runner creates exactly one immutable
`benchmarks/panel3_cross_power_calibration_raw.json`. Existing spool or raw
files are never overwritten.

The analyzer reopens and verifies all spool records, the raw artifact, source
freeze, complete grid, task/data/split identity, result hashes, predictions,
strict per-arm metadata schemas, finite validation ratios whenever a validation
comparison exists, decline reasons, selected lanes, best-prefix/refit fields,
guard consistency, prediction timing, and strictly positive finite RMSE
values. T5 size-gate declines use their smaller exact metadata schema and must
remain byte-identical to the control. It then creates exactly one immutable
`benchmarks/panel3_cross_power_calibration_summary.json`. Existing summaries
are never overwritten.

## Frozen analysis

For candidate `c`, control `d`, dataset `j`, and coordinate `s`, form:

```text
r[j,s;c/d] = RMSE[j,s;c] / RMSE[j,s;d]
R[j;c/d]   = exp(mean_s(log(r[j,s;c/d])))
G[c/d]     = exp((1/13) * sum_j(log(R[j;c/d])))
```

All 39 coordinate ratios and all 13 dataset ratios are retained. Exact ratio
1 is a tie, below 1 is a win, and above 1 is a loss. Nothing is filtered by
engagement, applicability, validation score, direction, magnitude, dataset, or
mechanism. The summary separately reports:

- coordinate and dataset wins/losses/ties;
- each candidate's complete 13-value fixed-panel power-input vector;
- all 39 coordinate ratios;
- T5 outer-training applicability (`>=2,000` rows) and its exact decline
  behavior;
- guarded-cross engagement/decline counts and validation ratios; and
- equal-dataset and worst-dataset ratios as descriptions only.

The fixed-panel vectors may be used only as spent-data, conditional design
inputs for a separately frozen Panel 3 power calculation. This analyzer does
not calculate a promotion decision, does not choose between candidates, and
always records:

```text
panel3_authorized = false
default_promotion_authorized = false
product_claim_authorized = false
```
