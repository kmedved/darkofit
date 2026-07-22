# M6 quality successor v2 contract

_Frozen before v2 historical backtest execution on 2026-07-21._

Contract identity: `m6-quality-successor-v2`.

## Reason for the new identity

The v1 artifact-only calculation reproduced its declared advance and kill, but
the pre-activation audit found a self-referential analyzer hash and missing
repeat attestation. The create-only v1 calculation remains immutable and has no
ranking authority. V2 is a structural correction, not a threshold repair.

The v2 quality thresholds, medium 60-cell grid, historical subset, input
artifact hashes, and non-shipping boundary are unchanged from v1. Their v1
outcomes are already known, so v2 claims no outcome blindness. A new identity
and one-shot result are required because the implementation binding changes.

## Immutable decision rule

`m6_quality_rule_v2.py` is the complete immutable ranking rule and exact grid.
It contains no activation flag or result hash. The backtest result binds that
file's whole-file SHA-256. Later activation does not edit it.

The ten exact dataset ids run at `medium`, seeds 0--2, weights `none` and
`stress`, four threads, and three repeats: 60 paired cells and 120 rows. A
quality candidate advances only if its equal-cell primary-loss ratio is at
most `0.98`, it wins at least 60% of cells, and no cell is above `1.02`.
Per-dataset and leave-one-dataset-out ratios are always reported.

The aggregate band is the minimum movement worth later mechanism/evidence
cost; the breadth rule prevents a narrow subgroup from carrying it; the cell
bound prevents hidden local harm. These rules rank spent development work
only. They cannot ship, change defaults, access new evidence, or replace Tier-D
requirements.

## Exact execution wrapper

`run_m6_quality_successor_v2.py` is the sole eligible execution path. It:

- requires a clean committed harness and a committed create-only v2 backtest
  result whose bindings match the current immutable rule, contract, wrapper,
  comparison runner, and paired-execution foundation;
- requires clean, distinct control and candidate commits before execution and
  verifies that neither source changes during execution;
- invokes the ten dataset names explicitly, never `--datasets all`;
- owns the exact `medium`, three-seed, three-repeat, two-weight, four-thread
  command and records that command and repeat count in its manifest;
- uses `paired-evidence-v1` to validate exact pairs, inputs, probabilities,
  fitted/runtime thread state, implementation paths, and model metadata;
- writes the raw CSV, result, and manifest create-only; and
- consumes a stable mechanism id and positive inspection index even if a later
  step fails. Failed/inspected attempts require a testing-log note, and index
  gaps or reuse invalidate the mechanism audit.

All output paths must be outside the harness during execution so the harness
can remain clean. Every result is spent for that mechanism. Individual cells
may not become tuning targets.

## Predeclared v2 backtest

The subset and rules are unchanged:

- known advance: combined B1+B2 ensemble-v3 ratios from
  `m3b_ensemble_v3_r3_vs_single_readout_20260721.json`, SHA-256
  `99d693063c46a0708eb45a704af0b46611fa8ed89dbe4d6469b47c7cd4a27c64`;
- known kill: 3% linear-leaf selector ratios from
  `fresh_selector_confirmation.json`, SHA-256
  `4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d`.

The one-shot v2 runner must return advance and kill, bind every code/input
hash, and run from the clean first commit containing all v2 files and tests.
A disagreement is terminal. The backtest does not create new quality evidence
or imply independence among historical cases.
