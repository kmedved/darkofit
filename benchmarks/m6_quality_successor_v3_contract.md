# M6 quality successor v3 contract

_Frozen after the Phase F rule-victim audit and before v3 historical backtest
execution on 2026-07-22._

Contract identity: `m6-quality-successor-v3`.

## Why v3 exists

M6 v2 remains immutable, but its ranking rule has lost forward authority. It
requires a candidate to win at least 60% of cells and its negative backtest
expects the 3% smooth-data linear-leaf selector to be killed. The Phase F audit
found that verdict depended on the since-abolished win-count gate: the selector
had a `0.989264` aggregate ratio, zero lineage or split losses, and twelve
intentional no-ops. Reusing v2 would let an abolished gate kill the same
candidate from inside the replacement development rig.

V3 supersedes only future M6 quality ranking. It does not edit v1/v2 artifacts,
retroactively change their outcomes, or authorize the revived selector. Every
candidate still needs its own new identity and downstream evidence.

## Immutable development rule

`m6_quality_rule_v3.py` is the complete immutable ranking rule and exact grid.
It contains no activation flag or result hash. The ten exact dataset ids run at
`medium`, seeds 0--2, weights `none` and `stress`, four threads, and three
repeats: 60 paired cells and 120 rows.

V3 deliberately has **no win count and no minimum worthwhile effect size**. A
candidate advances from this spent development slice only when:

1. its equal-cell aggregate primary-loss ratio is at most `1.000`;
2. its worst dataset geometric-mean ratio is at most `1.020`; and
3. its worst leave-one-dataset-out aggregate ratio is at most `1.003`.

The first condition prevents ranking an overall regression. The second is the
current Tier-D unguarded harm bound, reused conservatively for development
triage. The third is the current default concentration slack (`1.000 + 0.003`)
and prevents a single favorable dataset from carrying an otherwise harmful
candidate. Individual-cell maxima are telemetry, not a gate: seeds and weight
modes are not independent datasets and this rung cannot ship anything.

Per-dataset and leave-one-dataset-out ratios are always reported. `advance`
means only “eligible for mechanism-specific work”; `kill` means “do not spend
the next evidence rung on this exact candidate.” Neither disposition can
change a default, expose an API, access fresh/lockbox evidence, or substitute
for `SHIPPING_POLICY.md` Tier-D uncertainty, power, harm, cost, and no-rerun
requirements.

## Exact execution wrapper

`run_m6_quality_successor_v3.py` is the sole eligible execution path. It:

- requires a clean committed harness and a committed, create-only v3 backtest
  result whose bindings match the current audit, v2 supersession, immutable rule, contract,
  backtest runner, execution wrapper, comparison runner, and paired-execution
  foundation;
- requires clean, distinct control and candidate commits before execution and
  verifies that neither source changes during execution;
- invokes the ten dataset names explicitly, never `--datasets all`;
- owns the exact `medium`, three-seed, three-repeat, two-weight, four-thread
  command and records that command and repeat count in its manifest;
- uses `paired-evidence-v1` to validate exact pairs, inputs, probabilities,
  fitted/runtime thread state, implementation paths, and model metadata;
- writes the raw CSV, result, and manifest create-only; and
- consumes a stable mechanism id and positive inspection index even if a later
  step fails. Failed or inspected attempts require a testing-log note, and
  index gaps or reuse invalidate the mechanism audit.

All output paths must be outside the harness during execution so the harness
can remain clean. Every result is spent for that mechanism. Individual cells
may not become tuning targets.

## Outcome-known v3 backtest

The subset was declared by the Phase F audit. All outcomes are already known;
v3 claims no outcome blindness. It contains one surviving positive, one
surviving negative, and one explicit obsolete-verdict tripwire:

- known advance: combined B1+B2 ensemble-v3 ratios from
  `m3b_ensemble_v3_r3_vs_single_readout_20260721.json`, SHA-256
  `99d693063c46a0708eb45a704af0b46611fa8ed89dbe4d6469b47c7cd4a27c64`;
- known kill: native ordinal C2 task ratios from
  `native_ordinal_c2_development_result.json`, SHA-256
  `7aeb83131bb7604a3eaabc2789f048d40dabb58791b6ab6aad0ac26f0f0f566f`;
  its `1.317510` worst-task ratio is genuine harm under the current rule; and
- retired selector verdict: the 3% linear-leaf selector ratios from
  `fresh_selector_confirmation.json`, SHA-256
  `4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d`.
  V3 must return `advance`, proving the old win-count kill is absent. This is a
  tripwire, not new evidence or authorization for the selector.

The one-shot runner must agree on all three replays, bind every code, audit,
and input hash, and run from the clean first commit containing all v3 files and
tests. Any disagreement is terminal. The backtest does not imply independence
among historical cases and creates no new quality evidence.
