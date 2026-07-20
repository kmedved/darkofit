# Standing M5/M6 evidence protocol

Status: Wave 1 infrastructure draft, authorized 2026-07-20.

This protocol creates the cheap middle rung in
[`COUNTERPUNCH_PLAN.md`](../COUNTERPUNCH_PLAN.md). It is deliberately split
between a non-ranking diversity guard and a spent comparative development
slice:

- M5 is a sentinel suite. It detects crashes, invalid outputs, invariant
  failures, and unexplained drift. It is not an acceptance score.
- M6 compares one frozen pre-mechanism DarkoFit control with one candidate.
  Its repeatedly inspected results may rank or kill development ideas, but
  cannot authorize shipping or a default change.

The machine-readable draft contract is
[`standing_evidence.py`](standing_evidence.py). The M5 domain registry is in
place, including binary, multiclass, weighted, grouped/entity, categorical,
missing-value, and high-row coverage. Its generators, fingerprints, and
expected ranges are not frozen yet; calling those registry entries completed
sentinels would overstate the current infrastructure.

## M6 draft-v2 coordinates

M6 reuses the ten deterministic builders in
[`benchmark_adapters.py`](benchmark_adapters.py): four regression, three
binary, and three multiclass datasets, with numeric and categorical coverage.
The first executable draft pins:

- size `small` (2,500 requested rows; pinned real datasets retain their
  naturally smaller row counts);
- seeds 0, 1, and 2;
- unweighted and deterministic stress-weighted fits;
- public defaults for both source trees;
- four threads; and
- one fresh worker per arm and cell, with control/candidate order alternating
  across dataset/size/seed blocks within each weight stratum and opposite
  orders for the two weight modes.

That is 60 matched cells and 120 raw rows. Missing-value coverage belongs to
M5 until an adapter supports it. This is intentionally not the freeze shape:
the executable contract refuses `contract_frozen=true` until a medium size
and exact ChimeraBoost and CatBoost release anchors are present. Peak RSS
must also be added before the freeze review. The current small-only draft is
not allowed to become the comparative contract by inertia.

The runner records task-appropriate primary loss (RMSE or log loss), the
existing secondary metrics, fit and prediction time, source commit/tree
identity, machine details, contract hash, CSV hash, and paired candidate/control
ratios. A full run requires clean committed source trees, a candidate tree
different from the control, a stable mechanism id, and a positive one-based
inspection index. The manifest records that index and marks every full-run
outcome spent. Assign the index before launch and increment it for every
material inspection of that mechanism; failed launches also consume an index
and need an invalid-attempt note. The adjacent `TESTING_LOG.md` entry repeats
the id and index, making panel grinding visible. A skipped or reset index
invalidates the mechanism's M6 audit.

Output and its adjacent `.manifest.json` are create-only. While the contract's
`contract_frozen` and `backtest_complete` flags remain false, even a full run
is labeled contract-development evidence and is not ranking-eligible. The
checkout supplying this runner and the dataset builders is also required
clean, recorded in the manifest, and checked for changes across execution.

## Harness null check

The three-dataset smoke is a harness test, not evidence and not a testing-log
entry. Point both arms at the same clean checkout:

```bash
python benchmarks/run_standing_evidence.py \
  --smoke \
  --control /path/to/darkofit \
  --candidate /path/to/darkofit \
  --csv /private/tmp/darkofit-m6-smoke.csv
```

It must produce 12 successful rows, complete matched pairs, identical primary
loss within each pair, and a provenance manifest. Timing ratios need not equal
one.

## Spent development run

Use separate clean worktrees for the exact pre-mechanism control and candidate:

```bash
python benchmarks/run_standing_evidence.py \
  --control /path/to/darkofit-control \
  --candidate /path/to/darkofit-candidate \
  --mechanism-id candidate-name \
  --inspection-index 1 \
  --csv /private/tmp/darkofit-m6-YYYYMMDD-candidate.csv
```

## Predeclared historical backtest

The following subset was committed before any M6 backtest execution. Its
machine-readable form, including the historical-result hashes and replay
adapters, lives in `standing_evidence.py`.

| Mechanism | Historical verdict | Primary axis | Frozen source pair |
| --- | --- | --- | --- |
| Fused variable-Hessian lane | Advance | Fit speed | `7097e7a` → `1016e7e` |
| Forest-work packed router | Kill | Prediction speed | `e089943` → `e961bcc` |
| 3% linear-leaf selector | Kill | Quality | control and candidate policy at `29bd30c` |

The selector is deliberately included so the backtest must demonstrate that
M6 can reject a quality mechanism, not merely confirm profiler-visible speed
winners. Its named policy adapter must be implemented and frozen before the
backtest; treating the same source pin as two public-default arms would be a
null comparison, not a replay.

Before M6 may rank new mechanisms, run every declared replay and record
`agree`, `disagree`, or `lacks_power` against its historical disposition.
Changing the subset after seeing any replay restarts the backtest under a new
contract version. `contract_frozen` is necessary but not sufficient:
`backtest_complete` remains false until the full declared subset is reported.
Every material full run receives the 12-field entry required by
[`TESTING_LOG.md`](TESTING_LOG.md), including its mechanism id and inspection
index in the notes. No individual cell may be used as a tuning target, and a
candidate that advances still needs the sports and milestone evidence required
by the plan and shipping policy.
