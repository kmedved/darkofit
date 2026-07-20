# Standing M5/M6 evidence protocol

Status: M6 contract frozen; historical backtest terminal-failed 2026-07-20,
so candidate ranking remains disabled.

This protocol creates the cheap middle rung in
[`COUNTERPUNCH_PLAN.md`](../COUNTERPUNCH_PLAN.md). It is deliberately split
between a non-ranking diversity guard and a spent comparative development
slice:

- M5 is a sentinel suite. It detects crashes, invalid outputs, invariant
  failures, and unexplained drift. It is not an acceptance score.
- M6 compares one frozen pre-mechanism DarkoFit control with one candidate.
  Its repeatedly inspected results may rank or kill development ideas, but
  cannot authorize shipping or a default change.

The machine-readable contract is
[`standing_evidence.py`](standing_evidence.py).

## M5 v1 coordinates

| Domain | Dataset | Seeds | Profile |
| --- | --- | --- | --- |
| Grouped/entity regression | generic group-disjoint generator v1 | 0, 1 | three-member group bootstrap |
| Smooth numeric regression | SynthGen df1/311 | 0, 1 | fixed 300-tree profile |
| Noisy numeric regression | SynthGen df1/241 | 0, 1 | fixed 300-tree profile |
| Categorical + missing regression | SynthGen df1/234 | 0, 1 | fixed 300-tree profile |
| High-row numeric | Friedman adapter, 50,000 rows | 0 | fixed 120-tree profile |
| Binary classification | earned SynthGen canary df1/647 | 0, 1, 2 | frozen canary profile |
| Multiclass classification | earned SynthGen canary df1/077 | 0, 1, 2 | frozen canary profile |
| Weighted regression | wide-numeric adapter, 10,000 rows | 0, 1 | stress weights |
| Weighted classification | numeric-binary adapter, 10,000 rows | 0, 1 | stress weights |

That is 19 cells and 38 fresh-worker rows against the exact post-H1 control
source `726e5d8`. Every row requires finite task-appropriate quality, a
normalized loss no worse than `1.10` times a train-only trivial predictor,
valid probabilities where applicable, exact save/load predictions, resolved
metadata, prediction fingerprints, and dataset/split hashes. Each earned
canary must retain mean excess Brier at most `0.005` and worst-seed excess at
most `0.01`. The initial behavior-identical control/candidate run establishes
the fingerprints and same-machine paired performance ratios; future checks
bind to that artifact. No value is an acceptance or ranking score.

The baseline contract remains unfrozen until the complete create-only
artifact hash is embedded:

```bash
/opt/anaconda3/envs/darko311/bin/python benchmarks/run_m5_sentinels.py \
  --control /private/tmp/darkofit-wave1-source-726e5d8 \
  --candidate /Users/konstantinmedvedovsky/code/darkofit \
  --output benchmarks/m5_sentinel_baseline.json
```

## M6 v3 coordinates

M6 reuses the ten deterministic builders in
[`benchmark_adapters.py`](benchmark_adapters.py): four regression, three
binary, and three multiclass datasets, with numeric and categorical coverage.
The first executable draft pins:

- sizes `small` and `medium` (2,500 and 10,000 requested rows; pinned real
  datasets retain their naturally smaller row counts);
- seeds 0, 1, and 2;
- unweighted and deterministic stress-weighted fits;
- public defaults for both source trees;
- four threads; and
- one fresh worker per arm and cell, with control/candidate order alternating
  across dataset/size/seed blocks within each weight stratum and opposite
  orders for the two weight modes.

That is 120 matched cells and 240 raw rows. Missing-value coverage belongs to
M5 until an adapter supports it. The exact release anchors are ChimeraBoost
commit `f14be606b641f1bf0dc92bb14b3951f1fe631c6b` and CatBoost 1.2.10 whose
installed wheel `RECORD` hashes to
`9c20fb35750d9ff814309323b225e836b538c1496745f357c8fd50187e7824ed`.
The executable contract still refuses `contract_frozen=true` until a
create-only release-anchor artifact is complete and its SHA-256 is embedded
in the contract. Thus adding the names alone cannot satisfy the freeze.

The runner records task-appropriate primary loss (RMSE or log loss), the
existing secondary metrics, fit and prediction time, worker peak RSS,
source commit/tree
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

Release anchors are established once with one fresh worker per
product/cell, a same-product three-tree warmup outside timing, and only the
thread count and random seed fixed around product defaults:

```bash
/opt/anaconda3/envs/darko311/bin/python \
  benchmarks/run_m6_release_anchors.py \
  --chimeraboost-source /Users/konstantinmedvedovsky/code/chimeraboost \
  --output benchmarks/m6_release_anchors.json
```

The create-only artifact contains all 240 product rows and is embedded at
SHA-256
`59747bc08d48a2ddad9b3cec05c965ecbd9edf21025c537f17dc58d816385409`.
The contract is therefore frozen. The separate historical backtest later
failed and is hash-bound at
`18b902e6099a4686b8eda71fac9ac327a0b5243872b80b5da79c5e01e5e2c201`;
candidate ranking remains disabled.

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

It must produce 24 successful rows, complete matched pairs, identical primary
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

The replay gates are also predeclared, not chosen from replay output:

- fused variable Hessian uses its 50,000-row binary-Logloss and
  stress-weighted-RMSE cases; exact behavior and measured engagement are
  mandatory, the fit-ratio geometric mean must be at most `0.90`, and each
  paired series must have IQR/median at most `0.10`;
- the packed router uses the frozen repeated-row cases at 127, 525, 585,
  2,409, 8,192, and 100,000 rows; predictions and observed dispatch must be
  exact, the two small gated cases must be at least 2x faster than legacy,
  both large cases must have candidate/legacy core ratios at most `1.10`, and
  every timing series must have IQR/median at most `0.30`; and
- the selector uses the small and medium Friedman, wide-numeric, and
  categorical-regression cells. Its exact deterministic 20% internal
  validation split selects linear leaves only for relative validation-RMSE
  improvement at least `0.03`. Advancement requires selector/default
  geometric-mean outer RMSE at most `0.98`, at least four wins in six cells,
  and no cell above `1.02`.

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

The committed replay executor is
[`run_m6_historical_backtest.py`](run_m6_historical_backtest.py). It runs the
fused and packed mechanisms through the exact historical candidate-source
runners (including their private reference/legacy and observed-engagement
paths), and runs the selector at source `29bd30c` on the six declared M6
cells. All source clones must be clean and exact; the packed replay also
requires the SHA-pinned historical basketball cache and ChimeraBoost 0.15
source. The executor and its frozen-rule unit tests are committed before the
first replay. Historical runners use an explicitly named clean Python
installation because the active development environment exposes an unrelated
regular `benchmarks` package through a site path; the exact historical runners
spawn nested workers and therefore cannot rely only on the parent's import
shim.

The first outcome-bearing launch was terminal: the fused replay disagreed
with its historical positive verdict, then the exact packed runner proved
unexecutable on the current 14-thread machine because it hard-requires 18
Numba threads. The selector was not opened. The failure record is
[`m6_historical_backtest_result.md`](m6_historical_backtest_result.md).
`backtest_complete` remains false, `backtest_terminal` is true, and the
executor refuses another launch under v3.
