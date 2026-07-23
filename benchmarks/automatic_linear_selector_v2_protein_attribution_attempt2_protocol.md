# Automatic linear-selector v2 Protein attribution protocol, attempt 2

_Frozen before any attempt-2 Protein fit under candidate
`a53d4bf543534678189d87d88dcad87dd2a8bd8f`._

Contract identity:
`automatic-linear-selector-v2-protein-attribution-attempt2-20260722`.

## Authority, supersession, and evidence class

Revision R1 in `BEAT_CHIMERABOOST_PLAN.md`, adopted and published at
`d938d99bbc6324a0d8d34129a4c3b1c0ba2da5a9`, explicitly authorizes Protein
attribution attempt 2. Attempt 1 remains terminal and spent under contract
`automatic-linear-selector-v2-protein-attribution-20260722`; it is never
rerun or edited. Its manifest and failure result prove that worker zero died
while importing the data loader, before any model fit or outcome existed.

Attempt 2 is a new execution identity, not a favorable scientific rerun. It
binds the exact same candidate, coordinates, arms, quality rule, invariants,
thread budget, and telemetry policy as attempt 1. The only execution repair
replaces TabArena's dependency-heavy context import with the source-equivalent
OpenML task call used underneath its `OpenMLTaskWrapper`. The resulting split
fingerprints must equal the immutable v0.11 release-ladder raw rows for all
three coordinates before the launch manifest is created. A loader-preflight
failure therefore spends no attempt.

Protein and all three coordinates are already-spent development evidence.
This attempt can close the exact candidate or advance it to
`ready_for_powered_fresh_design`; it cannot merge, ship, change a default,
inspect fresh confirmation, run TabArena as a scoreboard, or access a
lockbox.

## Bound lineage

Attempt 2 binds:

- selector candidate
  `a53d4bf543534678189d87d88dcad87dd2a8bd8f`;
- selector development contract
  `automatic-linear-selector-v2-development-20260722`;
- the advancing M6 v3 inspection-1 result and manifest;
- the original attempt-1 protocol, launch manifest, terminal failure result,
  and create-only result note;
- the repaired base runner and its tests from published commit `370b892`;
- the R1 authorization record at published commit `d938d99`;
- the v0.11 release-ladder v3 contract and its pinned TabArena split source;
  its immutable raw artifact supplies the three expected split fingerprints;
  and
- OpenML task 363693, `physiochemical_protein`.

## Frozen coordinates, arms, and decision rule

The exact attempt-1 grid remains unchanged: coordinates `(repeat, fold,
seed)` are `(0, 0, 0)`, `(1, 1, 1001)`, and `(2, 2, 2002)`; the cyclic Latin
rotation covers `constant` (`linear_leaves=False`), `automatic`
(`linear_leaves="auto"`), and `explicit_linear` (`linear_leaves=True`). Each
cell runs in a fresh process with 14 threads. Warmup, fit, prediction timing,
process-tree RSS, and exact source checks are inherited unchanged from the
attempt-1 runner.

For every coordinate the automatic arm must be eligible, select linear at a
relative validation improvement of at least `0.03`, use the declared
disjoint weighted-target-stratified holdout, finish with linear leaves active,
and be prediction-byte and normalized-core-state identical to
`explicit_linear`.

Quality is `automatic RMSE / constant RMSE`. Both the equal-coordinate
geometric mean and every coordinate must be at most `1.02`. There is no
minimum-effect gate. All seven inherited exactness/selector invariants and
both harm checks must pass. Passing yields only
`ready_for_powered_fresh_design`; otherwise the candidate closes.

## Execution and no-rerun discipline

The protocol, attempt-2 wrapper, and tests must be committed and published
before the preflight or first fit. Execution requires clean, exact, published
harness and candidate sources plus the exact clean TabArena source. The
preflight loads and fingerprints all three registered splits before any output
is created and records the interpreter plus OpenML, NumPy, pandas, and
scikit-learn versions. The frozen direct loader uses OpenML `0.15.1`; its
content fingerprints, not a network response or package-level assumption,
are the binding proof of split identity. After preflight succeeds, the
create-only launch manifest is written
before worker zero and spends attempt 2 even if execution later fails.

No failed or inspected attempt-2 launch may be rerun under this contract
identity, and no gate may be relaxed after inspection. A passing Protein
result must be followed by the development contract's historical guardrail
replay, computed from existing artifacts without refitting, before the
selector receives its final campaign disposition. A result note and 12-field
`benchmarks/TESTING_LOG.md` entry are required.
