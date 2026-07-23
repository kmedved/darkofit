# B3 parallel ensemble-members v1 development contract

_Frozen before candidate implementation or candidate timing inspection._

Contract identity: `b3-parallel-ensemble-members-v1-20260723`.

## Authority and claim

Revision R1 in `BEAT_CHIMERABOOST_PLAN.md` authorizes B3 as the next
one-at-a-time speed mechanism after the categorical-crosses development slot.
`NEXT_STEPS.md` section 4.3 supplies the binding memory-accounting rule. The
DarkoFit control pin is
`c4dae58fcf7a8d456533ba2d9b469f039adc453c`. The source-pinned donor reference
is ChimeraBoost `v0.21.0` at
`26fed8a715fe172518472f4fec1a663492db6f61` under Apache-2.0.

The claim is narrow: scheduling the eight already-public ensemble-v3 members
across independent worker processes can reduce fit wall time under the same
total CPU budget without changing fitted predictions, member seeds, sampled
rows, early-stopping outcomes, or the public sequential behavior. This
contract adds no public constructor parameter, changes no default, and does
not authorize a merge or release.

## Exact private candidate

The only candidate behavior is a private B3 route for `ensemble_mode="v3"`,
`n_ensembles=8`. All public fit routes remain sequential.

On the 14-physical-core campaign machine, the private automatic topology is:

- total CPU budget `B = 14`;
- worker count `W = min(K, max(1, floor(B / 2))) = 7` for `K = 8` members;
- per-worker Numba/BLAS budget `T = floor(B / W) = 2`; and
- maximum simultaneous model threads `W * T = 14`.

The general private resolver is deterministic for positive `B` and `K`, never
returns more workers than members, never returns fewer than one thread per
worker, and never allocates more than `B` total model threads. Sequential
control is `W=1, T=14`. The fixed same-thread equivalence control is `W=1,
T=2`; it is not a timing comparator.

Workers must receive precomputed deterministic member plans. Results are
reassembled strictly by member index, not completion order. Any worker error,
missing member, duplicate member, invalid result, or cancelled future fails
the whole fit and restores the estimator's previous state. Each worker gets a
fixed thread environment before importing/using Numba; nested BLAS/Numba
oversubscription is forbidden. The caller's thread-local ambient Numba mask
must be restored after success and failure.

The private candidate may refactor member fitting only enough to share the
same top-level worker implementation between sequential and parallel routes.
Sampling, member policy, preprocessing eligibility, OOB evaluation,
early-stopping, fitted metadata, serialization, and prediction aggregation are
otherwise frozen to control behavior. No member-count, recipe, quality,
prediction-batching, archive-format, or shared-categorical-transform change is
part of B3 v1.

## Correctness invariants before timing

Before timed evidence, tests must establish:

1. the topology resolver produces `(7, 2)` for `(K=8, B=14)`, respects the
   CPU bound over boundary cases, and leaves public fits sequential;
2. parallel and sequential fits at the same `T=2` have exact prediction and
   probability hashes, member order, seeds, sampled/OOB index hashes,
   best iterations, class metadata, and normalized fitted state after removal
   of scheduling-only telemetry;
3. the formal `1x14` control and `7x2` candidate also have exact prediction or
   probability hashes; a thread-count-dependent model result kills B3 v1;
4. numeric, categorical, weighted, grouped, binary, and multiclass fits work,
   including group-disjoint OOB behavior and zero-weight rows;
5. safe-NPZ save/load and deterministic resave preserve all predictions,
   member order, and B3 provenance, with schema-derived corruption rejection;
6. parent and worker failures are propagated, partial results cannot be
   adopted, repeated fits do not retain an executor, and clone/get-params stay
   unchanged because the route is private;
7. the parent ambient Numba mask is restored after success/failure and every
   returned member records both its fit-time thread count and the thread count
   actually used by sequential ensemble prediction; and
8. prediction, staged prediction, feature names/importances, empty batches,
   and existing sequential/public ensemble tests remain valid.

## Frozen timing and resource panel

After invariants pass, execute one numbered, create-only inspection using the
four already-spent general cases from the ensemble-v3 characterization:

- `general_friedman_numeric`;
- `general_categorical_reg`;
- `general_numeric_binary`; and
- `general_categorical_multiclass`.

Use the exact existing case generators/splits, 600 maximum rounds, patience
30, validation fraction 0.15, random state 4, eight v3 members, three paired
blocks, and fresh outer workers. Each block rotates arm order. For each arm,
record:

- the first fit after process start (`cold_executor`), including executor
  creation/import overhead;
- the immediately repeated fit in the same outer worker (`steady_executor`),
  including no model reuse but allowing the process backend to reuse workers;
- complete predict latency on the fixed test rows after each fit;
- parent-plus-recursive-child peak RSS, start RSS, peak-minus-start bytes,
  archive bytes, worker topology, thread masks, member fit counts, and output
  hashes; and
- exact source, contract, case, split, weight, implementation, and hardware
  fingerprints.

The sequential comparator uses `1x14`; the candidate uses `7x2`. Timing begins
immediately before the public-control/private-candidate fit call and ends only
after all members are adopted or the fit fails. No arm-specific omitted work
or prebuilt fitted state is allowed.

## Frozen disposition

B3 v1 advances only if all of the following hold:

1. every correctness, source, execution, and output-integrity invariant
   passes;
2. candidate/control fit-time ratios are `<= 1.0` for the equal-case geometric
   mean in both `cold_executor` and `steady_executor` views;
3. every case's three-block median fit-time ratio is `<= 1.0` in both views;
4. the worst leave-one-case-out geometric-mean ratio is `<= 1.0` in both
   views; and
5. the hybrid RSS rule below passes.

There is no arbitrary minimum speedup. The all-case and leave-one-out rules
establish stable direction inside the declared envelope instead of allowing a
single large win to hide a workload slowdown. Prediction timing and archive
bytes are reported telemetry; prediction correctness and thread restoration
are hard invariants.

The 24 GiB campaign machine gives a harm-derived absolute process-tree peak
RSS ceiling of one quarter of physical memory: `6 GiB`. The ratio allowance is
`1 + ceil(W/2) = 5x` the paired sequential peak. The absolute-delta allowance
is one twelfth of physical memory: `2 GiB`. Memory fails if candidate peak RSS
exceeds `6 GiB`, **or** if it exceeds both `5x` the paired sequential peak and
the sequential peak by more than `2 GiB`. Ratios and deltas are telemetry when
only one allowance is exceeded.

`advance` means only `eligible_for_public_b3_contract_design`; `kill` closes
this exact topology and implementation. Any harness or worker failure spends
inspection 1 and closes that execution identity. A repair requires a new
explicit identity and owner authority; no favorable rerun is allowed.

## Execution discipline and non-claims

Candidate, control, and harness must be clean, committed, and published before
the formal run. The launch manifest is written create-only before work begins,
which spends inspection 1. Use fresh workers, exclusive-machine audit,
complete process-tree monitoring, and a 12-field `TESTING_LOG.md` entry.

No fresh data, sports panel, M2, TabArena, lockbox, public API/default, candidate
merge, release, quality claim, or rival-comparison claim is authorized.
