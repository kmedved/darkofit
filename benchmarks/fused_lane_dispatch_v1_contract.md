# Behavior-exact fused-lane dispatch v1 design contract

_Design frozen on 2026-07-21 before selector implementation, calibration
outcome access, or validation outcome access._

## Purpose and authority

This contract funds one bounded internal performance mechanism: choose between
DarkoFit's existing, behavior-exact fused and unfused oblivious-tree kernels
from hardware and workload shape. It starts from the clean, published
DarkoFit source pin
`bf2b0622a4b89f28850ad769b7836962721c9e77`.

The historical signal is deliberately narrow:

- Wave 1 Q0 at SHA-256
  `9111f14ae4d0d89e122f541b53f85c76c6bd5e76f4fa781c69039c1020c04e1c`
  found the forced-unfused fit path behavior-exact and faster than production
  fused at 500,000 and 1,000,000 rows on a 14-logical-CPU Apple-arm64 host;
  paired fit ratios were `0.901011` and `0.981264`.
- The earlier fused-subset artifact at SHA-256
  `ed45820d74733ebcc6fca3ed1524a49eb9d73ae7ef22925bec41e7dea22d9d01`
  found the fused path materially faster on the 5,241-row, 15-feature,
  18-thread workload.

Those spent results establish a crossover hypothesis, not a dispatch rule.
No Q0 coordinate may be reused as a validation coordinate, and neither
historical artifact may be reinterpreted as prospective confirmation.

The opportunity score is `3 impact * 4 confidence / 2 effort = 6.0`. The
profiled fused histogram-and-split path accounted for 52--63% of Q0 fit time,
the alternative arithmetic already exists, and prior tests establish exact
behavior. The funded work is therefore a selector and its evidence harness,
not a new kernel.

This design contract authorizes the smallest private selector implementation
and invariant tests. It does not yet authorize calibration execution,
validation execution, an internal-default change, a public estimator option,
a release claim, M2, Q re-entry, TabArena/M4, fresh data, or lockbox access.
A separate create-only execution contract must bind the finished harness and
the exact source used by each phase before outcomes are opened.

## Mechanism and preserved behavior

The only candidate mechanism is a pure internal selector between:

1. `fused`: the current production fused histogram-plus-split kernel; and
2. `unfused`: the current histogram builder followed by the current split
   search.

The implementation may not change histogram arithmetic, accumulation order,
split legality, gain arithmetic, tie-breaking, leaf routing, leaf values,
RNG consumption, preprocessing, sampling, early stopping, prediction, or
serialization. There is no new public parameter, constructor state, fitted
attribute, or archive field. Benchmark-only counters may report the selected
lane but must not influence the model or canonical archive payload.

V1 may switch only the scalar, full-row, full-feature CatBoost/oblivious lane
that is already eligible for fusion: unit or variable Hessian,
`random_strength=0`, more than two threads, no row-parallel buffers, no level
subtraction, and no precomputed root histogram. Sampled row/feature lanes,
multiclass root-copy, randomized splits, one/two-thread execution, leaf-wise
or hybrid trees, and every currently ineligible path retain today's dispatch.

The automatic candidate is active only inside the measured envelope:

- macOS on arm64;
- 3--14 resolved Numba threads;
- 8--64 active features;
- 8,192--1,200,000 fit rows;
- requested depth 4--8; and
- 64--254 maximum realized bins.

Every coordinate outside that envelope falls back to fused. A later campaign
may extend the envelope; this campaign may not infer portability to x86,
Linux, more than 14 threads, or unmeasured shapes.

## One-dimensional rule family

The selector may use exactly one fitted scalar threshold. Define

`scan_work = n_rows * ceil(n_active_features / min(n_threads, n_active_features))`.

Inside the eligible envelope, work below the frozen threshold selects fused
and work at or above it selects unfused. No second threshold, exception table,
task-specific branch, sports-specific branch, CPU-model lookup, learned
classifier, online timing probe, or outcome-dependent fallback is allowed.
Ties at the threshold select unfused. All dispatch inputs must already be
available before tree construction; the selector may not scan training data.

## Calibration phase and threshold freeze

Calibration is synthetic, generic, kernel-level, and spent. It uses NumPy
seed `20260721`, depth 6, 128 bins, zero split randomness, and both unit- and
deterministic-positive-variable-Hessian cases. The exact grid is the Cartesian
product of:

- rows: `8_192`, `32_768`, `131_072`, `524_288`, `1_048_576`;
- `(features, threads)`: `(15, 4)`, `(24, 9)`, `(48, 14)`; and
- paths: forced fused and forced unfused.

Each shape/Hessian coordinate uses a fresh worker. Within that worker, both
paths receive two untimed same-coordinate warmups, followed by seven paired
timed repetitions that alternate which path runs first. The execution contract
must bind generated-data hashes, source hashes, thread environment, actual
Numba thread count, engagement counters, and raw timings. Timing excludes
process startup, JIT warmup, allocation of shared input arrays, and result
serialization.

For each coordinate, compute paired unfused/fused ratios and require
`IQR / median <= 0.10`. Candidate thresholds are `never switch` plus the
midpoints between distinct observed `scan_work` values. For every threshold,
choose the predicted path in every calibration cell and compute:

- geometric-mean selected/current-fused time;
- worst selected/current-fused ratio; and
- geometric-mean regret against the faster measured path in each cell.

Select the threshold with the lowest geometric-mean regret. Ratios within
`0.001` are tied; ties choose the largest threshold, with `never switch` the
largest and therefore most conservative choice. A threshold qualifies only
when all cells are behavior-exact and stable, both paths are selected at least
once, selected/current-fused geometric mean is `<= 0.97`, and no cell exceeds
`1.02`. Otherwise the campaign closes before product dispatch implementation.

Publish the calibration rows, analysis, and selected threshold as create-only
artifacts. Commit the threshold artifact before adding it to the product
selector. Calibration artifacts are spent and may not be rerun or adjusted
after inspection.

## Validation phase

After a qualifying threshold is committed, one source commit may wire exactly
that threshold into the private selector. A new create-only execution freeze
must bind that candidate source before validation. Validation uses generic
synthetic data and these outcome-unseen full-fit cells:

| Cell | Task | Rows | Features | Threads | Depth | Bins | Rounds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-unit | RMSE, unit Hessian | 12,000 | 15 | 14 | 6 | 254 | 200 |
| mid-weighted | weighted RMSE | 75,000 | 37 | 6 | 4 | 64 | 80 |
| mid-binary | binary Logloss | 280,000 | 19 | 11 | 8 | 254 | 60 |
| large-unit | RMSE, unit Hessian | 650,000 | 24 | 14 | 6 | 128 | 40 |
| large-binary | binary Logloss | 900,000 | 47 | 9 | 5 | 192 | 40 |
| large-weighted | weighted RMSE | 1,100,000 | 63 | 14 | 7 | 254 | 30 |

All cells use the distinct validation seed `20260722`, learning rate `0.1`,
L2 `3`, full row and
feature fractions, no ordered boosting, no early stopping, and deterministic
targets/weights declared by the execution harness. Each cell runs current
forced-fused control and automatic candidate in three reciprocal blocks:
control/candidate, candidate/control, control/candidate. Every timed fit runs
in a fresh worker after a same-cell, three-round warmup. Calibration and
validation generators use distinct fixed seed streams and distinct row/shape
coordinates.

The threshold must select fused in at least two validation cells and unfused
in at least two. Otherwise validation is inconclusive and the dispatch is not
retained.

## Exactness and resource gates

Before timing may count, every paired validation cell must have:

1. the expected candidate lane and positive/zero engagement counters;
2. bitwise-identical predictions and probabilities where applicable;
3. identical canonical safe-NPZ model payloads and exact safe-load round trips;
4. identical tree structures, split gains, leaf values, feature importances,
   fitted tree counts, stop reasons, resolved learning rate/depth/thread count,
   feature schema, and fitted metadata other than benchmark telemetry;
5. identical RNG and data fingerprints; and
6. the caller's thread-local Numba mask restored after fit and predict.

The candidate passes performance only when all of the following hold:

- every fit and tree-build paired series has `IQR / median <= 0.10`;
- all-cell candidate/control geometric-mean fit ratio is `<= 0.98`;
- unfused-selected-cell geometric-mean fit ratio is `<= 0.97`;
- all-cell candidate/control geometric-mean tree-build ratio is `<= 0.98`;
- unfused-selected-cell geometric-mean tree-build ratio is `<= 0.95`;
- no individual fit or tree-build median ratio exceeds `1.02`; and
- every peak-RSS paired series is stable and has median ratio `<= 1.05`.

These are conjunctive gates. There is no averaging away an exactness failure,
unstable coordinate, or individual regression.

## Isomorphism proof and rollback

The implementation record must state:

- ordering preserved: yes, because each selected path retains its existing
  loop and reduction order;
- tie-breaking unchanged: yes, proven by exact tree and archive payloads;
- floating-point behavior: bitwise identical within every paired cell;
- RNG seeds: unchanged and fingerprinted; and
- golden outputs: captured from the forced-fused source before candidate
  wiring and verified after it.

The rollback is one commit: revert the selector/default wiring and retain the
current fused default. No archive migration or API deprecation is involved.

## Stop rule and downstream authority

Any calibration or validation infrastructure failure publishes a terminal
record, exposes no partial final-looking result, and makes that execution
identity non-rerunnable. A corrected harness or source requires a new identity
with unchanged scientific thresholds.

Failure or inconclusive evidence restores the current fused dispatch and
closes v1. Passing retains only the internal macOS-arm64 dispatch inside the
measured envelope. It authorizes re-baselining the Q re-entry microbenchmark
against the post-dispatch engine, but it does not itself reopen Q or authorize
M2, a public/default modeling change, a broad speed claim, fresh evidence,
TabArena/M4, or lockbox access.
