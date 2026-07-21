# Fused-lane dispatch calibration execution protocol v1

_Prospective and outcome-blind. This protocol implements, but does not itself
authorize execution of, the calibration phase frozen in the fused-lane v1/v2
design contracts and realized-bin-width erratum._

## Boundary

The phase is generic synthetic kernel evidence. It is not sports evidence, a
quality campaign, a public speed claim, a release gate, M2/M4, Q re-entry, or
lockbox access. The product's `"auto"` mode remains fused because the source
pin contains no crossover threshold.

The execution contract records `execution_authorized=false`. The runner must
refuse to open outcomes unless it is also given a separate create-only owner
authorization whose contract SHA-256, phase, campaign identity, and source pin
all match. Creating or freezing this protocol is not that authorization.

## Immutable source layout

Execution uses two clean trees:

1. the harness checkout containing the committed execution contract and
   unchanged hash-bound runner/analyzer files; and
2. a separate clean Git worktree detached at the execution contract's exact
   `source` commit, from which `darkofit` is imported.

This prevents the later contract or authorization commits from changing the
measured product source. Every worker checks the detached source before and
after its coordinate. The parent checks the harness and source across the
whole phase.

The execution freeze also records and the runner rechecks the exact Python,
NumPy, Numba, and llvmlite versions plus the OS release, architecture, hardware
model, CPU identifier/counts, and installed memory. A runtime or machine change
requires a new outcome-blind execution identity; evidence from this campaign
is hardware-scoped to the recorded host.

## Calibration grid and generation

The grid is the Cartesian product declared by the v1 design contract:

- rows: 8,192; 32,768; 131,072; 524,288; 1,048,576;
- `(features, threads)`: `(15, 4)`, `(24, 9)`, `(48, 14)`;
- Hessian: unit and deterministic positive-variable;
- depth 6, 128 realized kernel bins, L2 3, learning rate 0.1, no split
  randomness, no sampling, no row-parallel buffers, no root-copy, and level
  subtraction off.

The generator is pinned by source hash and seed `20260721`. It emits uint8
binned predictors, deterministic structured float64 gradients, and either
unit or `[0.25, 2.0)` deterministic float64 Hessians. Each worker records
per-array and combined dataset hashes.

## Fresh-process timing

One fresh worker owns one shape/Hessian coordinate and both lanes. Its process
environment fixes every common thread variable and starts Numba with the
coordinate's exact thread ceiling. Shared data, Fortran histogram view,
histogram buffers, and split scratch are allocated before timing.

Both forced lanes receive two same-coordinate untimed warmups. Seven paired
timed repetitions then alternate order:

1. fused, unfused;
2. unfused, fused;
3. repeat that alternation through repetition seven.

Timing covers only `build_oblivious_tree`. Each repetition records actual
fused and unfused level counters. Exactness requires identical trees, split
gains, leaf values, leaf assignments, leaf gradient/Hessian totals, and probe
predictions. The caller's thread mask and Numba ceiling/current count must be
unchanged. Analysis treats a malformed fingerprint, runtime record, counter,
state hash, or thread-restoration record as an exactness failure.

## Create-only failure and result behavior

The raw output path and its terminal-failure sibling must both be absent before
the first worker starts. Rows stay in parent memory until all coordinates and
source checks complete. Success creates exactly one raw JSON artifact.
Infrastructure failure creates exactly one terminal JSON record, publishes no
partial rows, and makes that execution identity non-rerunnable. A corrected
harness requires a new contract identity with the scientific grid and gates
unchanged.

The authorization, raw, terminal, and analysis paths are themselves frozen in
the execution contract. The runner rejects copied authorization records and
alternate output names, so changing a filename cannot manufacture another run
under the same campaign identity.

## Frozen analysis

For each coordinate, analyze paired `unfused / fused` ratios. Exactness is
conjunctive. Stability requires `IQR / median <= 0.10` at every coordinate.
Candidate thresholds are `never switch` and the integral midpoints between
distinct observed `scan_work` values, where:

`scan_work = rows * ceil(features / min(threads, features))`.

For each threshold, compute selected/current-fused geometric mean, worst
selected/current-fused ratio, and geometric-mean regret against the faster
measured lane. Select minimum regret; values within `0.001` tie, and the
largest threshold wins, with `never switch` largest. Qualification additionally
requires both lanes selected, selected/current-fused geometric mean `<=0.97`,
and no coordinate above `1.02`.

Failure closes Wave 4 with the fused default unchanged. Qualification permits
only a create-only threshold artifact. That artifact must be committed before
candidate wiring, followed by a separate source pin and separate create-only
validation execution contract. Calibration never directly authorizes
validation or a default change.
