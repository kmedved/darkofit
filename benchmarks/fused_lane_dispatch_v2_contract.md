# Behavior-exact fused-lane dispatch v2 design contract

_Frozen on 2026-07-21 before selector implementation, calibration outcome
access, or validation outcome access._

## Identity and reason for supersession

This is a create-only, pre-outcome successor to
[`fused_lane_dispatch_v1_contract.md`](fused_lane_dispatch_v1_contract.md),
whose SHA-256 is
`68d0dd6ef42f29d164943ef16e766821c5bd53319840b22a59b1bd449191cf1a`.
The unmodified v1 contract remains the scientific base; this document changes
only the product-observability and explicit-override clauses below. Where the
two documents conflict, v2 controls. Every calibration grid, seed, threshold
selection rule, validation cell, performance/resource limit, stop rule, and
downstream restriction in v1 remains unchanged.

V1 was frozen at commit
`b65337689071fda6b4806fc7e3b6f3683f2ac29e`. Before implementation began, an
owner-supplied review correctly identified that a static automatic dispatch
must be auditable from a saved model and explicitly overridable. V1 instead
forbade a fitted attribute, archive field, and public option. No calibration
or validation outcome has been accessed, so the clean correction is this new
contract identity rather than an edit to v1.

The implementation source pin is
`2e9e14cb8af40caf9672c04bb0933802d324fb6d`. That commit contains one separate
hygiene fix discovered while establishing the pre-implementation baseline:
Logloss evaluation now combines fixed input-order blocks, eliminating
scheduler-dependent one-ULP fitted-score/archive drift. It does not alter any
tree, histogram, split, prediction, dispatch, or campaign threshold.

## Public override and staged default

V2 authorizes one bounded estimator option:

`oblivious_kernel={"auto", "fused", "unfused"}`

Its constructor default is `"auto"`. The scalar booster and the public
regressor/binary-classifier wrappers expose and serialize the same value.
Multiclass, distributional, LightGBM, hybrid, depthwise, and every other
non-scalar-oblivious fit retain their current kernels under `"auto"`.

The modes mean:

- `"auto"`: resolve exactly once after preprocessing and before the first
  tree. Before a qualifying calibration threshold is committed, this mode
  resolves to today's fused behavior with reason `threshold_unavailable`.
- `"fused"`: retain the existing fused-oblivious request for the functionally
  eligible scalar lane, independent of the automatic performance envelope.
- `"unfused"`: use the existing histogram-builder-then-split-search reference
  for the same functionally eligible scalar lane, independent of the automatic
  performance envelope.

An explicit non-`"auto"` value on an ineligible configuration raises a
`ValueError` before the first tree and names the failed eligibility condition;
it may not silently decline. `"auto"` outside the measured hardware/shape
envelope keeps today's fused request and records the exact fallback reason.
Expected automatic fallback does not emit a warning on every ordinary fit.

This option is an observability and escape-hatch surface, not a promise that
either forced lane is faster on unmeasured hardware. No additional lane,
online race, adaptive timing probe, environment-variable override, or hidden
exception table is allowed.

## Functional eligibility

Automatic switching and both explicit overrides operate only when all of the
following are true after preprocessing:

- scalar `GradientBoosting` with `tree_mode="catboost"` (including the
  normalized oblivious alias);
- full-row uniform sampling with no active MVS, GOSS, weighted-GOSS, or
  Bayesian bootstrap;
- full-feature sampling;
- `random_strength == 0`;
- resolved Numba thread count greater than two;
- no row-parallel histogram buffers;
- no level-histogram subtraction;
- no precomputed root histogram; and
- float64 histograms with the existing scalar unit- or variable-Hessian
  semantics.

The automatic mode additionally requires every macOS-arm64, row, feature,
thread, depth, and realized-bin envelope bound declared by v1. Explicit modes
may be used outside the measured hardware/shape envelope when functional
eligibility holds, but no automatic-performance claim follows from that use.

## Deterministic resolution and persisted metadata

Resolution is a pure function of the requested mode, functional eligibility,
the v1 platform/shape envelope, and one frozen `scan_work` threshold. Logical
CPU count is recorded only as a hardware fingerprint; it may not introduce a
second decision branch. The selector may not inspect targets, weights, feature
values, timings, task names, sports provenance, or earlier fits. The v1
`scan_work` formula and threshold tie rule are unchanged.

Each completed fit persists exactly one schema-validated record at both
`oblivious_kernel_dispatch_` and
`auto_params_["oblivious_kernel_dispatch"]`; the two dictionaries must be
equal. Safe-NPZ save/load and deterministic resave preserve it. Required
fields are:

- `schema_version` (integer `1`);
- `requested` (`"auto"`, `"fused"`, or `"unfused"`);
- `resolved` (`"fused"` or `"unfused"`);
- `reason` (a closed, tested reason code);
- `functional_eligible` and `automatic_eligible` (booleans);
- `threshold` and `scan_work` (nonnegative integers or `null`);
- `engaged` (boolean, derived from actual level engagement rather than the
  request alone); and
- `fused_level_count` and `unfused_level_count` (nonnegative integers whose
  sum agrees with `engaged` and whose zero/nonzero pattern agrees with the
  resolved lane); and
- an `inputs` object containing platform system/machine, logical CPU count,
  rows, active features, resolved threads, resolved depth, and maximum
  realized bins.

Private per-fit counters record actual fused and unfused level engagement for
invariants and the evidence harness. They are not constructor state and do not
choose the lane. A saved record claiming an unknown mode/reason, disagreeing
with the serialized constructor request, containing malformed inputs, or
disagreeing with the fitted trees/counter-derived engagement fails safe load.

## Exactness projection and goldens

V1's behavior-exact requirement remains binding. Because v2 deliberately
records the request and resolution, byte-identical whole archives are neither
possible nor the correct oracle between a forced-fused control and an
automatic/unfused candidate. The paired exactness projection removes only:

1. the booster and wrapper `oblivious_kernel` constructor fields;
2. `auto_params.oblivious_kernel_dispatch`; and
3. the duplicate fitted `oblivious_kernel_dispatch_` view, if represented
   separately in a harness record.

After that declared projection, every safe-NPZ member, header value, array,
tree, split, gain, leaf value, feature importance, fitted round/stop field,
prediction, probability, RNG fingerprint, and deterministic-resave result
must be exact. The harness must also prove that each arm's unprojected dispatch
metadata is internally consistent and survives load exactly.

Golden outputs cover at least unit-Hessian RMSE, weighted RMSE, binary
Logloss, categorical preprocessing, callback stop, early stopping/refit,
automatic fallback, both explicit overrides, malformed override values,
unsupported explicit configurations, save/load, and caller Numba-mask
restoration. Sampled, multiclass, distributional, randomized-split,
one/two-thread, leaf-wise, hybrid, root-copy, subtraction, and row-parallel
cases prove unchanged automatic fallback or explicit rejection as applicable.

## Execution authority and downstream sequence

This combined v1+v2 design contract authorizes implementation of the pure
selector, override plumbing, fitted metadata, safe-load validation, private
engagement counters, and invariant/golden tests. It still does **not**
authorize calibration execution, validation execution, committing a selected
threshold, changing the effective `"auto"` default away from current fused
behavior, making a speed or portability claim, cutting a release, opening Q,
running M2/M4, using fresh data, or opening a lockbox.

A separate create-only execution contract must hash both design contracts,
the finished harness, generators, analyzers, tests/goldens, runtime, and exact
source pin before calibration outcomes are opened. If calibration qualifies,
the threshold artifact is committed before automatic candidate wiring, and a
second create-only freeze binds validation exactly as v1 requires.

Regardless of the dispatch outcome, B-archive remains closed. A passing
dispatch re-baselines any later Q microbenchmark. Because this follows several
speed/packaging mechanisms, the next funded mechanism slot after Wave 4 is
reserved for the quality shortlist unless an owner decision records a
specific exception; the current leading nominees are the T7b-derived
`l2_leaf_reg` and samples-per-feature depth-policy mechanisms.
