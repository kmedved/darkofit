# Implementation roadmap

Roadmap for the remaining performance, serialization, benchmark, and modeling
work after the July 2026 correctness/API fix rounds. This document is meant to
be handed to implementers, so each item includes the implementation shape and
the gate that must pass before the next stage opens.

Current baseline assumptions:

- Base commit: `3029388`.
- Current worktree also includes the serialization hardening follow-up and the
  small F1-F3 follow-ups listed below.
- Current worktree suite after the roadmap implementation, the post-review
  payload-hardening round (`tests/test_payload_hardening.py`), and the
  lane-equivalence/A-B infrastructure (`tests/test_lane_equivalence.py`,
  `benchmarks/ab_compare.py`): `447 passed, 4 skipped` of 451 collected in
  the repo venv (no pandas; two pandas-gated and two thread-gated skips).
  `ab_compare.py . .` self-comparison: 17 cases bit-identical.
- Risk labels below are hypotheses until the item-specific acceptance gate
  proves them.

Stable IDs:

| ID | Status | Item | Value | Risk posture |
|----|--------|------|-------|--------------|
| R1 | Done in current worktree | Vectorize `_unique_if_at_most` | Faster wide low-cardinality binning | Behavior-identical tests and timing note landed |
| R2 | Done in current worktree | F-order `X_binned` for fixed-column row passes | Fixed-column routing cache behavior | Bit-identical lane/model checks landed |
| R3 | Done in current worktree | Segmented row layout resolver | Large-n leafwise fit time, regime-limited | Resolver excludes fast-lane regressions; no unsafe blanket flip |
| R4 | Done in current worktree | Exact MVS / weighted-GOSS threshold solve | Lower sampler cost at very large n | Tolerance/statistical sampler tests landed |
| R5 | Opt-in done in current worktree | Float32 grad/hess streams and narrower leaf ids | Major histogram bandwidth lever | Defaults unchanged pending regret/perf gates |
| R6 | Done in current worktree | Multiclass histogram and gradient layout | Large multiclass per-iteration win | Shared-vector class-minor layout landed |
| R7 | Done in current worktree | Shared preprocessing in auto/probe fits | Removes repeated target-encoding/binning work | Cache-key and monkeypatch-count tests landed |
| R8 | Done in current worktree | Depthwise/levelwise serialization | Closes train-but-cannot-save mode gap | v2 round-trip and corrupt-payload tests landed |
| R9 | Done in current worktree | Benchmark config profiles | Auditable benchmark comparisons | Matched/native profile schema landed |
| R10 | Opt-in done in current worktree | Encoding upgrades | Accuracy on categorical data | Defaults unchanged pending categorical regret/perf gates |
| F1-F3 | Done in current worktree | Retry cap, stopper wrapper, CI Numba cache | Robustness and CI time | Already covered by focused tests |

## Global acceptance rules

These rules apply to every item.

1. **Behavior proof before performance claims.** For any item labeled
   bit-identical, acceptance means `np.array_equal`, not `allclose`, for
   predictions, relevant training histories, and direct tree arrays. Fix the
   same histogram lane, thread count, and resolver settings when making a
   bitwise comparison; otherwise a test may measure an unrelated lane change. If
   an item only passes within tolerance, change its risk label and do not flip a
   default in the same stage.

2. **Performance proof before default flips.** Any default-changing performance
   item must ship with a dated `benchmarks/FINDINGS.md` entry containing:
   commit/revision identifiers, machine/threads, command lines, resolved
   benchmark configs, before/after phase timings, and the decision.

3. **Opt-in before defaults for modeling changes.** Anything that can change
   split choice or predictions for numerical reasons lands behind an explicit
   flag first. A default flip requires a separate gate: regret/quality parity,
   performance win, deterministic reruns, and serialization/refit coverage.

4. **No stale roadmap items.** Before starting a stage, update this file if the
   previous stage changed scope, risk, test count, or ordering.

## Stage order and handoff gates

Order: Stage 0 -> Stage 1 (R8, R1) -> Stage 2 (R9) -> Stage 3 (lane infra,
R2, R3) -> Stage 4 (R4) -> Stage 5 (R7) -> Stage 6 (R5a -> R6 -> R5b ->
optional R5c) -> Stage 7 (R10).

Rationale: R8 is isolated and closes a concrete train-but-cannot-save contract
gap before any tree contracts change; R9 lands before every default-changing
performance decision so Global Rule 2 is satisfiable from the start; R7 lands
before the kernel work so auto/probe experimentation is cheap during the
benchmark-gated stages (its bit-identical gate must pass before R5a
benchmarking begins, so a latent cache bug cannot masquerade as a float32
quality effect); R5b lands after R6 so the uint32 leaf audit runs once against
the final multiclass layout instead of twice.

### Stage 0: keep the plan current

This document is the first artifact in the chain. It should not describe
completed work as remaining work, and it should not promise "risk: none" without
a proof gate.

Acceptance:

- F1-F3 are marked complete.
- R2, R3/R9 ordering, R4, R5, R7, and R10b include the caveats called out in
  review.
- R8 has one clear versioning policy.
- `ROADMAP.md` is the only file changed by this stage unless the user asks for
  implementation work too.

### Stage 1: isolated wins — R8, then R1

Land R8 first: it is isolated, the unsupported path is concrete today
(`save_model` raises for depthwise), and it closes a train/predict/save
contract gap without touching model semantics or requiring any shared
infrastructure. R1 follows (or lands in parallel); neither blocks the other.

Gate to Stage 2:

- R8: depthwise/levelwise models round-trip bitwise for regression, MAE,
  binary, multiclass per-class, categorical missing/unseen cases, and empty
  models; corrupt payload tests reject bad offsets and bad per-tree shape
  counts; `FlatLevelwiseEnsemble` predictions after load equal the pre-save
  per-tree loop bitwise; the save-raises expectation test is replaced with
  round-trip coverage.
- R1: focused binning tests and the full suite pass; a short local
  microbenchmark note shows the expected low-cardinality speedup or explains
  why the item should stop.

### Stage 2: benchmark honesty — R9

Land R9 before any benchmark-gated decision downstream: R3's segmented
`"auto"` threshold, R4's sampler benchmarks, and every R5/R6 default flip all
cite FINDINGS entries that must be profile-labeled to be auditable.

Gate to Stage 3:

- `bench_vs_lightgbm.py --profile matched` and `--profile native` produce
  distinct, resolved configs, with profile resolution applied before
  `_resolve_benchmark_capacity`.
- CSV rows include the resolved knobs for both libraries needed to audit the
  comparison.
- Stale committed CSVs are regenerated with the new schema or removed.

### Stage 3: lane/layout proof infrastructure, then R2, then R3

Build only the verification infrastructure needed for tree-lane work, then land
R2 and R3 one at a time. R9 has already landed, so R3's resolver and any
default threshold can be completed in this stage with profile-labeled
provenance.

The infrastructure has two named deliverables (referenced by R2/R3/R5/R6
acceptance):

- **`benchmarks/ab_compare.py`** — takes two repo paths (use
  `git worktree add <dir> <rev>` for the baseline revision), runs a fixed
  matrix of seeded fits in each ({catboost ordered, catboost non-ordered,
  catboost MAE, lightgbm, hybrid, depthwise} x {RMSE, binary, multiclass
  per_class, multiclass shared_vector} x {unweighted, weighted}, plus one
  config each for goss/mvs/bayesian), saves `predict(Xte)` **and**
  `model_.train_history_` per config, and diffs with `np.array_equal`.
  Comparing training histories catches training-loop divergence that cancels
  at the final prediction; this exact pattern verified the fix round.
- **`tests/test_lane_equivalence.py`** — parametrized equivalence over
  lane-forcing kwargs at 1/2/4 numba threads, asserting `array_equal` trees
  and predictions between lane pairs documented as exact, and `allclose` with
  a tight labeled tolerance for pairs documented as float64-rounding-only
  (level subtraction).

Gate before R2:

- Direct builder tests can compare C-order vs F-order inputs within the same
  thread/lane configuration.
- Existing prefix-vs-segmented coverage is upgraded from `allclose` to
  `array_equal` where bitwise equality is claimed.

Gate after R2:

- C-order and F-order direct-builder tests are bit-identical for oblivious,
  hybrid shared trunk, leafwise prefix, leafwise segmented, and levelwise
  routes that use fixed-column row passes.
- Model-level A/B checks against the pre-R2 revision are bit-identical.
- Profiler shows a measurable win at 200k+ rows, or the change is reverted /
  kept disabled.

Gate after R3:

- Prefix and segmented layouts are bit-identical across the expanded matrix.
- The resolver provably never selects segmented where an eligible fast lane
  would be disabled (see R3).
- Any `"auto"` threshold is chosen from profile-labeled benchmarks and
  documented in `FINDINGS.md`; until that decision, segmented remains an
  explicit opt-in.
- The resolved layout is auditable, either via `auto_params_["tree"]` or a
  documented internal resolver covered by tests.
- Default flip A/B checks remain bit-identical for affected configs.

### Stage 4: sampler optimization

Implement R4 with probability-vector and sampling-distribution tests. Treat the
exact solve as a behavior-perturbing numerical change unless the tests prove
otherwise.

Gate to Stage 5:

- Final probability vectors, row indices, and grad/hess scaling match the old
  implementation within the declared tolerance on edge cases and randomized
  cases.
- Sampling fractions/masses match binomial or mass-based confidence intervals.
- Model quality parity holds on paired-seed MVS and weighted-GOSS configs.
- Sampler phase benchmarks show a real win at the intended large-n sizes. If
  sort cost loses at smaller n, use a thresholded hybrid or keep the old path.

### Stage 5: shared preprocessing — R7

Land R7 before the kernel work so every subsequent benchmark-gated stage
iterates faster on auto/probe fits. Its bit-identical gate must pass before
R5a benchmarking begins, so a latent cache bug cannot masquerade as a float32
quality effect.

Gate after R7:

- Auto-mode and learning-rate probe fits are bit-identical before/after when
  the cache key matches.
- Cache keys include all preprocessing-affecting state: `max_bins`, resolved
  `cat_smoothing`, `target_encoding_mode`, folds, `include_cat_codes`,
  `bin_sample_count`, fit random seed, `cat_features`, input feature count,
  the actual train/eval feature matrices, exact train/eval sample weights when
  present, the train/eval split identity, and the target signature.
- Scalar and multiclass caches are never shared with each other.
- Monkeypatched counts prove the expected reduction in training-prep passes
  without timing assertions.

### Stage 6: histogram stream and multiclass layout work

Order within the stage: R5a -> R6 -> R5b -> optional R5c. Do not combine
float32 gradients, class-minor histograms, uint32 leaf ids, and default flips
in one patch.

Gate after R5a, float32 grad/hess streams:

- The option is explicit and defaults to current behavior.
- Scalar fits are deterministic for the same seed and thread count.
- Well-separated synthetic split tests remain structurally identical.
- Real-data prediction and metric deltas stay within declared tolerances.
- Full regret report passes the quality gate.

Gate after R6:

- Shared-vector multiclass is bit-identical against the old class-major
  histogram layout for predictions and `train_history_`.
- Per-class multiclass is included in the A/B and remains bit-identical, which
  proves the fused-root fork/parametrization worked (see R6).
- Direct old-layout reference tests transpose and compare class-minor
  histograms with `array_equal`.
- FINDINGS records K in `{3, 10}` and shows the expected fused-root/refill win.

Gate after R5b, uint32 leaf ids (audited against the final R6 layout):

- All kernels that read or write `leaf` accept the narrower dtype.
- Training-state returns, `np.bincount` paths, prediction helpers, row-order
  partitioning, and serialization/flat prediction tests still pass.
- Direct tree arrays and predictions are bit-identical to int64 leaf ids for
  the same float dtype.

Gate before any R5/R6 default flip:

- R9 benchmark profiles are in place.
- Regret suite passes.
- Large-n performance improves materially.
- No thread-count determinism regressions.
- The default flip lands in a separate patch from the opt-in implementation.

### Stage 7: modeling upgrades

Land R10 last, behind explicit flags and benchmark gates.

Status in current worktree: R10a and R10b are implemented behind explicit
non-default options. `ts_permutations` defaults to `1`. Target-ordered raw
category codes default to `"off"` and require the explicit
`"leaky_full"` opt-in, which records the full-target leakage policy in model
archives. No categorical modeling default was flipped.

Gate after R10a:

- `ts_permutations=1` is bit-identical to current ordered target statistics.
- Own-label exclusion tests pass for all P values.
- Variance reduction is demonstrated on early-permutation rows.
- Serialization/refit params preserve the new parameter.
- No default flip without regret-suite improvement.

Gate after R10b:

- Target-ordered remaps apply only to the raw-code block, or the encoder
  internals are remapped consistently. Do not mutate the shared `codes` matrix
  before target-stat encoders consume it.
- Training-time raw-code ordering has an explicit leakage policy: ordered /
  out-of-fold remaps, or a deliberately leaky opt-in path with strong
  regularization and benchmark gates. No full-target remap becomes a default
  training feature.
- Ties are deterministic with category code as a secondary key.
- Files that require remaps bump to a format version that older loaders reject
  cleanly.
- LightGBM/hybrid categorical benchmarks improve; CatBoost mode is unchanged.

## R1. Vectorize `_unique_if_at_most` in binning

Problem:

`_unique_if_at_most` in `darkofit/binning.py` is a Python `set` loop called
from `_feature_borders` for every numeric block column. The early exit is good
for continuous columns, but low-cardinality columns run to completion in Python.
Wide dummy/code matrices can spend seconds here.

Implementation shape:

```python
def _unique_if_at_most(values, max_unique):
    if values.size > 4096:
        probe = np.unique(values[:4096])
        if probe.size > max_unique:
            return None
    uniq = np.unique(values)
    return uniq.astype(np.float64, copy=False) if uniq.size <= max_unique else None
```

The probe is safe because seeing more than `max_unique` distinct values in a
prefix proves the full column also exceeds the limit. Input has already been
filtered to finite values by `_feature_borders`.

Acceptance:

- Add a test-local reference implementation matching the old `float()` set
  loop.
- Compare reference and new output on empty arrays, all-equal arrays,
  low-cardinality arrays, high-cardinality arrays, long arrays whose first
  4096 rows already exceed the limit, and long arrays whose first 4096 rows do
  not exceed the limit but the full array does.
- Include signed-zero coverage (`-0.0` and `0.0`) and verify final borders, not
  just unique values.
- Preserve the old dtype contract: returned uniques are `float64`.
- Run focused binning tests and the full suite.
- Record a small local timing note for many low-cardinality columns. This note
  does not need to be a CI assertion.

## R2. F-order `X_binned` for fixed-column row passes

Problem:

Several routing/partition kernels read one fixed feature for many rows. With
C-order `X_binned`, each byte read can pull an entire row cache line. The
booster already creates `X_hist_binned = np.asfortranarray(X_binned)` for
multi-threaded histogram work, but some fixed-column row passes still route
through the C-order array.

Contract constraint:

`X_hist_binned` is currently documented and validated as a histogram matrix, not
a routing matrix. It is only shape-checked against `X_binned`. Do not silently
reuse it for routing, because direct builder callers could pass a same-shape
non-equivalent matrix that is harmless for today's routing semantics but wrong
after R2.

Relevant call sites:

- `_update_leaves_with_split` in the oblivious builder around `tree.py:4132`.
- `_update_leaves_with_split` in the hybrid shared-trunk path around
  `tree.py:4759`.
- `_partition_leaf_segment_rows`, `_partition_leaf_rows`, and
  `_update_leafwise_leaves_with_split` around `tree.py:5232-5246`.
- `_partition_leaf_rows` in the multiclass shared-vector builder around
  `tree.py:5520`.

Implementation shape:

Add a separate internal routing contract, for example an `X_route_binned`
builder kwarg whose documented contract is "value-identical to `X_binned`, but
possibly a different memory layout." The booster may pass
`np.asfortranarray(X_binned)` through this kwarg. Direct builders should keep
`X_route_binned=None` by default, which resolves to `X_binned`. Do not change
prediction traversal or multi-column histogram lanes as part of R2. Numba will
compile separate specializations for C- and F-contiguous arrays.

Acceptance:

- Add direct-builder tests that compare C-order and F-order routing inputs
  within the same thread count and same lane. Do not rely only on 1-thread vs
  multi-thread model tests, because those also change other lanes.
- Add a direct-builder test with a deliberately different same-shape
  `X_hist_binned`; routing must not change unless the explicit routing kwarg is
  used.
- Update builder docstrings so `X_hist_binned` remains histogram-only and the
  new route matrix has its own value-equivalence contract.
- Cover oblivious, hybrid shared trunk, leafwise prefix, leafwise segmented,
  and multiclass shared-vector partitioning where applicable.
- Assert `array_equal` for tree structure, values, returned leaves, leaf sums,
  predictions, and training histories.
- Run the A/B harness or equivalent model-level checks against the pre-R2
  revision.
- Profile the fixed-column partition/update phase at 200k and 500k rows. If
  the profiler does not show a measurable win, do not keep a default-path
  change just because it is theoretically cache-friendly.

## R3. Segmented row layout as the leafwise default at large n

Problem:

The prefix row layout keeps `row_order` globally compact by shifting tails after
splits. The segmented layout avoids those shifts with segment-local partitioning.
The machinery exists, but `"auto"` currently stays on prefix because earlier
large-fit evidence was not decisive.

Implementation shape:

1. First strengthen tests: existing segmented-vs-prefix coverage should use
   `array_equal` for gains, values, leaf sums, and predictions wherever the
   roadmap claims bitwise equivalence.
2. Benchmark prefix vs segmented at:
   `n in {10k, 50k, 200k, 500k}`, `p in {20, 100}`, tree modes
   `{lightgbm, hybrid}`, and thread counts `{1, 4, 8}` where available.
3. If segmented wins only above a size threshold, add a private thresholded
   resolver for `"auto"` rather than a blanket flip.
4. Keep prefix whenever segmented preconditions fail: row sampling, feature
   sampling, `reuse_leaf_histograms=False`, or a non-all-ones feature mask.
5. Keep prefix whenever a fast lane gated on `not use_segmented_rows` would
   otherwise be active. Segmented rows currently disable the full-feature
   positive-split lane (`tree.py:4590`), the row-parallel segment lane
   (`tree.py:4611`), and fused changed-leaf scoring (`tree.py:4800`) — the
   fastest current path for unweighted `Logloss` lightgbm/hybrid fits
   (`hessian_always_positive`). A size-only `"auto"` flip would ship a
   binary-classification regression under a performance banner. This shrinks
   the benchmark regime in step 2 to configs where those lanes are off
   (weighted fits and non-positive-Hessian losses); be prepared for the honest
   outcome that the better investment is making the fast lanes
   segmented-compatible, and record that decision in FINDINGS either way.
6. Define a resolver such as
   `_resolve_leafwise_row_layout(requested, n_samples, n_features, row_indices,
   feature_indices, feature_mask, reuse_leaf_histograms, max_leaves,
   fast_lane_eligible)` and use it as the single source of truth for tests and
   metadata, where `fast_lane_eligible` captures the lane gates from step 5.

Acceptance:

- Prefix and segmented are bit-identical across RMSE, logloss, weighted fits,
  `min_child_samples > 1`, deeper trees, and 1/2/4 thread counts.
- The resolver provably never selects segmented where an eligible
  `not use_segmented_rows` fast lane would be disabled (unit tests on the
  resolver with lane-eligibility fixtures).
- Default flip A/B checks are bit-identical for affected configs.
- `FINDINGS.md` records threshold choice and small-n regressions if any.
- The resolved layout is auditable. If this requires plumbing from tree builder
  to booster metadata, include that plumbing and tests in R3.

## R4. Exact threshold solve for MVS / weighted-GOSS

Problem:

`_mvs_probabilities` and `_weighted_goss_probabilities` currently use fixed
bisection loops over O(n) vector operations. At very large n, this creates
substantial temporary allocation churn every boosting round.

Implementation shape:

For MVS, solve the piecewise-linear equation for
`sum(min(1, importance / theta)) = target`. For weighted-GOSS, derive the exact
piecewise solve from the current predicate:
`sum(min(1, alpha * mass) * mass) = target_mass`.

Important caveat: current MVS code rescales and clips the probability vector
after bisection. Weighted-GOSS also combines deterministic top rows at scale 1
with Bernoulli-sampled "other" rows scaled by `1 / p`. Acceptance should compare
the final observable sampled gradients/hessians and row indices, not only theta
or the returned probability vector. A probability shift near machine precision
can change a Bernoulli draw, so this is not a bit-identical item.

Acceptance:

- Unit tests compare old and new final probability vectors and final
  grad/hess scaling within declared tolerances on ties, all-equal values,
  zero-importance rows, tiny n, `target >= n`, invalid/non-finite inputs, and
  weighted mass edge cases.
- Weighted-GOSS tests cover top/other composition, top-row scale 1, other-row
  `1 / p` scaling, scalar rows, and multiclass shared-row-sample semantics.
- Statistical tests over many seeds confirm realized sample count/mass matches
  the intended target within confidence intervals.
- Existing MVS and weighted-GOSS model configs show paired-seed metric parity.
- The uniform-mass weighted-GOSS fast path is preserved:
  `_weighted_goss_subsample_from_score` intentionally bypasses the general
  weighted solve when mass is constant
  (`_weighted_goss_uniform_mass_subsample`, `booster.py:1337/1406`), and tests
  lock that behavior. The exact solve must not reroute it.
- Benchmarks time the sampler end to end (`_maybe_subsample`, including
  `_weighted_goss_top_indices` top-mass selection — the threshold solve is not
  the only cost center) and show the exact path wins at the intended large-n
  sizes. If sorting loses for small or medium n, use a thresholded hybrid
  resolver.

## R5. Float32 grad/hess streams and narrower leaf ids

Problem:

Histogram kernels repeatedly stream gradient, hessian, and leaf-id arrays. The
current scalar path uses float64 grad/hess and int64 leaf ids, so the hot stream
is 24 bytes per row-feature visit. Narrower streams can reduce bandwidth, but
they touch numba typing, split choice, determinism, and quality.

Stage R5a: float32 grad/hess streams, int64 leaves unchanged.

- Add an explicit `histogram_dtype` option with default `"float64"`.
- Scope R5a to scalar `GradientBoosting` first. Multiclass float32 streams wait
  until R6's shared-vector layout is stable.
- Define `histogram_dtype` as a stream dtype, not a histogram-cell dtype.
- The numerical contract is fixed, not a choice: losses keep producing float64
  into the existing fit-loop buffers, and grad/hess are cast **once per tree**
  into reused float32 buffers immediately before the builder call. Allocating
  the fit-loop buffers as float32 is not acceptable — those buffers feed
  bootstrap and the MVS/GOSS/weighted-GOSS importance computations *before*
  any builder runs (`booster.py:1796-1798`), so float32 there changes sampling
  probabilities and smuggles a sampler change into a bandwidth change. The
  fit loop, losses, samplers, bootstrap, and leaf-value math stay float64;
  one O(n) cast per tree is trivial next to per-feature re-reads.
- Histogram cells stay float64, so accumulation remains float64 after the
  one-time grad/hess quantization.
- Keep leaf ids int64 in this sub-stage. Stream traffic becomes 16 bytes
  (f32 + f32 + i64), not 12 or 10.

Stage R5b: uint32 leaf ids.

- Change leaf arrays only after R5a **and** R6 are stable, so the dtype audit
  runs once against the final multiclass shared-vector layout instead of
  twice.
- Audit every numba kernel and Python helper that indexes, returns, bins, or
  serializes leaves.
- Keep Python-facing returned leaves signed, or explicitly cast at every
  `np.bincount`, prediction, and training-state boundary where callers expect
  signed integer leaves.
- uint32 gives 12-byte streams with f32 grad/hess. uint16 is not acceptable for
  oblivious depth 16 because 2^16 leaves overflow.

Stage R5c: optional float32 histogram cells.

- Only attempt after parent-sibling subtraction error analysis.
- This is a separate quality-risk item and should not be bundled with R5a/R5b.

Acceptance:

- R5a is opt-in, deterministic for same seed/thread count, and quality-gated by
  the regret suite.
- Samplers provably see float64: for a fixed seed, the sampled row sets are
  identical with `histogram_dtype` set to `"float32"` and `"float64"`.
- R5b proves bit-identical behavior relative to int64 leaves for the same
  grad/hess dtype.
- Well-separated synthetic splits remain structurally identical.
- Real datasets use tolerance/metric parity, not bit-identical assertions, for
  float32 vs float64 comparisons.
- No default flip until R9 profiles, regret suite, large-n phase profiles, and
  thread-determinism tests pass in a separate patch.

## R6. Multiclass histogram and gradient layout

Problem:

Shared-vector multiclass histograms are currently class-major
`(K, f, leaf, bin)`, while the inner loops frequently iterate over `k`. That
creates large strides for class-adjacent writes. Grad/hess are also class-major
`(K, n)`, which is good for per-class builders but bad for shared-vector
row-major access.

Implementation shape:

- Scope this item to the existing shared-vector eligibility rules only:
  LightGBM mode, no row sampling, no Bayesian bootstrap, full colsample, and no
  ordered boosting. Per-class builders stay on the existing layout.
- Shared-kernel trap: `_build_multiclass_histograms_counts_into`
  (`tree.py:3081`) serves **both** strategies — it fills shared-vector
  histograms *and* the per-class fused-root buffers (`root_g/root_h/root_c`).
  "Per-class stays bit-identical" therefore requires forking or parametrizing
  that kernel so the per-class fused root keeps its current class-major
  layout, or converting `root_g` and all of its consumers in the same patch.
  Pick one explicitly; do not let the two layouts drift apart silently.
- The booster buffer allocator (`_alloc_multiclass_hist_buffers`), the
  fused-root path, every `_refill_multiclass_*` kernel, every
  `_best_multiclass_*` scorer, and the shared-vector training update helpers
  move together in one atomic patch — a partial conversion leaves the lane
  internally inconsistent.
- Keep the master grad/hess arrays as `(K, n)` so per-class builders retain
  contiguous class rows.
- For shared-vector only, maintain a reused transposed `(n, K)` copy refreshed
  once per boosting round.
- Change shared-vector histogram buffers to class-minor
  `(n_features, max_leaves, max_bins, K)` for `hg/hh`. Keep the shared count
  histogram `hc` as `(n_features, max_leaves, max_bins)`.
- Preserve the existing `hc[f, leaf, bin]` semantics: it counts rows whose
  summed class Hessian is positive, not per-class Hessian mass.
- Preserve per-cell row accumulation order. Iterating rows outer and classes
  inner should keep each `(f, leaf, bin, k)` sum in the same row order.

Acceptance:

- Direct kernel reference tests fill old-layout and new-layout histograms from
  the same inputs, transpose as needed, and assert `array_equal`.
- Shared-vector multiclass model A/B is bit-identical for weighted and
  unweighted configs.
- Per-class multiclass is included in A/B and remains bit-identical, which
  proves the fused-root fork/parametrization worked.
- FINDINGS records fused-root/refill timings for `K in {3, 10}` and confirms
  whether the expected improvement materialized.

## R7. Share preprocessing across `tree_mode="auto"` and learning-rate probes

Problem:

Auto mode and learning-rate probing can refit many candidates on the same rows.
Each fit currently repeats target encoding and binning. With default probe
settings, auto mode can do roughly 21 training-prep passes where two categorical
prep configurations often suffice: ordered/no raw codes and kfold/raw codes.
Note "two" holds only for RMSE/Logloss: `_include_cat_codes` is loss-dependent
(`booster.py:1643`), so MAE/Quantile lightgbm fits use a third config
(kfold/no raw codes) — write the prep-count assertions per loss family.

Implementation shape:

- Add an internal `prep_override=(fitted_prep, X_binned, optional_eval_binned)`
  path to `GradientBoosting.fit` and `MulticlassBoosting.fit`.
- Make the cache a short-lived object scoped to one orchestrated wrapper
  fit/probe call. Do not make it global or reusable across independent fits.
- Cache only when every preprocessing-affecting input matches: `max_bins`,
  resolved `cat_smoothing`, target-encoding mode/folds, `include_cat_codes`,
  `bin_sample_count`, random seed, `cat_features`, input feature count, raw
  training `X` identity/content signature, eval `X` identity/content signature
  when present, exact normalized train/eval sample-weight bytes or digest when
  weights are provided, target arrays, and the resolved train/eval split
  identity.
- Treat `sample_weight=None`, uniform weights, and nonuniform weights as
  different unless a code path proves bitwise equivalence.
- Do not share scalar and multiclass preps. Multiclass target encoders encode K
  targets; scalar/binary encode one target.
- Refit-on-full-data must never reuse selection-split preprocessing.
- Be careful with object identity: candidate models may hold `prep_`. Shared
  fitted preps must be immutable after construction or copied before mutation.

Acceptance:

- Before/after auto-mode and probe-enabled predictions are `array_equal` when
  cache keys match.
- Monkeypatched `FeaturePreprocessor.fit_transform` counts prove the intended
  reduction. For the default categorical auto+probe path, target training-prep
  count is two unless a documented cache-key difference requires more.
- Weighted binning, eval-set transform, categorical missing/unseen prediction,
  and multiclass cases are covered.
- Two candidates with identical preprocessing params but different raw `X`, eval
  `X`, targets, split identity, or nonuniform sample-weight values must not
  share cached bins.
- Incompatible cache keys fail loudly instead of silently reusing wrong binned
  data.

## R8. Depthwise / levelwise serialization

Problem:

`tree_mode="depthwise"` trains and predicts, but save currently raises for
`LevelwiseTree`. All required state is plain arrays:
`node_features`, `node_thresholds`, `values`, `splits_feat`, `splits_thr`, and
`gains`. `FlatLevelwiseEnsemble` already demonstrates the flattened prediction
layout.

Implementation shape:

- Add `tree_kind="levelwise"` and `"levelwise_per_class"`.
- Import `LevelwiseTree` in the serialization module and add both kinds to the
  save and load tree-kind whitelist.
- Pack per-tree `depths`, flat node feature/threshold arrays with node offsets,
  flat values with value offsets, and flat split/gain arrays with split offsets.
- Preserve the actual `LevelwiseTree` shapes. `node_features` and
  `node_thresholds` are depth by max-leaf-width tables, with only
  `2^d` entries meaningful per level; `values` has `2^actual_depth` entries.
- Do not route levelwise payloads through `_pack_nonoblivious`; levelwise tables
  need their own pack/unpack path.
- Reuse existing validator helpers and add per-tree invariants:
  `node_features.shape == node_thresholds.shape`,
  `values_len == 1 << depth`, and matching split/gain lengths.

Version policy:

Keep format version 2 for levelwise tree kinds. Current v2 loaders already
reject unknown tree kinds cleanly; they do not silently mispredict. Bump to v3
only for future payloads older v2 loaders would ignore or misinterpret.

Acceptance:

- Round-trip bitwise tests for depthwise RMSE, MAE, binary, multiclass
  per-class, numeric data, categorical missing/unseen categories, and
  `iterations=0`.
- Corrupt archive negatives for bad offsets, truncated arrays, node-count
  mismatch, value-count mismatch, and split/gain mismatch.
- Post-load flat prediction equals pre-save per-tree prediction bitwise.
- Existing expectation that save raises for depthwise is replaced with
  round-trip coverage.

## R9. Benchmark config profiles

Problem:

`bench_vs_lightgbm.py` still mixes policy and implementation differences:
different bin counts, leaf budgets, learning rates, and l2 defaults can make
comparisons hard to audit. Committed CSVs also need resolved-config provenance.

Implementation shape:

- Add `--profile {matched,native}`, defaulting to `matched`.
- `matched`: same effective bin budget, same fixed learning rate, same leaf
  budget, mapped l2/min-child settings where possible.
- `native`: each library's true defaults. It must disable the current
  matched-leaf mutation in `_resolve_benchmark_capacity`, otherwise DarkoFit
  auto/lightgbm/hybrid modes still inherit LightGBM leaf capacity.
- Run profile resolution before `_resolve_benchmark_capacity`, and make
  `matched` set explicit resolved knobs for both estimators.
- Explicit CLI knobs still override profile defaults.
- Write resolved configs into every result row: profile, bins, learning rate,
  l2/lambda, leaves, min-child settings, threads, and selected DarkoFit mode.
- Normalize `--threads 0` consistently across DarkoFit and LightGBM.

Acceptance:

- Adapter/fake tests verify each profile's estimator kwargs without requiring
  LightGBM to be installed.
- CSV schema tests prove resolved columns are written.
- Stale committed CSVs are regenerated with the new schema or removed.
- FINDINGS uses profile-labeled tables after this lands.

## R10. Encoding upgrades

R10 changes model behavior for accuracy. Keep both sub-items opt-in until they
clear quality gates.

### R10a. Multi-permutation ordered target statistics

Problem:

`OrderedTargetEncoder` uses a single ordered permutation. Rows early in that
permutation get high-variance near-prior encodings. Averaging multiple
permutations can reduce this variance.

Implementation shape:

- Add `ts_permutations`, default `1`.
- `ts_permutations=1` must preserve today's exact seed sequence and output.
- For P > 1, average P ordered-stat columns into the same number of output
  columns as today. Do not expand to P feature columns unless that feature
  expansion is deliberately benchmarked as a separate modeling change.
- Reset per-category running sums/counts for each permutation.
- Prediction-time `sums_` and `counts_` remain full-data totals; serialization
  needs to preserve `ts_permutations`, not P separate prediction totals.
- Thread through `FeaturePreprocessor`, wrappers, serialization prep config,
  and refit params.

Acceptance:

- P=1 A/B is bit-identical.
- Own-label exclusion tests pass for P > 1.
- Variance-shrink property test demonstrates lower variance for early rows.
- Serialization and `get_refit_params` preserve the parameter.
- No default flip without regret-suite improvement on categorical data.

### R10b. Target-ordered raw category codes

Problem:

LightGBM/hybrid mode includes raw category-code features. Current codes are
order-of-appearance, so ordinal splits group unrelated categories. Ordering raw
codes by target statistic can make single-threshold splits meaningful.

Implementation shape:

- Compute a deterministic per-column remap from smoothed target means.
- Define a sample-weighted, leakage-controlled remap plan before coding. Either
  use ordered/out-of-fold training-time raw-code values with a separate
  full-data remap persisted for inference, or keep full-target remaps behind an
  explicit leaky opt-in with smoothing/min-effective-count restrictions and
  benchmark gates.
- Apply the remap only to the raw-code block unless encoder internals are also
  remapped consistently. The target-stat encoders currently consume the same
  `codes` matrix, so do not mutate it before encoder fitting.
- Transform paths must apply the same raw-code remap for both dict and pandas
  lookup paths. Handle fitted missing categories separately from unseen `-1`;
  unseen categories stay `-1 -> NaN`.
- Start scalar-only unless multiclass behavior is explicitly benchmarked and
  specified.
- Persist remaps in the archive.

Version policy:

Files that depend on raw-code remaps must bump to format version 3, because a
v2-era loader could otherwise ignore the new remap arrays and bin unremapped
codes incorrectly.

Acceptance:

- Unit test with monotone category means proves the raw-code split can isolate
  high/low response groups.
- Tie handling is deterministic with category code as secondary key.
- CatBoost mode is unchanged because it has no raw-code block.
- LightGBM/hybrid categorical benchmarks improve or the feature stays opt-in.
- Singleton or low-count categories must not receive training raw-code values
  determined by their own labels unless the feature is explicitly documented and
  accepted as a leaky opt-in.
- Round-trip tests include unseen categories.
- Version bump is written when remaps are present, and old loaders reject the
  file cleanly where feasible to test.

## F1-F3. Completed small follow-ups

These items were follow-ups from fix-round verification and are done in the
current worktree.

- F1: capped consecutive fruitless sampled depth-0 retries in scalar and
  multiclass training. Covered by constant-target/no-signal tests.
- F2: wrapped user `study_stopper` callbacks so `study.stop()` propagates via
  the shared Optuna user attribute used by multiprocess search. Covered by a
  real Optuna callback test.
- F3: GitHub Actions Numba cache. The first draft set
  `NUMBA_CACHE_DIR: ~/.cache/numba` in the env block — Numba joins that value
  literally (`os.path.join`, no `~` expansion), so kernels would have cached
  into a literal `./~` directory while `actions/cache` saved an always-empty
  `$HOME/.cache/numba`. Fixed by exporting the resolved path via
  `$GITHUB_ENV` in a step. Acceptance is **not** workflow inspection: it is an
  observed warm second CI run whose pytest step drops from ~4-5 min toward the
  local warm-cache time.

Do not implement these again. If they are not yet committed when this roadmap
is committed, keep them in the same branch history or note the dependency in
the PR description.
