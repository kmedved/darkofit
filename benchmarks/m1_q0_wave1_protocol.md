# Wave 1 M1/Q0 protocol

_Frozen before outcome access on 2026-07-20._

## Purpose and evidence boundary

This Tier-E campaign supplies two inputs to the owner-facing G-M portfolio
decision in `COUNTERPUNCH_PLAN.md`:

1. **M1** characterizes current large-n matched-capacity product-path
   throughput for DarkoFit and pinned ChimeraBoost, including a causal
   quantized-versus-float comparison inside the same ChimeraBoost source.
2. **Q0 profile** measures where DarkoFit's current scalar CatBoost-mode fit
   spends wall time and screens whether a private quantization prototype could
   plausibly clear a speed budget declared below.

The campaign is descriptive engineering evidence. It cannot authorize a
public option, a default change, a shipping claim, or fresh/lockbox access.
M1 does not compare byte-identical preprocessing: DarkoFit retains its public
200,000-row numeric-border sample while ChimeraBoost retains full-data border
construction. The optional ChimeraBoost 0.15 arm is not included, so this
campaign may not attribute total movement since the historical comparison to
0.18 alone.

## Frozen source boundary

- DarkoFit package source:
  `726e5d8e6131c580bce948db833a5007d0692dca`, the single clean
  post-H1-hygiene pin recorded in `h1_hygiene_audit_result.md`.
- ChimeraBoost package source:
  `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`
  (`v0.18.0-6-gf14be60`).
- The runner and this protocol are hash-bound into each raw artifact. The
  execution harness, DarkoFit source worktree, and ChimeraBoost source tree
  must all be clean and unchanged across every worker.

Workers import DarkoFit from a detached clean worktree at the exact package
pin, not from the later documentation/benchmark harness checkout. Numba and
Python caches live outside both source trees. Every worker is a fresh process.
Timed M1 and Q0 workers run sequentially on an otherwise idle machine and may
not overlap each other or unrelated heavy work.

## Shared deterministic workload

The data generator is copied into the runner so its identity does not depend
on whichever package source is first on `PYTHONPATH`:

- NumPy `default_rng` seed `20260717`;
- 24 independent standard-normal numeric features;
- signal
  `1.4*x0 - 0.9*x1 + 0.35*x2*x3 + 0.2*x4**2`;
- target equal to signal plus independent Normal(0, 0.5) noise;
- training sizes 500,000 and 1,000,000 rows; and
- the following 100,000 generated rows as the holdout at each size.

All arms use 300 constant-leaf symmetric trees, learning rate 0.1, depth 6,
L2 1, 128 bins, full row and feature fractions, minimum child weight 1,
ordered boosting off, early stopping off, seed 4, and 14 Numba threads.
DarkoFit additionally uses `tree_mode="catboost"`,
`min_child_samples=1`, `linear_leaves=False`, no train-loss evaluation, and
no diagnostic warnings. ChimeraBoost additionally fixes
`linear_leaves=False`, `cross_features=False`,
`cat_combinations=False`, and no ensemble.

The inherited historical protocol used 18 threads on an 18-logical-CPU
machine. A pre-freeze, 5,000-row API smoke on the current execution machine
found only 14 logical CPUs; requesting 18 made TBB warn and silently cap its
worker pool. This protocol therefore fixes all arms and thread-limit
environment variables at the honest current-machine budget of 14. That
departure is another reason not to attribute movement from the old result;
only current within-machine arm ratios are compared.

Each fresh worker first performs a same-arm, same-kernel 5,000-row,
three-tree fit and 256-row prediction warmup outside timing. The worker
records the environment, actual package import path,
fitted tree/thread/depth/learning-rate metadata, full prediction and data
fingerprints, holdout RMSE, peak RSS, common pre-predict pickle size, native
archive size where the product exposes one, and phase timing where available.

## M1 arms and order

The three primary arms are:

1. `darkofit_float`: DarkoFit's current float-histogram product path;
2. `chimeraboost_quantized`: pinned ChimeraBoost with
   `quantize_gradients=True`; and
3. `chimeraboost_float`: the same pinned ChimeraBoost source with
   `quantize_gradients=False`.

At each training size, run the six permutations of these three arms as six
balanced order blocks, for 36 fresh timed workers total. Pair ratios only
within the same block and size. Report every repeat, paired medians,
`IQR / median`, equal-size geometric means, holdout quality, prediction
timing, peak RSS, serialized size, metadata, and DarkoFit phase timing.

M1 publishes, without converting them into certification gates:

- whether DarkoFit is faster than current quantized ChimeraBoost at both sizes
  and in the equal-size geometric mean on this machine;
- the DarkoFit/ChimeraBoost float and quantized fit ratios; and
- the causal current-source quantized/float fit and RMSE ratios.

For G-M only, M1 supplies a **material donor signal** when all integrity
checks pass, quantized/float paired timing is stable
(`IQR / median <= 0.10`), its equal-size geometric-mean fit ratio is at most
`0.90`, neither size exceeds `1.02`, and quantized/float holdout RMSE is at
most `1.002` at each size. This is evidence that the mechanism can matter in
the donor; it is not evidence that DarkoFit has the same bottleneck.

## Q0 production profile and diagnostic decomposition

Q0 uses the same two training sizes and DarkoFit configuration, but 40 trees
per fit and three reciprocal blocks. Each block contains:

- `production`: the current fused histogram-plus-split path; and
- `unfused_reference`: the private behavior-exact reference path with
  `fused_oblivious_kernel=False`.

Orders alternate production/reference, reference/production,
production/reference. The production path is the only source of end-to-end
shares and the funding projection. Python timing shims record calls and
inclusive wall time for:

- fused histogram plus split;
- sibling subtraction;
- leaf-value calculation; and
- leaf routing.

DarkoFit's built-in timer supplies preprocessing, gradient/Hessian
preparation, whole tree build, and train-update time. Because the current
14-thread product kernel fuses histogram construction and split scanning,
those two production components are deliberately reported as one. The
unfused reference is run only to report separate histogram-construction and
split-search diagnostics and to verify prediction identity. Its slower wall
time and component shares must not enter the Q funding projection. Sibling
subtraction is expected to have zero production calls at 14 threads; that
zero is an engagement fact, not an inferred cost estimate.

All production repeats must engage the fused kernel, all unfused repeats must
avoid it and exercise both reference components, sibling subtraction must
remain inactive, behavior fingerprints must be stable and identical across
the two modes, and workers must complete without stderr.

## Predeclared Q speed budget and stop rule

The minimum worthwhile private-prototype result is a repeatable **10% lower
end-to-end fit time**:

- equal-size geometric-mean candidate/control fit ratio at most `0.90`;
- no size ratio above `1.02`;
- paired `IQR / median <= 0.10`;
- exact deterministic predictions and all required safe-accumulation and
  fallback invariants.

This budget is fixed before the Q0 profile is inspected. It matches the
maintenance burden of a second arithmetic representation: a smaller gain is
not enough to justify public implementation.

The Q0 screen uses a deliberately conservative donor prior of `1.30x` for
the quantization-eligible fused kernel. For production eligible share `s`,
the projected end-to-end ratio is:

`(1 - s) + s / 1.30`.

The equal-size geometric mean of those projected ratios must be at most
`0.90`. With equal shares this requires at least
`0.10 / (1 - 1/1.30) = 0.433333...` of fit wall time in the eligible fused
kernel. The artifact also reports the infinite-kernel-speed ceiling
`1 - s`, but that optimistic ceiling cannot fund work.

If integrity fails, Q0 is inconclusive and must be repaired without inspecting
or relaxing the speed rule. If integrity passes but the conservative
projection misses 10%, close Q before a prototype. If it clears 10%, Q
becomes eligible for the G-M portfolio decision; that result alone does not
authorize implementation. A funded prototype is a new, private measurement
against this same 10% budget.

## Artifacts and terminal handling

M1 and Q0 each produce one create-only JSON artifact. The runner records this
protocol hash, its own hash, exact sources, machine/dependency details,
coordinates, full worker stdout/stderr, raw results, and deterministic
analysis. Each completed campaign gets a result note and a 12-field testing
log entry. Results are published once; failures are recorded rather than
rerun to improve an outcome.
