# Q post-dispatch packed-histogram microprototype

This normal development benchmark asks one narrow causal question: does a
packed integer histogram kernel reduce end-to-end fit time enough to justify a
real DarkoFit quantization project after the fused/unfused automatic dispatch
shipped?

It is not a public implementation. The candidate is installed only inside
fresh benchmark workers and supports the first Q lane only:

- scalar unweighted RMSE;
- constant Hessian;
- CatBoost-style oblivious trees;
- full rows and features, without sampling or split randomness; and
- the current 14-thread macOS-arm64 workload.

The control uses `oblivious_kernel="auto"` and therefore exercises the current
post-dispatch engine independently at each shape. The candidate replaces only
the full-row unit-Hessian fused histogram/split kernel and explicitly requests
that lane. It quantizes each tree's gradient once, packs the signed gradient
into the high 32 bits and the exact unit-Hessian count into the low 32 bits,
and retains float64 gradients for leaf values. The benchmark-local prototype
adapts ChimeraBoost's Apache-2.0 packed-int64 design to DarkoFit's row-major
kernel and split-legality rules; it does not import or execute ChimeraBoost.

The two Q0 shapes remain fixed: 500,000 and 1,000,000 training rows, 24
features, 128 bins, depth 6, 40 trees, and 100,000 holdout rows. Three
reciprocal blocks alternate control/candidate order. Each arm runs in a fresh
worker after an untimed compile warmup. Data, model seeds, total CPU, and
thread count are shared.

## Arithmetic and reproducibility checks

- `qmax = min(32767, floor((2**31 - 1) / n_rows))`;
- therefore `n_rows * qmax <= 2**31 - 1`, so every signed gradient
  histogram/prefix sum fits in 32 bits;
- every packed low-half count is at most `n_rows < 2**32`, so it cannot carry
  into the signed high half;
- counter-based stochastic rounding is keyed by tree and row, independent of
  work scheduling;
- a slow unpacked integer oracle covers packing, accumulation, and unpacking;
- exactly representable inputs reproduce the float split decision; and
- repeated candidate workers at a fixed size must produce one prediction and
  fitted-structure fingerprint.

## Decision

The historical Q budget remains the deliberately demanding maintenance bar:

- equal-size geometric mean of the per-size paired-median end-to-end fit
  ratios at most `0.90`;
- no size ratio above `1.02`;
- paired-ratio `IQR / median <= 0.10` at each size;
- all arithmetic, coverage, dispatch, engagement, and determinism checks pass.

Meeting the bar funds design of a real private Q1 implementation; it does not
ship a parameter or default. Missing it closes Q at the microprototype stage.
RMSE ratios, prediction time, and peak RSS are descriptive telemetry and must
be disclosed either way.
