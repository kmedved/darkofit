# Benchmark Notes

This file is the current checkpoint for ChimeraBoost benchmark work. For the
longer historical speed investigation, see [benchmarks/FINDINGS.md](benchmarks/FINDINGS.md).

## Current Default Lanes

`tree_mode="catboost"` remains the default CatBoost-like path. The recent
"best of both worlds" work was aimed at preserving bbstats v2 behavior while
keeping exact, gated speed wins from this fork where they were proven.

`tree_mode="lightgbm"` is ChimeraBoost's leaf-wise, non-oblivious tree builder.
It is not LightGBM model compatibility. It is a native ChimeraBoost training
mode intended to be compared against LightGBM-style leaf-wise boosting.

## Experimental LightGBM-Mode Hooks

The current main branch includes a few infrastructure hooks that are intentionally
not default speed wins:

- `leafwise_row_layout="segmented"` is an internal opt-in row layout for
  leaf-wise trees. It has exact parity coverage, but default full-fit benchmarks
  regressed, so `auto` keeps the normal prefix layout.
- `fused_changed_leaf_scoring=True` is an internal opt-in fused refill/subtract
  plus split-scoring path for scalar LightGBM-mode trees. It improved a direct
  tree microbenchmark but regressed a full large `numeric_binary` fit, so it is
  not routed from public estimators.
- `multiclass_tree_strategy="shared_vector"` is an explicit classifier option
  for compatible LightGBM-mode multiclass fits. Forced shared-vector trees were
  slower on numeric multiclass in the focused benchmark, so `auto` preserves the
  previous default behavior.
- Histogram buffers are interleaved lane views of one
  `(features, leaves, bins, 2-or-3)` base array for fits at <= 4 threads, so
  each bin's grad/hess(/count) share a cache line. Results are bitwise
  identical (the kernels are layout-agnostic and summation order is
  unchanged). Measured end-to-end on the Apple-silicon dev box at 200k x 40:
  +27% at 1 thread, +6% at 2 threads, neutral at 4; at 8+ threads the effect
  was neutral-to-negative, so larger fits keep separate buffers.
- `histogram_parallelism="row"` is an opt-in lane that fills histograms with
  row-chunked thread-local accumulators (one read of grad/hess/leaf per scan)
  instead of the feature-parallel kernels (one read per feature). It has exact
  parity coverage at the tree level. On the Apple-silicon dev machine it
  measured 5-20% slower than feature-parallel at 400k x 40 and 1M x 80 (the
  redundant streams stay cache-resident there), so `auto` keeps the
  feature-parallel kernels. Re-evaluate on machines with smaller caches /
  lower memory bandwidth, where FINDINGS.md attributes the 500k-row
  per-iteration deficit to exactly these redundant reads.

Treat these as scaffolding for future architecture work, not as current default
performance claims.

## Fair LightGBM-Mode Benchmarking

For LightGBM-mode comparisons, match leaf capacity explicitly:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_vs_lightgbm.py \
  --tree-mode lightgbm \
  --sizes medium large \
  --datasets friedman_numeric wide_numeric_reg categorical_reg numeric_binary numeric_multiclass \
  --seeds 3 \
  --threads 8 \
  --iterations 800 \
  --patience 50 \
  --chimera-num-leaves 64 \
  --lightgbm-num-leaves 64 \
  --repeat 2 \
  --csv /tmp/chimeraboost_lightgbm_mode.csv
```

The benchmark harness also has `--match-lightgbm-leaves`, which defaults an
unspecified ChimeraBoost leaf count to the LightGBM leaf count in LightGBM mode.
Passing both values explicitly is still the least ambiguous recipe for reports.

For ChimeraBoost timings, use a warm numba cache and `--repeat >= 2`. Cold-cache
or single-repeat timings can include one-time numba compilation and should not be
used for speed conclusions.

## Current Performance Interpretation

The recent LightGBM-mode optimization probes did not find a small default kernel
change worth promoting. The useful conclusions are:

- Matching leaf capacity removes an old comparison confounder.
- Quality and best-iteration behavior are stable against the opt-in scaffolding.
- The default LightGBM-mode speed frontier is now architectural, not cleanup-level.

The next serious optimization tracks are:

- reduce numeric and multiclass round count with stronger tree strategy;
- redesign leaf-wise histogram refill, scoring, and reuse so large fits do less
  repeated work per split.

Start those on a fresh branch with a fresh profile rather than by widening the
current experimental hooks.

