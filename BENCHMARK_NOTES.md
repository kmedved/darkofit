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

`tree_mode="levelwise"` is accepted by the benchmark harness as an alias for
the experimental depth-wise builder (`tree_mode="depthwise"` internally). A
focused medium-size probe on 2026-06-19 used:

```bash
/Users/kmedved/.venvs/darko311/bin/python benchmarks/bench_vs_lightgbm.py \
  --tree-mode levelwise \
  --sizes medium \
  --datasets friedman_numeric numeric_binary numeric_multiclass wide_numeric_reg \
  --seeds 2 \
  --threads 8 \
  --iterations 800 \
  --patience 50 \
  --repeat 2 \
  --csv /tmp/chimeraboost_levelwise_medium.csv
```

Compared with the same current-main CatBoost-mode run, levelwise reduced the
round count and fit time on `friedman_numeric` and `numeric_multiclass`, but it
was not a safe default candidate: `numeric_binary` fit time was
neutral-to-slower, and `wide_numeric_reg` RMSE regressed from 40.08 to 88.29
while fitting 1.23x slower.

A follow-up flat-prediction pass added level-wise ensemble flattening and reran
the same medium probe. Prediction improved 3.38-4.01x over the previous
levelwise path (`friedman_numeric` 0.0054s -> 0.0013s, `numeric_binary` 0.0276s
-> 0.0082s, `numeric_multiclass` 0.0420s -> 0.0105s, `wide_numeric_reg`
0.0135s -> 0.0038s). Treat depth-wise trees as an experimental comparison lane
until the wide-regression quality/defaults issue is addressed.

A first quality sweep pointed to a regression-specific depth/default issue, not
a global depth-wise default. On the same medium benchmark matrix, `depth=2`
with the original 800-round budget improved `wide_numeric_reg` RMSE from 88.29
to 52.71 and fit speed from x1.52 to x4.21 versus LightGBM, but it damaged
classification quality (`numeric_binary` F1 0.9166 -> 0.8466,
`numeric_multiclass` F1 0.8786 -> 0.8175). With a 1500-round budget, compact
two-seed `wide_numeric_reg` probes found `depth=2` near 45.44 RMSE, much closer
to CatBoost mode's 40.08 than the depth-6 path.

A follow-up regression-only matrix over `diabetes_resampled`, `friedman_numeric`,
`wide_numeric_reg`, and `categorical_reg` at small/medium sizes found that
shallow depth improved depth-wise RMSE versus the old depth-6 default in every
dataset/size cell. Best depth-wise configurations used depth 2 or 3; L2 did not
produce a stable cross-dataset rule. The current estimator therefore resolves
omitted `depth` to 2 only for `tree_mode="depthwise"` RMSE regression. Explicit
depths, `depth="auto"`, CatBoost mode, LightGBM mode, and depth-wise
classification keep their existing behavior.

Validation after the default-rule change reran the benchmark harness with
`--tree-mode levelwise` and no explicit `--depth`, so the estimator resolved
depth per task. At medium size and two seeds, the depth-wise RMSE default beat
LightGBM on `diabetes_resampled` (+2.42%), `friedman_numeric` (+3.45%), and
`wide_numeric_reg` (+42.95%), while `categorical_reg` trailed LightGBM by
6.42% but remained far better than the old depth-6 depth-wise regression lane.
At small size and two seeds, it beat LightGBM on all four regression tasks. A
large one-seed smoke on `friedman_numeric`, `wide_numeric_reg`, and
`categorical_reg` kept fit-speed ratios above x1.5 and quality roughly tied or
better (`wide_numeric_reg` RMSE 42.69 vs LightGBM 68.57).

## Experimental LightGBM-Mode Hooks

The current main branch includes a few infrastructure hooks that are intentionally
not default speed wins:

- `leafwise_row_layout="segmented"` is an internal opt-in row layout for
  leaf-wise trees. It has exact parity coverage, but default full-fit benchmarks
  regressed, so `auto` keeps the normal prefix layout.
- `fused_changed_leaf_scoring=True` is routed for the narrow scalar
  LightGBM-mode lane where retesting kept exact predictions and improved speed:
  unweighted logloss fits with full rows/features, positive Hessians, no
  row-parallel buffers, no split-score noise, and more than two threads. Keep
  it off for `random_strength > 0` because noisy split scoring dominates that
  path.
- Full-row/full-feature positive-Hessian split scoring reuses the parent
  gradient/Hessian totals once per leaf instead of recomputing them for every
  feature. The invariant only holds in that full-feature lane. Direct
  leaf-wise tree timings at 120k x 80, 64 leaves, 128 bins kept identical split
  signatures and improved the 2-thread median from 0.0295s to 0.0282s; the
  4-thread median was neutral because changed-leaf scoring is dominated by the
  fused/feature-parallel lanes.
- `multiclass_tree_strategy="shared_vector"` is the `auto` default for
  compatible LightGBM-mode multiclass fits. The full harness retest found the
  shared-vector path materially faster on numeric multiclass, neutral-to-better
  on F1 in the synthetic multiclass cases, and already equivalent to categorical
  auto where it was previously selected. Explicit
  `multiclass_tree_strategy="per_class"` remains available as a fallback and
  comparison lane.
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

- reduce numeric and multiclass round count with a stronger tree strategy that
  does not lose the current CatBoost-mode wide-regression quality;
- redesign leaf-wise histogram refill, scoring, and reuse so large fits do less
  repeated work per split.

Start those on a fresh branch with a fresh profile rather than by widening the
current experimental hooks.
