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

