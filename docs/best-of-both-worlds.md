# Best-of-Both-Worlds Integration Plan

This branch uses upstream `bbstats/chimeraboost@ddaf272` as the trunk and ports
fork work forward behind benchmark-gated seams. The target shape is:

- `tree_mode="catboost"`: upstream/product path; oblivious trees, exact SHAP,
  linear leaves, hierarchical shrinkage, bagging, and current docs/API.
- `tree_mode="lightgbm"`: opt-in non-oblivious level-wise tree path; no exact
  SHAP, no linear leaves, and no hierarchical shrinkage in v1.
- `tree_mode="auto"`: future validation-selected mode, only after broader
  holdout evidence shows it beats the catboost default on the primary metric.

## Completed Integration Phases

1. Added a three-way revision benchmark harness that compares upstream,
   fork-style modes, and the current integration candidate in isolated
   subprocesses.
2. Added a no-op `tree_mode` seam on the upstream trunk, preserving catboost
   behavior and blocking unimplemented levelwise aliases.
3. Fixed validation/sample-weight semantics before speed work.
4. Ported safe catboost-path performance work: timing diagnostics, compact
   binned dtypes, guarded constant-Hessian histograms, selected-feature and
   selected-row histogram fills, and class-major multiclass buffers.
5. Added opt-in `tree_mode="lightgbm"` with a level-wise tree representation
   for regression, binary classification, and multiclass classification.

## Current Mode-Gate Benchmark

Raw rows are tracked in:

`benchmarks/best_of_both_worlds_mode_gate_medium_20260605.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --upstream /private/tmp/chimeraboost-upstream-ddaf272 \
  --candidate . \
  --models upstream_matched candidate_catboost candidate_lightgbm \
  --datasets friedman_numeric wide_numeric_reg numeric_binary numeric_multiclass \
    categorical_binary categorical_multiclass \
  --sizes medium \
  --seeds 1 \
  --repeat 2 \
  --iterations 300 \
  --patience 25 \
  --threads 4 \
  --weight-modes none uniform \
  --csv /tmp/cb_p5_mode_gate_medium.csv
```

Summary:

| Dataset family | Winner on primary metric | Levelwise behavior |
| --- | --- | --- |
| Friedman regression | upstream/candidate catboost tie | Faster than candidate catboost, worse RMSE |
| Wide numeric regression | upstream/candidate catboost tie | Similar speed, much worse RMSE |
| Numeric binary | upstream/candidate catboost tie | Faster, fewer rounds, worse log loss |
| Numeric multiclass | upstream/candidate catboost tie | Faster, fewer rounds, worse log loss |
| Categorical binary | upstream/candidate catboost tie | Faster, fewer rounds, worse log loss |
| Categorical multiclass | upstream/candidate catboost tie | Similar speed, fewer rounds, worse log loss |

Decision from this gate: keep `tree_mode="catboost"` as the default. The
levelwise path is real and useful as an opt-in research/performance mode, but it
does not yet clear the out-of-sample metric bar for `auto` or default selection.

## Next Work

1. Retune levelwise-specific defaults (`min_child_weight`, `l2_leaf_reg`, depth,
   learning rate, and early stopping) against the revision harness.
2. Add row-parallel/thread-local histogram kernels if 500k-scale runs still show
   per-iteration scaling inversion after levelwise tuning.
3. Add vector-valued multiclass trees only after the scalar levelwise mode has a
   stable quality/speed profile.
4. Consider `tree_mode="auto"` only after broader multi-seed and large-data
   benchmarks show levelwise wins primary weighted holdout metrics often enough
   to justify the added default complexity.
