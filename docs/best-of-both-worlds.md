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

## Levelwise Tuning Follow-Up

Raw tuning rows are tracked in:

- `benchmarks/levelwise_tuning_numeric_binary_capacity_20260605.csv`
- `benchmarks/levelwise_tuning_numeric_binary_leafiters_20260605.csv`
- `benchmarks/levelwise_tuning_mixed_best_candidates_20260605.csv`

The tuning script is:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_levelwise_tuning.py
```

Result: no safe levelwise default change yet. On medium `numeric_binary`, deeper
levelwise trees plus lower learning rate improved log loss from the original
mode-gate row (`0.2123`) to `0.1853`, but still trailed catboost/upstream
(`0.1679`) and cost more rounds. Applying that best numeric-binary candidate
across the mixed panel worsened regression materially (`friedman_numeric`
`1.2679` RMSE and `wide_numeric_reg` `96.58` RMSE versus catboost `1.0840` and
`40.50`) and did not beat catboost on classification log loss.

Decision: keep `tree_mode="lightgbm"` opt-in with no mode-specific public
defaults for now. The next useful work is not another coarse global default
sweep; it is either a quality fix in the levelwise implementation/objective or
the separate large-data histogram-scaling work.

## Row-Parallel Histogram Probe

A first thread-local row-parallel histogram implementation was tested against the
current feature-parallel kernel and rejected. The implementation was correct
(`np.allclose` to feature-parallel histograms), but slower:

- 12 features, 300k rows, 16 leaves, 128 bins, 4 threads:
  feature-parallel `0.0031s` best vs row-parallel `0.0049s` best.
- 2 features, 1M rows, 16 leaves, 128 bins, 4 threads:
  feature-parallel `0.0045s` best vs row-parallel `0.0059s` best.
- A row-major binned view did not fix it; feature-parallel still won on 2,
  12, and 16 feature microbenchmarks.

Decision: do not merge this row-parallel/thread-local histogram shape. The
existing feature-major layout is still extremely efficient. The large-data
per-iteration gap likely needs a different attack, such as histogram
subtraction/leaf-partitioned row order, lower histogram width for encoded
target-stat columns, or a more substantial grower rewrite.

## Target-Stat Bin Budget Probe

Implemented an opt-in `max_bins_ts` parameter that caps only ordered-target-stat
encoded categorical columns. Raw numeric columns still use `max_bins`, and
`max_bins_ts=None` preserves the old behavior exactly.

Raw medium categorical rows are tracked in:

- `benchmarks/ts_bin_cap_default_medium_20260605.csv`
- `benchmarks/ts_bin_cap64_medium_20260605.csv`
- `benchmarks/ts_bin_cap32_medium_20260605.csv`

Summary:

| Dataset | Default metric | `max_bins_ts=64` | `max_bins_ts=32` | Decision |
| --- | ---: | ---: | ---: | --- |
| `categorical_reg` RMSE | 2.5458 | 2.5362 | 2.5816 | 64 slightly better |
| `categorical_binary` log loss | 0.2546 | 0.2538 | 0.2535 | 32/64 slightly better |
| `categorical_multiclass` log loss | 0.5572 | 0.5660 | 0.5611 | default better |

Decision: keep `max_bins_ts` opt-in. It is a useful categorical-speed/regularity
knob, but not a universal default because multiclass quality regressed in the
first gate.
