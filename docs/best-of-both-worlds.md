# Best-of-Both-Worlds Integration Plan

This branch uses upstream `bbstats/chimeraboost@ddaf272` as the trunk and ports
fork work forward behind benchmark-gated seams. The target shape is:

- `tree_mode="catboost"`: upstream/product path; oblivious trees, exact SHAP,
  linear leaves, hierarchical shrinkage, bagging, and current docs/API.
- `tree_mode="lightgbm"`: opt-in non-oblivious level-wise tree path; no exact
  SHAP, no linear leaves, and no hierarchical shrinkage in v1.
- `tree_mode="auto"`: future validation-selected mode, only after broader
  holdout evidence shows it beats the catboost default on the primary metric.

## Integration Roles

The upstream branch is the product trunk. It owns the modern public API
(`n_estimators`, constructor-level `cat_features`), validation defaults, exact
oblivious-tree SHAP, linear leaves, hierarchical shrinkage, bagging, docs, and
CI. The legacy fork is the performance research source: phase timing, compact
binning, histogram kernel specializations, class-major multiclass buffers, and
the non-oblivious `tree_mode="lightgbm"` experiment.

Port fork ideas forward into upstream-shaped code; do not port upstream product
surface backward into the old fork. `tree_mode="catboost"` must preserve the
upstream product guarantees. `tree_mode="lightgbm"` is allowed to be less
feature-complete in v1, but it must fail explicitly for unsupported guarantees
such as exact SHAP, linear leaves, or hierarchical shrinkage.

Default selection is metric-gated, not ideology-gated. The primary decision
criterion is weighted out-of-sample loss when weights exist, ordinary holdout
loss otherwise; speed breaks ties.

## Completed Integration Phases

1. Added a three-way revision benchmark harness that compares upstream,
   legacy fork, and the current integration candidate in isolated
   subprocesses.
2. Added a no-op `tree_mode` seam on the upstream trunk, preserving catboost
   behavior and blocking unimplemented levelwise aliases.
3. Fixed validation/sample-weight semantics before speed work.
4. Ported safe catboost-path performance work: timing diagnostics, compact
   binned dtypes, guarded constant-Hessian histograms, selected-feature and
   selected-row histogram fills, and class-major multiclass buffers.
5. Added opt-in `tree_mode="lightgbm"` with a level-wise tree representation
   for regression, binary classification, and multiclass classification.

## Current Upstream/Fork/Candidate Benchmark

Raw rows and summary rows are tracked in:

- `benchmarks/tri_compare_medium_20260605.csv`
- `benchmarks/tri_compare_medium_summary_20260605.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --upstream /private/tmp/chimeraboost-upstream-ddaf272-bobw \
  --fork /private/tmp/chimeraboost-fork-origin-main-bobw \
  --candidate . \
  --models upstream_matched fork_matched candidate_catboost candidate_lightgbm \
  --datasets friedman_numeric wide_numeric_reg categorical_reg numeric_binary \
    numeric_multiclass categorical_binary categorical_multiclass \
  --sizes medium \
  --seeds 3 \
  --repeat 2 \
  --iterations 300 \
  --patience 25 \
  --threads 4 \
  --weight-modes none stress \
  --csv benchmarks/tri_compare_medium_20260605.csv
```

All 168 rows completed successfully. Ratios below are against
`upstream_matched`; lower is better for both primary metric and fit time.

| Variant | Primary-metric wins/ties | Mean metric ratio | Mean fit ratio | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `candidate_catboost` | 14 / 14 | 1.000 | 1.196 | Preserves upstream quality exactly; speed is mixed and not yet a broad win. |
| `fork_matched` | 2 / 14 | 1.093 | 1.275 | Legacy fork can be faster, but usually gives up holdout quality and often runs to the iteration cap. |
| `candidate_lightgbm` | 0 / 14 | 1.222 | 0.941 | Often uses fewer rounds and can be faster, but still fails the primary-metric gate. |

Decision: keep upstream-shaped `tree_mode="catboost"` as the product/default
path. The legacy fork is useful as a source of implementation ideas, not as the
integration base. `tree_mode="lightgbm"` remains opt-in until it wins weighted
or ordinary holdout loss, not just speed.

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

## Histogram Subtraction Probe

Implemented an experimental full-row oblivious-tree builder,
`build_oblivious_tree_hist_subtract`, using leaf-grouped row order plus
histogram subtraction. It is not wired into `fit`; the production grower still
uses direct per-level histogram rebuilds.

Raw microbenchmark rows are tracked in:

`benchmarks/hist_subtraction_microbench_20260605.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_hist_subtraction.py \
  --rows 50000 250000 \
  --features 12 40 \
  --repeat 5 \
  --csv benchmarks/hist_subtraction_microbench_20260605.csv
```

Summary:

| Rows | Features | Subtract/direct time | Same tree? | Max prediction diff |
| ---: | ---: | ---: | --- | ---: |
| 50k | 12 | 1.33x | yes | 0.0 |
| 50k | 40 | 0.88x | yes | 0.0 |
| 250k | 12 | 1.45x | no | 0.0104 |
| 250k | 40 | 1.08x | yes | 0.0 |

Decision: keep this as an experimental builder and do not route training through
it. The current feature-major direct rebuild is still very hard to beat; the
subtraction path adds row partitioning overhead and can change split decisions
through floating accumulation order. A production version would need a stronger
gate, likely a more integrated grower rewrite that uses row grouping for both
split search and leaf-value accumulation instead of bolting subtraction onto the
existing feature-parallel layout.

## Fused Classification Gradient/Hessian Probe

Prototyped an in-place classification loss pipeline for binary Logloss and
class-major multiclass softmax, then rejected it after before/after timing. The
production code remains on the existing allocating NumPy loss path.

Raw medium classification rows are tracked in:

`benchmarks/fused_grad_hess_medium_20260605.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --fork /private/tmp/chimeraboost-before-fused-grad-29f6c91 \
  --candidate . \
  --models fork_catboost_matched fork_lightgbm_matched candidate_catboost candidate_lightgbm \
  --datasets numeric_binary numeric_multiclass categorical_binary categorical_multiclass \
  --sizes medium \
  --seeds 2 \
  --repeat 2 \
  --iterations 300 \
  --patience 25 \
  --threads 4 \
  --weight-modes none stress \
  --csv benchmarks/fused_grad_hess_medium_20260605.csv
```

The vectorized in-place version preserved metrics, but did not improve wall
clock:

| Mode | Metric ratio | Fit-time ratio | Grad/Hess timing ratio |
| --- | ---: | ---: | ---: |
| `candidate_catboost` vs previous | 0.9998 | 1.2121 | 0.7477 |
| `candidate_lightgbm` vs previous | 0.9998 | 2.2088 | 1.4102 |

Decision: do not merge the fused loss pipeline. Even where the measured
Grad/Hess phase improved, the end-to-end fit did not. The current NumPy path is
fast enough relative to tree building, and a useful replacement likely needs a
deeper fused histogram accumulator rather than only in-place loss arrays.

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

## Shared Multiclass Levelwise Trees

Implemented shared-structure, vector-leaf multiclass trees for
`tree_mode="lightgbm"` when training is full-row and unordered. This replaces
the previous K scalar levelwise trees per boosting round with one shared split
structure and K-dimensional leaf values. The old per-class path remains the
fallback for subsampling or ordered boosting.

Raw medium rows are tracked in:

`benchmarks/shared_multiclass_levelwise_medium_20260605.csv`

Summary versus the previous levelwise gate:

| Dataset | Catboost log loss | Previous levelwise | Shared vector levelwise | Fit result |
| --- | ---: | ---: | ---: | --- |
| `numeric_multiclass` | 0.4718 | 0.5121 | 0.4847 | Quality much closer to catboost, but fit slower than old levelwise |
| `categorical_multiclass` | 0.5572 | 0.5866 | 0.5824 | Slight quality and speed improvement over old levelwise |

Decision: keep the shared vector tree for opt-in levelwise multiclass because it
moves the quality gap in the right direction and preserves the fallback path.
Do not use it to justify `auto` or default selection yet; catboost still wins
the primary metric.
