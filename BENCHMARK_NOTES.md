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

## Deferred Speed Gates

Paged/fused leaf-wise building is still the right xlarge-speed direction, but it
is not promoted by this pass. The current code already has histogram reuse,
smaller-child refill/subtraction, an opt-in segmented row layout, row-parallel
histogram buffers, narrow fused changed-leaf scoring, and fused multiclass root
histograms. The next win is therefore a larger builder redesign: page row
segments and histogram state together, fuse refill plus scoring across changed
leaves, and preserve exact row-order/leaf-update semantics.

Promotion requires the leaf-wise phase profiler and the end-to-end benchmark
suite to show lower tree-build time on large/xlarge cases without quality,
prediction, save/load, or flat-prediction regressions. A small extra hook should
stay experimental unless it beats the current narrow fused/row-layout lanes on
the same warm-cache profile.

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
unspecified ChimeraBoost leaf count to the LightGBM leaf count for
`tree_mode="lightgbm"`, `tree_mode="hybrid"`, and leaf-wise auto candidates.
Passing both values explicitly is still the least ambiguous recipe for reports.

For ChimeraBoost timings, use a warm Numba cache and `--repeat >= 2`.
Cold-cache or single-repeat timings can include one-time Numba compilation and
should not be used for speed conclusions.

## Default-Regret Benchmarking

Use the revision harness in policy-suite mode when the question is whether a
public default policy improved, rather than whether one checkout beat another:

```bash
python benchmarks/bench_compare_revisions.py \
  --policy-suite default-regret \
  --candidate . \
  --datasets all \
  --sizes small medium large \
  --seeds 3 \
  --weight-modes none uniform stress \
  --repeat 2 \
  --threads 8 \
  --csv benchmarks/default_regret_raw.csv
```

Then summarize the raw rows by default regret:

```bash
python benchmarks/default_regret_report.py \
  benchmarks/default_regret_raw.csv \
  --default-policy candidate_default \
  --output-csv benchmarks/default_regret_cases.csv
```

The report compares `candidate_default` against the best policy available for
each matched dataset/size/seed/weight case, then reports median, p90, and worst
quality regret plus Pareto-dominated cases. Treat this as the default-change
decision layer; use the raw CSV for drill-down when a worst case needs profiling.

## Promotion Contract

Use the benchmark suite as the gate for any default-facing model, tree-mode, or
speed claim. A change is eligible for promotion only when all of these hold:

- It is measured on warm Numba caches with `--repeat >= 2` for ChimeraBoost and
  at least three seeds for quality-sensitive comparisons.
- It reports package-default, equal-capacity or equal-compute, and explicit-lane
  comparisons when the change affects public defaults.
- It includes weighted and unweighted cases, and the primary decision metric is
  the task-appropriate weighted held-out loss or score.
- It reports fit time, prediction time, best round, preprocessing time, and
  quality regret; speed-only wins do not promote when quality regret worsens.
- It preserves raw per-case CSV rows so regressions can be paired by dataset,
  size, seed, task, and weight mode.
- It is demoted or kept experimental when the default-regret report shows a new
  worst-case regression, an increased p90 quality regret, or materially more
  Pareto-dominated cases without an explicit product reason.

## Deferred Accuracy Gates

Learned missing-value direction remains a promising missing-heavy-data idea, but
it is not promoted by this pass. Missing numeric values are binned as the top
bin, so current tree prediction routes them to the `> threshold` side. A real
learned-direction implementation must carry, at minimum, a per-split direction
bit and the feature's missing-bin id through tree scoring, tree objects,
flattened prediction kernels, and model serialization. A partial scorer-only
change would be incorrect at prediction time because finite high bins and the
missing top bin cannot be distinguished from the threshold alone.

Promotion requires a missing-heavy benchmark slice that proves lower held-out
loss without regressing non-missing cases, plus save/load and flat-prediction
parity for every tree representation the feature supports.

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

## Distributional Regression Benchmark

Native Gaussian distributional regression is measured with:

```bash
python benchmarks/bench_distributional.py \
  --datasets synthetic_100k synthetic_500k \
  --models chimera_gaussian chimera_gaussian_es \
           chimera_gaussian_es_calibrated chimera_rmse_const_sigma \
           chimera_quantile_pair ngboost catboost_uncertainty lightgbm_twin \
  --seeds 0 1 2 \
  --iterations 80 \
  --early-stop-iterations 400 \
  --early-stopping-rounds auto \
  --validation-fraction 0.1 \
  --learning-rate 0.06 \
  --num-leaves 31 \
  --threads 8 \
  --csv benchmarks/distributional_raw.csv \
  --markdown benchmarks/distributional_summary.md
```

The benchmark reports validation NLL, Gaussian CRPS, empirical 90% interval
coverage, coverage binned by predicted sigma, mean interval width, fit time,
and prediction time on warm ChimeraBoost kernels. Optional competitors are
soft imports: NGBoost, CatBoost `RMSEWithUncertainty`, and the LightGBM
twin-model variance baseline print explicit skip rows when their packages are
unavailable.

Full local promotion run after installing `ngboost==0.5.11`,
`catboost==1.2.10`, and `lightgbm==4.6.0`: all comparison lanes ran
successfully. Raw per-seed rows are in
`benchmarks/distributional_raw.csv`; the generated table is in
`benchmarks/distributional_summary.md`.

| dataset | model | fit_s | nll | crps | cov90 | width90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| synthetic_100k | chimera_gaussian | 0.972 | 1.04395 | 0.40490 | 0.918 | 2.487 |
| synthetic_100k | chimera_gaussian_es | 3.354 | 0.99145 | 0.39060 | 0.885 | 2.197 |
| synthetic_100k | chimera_gaussian_es_calibrated | 3.343 | 0.98975 | 0.39050 | 0.899 | 2.284 |
| synthetic_100k | chimera_rmse_const_sigma | 0.792 | 1.10794 | 0.40356 | 0.897 | 2.385 |
| synthetic_100k | chimera_quantile_pair | 2.028 | - | - | 0.906 | 2.530 |
| synthetic_100k | ngboost | 20.203 | 1.01390 | 0.39712 | 0.904 | 2.332 |
| synthetic_100k | catboost_uncertainty | 0.255 | 1.05816 | 0.41034 | 0.908 | 2.458 |
| synthetic_100k | lightgbm_twin | 1.841 | 1.64377 | 0.41959 | 0.618 | 1.211 |
| synthetic_500k | chimera_gaussian | 2.815 | 1.04370 | 0.40426 | 0.921 | 2.508 |
| synthetic_500k | chimera_gaussian_es | 10.952 | 0.98289 | 0.38881 | 0.894 | 2.237 |
| synthetic_500k | chimera_gaussian_es_calibrated | 10.469 | 0.98265 | 0.38879 | 0.899 | 2.270 |
| synthetic_500k | chimera_rmse_const_sigma | 1.966 | 1.10681 | 0.40316 | 0.899 | 2.403 |
| synthetic_500k | chimera_quantile_pair | 6.247 | - | - | 0.910 | 2.527 |
| synthetic_500k | ngboost | 125.871 | 1.00773 | 0.39523 | 0.905 | 2.329 |
| synthetic_500k | catboost_uncertainty | 0.848 | 1.05592 | 0.40944 | 0.909 | 2.457 |
| synthetic_500k | lightgbm_twin | 3.463 | 1.63048 | 0.41888 | 0.619 | 1.209 |

Promotion-gate read: fixed-round Chimera Gaussian is 1.43x the 500k RMSE
constant-sigma fit time, comfortably below the <=2.5x equal-round gate, and
44.7x faster than NGBoost at the same row count and round budget. The
early-stopped Gaussian lane uses a larger 400-round budget and therefore is not
the equal-round speed gate, but it materially improves quality: calibrated
early-stopped Chimera has the best NLL/CRPS on both synthetic sizes and is
12.0x faster than NGBoost at 500k rows. Scalar sigma calibration moves
early-stopped coverage from mild undercoverage (0.894-0.885) back to about
0.90, with sigma-bin coverage near flat in the generated per-seed tables.
CatBoost uncertainty remains the fastest external uncertainty lane but has
worse NLL/CRPS. The LightGBM twin model has strong point RMSE and sharply
under-covers, with only about 62% empirical coverage for nominal 90% intervals.
Treat the sigma-quality conclusion as synthetic-gate evidence. Before using
`sigma` downstream as observation noise, run the same lanes on real
heteroscedastic regression data and the intended domain data.

## WNBA Real-Data Distributional Validation

The first domain-data sigma check uses WNBA DARKO game-level metric observations:

```bash
PYTHONPATH=. /Users/kmedved/.venvs/darko311/bin/python \
  benchmarks/bench_wnba_realdata_distributional.py
```

The source is
`/Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/calculated_data/research/observation_covariance_measurement/game_metric_observations.parq`.
Rows use source-column `z_observed`, a transformed observation scale for six
game metrics (`fg_pct`, `fta_100`, `pace`, `pf_100`, `pts_100`, `tov_100`),
with `sample_weight` observation weights.  The split is time ordered: train on
2009-2021, early-stop/calibrate on 2022-2023, and test on 2024-2026.  Features
are date/context fields plus causal prior metric aggregates computed from
previous dates only.

Generated outputs:

- `benchmarks/wnba_realdata_distributional.csv`
- `benchmarks/wnba_realdata_distributional_summary.md`

| model | NLL | CRPS | RMSE mu | cov90 | std-resid RMS | mean sigma | affine b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unit_normal_observation_baseline | 51.648 | 6.068 | 10.073 | 0.304 | 10.073 | 1.000 | |
| chimera_rmse_const_sigma | 1.435 | 0.501 | 1.015 | 0.901 | 1.027 | 0.989 | |
| chimera_gaussian_raw | 0.430 | 0.394 | 1.013 | 0.873 | 1.086 | 0.588 | |
| chimera_gaussian_scalar_calibrated | 0.423 | 0.393 | 1.013 | 0.893 | 1.014 | 0.630 | |
| chimera_gaussian_affine_calibrated | **0.407** | **0.392** | 1.013 | 0.900 | 1.009 | 0.695 | 1.104 |

Interpretation: affine-calibrated Gaussian passes this real-data one-step
scale calibration check and improves the scalar lane on NLL, CRPS, overall
90% coverage, and standardized-residual RMS.  The fitted affine slope
(`b=1.104`) matches the expected mild sigma-range stretch: scalar calibration
left sigma-bin RMS at `0.824/0.856/0.990/1.111/1.127`, while affine moves it
to `0.967/0.934/1.058/1.084/0.978`; coverage by increasing predicted sigma is
`0.911/0.926/0.879/0.871/0.915`.  This is enough to retire the "synthetic
only" caveat for one-step observation scale, but not enough to declare
production Kalman readiness.  A downstream Kalman replay should still test
whether injecting `sigma^2` as observation variance improves filtering
outcomes versus the current covariance schedule.
