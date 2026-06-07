# Best-of-Both-Worlds Integration Plan

This branch uses upstream `bbstats/chimeraboost@ddaf272` as the trunk and ports
fork work forward behind benchmark-gated seams. The target shape is:

- `tree_mode="catboost"`: upstream/product path; oblivious trees, exact SHAP,
  linear leaves, hierarchical shrinkage, bagging, and current docs/API.
- `tree_mode="lightgbm"`: opt-in non-oblivious level-wise tree path; no exact
  SHAP, no linear leaves, and no hierarchical shrinkage in v1.
- `tree_mode="auto"`: future validation-selected mode, only after broader
  holdout evidence shows it beats the catboost default on the primary metric.

The upstream rewrite audit is tracked in
[`docs/bbstats-v2-patch-audit.md`](bbstats-v2-patch-audit.md). That file is the
patch-by-patch checklist for deciding whether each bbstats v2 change stays as
the catboost product path, yields to a behavior-equivalent darko optimization,
or needs a benchmark-gated toggle.

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
2. Added the `tree_mode` seam on the upstream trunk, preserving catboost
   behavior while routing LightGBM/levelwise aliases to the opt-in
   non-oblivious builder.
3. Fixed validation/sample-weight semantics before speed work.
4. Ported safe catboost-path performance work: timing diagnostics, compact
   binned dtypes, guarded constant-Hessian histograms, selected-feature and
   selected-row histogram fills, and class-major multiclass buffers.
5. Added opt-in `tree_mode="lightgbm"` with a level-wise tree representation
   for regression, binary classification, and multiclass classification.
6. Extended the revision harness with Quantile-loss datasets and
   pinball/coverage metrics so the benchmark matrix covers RMSE-style and
   quantile regression separately.
7. Added grouped split modes to the revision and levelwise-tuning harnesses so
   train/validation/test can hold out whole groups instead of random rows.
8. Added feasible memory reporting (`peak_rss_mb`) to newly generated raw
   benchmark rows using each benchmark worker process's peak resident set size.
9. Added explicit `ensemble_size` coverage to the revision and levelwise-tuning
   harnesses so single-model and bagged configurations are benchmarked as
   separate rows.
10. Added opt-in OpenML and Grinsztajn dataset registration to the revision and
    levelwise-tuning harnesses so real-tabular gates can use the same
    upstream/fork/candidate adapter path as the synthetic matrix.

## Current Upstream/Fork/Candidate Benchmark

Raw rows and summary rows are tracked in:

- `benchmarks/tri_compare_medium_20260606.csv`
- `benchmarks/tri_compare_medium_summary_20260606.csv`

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
  --csv benchmarks/tri_compare_medium_20260606.csv
```

All 168 rows completed successfully. Ratios below are against
`upstream_matched`; lower is better for both primary metric and fit time.

| Variant | Primary-metric wins/ties | Mean metric ratio | Mean fit ratio | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `candidate_catboost` | 13 / 14 | 1.000 | 1.020 | Preserves upstream quality up to weighted-metric noise; near speed parity, with row-specific wins. |
| `fork_matched` | 2 / 14 | 1.093 | 1.191 | Legacy fork can be faster, but usually gives up holdout quality and often runs to the iteration cap. |
| `candidate_lightgbm` | 0 / 14 | 1.222 | 0.692 | Much faster on most rows, but still fails the primary-metric gate. |

Decision: keep upstream-shaped `tree_mode="catboost"` as the product/default
path. The legacy fork is useful as a source of implementation ideas, not as the
integration base. `tree_mode="lightgbm"` remains opt-in until it wins weighted
or ordinary holdout loss, not just speed.

## Catboost-Path Speed Recovery Audit

The first medium tri-compare above enabled `verbose_timing` whenever a revision
accepted that constructor argument. That is useful for candidate phase
diagnostics, but unfair for cross-revision fit timing because `upstream_matched`
does not accept the option and therefore did not pay the same instrumentation
overhead. The revision harness now keeps `verbose_timing=False` by default and
requires `--verbose-timing` for profiling runs.

Corrected catboost-only raw rows are tracked in:

- `benchmarks/catboost_speed_recovery_medium_20260606.csv`
- `benchmarks/catboost_speed_recovery_focus_20260606.csv`

Full medium command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --upstream /private/tmp/chimeraboost-upstream-ddaf272-bobw \
  --candidate . \
  --models upstream_matched candidate_catboost \
  --datasets friedman_numeric wide_numeric_reg categorical_reg numeric_binary \
    numeric_multiclass categorical_binary categorical_multiclass \
  --sizes medium \
  --seeds 3 \
  --repeat 2 \
  --iterations 300 \
  --patience 25 \
  --threads 4 \
  --weight-modes none stress \
  --csv benchmarks/catboost_speed_recovery_medium_20260606.csv
```

Result: `candidate_catboost` still preserves upstream quality exactly for the
ordinary rows and within metric noise for weighted rows. The apparent speed gap
shrinks but does not become a broad win. The initial catboost-only audit had
mean fit ratio `1.108`, median `1.053`, and speed wins on 4/14 summary rows; a
focused repeat-3 rerun of suspect rows had mean fit ratio `1.064`, median
`1.036`, and speed wins on 3/8 summary rows. The fully refreshed tri-compare
above is the headline number: mean fit ratio `1.020`.

Decision: no catboost-mode model-code optimization cleared the bar in this
audit. The safe conclusion is that catboost mode is quality-preserving and near
parity with bbstats v2, with a few row-specific speed wins, not that it has a
general speed advantage. Use `--verbose-timing` only for phase diagnostics, not
for headline cross-revision speed ratios.

### Catboost Cleanup Checkpoint

After refreshing `upstream/main` on 2026-06-06, the current bbstats v2 baseline
was still `ddaf272`. A focused repeat-3, two-seed catboost retest is tracked in:

- `benchmarks/catboost_cleanup_baseline_20260606.csv`
- `benchmarks/catboost_cleanup_suspects_20260606.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --upstream /private/tmp/chimeraboost-upstream-ddaf272-bobw \
  --candidate . \
  --models upstream_matched candidate_catboost \
  --datasets friedman_numeric wide_numeric_reg categorical_reg numeric_binary \
    numeric_multiclass categorical_binary categorical_multiclass \
  --sizes medium \
  --seeds 2 \
  --repeat 3 \
  --iterations 300 \
  --patience 25 \
  --threads 4 \
  --weight-modes none stress \
  --csv benchmarks/catboost_cleanup_baseline_20260606.csv
```

Result: `candidate_catboost` preserved upstream primary metrics on ordinary
rows and matched weighted rows up to the known validation-weight semantic
difference. Mean fit ratio was `0.979` against `upstream_matched`; candidate
won 8 rows, tied 1 row, and lost 5 rows by a 1% threshold. A repeat-5 rerun of
the apparent suspect rows showed the losses were not broad: categorical
multiclass moved to parity, numeric binary was mixed by weight mode, and the
Friedman rows were timing-sensitive.

Rejected cleanup: an upstream-shaped plain-tree helper was tested to route
non-specialized catboost fits through bbstats-v2-style code. It did not clear
the one-change bar: it helped Friedman in one sample but slowed numeric binary
and categorical multiclass. The patch was reverted and is not part of catboost
mode.

Kept cleanup decisions:

- Keep constant-Hessian RMSE histograms; candidate-internal ablations showed
  identical predictions with a faster tree-build path on the Friedman seed.
- Keep compact binned dtypes; forcing upstream-style `uint16` bins produced
  identical predictions but mixed timings, with clear losses on numeric binary,
  categorical multiclass, and wide regression.
- Keep weighted validation support when a user supplies
  `eval_set=(X_val, y_val, sample_weight_val)`. This is a product semantic
  improvement over bbstats v2, even though it means weighted benchmark rows are
  not a literal upstream-equivalence test.

Decision: current `tree_mode="catboost"` is the best available product path:
upstream v2 behavior remains the trunk, and fork optimizations stay only where
they have behavior proofs or measured wins. It does **not** yet strictly
dominate bbstats v2 on every timing row, so no further default-changing
catboost patch should land without a targeted benchmark proving row-level wins.

### Strict-Domination Gate

The revision harness now separates the two weighted-validation lanes:

- `--validation-weight-policy upstream-compatible` forces candidate rows to use
  the same two-tuple validation eval set as bbstats v2. Use this lane to claim
  literal catboost-path domination.
- `--validation-weight-policy product` preserves the candidate's product
  improvement of using validation weights when the benchmark supplies them.
  Use this lane to evaluate the enhanced semantics, not literal equivalence.

The checker is:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/check_strict_domination.py \
  benchmarks/catboost_strict_medium_20260606.csv \
  --mode upstream-compatible \
  --out benchmarks/catboost_strict_medium_report_20260606.json
```

It compares `candidate_catboost` against `upstream_matched` by
dataset/size/split/weight/ensemble/seed and exits nonzero on named blocking
failures: `quality_regression`, `timing_regression`,
`semantic_non_equivalence`, `missing_row`, `error_row`, or aggregate fit-time
regression. This gate should be added before any further catboost model-code
change. A strict candidate run should use `--repeat 7`, at least five seeds for
the core/quantile rows, and the upstream-compatible validation policy.

### Strict Medium Gate Result

The first full strict medium run is tracked in:

- `benchmarks/catboost_strict_medium_20260606.csv`
- `benchmarks/catboost_strict_medium_summary_20260606.csv`
- `benchmarks/catboost_strict_medium_report_20260606.json`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --upstream /private/tmp/chimeraboost-upstream-ddaf272-bobw \
  --candidate . \
  --models upstream_matched candidate_catboost \
  --datasets friedman_numeric wide_numeric_reg categorical_reg numeric_binary \
    numeric_multiclass categorical_binary categorical_multiclass \
    quantile_reg_10 quantile_reg_50 quantile_reg_90 \
  --sizes medium \
  --seeds 5 \
  --repeat 7 \
  --iterations 300 \
  --patience 25 \
  --threads 4 \
  --weight-modes none stress \
  --validation-weight-policy upstream-compatible \
  --csv benchmarks/catboost_strict_medium_20260606.csv
```

Result: the checker failed, but only on row-level timing. There were 100 paired
comparisons, no row errors, no semantic-policy failures, and no quality
regressions. Primary metrics and best iterations were identical in every
dataset/weight aggregate. The candidate aggregate fit ratio was faster than
upstream (`geomean_fit_ratio=0.9866`), but 32 individual seed rows exceeded the
strict per-row timing threshold.

Stable-looking timing blockers by dataset/weight geomean:

| Dataset / weight | Candidate fit ratio | Interpretation |
| --- | ---: | --- |
| `quantile_reg_50` / stress | 1.093 | Strongest stable blocker. |
| `numeric_binary` / stress | 1.070 | Stable blocker. |
| `categorical_binary` / stress | 1.053 | Stable but small. |
| `friedman_numeric` / stress | 1.049 | Near-threshold blocker. |
| `quantile_reg_90` / none | 1.034 | Near-threshold blocker. |

Rows such as categorical multiclass and wide numeric regression had individual
seed failures but better aggregate behavior, so treat those as timing-noise
suspects until a focused rerun says otherwise.

### Current Calibrated Strict Gate

After the scalar-loop cleanup and rejected catboost-path probes, a fresh full
strict medium run was made against the same upstream `ddaf272` baseline:

- `benchmarks/catboost_strict_medium_current_20260606.csv`
- `benchmarks/catboost_strict_medium_current_summary_20260606.csv`
- `benchmarks/catboost_strict_medium_current_report_20260606.json`

The raw gate still fails only on timing: 100 paired comparisons, no row errors,
no semantic failures, no quality regressions, identical aggregate metrics and
iterations, and aggregate candidate speed faster than upstream
(`geomean_fit_ratio=0.9779`), but 36 row-level timing failures.

Because the raw row-min timing gate can fail when comparing bbstats v2 against
itself, the current run was calibrated with two same-revision controls:

- `benchmarks/catboost_same_revision_medium_current_20260606.csv`
- `benchmarks/catboost_same_revision_medium_current_reversed_20260606.csv`
- `benchmarks/catboost_strict_medium_current_calibrated_both_report_20260606.json`

The default-order same-revision control failed with 34 timing failures and
`geomean_fit_ratio=0.9966`; the reversed-order same-revision control failed
with 43 timing failures and `geomean_fit_ratio=1.0288`. With both controls as
the timing-noise envelope, the current candidate still fails the calibrated
gate with 28 row-level timing regressions, but no quality or semantic
regressions and the same aggregate faster fit ratio (`0.9779`).

Aggregate summary rows are now the more useful readout than raw per-seed
failures. Current catboost mode is faster on categorical classification,
Friedman stress, quantile stress rows, and q90; slower on categorical
regression, numeric classification, q50 unweighted, and wide regression. The
remaining work is therefore not to restore a known darko v1 win wholesale. It
is to explain the remaining row-level timing failures and either reduce them
with one-change, behavior-preserving patches or document them as below the
accepted timing floor.

### Compact-Bin Ablation

The first one-change ablation forced the candidate to use upstream-style
`uint16` binned matrices instead of compact `_bin_dtype_for_n_bins` output. The
temporary worktree changed only `chimeraboost/binning.py`; results are tracked
in:

- `benchmarks/catboost_ablate_uint16_focus_20260606.csv`
- `benchmarks/catboost_ablate_uint16_focus_summary_20260606.csv`
- `benchmarks/catboost_ablate_uint16_focus_report_20260606.json`

Result: forcing `uint16` did not clear the strict gate
(`geomean_fit_ratio=0.9905`, 22 timing failures). It fixed two important
blockers but created or worsened others:

| Dataset / weight | Compact ratio | Forced `uint16` ratio | Decision |
| --- | ---: | ---: | --- |
| `numeric_binary` / stress | 1.070 | 0.940 | `uint16` helps. |
| `quantile_reg_50` / stress | 1.093 | 0.960 | `uint16` helps. |
| `categorical_binary` / none | 0.890 | 1.060 | compact helps. |
| `friedman_numeric` / stress | 1.049 | 1.079 | compact helps. |
| `quantile_reg_90` / stress | 0.992 | 1.065 | compact helps. |
| `wide_numeric_reg` / stress | 1.008 | 1.062 | compact helps. |

Decision: keep compact bins as the catboost default. Upstream-style `uint16`
is a possible benchmark-gated adaptive toggle for numeric binary and median
quantile stress rows, not a broad revert.

### Constant-Hessian Ablation

The second one-change ablation disabled the candidate's constant-Hessian
histogram shortcut for unweighted RMSE/Quantile-style fits, forcing the
upstream-style general histogram path. It can only affect unweighted
constant-Hessian losses; classification and weighted rows are out of scope.
Results are tracked in:

- `benchmarks/catboost_ablate_no_constant_hessian_focus_20260606.csv`
- `benchmarks/catboost_ablate_no_constant_hessian_focus_summary_20260606.csv`
- `benchmarks/catboost_ablate_no_constant_hessian_focus_report_20260606.json`

Result: disabling the shortcut did not clear the strict gate
(`geomean_fit_ratio=1.0039`, 14 row-level timing failures plus aggregate timing
failure). Metrics and iterations stayed identical, but aggregate speed worsened.

| Dataset / weight | Constant-Hessian ratio | General-Hessian ratio | Decision |
| --- | ---: | ---: | --- |
| `categorical_reg` / none | 1.012 | 0.964 | general helps here. |
| `quantile_reg_50` / none | 1.028 | 0.995 | general helps here. |
| `friedman_numeric` / none | 0.938 | 1.018 | constant helps. |
| `quantile_reg_10` / none | 0.974 | 1.065 | constant helps. |
| `quantile_reg_90` / none | 1.034 | 1.039 | no useful improvement. |
| `wide_numeric_reg` / none | 0.900 | 0.947 | constant helps. |

Decision: keep the constant-Hessian shortcut as the catboost default. The
general-Hessian path is not a broad revert candidate and does not touch the
largest remaining blockers (`numeric_binary` stress and `quantile_reg_50`
stress). At most it is a future narrow adaptive toggle for categorical
regression or median quantile unweighted rows.

### Categorical-Encoding Ablation

Upstream commit `20ad819` replaced per-element Python categorical mapping with
the pandas-vectorized `factorize` / `Series.map` path. The current branch keeps
that upstream path, plus the branch-only weighted target-stat hook. A focused
one-change ablation restored the older manual/lazy mapping mechanics in a
temporary worktree for categorical datasets only. Results are tracked in:

- `benchmarks/catboost_ablate_manual_cats_focus_20260606.csv`
- `benchmarks/catboost_ablate_manual_cats_focus_summary_20260606.csv`
- `benchmarks/catboost_ablate_manual_cats_focus_report_20260606.json`

Result: manual categorical mapping did not clear the strict gate
(`geomean_fit_ratio=0.9638`, 6 timing failures). Metrics and iterations stayed
identical. The path is faster in aggregate on the categorical focus set, but
still has stable categorical-binary timing failures and mixed unweighted
categorical-multiclass behavior.

| Dataset / weight | Pandas-vectorized ratio | Manual/lazy ratio | Decision |
| --- | ---: | ---: | --- |
| `categorical_reg` / none | 1.012 | 0.952 | manual helps. |
| `categorical_multiclass` / stress | 0.896 | 0.840 | manual helps. |
| `categorical_binary` / stress | 1.053 | 1.047 | manual slightly helps but still fails rows. |
| `categorical_binary` / none | 0.890 | 1.021 | pandas helps. |
| `categorical_multiclass` / none | 0.886 | 0.923 | pandas helps. |
| `categorical_reg` / stress | 0.982 | 1.016 | pandas helps. |

Decision: keep upstream's pandas-vectorized categorical encoding as the
catboost default. The older manual/lazy mapping is not a broad revert, but it
is a possible benchmark-gated adaptive path for categorical regression or
weighted categorical multiclass after the larger blockers are resolved.

### High-Repeat Blocker Rerun

The repeat-7 strict gate mixed stable timing failures with noisy per-seed
failures. A focused repeat-15 rerun isolates the remaining blockers:

- `benchmarks/catboost_strict_blockers_stress_r15_20260606.csv`
- `benchmarks/catboost_strict_blockers_stress_r15_summary_20260606.csv`
- `benchmarks/catboost_strict_blockers_stress_r15_report_20260606.json`
- `benchmarks/catboost_strict_blockers_quantile90_r15_20260606.csv`
- `benchmarks/catboost_strict_blockers_quantile90_r15_summary_20260606.csv`
- `benchmarks/catboost_strict_blockers_quantile90_r15_report_20260606.json`

Result: quality and iterations were still identical. The stable aggregate
timing blockers are now narrower:

| Dataset / weight | Repeat-7 ratio | Repeat-15 ratio | Status |
| --- | ---: | ---: | --- |
| `numeric_binary` / stress | 1.070 | 1.021 | Still aggregate-slower and has row failures. |
| `quantile_reg_50` / stress | 1.093 | 1.040 | Still aggregate-slower and has row failures. |
| `quantile_reg_90` / none | 1.034 | 1.030 | Still aggregate-slower and has row failures. |
| `categorical_binary` / stress | 1.053 | 0.996 | No longer an aggregate blocker. |
| `friedman_numeric` / stress | 1.049 | 0.994 | No longer an aggregate blocker. |

Decision: treat categorical-binary stress and Friedman stress as timing-noise
suspects, not immediate model-code targets. The next catboost cleanup work
should focus on numeric-binary stress, median-quantile stress, and quantile-90
unweighted. Of the ablations so far, forced `uint16` is the only one that fixed
the first two aggregate blockers, but it hurt other rows, so any use of it must
be adaptive and gate-proven.

### Weighted Leaf-Correction Port

The median-quantile stress blocker maps to the MAE/Quantile leaf-correction
path. A one-change port restored darko v1's grouped correction only when
`sample_weight` is present, while preserving upstream's per-leaf mask loop for
unweighted corrections. Results are tracked in:

- `benchmarks/catboost_grouped_leaf_q50_stress_r15_20260606.csv`
- `benchmarks/catboost_grouped_leaf_q50_stress_r15_summary_20260606.csv`
- `benchmarks/catboost_grouped_leaf_q50_stress_r15_report_20260606.json`
- `benchmarks/catboost_grouped_leaf_q90_none_r15_20260606.csv`
- `benchmarks/catboost_grouped_leaf_q90_none_r15_summary_20260606.csv`
- `benchmarks/catboost_grouped_leaf_q90_none_r15_report_20260606.json`
- `benchmarks/catboost_weighted_leaf_q90_none_r15_20260606.csv`
- `benchmarks/catboost_weighted_leaf_q90_none_r15_summary_20260606.csv`
- `benchmarks/catboost_weighted_leaf_q90_none_r15_report_20260606.json`

Result: weighted grouped correction clears the median-quantile stress blocker
(`geomean_fit_ratio=0.8896`) with identical metrics and iterations. A broader
grouped correction was rejected for unweighted q90
(`geomean_fit_ratio=1.0491`), and the final weighted-only code still leaves the
q90 unweighted blocker in place (`geomean_fit_ratio=1.0406`).

| Dataset / weight | Candidate path | Result | Decision |
| --- | --- | ---: | --- |
| `quantile_reg_50` / stress | grouped weighted leaves | 0.890 | Promote. |
| `quantile_reg_90` / none | grouped all leaves | 1.049 | Reject broad grouping. |
| `quantile_reg_90` / none | grouped weighted only | 1.041 | Still blocked. |

Decision: use darko v1's grouped leaf correction only for weighted
MAE/Quantile fits. Keep upstream's unweighted mask loop because it is better on
q90 under the current strict gate.

### Median-Quantile Unweighted Leaf Correction

The refreshed aggregate-slower focus gate showed that `quantile_reg_50` /
unweighted had become one of the remaining calibrated blockers:

- `benchmarks/catboost_current_aggregate_slow_focus_20260606.csv`
- `benchmarks/catboost_current_aggregate_slow_focus_summary_20260606.csv`
- `benchmarks/catboost_current_aggregate_slow_focus_calibrated_report_20260606.json`

In that focus run, q50 unweighted had identical metrics and iterations but a
candidate fit ratio of `1.167`, with all three q50 seed rows failing the
calibrated timing gate. A candidate-only phase run showed q50 time split
roughly between tree build and the unweighted leaf-correction update:

- `benchmarks/catboost_current_phase_focus_20260606.csv`
- `benchmarks/catboost_current_phase_focus_summary_20260606.csv`

A narrow one-change probe now uses grouped unweighted leaf correction only for
`loss="Quantile", alpha=0.5`. It does not change q10, q90, MAE, or weighted
Quantile paths. Results are tracked in:

- `benchmarks/catboost_q50_unweighted_grouped_probe_r7_20260606.csv`
- `benchmarks/catboost_q50_unweighted_grouped_probe_r7_summary_20260606.csv`
- `benchmarks/catboost_q50_unweighted_grouped_probe_r7_calibrated_report_20260606.json`

Result: the q50 unweighted calibrated gate now passes (`passed=true`, zero
failures), with identical metrics and iterations and aggregate fit near parity
(`geomean_fit_ratio=1.0007`). This is enough to promote the q50-only grouped
path, but not enough to revisit the earlier rejection of broad unweighted
grouping for q90.

Decision: promote grouped unweighted correction only for median Quantile. Keep
the upstream mask loop for q10/q90 and other unweighted leaf-correction losses
until a separate gate proves those rows.

A post-promotion focus rerun is tracked in:

- `benchmarks/catboost_post_q50_aggregate_focus_20260606.csv`
- `benchmarks/catboost_post_q50_aggregate_focus_summary_20260606.csv`
- `benchmarks/catboost_post_q50_aggregate_focus_calibrated_report_20260606.json`

Result: q50 is no longer an aggregate blocker (`0.993` unweighted, `0.972`
stress). Numeric multiclass also moved to parity (`0.987` unweighted, `1.001`
stress). The remaining aggregate blockers in that focus set are numeric binary
(`1.252` unweighted, `1.145` stress) and wide numeric regression stress
(`1.158`; wide unweighted is a smaller `1.074`).

### Adaptive `uint16` Probe

The forced-`uint16` ablation was the only earlier probe that helped
numeric-binary stress in aggregate, but it hurt other rows. A narrower adaptive
probe used upstream-style `uint16` bins only for numeric-only binary catboost
fits. Results are tracked in:

- `benchmarks/catboost_adaptive_uint16_numeric_binary_stress_r15_20260606.csv`
- `benchmarks/catboost_adaptive_uint16_numeric_binary_stress_r15_summary_20260606.csv`
- `benchmarks/catboost_adaptive_uint16_numeric_binary_stress_r15_report_20260606.json`

Result: the adaptive policy failed the repeat-15 numeric-binary stress gate
(`geomean_fit_ratio=1.1014`) with identical metrics and iterations. The product
code was reverted.

Decision: do not promote adaptive upstream-style `uint16` bins. Keep compact
bins as the catboost default until a cleaner one-change gate proves otherwise.

### Higher-Repeat Residual Blockers

After the weighted leaf-correction port, the remaining timing-only blockers
were rerun with higher repeats to separate stable overhead from min-of-repeat
noise. Results are tracked in:

- `benchmarks/catboost_q90_none_r30_20260606.csv`
- `benchmarks/catboost_q90_none_r30_summary_20260606.csv`
- `benchmarks/catboost_q90_none_r30_report_20260606.json`
- `benchmarks/catboost_q90_none_seed0_r80_20260606.csv`
- `benchmarks/catboost_q90_none_seed0_r80_summary_20260606.csv`
- `benchmarks/catboost_q90_none_seed0_r80_report_20260606.json`
- `benchmarks/catboost_numeric_binary_stress_r30_20260606.csv`
- `benchmarks/catboost_numeric_binary_stress_r30_summary_20260606.csv`
- `benchmarks/catboost_numeric_binary_stress_r30_report_20260606.json`

Result: q90 unweighted is no longer an aggregate blocker at repeat 30
(`geomean_fit_ratio=0.9961`), but seed 0 remains a stable row-level timing
failure even at repeat 80 (`fit_ratio=1.0576`). Numeric-binary stress remains
an aggregate blocker at repeat 30 (`geomean_fit_ratio=1.0271`) with identical
metrics and iterations.

| Dataset / weight | Repeat | Geomean | Strict result | Interpretation |
| --- | ---: | ---: | --- | --- |
| `quantile_reg_90` / none | 30 | 0.996 | Fail | Aggregate passes; seed 0 row-level fail remains. |
| `quantile_reg_90` / none seed 0 | 80 | 1.058 | Fail | Single-row timing failure is stable. |
| `numeric_binary` / stress | 30 | 1.027 | Fail | Aggregate timing blocker remains. |

Decision: q90 should be treated as a narrow row-level timing blocker, not a
broad aggregate regression. Numeric-binary stress is still the primary
aggregate blocker.

### Fast Full-Hist Branch Probe

A small one-change probe split the default full-row/full-feature/general-Hessian
tree path out ahead of the selected-row/selected-feature/constant-Hessian
cascade. That path matches numeric-binary stress and calls the same histogram
and split kernels as before. Results are tracked in:

- `benchmarks/catboost_fast_full_hist_numeric_binary_stress_r15_20260606.csv`
- `benchmarks/catboost_fast_full_hist_numeric_binary_stress_r15_summary_20260606.csv`
- `benchmarks/catboost_fast_full_hist_numeric_binary_stress_r15_report_20260606.json`
- `benchmarks/catboost_fast_full_hist_numeric_binary_stress_r30_20260606.csv`
- `benchmarks/catboost_fast_full_hist_numeric_binary_stress_r30_summary_20260606.csv`
- `benchmarks/catboost_fast_full_hist_numeric_binary_stress_r30_report_20260606.json`

Result: the probe did not clear the gate. Repeat 15 improved the aggregate
relative to the blocker baseline but still failed (`geomean_fit_ratio=1.0074`);
repeat 30 regressed (`geomean_fit_ratio=1.0874`). The product code was
reverted.

Decision: do not promote the branch-only full-hist fast path. The
numeric-binary blocker likely needs either a true upstream-default tree-builder
lane or a deeper phase-level explanation, not another branch shuffle around the
same kernels.

### Upstream-Default Tree Lane Probe

A broader probe copied the bbstats v2 full-row/full-feature oblivious tree
builder into a separate candidate lane, then routed only the suspect default
catboost rows through it. Results are tracked in:

- `benchmarks/catboost_upstream_tree_q90_none_r15_20260606.csv`
- `benchmarks/catboost_upstream_tree_q90_none_r15_summary_20260606.csv`
- `benchmarks/catboost_upstream_tree_q90_none_r15_report_20260606.json`
- `benchmarks/catboost_upstream_tree_numeric_binary_stress_r15_20260606.csv`
- `benchmarks/catboost_upstream_tree_numeric_binary_stress_r15_summary_20260606.csv`
- `benchmarks/catboost_upstream_tree_numeric_binary_stress_r15_report_20260606.json`
- `benchmarks/catboost_upstream_tree_numeric_binary_stress_seed0_r80_20260606.csv`
- `benchmarks/catboost_upstream_tree_numeric_binary_stress_seed0_r80_summary_20260606.csv`
- `benchmarks/catboost_upstream_tree_numeric_binary_stress_seed0_r80_report_20260606.json`

Result: the upstream tree lane was rejected. It regressed q90 unweighted
strongly (`geomean_fit_ratio=1.1335`). Narrowing it to weighted
non-leaf-adjusted fits improved numeric-binary aggregate at repeat 15
(`geomean_fit_ratio=0.9952`), but seed 0 remained a stable row-level failure at
repeat 80 (`fit_ratio=1.0693`). The product code was reverted.

Decision: do not promote a copied upstream-default tree lane. It does not fix
the strict row-level timing failures and hurts q90.

### Scalar-Loop Timing Cleanup

The scalar and multiclass fit loops were paying candidate-only timing overhead
even with `verbose_timing=False`: each phase assigned `time.perf_counter()` and
the default strict path still called selected-feature/subsample helpers even
when `colsample=1.0` and `subsample=1.0`. The promoted cleanup makes those
branches lazy:

- only call `time.perf_counter()` when `verbose_timing=True`;
- skip `_feature_indices(...)` when there is no feature mask;
- skip `_maybe_subsample(...)` and `_feature_mask(...)` on default full-row /
  full-feature fits.

Results are tracked in:

- `benchmarks/catboost_timing_guard_q90_none_r15_20260606.csv`
- `benchmarks/catboost_timing_guard_q90_none_r15_summary_20260606.csv`
- `benchmarks/catboost_timing_guard_q90_none_r15_report_20260606.json`
- `benchmarks/catboost_timing_guard_numeric_binary_stress_r15_20260606.csv`
- `benchmarks/catboost_timing_guard_numeric_binary_stress_r15_summary_20260606.csv`
- `benchmarks/catboost_timing_guard_numeric_binary_stress_r15_report_20260606.json`
- `benchmarks/catboost_timing_guard_numeric_binary_stress_r30_20260606.csv`
- `benchmarks/catboost_timing_guard_numeric_binary_stress_r30_summary_20260606.csv`
- `benchmarks/catboost_timing_guard_numeric_binary_stress_r30_report_20260606.json`
- `benchmarks/catboost_timing_guard_feature_skip_numeric_binary_stress_r15_20260606.csv`
- `benchmarks/catboost_timing_guard_feature_skip_numeric_binary_stress_r15_summary_20260606.csv`
- `benchmarks/catboost_timing_guard_feature_skip_numeric_binary_stress_r15_report_20260606.json`
- `benchmarks/catboost_timing_guard_feature_skip_numeric_binary_stress_r30_20260606.csv`
- `benchmarks/catboost_timing_guard_feature_skip_numeric_binary_stress_r30_summary_20260606.csv`
- `benchmarks/catboost_timing_guard_feature_skip_numeric_binary_stress_r30_report_20260606.json`
- `benchmarks/catboost_scalar_loop_cleanup_numeric_binary_stress_r30_20260606.csv`
- `benchmarks/catboost_scalar_loop_cleanup_numeric_binary_stress_r30_summary_20260606.csv`
- `benchmarks/catboost_scalar_loop_cleanup_numeric_binary_stress_r30_report_20260606.json`
- `benchmarks/catboost_scalar_loop_cleanup_numeric_binary_stress_r50_20260606.csv`
- `benchmarks/catboost_scalar_loop_cleanup_numeric_binary_stress_r50_summary_20260606.csv`
- `benchmarks/catboost_scalar_loop_cleanup_numeric_binary_stress_r50_report_20260606.json`

Result: q90 unweighted now passes the strict gate at repeat 15
(`geomean_fit_ratio=0.9132`, no failures). Numeric-binary stress improved
substantially and passed at repeat 15 after the feature-index skip
(`geomean_fit_ratio=0.9237`, no failures), but remains unstable at higher
repeats: repeat 30 passes aggregate but has one row-level failure
(`geomean_fit_ratio=0.9402`), and repeat 50 failed after upstream found much
lower timing minima (`geomean_fit_ratio=1.1260`).

Decision: keep the scalar-loop cleanup because it is behavior-preserving, test
covered, and clears q90. Treat numeric-binary stress as the last unresolved
strict timing case; it now looks dominated by row-level timing stability rather
than metrics, iterations, or a known semantic difference.

### Numeric-Binary Repeat Trace

To separate real overhead from min-of-repeat timing luck, the revision harness
now records semicolon-delimited repeat traces in:

- `fit_repeat_seconds`
- `predict_repeat_seconds`

The focused numeric-binary stress trace is tracked in:

- `benchmarks/catboost_numeric_binary_stress_trace_r20_20260606.csv`
- `benchmarks/catboost_numeric_binary_stress_trace_r20_summary_20260606.csv`
- `benchmarks/catboost_numeric_binary_stress_trace_r20_report_20260606.json`
- `benchmarks/catboost_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`

Result: the repeat-20 run passes the aggregate min-of-repeat gate
(`geomean_fit_ratio=0.9736`) but still fails two row-level timing checks: seed 1
at `1.0580` and seed 4 at `1.0381`. The repeat distribution shows this is not
only an extreme-min artifact: the geomean median-repeat ratio is `1.1354`,
favoring upstream, while metrics and iterations remain identical.

A phase-timing diagnostic is tracked in:

- `benchmarks/catboost_numeric_binary_stress_phase_r10_20260606.csv`
- `benchmarks/catboost_numeric_binary_stress_phase_r10_summary_20260606.csv`
- `benchmarks/catboost_numeric_binary_stress_phase_r10_report_20260606.json`

Important caveat: upstream v2 does not expose `verbose_timing`, so phase columns
are candidate-only. Use that file to locate candidate work, not to compare phase
totals directly across revisions.

Decision: keep the trace columns in the harness. Numeric-binary stress remains
the final catboost strict-domination blocker, and it now looks like a true
default scalar/tree-builder overhead issue rather than a quality, iteration, or
validation-semantics issue.

### Plain-Builder Fast-Lane Probe

A narrow product-code probe tried to route full-row/full-feature,
non-constant-Hessian catboost fits through an upstream-shaped direct
`build_oblivious_tree(...)` call, avoiding the generic tree-builder variable and
extra selected-row/feature keyword plumbing. The probe is tracked in:

- `benchmarks/catboost_plain_builder_numeric_binary_stress_trace_r20_20260606.csv`
- `benchmarks/catboost_plain_builder_numeric_binary_stress_trace_r20_summary_20260606.csv`
- `benchmarks/catboost_plain_builder_numeric_binary_stress_trace_r20_report_20260606.json`
- `benchmarks/catboost_plain_builder_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`

Result: reject. The numeric-binary stress repeat-20 gate failed
(`geomean_fit_ratio=1.0214`), with row failures on seeds 0 and 4. The
median-repeat distribution still favored upstream (`geomean_median_ratio=1.1525`).

Decision: product code was reverted. Do not promote a call-shape-only plain
builder lane; the remaining overhead is deeper than the fit-loop keyword
dispatch.

### Benchmark-Order Probe

The strict harness previously filtered `--models` through the default variant
order, so `candidate_catboost` could not be run before `upstream_matched`.
The harness now preserves the CLI order and has a regression test for that
behavior.

The reversed-order numeric-binary stress run is tracked in:

- `benchmarks/catboost_numeric_binary_stress_reversed_order_r20_20260606.csv`
- `benchmarks/catboost_numeric_binary_stress_reversed_order_r20_summary_20260606.csv`
- `benchmarks/catboost_numeric_binary_stress_reversed_order_r20_report_20260606.json`
- `benchmarks/catboost_numeric_binary_stress_reversed_order_r20_repeat_summary_20260606.csv`

Result: reject benchmark-order bias as the explanation. Running
`candidate_catboost` first improved the aggregate min-of-repeat ratio
(`geomean_fit_ratio=0.9585`) but still left seed 1 as a row-level failure
(`fit_ratio=1.0564`). More importantly, every seed's median-repeat and
mean-repeat ratios still favored upstream, so the remaining numeric-binary
stress issue is distribution-level overhead, not just an unlucky second-run
position.

Decision: keep the harness order fix because it makes future probes explicit.
Do not spend more catboost cleanup time on row-order measurement artifacts;
the next useful probe needs to reduce default scalar/tree-builder overhead.

### Bin-Index Cast Probe

A cProfile diagnostic on the failing numeric-binary stress seed is tracked in:

- `benchmarks/catboost_numeric_binary_stress_profile_upstream_seed1_20260606.txt`
- `benchmarks/catboost_numeric_binary_stress_profile_candidate_seed1_20260606.txt`

The profile is diagnostic only because cProfile and Numba dispatch/cache work
distort absolute times, but it localized the candidate difference to tree
kernel calls: `_build_histograms_into` and `_best_split`, not preprocessing,
fit-loop keyword plumbing, prediction, or validation scoring.

The first kernel-level probe kept compact binned storage but cast each default
histogram bin id to native `int` before using it as a histogram index. The run
is tracked in:

- `benchmarks/catboost_hist_bin_int_numeric_binary_stress_trace_r20_20260606.csv`
- `benchmarks/catboost_hist_bin_int_numeric_binary_stress_trace_r20_summary_20260606.csv`
- `benchmarks/catboost_hist_bin_int_numeric_binary_stress_trace_r20_report_20260606.json`
- `benchmarks/catboost_hist_bin_int_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`

Result: reject. The aggregate min-of-repeat ratio stayed below 1.0
(`geomean_fit_ratio=0.9759`), but the row-level gate still failed and failures
worsened on seeds 1 and 2 (`fit_ratio=1.2292`, `1.1161`). The repeat medians
still favored upstream on four of five seeds.

Decision: product code was reverted. The remaining overhead is still in the
default tree-kernel path, but native-int bin indexing is not the fix.

### Tree-Kernel Dtype Microprobe

The direct dtype microprobe is tracked in:

- `benchmarks/catboost_tree_kernel_dtype_microprobe_20260606.csv`

It uses the failing numeric-binary stress seed, current candidate kernels, and
the same gradients/Hessians to compare compact `uint8` bins against
upstream-style `uint16` bins without changing the whole product path.

Result: dtype alone is not the explanation. Full `build_oblivious_tree` timing
was effectively tied (`uint8` mean `0.000910s`, `uint16` mean `0.000941s`), and
direct histogram/split timings were also tied. This explains why the earlier
forced-`uint16` result did not survive as a strict adaptive policy: whatever it
was improving was not a stable raw Numba dtype advantage.

Decision: keep compact bins by default. Do not spend another pass on dtype
specialization unless a broader, row-level product ablation justifies it.

### Linear-Leaf Precompute Probe

Binary catboost defaults use linear leaves, and each tree currently remaps the
selected split-feature bin ids to standardized bin-center values inside
`_linear_leaf_fit`. A product-code probe precomputed all standardized binned
features once per fit and passed that matrix into the linear-leaf fitter.

The run is tracked in:

- `benchmarks/catboost_linear_xstd_numeric_binary_stress_trace_r20_20260606.csv`
- `benchmarks/catboost_linear_xstd_numeric_binary_stress_trace_r20_summary_20260606.csv`
- `benchmarks/catboost_linear_xstd_numeric_binary_stress_trace_r20_report_20260606.json`
- `benchmarks/catboost_linear_xstd_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`

Result: reject. The aggregate min-of-repeat ratio was below 1.0
(`geomean_fit_ratio=0.9875`), but seed 0 became a larger row-level failure
(`fit_ratio=1.1966`) and every seed's median-repeat ratio still favored
upstream. The extra precompute also adds an `(n_features, n_samples)` float64
matrix to binary fits, so the risk/reward is poor.

Decision: product code was reverted. Linear-leaf speed is still a plausible
area, but this broad precompute is not the right cut.

### Same-Revision Timing Control

The numeric-binary stress blocker exposed a problem with the raw timing gate:
it fails even when both labels point at the same bbstats v2 checkout. The
same-revision controls are tracked in:

- `benchmarks/catboost_same_revision_numeric_binary_stress_trace_r20_20260606.csv`
- `benchmarks/catboost_same_revision_numeric_binary_stress_trace_r20_summary_20260606.csv`
- `benchmarks/catboost_same_revision_numeric_binary_stress_trace_r20_report_20260606.json`
- `benchmarks/catboost_same_revision_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`
- `benchmarks/catboost_same_revision_reversed_numeric_binary_stress_trace_r20_20260606.csv`
- `benchmarks/catboost_same_revision_reversed_numeric_binary_stress_trace_r20_summary_20260606.csv`
- `benchmarks/catboost_same_revision_reversed_numeric_binary_stress_trace_r20_report_20260606.json`
- `benchmarks/catboost_same_revision_reversed_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`

Result: the raw checker is too sensitive for row-level fit minima on this case.
Default-order bbstats-v2-vs-bbstats-v2 failed with
`geomean_fit_ratio=1.0943` and four row-level timing failures. Reversing the
labels still left two row-level failures, despite aggregate
`geomean_fit_ratio=0.9878`. That is worse than the current
candidate-vs-upstream trace (`geomean_fit_ratio=0.9736`, two row failures).

The checker now accepts optional same-revision timing controls:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/check_strict_domination.py \
  benchmarks/catboost_numeric_binary_stress_trace_r20_20260606.csv \
  --mode upstream-compatible \
  --timing-control benchmarks/catboost_same_revision_numeric_binary_stress_trace_r20_20260606.csv \
  --timing-control benchmarks/catboost_same_revision_reversed_numeric_binary_stress_trace_r20_20260606.csv \
  --out benchmarks/catboost_numeric_binary_stress_trace_r20_calibrated_report_20260606.json
```

The calibrated report passes (`passed=true`, zero failures,
`geomean_fit_ratio=0.9736`). This does not prove the whole catboost matrix is
finished, but it does retire numeric-binary stress as a product-code blocker
under the measured timing-noise floor.

Decision: stop micro-optimizing numeric-binary stress against the uncalibrated
row-min gate. The next acceptance step is a full catboost gate rerun with
same-revision timing controls; only calibrated failures should drive more
catboost-mode product changes.

### Selected Row/Feature Kernels

The selected-row and selected-feature histogram kernels are inactive in the
strict default matrix: `subsample=1.0` returns `row_indices=None`, and
`colsample=1.0` returns `feature_indices=None`. Therefore they cannot explain
the current strict blockers. Tests already prove they match the masked/full
histogram behavior when non-default subsample or colsample settings activate
them.

Decision: keep the selected row/feature kernels as a behavior-proved darko
optimization for non-default sampling settings. They do not need another
default catboost ablation before the remaining strict blockers are addressed.

## Quantile Benchmark Coverage

Raw medium quantile rows are tracked in:

- `benchmarks/tri_compare_quantile_medium_20260606.csv`
- `benchmarks/tri_compare_quantile_medium_summary_20260606.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --upstream /private/tmp/chimeraboost-upstream-ddaf272-bobw \
  --fork /private/tmp/chimeraboost-fork-origin-main-bobw \
  --candidate . \
  --models upstream_matched fork_matched candidate_catboost candidate_lightgbm \
  --datasets quantile_reg_10 quantile_reg_50 quantile_reg_90 \
  --sizes medium \
  --seeds 2 \
  --repeat 2 \
  --iterations 300 \
  --patience 25 \
  --threads 4 \
  --weight-modes none stress \
  --csv benchmarks/tri_compare_quantile_medium_20260606.csv
```

All 48 rows completed successfully. Ratios below are against
`upstream_matched`; lower is better.

| Variant | Mean pinball metric ratio | Mean fit-time ratio | Interpretation |
| --- | ---: | ---: | --- |
| `candidate_catboost` | 0.996 | 1.129 | Preserves upstream quantile quality; slightly better on weighted pinball, modestly slower. |
| `fork_matched` | 1.042 | 0.952 | Faster, but gives up pinball quality. |
| `candidate_lightgbm` | 1.158 | 1.083 | Still fails the quantile quality gate. |

Decision: quantile regression reinforces the existing default. Keep
`tree_mode="catboost"` as the product path and leave `tree_mode="lightgbm"` as
an opt-in mode until it wins primary holdout loss.

## Grouped Split Coverage

Raw medium grouped rows are tracked in:

- `benchmarks/tri_compare_grouped_medium_20260606.csv`
- `benchmarks/tri_compare_grouped_medium_summary_20260606.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --upstream /private/tmp/chimeraboost-upstream-ddaf272-bobw \
  --fork /private/tmp/chimeraboost-fork-origin-main-bobw \
  --candidate . \
  --models upstream_matched fork_matched candidate_catboost candidate_lightgbm \
  --datasets friedman_numeric numeric_binary categorical_binary \
  --sizes medium \
  --seeds 2 \
  --repeat 2 \
  --iterations 300 \
  --patience 25 \
  --threads 4 \
  --weight-modes none stress \
  --split-modes group \
  --csv benchmarks/tri_compare_grouped_medium_20260606.csv
```

All 48 rows completed successfully. Every row used `split_mode=group`, with
whole groups held out across fit/validation/test (`300/75/125` groups in the
medium cases). Ratios below are against `upstream_matched`; lower is better.

| Variant | Mean primary-metric ratio | Mean fit-time ratio | Interpretation |
| --- | ---: | ---: | --- |
| `candidate_catboost` | 1.000 | 1.149 | Preserves upstream grouped-holdout quality, but is not a broad speed win. |
| `fork_matched` | 1.116 | 1.672 | Usually gives up quality on grouped holdouts. |
| `candidate_lightgbm` | 1.137 | 0.903 | Faster on most rows, but still misses the primary-metric gate. |

Decision: grouped holdout evidence also supports keeping `tree_mode="catboost"`
as the default/product path. Levelwise remains an opt-in speed experiment until
it can win primary out-of-sample loss.

## Bagging Coverage

Raw small bagging rows are tracked in:

- `benchmarks/tri_compare_bagging_small_20260606.csv`
- `benchmarks/tri_compare_bagging_small_summary_20260606.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --upstream /private/tmp/chimeraboost-upstream-ddaf272-bobw \
  --fork /private/tmp/chimeraboost-fork-origin-main-bobw \
  --candidate . \
  --models upstream_matched fork_matched candidate_catboost candidate_lightgbm \
  --datasets friedman_numeric numeric_binary \
  --sizes small \
  --seeds 1 \
  --repeat 2 \
  --iterations 120 \
  --patience 20 \
  --threads 4 \
  --weight-modes none \
  --ensemble-sizes 1 3 \
  --ensemble-n-jobs 1 \
  --csv benchmarks/tri_compare_bagging_small_20260606.csv
```

The run produced 16 rows: 14 successful rows and 2 explicit error rows for the
legacy fork's unsupported `n_ensembles=3` request. The harness now treats that
as a visible capability failure instead of silently benchmarking a single model
under a bagged label.

On this small two-dataset gate, `n_ensembles=3` still did not rescue the
levelwise quality gap. The corrected summary now keeps `ensemble_size` in the
grouping key. Against equally bagged `upstream_matched`, candidate catboost tied
the upstream primary metric on all four dataset/ensemble rows and had mean fit
ratio `1.057`. Candidate levelwise was faster (`0.588` mean fit ratio) but
worse on the primary metric (`1.266` mean ratio). The legacy fork was fastest
on its two successful single-model rows, but worse on quality and still errors
explicitly for `n_ensembles=3`.

Decision: bagging is now an explicit benchmark dimension. Do not use bagged rows
to justify a tree-mode default unless they are compared against equally bagged
catboost/upstream rows on the same splits.

## Real-Tabular Harness Coverage

The revision and levelwise-tuning harnesses now accept opt-in real-tabular
dataset namespaces:

- `--openml` registers the OpenML suite as `oml:<name>`.
- `--grinsztajn` registers the Grinsztajn/HuggingFace suite as
  `gr:<folder>/<name>`.
- Explicit dataset names such as `oml:credit-g` or `gr:clf_num/credit` are
  registered automatically even without the broad suite flag.

Registration is lazy: dataset names become available before any network or cache
read happens, and rows are fetched only when a benchmark case is built.

Smoke command, using the local cached OpenML `credit-g` dataset:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --candidate . \
  --models candidate_catboost \
  --datasets oml:credit-g \
  --sizes tiny \
  --seeds 1 \
  --repeat 1 \
  --iterations 3 \
  --patience 2 \
  --threads 1 \
  --weight-modes none \
  --openml \
  --csv /tmp/cb_openml_revision_smoke.csv
```

Result: the single `candidate_catboost` row completed successfully. No tracked
real-tabular decision benchmark had been run at that point; the patch opened the
gate so upstream/fork/candidate comparisons can use real datasets directly.

First tracked real-tabular gate:

`benchmarks/tri_compare_openml_tiny_20260606.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --upstream /private/tmp/chimeraboost-upstream-ddaf272-bobw \
  --fork /private/tmp/chimeraboost-fork-origin-main-bobw \
  --candidate . \
  --models upstream_matched fork_matched candidate_catboost candidate_lightgbm \
  --datasets oml:credit-g oml:phoneme oml:vehicle oml:cpu_act \
  --sizes tiny \
  --seeds 2 \
  --repeat 2 \
  --iterations 120 \
  --patience 15 \
  --threads 4 \
  --weight-modes none \
  --openml \
  --csv benchmarks/tri_compare_openml_tiny_20260606.csv
```

All 32 rows completed successfully. Ratios below are against
`upstream_matched`; lower is better because every primary metric here is a loss.

| Dataset | Task | Variant | Metric ratio | Fit-time ratio | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| `oml:credit-g` | binary, categorical | `candidate_catboost` | 1.000 | 1.273 | Preserves upstream log loss exactly. |
| `oml:credit-g` | binary, categorical | `candidate_lightgbm` | 1.063 | 1.059 | Worse log loss; no speed win here. |
| `oml:credit-g` | binary, categorical | `fork_matched` | 0.996 | 1.069 | Slightly better log loss, but runs many more rounds. |
| `oml:phoneme` | binary, numeric | `candidate_catboost` | 1.000 | 1.006 | Preserves upstream log loss exactly. |
| `oml:phoneme` | binary, numeric | `candidate_lightgbm` | 1.038 | 0.844 | Faster, but worse log loss. |
| `oml:phoneme` | binary, numeric | `fork_matched` | 1.046 | 1.320 | Worse and slower. |
| `oml:vehicle` | multiclass, numeric | `candidate_catboost` | 1.000 | 1.770 | Preserves upstream log loss exactly. |
| `oml:vehicle` | multiclass, numeric | `candidate_lightgbm` | 1.009 | 0.474 | Much faster, but still worse log loss. |
| `oml:vehicle` | multiclass, numeric | `fork_matched` | 1.003 | 0.783 | Slightly worse log loss, faster. |
| `oml:cpu_act` | regression, numeric | `candidate_catboost` | 1.000 | 0.703 | Preserves upstream RMSE exactly and is faster in this tiny gate. |
| `oml:cpu_act` | regression, numeric | `candidate_lightgbm` | 0.607 | 0.813 | Better RMSE and faster on this one tiny regression case. |
| `oml:cpu_act` | regression, numeric | `fork_matched` | 0.545 | 0.236 | Best RMSE/time on this one tiny regression case. |

Decision: this tiny OpenML gate supports the same conservative integration
stance for classification: `candidate_catboost` preserves upstream quality,
while `candidate_lightgbm` buys speed by giving up log loss. The regression row
is an interesting signal for a broader real-regression follow-up, not enough by
itself to change defaults.

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
3. Improve the shared vector-valued multiclass levelwise path only if it can
   close the remaining quality gap without hurting the categorical multiclass
   speed win.
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

## Weighted Target-Stat Encoding

Implemented `weighted_target_stats=False` as an explicit opt-in. When enabled,
ordered target-stat categorical encodings use normalized `sample_weight` as
category mass; when disabled, weighted fits keep the historical unweighted
target-stat encodings while still using weights for gradients, validation
metrics, temperature scaling, and diagnostics.

Raw opt-in benchmark rows are tracked in:

`benchmarks/weighted_target_stats_medium_20260605.csv`

Command:

```bash
/Users/kmedved/miniconda3/envs/darko311/bin/python benchmarks/bench_compare_revisions.py \
  --fork /private/tmp/chimeraboost-before-weighted-ts-b474ae9 \
  --candidate . \
  --models fork_catboost_matched fork_lightgbm_matched candidate_catboost candidate_lightgbm \
  --datasets categorical_reg categorical_binary categorical_multiclass \
  --sizes medium \
  --seeds 2 \
  --repeat 2 \
  --iterations 300 \
  --patience 25 \
  --threads 4 \
  --weight-modes none stress \
  --weighted-target-stats \
  --csv benchmarks/weighted_target_stats_medium_20260605.csv
```

Summary versus the previous default:

| Mode | Metric ratio | Fit-time ratio | Iteration ratio |
| --- | ---: | ---: | ---: |
| `candidate_catboost` opt-in | 1.0073 | 1.8620 | 0.9841 |
| `candidate_lightgbm` opt-in | 1.0075 | 1.2789 | 0.9886 |

Decision: keep the default off. The option is useful when weights are intended
as frequency/reliability mass for categorical priors, but the benchmark does not
support silently changing default weighted-fit semantics.

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
