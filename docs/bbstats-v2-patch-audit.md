# bbstats v2 Patch Audit

This document tracks the work to go through every upstream `bbstats` v2 patch
relative to the original fork point and decide how the best-of-both-worlds
branch should handle it.

The target is not to preserve the old fork for its own sake. The target is a
`tree_mode="catboost"` path that strictly dominates current bbstats v2 under a
literal upstream-compatible benchmark gate: same quality, same public behavior,
and no stable speed regression. When upstream v2 is better, use upstream. When
the darko v1 fork is better and behavior-equivalent, preserve the fork
optimization. When the answer is workload-dependent, keep it behind a
benchmark-gated toggle instead of changing the product default.

## Reference Points

| Role | Ref | Notes |
| --- | --- | --- |
| Fork point / bbstats v1 | `78397b27d7d27fc055490bb21ab8dd1b68893e13` | Merge base of `origin/main` and `upstream/main`. |
| darko v1 | `origin/main@1eacb51e8945df90f48b3c0dfb5719f91f44dcf7` | Local fork work before the bbstats v2 rewrite was integrated. |
| bbstats v2 | `upstream/main@ddaf2725c1f13d567bf8828963377d152ca9a8c8` | Current upstream rewrite baseline. |
| Best-of-both-worlds branch | `codex/best-of-both-worlds` | Upstream v2 trunk plus benchmark-gated darko patches. |

Patch counts from the fork point:

- bbstats v2: 86 commits, 81 non-merge commits.
- darko v1: 14 changed files, mostly benchmark harnesses and kernel speed work.
- current branch over bbstats v2: benchmark gates, docs, and opt-in
  tree/kernel paths. The exact file count changes as tracked benchmark
  artifacts land.

## Decision Labels

| Label | Meaning |
| --- | --- |
| `KEEP-UPSTREAM` | Upstream v2 is the product behavior. Keep it in catboost mode. |
| `KEEP-UPSTREAM + STRICT-GATE` | Keep upstream behavior, but require the strict-domination gate before claiming an improvement. |
| `PRESERVE-DARKO` | Keep a darko v1 optimization because it is behavior-equivalent or product-positive. |
| `BENCHMARK-TOGGLE` | Keep both paths or tune later; do not make a default change without row-level benchmark evidence. |
| `DOCS/RELEASE/ADVISORY` | Documentation, charts, release metadata, or process changes. Not a catboost-mode blocker. |
| `MERGE/NOOP` | Merge-only commit. Covered by the commits it merges. |

## Patch Family Decisions

### Product API And Defaults

Decision: `KEEP-UPSTREAM + STRICT-GATE`

Upstream owns the modern public API and defaults:

- `n_estimators` replaces `iterations`.
- `cat_features` can be supplied on the estimator constructor.
- `cat_features` may be column names for pandas inputs.
- classifier defaults include `ordered_boosting=False`,
  `leaf_estimation_iterations=3`, `early_stopping_rounds=50`,
  `l2_leaf_reg=1.0`, and size-adaptive `min_child_weight`.
- regressor defaults include `ordered_boosting=False` and the upstream
  validation/early-stopping shape.

The current branch should not second-guess these defaults in catboost mode
unless the strict gate proves an alternative is quality-equivalent and faster.

### Validation And Robustness

Decision: `KEEP-UPSTREAM`

Keep upstream validation hardening, estimator compliance fixes, pyarrow feature
name handling, masked-array rejection, `cat_smoothing > 0`, and quantile
calibration fixes. These are correctness and product-surface patches, not speed
experiments.

One intentional product improvement remains in the candidate: validation
weights may be passed with `eval_set=(X_val, y_val, sample_weight_val)`. For
literal upstream comparisons, use:

```bash
--validation-weight-policy upstream-compatible
```

For the product semantics lane, use:

```bash
--validation-weight-policy product
```

### Catboost Product Features

Decision: `KEEP-UPSTREAM + STRICT-GATE`

Keep upstream's exact oblivious-tree SHAP, linear leaves, hierarchical
shrinkage, bagging/ensembles, and OOB early stopping in catboost mode. The
opt-in `tree_mode="lightgbm"` path may fail explicitly for unsupported
catboost-only features, but catboost mode must remain the upstream product path.

### Darko Optimizations Already Preserved

Decision: `PRESERVE-DARKO`

The current branch preserves darko v1 speed work only where it is isolated or
has behavior proof:

- compact binned dtypes through `_bin_dtype_for_n_bins`;
- constant-Hessian histogram paths for RMSE-style fits;
- selected-feature and selected-row histogram fills;
- class-major multiclass buffers;
- grouped weighted MAE/Quantile leaf correction plus the median-Quantile
  unweighted correction;
- weighted target stats as an opt-in product improvement;
- scalar-loop cleanup for inactive timing, feature-sampling, and row-sampling
  branches;
- timing diagnostics, strict benchmark adapters, and strict-domination checker;
- opt-in `tree_mode="lightgbm"` / `tree_mode="levelwise"` research path.

These stay in catboost mode only where the evidence below shows they preserve
upstream quality and public behavior. Rejected darko restorations, including
the reusable split-scratch probe, remain documented in the completed queue and
should not be reintroduced without a new one-change gate.

### Workload-Dependent Areas And Current Defaults

Decision: `BENCHMARK-TOGGLE`

The initial audit left these as suspect surfaces. Current evidence resolves the
default choice for each one, but keeps the "one-change gate before promotion"
rule for future revisits:

- plain upstream-style tree helper versus specialized kernels: rejected by the
  fast full-hist, upstream-tree-lane, and plain-builder probes;
- compact bins versus upstream-style `uint16` bins: keep compact bins by
  default; forced/adaptive `uint16` was not broadly safe;
- categorical factorization versus upstream's pandas-vectorized categorical
  encoding: keep upstream's pandas-vectorized path by default;
- selected-row/selected-feature kernels outside strict default settings: keep
  the darko kernels for sampling modes, but do not claim default-gate wins from
  them because they are inactive at `subsample=1.0` / `colsample=1.0`;
- learning-rate or ordered-boosting default changes: keep upstream defaults;
  use these as explicit user knobs rather than automatic catboost-mode changes.

Future changes in these areas need a new one-change ablation against a stable
strict-gate failure.

### Docs, Charts, CI, And Release Metadata

Decision: `DOCS/RELEASE/ADVISORY`

Keep useful upstream docs/CI/release metadata when it does not conflict with
the branch's benchmark claims. Benchmark images and README performance claims
are treated as upstream historical context; current branch claims must point to
tracked CSVs and the strict-domination checker.

## Source-File Patch Coverage

The table below is generated from `git diff --numstat` over the three important
comparisons:

- `78397b27..upstream/main` for bbstats v2;
- `78397b27..origin/main` for darko v1;
- `upstream/main..HEAD` for the current best-of-both-worlds branch.

Counts are insertion/deletion totals. They are not quality judgments by
themselves; the decision column ties each file-level patch surface back to the
measured decisions in this document.

| File | bbstats v2 | darko v1 | current vs v2 | Decision |
| --- | ---: | ---: | ---: | --- |
| `chimeraboost/__init__.py` | +1/-1 | +0/-0 | +1/-1 | Keep upstream package surface; current branch only reflects branch-local metadata. |
| `chimeraboost/binning.py` | +38/-0 | +13/-2 | +39/-4 | Keep compact-bin default and validation; forced/adaptive upstream-style `uint16` was rejected except as a future gated experiment. |
| `chimeraboost/booster.py` | +364/-149 | +196/-66 | +355/-59 | Keep upstream catboost semantics, defaults, linear leaves, and OOB behavior; preserve measured darko/current loop cleanup and diagnostics; darko-style in-place prediction is low-ceiling; scalar tree-build timing remains the open blocker. |
| `chimeraboost/losses.py` | +20/-40 | +26/-1 | +26/-1 | Keep upstream loss behavior; preserve weighted metric/eval support where gates cover it. |
| `chimeraboost/preprocessing.py` | +77/-22 | +10/-6 | +13/-4 | Keep upstream pandas-vectorized categorical encoding; manual/lazy categorical mapping was rejected as a default; preserve only gated bin/target-stat extensions. |
| `chimeraboost/sklearn_api.py` | +1026/-98 | +15/-3 | +154/-55 | Keep upstream public API, validation, `n_estimators`, constructor `cat_features`, and cat-feature names; current additions are compatibility toggles, timing diagnostics, and weighted-validation policy lanes. |
| `chimeraboost/target_encoding.py` | +31/-22 | +38/-1 | +28/-5 | Keep upstream ordered target-encoding behavior; preserve weighted target stats as an opt-in product improvement. |
| `chimeraboost/tree.py` | +563/-94 | +692/-41 | +876/-5 | Keep upstream oblivious-tree semantics; preserve only proven darko/current kernels and research modes; split scratch, darko serial histogram kernels, upstream-default lane, plain-builder lane, dtype-only, and branch-only probes are rejected. |
| `tests/test_chimeraboost.py` | +799/-54 | +871/-0 | +496/-0 | Preserve upstream compliance/validation tests and darko/current behavior coverage. |
| `tests/test_benchmark_adapters.py` | +0/-0 | +0/-0 | +702/-0 | Preserve current strict-domination adapter coverage; this is the evidence harness for bbstats-v2 comparisons. |

### Benchmark Harness Patch Family

Decision: `PRESERVE-DARKO + KEEP-UPSTREAM`

Darko v1 changed several legacy benchmark scripts
(`bias_variance.py`, `capacity_sweep.py`, `diagnose_openml.py`,
`plot_frontier.py`, `profile_tree_kernels.py`, and `run_benchmarks.py`). Current
best-of-both-worlds keeps upstream v2's user-facing benchmark suite and adds a
new isolated revision harness instead:

- `benchmarks/bench_compare_revisions.py`
- `benchmarks/benchmark_adapters.py`
- `benchmarks/check_strict_domination.py`
- focused CSV/JSON artifacts for one-change gates.

Do not restore the removed darko v1 one-off scripts as maintained entrypoints.
Their useful lessons are now represented by the revision-isolated harness,
repeat traces, same-revision timing controls, phase summaries, and direct
microbench notes recorded in this audit. Keep upstream's `run_benchmarks.py`
for package-facing benchmark charts; use `bench_compare_revisions.py` for
fork/upstream/candidate decisions.

## Complete Upstream v2 Commit Inventory

Inventory check: `git log --no-merges 78397b27..upstream/main` returns 81
upstream code/doc/release commits, and all 81 short hashes are represented
below. A stricter all-commit check on 2026-06-07 returned
`upstream commits 86 missing 0` for:
`git rev-list --reverse --abbrev-commit --abbrev=7 --oneline 78397b27d7d27fc055490bb21ab8dd1b68893e13..upstream/main`
against this file. The additional documented short hashes are merge commits
marked `MERGE/NOOP`.

| Commit | Date | Subject | Decision |
| --- | --- | --- | --- |
| `d3f4d24` | 2026-05-28 | clod | `DOCS/RELEASE/ADVISORY` |
| `109a4e9` | 2026-05-28 | fix readme | `DOCS/RELEASE/ADVISORY` |
| `109d6ff` | 2026-05-28 | Revise parameter description and benchmark details | `DOCS/RELEASE/ADVISORY` |
| `6fced7c` | 2026-05-28 | Correct 'is' to 'are' in results description | `DOCS/RELEASE/ADVISORY` |
| `dc9db89` | 2026-05-28 | cleanup | `DOCS/RELEASE/ADVISORY` |
| `886516c` | 2026-05-28 | table fixes | `DOCS/RELEASE/ADVISORY` |
| `de26b84` | 2026-05-28 | Merge branch 'main' of https://github.com/bbstats/chimeraboost | `MERGE/NOOP` |
| `e3c66ac` | 2026-05-28 | de-slop | `DOCS/RELEASE/ADVISORY` |
| `d5b7bc0` | 2026-05-28 | update benchmarks | `DOCS/RELEASE/ADVISORY` |
| `ed4db3d` | 2026-05-28 | Add subtitle to README for accuracy and speed | `DOCS/RELEASE/ADVISORY` |
| `95c6350` | 2026-05-28 | Update README.md | `DOCS/RELEASE/ADVISORY` |
| `4b1843f` | 2026-05-28 | Fix formatting in README.md for performance comparison | `DOCS/RELEASE/ADVISORY` |
| `08851ad` | 2026-05-28 | pypi release 0.6.0 | `DOCS/RELEASE/ADVISORY` |
| `79ff677` | 2026-05-28 | Merge branch 'main' of https://github.com/bbstats/chimeraboost | `MERGE/NOOP` |
| `ac2e228` | 2026-05-28 | Stop tracking compiled test bytecode | `KEEP-UPSTREAM` |
| `10c0d2b` | 2026-05-29 | cleanup | `DOCS/RELEASE/ADVISORY` |
| `2a6b5d7` | 2026-05-29 | Release 0.7.0: faster predict/fit, slowdown histogram, de-slop | `DOCS/RELEASE/ADVISORY` |
| `eea3c94` | 2026-05-29 | Raise default early_stopping_rounds 10 -> 50 | `KEEP-UPSTREAM + STRICT-GATE` |
| `e3c74be` | 2026-05-29 | Release 0.7.1: default early_stopping_rounds 10 -> 50 | `DOCS/RELEASE/ADVISORY` |
| `001e3cb` | 2026-05-29 | Merge branch 'main' of https://github.com/bbstats/chimeraboost | `MERGE/NOOP` |
| `5e68f68` | 2026-05-30 | Release 0.8.0: first-class bagging (n_ensembles) + Brier benchmark metric | `KEEP-UPSTREAM + STRICT-GATE` |
| `36c2174` | 2026-05-31 | Default ordered_boosting=False for ChimeraBoostRegressor | `KEEP-UPSTREAM + STRICT-GATE` |
| `88d5d55` | 2026-05-31 | Add leaf_estimation_iterations parameter (default 1, no behavior change) | `KEEP-UPSTREAM` |
| `ba83df5` | 2026-05-31 | Add /bench status command + reusable summarize module + live progress | `DOCS/RELEASE/ADVISORY` |
| `d54aa80` | 2026-05-31 | Default ordered_boosting=False for ChimeraBoostClassifier | `KEEP-UPSTREAM + STRICT-GATE` |
| `1a15c24` | 2026-06-01 | Default leaf_estimation_iterations=3 for ChimeraBoostClassifier (+1.0pp Brier, +0.2pp F1, reg unchanged) | `KEEP-UPSTREAM + STRICT-GATE` |
| `fe0452c` | 2026-06-01 | Update benchmark images (94.2% Brier, 98.7% F1 after clf lei=3 default) | `DOCS/RELEASE/ADVISORY` |
| `da6b1f3` | 2026-06-01 | Cite Grinsztajn et al. (2022) in benchmark captions and images | `DOCS/RELEASE/ADVISORY` |
| `ab810f9` | 2026-06-01 | Add blended-strength Pareto + guard near-solved datasets from RMSE | `DOCS/RELEASE/ADVISORY` |
| `1b2d9f3` | 2026-06-01 | Exempt empty children from min_child_weight veto (fix oblivious depth cap) | `KEEP-UPSTREAM + STRICT-GATE` |
| `2ea3b4d` | 2026-06-01 | Release 0.9.0: oblivious depth-cap fix + blended-strength Pareto | `DOCS/RELEASE/ADVISORY` |
| `70ecf96` | 2026-06-01 | Release 0.9.1: tidy README + benchmark tables, correct speed claims | `DOCS/RELEASE/ADVISORY` |
| `71726da` | 2026-06-01 | Robustness pass: input validation + scikit-learn check_estimator compliance | `KEEP-UPSTREAM` |
| `9f76502` | 2026-06-01 | Size-adaptive classifier min_child_weight (closes the Brier gap, reaches Pareto frontier) | `KEEP-UPSTREAM + STRICT-GATE` |
| `0699407` | 2026-06-01 | Update README to remove scikit-learn compatibility section | `DOCS/RELEASE/ADVISORY` |
| `20ad819` | 2026-06-01 | Vectorize categorical encoding via pandas (~15% faster cat fits, bit-identical) | `KEEP-UPSTREAM + BENCHMARK-TOGGLE` |
| `f1fb307` | 2026-06-01 | Update benchmark images (cat-encoding speedup: 2.7x -> 2.6x slowdown, accuracy unchanged) | `DOCS/RELEASE/ADVISORY` |
| `ae884ed` | 2026-06-02 | Lower l2_leaf_reg default 3.0 -> 1.0 (Brier +1.5pp, RMSE flat) | `KEEP-UPSTREAM + STRICT-GATE` |
| `9349302` | 2026-06-02 | Bump to 0.9.2; update benchmark images and README | `DOCS/RELEASE/ADVISORY` |
| `178ce96` | 2026-06-02 | Fix ensemble early stopping: use OOB rows instead of auto-splitting bootstrap | `KEEP-UPSTREAM` |
| `422270c` | 2026-06-02 | Add ChimeraBoostEns2/Ens5 to benchmark harness; update charts | `DOCS/RELEASE/ADVISORY` |
| `e75b890` | 2026-06-02 | Fix Pareto label overlap: nudge ChimeraBoostEns2 label left | `DOCS/RELEASE/ADVISORY` |
| `22555e7` | 2026-06-02 | Add Ens2/Ens5 to summary table and slowdown histogram | `DOCS/RELEASE/ADVISORY` |
| `3079eed` | 2026-06-02 | Pareto: bold all ChimeraBoost labels, drop (ours) suffix | `DOCS/RELEASE/ADVISORY` |
| `5daf0bc` | 2026-06-02 | Show model config (max trees, patience, val split, seeds) in all benchmark charts | `DOCS/RELEASE/ADVISORY` |
| `1d3cd12` | 2026-06-02 | Add cores/model to all benchmark charts; fix sklearn_HGB thread count | `DOCS/RELEASE/ADVISORY` |
| `e240acf` | 2026-06-02 | README: fix broken tagline and drop duplicated feature bullets | `DOCS/RELEASE/ADVISORY` |
| `393a8d7` | 2026-06-02 | Remove dead speculative code and redundant comments | `KEEP-UPSTREAM` |
| `1cdbf20` | 2026-06-02 | infra: fix sklearn baseline bug and establish Stage-1 diagnostic loop | `DOCS/RELEASE/ADVISORY` |
| `73b5c0d` | 2026-06-02 | docs: capacity gate verdict + pol depth-tuning guidance | `DOCS/RELEASE/ADVISORY` |
| `06063df` | 2026-06-02 | Release 0.9.2: sync version, changelog, license metadata | `DOCS/RELEASE/ADVISORY` |
| `7119261` | 2026-06-02 | 0.10.0: early-stopping defaults so out-of-box == benchmarked | `KEEP-UPSTREAM + STRICT-GATE` |
| `042bbb7` | 2026-06-02 | benchmarks: restore Ens2/Ens5 in charts | `DOCS/RELEASE/ADVISORY` |
| `fcdc874` | 2026-06-03 | README: lead with the TabArena-Lite Elo/speed story | `DOCS/RELEASE/ADVISORY` |
| `fba45b2` | 2026-06-03 | Drop dev-narrative from source docstrings | `KEEP-UPSTREAM` |
| `aa3675b` | 2026-06-03 | README: drop the TabArena-Lite text bullet (the Pareto chart says it) | `DOCS/RELEASE/ADVISORY` |
| `49071aa` | 2026-06-03 | tabarena pareto: add Linear baseline | `DOCS/RELEASE/ADVISORY` |
| `9a46f01` | 2026-06-03 | tabarena pareto: label the linear baseline "Linear regression" | `DOCS/RELEASE/ADVISORY` |
| `2f30f62` | 2026-06-03 | tabarena pareto: label the linear baseline "Linear" | `DOCS/RELEASE/ADVISORY` |
| `425dfb3` | 2026-06-03 | Add linear leaf models (default-on for binary) + hs_lambda knob | `KEEP-UPSTREAM + STRICT-GATE` |
| `dd9ff6b` | 2026-06-03 | benchmarks: --chimera-linear-leaves flag + linear-leaf/HS dev panels | `DOCS/RELEASE/ADVISORY` |
| `a288f3a` | 2026-06-03 | tabarena pareto: refresh for linear-leaf default (Elo 1212->1219) | `DOCS/RELEASE/ADVISORY` |
| `e8e7961` | 2026-06-03 | README: cite linear-leaf trees + hierarchical shrinkage; drop base GBM ref | `DOCS/RELEASE/ADVISORY` |
| `b3f5591` | 2026-06-03 | README: order citations by year | `DOCS/RELEASE/ADVISORY` |
| `ff6f248` | 2026-06-03 | Add exact SHAP feature attributions (shap_values) | `KEEP-UPSTREAM` |
| `e4ed6f0` | 2026-06-03 | Add MkDocs Material docs site with task recipes | `DOCS/RELEASE/ADVISORY` |
| `2087ced` | 2026-06-03 | Add parameter reference, SHAP guide, and auto API docs | `DOCS/RELEASE/ADVISORY` |
| `c569162` | 2026-06-03 | Tighten docs prose and add a complete API reference | `DOCS/RELEASE/ADVISORY` |
| `35272d9` | 2026-06-04 | Restructure docs into a user guide + reference (River/Polars/FLAML style) | `DOCS/RELEASE/ADVISORY` |
| `0786e93` | 2026-06-04 | README: link the documentation site | `DOCS/RELEASE/ADVISORY` |
| `756f712` | 2026-06-04 | docs CI: rebuild on chimeraboost/** changes too | `KEEP-UPSTREAM` |
| `7a0dd29` | 2026-06-04 | Harden input/param validation and fix quantile calibration | `KEEP-UPSTREAM` |
| `3c56c82` | 2026-06-04 | Fix pyarrow feature-name pollution and reject masked arrays | `KEEP-UPSTREAM` |
| `11c28a1` | 2026-06-04 | docs: loss-adaptive depth, input validation, and assume_finite | `DOCS/RELEASE/ADVISORY` |
| `cea241f` | 2026-06-04 | Accept cat_features as a constructor argument | `KEEP-UPSTREAM` |
| `cf53a67` | 2026-06-04 | docs: note cat_features can be set on the constructor | `DOCS/RELEASE/ADVISORY` |
| `a22ac14` | 2026-06-04 | Reject cat_smoothing<=0 (was 0/0 ZeroDivisionError in ordered TS) | `KEEP-UPSTREAM` |
| `b795b96` | 2026-06-04 | Add knob-characterization harness (does each hyperparameter work/help) | `DOCS/RELEASE/ADVISORY` |
| `f2fee8b` | 2026-06-04 | docs: cat_smoothing must be >0; note ordered_boosting tends to hurt | `DOCS/RELEASE/ADVISORY` |
| `d40b0f1` | 2026-06-04 | docs: make the site homepage the README verbatim (single source) | `DOCS/RELEASE/ADVISORY` |
| `a3955fd` | 2026-06-04 | docs: make the site homepage the README verbatim (single source) | `DOCS/RELEASE/ADVISORY` |
| `f1aa821` | 2026-06-04 | ci: run pytest on PRs and main; flesh out CHANGELOG [Unreleased] | `KEEP-UPSTREAM` |
| `cc548e4` | 2026-06-04 | Merge pull request #1 from bbstats/validation-hardening | `MERGE/NOOP` |
| `9b8fff8` | 2026-06-04 | Rename iterations -> n_estimators; accept cat_features by column name | `KEEP-UPSTREAM` |
| `f3bd6d9` | 2026-06-04 | Release 0.11.0 | `DOCS/RELEASE/ADVISORY` |
| `ddaf272` | 2026-06-04 | Merge pull request #2 from bbstats/rename-n-estimators-and-cat-feature-names | `MERGE/NOOP` |

## Remaining Work Queue

Completed:

- Strict upstream-compatible medium gate:
  `benchmarks/catboost_strict_medium_20260606.csv`.
  Outcome: quality and iterations are identical for all paired rows; no
  semantic failures; aggregate fit is faster (`geomean_fit_ratio=0.9866`), but
  32 seed-level timing rows still fail the strict checker.
- Compact-bin ablation:
  `benchmarks/catboost_ablate_uint16_focus_20260606.csv`.
  Outcome: forced `uint16` helps numeric-binary stress and quantile-50 stress,
  but hurts categorical-binary unweighted, Friedman stress, quantile-90 stress,
  and wide-regression stress. Keep compact bins by default; treat upstream-style
  `uint16` as rejected for now. The later adaptive numeric-binary probe also
  failed the strict gate.
- Constant-Hessian ablation:
  `benchmarks/catboost_ablate_no_constant_hessian_focus_20260606.csv`.
  Outcome: disabling the constant-Hessian shortcut helps categorical regression
  and median quantile unweighted rows, but hurts Friedman, quantile-10,
  quantile-90, and wide regression. Keep the shortcut by default; do not add a
  general-Hessian adaptive toggle without a new full-gate win.
- Categorical-encoding ablation:
  `benchmarks/catboost_ablate_manual_cats_focus_20260606.csv`.
  Outcome: restoring the older manual/lazy categorical mapping improves
  aggregate categorical timing and helps categorical regression plus weighted
  categorical multiclass, but it hurts categorical-binary unweighted and still
  leaves row-level strict failures. Keep upstream's pandas-vectorized path by
  default; do not add a manual-mapping adaptive toggle without a new full-gate
  win.
- High-repeat blocker rerun:
  `benchmarks/catboost_strict_blockers_stress_r15_20260606.csv` and
  `benchmarks/catboost_strict_blockers_quantile90_r15_20260606.csv`.
  Outcome: categorical-binary stress and Friedman stress are no longer
  aggregate blockers at repeat 15. Numeric-binary stress, median-quantile
  stress, and quantile-90 unweighted remain the stable aggregate blockers.
- Weighted leaf-correction port:
  `benchmarks/catboost_grouped_leaf_q50_stress_r15_20260606.csv`.
  Outcome: restoring the darko v1 grouped correction only for weighted
  MAE/Quantile leaves clears the median-quantile stress blocker
  (`geomean_fit_ratio=0.8896`) with identical metrics and iterations. The same
  grouped correction is rejected for unweighted leaves:
  `benchmarks/catboost_grouped_leaf_q90_none_r15_20260606.csv` regressed q90
  (`geomean_fit_ratio=1.0491`). Keep upstream's mask loop for unweighted leaf
  correction and use the grouped path only when `sample_weight` is present.
- Weighted Quantile mask-fallback probes:
  `benchmarks/catboost_weighted_quantile_mask_nonmedian_r7_20260607.csv`,
  `benchmarks/catboost_weighted_quantile_mask_nonmedian_r7_report_20260607.json`,
  `benchmarks/catboost_weighted_quantile_lower_tail_mask_r7_20260607.csv`, and
  `benchmarks/catboost_weighted_quantile_lower_tail_mask_r7_report_20260607.json`.
  Outcome: restoring upstream's per-leaf mask loop for weighted non-median
  Quantile corrections failed the strict gate (`geomean_fit_ratio=1.0843`, 9
  failures), and the lower-tail-only variant was worse
  (`geomean_fit_ratio=1.1303`, 12 failures). Metrics and iterations were
  unchanged, so this was pure timing noise/regression. Reject both variants and
  keep the current grouped weighted correction.
- Adaptive upstream-style `uint16` probe:
  `benchmarks/catboost_adaptive_uint16_numeric_binary_stress_r15_20260606.csv`.
  Outcome: a narrow numeric-binary upstream-dtype policy failed the repeat-15
  gate (`geomean_fit_ratio=1.1014`). Do not promote it; keep compact bins as
  the default until a cleaner one-change gate proves otherwise.
- Higher-repeat residual blockers:
  `benchmarks/catboost_q90_none_r30_20260606.csv`,
  `benchmarks/catboost_q90_none_seed0_r80_20260606.csv`, and
  `benchmarks/catboost_numeric_binary_stress_r30_20260606.csv`.
  Outcome: q90 unweighted is no longer an aggregate blocker at repeat 30
  (`geomean_fit_ratio=0.9961`), but seed 0 remains a stable row-level timing
  failure at repeat 80 (`fit_ratio=1.0576`). Numeric-binary stress remains an
  aggregate blocker at repeat 30 (`geomean_fit_ratio=1.0271`).
- Fast full-hist branch probe:
  `benchmarks/catboost_fast_full_hist_numeric_binary_stress_r30_20260606.csv`.
  Outcome: splitting the default full-row/full-feature/general-Hessian path
  ahead of the selected-row/feature cascade failed the repeat-30 gate
  (`geomean_fit_ratio=1.0874`). The product code was reverted; do not promote
  this branch shuffle.
- Upstream-default tree lane:
  `benchmarks/catboost_upstream_tree_q90_none_r15_20260606.csv` and
  `benchmarks/catboost_upstream_tree_numeric_binary_stress_seed0_r80_20260606.csv`.
  Outcome: a copied bbstats v2 full-row/full-feature tree lane regressed q90
  (`geomean_fit_ratio=1.1335`) and still left numeric-binary seed 0 as a stable
  row-level failure (`fit_ratio=1.0693`). The product code was reverted; do not
  promote this lane.
- Scalar-loop timing cleanup:
  `benchmarks/catboost_timing_guard_q90_none_r15_20260606.csv`,
  `benchmarks/catboost_timing_guard_feature_skip_numeric_binary_stress_r15_20260606.csv`,
  and `benchmarks/catboost_scalar_loop_cleanup_numeric_binary_stress_r30_20260606.csv`.
  Outcome: guarding inactive `verbose_timing` timers and skipping default
  no-sampling/no-colsampling helper calls clears q90 at repeat 15
  (`geomean_fit_ratio=0.9132`) and clears numeric-binary stress at repeat 15
  (`geomean_fit_ratio=0.9237`). Numeric-binary remains unstable at higher
  repeats: repeat 30 passes aggregate but has one row failure
  (`geomean_fit_ratio=0.9402`), and repeat 50 failed after upstream found lower
  timing minima (`geomean_fit_ratio=1.1260`). Keep the behavior-preserving
  cleanup, but do not call strict domination complete yet.
- Numeric-binary repeat trace:
  `benchmarks/catboost_numeric_binary_stress_trace_r20_20260606.csv`,
  `benchmarks/catboost_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`,
  and `benchmarks/catboost_numeric_binary_stress_phase_r10_20260606.csv`.
  Outcome: the harness now records semicolon-delimited `fit_repeat_seconds`
  and `predict_repeat_seconds` columns. The real repeat-20 trace passes the
  aggregate min-of-repeat gate (`geomean_fit_ratio=0.9736`) but still has two
  row-level timing failures (seed 1 ratio `1.0580`, seed 4 ratio `1.0381`).
  The repeat distribution is not pure noise: median-repeat geomean is `1.1354`
  in favor of upstream. The phase run is candidate-only because upstream v2
  does not expose `verbose_timing`; use it only to locate candidate work, not
  to compare phases across revisions.
- Plain-builder fast lane:
  `benchmarks/catboost_plain_builder_numeric_binary_stress_trace_r20_20260606.csv`
  and
  `benchmarks/catboost_plain_builder_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`.
  Outcome: routing full-row/full-feature non-constant-Hessian catboost fits
  through an upstream-shaped direct builder call failed the numeric-binary
  stress repeat-20 gate (`geomean_fit_ratio=1.0214`) and still had a worse
  median-repeat geomean (`1.1525`). The product code was reverted; do not
  promote this call-shape-only fast lane.
- Benchmark-order probe:
  `benchmarks/catboost_numeric_binary_stress_reversed_order_r20_20260606.csv`,
  `benchmarks/catboost_numeric_binary_stress_reversed_order_r20_report_20260606.json`,
  and
  `benchmarks/catboost_numeric_binary_stress_reversed_order_r20_repeat_summary_20260606.csv`.
  Outcome: the revision harness now honors the order supplied to `--models`,
  and reversing the prior run order (`candidate_catboost` before
  `upstream_matched`) did not clear the blocker. The aggregate min-of-repeat
  gate improved (`geomean_fit_ratio=0.9585`) but seed 1 still failed at
  `1.0564`, and every seed's median and mean repeat ratios still favored
  upstream. Do not treat the remaining numeric-binary stress issue as a simple
  benchmark-order artifact.
- Bin-index cast probe:
  `benchmarks/catboost_numeric_binary_stress_profile_upstream_seed1_20260606.txt`,
  `benchmarks/catboost_numeric_binary_stress_profile_candidate_seed1_20260606.txt`,
  `benchmarks/catboost_hist_bin_int_numeric_binary_stress_trace_r20_20260606.csv`,
  and
  `benchmarks/catboost_hist_bin_int_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`.
  Outcome: cProfile on the failing seed pointed at tree-kernel time
  (`_build_histograms_into` and `_best_split`) rather than fit-loop keyword
  plumbing. A one-change ablation that cast default histogram bin ids to native
  `int` before indexing did not clear the gate (`geomean_fit_ratio=0.9759`) and
  worsened row failures on seeds 1 and 2 (`1.2292`, `1.1161`). The product code
  was reverted; do not promote this kernel-indexing tweak.
- Tree-kernel dtype microprobe:
  `benchmarks/catboost_tree_kernel_dtype_microprobe_20260606.csv`.
  Outcome: direct candidate-kernel timings on the failing seed show compact
  `uint8` and upstream-style `uint16` binned matrices are essentially tied for
  full `build_oblivious_tree`, direct histogram fill, and direct split search.
  That rules out dtype specialization alone as the remaining numeric-binary
  stress explanation.
- Linear-leaf precompute probe:
  `benchmarks/catboost_linear_xstd_numeric_binary_stress_trace_r20_20260606.csv`
  and
  `benchmarks/catboost_linear_xstd_numeric_binary_stress_trace_r20_repeat_summary_20260606.csv`.
  Outcome: precomputing standardized binned feature values once per binary fit
  improved the aggregate min-of-repeat ratio (`geomean_fit_ratio=0.9875`) but
  still failed seed 0 (`fit_ratio=1.1966`) and every seed's median-repeat ratio
  still favored upstream. The product code was reverted; do not promote this
  linear-leaf precompute without a broader redesign.
- Same-revision timing control:
  `benchmarks/catboost_same_revision_numeric_binary_stress_trace_r20_20260606.csv`,
  `benchmarks/catboost_same_revision_reversed_numeric_binary_stress_trace_r20_20260606.csv`,
  and
  `benchmarks/catboost_numeric_binary_stress_trace_r20_calibrated_report_20260606.json`.
  Outcome: the raw strict timing gate fails when comparing bbstats v2 against
  itself. In default order the same-code control failed with
  `geomean_fit_ratio=1.0943`; in reversed order it still had two row failures
  despite aggregate `geomean_fit_ratio=0.9878`. The current candidate's
  numeric-binary stress trace passes once timing limits are calibrated to that
  same-revision noise envelope (`passed=true`, zero failures). Treat the
  remaining raw row failures as below the current harness timing floor, not as
  proved product regressions.
- Full current calibrated strict medium gate:
  `benchmarks/catboost_strict_medium_current_20260606.csv`,
  `benchmarks/catboost_same_revision_medium_current_20260606.csv`,
  `benchmarks/catboost_same_revision_medium_current_reversed_20260606.csv`,
  and
  `benchmarks/catboost_strict_medium_current_calibrated_both_report_20260606.json`.
  Outcome: current catboost mode is quality- and iteration-equivalent to
  bbstats v2 across all 100 medium comparisons and is faster in aggregate
  (`geomean_fit_ratio=0.9779`). The uncalibrated gate fails with 36 row-level
  timing failures. Same-revision controls also fail the raw row-min timing
  gate, confirming that raw per-row timing is too strict by itself; however,
  calibrating against both same-code controls still leaves 28 row-level timing
  failures. These survivors are timing-only and do not justify reverting any
  accepted product behavior, but they do block a strict-domination claim.
- Post-audit full current gate:
  `benchmarks/catboost_strict_medium_current_postaudit_20260606.csv`,
  `benchmarks/catboost_strict_medium_current_postaudit_summary_20260606.csv`,
  `benchmarks/catboost_same_revision_medium_postaudit_20260606.csv`,
  `benchmarks/catboost_same_revision_medium_postaudit_reversed_20260606.csv`,
  and
  `benchmarks/catboost_strict_medium_current_postaudit_calibrated_both_report_20260606.json`.
  Outcome: after q50 cleanup and the rejected tree-builder restorations,
  current catboost mode still preserves upstream quality and iterations exactly
  across all 100 paired medium comparisons, but the full speed gate regressed.
  Raw strict checking fails with 63 failures (`geomean_fit_ratio=1.1160`);
  calibration against both fresh same-revision controls still leaves 57 timing
  failures. This is broader than the earlier numeric-binary-only focus:
  categorical, quantile, and scalar rows all contribute. Do not claim strict
  domination yet; the next pass should profile shared full-matrix overhead
  against bbstats v2.
- Post-audit phase focus:
  `benchmarks/catboost_postaudit_phase_focus_20260607.csv`.
  Outcome: a candidate-only verbose-timing probe on representative blockers
  (`categorical_reg`, `categorical_binary`, `quantile_reg_10`, and
  `numeric_binary`) shows the broad regression is mostly tree-build dominated:
  tree build is roughly `69%` to `85%` of candidate phase time for
  categorical/numeric blockers. Quantile q10 also has a large leaf-correction
  component (`29%` stress, `46%` unweighted). Follow-up weighted Quantile
  mask-fallback probes failed, so next code probes should target the
  generalized tree-builder path, not wrapper, validation, prediction,
  benchmark-order, or alpha-specific Quantile leaf-correction explanations.
- Median-quantile unweighted grouped leaf correction:
  `benchmarks/catboost_current_aggregate_slow_focus_20260606.csv`,
  `benchmarks/catboost_current_phase_focus_20260606.csv`, and
  `benchmarks/catboost_q50_unweighted_grouped_probe_r7_20260606.csv`.
  Outcome: the focused aggregate-slower rerun isolated `quantile_reg_50` /
  unweighted as a stable timing-only blocker (`fit_vs_base=1.167`, three q50
  row failures) while q50 stress was already faster. A narrow one-change probe
  using grouped unweighted correction only for `loss="Quantile", alpha=0.5`
  cleared the q50 calibrated gate (`passed=true`, zero failures,
  `geomean_fit_ratio=1.0007`) with identical metrics and iterations. Promote
  this median-only path; keep q10/q90 and MAE on the upstream mask loop unless
  a separate gate proves otherwise.
- Post-q50 aggregate-slower focus:
  `benchmarks/catboost_post_q50_aggregate_focus_20260606.csv` and
  `benchmarks/catboost_post_q50_aggregate_focus_calibrated_report_20260606.json`
  plus
  `benchmarks/catboost_post_q50_aggregate_focus_repeat_summary_20260606.csv`.
  Outcome: q50 is no longer an aggregate blocker (`0.993` unweighted, `0.972`
  stress), and numeric multiclass moved to parity (`0.987` unweighted, `1.001`
  stress). The focus gate still fails on timing, with numeric binary as the
  main aggregate blocker (`1.252` unweighted, `1.145` stress) and wide numeric
  regression stress as the next blocker (`1.158`; wide unweighted `1.074`).
  Repeat-distribution analysis confirms these are stable median-repeat
  slowdowns, not merely bad row minima: numeric binary is slower on all six
  medians (`1.34x` to `1.91x`), and wide numeric regression is slower on all
  six medians (`1.21x` to `1.68x`). The shared target is scalar catboost
  overhead, not q50 leaf correction or multiclass.
- Numeric-binary no-linear probe:
  `benchmarks/catboost_numeric_binary_no_linear_probe_r7_20260606.csv`.
  Outcome: forcing current candidate `linear_leaves=False` roughly halves fit
  time versus upstream on numeric-binary medium rows (`geomean_fit_ratio=0.458`)
  but materially worsens primary log loss on all six seed/weight rows
  (`+0.0175` to `+0.0265`). Reject this as a default or automatic fallback;
  bbstats v2's binary linear-leaf quality improvement must be preserved.
- Linear-leaf no-design-matrix probe:
  `benchmarks/catboost_linear_leaf_no_design_matrix_numeric_binary_r7_20260606.csv`
  and
  `benchmarks/catboost_linear_leaf_no_design_matrix_numeric_binary_r7_report_20260606.json`.
  Outcome: accumulating `_linear_leaf_fit`'s ridge equations directly from a
  per-row scratch vector preserved metrics and iterations, but made the
  numeric-binary focused gate worse (`geomean_fit_ratio=1.236`, all six timing
  rows failed). Reject this kernel rewrite; the current design-matrix temporary
  is faster under Numba for this workload.
- Scalar-blocker phase diagnostic:
  `benchmarks/catboost_scalar_blockers_phase_current_20260606.csv` and
  `benchmarks/catboost_scalar_blockers_phase_current_summary_20260606.csv`.
  Outcome: current numeric-binary and wide-regression blockers are still
  tree-build dominated. Numeric binary spends about `84%` to `90%` of booster
  time in `tree_build`; wide numeric regression spends about `95%` to `97%`.
  Continue testing tree-builder changes; do not prioritize wrapper,
  calibration, validation-prediction, or train-update work.
- Darko v1 in-place prediction surface:
  code comparison shows darko v1 had an `ObliviousTree.add_predict(...)` path
  for training and validation updates. The focused product-code probe is tracked
  in `benchmarks/catboost_add_predict_scalar_blockers_r7_20260607.csv` and
  `benchmarks/catboost_add_predict_scalar_blockers_r7_report_20260607.json`.
  Outcome: metrics and iterations stayed identical, but the gate failed
  (`geomean_fit_ratio=1.0111`, 10 failures). It helped some rows but badly
  regressed categorical regression stress (`mean fit ratio=1.4751`). Reject this
  restoration; product code was reverted.
- Darko v1 in-place leaf-routing surface:
  `benchmarks/catboost_inplace_leaf_update_scalar_blockers_r7_20260606.csv` and
  `benchmarks/catboost_inplace_leaf_update_scalar_blockers_r7_report_20260606.json`.
  Outcome: restoring in-place leaf-id updates inside the current tree builder
  preserved metrics and iterations, but failed the strict scalar-blocker gate
  (`geomean_fit_ratio=1.014`, seven failures). It helped wide numeric regression
  but regressed every numeric-binary row, including stress weights. Reject this
  restoration; keep the current NumPy leaf-routing expression.
- Darko v1 serial histogram surface:
  code comparison shows darko v1 had explicit single-thread row-major histogram
  kernels selected when Numba's thread count was one. A standalone current-tree
  microbench compared that row-major serial shape against the current
  feature-major histogram kernels with `numba.set_num_threads(1)`. Current
  feature-major stayed faster on the scalar-blocker-like shapes: darko-style
  serial was `1.43x` slower on `numeric_binary_medium`, `2.04x` slower on
  `wide_reg_medium`, `1.62x` slower on `numeric_binary_large`, and `1.12x`
  slower even on a low-feature categorical shape. Reject restoring the serial
  histogram lane; it is not the missing catboost-mode speed path.
- Split-scratch probe:
  `benchmarks/catboost_split_scratch_scalar_blockers_r7_20260606.csv` and
  `benchmarks/catboost_split_scratch_scalar_blockers_r7_report_20260606.json`.
  Outcome: restoring a darko-v1-style reusable split scratch inside the current
  bbstats-v2-compatible interleaved tree builder preserved metrics and
  iterations but worsened scalar-blocker timing (`geomean_fit_ratio=1.142`,
  `geomean_boost_ratio=1.133`, eleven timing failures). Reject this restoration;
  the current local-array split kernel is faster in this layout.
- Selected row/feature kernels:
  code inspection plus `tests/test_chimeraboost.py` show these kernels are
  inactive under the strict default matrix (`subsample=1.0`,
  `colsample=1.0`) and behavior-equivalent when non-default sampling activates
  them. Keep them as a behavior-proved darko optimization for sampling modes.

Next:

1. Treat the post-audit full gate as authoritative current status:
   `catboost_strict_medium_current_postaudit_calibrated_both_report_20260606.json`
   still fails with 57 timing-only failures despite exact metric and iteration
   parity. The remaining blocker is now full-matrix timing overhead, not a
   single numeric-binary or q50 issue.
   `catboost_postaudit_phase_focus_20260607.csv` localizes the representative
   current blockers to tree build, with q10 leaf correction as a secondary
   target. The follow-up upstream-vs-candidate tree-phase harness
   `benchmarks/catboost_tree_phase_compare_r5_20260607.csv` narrows this again:
   candidate tree building is faster on the categorical rows, but slower on
   numeric binary and q10. The next pass should target numeric/quantile
   tree-kernel overhead, not a universal tree-builder revert. The subphase run
   `benchmarks/catboost_tree_subphase_numeric_quantile_r5_20260607.csv` narrows
   numeric binary further: histogram fill is `1.40x` to `1.42x` slower and
   split search is `1.53x` to `1.54x` slower, while linear-leaf fitting is near
   parity. Do not spend the next pass on `_linear_leaf_fit`.
2. Run exactly one ablation at a time. Call-shape-only routing and
   benchmark-order bias are already rejected, native-int bin indexing is
   rejected, dtype alone is not explanatory, linear-leaf precompute is
   rejected, dropping binary linear leaves is rejected, linear-leaf
   design-matrix removal is rejected, split-scratch restoration is rejected,
   and full upstream-tree-lane restoration is rejected.
3. Promote only ablations that improve the full calibrated gate or a clearly
   named full-gate aggregate blocker without introducing a new quality,
   semantic, or timing regression.
4. Keep `tree_mode="lightgbm"` work paused until catboost mode is either
   strictly dominating bbstats v2 or the remaining non-domination cases are
   documented as irreducible timing noise under the accepted gate.
