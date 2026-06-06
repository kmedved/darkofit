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
| Best-of-both-worlds branch | `codex/best-of-both-worlds@9e13d694b0d57918d312926da286909e384fb477` | Upstream v2 trunk plus benchmark-gated darko patches. |

Patch counts from the fork point:

- bbstats v2: 86 commits, 81 non-merge commits.
- darko v1: 14 changed files, mostly benchmark harnesses and kernel speed work.
- current branch over bbstats v2: 47 changed files, mostly benchmark gates,
  docs, and opt-in tree/kernel paths.

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
- reusable split scratch in the tree builder;
- class-major multiclass buffers;
- weighted target stats as an opt-in product improvement;
- timing diagnostics, strict benchmark adapters, and strict-domination checker;
- opt-in `tree_mode="lightgbm"` / `tree_mode="levelwise"` research path.

These stay in catboost mode only while the strict gate continues to pass.

### Workload-Dependent Or Unproven Areas

Decision: `BENCHMARK-TOGGLE`

Do not change the catboost default yet for these areas:

- plain upstream-style tree helper versus specialized kernels;
- compact bins versus upstream-style `uint16` bins on suspect rows;
- categorical factorization versus upstream's pandas-vectorized categorical
  encoding;
- selected-row/selected-feature kernels outside subsample/colsample settings;
- learning-rate or ordered-boosting default changes.

These need one-change ablations against stable strict-gate failures.

### Docs, Charts, CI, And Release Metadata

Decision: `DOCS/RELEASE/ADVISORY`

Keep useful upstream docs/CI/release metadata when it does not conflict with
the branch's benchmark claims. Benchmark images and README performance claims
are treated as upstream historical context; current branch claims must point to
tracked CSVs and the strict-domination checker.

## Complete Upstream v2 Commit Inventory

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
| `20ad819` | 2026-06-01 | Vectorize categorical encoding via pandas (~15% faster cat fits, bit-identical) | `BENCHMARK-TOGGLE` |
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
  `uint16` as a possible adaptive toggle only.

Next:

1. For stable blockers, run exactly one ablation at a time:
   constant-Hessian path, categorical encoding path, class-major multiclass,
   selected row/feature kernels, validation semantics lane, or an adaptive bin
   dtype policy if the row pattern remains stable.
2. Promote only ablations that improve the blocker without introducing a new
   quality, semantic, or timing regression.
3. Keep `tree_mode="lightgbm"` work paused until catboost mode is either
   strictly dominating bbstats v2 or the remaining non-domination cases are
   documented as irreducible timing noise under the accepted gate.
