# ChimeraBoost vs LightGBM — speed investigation findings

_Last updated: 2026-05-28. Environment: `darko311` (Python 3.11.8, lightgbm 4.6.0,
numpy 2.2.4, numba 0.63.1). Harness: `benchmarks/bench_vs_lightgbm.py`._

## TL;DR

- **ChimeraBoost is accuracy-competitive with LightGBM** (often better, especially
  regression) at every size tested. The gap is **purely speed**, not quality.
- The speed gap has **two independent causes**, and which one dominates depends on
  dataset size:
  1. **Round count** (the oblivious-tree tax) — CB needs 3.5–8× more boosting rounds
     at matched learning rate. Dominant at 10k–50k; at 500k on numeric data CB does
     not even converge within the 1500-round cap.
  2. **Per-iteration scaling** — CB's per-tree cost grows ~linearly with rows while
     LightGBM's stays roughly flat. CB is ~10–17× *faster* per tree at 10k but
     ~1–4× *slower* per tree at 500k. This is a small-data advantage that inverts at
     scale.
- **Learning rate is not a free speed knob.** Raising the auto-lr from 0.1 trades
  accuracy on regression and numeric classification. Keep the default at **0.1**.
- No engine changes were made. Both structural levers below are evidence-backed but
  **deferred**.

## Trustworthy benchmarking recipe (read this before re-running)

ChimeraBoost fit-times are dominated by one-time **numba JIT compilation** on a cold
run. To get clean timings:

1. **Warm the on-disk numba cache** (`chimeraboost/__pycache__/*.nbi,*.nbc`). It is
   gitignored and self-heals; just run the harness once to populate it.
2. **Use `--repeat >= 2`** for ChimeraBoost. The harness reports `min` over repeats,
   which strips the one-time compile. `_warm_up()` helps but is **not sufficient on
   its own** — a matching-shape warmup still misses some numba specializations
   (observed: a post-warmup first fit at 13.8s vs 1.7s under `--repeat 2`).
3. **Use `--seeds >= 3`** for metric stability.
4. LightGBM has no per-process JIT, so `--repeat 1` is already clean for it.

Treat any single-seed, cold-cache, or `--repeat 1` ChimeraBoost fit-time as
compile-contaminated.

## 2026-07-07 — R1 `_unique_if_at_most` local microbenchmark

Command: one-off `timeit` comparison between the old Python `set` loop and the
two-stage `np.unique` implementation, run in the repo checkout on synthetic
finite float columns.

| case | old | new | result |
|---|---:|---:|---:|
| 200k rows, 16 distinct values, `max_unique=64` | 0.009533s | 0.001258s | 7.6× faster |
| 200k rows, high-cardinality prefix | 0.000003s | 0.000064s | slower, but still ~64µs before quantiles |
| first 4096 rows low-cardinality, later high-cardinality | 0.000186s | 0.000402s | slower until full `np.unique` rejects |

Decision: keep R1. The intended wide low-cardinality path improves materially.
High-cardinality columns pay a small absolute probe cost before falling back to
the existing quantile path; no default or modeling behavior changes.

## 2026-07-07 — R9 benchmark profile provenance

`benchmarks/bench_vs_lightgbm.py` now requires every raw CSV row to carry the
resolved comparison profile and audit knobs: profile, requested/selected
Chimera tree mode, threads, bin budgets, learning rates, l2/lambda,
min-child settings, matched-leaf policy, and leaf budgets.

Use `--profile matched` when comparing implementation speed under aligned
capacity/regularization assumptions. Use `--profile native` when measuring each
library as a user would get it from defaults. Future tables in this file should
label the profile explicitly; old committed LightGBM raw CSV snapshots were
removed because they used the pre-profile schema.

## Learning rate: keep the default at 0.1

A clean 3-seed sweep over lr ∈ {0.10, 0.15, 0.20, 0.30} at medium and large.
Higher lr cuts rounds everywhere (0.1→0.2 ≈ −50% rounds), but the accuracy cost is
**task-dependent**:

| Task family | lr 0.1 → 0.2 effect | Verdict |
|---|---|---|
| Categorical classification | f1 / log-loss flat (often flat to 0.3) | **Free** |
| Numeric classification | log-loss consistently worse; f1 −0.15 to −0.3 pt | Small real cost |
| Regression | monotonic RMSE cost (e.g. `wide_numeric_reg` medium 38.8 → 42.3, +9%) | Clear cost |

The "free 2× speedup from lr 0.2" seen in an earlier contaminated run was a
**categorical-classification-only artifact**. Blanket-raising lr trades away accuracy
on 2 of 3 task families. **Decision: keep auto-lr at 0.1; document lr as a speed knob**
the user can opt into. (0.15 is a defensible classification-only nudge but still taxes
regression ~3–4%.)

Note: `ordered_boosting` is **not** a round-count lever (disabling it does not reduce
rounds). It is a regularization/accuracy knob and the categorical-leakage defense in
`OrderedTargetEncoder` is always-on regardless.

## The speed gap, decomposed

### Lever 1 — round count (oblivious-tree tax)

CB's symmetric (oblivious) trees make only `depth` split decisions per tree (6 at
depth 6) vs LightGBM's ~`num_leaves` (64). At matched lr 0.1 this costs **3.5–8× more
rounds** at medium/large. At 500k on numeric data it is so severe that CB **hits the
1500-iteration cap without converging** (`numeric_binary`, `numeric_multiclass`).

→ Fix: **level-wise (non-oblivious) trees**. Size-universal; it is the *entire* gap at
10k–50k (where CB already wins per-tree).

### Lever 2 — per-iteration scaling (the crossover)

CB's per-tree cost grows ~linearly with rows (an O(rows) histogram scan); LightGBM's
stays roughly flat (it amortizes/parallelizes fixed per-iteration overhead extremely
well). So CB's lightweight-per-tree edge is a **small-data effect that inverts at
scale**:

| ms / iteration | 10k (medium) | 50k (large¹) | 500k (xlarge) |
|---|---|---|---|
| `numeric_binary` — **CB** | 1.6 | 6.2 | **59.0** |
| `numeric_binary` — LGB | 16.0 | 11.1 | 14.8 |
| `numeric_multiclass` — **CB** | 5.4 | 18.7 | **221.9** |
| `numeric_multiclass` — LGB | 78.4 | 54.4 | 70.2 |
| `wide_numeric_reg` — **CB** | 2.3 | 7.3 | **42.8** |
| `wide_numeric_reg` — LGB | 13.6 | 13.3 | 21.3 |

At 500k the per-iteration deficit is **co-equal with or larger than** the round gap.
On `wide_numeric_reg` the round gap is ~1.0× yet CB is still 2× slower per tree — i.e.
**100% of that gap is per-iteration**, and level-wise trees alone would not fix it.

→ Fix: **row-parallel / thread-local histograms**. Scale-specific (500k+); does not
help medium, where CB already wins per-tree. **Mechanism is still a hypothesis**
(histogram threading vs ordered-boosting per-row work vs cache effects) — confirm with
a thread-scaling/profiling pass (`benchmarks/profile_tree_kernels.py --threads 1 2 4 8
--samples 500000`) before building.

## Accuracy is not the problem

CB matches or beats LightGBM on 6/7 datasets at 500k, even where its fit was truncated
at the iteration cap:

- `wide_numeric_reg` RMSE **28.5 vs 39.5** (CB much better)
- All three categorical datasets: CB better
- `numeric_multiclass` f1 0.9495 vs 0.9488 (~par, despite CB being capped)
- `numeric_binary` f1 0.9594 vs 0.9641 (LGB better — but CB hit the 1500 cap and did
  not converge)

## Lever map and deferred next steps

| Lever | Attacks | Best for | Cost / risk | Status |
|---|---|---|---|---|
| Level-wise non-oblivious trees | round count | all sizes; whole gap at 10k–50k | large, invasive, accuracy risk | deferred |
| Row-parallel / thread-local histograms | per-iteration scaling | 500k+ | contained, no accuracy risk | deferred; profile first |
| Vector-valued multiclass trees | multiclass per-iter | multiclass | modest | low priority |
| Histogram subtraction / in-place buffers / TS bins | micro-opt | — | low value | low priority |
| ~~ordered-boosting auto-off~~ | — | — | — | **falsified** (not a round-count lever) |

## Appendix — full per-dataset tables

All at lr 0.1, 3 seeds. `xLGB` = LightGBM_seconds / CB_seconds (>1 ⇒ CB faster
overall). Metric is f1_macro (higher better) for classification, RMSE (lower better)
for regression.

### Medium (10k)  ·  CB `--repeat 2`, LGB `--repeat 2`

| dataset | task | CB fit / iters | LGB fit / iters | xLGB | CB metric | LGB metric |
|---|---|---|---|---|---|---|
| categorical_binary | binary | 0.33s / 310 | 1.01s / 56 | 3.02 | 0.8579 | 0.8584 |
| categorical_multiclass | mc | 0.55s / 183 | 2.55s / 53 | 4.61 | 0.7442 | 0.7231 |
| categorical_reg | reg | 0.26s / 372 | 2.36s / 212 | 9.17 | 2.6668 | 2.5473 |
| friedman_numeric | reg | 0.44s / 419 | 1.84s / 112 | 4.18 | 1.1011 | 1.2151 |
| numeric_binary | binary | 1.64s / 1034 | 2.02s / 126 | 1.23 | 0.9189 | 0.9171 |
| numeric_multiclass | mc | 4.18s / 771 | 7.63s / 97 | 1.83 | 0.8664 | 0.8726 |
| wide_numeric_reg | reg | 1.76s / 766 | 4.02s / 296 | 2.28 | 38.81 | 87.39 |

### Large (50k)  ·  CB `--repeat 1`¹, LGB `--repeat 1`

| dataset | task | CB fit / iters | LGB fit / iters | xLGB | CB metric | LGB metric |
|---|---|---|---|---|---|---|
| categorical_binary | binary | 1.50s / 413 | 1.20s / 93 | 0.80 | 0.8819 | 0.8763 |
| categorical_multiclass | mc | 1.89s / 230 | 3.33s / 66 | 1.76 | 0.7722 | 0.7579 |
| categorical_reg | reg | 0.93s / 437 | 2.86s / 286 | 3.08 | 2.2060 | 2.3425 |
| friedman_numeric | reg | 1.65s / 515 | 1.41s / 105 | 0.85 | 1.0431 | 1.0856 |
| numeric_binary | binary | 9.10s / 1469 | 4.13s / 373 | 0.45 | 0.9457 | 0.9477 |
| numeric_multiclass | mc | 20.62s / 1104 | 12.14s / 223 | 0.59 | 0.9261 | 0.9274 |
| wide_numeric_reg | reg | 7.05s / 968 | 9.16s / 691 | 1.30 | 32.85 | 66.73 |

### Xlarge (500k)  ·  CB `--repeat 2`, LGB `--repeat 1`

| dataset | task | CB fit / iters | LGB fit / iters | xLGB | round gap | per-iter | CB metric | LGB metric |
|---|---|---|---|---|---|---|---|---|
| categorical_binary | binary | 24.5s / 557 | 3.4s / 241 | 0.14 | 2.31× | **3.08×** | 0.8838 | 0.8806 |
| categorical_multiclass | mc | 33.4s / 344 | 6.8s / 119 | 0.20 | 2.89× | **1.69×** | 0.7754 | 0.7714 |
| categorical_reg | reg | 12.5s / 1042 | 4.7s / 409 | 0.37 | 2.55× | 1.05× | 2.0643 | 2.1149 |
| friedman_numeric | reg | 14.5s / 862 | 2.9s / 167 | 0.20 | 5.17× | 0.97× | 1.0214 | 1.0393 |
| numeric_binary² | binary | 88.5s / 1500 | 18.1s / 1227 | 0.21 | 1.22× | **3.99×** | 0.9594 | 0.9641 |
| numeric_multiclass² | mc | 332.9s / 1500 | 56.9s / 812 | 0.17 | 1.85× | **3.16×** | 0.9495 | 0.9488 |
| wide_numeric_reg | reg | 41.9s / 978 | 21.1s / 989 | 0.50 | 0.99× | **2.01×** | 28.55 | 39.48 |

¹ Large CB fit-times were collected at `--repeat 1` (warm disk cache, so mostly clean,
but not min-stripped). Treat the medium (`--repeat 2`) and xlarge (`--repeat 2`)
numbers as the rigorous anchors; large is the crossover region between them.

² `numeric_binary` and `numeric_multiclass` at 500k hit the 1500-iteration cap without
converging, so their `xLGB` and round-gap figures **understate** CB's true cost.
`round gap` = CB iters / LGB iters; `per-iter` = CB ms-per-iter / LGB ms-per-iter
(>1 ⇒ CB slower per tree).

## 2026-07-07 — Stage 3 R3 row-layout resolver decision

R3 did not flip `leafwise_row_layout="auto"` to segmented. The implementation now
routes automatic row-layout selection through `_resolve_leafwise_row_layout`, but
the resolver deliberately returns `prefix` until the full profile-labeled matrix
in `ROADMAP.md` is run and a threshold is chosen from measured evidence.

The resolver also keeps automatic selection on prefix whenever segmented
preconditions fail or a current `not use_segmented_rows` fast lane would be
disabled (positive-Hessian full-feature scoring, row-parallel segment scans, or
fused changed-leaf scoring). This avoids turning binary/lightgbm fast-path fits
into an accidental regression under an `"auto"` performance flag.

Focused acceptance run:

```text
python -m pytest -q tests/test_lane_equivalence.py \
  tests/test_chimeraboost.py::test_leafwise_segmented_row_layout_matches_prefix_layout \
  tests/test_chimeraboost.py::test_leafwise_segmented_row_layout_guard_and_auto_fallback
24 passed
```

No small-n/large-n segmented threshold is recorded yet; segmented remains an
explicit benchmarking opt-in.

## 2026-07-07 — Stage 3 R2 F-order route matrix check

Direct-builder microbenchmark isolating the fixed-column row-routing matrix:
`build_leafwise_tree` with `n=200000`, `p=32`, `bins=64`, `max_leaves=31`,
4 Numba threads, F-order `X_hist_binned` held constant, and only
`X_route_binned` changed between C-order and F-order.

```text
C-route 0.016412,0.020160,0.017861 best 0.016412 mean 0.018144
F-route 0.014641,0.014787,0.014722 best 0.014641 mean 0.014716
speedup_best 1.1210
speedup_mean 1.2329
```

Decision: keep R2 enabled. Focused lane tests prove C-order and F-order route
matrices remain bit-identical for the direct builders covered by
`tests/test_lane_equivalence.py`.

## 2026-07-07 — Stage 4 R4 exact MVS / weighted-GOSS solve

Replaced the fixed 48-iteration vectorized bisection in `_mvs_probabilities`
and `_weighted_goss_probabilities` with sort + prefix-sum piecewise-linear
solves. This is not a bit-identical modeling change: probabilities can move at
machine precision, which can flip Bernoulli draws.

Focused acceptance:

```text
python -m pytest -q \
  tests/test_chimeraboost.py::test_exact_mvs_probabilities_match_bisection_reference \
  tests/test_chimeraboost.py::test_exact_weighted_goss_probabilities_match_bisection_reference \
  tests/test_chimeraboost.py::test_mvs_realized_sample_count_matches_probability_mass \
  tests/test_chimeraboost.py::test_weighted_goss_realized_sample_mass_matches_probability_mass \
  tests/test_chimeraboost.py::test_weighted_goss_uniform_mass_fast_path_avoids_full_sort \
  tests/test_chimeraboost.py::test_weighted_goss_nonuniform_top_mass_avoids_full_score_sort \
  tests/test_chimeraboost.py::test_weighted_goss_nonuniform_final_scaling_and_multiclass_shared_rows \
  tests/test_chimeraboost.py::test_weighted_goss_empty_other_draw_does_not_force_biased_row \
  tests/test_chimeraboost.py::test_multiclass_mvs_uses_shared_row_sample_per_round
9 passed
```

Sampler end-to-end timing, same process, old bisection monkeypatched back in
for comparison, synthetic lognormal Hessian masses, `repeat=3`:

| n | MVS old best | MVS exact best | weighted-GOSS old best | weighted-GOSS exact best |
|---:|---:|---:|---:|---:|
| 10k | 0.000496s | 0.000465s | 0.000979s | 0.000782s |
| 200k | 0.011976s | 0.009681s | 0.017893s | 0.016017s |
| 500k | 0.031337s | 0.023541s | 0.045002s | 0.032435s |

Paired-seed model parity against old bisection methods monkeypatched on
`_BaseBooster`:

| config | prediction max abs diff | old metric | new metric |
|---|---:|---:|---:|
| MVS regression, diabetes, 40 iters | 1.71e-13 | RMSE 36.96661257369858 | RMSE 36.966612573698576 |
| weighted-GOSS regression, diabetes, 40 iters | 2.27e-13 | RMSE 37.7715373814183 | RMSE 37.7715373814183 |
| MVS binary, breast-cancer, 30 iters | 1.04e-1 | logloss 0.04854724505928547 | logloss 0.0512779898248578 |

Decision: keep the exact solve for the opt-in MVS/weighted-GOSS samplers. The
uniform-mass weighted-GOSS fast path remains untouched.

## 2026-07-07 — Stage 5 R7 shared preprocessing cache

Tree-mode auto and learning-rate probe fits now share a private preprocessing
cache across candidate/probe core boosters. The key is conservative: booster
class/loss, concrete preprocessor config, categorical feature set, train/eval
content signatures, target signatures, normalized train/eval sample-weight
signatures, and eval split content. Refit-on-full-data models do not receive the
selection/probe cache.

Focused acceptance:

```text
python -m pytest -q \
  tests/test_chimeraboost.py::test_preprocessing_cache_reduces_auto_probe_fit_transform_count \
  tests/test_chimeraboost.py::test_preprocessing_cache_key_separates_data_targets_weights_and_eval \
  tests/test_chimeraboost.py::test_preprocessing_cache_does_not_share_scalar_and_multiclass \
  tests/test_chimeraboost.py::test_multiclass_preprocessor_receives_class_major_target_views
4 passed
```

The categorical `tree_mode="auto"` + learning-rate-probe path now performs two
training `FeaturePreprocessor.fit_transform` calls under the monkeypatched
counter: one ordered target-stat prep for CatBoost, and one shared K-fold + raw
category-code prep for LightGBM/hybrid.

## 2026-07-07 — Stage 6 R5a opt-in float32 histogram streams

Added explicit `histogram_dtype`, defaulting to `"float64"`. The `"float32"`
mode is scoped to scalar `GradientBoosting`: losses, fit-loop grad/hess
buffers, bootstrap, samplers, and final leaf-value computation remain float64;
only the per-tree histogram builder streams are cast into reused float32
buffers. Multiclass rejects `"float32"` until the R6 shared-vector layout lands.

Focused acceptance:

```text
python -m pytest tests/test_chimeraboost.py -q -k \
  "histogram_dtype or float32_histogram_streams"
6 passed, 283 deselected, 1 warning
```

Full-suite acceptance:

```text
python -m pytest -q
410 passed, 1 warning
```

The focused tests cover dtype validation including `np.float32`, fixed-seed
sampler row-set parity between `"float64"` and `"float32"`, well-separated
structural split parity, same-seed/thread determinism, real-data metric parity,
and save/load persistence. The default-regret/default-flip gate remains pending
because this patch does not change the default policy.

## 2026-07-07 — Stage 6 R6 shared-vector multiclass class-minor layout

Shared-vector multiclass now stores histogram buffers as class-minor
`(feature, leaf, bin, class)` arrays and refreshes reused row-major
`(n_samples, n_classes)` grad/hess copies once per boosting round. The
per-class fused-root path intentionally stays on the old class-major
`(class, feature, leaf, bin)` buffers and old kernel, so per-class behavior is
not coupled to the shared-vector layout change.

Focused acceptance:

```text
python -m pytest tests/test_chimeraboost.py tests/test_lane_equivalence.py -q -k \
  "class_minor_refill or (multiclass and (histogram or class_minor or shared or leafwise or fused_root or route_binned))"
19 passed, 293 deselected, 1 warning
```

Full-suite and A/B acceptance:

```text
python -m pytest -q
413 passed, 1 warning

python benchmarks/ab_compare.py . .
ab_compare clean: 17 cases bit-identical
```

The direct kernel tests compare class-minor histogram fills against the legacy
class-major reference after transposition, pin class-minor refill/subtract
against two-step refill, and check both non-noisy and noisy shared split
scorers. The A/B suite includes both per-class LightGBM multiclass and
shared-vector LightGBM multiclass cases.

Kernel phase microbenchmark, 30k rows, 24 features, 64 bins, 7 leaves,
4 Numba threads, best of 5 after warm-up:

| K | build old best | build class-minor best | build speedup | refill old best | refill class-minor best | refill speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.000650s | 0.000619s | 1.050x | 0.000143s | 0.000149s | 0.959x |
| 10 | 0.002020s | 0.001125s | 1.795x | 0.000276s | 0.000182s | 1.515x |

Decision: keep the class-minor shared-vector layout. The K=3 refill path is
effectively a wash in this microbenchmark, while the K=10 build/refill phases
show the intended class-adjacent write benefit.

## 2026-07-07 — Stage 6 R5b opt-in uint32 leaf-id streams

Added explicit `leaf_dtype`, defaulting to `"int64"`. The `"uint32"` option is
opt-in and changes only the per-row leaf-id arrays used by scalar and
shared-vector multiclass builders; tree topology arrays, row-order structures,
split arrays, and serialized tree structures remain signed `int64`.

Focused acceptance:

```text
python -m pytest tests/test_chimeraboost.py -q -k "uint32_leaf_dtype"
6 passed, 292 deselected, 1 warning
```

Full-suite and default-path A/B acceptance:

```text
python -m pytest -q
419 passed, 1 warning

python benchmarks/ab_compare.py . .
ab_compare clean: 17 cases bit-identical
```

The focused tests compare direct int64 and uint32 training-state leaves across
oblivious, levelwise, leafwise, hybrid, and shared-vector multiclass builders;
pin `np.bincount` compatibility on uint32 leaves; exercise ordered scalar
training updates; and verify save/load persistence of the opt-in setting.
The default remains `"int64"`, so default-flip regret and large-n gates remain
pending.

## 2026-07-07 — Stage 7 R10 opt-in categorical encoding upgrades

Added two opt-in categorical modeling controls:

- `ts_permutations`, default `1`, averages ordered target statistics over
  multiple independent permutations when set above 1. The P=1 path still calls
  the original single-permutation kernels directly to preserve the exact seed
  sequence and output.
- `target_ordered_cat_codes`, default `"off"`, with explicit
  `"leaky_full"` opt-in for full-target smoothed raw-code ordering in
  LightGBM/hybrid scalar raw-code blocks. The remap is applied only to the
  raw-code block; target-stat encoders continue to consume the original
  factorized codes.

Focused acceptance:

```text
python -m pytest tests/test_chimeraboost.py -q -k \
  "target_ordered or ts_permutations or multi_permutation_ordered_statistics or load_legacy_v1_missing_category_archive"
13 passed, 297 deselected, 1 warning
```

Full-suite and default-path A/B acceptance:

```text
python -m pytest -q
431 passed, 1 warning

python benchmarks/ab_compare.py . .
ab_compare clean: 17 cases bit-identical
```

The tests cover P=1 compatibility, own-label exclusion for P>1, early-row
variance shrinkage, save/load and refit-param persistence, deterministic
target-mean tie ordering by category code, CatBoost-mode non-effect,
scalar-only enforcement for raw-code remaps, v3 archive round-trips with
unseen/missing categories, pandas/dict lookup parity, and corrupt/mis-versioned
remap payload rejection.

Decision: keep both features opt-in. No default flip occurs until categorical
regret/quality reports and LightGBM/hybrid benchmark gates show a durable win.
