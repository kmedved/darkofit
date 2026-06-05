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
