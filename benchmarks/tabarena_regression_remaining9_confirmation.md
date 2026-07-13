# TabArena remaining-nine regression confirmation

_Protocol frozen on 2026-07-12 before candidate results were generated. The
source under test is DarkoFit commit `224bd46`. Results are pending._

## Objective and frozen candidate

This campaign tests whether the candidate selected on Airfoil, Diamonds,
Physiochemical Protein, and Superconductivity generalizes to the other nine
datasets in the original 13-dataset regression panel.

The candidate is immutable for this confirmation:

```text
l2_leaf_reg=1.0
max_bins=128
learning_rate=0.1
ts_permutations=1
```

The control is the corrected DarkoFit 0.9.0 default: `l2_leaf_reg=3`,
`max_bins=254`, automatic learning rate, and `ts_permutations=1`. All other
settings are shared: CatBoost/symmetric trees, depth 6, 1,000-round cap,
validation early stopping, eight AutoGluon bag folds, fold-wise model seeds,
and sequential-local fold fitting.

No per-dataset tuning, result-dependent configuration change, or target-stat
permutation experiment is permitted. If the candidate changes after any
confirmation result is inspected, the entire panel becomes exploratory and a
new external confirmation is required.

## Bounded current-main performance preflight

Before the outer-split campaign, current `main` must pass three steady-state
regression alarms. The preflight is not a policy-selection stage and cannot
change the frozen candidate.

1. Flattened prediction compares the fused contiguous ensemble with the exact
   per-tree loop on one deterministic numeric CatBoost-mode regressor. After
   warmup, seven alternating-order repeats must be bitwise equal, the router
   must select the flat path, median speedup must be at least 1.25x, and timing
   IQR/median must not exceed 15% after at most one rerun.
2. The block-binning preprocessing lane compares `Binner.fit_transform_blocks` with the old
   `np.hstack` plus `fit_transform` reference on 250,000 rows and 40 float64
   columns. Borders and binned outputs must be bitwise equal. The block path
   may not be more than 5% slower, and the avoided temporary allocation is
   recorded; a speedup below 1.10x is informational rather than a blocker.
3. Representative training compares 50-round four-class fits with
   `eval_train_loss=False` and `True` on the same 100,000-row synthetic case.
   Seven alternating-order repeats time `.fit()` only; predicted probabilities
   are compared outside the timed region. They must be bitwise equal, the
   default lane must be at least 1.05x faster, and timing IQR/median must not
   exceed 15% after at most one rerun. The existing phase profiler additionally
   must complete 50 finite rounds without timer-accounting inconsistencies.

All raw repeats, environment metadata, thresholds, parity results, and gate
decisions are retained under `.cache/perf-preflight-remaining9/`.

## Frozen dataset and split matrix

The split set is the full curated intersection between current TabArena task
metadata and non-imputed registered `CHIMERA (default)` results. TabArena folds
flatten as `3 * repeat + fold`.

| Dataset | Task ID | Adapter-visible categoricals | Matched splits | Two-config jobs |
| --- | ---: | --- | ---: | ---: |
| Another Dataset on used Fiat 500 | 363615 | `model` | 30 (`r0f0`-`r9f2`) | 60 |
| Concrete compressive strength | 363625 | None | 30 (`r0f0`-`r9f2`) | 60 |
| Food Delivery Time | 363672 | `Delivery_person_ID`, `Type_of_order`, `Type_of_vehicle` | 9 (`r0f0`-`r2f2`) | 18 |
| Healthcare insurance expenses | 363675 | `sex`, `smoker`, `region` | 30 (`r0f0`-`r9f2`) | 60 |
| Houses | 363678 | None | 9 (`r0f0`-`r2f2`) | 18 |
| Miami housing | 363686 | `avno60plus` | 9 (`r0f0`-`r2f2`) | 18 |
| QSAR-TID-11 | 363697 | None | 9 (`r0f0`-`r2f2`) | 18 |
| QSAR fish toxicity | 363698 | None | 30 (`r0f0`-`r9f2`) | 60 |
| Wine quality | 363708 | `wine_color` | 9 (`r0f0`-`r2f2`) | 18 |
| **Total** | | | **165 dataset-splits** | **330 outer jobs** |

Every outer job fits eight child models, for 2,640 underlying fits. The earlier
rough estimate of 162 jobs assumed nine splits on every dataset and would have
discarded 21 valid splits from each of four tasks.

## Primary estimand and quality gates

For each dataset, compute the geometric mean of paired candidate/default RMSE
ratios over all its splits. The primary aggregate is the geometric mean of the
nine dataset-level ratios, giving every dataset equal weight regardless of
sample count or number of available repeats.

The candidate advances only if all gates pass:

1. All 330 jobs complete successfully with zero failure or imputation.
2. Equal-dataset test RMSE improves by at least 0.5%.
3. No dataset-level geometric-mean test RMSE regresses by more than 0.5%.
4. No individual split regresses by more than 2%.
5. The equal-dataset aggregate improves in each of repeats 0, 1, and 2, which
   are shared by all nine tasks.
6. Each three-repeat task improves in at least two repeat means; each ten-repeat
   task improves in at least seven repeat means.
7. Validation RMSE is reported as a diagnostic and must not reveal a broad
   direction reversal hidden by test aggregation.
8. Paired geometric-mean training time may not regress by more than 20%, batch
   inference by more than 10%, or observed peak CPU memory by more than 10%.

Registered ChimeraBoost quality is compared on the exact same split coordinates
but does not enter candidate selection. Its historical timing remains excluded
from same-machine claims.

## Conditional same-machine performance campaign

Only if the quality gates pass, compare three product configurations on the
same machine:

- corrected DarkoFit default;
- frozen DarkoFit candidate;
- ChimeraBoost 0.14.1 default from local clean commit `07995af`.

The fixed performance panel uses `r0f0`, `r1f1`, and `r2f2` for all nine
datasets: 81 outer jobs and 648 bagged child fits. It uses the same TabArena CPU
budget, explicit validation data, eight bag folds, fold-wise seeds,
sequential-local fitting, and 3,600-second per-model cap. This is a
product-default comparison, not hyperparameter parity; differences in round
caps, target-stat permutations, linear-leaf selection, and automatic policies
must be disclosed.

Report paired, equal-dataset summaries for:

- wall-clock training time;
- directly instrumented preprocessing `fit_transform` time, including every
  lane ChimeraBoost evaluates during linear-leaf selection;
- batch inference time;
- process peak RSS as a no-regression diagnostic;
- AutoGluon all-children and low-memory retained-model sizes;
- RMSE and validation RMSE so speed cannot conceal a quality change.

Numba warmup is performed and recorded separately from steady-state timed fits.
Preprocessing is measured by benchmark-only instrumentation around each
package's own preprocessor; it must not be inferred by subtracting internal fit
timers from wall time.

## Completion

The final report must retain a tidy per-split CSV, exact source/environment
provenance, failures and imputation status, raw timing artifacts, gate outcomes,
matched ChimeraBoost quality, and limitations. Public defaults change only if
this gate and the later classification, weighted-data, other-loss, and
tree-mode checks support the same narrowly scoped policy.
