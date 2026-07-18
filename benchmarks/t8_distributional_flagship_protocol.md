# T8 distributional flagship protocol

**Frozen:** 2026-07-17, before the campaign outcome existed. The code commit
containing this file is the implementation/protocol freeze. This is a Tier-E
capability measurement under `benchmarks/SHIPPING_POLICY.md`, not a Tier-D
default-policy nomination.

## Question and claim boundary

The campaign asks how DarkoFit's opt-in Gaussian distributional model and new
held-out split-conformal 90% interval compare with NGBoost Normal, CatBoost
`RMSEWithUncertainty`, and quantile LightGBM under one reproducible,
same-machine nominal-round budget.

The output may support:

- the correctness and practical usefulness of
  `predict_interval(..., calibrate="conformal")`;
- descriptive per-dataset CRPS, Gaussian NLL, marginal coverage, interval
  width, fit time, and prediction time;
- a scoped comparison of the tested package versions and configurations.

It cannot support a default change, universal superiority, conditional
coverage, a tuned-leaderboard claim, or a timing claim outside this machine.
Coverage and width must always be adjacent; no scalar score may hide width.

## Frozen implementation

- Runner: `benchmarks/bench_distributional.py`
- Runner SHA-256:
  `382ba9059fcf430654748c0cc0c15427f42d9be98cf37aa3becd76d19f471d80`
- Analyzer: `benchmarks/analyze_t8_distributional_flagship.py`
- Analyzer SHA-256:
  `c8b52ee6313b7b3406648277aec53661f993ec4f53fe276201588044b31d4c0e`
- The runner warms every requested model family once on a discarded small
  problem. Warmup is excluded from recorded times.
- Every dataset/seed train/test payload is SHA-256 fingerprinted in every
  output row. The analyzer requires all models at a coordinate to agree.

## Frozen environment

- Python 3.13.13
- NumPy 2.4.6
- scikit-learn 1.9.0
- Numba 0.66.0
- NGBoost 0.5.11
- CatBoost 1.2.10
- LightGBM 4.6.0
- Apple M5 Max, 18 physical cores, 128 GiB RAM
- macOS 26.5.2 arm64
- Four model threads; BLAS thread pools limited to one

## Data and coordinates

Five datasets, three seeds, five models: **75 required coordinates**.

| Dataset | Definition |
| --- | --- |
| `synthetic_100k` | 100,000 train / 25,000 test, six numeric features, known heteroscedastic Gaussian noise |
| `synthetic_t3_100k` | same feature/scale mechanism, variance-matched Student-t(3) noise |
| `openml_cpu_act` | OpenML data ID 197, seeded 75/25 split |
| `openml_wine_quality` | OpenML data ID 287, seeded 75/25 split |
| `openml_boston` | OpenML data ID 531, seeded 75/25 split |

Seeds are exactly `0, 1, 2`. Missing numeric OpenML values are filled after
the seeded split using training-fold medians.
OpenML IDs are public and previously used in this repository, so the real-data
cells are development/measurement evidence rather than fresh confirmation.
The two synthetic generators are specified in the frozen runner and do not
constitute an independent lockbox.

## Frozen model configurations

Common nominal budget: 120 rounds, learning rate 0.06, no sample weights.

1. `darkofit_gaussian_es_calibrated`: Gaussian, leaf-wise, 15 leaves,
   `min_child_samples=10`, early stopping, affine scale calibration, 20%
   automatic validation.
2. `darkofit_gaussian_es_conformal`: the same public configuration plus
   `interval_calibration="conformal"`. It trains on the same 80% as lane 1;
   the 20% holdout is deterministically divided into 10% selection/affine
   calibration and 10% untouched conformal calibration.
3. `ngboost`: Normal/LogScore, 120 estimators, learning rate 0.06, full-row and
   full-column sampling, package-default base learner.
4. `catboost_uncertainty`: `RMSEWithUncertainty`, depth 6, 120 iterations,
   learning rate 0.06.
5. `lightgbm_quantile_pair`: two 15-leaf quantile regressors at alpha 0.05 and
   0.95, 120 estimators, learning rate 0.06, `min_child_samples=10`.

The comparators train on all outer-training rows, while the DarkoFit lanes
reserve 20% for selection/calibration. This intentionally does not create a
data-budget advantage for DarkoFit. Nominal rounds and learning rates do not
equalize algorithmic compute; timing is descriptive.

## Metrics and analysis

- Primary interval outputs: empirical marginal 90% coverage and mean width,
  per dataset and model, averaged over seeds.
- Coverage summaries: mean coverage, mean absolute dataset-level deviation
  from 0.90, and worst/best individual cell coverage.
- Width summary: per-dataset width relative to the DarkoFit conformal lane,
  then geometric mean. Raw widths are never averaged across differently scaled
  datasets.
- Gaussian-distribution outputs: NLL and closed-form Gaussian CRPS for
  DarkoFit Gaussian, NGBoost Normal, and CatBoost uncertainty.
- Quantile LightGBM reports only coverage and width. It does not expose a
  Gaussian distribution; midpoint RMSE, NLL, and CRPS remain blank.
- Fit and prediction wall time are descriptive means after warmup.
- The analyzer separately reports the conformal-versus-parametric change in
  absolute coverage error and the width ratio. It does not combine them.

## Failure and decision rules

The analyzer fails closed on a missing/duplicate/non-success coordinate,
non-finite required metric, mismatched data fingerprint, wrong interval-method
label, or empty conformal calibration set. Failed coordinates are not
imputed, dropped, or rerun under changed settings.

Because this is Tier-E, there is no quality threshold for shipping the opt-in
API. Shipping requires the correctness suite, persistence coverage, honest
documentation of limitations, and a complete result. An adverse empirical
result remains publishable and does not trigger post-outcome tuning.

## Exact command

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=4 \
NUMBA_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MKL_NUM_THREADS=1 \
PYTHONPATH=. \
python benchmarks/bench_distributional.py \
  --datasets synthetic_100k synthetic_t3_100k \
    openml_cpu_act openml_wine_quality openml_boston \
  --models darkofit_gaussian_es_calibrated \
    darkofit_gaussian_es_conformal ngboost catboost_uncertainty \
    lightgbm_quantile_pair \
  --seeds 0 1 2 \
  --iterations 120 \
  --early-stop-iterations 120 \
  --early-stopping-rounds auto \
  --validation-fraction 0.2 \
  --learning-rate 0.06 \
  --num-leaves 15 \
  --min-child-samples 10 \
  --threads 4 \
  --weight-modes none \
  --csv benchmarks/t8_distributional_flagship_raw.csv \
  --markdown benchmarks/t8_distributional_flagship_raw.md

PYTHONPATH=. python benchmarks/analyze_t8_distributional_flagship.py
```
