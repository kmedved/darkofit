# Distributional Standardization Check

Post-0.7 target-standardization closeout run on 2026-07-08. This reruns the
public synthetic calibrated DarkoFit Gaussian lane only; the full external
competitor matrix remains in `benchmarks/distributional_summary.md`.

```bash
python benchmarks/bench_distributional.py \
  --datasets synthetic_100k synthetic_500k \
  --models darkofit_gaussian_es_calibrated \
  --seeds 0 1 2 \
  --iterations 80 \
  --early-stop-iterations 400 \
  --early-stopping-rounds auto \
  --validation-fraction 0.1 \
  --learning-rate 0.06 \
  --num-leaves 31 \
  --threads 8 \
  --csv benchmarks/distributional_standardization_check.csv \
  --markdown benchmarks/distributional_standardization_check.md
```

## Averages

| dataset | model | fit_s | nll | crps | cov90 | width90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| synthetic_100k | darkofit_gaussian_es_calibrated | 5.809 | 0.99006 | 0.39062 | 0.899 | 2.277 |
| synthetic_500k | darkofit_gaussian_es_calibrated | 15.416 | 0.98260 | 0.38877 | 0.899 | 2.267 |

## Per-Seed Rows

| dataset | weight_mode | model | seed | status | fit_s | rmse_mu | nll | crps | cov90 | width90 | cov90_by_sigma | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_100k | none | darkofit_gaussian_es_calibrated | 0 | ok | 6.128 | 0.73527 | 0.99543 | 0.39282 | 0.900 | 2.299 | 0.902/0.899/0.902/0.897/0.900 |  |
| synthetic_100k | none | darkofit_gaussian_es_calibrated | 1 | ok | 5.102 | 0.72801 | 0.98567 | 0.38912 | 0.900 | 2.285 | 0.897/0.900/0.899/0.907/0.899 |  |
| synthetic_100k | none | darkofit_gaussian_es_calibrated | 2 | ok | 6.197 | 0.72815 | 0.98909 | 0.38993 | 0.896 | 2.247 | 0.892/0.891/0.898/0.897/0.900 |  |
| synthetic_500k | none | darkofit_gaussian_es_calibrated | 0 | ok | 15.688 | 0.72594 | 0.98043 | 0.38793 | 0.898 | 2.253 | 0.899/0.902/0.894/0.894/0.901 |  |
| synthetic_500k | none | darkofit_gaussian_es_calibrated | 1 | ok | 15.025 | 0.72694 | 0.98155 | 0.38838 | 0.900 | 2.270 | 0.901/0.897/0.896/0.901/0.903 |  |
| synthetic_500k | none | darkofit_gaussian_es_calibrated | 2 | ok | 15.534 | 0.73029 | 0.98584 | 0.38998 | 0.900 | 2.277 | 0.902/0.897/0.895/0.900/0.903 |  |
