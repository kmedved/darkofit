# WNBA Real-Data Distributional Validation

This is a time-ordered validation on WNBA DARKO game-level metric observations.  The target is source-column `z_observed`, a transformed metric observation scale; weights are `sample_weight` from the observation rows.

## Data

- Source: `/Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/calculated_data/research/observation_covariance_measurement/game_metric_observations.parq`
- Date range: 2009-06-06 to 2026-07-05
- Metrics: fg_pct, fta_100, pace, pf_100, pts_100, tov_100
- Train seasons: 2009-2021
- Validation seasons: 2022-2023
- Test seasons: 2024-2026
- Rows: train 16,866, validation 2,994, test 4,356
- Features: 35 causal/date/context features; `metric_code` is categorical.

## Results

| Model | NLL | CRPS | RMSE mu | 90% cov | std-resid RMS | mean sigma | affine b | sigma range | sigma-|resid| corr | fit s | best iter | config |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| unit_normal_observation_baseline | 51.648 | 6.068 | 10.073 | 0.304 | 10.073 | 1.000 |  | 1.000-1.000 | nan | 0.00 |  |  |
| chimera_rmse_const_sigma | 1.448 | 0.507 | 1.028 | 0.902 | 1.034 | 0.994 |  | 0.994-0.994 | 0.000 | 2.98 | 132 |  |
| chimera_gaussian_raw | 0.432 | 0.394 | 1.015 | 0.881 | 1.074 | 0.593 |  | 0.057-2.003 | 0.687 | 7.71 | 280 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=1.0, rho_l2=1.0` |
| chimera_gaussian_scalar_calibrated | 0.426 | 0.393 | 1.015 | 0.899 | 1.001 | 0.637 |  | 0.061-2.149 | 0.687 | 5.09 | 280 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=1.0, rho_l2=1.0` |
| chimera_gaussian_affine_calibrated | 0.409 | 0.392 | 1.015 | 0.903 | 0.994 | 0.705 | 1.107 | 0.050-2.568 | 0.686 | 4.90 | 280 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=1.0, rho_l2=1.0` |
| chimera_gaussian_affine_tuned | 0.419 | 0.397 | 1.029 | 0.901 | 1.003 | 0.712 | 1.145 | 0.051-2.788 | 0.677 | 5.38 | 359 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=0.75, rho_l2=2.0` |

## Sigma-Binned Calibration

Rows are sorted by predicted sigma and split into equal-count bins. Good observation-noise calibration should keep coverage near 0.90 and standardized-residual RMS near 1 in each bin.

| Model | 90% coverage by sigma bin | std-resid RMS by sigma bin | E[z^2] by sigma bin |
|---|---|---|---|
| unit_normal_observation_baseline | 0.329/0.283/0.307/0.287/0.313 | 10.100/10.352/10.036/9.872/9.997 | 102.002/107.154/100.730/97.453/99.934 |
| chimera_rmse_const_sigma | 0.902/0.888/0.899/0.921/0.900 | 0.974/1.194/1.080/0.909/0.987 | 0.948/1.425/1.167/0.827/0.975 |
| chimera_gaussian_raw | 0.949/0.929/0.896/0.844/0.835 | 0.851/0.916/1.026/1.192/1.194 | 0.725/0.839/1.052/1.421/1.426 |
| chimera_gaussian_scalar_calibrated | 0.961/0.940/0.916/0.862/0.860 | 0.793/0.854/0.956/1.111/1.113 | 0.629/0.729/0.913/1.234/1.239 |
| chimera_gaussian_affine_calibrated | 0.925/0.924/0.896/0.868/0.914 | 0.935/0.932/1.021/1.080/0.962 | 0.875/0.868/1.043/1.167/0.925 |
| chimera_gaussian_affine_tuned | 0.926/0.926/0.900/0.858/0.913 | 0.929/0.947/1.041/1.101/0.958 | 0.863/0.898/1.084/1.212/0.919 |

## Affine Calibration Diagnostics

Diagnostics below use the affine-calibrated Gaussian lane on the held-out test split.

- PIT histogram deciles: 0.070/0.073/0.087/0.095/0.108/0.106/0.108/0.114/0.120/0.119
- Weighted PIT KS distance vs uniform: 0.076
- Lag-1 residual autocorrelation is computed over metric-ordered game-level rows because this source artifact has no player/entity identifier; per-player whiteness remains a downstream DARKO replay gate.

### Per-Metric Sigma Terciles

| Metric | sigma bin | n | 90% cov | std-resid RMS | E[z^2] | mean sigma |
|---|---:|---:|---:|---:|---:|---:|
| fg_pct | 1 | 242 | 0.901 | 0.989 | 0.978 | 0.179 |
| fg_pct | 2 | 242 | 0.927 | 0.945 | 0.894 | 0.182 |
| fg_pct | 3 | 242 | 0.906 | 0.998 | 0.996 | 0.185 |
| fta_100 | 1 | 242 | 0.874 | 1.062 | 1.128 | 0.202 |
| fta_100 | 2 | 242 | 0.896 | 1.016 | 1.032 | 0.222 |
| fta_100 | 3 | 242 | 0.942 | 0.858 | 0.736 | 0.247 |
| pace | 1 | 242 | 0.914 | 0.941 | 0.885 | 0.050 |
| pace | 2 | 242 | 0.951 | 0.900 | 0.810 | 0.051 |
| pace | 3 | 242 | 0.932 | 0.886 | 0.784 | 0.052 |
| pf_100 | 1 | 242 | 0.788 | 1.351 | 1.825 | 0.634 |
| pf_100 | 2 | 242 | 0.877 | 1.042 | 1.085 | 0.736 |
| pf_100 | 3 | 242 | 0.900 | 0.886 | 0.786 | 0.854 |
| pts_100 | 1 | 242 | 0.881 | 1.102 | 1.214 | 1.926 |
| pts_100 | 2 | 242 | 0.930 | 0.926 | 0.858 | 2.126 |
| pts_100 | 3 | 242 | 0.938 | 0.870 | 0.756 | 2.309 |
| tov_100 | 1 | 242 | 0.917 | 0.981 | 0.963 | 0.151 |
| tov_100 | 2 | 242 | 0.906 | 0.977 | 0.955 | 0.161 |
| tov_100 | 3 | 242 | 0.942 | 0.855 | 0.731 | 0.172 |

### Lag-1 Standardized Residual Correlation

| Group | pairs | lag-1 corr |
|---|---:|---:|
| pooled_metric_order | 4350 | -0.008 |
| fg_pct | 725 | -0.058 |
| fta_100 | 725 | -0.005 |
| pace | 725 | -0.044 |
| pf_100 | 725 | 0.024 |
| pts_100 | 725 | -0.042 |
| tov_100 | 725 | -0.047 |

## W3 Gaussian Source/Tuning Sweep

Candidates are selected on the validation fold by worst per-metric sigma-tercile standardized-residual RMS deviation, then mean deviation, then NLL/CRPS. The selected row is scored against the untouched future test seasons in the main results.

| selected | candidate | validation NLL | validation CRPS | 90% cov | std-resid RMS | max metric RMS dev | mean metric RMS dev | failed bins | config |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|  | 1 | 0.395 | 0.380 | 0.905 | 1.000 | 0.245 | 0.091 | 10 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=1.0, rho_l2=1.0` |
|  | 2 | 0.399 | 0.381 | 0.907 | 1.000 | 0.299 | 0.079 | 10 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=1.0, rho_l2=2.0` |
|  | 3 | 0.397 | 0.380 | 0.906 | 1.000 | 0.281 | 0.097 | 12 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=1.0, rho_l2=4.0` |
|  | 4 | 0.410 | 0.385 | 0.904 | 1.000 | 0.331 | 0.099 | 13 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=0.75, rho_l2=1.0` |
| yes | 5 | 0.399 | 0.382 | 0.905 | 1.000 | 0.237 | 0.097 | 12 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=0.75, rho_l2=2.0` |
|  | 6 | 0.401 | 0.382 | 0.903 | 1.000 | 0.279 | 0.099 | 11 | `lr=0.04, leaves=31, min_child=25, l2=auto, rho_lr=0.75, rho_l2=4.0` |
|  | 7 | 0.397 | 0.381 | 0.904 | 1.000 | 0.303 | 0.086 | 10 | `lr=0.04, leaves=31, min_child=25, l2=1.0, rho_lr=1.0, rho_l2=1.0` |
|  | 8 | 0.393 | 0.380 | 0.908 | 1.000 | 0.301 | 0.090 | 12 | `lr=0.04, leaves=31, min_child=25, l2=1.0, rho_lr=1.0, rho_l2=2.0` |
|  | 9 | 0.398 | 0.381 | 0.907 | 1.000 | 0.238 | 0.077 | 10 | `lr=0.04, leaves=31, min_child=25, l2=1.0, rho_lr=1.0, rho_l2=4.0` |
|  | 10 | 0.405 | 0.384 | 0.901 | 1.000 | 0.255 | 0.088 | 9 | `lr=0.04, leaves=31, min_child=25, l2=1.0, rho_lr=0.75, rho_l2=1.0` |
|  | 11 | 0.400 | 0.383 | 0.904 | 1.000 | 0.283 | 0.084 | 12 | `lr=0.04, leaves=31, min_child=25, l2=1.0, rho_lr=0.75, rho_l2=2.0` |
|  | 12 | 0.407 | 0.384 | 0.905 | 1.000 | 0.301 | 0.093 | 11 | `lr=0.04, leaves=31, min_child=25, l2=1.0, rho_lr=0.75, rho_l2=4.0` |

### Selected Tuned Lane Per-Metric Sigma Terciles

| Metric | sigma bin | n | 90% cov | std-resid RMS | E[z^2] | mean sigma |
|---|---:|---:|---:|---:|---:|---:|
| fg_pct | 1 | 242 | 0.927 | 0.955 | 0.912 | 0.178 |
| fg_pct | 2 | 242 | 0.886 | 1.019 | 1.037 | 0.181 |
| fg_pct | 3 | 242 | 0.917 | 0.975 | 0.952 | 0.186 |
| fta_100 | 1 | 242 | 0.875 | 1.121 | 1.257 | 0.197 |
| fta_100 | 2 | 242 | 0.904 | 1.033 | 1.068 | 0.213 |
| fta_100 | 3 | 242 | 0.904 | 0.956 | 0.914 | 0.229 |
| pace | 1 | 242 | 0.930 | 0.908 | 0.824 | 0.051 |
| pace | 2 | 242 | 0.955 | 0.876 | 0.767 | 0.052 |
| pace | 3 | 242 | 0.943 | 0.879 | 0.772 | 0.053 |
| pf_100 | 1 | 242 | 0.788 | 1.320 | 1.742 | 0.644 |
| pf_100 | 2 | 242 | 0.860 | 1.068 | 1.140 | 0.768 |
| pf_100 | 3 | 242 | 0.933 | 0.816 | 0.666 | 0.895 |
| pts_100 | 1 | 242 | 0.858 | 1.159 | 1.344 | 1.878 |
| pts_100 | 2 | 242 | 0.925 | 0.933 | 0.870 | 2.148 |
| pts_100 | 3 | 242 | 0.938 | 0.846 | 0.716 | 2.392 |
| tov_100 | 1 | 242 | 0.891 | 1.013 | 1.026 | 0.153 |
| tov_100 | 2 | 242 | 0.943 | 0.918 | 0.842 | 0.162 |
| tov_100 | 3 | 242 | 0.937 | 0.890 | 0.792 | 0.171 |

## Rolling-Origin Affine Checks

Each origin uses seasons before the origin for training, the origin season for early stopping/calibration, and the next season for testing.

| origin val season | test season | NLL | CRPS | 90% cov | std-resid RMS | sigma-bin RMS |
|---:|---:|---:|---:|---:|---:|---|
| 2021 | 2022 | 0.451 | 0.393 | 0.904 | 0.987 | 0.822/0.904/0.978/1.169/0.940 |
| 2022 | 2023 | 0.382 | 0.376 | 0.924 | 0.926 | 0.917/0.889/0.992/0.936/0.907 |
| 2023 | 2024 | 0.356 | 0.371 | 0.913 | 0.968 | 1.010/1.055/0.833/0.953/0.961 |

## Interpretation

- Calibrated Gaussian verdict: passes this real-data scale calibration check, but sigma still needs downstream Kalman replay validation before use as production observation noise.
- This benchmark checks one-step observation calibration on held-out future seasons; it does not prove Kalman filtering improves when these sigmas are injected as observation variances.
- The unit-normal baseline is included only as a sanity check for the source scale; the constant-sigma RMSE lane is the practical calibration baseline.

## Metadata

```json
{
  "data_path": "/Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/calculated_data/research/observation_covariance_measurement/game_metric_observations.parq",
  "date_max": "2026-07-05",
  "date_min": "2009-06-06",
  "early_stopping_rounds": 40,
  "feature_cols": [
    "metric_code",
    "season",
    "playoffs_fl",
    "season_day",
    "day_sin",
    "day_cos",
    "dow_sin",
    "dow_cos",
    "log_sample_weight",
    "all_prior_weight",
    "all_prior_mean",
    "all_prior_std",
    "season_prior_weight",
    "season_prior_mean",
    "season_prior_std",
    "recent7_w",
    "recent7_mean",
    "recent7_std",
    "recent7_std_to_all",
    "recent7_std_to_season",
    "recent30_w",
    "recent30_mean",
    "recent30_std",
    "recent30_std_to_all",
    "recent30_std_to_season",
    "recent90_w",
    "recent90_mean",
    "recent90_std",
    "recent90_std_to_all",
    "recent90_std_to_season",
    "recent365_w",
    "recent365_mean",
    "recent365_std",
    "recent365_std_to_all",
    "recent365_std_to_season"
  ],
  "iterations": 400,
  "l2_leaf_reg": "auto",
  "lag_diagnostic_group": "metric (source has no player/entity column)",
  "learning_rate": 0.04,
  "metrics": [
    "fg_pct",
    "fta_100",
    "pace",
    "pf_100",
    "pts_100",
    "tov_100"
  ],
  "min_child_samples": 25,
  "n_features": 35,
  "num_leaves": 31,
  "random_state": 0,
  "rho_l2_leaf_reg_multiplier": 1.0,
  "rho_learning_rate_multiplier": 1.0,
  "rolling_origins": [
    2021,
    2022,
    2023
  ],
  "test_rows": 4356,
  "test_seasons": "2024-2026",
  "thread_count": 1,
  "train_rows": 16866,
  "train_seasons": "2009-2021",
  "tune_gaussian": true,
  "val_rows": 2994,
  "val_seasons": "2022-2023"
}
```
