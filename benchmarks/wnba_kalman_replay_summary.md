# WNBA Kalman Replay

Scalar per-metric random-walk Kalman replay on the WNBA DARKO game-level metric observation artifact. The Chimera lane injects `predict_variance()` as row-level `R_t`; the incumbent lane uses `sigma2 / sample_weight` with validation-tuned `sigma2` scale.

- Data: `/Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/calculated_data/research/observation_covariance_measurement/game_metric_observations.parq`
- Train through season: 2022
- Validation seasons: 2023-2023
- Test seasons: 2024, 2025, 2026
- Chimera best iteration: 286
- Per-metric calibration groups: 6
- Runtime seconds: 61.25

## Overall

| model | season | n | NLL | RMSE | 90% cov | NIS mean | z RMS | lag1 z corr | mean R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chimera_variance | ALL | 4356 | 0.186 | 0.552 | 0.910 | 0.922 | 0.935 | -0.043 | 0.735 |
| chimera_variance | 2024 | 1572 | 0.159 | 0.533 | 0.931 | 0.827 | 0.883 | -0.007 | 0.749 |
| chimera_variance | 2025 | 1860 | 0.189 | 0.542 | 0.901 | 0.955 | 0.951 | -0.068 | 0.750 |
| chimera_variance | 2026 | 924 | 0.223 | 0.599 | 0.895 | 1.015 | 0.975 | -0.051 | 0.680 |
| incumbent_weight_heuristic | ALL | 4356 | 0.117 | 0.552 | 0.889 | 1.088 | 1.042 | -0.047 | 0.729 |
| incumbent_weight_heuristic | 2024 | 1572 | 0.080 | 0.533 | 0.907 | 1.017 | 1.007 | -0.013 | 0.727 |
| incumbent_weight_heuristic | 2025 | 1860 | 0.128 | 0.542 | 0.882 | 1.104 | 1.048 | -0.070 | 0.736 |
| incumbent_weight_heuristic | 2026 | 924 | 0.158 | 0.599 | 0.873 | 1.179 | 1.081 | -0.060 | 0.720 |

## Season Wins

| season | Chimera lower NLL | Chimera lower RMSE | Chimera NIS closer to 1 |
|---:|---:|---:|---:|
| 2024 | False | True | False |
| 2025 | False | True | True |
| 2026 | False | True | True |

## Interpretation

- Verdict: does not clear the production replacement gate: Chimera variance wins NLL in 0/3 seasons, RMSE in 3/3 seasons, and NIS closeness in 2/3 seasons. Treat the variance as calibration-useful, but keep the incumbent R fallback for production.
- This is still a game-metric observation replay, not a mutation of the production player DARKO filter. A production rollout should wire the same row-level `R_t` contract into that pipeline and retain an automatic incumbent fallback.

## Metric Details

| model | metric | season | n | NLL | RMSE | 90% cov | NIS mean | mean R | q | r scale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chimera_variance | fg_pct | ALL | 726 | -0.299 | 0.179 | 0.883 | 1.105 | 0.029 | 1e-06 |  |
| incumbent_weight_heuristic | fg_pct | ALL | 726 | -0.299 | 0.179 | 0.877 | 1.136 | 0.028 | 1e-06 | 0.825 |
| chimera_variance | fta_100 | ALL | 726 | -0.151 | 0.206 | 0.913 | 0.907 | 0.047 | 1.81161e-05 |  |
| incumbent_weight_heuristic | fta_100 | ALL | 726 | -0.158 | 0.206 | 0.904 | 0.990 | 0.042 | 1.81161e-05 | 0.825 |
| chimera_variance | pace | ALL | 726 | -1.270 | 0.047 | 0.999 | 0.216 | 0.010 | 1e-06 |  |
| incumbent_weight_heuristic | pace | ALL | 726 | -1.646 | 0.047 | 0.888 | 1.107 | 0.002 | 1e-06 | 0.562 |
| chimera_variance | pf_100 | ALL | 726 | 1.205 | 0.779 | 0.860 | 1.255 | 0.493 | 0.00139689 |  |
| incumbent_weight_heuristic | pf_100 | ALL | 726 | 1.185 | 0.780 | 0.854 | 1.264 | 0.454 | 0.0022638 | 0.681 |
| chimera_variance | pts_100 | ALL | 726 | 2.091 | 1.949 | 0.902 | 1.008 | 3.807 | 7.71075e-05 |  |
| incumbent_weight_heuristic | pts_100 | ALL | 726 | 2.088 | 1.950 | 0.906 | 0.993 | 3.826 | 7.71075e-05 | 1.000 |
| chimera_variance | tov_100 | ALL | 726 | -0.461 | 0.152 | 0.906 | 1.038 | 0.022 | 1e-06 |  |
| incumbent_weight_heuristic | tov_100 | ALL | 726 | -0.467 | 0.152 | 0.906 | 1.040 | 0.022 | 1e-06 | 0.825 |

## Metadata

```json
{
  "args": {
    "data": "/Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/calculated_data/research/observation_covariance_measurement/game_metric_observations.parq",
    "early_stopping_rounds": 40,
    "iterations": 600,
    "l2_leaf_reg": "auto",
    "learning_rate": 0.04,
    "min_child_samples": 25,
    "num_leaves": 31,
    "output_csv": "/Users/kmedved/Code/GitHub/chimeraboost/benchmarks/wnba_kalman_replay.csv",
    "output_summary": "/Users/kmedved/Code/GitHub/chimeraboost/benchmarks/wnba_kalman_replay_summary.md",
    "r_ceil": 9.0,
    "r_floor": 0.01,
    "random_state": 42,
    "thread_count": 1,
    "train_through_season": 2022,
    "val_start_season": 2023,
    "val_through_season": 2023
  },
  "details": {
    "best_iteration": 286,
    "group_count": 6,
    "sigma_affine_b": 1.0873956451721396
  },
  "season_wins": [
    {
      "nis_closer": false,
      "nll_win": false,
      "rmse_win": true,
      "season": "2024"
    },
    {
      "nis_closer": true,
      "nll_win": false,
      "rmse_win": true,
      "season": "2025"
    },
    {
      "nis_closer": true,
      "nll_win": false,
      "rmse_win": true,
      "season": "2026"
    }
  ]
}
```
