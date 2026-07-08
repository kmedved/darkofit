# WNBA Kalman Replay

Scalar per-metric random-walk Kalman replay on the WNBA DARKO game-level metric observation artifact. The Chimera lane injects `predict_variance()` as row-level `R_t`; the incumbent lane uses `sigma2 / sample_weight` with validation-tuned `sigma2` scale.

- Data: `/Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/calculated_data/research/observation_covariance_measurement/game_metric_observations.parq`
- Train through season: 2021
- Validation seasons: 2022-2023
- Test seasons: 2024, 2025, 2026
- Distributional loss: StudentT
- Model best iteration: 202
- Per-metric calibration groups: 6
- R floor: 0.25 x each metric's incumbent tuned train mean R
- Gate noise band: paired row bootstrap, 2,000 resamples, 95% CI; differences whose interval crosses zero are ties.
- Runtime seconds: 67.65

## Overall

| model | season | n | NLL | RMSE | 90% cov | NIS mean | z RMS | lag1 z corr | mean R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| incumbent_weight_heuristic | ALL | 4356 | 0.114 | 0.864 | 0.904 | 0.982 | 0.991 | -0.054 | 0.742 |
| incumbent_weight_heuristic | 2024 | 1572 | 0.081 | 0.848 | 0.923 | 0.918 | 0.958 | -0.020 | 0.739 |
| incumbent_weight_heuristic | 2025 | 1860 | 0.124 | 0.836 | 0.897 | 0.998 | 0.999 | -0.077 | 0.748 |
| incumbent_weight_heuristic | 2026 | 924 | 0.150 | 0.945 | 0.886 | 1.061 | 1.030 | -0.063 | 0.733 |
| studentt30_incumbent_blend | ALL | 4356 | 0.114 | 0.864 | 0.904 | 0.994 | 0.997 | -0.055 | 0.741 |
| studentt30_incumbent_blend | 2024 | 1572 | 0.080 | 0.848 | 0.924 | 0.928 | 0.963 | -0.019 | 0.739 |
| studentt30_incumbent_blend | 2025 | 1860 | 0.124 | 0.836 | 0.896 | 1.013 | 1.007 | -0.078 | 0.748 |
| studentt30_incumbent_blend | 2026 | 924 | 0.150 | 0.945 | 0.886 | 1.069 | 1.034 | -0.063 | 0.733 |
| studentt30_replay_scaled | ALL | 4356 | 0.123 | 0.875 | 0.901 | 1.013 | 1.006 | -0.048 | 0.764 |
| studentt30_replay_scaled | 2024 | 1572 | 0.082 | 0.845 | 0.920 | 0.923 | 0.961 | -0.017 | 0.726 |
| studentt30_replay_scaled | 2025 | 1860 | 0.133 | 0.837 | 0.894 | 1.030 | 1.015 | -0.076 | 0.800 |
| studentt30_replay_scaled | 2026 | 924 | 0.173 | 0.992 | 0.883 | 1.131 | 1.064 | -0.061 | 0.756 |
| studentt30_variance | ALL | 4356 | 0.137 | 0.875 | 0.919 | 0.890 | 0.944 | -0.045 | 0.774 |
| studentt30_variance | 2024 | 1572 | 0.107 | 0.845 | 0.933 | 0.822 | 0.906 | -0.013 | 0.737 |
| studentt30_variance | 2025 | 1860 | 0.142 | 0.837 | 0.917 | 0.897 | 0.947 | -0.074 | 0.810 |
| studentt30_variance | 2026 | 924 | 0.180 | 0.992 | 0.900 | 0.993 | 0.996 | -0.057 | 0.765 |

## Season Results vs Incumbent

| model | season | NLL result | RMSE result | NIS result |
|---|---:|---:|---:|---:|
| studentt30_incumbent_blend | 2024 | tie | tie | win |
| studentt30_incumbent_blend | 2025 | tie | tie | tie |
| studentt30_incumbent_blend | 2026 | tie | tie | tie |
| studentt30_replay_scaled | 2024 | tie | tie | tie |
| studentt30_replay_scaled | 2025 | loss | tie | tie |
| studentt30_replay_scaled | 2026 | loss | loss | tie |
| studentt30_variance | 2024 | loss | tie | loss |
| studentt30_variance | 2025 | loss | tie | tie |
| studentt30_variance | 2026 | loss | loss | tie |

## Interpretation

- Verdict: does not clear the production replacement gate: best candidate `studentt30_incumbent_blend` wins NLL in 0/3 seasons with 3 ties, RMSE in 0/3 seasons with 3 ties, and NIS closeness in 1/3 seasons with 2 ties (majority threshold 2/3). Treat the variance as calibration-useful, but keep the incumbent R fallback for production.
- Best candidate overall gap vs incumbent: NLL -0.0000, RMSE -0.0000, NIS-closeness -0.0120.
- The best candidate is statistically indistinguishable from the incumbent on this scalar replay; the useful signal is parity plus better overall second-moment calibration, not a standalone replacement claim.
- This is still a game-metric observation replay, not a mutation of the production player DARKO filter. A production rollout should wire the same row-level `R_t` contract into that pipeline and retain an automatic incumbent fallback.

## Metric Details

| model | metric | season | n | NLL | RMSE | 90% cov | NIS mean | mean R | q | r scale | r mix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| studentt30_variance | fg_pct | ALL | 726 | -0.302 | 0.179 | 0.906 | 0.986 | 0.032 | 1e-06 |  |  |
| studentt30_replay_scaled | fg_pct | ALL | 726 | -0.302 | 0.179 | 0.906 | 0.986 | 0.032 | 1e-06 | 1.000 |  |
| studentt30_incumbent_blend | fg_pct | ALL | 726 | -0.303 | 0.179 | 0.912 | 0.966 | 0.033 | 1e-06 | 1.000 | 0.550 |
| incumbent_weight_heuristic | fg_pct | ALL | 726 | -0.303 | 0.179 | 0.912 | 0.946 | 0.034 | 1e-06 | 1.000 |  |
| studentt30_variance | fta_100 | ALL | 726 | -0.149 | 0.207 | 0.924 | 0.849 | 0.050 | 1.11786e-05 |  |  |
| studentt30_replay_scaled | fta_100 | ALL | 726 | -0.149 | 0.207 | 0.924 | 0.849 | 0.050 | 1.11786e-05 | 1.000 |  |
| studentt30_incumbent_blend | fta_100 | ALL | 726 | -0.158 | 0.206 | 0.908 | 0.963 | 0.043 | 1.81161e-05 | 1.000 | 0.150 |
| incumbent_weight_heuristic | fta_100 | ALL | 726 | -0.158 | 0.206 | 0.904 | 0.990 | 0.042 | 1.81161e-05 | 0.825 |  |
| studentt30_variance | pace | ALL | 726 | -1.550 | 0.047 | 0.981 | 0.502 | 0.004 | 1e-06 |  |  |
| studentt30_replay_scaled | pace | ALL | 726 | -1.646 | 0.046 | 0.884 | 1.108 | 0.002 | 1.62061e-06 | 0.445 |  |
| studentt30_incumbent_blend | pace | ALL | 726 | -1.649 | 0.046 | 0.895 | 1.076 | 0.002 | 1.62061e-06 | 0.445 | 0.000 |
| incumbent_weight_heuristic | pace | ALL | 726 | -1.649 | 0.046 | 0.895 | 1.076 | 0.002 | 1.62061e-06 | 0.562 |  |
| studentt30_variance | pf_100 | ALL | 726 | 1.186 | 0.780 | 0.875 | 1.091 | 0.550 | 0.0022638 |  |  |
| studentt30_replay_scaled | pf_100 | ALL | 726 | 1.195 | 0.781 | 0.861 | 1.222 | 0.490 | 0.0022638 | 0.891 |  |
| studentt30_incumbent_blend | pf_100 | ALL | 726 | 1.170 | 0.780 | 0.886 | 1.053 | 0.547 | 0.0022638 | 0.891 | 0.000 |
| incumbent_weight_heuristic | pf_100 | ALL | 726 | 1.170 | 0.780 | 0.886 | 1.053 | 0.547 | 0.0022638 | 0.825 |  |
| studentt30_variance | pts_100 | ALL | 726 | 2.100 | 1.972 | 0.913 | 0.987 | 3.984 | 1e-06 |  |  |
| studentt30_replay_scaled | pts_100 | ALL | 726 | 2.100 | 1.972 | 0.913 | 0.987 | 3.984 | 1e-06 | 1.000 |  |
| studentt30_incumbent_blend | pts_100 | ALL | 726 | 2.085 | 1.943 | 0.906 | 0.986 | 3.799 | 0.000531871 | 1.000 | 0.000 |
| incumbent_weight_heuristic | pts_100 | ALL | 726 | 2.085 | 1.943 | 0.906 | 0.986 | 3.799 | 0.000531871 | 1.000 |  |
| studentt30_variance | tov_100 | ALL | 726 | -0.461 | 0.152 | 0.916 | 0.927 | 0.024 | 2.9359e-05 |  |  |
| studentt30_replay_scaled | tov_100 | ALL | 726 | -0.461 | 0.152 | 0.916 | 0.927 | 0.024 | 2.9359e-05 | 1.000 |  |
| studentt30_incumbent_blend | tov_100 | ALL | 726 | -0.462 | 0.152 | 0.916 | 0.922 | 0.024 | 2.9359e-05 | 1.000 | 0.950 |
| incumbent_weight_heuristic | tov_100 | ALL | 726 | -0.462 | 0.152 | 0.923 | 0.844 | 0.026 | 2.9359e-05 | 1.000 |  |

## Metadata

```json
{
  "args": {
    "chimera_scale_max": 4.0,
    "chimera_scale_min": 0.25,
    "chimera_scale_steps": 25,
    "data": "/Users/kmedved/Library/CloudStorage/Dropbox/github/wnba_darko/calculated_data/research/observation_covariance_measurement/game_metric_observations.parq",
    "early_stopping_rounds": 40,
    "hybrid_mix_steps": 21,
    "iterations": 600,
    "l2_leaf_reg": "auto",
    "learning_rate": 0.04,
    "loss": "StudentT",
    "min_child_samples": 25,
    "num_leaves": 31,
    "output_csv": "/Users/kmedved/Code/GitHub/chimeraboost/benchmarks/wnba_kalman_replay.csv",
    "output_summary": "/Users/kmedved/Code/GitHub/chimeraboost/benchmarks/wnba_kalman_replay_summary.md",
    "r_ceil": 9.0,
    "r_floor": null,
    "r_floor_fraction": 0.25,
    "random_state": 42,
    "rho_l2_leaf_reg_multiplier": 1.0,
    "rho_learning_rate_multiplier": 1.0,
    "student_t_nu": 30.0,
    "thread_count": 1,
    "train_through_season": 2021,
    "val_start_season": 2022,
    "val_through_season": 2023
  },
  "bootstrap_summaries": [
    {
      "model": "studentt30_incumbent_blend",
      "n": 1572,
      "nis_closeness_ci_high": -0.0055815186986899286,
      "nis_closeness_ci_low": -0.014625225621498993,
      "nis_closeness_gap": -0.010235315828706848,
      "nis_closeness_result": "win",
      "nll_ci_high": 0.0013991986242488526,
      "nll_ci_low": -0.0021128214847942234,
      "nll_gap": -0.00045791830920083223,
      "nll_result": "tie",
      "rmse_ci_high": 6.202309456082565e-06,
      "rmse_ci_low": -1.9396718755215758e-05,
      "rmse_gap": -6.564290879862433e-06,
      "rmse_result": "tie",
      "season": "2024"
    },
    {
      "model": "studentt30_incumbent_blend",
      "n": 1860,
      "nis_closeness_ci_high": 0.019492687254298802,
      "nis_closeness_ci_low": -0.01864260434040736,
      "nis_closeness_gap": 0.010998491326259852,
      "nis_closeness_result": "tie",
      "nll_ci_high": 0.0018531581058296184,
      "nll_ci_low": -0.0016977352747592123,
      "nll_gap": 2.6133352739162717e-05,
      "nll_result": "tie",
      "rmse_ci_high": 5.324962228606122e-06,
      "rmse_ci_low": -8.862849350965706e-06,
      "rmse_gap": -1.424898008139941e-06,
      "rmse_result": "tie",
      "season": "2025"
    },
    {
      "model": "studentt30_incumbent_blend",
      "n": 924,
      "nis_closeness_ci_high": 0.013553725370638663,
      "nis_closeness_ci_low": -0.008330236507831806,
      "nis_closeness_gap": 0.007685852804336868,
      "nis_closeness_result": "tie",
      "nll_ci_high": 0.002950221805948866,
      "nll_ci_low": -0.0016278101210172005,
      "nll_gap": 0.0006024702029075511,
      "nll_result": "tie",
      "rmse_ci_high": 1.557022195536605e-05,
      "rmse_ci_low": -1.1462879364543265e-06,
      "rmse_gap": 6.9217283814593245e-06,
      "rmse_result": "tie",
      "season": "2026"
    },
    {
      "model": "studentt30_incumbent_blend",
      "n": 4356,
      "nis_closeness_ci_high": 0.013835639955183075,
      "nis_closeness_ci_low": -0.013939999705078582,
      "nis_closeness_gap": -0.011989056756825711,
      "nis_closeness_result": "tie",
      "nll_ci_high": 0.001128046667297335,
      "nll_ci_low": -0.0011294199074207813,
      "nll_gap": -2.6298686520268166e-05,
      "nll_result": "tie",
      "rmse_ci_high": 4.4925785649713144e-06,
      "rmse_ci_low": -7.0473964791706315e-06,
      "rmse_gap": -1.3082921408447135e-06,
      "rmse_result": "tie",
      "season": "ALL"
    },
    {
      "model": "studentt30_replay_scaled",
      "n": 1572,
      "nis_closeness_ci_high": 0.004931511474477432,
      "nis_closeness_ci_low": -0.016540956224890702,
      "nis_closeness_gap": -0.005909005613822904,
      "nis_closeness_result": "tie",
      "nll_ci_high": 0.005761517392597049,
      "nll_ci_low": -0.003212521219308251,
      "nll_gap": 0.0011399822348082689,
      "nll_result": "tie",
      "rmse_ci_high": 0.003595003783244295,
      "rmse_ci_low": -0.009122372185569643,
      "rmse_gap": -0.0029966477771287137,
      "rmse_result": "tie",
      "season": "2024"
    },
    {
      "model": "studentt30_replay_scaled",
      "n": 1860,
      "nis_closeness_ci_high": 0.04457557559615842,
      "nis_closeness_ci_low": -0.037663421211949,
      "nis_closeness_gap": 0.027628221593631985,
      "nis_closeness_result": "tie",
      "nll_ci_high": 0.014074614922725121,
      "nll_ci_low": 0.0023815165017378884,
      "nll_gap": 0.008288304865803365,
      "nll_result": "loss",
      "rmse_ci_high": 0.008811985016682244,
      "rmse_ci_low": -0.00478928558683252,
      "rmse_gap": 0.0019005142313512913,
      "rmse_result": "tie",
      "season": "2025"
    },
    {
      "model": "studentt30_replay_scaled",
      "n": 924,
      "nis_closeness_ci_high": 0.09702955825441083,
      "nis_closeness_ci_low": -0.013400563988097688,
      "nis_closeness_gap": 0.06992924233635023,
      "nis_closeness_result": "tie",
      "nll_ci_high": 0.035893640550645034,
      "nll_ci_low": 0.011603002810506087,
      "nll_gap": 0.023404281789489435,
      "nll_result": "loss",
      "rmse_ci_high": 0.07767086701308071,
      "rmse_ci_low": 0.016136827652973296,
      "rmse_gap": 0.047068702063878365,
      "rmse_result": "loss",
      "season": "2026"
    },
    {
      "model": "studentt30_replay_scaled",
      "n": 4356,
      "nis_closeness_ci_high": 0.03695316283499242,
      "nis_closeness_ci_low": -0.03521967655694437,
      "nis_closeness_gap": -0.004676618288740109,
      "nis_closeness_result": "tie",
      "nll_ci_high": 0.012905106186630778,
      "nll_ci_low": 0.004983203044921497,
      "nll_gap": 0.008915026514463064,
      "nll_result": "loss",
      "rmse_ci_high": 0.019395440012274164,
      "rmse_ci_low": 0.002230686981405108,
      "rmse_gap": 0.010842721156963386,
      "rmse_result": "loss",
      "season": "ALL"
    },
    {
      "model": "studentt30_variance",
      "n": 1572,
      "nis_closeness_ci_high": 0.11598320308830978,
      "nis_closeness_ci_low": 0.07534678916334472,
      "nis_closeness_gap": 0.09593970310114297,
      "nis_closeness_result": "loss",
      "nll_ci_high": 0.03525330823245406,
      "nll_ci_low": 0.017265533161383503,
      "nll_gap": 0.02598141200985551,
      "nll_result": "loss",
      "rmse_ci_high": 0.0034901919344351297,
      "rmse_ci_low": -0.009850952385472057,
      "rmse_gap": -0.0030410648536113305,
      "rmse_result": "tie",
      "season": "2024"
    },
    {
      "model": "studentt30_variance",
      "n": 1860,
      "nis_closeness_ci_high": 0.11639560161358735,
      "nis_closeness_ci_low": -0.024231097748585964,
      "nis_closeness_gap": 0.10030361969998447,
      "nis_closeness_result": "tie",
      "nll_ci_high": 0.028604160248761095,
      "nll_ci_low": 0.0073327203353904845,
      "nll_gap": 0.01790683052092927,
      "nll_result": "loss",
      "rmse_ci_high": 0.008574223568590122,
      "rmse_ci_low": -0.005135501564353833,
      "rmse_gap": 0.001791365892973129,
      "rmse_result": "tie",
      "season": "2025"
    },
    {
      "model": "studentt30_variance",
      "n": 924,
      "nis_closeness_ci_high": 0.07566769490084052,
      "nis_closeness_ci_low": -0.09687257922196016,
      "nis_closeness_gap": -0.05433981105969088,
      "nis_closeness_result": "tie",
      "nll_ci_high": 0.04497418225511677,
      "nll_ci_low": 0.014148149906459608,
      "nll_gap": 0.029937571952110546,
      "nll_result": "loss",
      "rmse_ci_high": 0.07693574364834986,
      "rmse_ci_low": 0.015672727566735184,
      "rmse_gap": 0.047021224109772164,
      "rmse_result": "loss",
      "season": "2026"
    },
    {
      "model": "studentt30_variance",
      "n": 4356,
      "nis_closeness_ci_high": 0.10433444621793626,
      "nis_closeness_ci_low": 0.04036739811697859,
      "nis_closeness_gap": 0.09199604370863912,
      "nis_closeness_result": "loss",
      "nll_ci_high": 0.029987815494055173,
      "nll_ci_low": 0.01729586142345979,
      "nll_gap": 0.02337277340040667,
      "nll_result": "loss",
      "rmse_ci_high": 0.018966242590012686,
      "rmse_ci_low": 0.0016428866987379443,
      "rmse_gap": 0.010771234062285973,
      "rmse_result": "loss",
      "season": "ALL"
    }
  ],
  "candidate_summaries": [
    {
      "clears_gate": false,
      "model": "studentt30_incumbent_blend",
      "nis_ties": 2,
      "nis_wins": 1,
      "nll_ties": 3,
      "nll_wins": 0,
      "overall_nis_closeness_gap": -0.011989056756825489,
      "overall_nll_gap": -2.6298686520279113e-05,
      "overall_rmse_gap": -1.3082921407336912e-06,
      "required_wins": 2,
      "rmse_ties": 3,
      "rmse_wins": 0,
      "season_count": 3
    },
    {
      "clears_gate": false,
      "model": "studentt30_replay_scaled",
      "nis_ties": 3,
      "nis_wins": 0,
      "nll_ties": 1,
      "nll_wins": 0,
      "overall_nis_closeness_gap": -0.004676618288739887,
      "overall_nll_gap": 0.008915026514463062,
      "overall_rmse_gap": 0.010842721156963386,
      "required_wins": 2,
      "rmse_ties": 2,
      "rmse_wins": 0,
      "season_count": 3
    },
    {
      "clears_gate": false,
      "model": "studentt30_variance",
      "nis_ties": 2,
      "nis_wins": 0,
      "nll_ties": 0,
      "nll_wins": 0,
      "overall_nis_closeness_gap": 0.09199604370863934,
      "overall_nll_gap": 0.023372773400406513,
      "overall_rmse_gap": 0.010771234062285973,
      "required_wins": 2,
      "rmse_ties": 2,
      "rmse_wins": 0,
      "season_count": 3
    }
  ],
  "details": {
    "best_iteration": 202,
    "dist_params": {
      "nu": 30.0
    },
    "group_count": 6,
    "loss": "StudentT",
    "r_floor": null,
    "r_floor_by_metric": {
      "fg_pct": 0.008467452176285157,
      "fta_100": 0.01070123752492613,
      "pace": 0.000485829590112519,
      "pf_100": 0.13829729985922617,
      "pts_100": 0.9602611260626465,
      "tov_100": 0.006677115944310615
    },
    "r_floor_fraction": 0.25,
    "r_tuning_by_metric": {
      "fg_pct": {
        "blend_chimera_mix": 0.55,
        "blend_q": 1e-06,
        "incumbent_q": 1e-06,
        "incumbent_scale": 1.0,
        "raw_q": 1e-06,
        "scaled_q": 1e-06,
        "scaled_scale": 1.0
      },
      "fta_100": {
        "blend_chimera_mix": 0.15000000000000002,
        "blend_q": 1.8116091942004134e-05,
        "incumbent_q": 1.8116091942004134e-05,
        "incumbent_scale": 0.8254041852680184,
        "raw_q": 1.1178591777554047e-05,
        "scaled_q": 1.1178591777554047e-05,
        "scaled_scale": 1.0
      },
      "pace": {
        "blend_chimera_mix": 0.0,
        "blend_q": 1.620605913741319e-06,
        "incumbent_q": 1.620605913741319e-06,
        "incumbent_scale": 0.5623413251903491,
        "raw_q": 1e-06,
        "scaled_q": 1.620605913741319e-06,
        "scaled_scale": 0.4454493590701697
      },
      "pf_100": {
        "blend_chimera_mix": 0.0,
        "blend_q": 0.002263803409521449,
        "incumbent_q": 0.002263803409521449,
        "incumbent_scale": 0.8254041852680184,
        "raw_q": 0.002263803409521449,
        "scaled_q": 0.002263803409521449,
        "scaled_scale": 0.8908987181403393
      },
      "pts_100": {
        "blend_chimera_mix": 0.0,
        "blend_q": 0.000531871171866456,
        "incumbent_q": 0.000531871171866456,
        "incumbent_scale": 1.0,
        "raw_q": 1e-06,
        "scaled_q": 1e-06,
        "scaled_scale": 1.0
      },
      "tov_100": {
        "blend_chimera_mix": 0.9500000000000001,
        "blend_q": 2.9359045735093353e-05,
        "incumbent_q": 2.9359045735093353e-05,
        "incumbent_scale": 1.0,
        "raw_q": 2.9359045735093353e-05,
        "scaled_q": 2.9359045735093353e-05,
        "scaled_scale": 1.0
      }
    },
    "sigma_affine_b": 1.0544134018500366
  },
  "season_results": [
    {
      "model": "studentt30_incumbent_blend",
      "nis_closeness_gap": -0.010235315828706848,
      "nis_result": "win",
      "nll_gap": -0.00045791830920081933,
      "nll_result": "tie",
      "rmse_gap": -6.564290879751411e-06,
      "rmse_result": "tie",
      "season": "2024"
    },
    {
      "model": "studentt30_incumbent_blend",
      "nis_closeness_gap": 0.010998491326259852,
      "nis_result": "tie",
      "nll_gap": 2.6133352739168836e-05,
      "nll_result": "tie",
      "rmse_gap": -1.424898008139941e-06,
      "rmse_result": "tie",
      "season": "2025"
    },
    {
      "model": "studentt30_incumbent_blend",
      "nis_closeness_gap": 0.007685852804336646,
      "nis_result": "tie",
      "nll_gap": 0.0006024702029075457,
      "nll_result": "tie",
      "rmse_gap": 6.9217283814593245e-06,
      "rmse_result": "tie",
      "season": "2026"
    },
    {
      "model": "studentt30_replay_scaled",
      "nis_closeness_gap": -0.005909005613822904,
      "nis_result": "tie",
      "nll_gap": 0.0011399822348083927,
      "nll_result": "tie",
      "rmse_gap": -0.0029966477771286026,
      "rmse_result": "tie",
      "season": "2024"
    },
    {
      "model": "studentt30_replay_scaled",
      "nis_closeness_gap": 0.027628221593632207,
      "nis_result": "tie",
      "nll_gap": 0.00828830486580344,
      "nll_result": "loss",
      "rmse_gap": 0.0019005142313512913,
      "rmse_result": "tie",
      "season": "2025"
    },
    {
      "model": "studentt30_replay_scaled",
      "nis_closeness_gap": 0.06992924233635023,
      "nis_result": "tie",
      "nll_gap": 0.023404281789489445,
      "nll_result": "loss",
      "rmse_gap": 0.047068702063878365,
      "rmse_result": "loss",
      "season": "2026"
    },
    {
      "model": "studentt30_variance",
      "nis_closeness_gap": 0.09593970310114297,
      "nis_result": "loss",
      "nll_gap": 0.02598141200985553,
      "nll_result": "loss",
      "rmse_gap": -0.0030410648536112195,
      "rmse_result": "tie",
      "season": "2024"
    },
    {
      "model": "studentt30_variance",
      "nis_closeness_gap": 0.10030361969998447,
      "nis_result": "tie",
      "nll_gap": 0.0179068305209293,
      "nll_result": "loss",
      "rmse_gap": 0.001791365892973129,
      "rmse_result": "tie",
      "season": "2025"
    },
    {
      "model": "studentt30_variance",
      "nis_closeness_gap": -0.05433981105969088,
      "nis_result": "tie",
      "nll_gap": 0.02993757195211058,
      "nll_result": "loss",
      "rmse_gap": 0.04702122410977205,
      "rmse_result": "loss",
      "season": "2026"
    }
  ]
}
```
