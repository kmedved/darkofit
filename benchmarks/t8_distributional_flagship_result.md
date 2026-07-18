# T8 distributional flagship result

- Raw CSV SHA-256: `eedf6e037a3ef1e6628fdf2a2ae1c46bf8fae09df93b6566dd3c8b22dba75578`
- Analyzer SHA-256: `c8b52ee6313b7b3406648277aec53661f993ec4f53fe276201588044b31d4c0e`
- Complete coordinates: **75/75**
- Status: descriptive Tier-E evidence; no default or automatic policy was tested.

Coverage is the first number in every cell and width is the second. They are intentionally not collapsed into a single score.

## Per-dataset 90% interval results

| Dataset | Model | Coverage | Absolute gap | Width | NLL | CRPS | Fit s | Predict s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_100k | darkofit_gaussian_es_calibrated | 0.9002 | 0.0002 | 2.3387 | 1.0210 | 0.3996 | 0.634 | 0.0208 |
| synthetic_100k | darkofit_gaussian_es_conformal | 0.8997 | 0.0003 | 2.3335 | 1.0211 | 0.3996 | 0.651 | 0.0494 |
| synthetic_100k | ngboost | 0.8978 | 0.0022 | 2.2667 | 0.9964 | 0.3922 | 31.176 | 0.1408 |
| synthetic_100k | catboost_uncertainty | 0.9025 | 0.0025 | 2.3395 | 1.0186 | 0.3985 | 0.392 | 0.0012 |
| synthetic_100k | lightgbm_quantile_pair | 0.9000 | 0.0000 | 2.3965 | - | - | 0.483 | 0.0403 |
| synthetic_t3_100k | darkofit_gaussian_es_calibrated | 0.9290 | 0.0290 | 2.2875 | 1.0142 | 0.3517 | 0.591 | 0.0198 |
| synthetic_t3_100k | darkofit_gaussian_es_conformal | 0.8987 | 0.0013 | 1.9530 | 1.0134 | 0.3514 | 0.586 | 0.0448 |
| synthetic_t3_100k | ngboost | 0.9197 | 0.0197 | 2.3998 | 0.9906 | 0.3604 | 31.158 | 0.1271 |
| synthetic_t3_100k | catboost_uncertainty | 0.9272 | 0.0272 | 2.2651 | 1.0145 | 0.3526 | 0.415 | 0.0013 |
| synthetic_t3_100k | lightgbm_quantile_pair | 0.8988 | 0.0012 | 2.0190 | - | - | 0.517 | 0.0371 |
| openml_cpu_act | darkofit_gaussian_es_calibrated | 0.9214 | 0.0214 | 8.2691 | 2.2502 | 1.2837 | 0.254 | 0.0072 |
| openml_cpu_act | darkofit_gaussian_es_conformal | 0.9085 | 0.0085 | 7.6122 | 2.2388 | 1.2679 | 0.255 | 0.0107 |
| openml_cpu_act | ngboost | 0.8861 | 0.0139 | 6.4534 | 2.2727 | 1.2198 | 3.522 | 0.0281 |
| openml_cpu_act | catboost_uncertainty | 0.9111 | 0.0111 | 7.6147 | 2.2299 | 1.3387 | 0.231 | 0.0004 |
| openml_cpu_act | lightgbm_quantile_pair | 0.8350 | 0.0650 | 11.2479 | - | - | 0.243 | 0.0036 |
| openml_wine_quality | darkofit_gaussian_es_calibrated | 0.9007 | 0.0007 | 2.1641 | 1.0182 | 0.3767 | 0.161 | 0.0029 |
| openml_wine_quality | darkofit_gaussian_es_conformal | 0.9104 | 0.0104 | 2.2368 | 1.0200 | 0.3755 | 0.162 | 0.0063 |
| openml_wine_quality | ngboost | 0.8726 | 0.0274 | 2.0231 | 1.0429 | 0.3840 | 1.565 | 0.0149 |
| openml_wine_quality | catboost_uncertainty | 0.8800 | 0.0200 | 2.0180 | 1.0318 | 0.3811 | 0.096 | 0.0004 |
| openml_wine_quality | lightgbm_quantile_pair | 0.9233 | 0.0233 | 2.2440 | - | - | 0.176 | 0.0016 |
| openml_boston | darkofit_gaussian_es_calibrated | 0.9134 | 0.0134 | 13.5615 | 2.9702 | 2.1477 | 0.152 | 0.0004 |
| openml_boston | darkofit_gaussian_es_conformal | 0.9344 | 0.0344 | 15.3609 | 2.9533 | 1.9862 | 0.153 | 0.0007 |
| openml_boston | ngboost | 0.5512 | 0.3488 | 3.9488 | 4.7914 | 1.7698 | 0.293 | 0.0076 |
| openml_boston | catboost_uncertainty | 0.7244 | 0.1756 | 5.8864 | 3.1847 | 1.7712 | 0.074 | 0.0002 |
| openml_boston | lightgbm_quantile_pair | 0.7349 | 0.1651 | 9.0505 | - | - | 0.171 | 0.0009 |

## Equal-dataset coverage and width

| Model | Mean coverage | Mean absolute coverage gap | Geomean width / conformal | Worst cell coverage | Best cell coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| darkofit_gaussian_es_calibrated | 0.9129 | 0.0129 | 1.0172 | 0.8425 | 0.9843 |
| darkofit_gaussian_es_conformal | 0.9103 | 0.0110 | 1.0000 | 0.8898 | 0.9843 |
| ngboost | 0.8255 | 0.0824 | 0.7487 | 0.5276 | 0.9218 |
| catboost_uncertainty | 0.8691 | 0.0473 | 0.8334 | 0.6535 | 0.9298 |
| lightgbm_quantile_pair | 0.8584 | 0.0509 | 0.9850 | 0.7008 | 0.9237 |

## Conformal-versus-parametric DarkoFit

- Equal-dataset change in absolute coverage error: **-0.0020** (negative is closer to 90%).
- Geometric-mean interval-width ratio: **0.9831×**.
- NLL and CRPS describe the fitted Gaussian distribution, not the conformal interval; they are never assigned to the quantile-only LightGBM lane.
- Split-conformal coverage is marginal. This campaign does not claim conditional coverage, superiority on every dataset, or a default change.

## Integrity checks

- Every preregistered coordinate completed successfully.
- Every model at a dataset/seed coordinate used the same fingerprinted train/test arrays.
- Conformal rows report a nonempty isolated calibration set.
- Interval-only baselines do not report midpoint RMSE, NLL, or CRPS as though they exposed a predictive distribution.
