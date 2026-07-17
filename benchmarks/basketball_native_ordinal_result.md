# Basketball native-ordinal no-engagement result

## Decision

The C1 basketball fatal screen **passed**. Native ordinal handling remained an exact inactive no-op on numeric basketball data and stayed within every preregistered runtime and memory gate.

This authorizes only the frozen C2 categorical development tier. It does not authorize a categorical default or quality claim.

Decision code: `authorize_frozen_c2_categorical_development`.

## Gates

| Gate | Passed |
|---|---:|
| `all_workers_succeeded` | **True** |
| `source_runtime_data_bound` | **True** |
| `cache_isolated_and_warmed` | **True** |
| `no_unexpected_warnings` | **True** |
| `historical_predictions_reproduced` | **True** |
| `paired_predictions_bitwise_exact` | **True** |
| `historical_fit_contract_reproduced` | **True** |
| `paired_logical_model_state_exact` | **True** |
| `preprocessing_contract_reproduced` | **True** |
| `archive_contract_reproduced` | **True** |
| `guardrail_score_contract_reproduced` | **True** |
| `ordinal_no_engagement_contract` | **True** |
| `behavior_reproduced_across_blocks` | **True** |
| `timing_values_valid` | **True** |
| `median_total_fit_ratio_at_most_1_02` | **True** |
| `median_held_prediction_ratio_at_most_1_05` | **True** |
| `median_cold_prediction_ratio_at_most_1_05` | **True** |
| `fit_paired_ratio_stable` | **True** |
| `held_prediction_paired_ratio_stable` | **True** |
| `cold_prediction_paired_ratio_stable` | **True** |
| `median_peak_rss_ratio_at_most_1_05` | **True** |

## Paired operating ratios

| Metric | Median candidate/control | IQR/median |
|---|---:|---:|
| Eleven-fit total | 1.0029× | 0.0484 |
| Held-team prediction | 1.0008× | 0.0528 |
| Cold-player prediction | 0.9996× | 0.0656 |
| Peak RSS | 0.9987× | 0.0090 |

Per-arm timing dispersion is diagnostic only. Stability gates use the preregistered same-block paired ratios.

Raw artifact SHA-256: `7118d8dad2e0a9f7deb5d2c2425255cadfa8d810404fc8431070e3fa86dc6ea2`.

No CTR23, TabArena, I3, fresh-confirmation, or lockbox coordinate was touched.
