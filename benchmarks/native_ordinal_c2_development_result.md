# Native ordinal C2 development result

**Decision: close_native_ordinal_c2_development (FAIL).**

## Headline

- Equal-task test RMSE ratio: `0.992755`.
- Task wins: `3` of `4`.
- Worst task ratio: `1.317510`.
- Worst split ratio: `2.435933`.
- Worst validation task ratio: `1.268106`.
- Median fit / predict / RSS ratios: `1.008193` / `1.040711` / `1.000671`.

## Per-task quality

| Task | Dataset | Test ratio | Validation ratio |
|---:|---|---:|---:|
| 361268 | fps_benchmark | 1.317510 | 1.268106 |
| 363375 | ae_price_prediction | 0.990822 | 0.989776 |
| 363471 | munich-rent-index-1999 | 0.992230 | 1.011009 |
| 363631 | diamonds | 0.749905 | 0.743866 |

## Gates

| Gate | Value | Pass |
|---|---:|:---:|
| integrity | 0 | yes |
| equal_task_test_rmse | 0.992755 | no |
| task_wins | 3 | yes |
| worst_task | 1.317510 | no |
| worst_split | 2.435933 | no |
| validation | 1.268106 | no |
| fit_time | 1.008193 | yes |
| predict_time | 1.040711 | yes |
| peak_rss | 1.000671 | yes |
| fit_dispersion | 0.036502 | yes |
| predict_dispersion | 0.157556 | no |
| confirmation_power | 1.000000 | yes |

Raw artifact: `benchmarks/native_ordinal_c2_development_raw.json` (`2599029d7f4c8f7464c26af27d0aadf8e8443f47f1a52b0f794b8dd912c10d8a`).

CTR23 lockbox touched: **no**.
