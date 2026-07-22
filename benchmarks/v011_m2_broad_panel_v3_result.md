# v0.11 M2 broad-panel result

Status: **complete descriptive spent evidence; no default, public exposure, M4,
release, fresh-confirmation, or lockbox action is authorized.**

## Execution and integrity

- The v3 contract ran from clean published DarkoFit commit `a2983ce` with the
  exact ChimeraBoost pin `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`,
  CatBoost `1.2.10`, TabArena `4cd1d2526874962daae048a6f2dcf34aa272f3fa`,
  and AutoGluon `1.5.1b20260712`.
- All `117` ordered fresh workers and `936` child fits completed at the common
  14-CPU/thread budget. Validation found zero model failures, imputations,
  deadline hits, or known time-limit stops, and all `117` worker attestations.
- V1 stopped before output creation because its historical 18-CPU requirement
  did not match this 14-core host. V2 passed dry-run but stopped before warmup
  or fitting because the warmup module retained its own 18-thread constant.
  Neither identity was rerun; v3 preserved the scientific protocol and fixed
  only that warmup binding.
- The run is defaults-only and singles-only across 13 fixed TabArena regression
  datasets, three registered coordinates per dataset, and eight sequential bag
  folds per outer job. The private ensemble is absent.

## Comparative result

All ratios are numerator/denominator and lower is better. Point estimates give
each fixed dataset equal weight after averaging its three paired log ratios.
Intervals bootstrap coordinates within each fixed dataset; they do not treat
the 13 datasets as a random population.

| Contrast | Test RMSE ratio (95% descriptive interval) | Fit-time ratio | Predict-time ratio | Incremental RSS ratio | Dataset quality W-L-T |
| --- | ---: | ---: | ---: | ---: | ---: |
| DarkoFit / ChimeraBoost | 1.0174 [1.0135, 1.0216] | 0.8126 [0.7789, 0.8483] | 1.3166 [1.2676, 1.3666] | 0.8425 [0.8184, 0.8671] | 6-7-0 |
| DarkoFit / CatBoost | 1.0538 [1.0511, 1.0569] | 0.0913 [0.0858, 0.0968] | 1.2706 [1.2162, 1.3249] | 0.3689 [0.3516, 0.3871] | 1-12-0 |
| ChimeraBoost / CatBoost | 1.0358 [1.0330, 1.0387] | 0.1124 [0.1071, 0.1177] | 0.9651 [0.9285, 1.0001] | 0.4379 [0.4236, 0.4533] | 2-11-0 |

DarkoFit therefore trades quality and prediction throughput for much faster
fit and lower memory on this frozen panel. Against ChimeraBoost it is 1.74%
worse in equal-dataset RMSE, takes 18.7% less fit time and 31.7% more prediction
time, and uses 15.8% less incremental RSS. Against CatBoost it is 5.38% worse
in RMSE, takes 90.9% less fit time (equivalent to about 10.95 times the fit
throughput), takes 27.1% more prediction time, and uses 63.1% less incremental
RSS. Peak-RSS ratios were `0.9627x` against ChimeraBoost and `0.7071x` against
CatBoost. CatBoost wins quality on 12 of 13 datasets; DarkoFit wins six of 13
against ChimeraBoost.

## Artifacts and provenance

- Contract SHA-256: `719213fd993b8626d7ece192fa9b9581ffa4ea6220d0f7d94a598683e098f846`.
- Protocol SHA-256: `d4c8bc3fbe980149a3528d13a7f9fd6393f4690517a862cef31d8e622796e403`.
- Ordered-job SHA-256: `ed35ca18a759b74ab9f26373e2d253c5970d2dab3788e1139d23466429cf0385`.
- Raw-result-set SHA-256: `81ee5327e7e2e4997af421ad6ab5579bbd12e1099552898c76552483c217cda3`.
- Manifest SHA-256: `c96b020c82a873091d984dc992eef6a78a8bb6607271c0f7c9921531fd97867c`.
- Warmup-history SHA-256: `a0e5f1104c7091a5fd60bdfc780688153cb27c53c73134233d6850b67241a32e`.
- Normalized analysis-payload SHA-256:
  `327f24f90383865ea8502118ea622ecf3b983a34926826c4ba96998bccb11f8d`.
- Completion-attestation SHA-256:
  `1fbd09e4e71e537d58479b4343e3269a1cb7d1a8b56e6f8d23a59aa4b96c4b5c`.
- Committed analyzer summary and newline-normalized CSV tables:
  [`v011_m2_broad_panel_v3_result.json`](v011_m2_broad_panel_v3_result.json),
  [`v011_m2_broad_panel_v3_paired_coordinates.csv`](v011_m2_broad_panel_v3_paired_coordinates.csv),
  and
  [`v011_m2_broad_panel_v3_per_dataset.csv`](v011_m2_broad_panel_v3_per_dataset.csv).
- The summary is byte-exact at SHA-256
  `e995b96760f0f48eff6ca0745a45055128c10c9a4b73bb0c7b25c55402157af0`.
  The analyzer-cache CSV hashes are
  `891b0d5e72e70e11c3cf2d2877aa8328d5f1cbe9b391de7ce2bcd1ac36cf095b`
  and
  `a1a309c2f2c71abbce16d758100f552978bdc620ab4862c3265d777204cba119`;
  the committed LF-normalized copies preserve every field and hash to
  `0ca6c1d139c138ca48d116332c56bbb747d16aa6b14e973c3960d0d8befa8020`
  and
  `9b0dcab3baccad428dbef40be82cab219d1a6c2fb7751413f947470cb5301ab3`.

## Limitations and disposition

The panel is a fixed, already-spent development set on one Apple-silicon
machine. It is not a fresh generalization estimate, leaderboard claim, or
certificate. The 13 fixed datasets are equally weighted but are not asserted
to be 13 independent draws. Competitor stop-reason metadata remains partly
unresolved (`443` child fits reported `unknown`), while the harness verified
that none was a known deadline/time-limit stop and no model failed.

Phase 2 is complete. The owner-authorized evidence phase stops here: exposing
ensemble-v3, running M4, or releasing v0.11 each still requires separate owner
authorization.
