# T7b automatic scalar-RMSE L2 v1 terminal result

Run once on 2026-07-22 under frozen contract
`t7b-automatic-scalar-rmse-l2-v1-20260722`.

The clean harness was
`454fee09794a5090c68614b46a2f2be455a53b38`. The M6 comparison used clean
pre-mechanism control `370b8924c034de0332a4b990817972cf0e876f3e`
and clean private candidate
`4bf425fcf3ef095679176b8326ea6621830b64cc`. The candidate changed only
the frozen allowlist: `darkofit/booster.py`, `tests/test_darkofit.py`, and
`tests/test_t7b_automatic_l2_policy.py`.

## Pre-quality evidence

The candidate's focused policy suite passed all nine cases. The full local
suite passed `3,146` tests with `34` skips after explicitly deselecting two
unavailable-environment prerequisites: the M3b sports-panel cache check and
the captured ChimeraBoost checkout check. The same run without those
deselections produced only those two prerequisite failures.

The cross-revision invariant probe showed exact predictions and exact logical
fitted state for all five declared no-op families: explicit CatBoost RMSE L2,
CatBoost classification, CatBoost MAE, LightGBM RMSE, and hybrid RMSE. Both
workers restored their thread-local Numba masks.

M5 then passed all 19 paired sentinel cells. Candidate and control behavior
fingerprints were equal, all earned floors passed, baseline drift was empty,
and advancement was not blocked. M5 resource ratios are non-ranking telemetry.

## Frozen M6 v3 result

The one permitted M6 inspection validated all 120 rows covering 60 paired
medium cells, ten datasets, three seeds, unweighted and stress-weighted
policies, three repeats, and four threads.

| Frozen gate | Result | Limit | Status |
| --- | ---: | ---: | --- |
| Equal-cell geometric-mean primary-loss ratio | `1.000818` | `<= 1.000000` | **fail** |
| Worst dataset-group ratio (`diabetes_resampled`) | `1.010896` | `<= 1.020000` | pass |
| Worst leave-one-dataset-out ratio (omit `wide_numeric_reg`) | `1.001370` | `<= 1.003000` | pass |

The worst individual coordinate was
`diabetes_resampled/medium/1/stress` at `1.032550`; the frozen harm gate is
the dataset-group geometric mean, not the largest individual cell. All six
classification datasets were exact quality no-ops and remained in the
aggregate as declared.

Adjacent non-gating candidate/control geometric-mean telemetry was `1.001920`
for fit time, `1.003624` for prediction time, and `1.001976` for worker peak
RSS. These timing and memory readings do not alter the quality disposition.

## Create-only artifacts

- [`t7b_automatic_l2_v1_invariants_20260722.json`](t7b_automatic_l2_v1_invariants_20260722.json),
  SHA-256 `c3dee2ecb521648e2f9521e280267d41301361cf9aeccfd84ef77b817f4443f9`;
- [`t7b_automatic_l2_v1_m5_20260722.json`](t7b_automatic_l2_v1_m5_20260722.json),
  SHA-256 `3bc489a9304ccd0021ed936b8eeec3bcfb1ab6b37476b8ecc87ffb3943a3c747`;
- [`t7b_automatic_l2_v1_m6_inspection1_launch_manifest_20260722.json`](t7b_automatic_l2_v1_m6_inspection1_launch_manifest_20260722.json),
  SHA-256 `593d44e331b9be14f1683947315e60eaefbe947b4460e7d99354074564fc4e1f`;
- [`t7b_automatic_l2_v1_m6_inspection1_raw_20260722.csv`](t7b_automatic_l2_v1_m6_inspection1_raw_20260722.csv),
  SHA-256 `dfa5560d752f1c17fa8dea0b497d90ebfaf1cb63275f2d69e7ca0afd57677a3a`;
- [`t7b_automatic_l2_v1_m6_inspection1_result_20260722.json`](t7b_automatic_l2_v1_m6_inspection1_result_20260722.json),
  SHA-256 `6fc5ececda62da257fd3e00fce7df1b8dba2978501e689d0d2a2ca678f296f26`;
- [`t7b_automatic_l2_v1_m6_inspection1_result_20260722.json.manifest.json`](t7b_automatic_l2_v1_m6_inspection1_result_20260722.json.manifest.json),
  SHA-256 `034bfbc47a2ef1fe872efa57cc52f3eb97d5986e269cc219e0bde802eab558d8`;
  and
- [`t7b_automatic_l2_v1_m6_inspection1_terminal_attestation_20260722.json`](t7b_automatic_l2_v1_m6_inspection1_terminal_attestation_20260722.json),
  SHA-256 `6bc045080cb6db0a38f912d5d7b31d10d5483e392520dcf682d057db43d05419`.

## Decision

The frozen disposition is `closed_in_m6`. Inspection 1 is spent and no rerun
is authorized. Candidate `4bf425fc` is not merged, the public automatic L2
policy remains unchanged, and this result authorizes no sports, fresh,
TabArena, lockbox, shipping, or default claim.

The samples-per-feature depth policy was explicitly excluded from this
identity. It remains a separate candidate-generation hypothesis and would
require a new one-mechanism contract, source identity, and inspection; this
L2 failure may not be used to impute its outcome.
