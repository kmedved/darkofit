# T7b automatic scalar-RMSE depth v1 general-development result

Run once on 2026-07-22 under frozen contract
`t7b-automatic-scalar-rmse-depth-v1-20260722`.

The clean harness was
`a1eb76071ca72a52494f6cca6022ea8ace8d5394`. The M6 comparison used clean
pre-mechanism control `e23d2b164f10374b1c0e02521c33fc96d48980da`
and clean private candidate
`41e948f0c53b1d124e16071a7fa66eba47d084d3`. The candidate changed only
the frozen allowlist: `darkofit/booster.py`, `tests/test_darkofit.py`, and
`tests/test_t7b_automatic_depth_policy.py`.

## Pre-quality evidence

The candidate full local suite passed `3,167` tests with `34` skips after
explicitly deselecting two unavailable-environment prerequisites: the M3b
sports-panel cache check and the captured ChimeraBoost checkout check. The
same run without those deselections produced only those two prerequisite
failures. A final focused policy, serialization, dispatch, and thread-state
suite passed all `131` tests.

The cross-revision invariant probe showed exact predictions and exact logical
fitted state for all seven declared no-op families. It also engaged the
low-, middle-, and high-density branches at the exact frozen boundaries and
confirmed that automatic L2 stayed unchanged. Both workers restored their
thread-local Numba masks.

M5 then passed all 19 paired sentinel cells. Candidate and control behavior
fingerprints were equal, all earned floors passed, baseline drift was empty,
and advancement was not blocked. M5 resource ratios are non-ranking telemetry.

## Frozen M6 v3 result

The one permitted M6 inspection validated all 120 rows covering 60 paired
medium cells, ten datasets, three seeds, unweighted and stress-weighted
policies, three repeats, and four threads.

| Frozen gate | Result | Limit | Status |
| --- | ---: | ---: | --- |
| Equal-cell geometric-mean primary-loss ratio | `0.992921` | `<= 1.000000` | pass |
| Worst dataset-group ratio (`diabetes_resampled`) | `1.011124` | `<= 1.020000` | pass |
| Worst leave-one-dataset-out ratio (omit `wide_numeric_reg`) | `1.001230` | `<= 1.003000` | pass |

The worst individual coordinate was
`diabetes_resampled/medium/1/stress` at `1.037673`; the frozen harm gate is
the dataset-group geometric mean, not the largest individual cell. The
`wide_numeric_reg` group ratio was `0.921178`. All six classification
datasets were exact quality no-ops and remained in the aggregate as declared.

Adjacent non-gating candidate/control geometric-mean telemetry was `0.849501`
for fit time, `0.933229` for prediction time, and `0.993834` for worker peak
RSS. On the four regression datasets, fit-time telemetry was `0.675402`.
These timing and memory readings do not alter the quality disposition.

## Create-only artifacts

- [`t7b_automatic_depth_v1_invariants_20260722.json`](t7b_automatic_depth_v1_invariants_20260722.json),
  SHA-256 `02362e5d7080c155add0846a58b6960db997bd29a0374e936a16a5a5364e5aff`;
- [`t7b_automatic_depth_v1_m5_20260722.json`](t7b_automatic_depth_v1_m5_20260722.json),
  SHA-256 `1d3eac70f81babeb628850cf19844d7b4c590c6df67ded723fcf7caba019bca1`;
- [`t7b_automatic_depth_v1_m6_inspection1_launch_manifest_20260722.json`](t7b_automatic_depth_v1_m6_inspection1_launch_manifest_20260722.json),
  SHA-256 `7eb95710c761f0682c00cf4b5971233089c70e654c5e5adc316d5388d933dc46`;
- [`t7b_automatic_depth_v1_m6_inspection1_raw_20260722.csv`](t7b_automatic_depth_v1_m6_inspection1_raw_20260722.csv),
  SHA-256 `e8e651459fafdea7ace0d298ccedd2c8d87145b945928111d475a007b955bafe`;
- [`t7b_automatic_depth_v1_m6_inspection1_result_20260722.json`](t7b_automatic_depth_v1_m6_inspection1_result_20260722.json),
  SHA-256 `7af0c480221b5886c7bbf41f810147663d9da6e2c4171a70bc9db3a431eebb28`;
- [`t7b_automatic_depth_v1_m6_inspection1_result_20260722.json.manifest.json`](t7b_automatic_depth_v1_m6_inspection1_result_20260722.json.manifest.json),
  SHA-256 `dbb47702f4e7992f34e653ea1155a8638e4e1945dbda0da1eb582345c73c32c7`;
  and
- [`t7b_automatic_depth_v1_m6_inspection1_terminal_attestation_20260722.json`](t7b_automatic_depth_v1_m6_inspection1_terminal_attestation_20260722.json),
  SHA-256 `b925aab09fdd71ca0f8887e1d3a4023c20412b2eefc337f2a2a7c1d5a267f598`.

## Decision

The frozen disposition is `advance`. Inspection 1 is spent and no rerun is
authorized. Candidate `41e948f0` is eligible only for one separately frozen
check on already-spent sports data. It remains private and unmerged; this
result authorizes no fresh, TabArena, lockbox, shipping, release, or default
claim.

Any sports result must preserve this exact candidate and use a new contract,
source identity, create-only artifacts, season-clustered uncertainty, and
concentration/harm guards. A favorable general result cannot waive those
requirements.
