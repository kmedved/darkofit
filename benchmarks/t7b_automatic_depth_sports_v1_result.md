# T7b automatic depth spent-sports v1 terminal result

Run once on 2026-07-22/23 under frozen contract
`t7b-automatic-depth-spent-sports-v1-20260722`.

The clean published harness was
`ac51c3e3379f855ba960f684375bee49cf0910e4`. It compared clean
pre-mechanism control `e23d2b164f10374b1c0e02521c33fc96d48980da`
with the exact private candidate
`41e948f0c53b1d124e16071a7fa66eba47d084d3`. The contract SHA-256 was
`ac5a745378a086ed119af1d55a68e961ea95e1f74c4f307dce72ad9b6717fe1b`.

## Frozen quality result

All 18 fresh-interpreter rows completed across the exact nine spent sports cases:
2014--2016 crossed with `minutes_per_game`, `game_score`, and
`box_plus_minus`. Each model fit on the frozen non-held-team rows with
player-group-aware validation. Primary RMSE was scored on cold-player rows
inside the held-team set; all-held-team RMSE was the secondary guard.

| Frozen gate | Result | Limit | Status |
| --- | ---: | ---: | --- |
| Cold-player equal-lineage geometric-mean ratio | `0.950266` | `<= 1.000000` | pass |
| All-held-team equal-lineage geometric-mean ratio | `0.951078` | `<= 1.010000` | pass |
| Season-cluster bootstrap p95 | `0.966591` | `<= 1.010000` | pass |
| Worst season (`2014`) | `0.972028` | `<= 1.020000` | pass |
| Worst lineage (`2016/minutes_per_game`) | `0.997200` | `<= 1.030000` | pass |
| Worst leave-one-season-out (omit `2015`) | `0.963884` | `<= 1.003000` | pass |

The candidate improved cold-player RMSE in all nine lineages. Season ratios
were `0.972028` (2014), `0.923605` (2015), and `0.955809` (2016). The
season-cluster bootstrap used exactly three clusters, 100,000 draws, and seed
`20260722`; its descriptive interval was `[0.923605, 0.972028]` at the 2.5th
and 97.5th percentiles. These are three dependent, already-spent seasons—not
nine independent datasets and not fresh confirmation.

Every candidate row engaged the frozen low-density policy at depth 4; every
control row retained depth 6. L2 stayed exactly `3.0` in both arms. All input,
thread-state, source-import, fitted-thread, safe-NPZ, and prediction-parity
invariants passed.

## Non-gating telemetry

The single-run equal-lineage candidate/control geometric-mean ratios were:

- fit time `0.601281`;
- prediction time `1.038138`;
- worker-plus-child peak RSS `0.988196`; and
- safe-NPZ bytes `0.732452`.

These are disclosure only. In particular, individual prediction intervals
were too short and noisy for a speed claim; their ratios ranged from
`0.235917` to `4.424803`. No timing or memory observation affected the
quality disposition.

## Create-only artifacts

- [`t7b_automatic_depth_sports_v1_contract.json`](t7b_automatic_depth_sports_v1_contract.json),
  SHA-256 `ac5a745378a086ed119af1d55a68e961ea95e1f74c4f307dce72ad9b6717fe1b`;
- [`t7b_automatic_depth_sports_v1_inspection1_launch_manifest_20260722.json`](t7b_automatic_depth_sports_v1_inspection1_launch_manifest_20260722.json),
  SHA-256 `07567f2585df0183bbd0f6dee9b3c18d678e28b3280ddd41c21331a23439bac1`;
- [`t7b_automatic_depth_sports_v1_inspection1_raw_20260722.json`](t7b_automatic_depth_sports_v1_inspection1_raw_20260722.json),
  SHA-256 `31b4d18576ed35efae3fe89e07375f18b82c02668586a562aa9969d1c9f0830d`;
- [`t7b_automatic_depth_sports_v1_inspection1_result_20260722.json`](t7b_automatic_depth_sports_v1_inspection1_result_20260722.json),
  SHA-256 `1ec0d2d37ef75195b66b779ec94920e05f5047147538de6eb17622947fd1a0da`;
  and
- [`t7b_automatic_depth_sports_v1_inspection1_terminal_attestation_20260722.json`](t7b_automatic_depth_sports_v1_inspection1_terminal_attestation_20260722.json),
  SHA-256 `180e7ea418b4a5e53c0672c2c5b5c1672824dc83fc3f9b3e279bca0cd19d9644`.

## Decision

The frozen disposition is `eligible_for_fresh_tier_d_design`. Inspection 1
is spent and no rerun is authorized. Candidate `41e948f0` remains private and
unmerged, and the public automatic-depth policy is unchanged.

This result authorizes only the design of a separately owner-approved,
prospective, powered Tier-D campaign on eligible fresh confirmation data. It
does not authorize that fresh-data access itself, a merge, default change,
public API, M2, TabArena, a release, or lockbox access.
