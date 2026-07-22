# Automatic linear-selector v2 M6 v3 inspection 1

Run once on 2026-07-22 from clean harness commit
`56a66700a354dfe90d4cfd72d4254a7d8e22b351`, clean control
`b11f013f7ba926e533c38db8261f1a569ebce6c6`, and clean published candidate
`a53d4bf543534678189d87d88dcad87dd2a8bd8f`.

## Result

The frozen runner validated all 120 rows covering 60 paired medium cells,
three seeds, two weight policies, and ten regression/classification datasets.
Every candidate/control primary-loss ratio was exactly `1.000000`. The
aggregate ratio, all ten dataset ratios, all ten leave-one-dataset-out ratios,
and the worst-cell ratio were therefore `1.000000`.

All three frozen gates passed:

- aggregate ratio at most `1.000`;
- worst-dataset ratio at most `1.020`; and
- worst leave-one-dataset-out ratio at most `1.003`.

The separately frozen engagement companion explains this exactness without
altering the ranking result: none of the 60 cells selected linear leaves. The
candidate used exact non-applicable or constant-leaf behavior on every cell.

Adjacent cost telemetry was not an M6 gate. Across the 60 paired cells, the
candidate/control geometric-mean ratios were `1.196564` for fit time,
`0.998746` for prediction time, and `1.022557` for worker peak RSS. The largest
fit-time ratio was `3.087023`; eligible regression auditions account for the
extra training work.

The create-only artifacts are:

- [`automatic_linear_selector_v2_m6_v3_inspection1_raw_20260722.csv`](automatic_linear_selector_v2_m6_v3_inspection1_raw_20260722.csv),
  SHA-256
  `e30d089e79d177eb866514e45a0a9ec921a25e46f15e293d72d95525a86cec66`;
- [`automatic_linear_selector_v2_m6_v3_inspection1_result_20260722.json`](automatic_linear_selector_v2_m6_v3_inspection1_result_20260722.json),
  SHA-256
  `7445b70ca3bc727bb24f8990ceef590ca933eb1dd45ccefe9ee5788eff211948`;
  and
- [`automatic_linear_selector_v2_m6_v3_inspection1_result_20260722.json.manifest.json`](automatic_linear_selector_v2_m6_v3_inspection1_result_20260722.json.manifest.json),
  SHA-256
  `601f069896cdf664fcab470abe8c3643f0c0aacf5f79572a6663e304af3d7782`.

## Decision

The frozen M6 disposition is `advance`. This spends inspection 1 and
authorizes only the contract's next spent-development step: Protein
attribution with constant, automatic, and explicit-linear arms. It creates no
shipping, default-on, fresh-confirmation, TabArena, or lockbox authority.
