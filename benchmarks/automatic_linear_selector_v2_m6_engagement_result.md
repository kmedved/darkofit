# Automatic linear-selector v2 M6 engagement result

Run once on 2026-07-22 from clean, published companion harness
`56a66700a354dfe90d4cfd72d4254a7d8e22b351` against clean, published
candidate `a53d4bf543534678189d87d88dcad87dd2a8bd8f`, before M6 inspection 1.

## Result

All 60 exact M6 quality-successor-v3 cells completed in fresh four-thread
workers. The artifact contains no primary quality metric, prediction, benchmark
timing, RSS value, or acceptance decision.

Selector disposition by cell:

| Dataset class | Cells | Result |
| --- | ---: | --- |
| Six classification datasets | 36 | `classification_not_applicable` |
| `diabetes_resampled` | 6 | exact constant fallback, `below_min_samples` |
| `friedman_numeric` | 6 | eligible; `margin_below_threshold` |
| `wide_numeric_reg` | 6 | eligible; `margin_below_threshold` |
| `categorical_reg` | 6 | eligible; `margin_below_threshold` |

No M6 cell selected linear leaves. Across the 18 eligible regression cells,
the observed relative validation margins ranged from `-0.056974` to
`0.028148`, below the frozen `0.03` threshold in every case. This is mechanism
engagement provenance, not a quality verdict.

The create-only raw artifact is
[`automatic_linear_selector_v2_m6_engagement_20260722.json`](automatic_linear_selector_v2_m6_engagement_20260722.json),
SHA-256
`6120efe99421403de1d64e7bff594bcf51d3aba18d8851de2e9f728860952405`.

## Decision

The observability obligation is complete. The separately frozen M6 v3
inspection 1 must still execute and apply its aggregate, worst-dataset, and
leave-one-dataset-out quality rule. This companion cannot advance or kill the
candidate and creates no shipping/default/fresh/TabArena/lockbox authority.
