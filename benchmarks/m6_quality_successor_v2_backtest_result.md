# M6 quality successor v2 backtest result

Run once on 2026-07-21 from clean detached checkpoint `e5ff51d`.

## Decision

`m6-quality-successor-v2` passed its structural backtest and may rank or kill
quality mechanisms on its frozen spent general-development slice. It may not
rank speed mechanisms, authorize shipping/default changes, or access later
evidence.

Both unchanged, predeclared historical replays agreed:

| Replay | Expected | Observed | Readout |
| --- | --- | --- | --- |
| Combined B1+B2 ensemble-v3 | Advance | Advance | `0.965513×`, 13/13 wins, worst `0.991888×` |
| 3% linear-leaf selector | Kill | Kill | `0.989264×`, 2/14 wins, worst `1.000000×` |

V2 makes no blindness claim because these outcomes were already visible after
the invalid v1 calculation. Its contribution is structural: the result binds
an immutable rule module, exact execution wrapper, exact comparison runner,
and paired-evidence foundation. The future wrapper explicitly records the ten
dataset ids, medium size, three seeds, two weight modes, four threads, and
three repeats.

## Binding

- Result:
  [`m6_quality_successor_v2_backtest_result.json`](m6_quality_successor_v2_backtest_result.json),
  SHA-256 `6880c679cd5f16aa61d13c2e57282e3f162769be87e478a6ddf18d8958c9cf57`.
- Contract checkpoint: `e5ff51d70a1db6e3ba7b50ba364fe1f3fa49d3e5`.
- Immutable rule SHA-256:
  `b80520a77f3b99f14209a89535b32ca3437141d9251353618db7f1151484cb55`.
- Contract SHA-256:
  `9458997b392ec9b560aca70f1dc7e3be8897c67d1145795a3e9f907923e35884`.
- Exact execution wrapper SHA-256:
  `3acdc64c7b8563def0fe01a3d4b14b65985a0390ad5fae9a9e37a07ba00061c2`.
- Comparison runner SHA-256:
  `0fcd849a13c0348c4c6802556d9a3d3b9f1d5b02c8c47a4e82c3e744f358760f`.
- Paired-execution foundation SHA-256:
  `63c63d4f0b7c6f649b7155325ee064faf6e5981094ed3cb79ac91b6b8fefedf9`.

V1 remains immutable and non-ranking under its separate invalidation record.
M6 v3 remains terminal. Each v2 mechanism inspection is numbered, create-only,
and spent, and any advance still needs every downstream plan and owner gate.
