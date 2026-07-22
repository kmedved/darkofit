# M6 quality successor v3 backtest result

_Executed once on 2026-07-22 from clean, published harness commit
`f3d19ebb4d9306e278a52534a7856650675d1166`._

## Decision

`m6-quality-successor-v3` passed its Phase-F-corrected historical backtest and
is the only M6 rule with forward quality-development ranking authority. It may
rank or kill a spent mechanism inspection on its exact medium general slice.
It cannot rank speed work, authorize shipping or default changes, or access
fresh confirmation, TabArena placement, or lockbox evidence.

All three outcome-known, predeclared replays agreed:

| Replay | Audit role | Expected | Observed | Aggregate | Worst group | Worst LOO |
|---|---|---|---|---:|---:|---:|
| Combined B1+B2 ensemble-v3 | Surviving known advance | advance | advance | `0.965513` | `0.991888` | `0.968329` |
| Native ordinal C2 | Surviving known kill | kill | kill | `0.992755` | `1.317510` | `1.090069` |
| 3% linear-leaf selector | Abolished-verdict tripwire | advance | advance | `0.989264` | `1.000000` | `0.998504` |

The selector replay is not new evidence or a retroactive product decision. It
proves only that the v3 development rule no longer reproduces v2's abolished
win-count kill. The selector still needs its separately authorized new
campaign and full automatic-policy Tier-D path.

## Integrity and no-rerun record

The backtest runner wrote exactly one create-only result. Its own terminal
payload reports `backtest_complete=true`, all three agreements true,
`candidate_ranking_eligible=true`, and `rerun_authorized=false`.

The surrounding shell command attempted to assign the runner's exit code to
zsh's reserved read-only variable `status` after the runner returned. That
post-run wrapper line exited nonzero, but it neither entered the Python runner
nor changed its already-fsynced result. The result was inspected in place and
the runner was **not rerun**.

V1 and v2 remain immutable. V3 supersedes only v2's forward ranking authority,
as recorded in
[`m6_quality_successor_v2_phase_f_supersession_20260722.md`](m6_quality_successor_v2_phase_f_supersession_20260722.md).

## Bindings

- Result SHA-256:
  `35cc54acfeb7de7950966445ed8248654f945072e5e5900e3333fff4b15129b6`.
- Phase F audit:
  `3717a080030788ded9fa12101dfad7e1b87ac811f517f1b1e1e16fb0fa35769f`.
- Contract:
  `1fedb2d2d2e043f56c8547fd67bf32ef028f98866f7455c05c2e8fa6c9d0e2b3`.
- Rule:
  `2415c7a7bde2bed23283067fdfe200892c15cf1c70d869153cc9cade81f9694c`.
- Execution runner:
  `950c3867f387112a65a5dd103f830cce71f7e74af42c5f5208499e787e609d39`.
- Backtest runner:
  `bcb733bdefd36fe4e6052f91f3453f42f7a21dd45011bd5705e7c91a098dd019`.
- V2 supersession:
  `24a755b8696256f76b7e4aaac55df3827e0e47ddd4506fdde34ee2dde04a7ef9`.
- Positive / negative / retired-selector artifacts:
  `99d693063c46a0708eb45a704af0b46611fa8ed89dbe4d6469b47c7cd4a27c64`,
  `7aeb83131bb7604a3eaabc2789f048d40dabb58791b6ab6aad0ac26f0f0f566f`,
  and
  `4dc158ec4fd11cf29a5822dc2a09aa76715ce9446773673fa9a2828da1b71a7d`.

The machine-readable source of truth is
[`m6_quality_successor_v3_backtest_result.json`](m6_quality_successor_v3_backtest_result.json).
