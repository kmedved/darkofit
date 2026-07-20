# M6 historical-backtest result

Run 2026-07-20 from frozen executor commit `59f7613`.

## Decision

M6 failed its historical-backtest prerequisite and remains non-ranking.
Do not rerun or relax the same declared replay.

The exact fused variable-Hessian replay completed first. The frozen M6
analyzer classified it `kill`, disagreeing with its known `advance` verdict.
That disagreement alone is terminal for M6's ranking eligibility.

The exact packed-router replay then stopped before loading data or fitting a
model: its historical runner hard-requires 18 Numba threads, while the current
machine/runtime permits at most 14. It is therefore `lacks_power` on this
machine, not an agreement. The selector replay was not opened after the
terminal prerequisite failure.

Three earlier launches failed before any replay outcome because of source-map
and historical namespace-import wiring. They are recorded in
[`m6_historical_backtest_invalid_attempt.md`](m6_historical_backtest_invalid_attempt.md).
The fourth launch is the only outcome-bearing attempt.

## Evidence boundary

- Failure artifact:
  [`m6_historical_backtest_failure.json`](m6_historical_backtest_failure.json),
  SHA-256
  `18b902e6099a4686b8eda71fac9ac327a0b5243872b80b5da79c5e01e5e2c201`.
- Frozen executor SHA-256:
  `80e451bc4a948794e169eb52cf0a4f61ec5b2b7e5a21aebd74bb20f7a02acdc8`.
- Fused source: `1016e7e8d70c403a70feab7762de8837ea8fd09c`.
- Packed source: `e961bcc2ea64706169641722b5935f9f31402fa3`.
- Selector source: `29bd30cdcf476139c30efe4e09773ca812ba443f`.
- ChimeraBoost 0.15 source:
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`.
- Basketball cache SHA-256:
  `43af3be46074da9130a757aa1db643e207e2c0ea5405d2086d698f85555fbcb2`.

The combined executor intentionally held replay shards in a temporary
directory and writes only after the full declared subset. The packed
preflight exception therefore removed the fused raw shard. The emitted frozen
analyzer disposition is preserved in the failure artifact, with this missing
raw shard disclosed. It is not reconstructed by rerunning the spent replay.

This failure does not invalidate M5 or M6's release-anchor characterization.
It means only that M6 has not earned authority to rank new mechanisms.
