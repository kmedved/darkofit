# Automatic linear-selector v2 M5 check

Run once on 2026-07-22 from clean, published candidate commit
`a53d4bf543534678189d87d88dcad87dd2a8bd8f`, against the exact frozen M5
control `726e5d8e6131c580bce948db833a5007d0692dca` and hash-bound baseline.

## Result

The candidate passed the non-ranking M5 diversity sentinel. All 38 fresh
workers completed across 19 paired cells and nine domains. Every row produced
finite task-appropriate metrics, valid fitted/thread metadata, and an exact
safe-NPZ prediction round trip. Both earned classification floors passed, and
the frozen analyzer reported no baseline drift and no advancement block.

The candidate and control behavior fingerprints differed in one of the 19
cells: noisy numeric regression seed 0 had a candidate/control primary-loss
ratio of `1.004434950`. The other 18 paired cells were behavior-identical.
This difference is within M5's invariant/drift envelope and is development
telemetry for M6, not an M5 ranking result.

The candidate/control median paired ratios were `1.031226` for fit,
`1.007107` for prediction, and `1.002855` for peak RSS. The maximum fit ratio
was `7.734942`, reflecting the automatic selector's extra internal fits on a
small eligible coordinate. M5 timing ratios are deliberately non-ranking and
are not portable performance claims.

The create-only raw artifact is
[`automatic_linear_selector_v2_m5_check_20260722.json`](automatic_linear_selector_v2_m5_check_20260722.json),
SHA-256
`1c765589ed303432d87009ca0330db8dcf35e3651fbd9b93d2f8bc576f9e494a`.
The frozen baseline SHA-256 is
`0971e06d4ed307d352d75e1e6400b849c0001b5e11f40243173d7080b6c5859d`.

## Decision

M5 does not rank or accept mechanisms. Its correctness and drift obligations
passed, so the exact candidate identity remains eligible for its already
authorized M6 quality-successor-v3 inspection 1. No shipping, default,
fresh-confirmation, TabArena, or lockbox authority is created.
