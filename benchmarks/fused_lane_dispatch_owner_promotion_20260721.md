# Owner promotion of the closed fused-lane dispatch candidate

_Recorded 2026-07-21 after the immutable calibration-v4 result was published._

## Decision

The owner directed DarkoFit to **promote** the calibration-selected
`scan_work` threshold `1048576` into the existing `oblivious_kernel="auto"`
product surface. This is an owner product decision, not a revision of the
campaign verdict.

For functionally and automatically eligible scalar CatBoost-mode fits on
macOS arm64, new fits resolve once per fit as follows:

- `scan_work < 1048576`: fused lane;
- `scan_work >= 1048576`: unfused lane.

The measured platform and shape envelope, static tie rule, persisted dispatch
metadata, safe-load validation, and explicit `"fused"`/`"unfused"` overrides
remain unchanged. Fits outside the automatic envelope continue to use the
fused lane. Archives retain the threshold and decision used when they were
fit, so models saved before this promotion retain their original
`threshold_unavailable` decision on load.

## Evidence status and non-claims

The immutable
[`calibration-v4 analysis`](fused_lane_dispatch_calibration_analysis_v4.json)
still has SHA-256
`c47314191eaec43e6ceb5fa7a2eca870b7af2308cc736dae23c12b9735f3bf9b`,
`qualifies=false`, and disposition `close_dispatch_campaign`. Its 30/30
behavior-exact cells, both-lane selection, and worst selected/current-fused
ratio `1.0` support the bounded product risk. Its six unstable cells and
`0.973846` selected/current-fused geomean against the frozen `<=0.970000`
require the following limitations to remain explicit:

- this promotion does not create or imply a validated 3% speed claim;
- it does not establish portability beyond the encoded macOS-arm64 envelope;
- it does not authorize a validation rerun, threshold relaxation, release,
  fresh-data access, or lockbox access; and
- it does not reopen the closed campaign or alter any immutable artifact.

The owner accepts the marginal and timing-stability risk in exchange for the
behavior-exact measured median improvement. Users can force
`oblivious_kernel="fused"` as an immediate rollback for a particular fit.
The next mechanism slot remains quality-first.
