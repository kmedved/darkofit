# Fused-lane dispatch calibration execution protocol v2

_Prospective, outcome-blind successor to the v1 execution freeze. No formal
calibration or validation worker ran and no timing outcome was opened under
v1._

## Reason for supersession

The published v1 execution contract has SHA-256
`3d7f8a653a71d6a9712f57f51bb01421765b42fcd105902f1fb0c6a611f7712d`.
Its first post-publication GitHub library matrix exposed one deterministic
test-portability bug: the product test intended to exercise the
`rows_outside_envelope` fallback used the ambient host platform. It therefore
observed `unsupported_platform` on Linux, although it observed the intended
row-bound reason on the frozen Darwin development host.

The selector and both reason codes were correct. The test now pins Darwin,
arm64, and the logical CPU count explicitly before fitting, as the nearby
pure-selector reason matrix already did. This changes no product code,
campaign generator, coordinate, seed, lane, timing region, repeat order,
threshold candidate, acceptance limit, or downstream authority.

Because that test is hash-bound by the v1 execution contract, v1 is preserved
and superseded rather than edited or silently reused. Its authorization, raw,
terminal, analysis, threshold, and validation artifacts never existed.

## Binding rules

Every scientific and operational clause in
[`fused_lane_dispatch_calibration_protocol.md`](fused_lane_dispatch_calibration_protocol.md)
remains binding. V2 additionally:

- uses execution identity `calibration_v2` and unique create-only
  authorization/raw/terminal/analysis paths;
- binds the immutable v1 execution contract and this successor protocol;
- requires the separate owner authorization to repeat the v2 execution
  identity as well as its contract hash, campaign, phase, and source; and
- keeps `execution_authorized=false`, `outcomes_opened=false`, and every
  downstream authority false at freeze time.

Freezing v2 is not authorization to run it. A qualifying calibration still
permits only a separately committed threshold artifact, new candidate source
pin, and separate validation execution freeze. Failure still closes Wave 4;
B remains closed, Q remains sequenced after any retained dispatch, and the
next mechanism slot remains quality-first.
