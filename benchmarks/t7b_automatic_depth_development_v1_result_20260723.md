# Automatic-depth development benchmark — 2026-07-23

This is **development evidence**, produced under `SHIP_RULES.md`. It is not a
holdout result and does not by itself authorize a default change.

The paired 32-lineage, three-coordinate benchmark completed all 192 arm rows
with integrity passing. The unchanged automatic-depth candidate
`41e948f0c53b1d124e16071a7fa66eba47d084d3` was compared with control
`e23d2b164f10374b1c0e02521c33fc96d48980da`.

## Result

- Equal-lineage RMSE ratio: **0.996860**
- Lineage-bootstrap upper ratio: **0.999869**
- Leave-one-most-favorable-lineage-out ratio: **0.998192**
- Worst lineage ratio: **1.016344**
- Low-density panel geomean: **0.994098**
- High-density panel geomean: **1.000000**

The candidate resolved depth 4 on all 51 low-density coordinates. All 45
historically labeled high-density coordinates resolved depth 6 from the
actual post-validation fit population and were behavior-identical to control.
This is the intended correction to the failed historical harness: panel labels
no longer override the policy's recorded fitted resolution.

Cost telemetry favored the candidate: fit `0.807402×`, prediction
`0.943500×`, RSS `0.991148×`, and mean RSS delta `-15,380,480` bytes. Timings
were paired on the same machine but remain secondary telemetry.

## Decision

The development effect is small but clear: its bootstrap upper bound and
leave-one-out sensitivity both remain below parity, integrity is complete,
and the changed branch improves without a large tail regression. Proceed to
the deliberate CTR23 plus newest-untouched-sports-season ship-check. Do not
tune the candidate from holdout results.

## Artifact hashes

- Preflight: `cfc94b9c57f86bc30b3654052490406c027b292002e27a2b10c0f3f441770334`
- Launch: `987f71bb45f19fa0a76bcb91b0478760eb8e5ad2a74b377f98f5088a5dc18b2d`
- Raw: `db7b96cbeec9ee21f1696453e16792560d57a6d6fe9ab5c7eae0f1fded19b30e`
- Result: `7e92d584b4adb8a96675d1f116a35682ddc2d4e2adc43051eadbca316d5c3307`
