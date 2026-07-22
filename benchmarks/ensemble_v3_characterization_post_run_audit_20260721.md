# Ensemble-v3 characterization post-run audit

_Dated, create-only correction record. The frozen v1 contract, raw artifact,
generated result, and interpretation remain byte-identical._

## Findings

1. The protocol promised an equal-case geometric mean for the paired
   peak-minus-start RSS ratios. The result retained every underlying value but
   omitted that aggregate. Recomputed from the immutable result, the four case
   medians are `2.995536x`, `2.708978x`, `3.003899x`, and `4.649776x`; their
   equal-case geometric mean is **`3.262867x` v3/single**. This sharpens the
   memory-cost interpretation alongside the already-reported `1.135581x`
   absolute process-tree peak-RSS ratio and `14.3–67.6 MB` v3 deltas.
2. Nine of 144 prediction intervals missed the declared `0.75 s` floor. The
   minimum was `0.006492584 s`: an anomalous `0.135360458 s` first warm call
   selected only eight formal calls, which then averaged `0.000811573 s` each.
   All misses were DarkoFit single at 8,192
   rows; all intervals at 65,536 rows and above met the floor. The immutable
   full-grid aggregate remains descriptive, but it is **not timing-decision
   eligible or a prediction certificate**. No favorable subset is recomputed.
3. The v1 contract loader verified the contract identity, frozen flag,
   execution grid, and bound-file records, but did not independently assert the
   `schema_version`, `outcome_blind`, `quality_uncertainty`, or `claims` fields.
   The committed contract contains the intended values and the completed raw
   and result artifacts bind its exact SHA-256, so this did not corrupt the
   completed evidence. It does prohibit reuse of the v1 harness.
4. Two unexercised failure paths reinforce that retirement: create-only writers
   can leave a partial target after a write failure, and RSS teardown can mask a
   primary fit exception if final telemetry sampling also fails. The completed
   run had neither a write failure nor an RSS sampling error.

## Disposition

Preserve all v1 artifacts without amendment or rerun. Retire the v1 harness.
Any successor must use a new identity, validate every declarative contract
field, guarantee partial-write cleanup, preserve primary exceptions during
telemetry teardown, and choose prediction loop counts from stabilized warmups
or fail closed when the formal duration floor is missed.

Machine-readable record:
[`ensemble_v3_characterization_post_run_audit_20260721.json`](ensemble_v3_characterization_post_run_audit_20260721.json).
