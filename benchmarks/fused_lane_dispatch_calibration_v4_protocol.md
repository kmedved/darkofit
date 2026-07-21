# Fused-lane dispatch calibration execution protocol v4

_Prospective, outcome-blind successor to the v3 execution freeze. No formal
calibration or validation worker ran and no timing outcome was opened under
v3._

## Reason for supersession

The published v3 execution contract has SHA-256
`c55ee50fccda5b9ba24e004ae8a27285e4db92e52a9c17a668bc1b417b0fa648`.
An independent pre-authorization review confirmed that v3 had closed the
actual-builder-counter defect, but found four remaining correctness or
evidence-contract defects:

1. a directly invoked worker could use the canonical authorization after the
   owner gate opened, permitting partial coordinate inspection outside the
   all-or-terminal parent execution;
2. single-wrapper and public-ensemble `oblivious_kernel` headers could diverge
   from their fitted boosters after parameter mutation or archive tampering;
3. ensemble class-safe retries required positive partition mass for observed
   classes whose full-data sample-weight mass was zero; and
4. calibration used the C-contiguous input for leaf routing while production
   multithreaded scalar fitting uses the same Fortran array for routing and
   histogram construction.

These were found before v3 authorization. V3 has no authorization, raw,
terminal, analysis, threshold, or validation artifact, and no formal worker
started. V3 therefore remains immutable and is superseded without opening or
responding to timing evidence.

## Gate and correctness repairs

V4 retains v3's worker-side contract, authorization, source, coordinate, and
environment checks, and adds a parent-only execution capability. Immediately
before every worker spawn, the formal parent revalidates the canonical
contract, canonical owner authorization, clean source pin, exact frozen
coordinate, and absence of both raw and terminal artifacts. It then sends a
one-use nonce and those bindings through an inherited anonymous pipe. Worker
functions reject a missing, non-pipe, stale, or mismatched capability before
case generation or any DarkoFit import. Possessing the authorization artifact
alone is insufficient to call a formal worker.

Single-wrapper saves derive `oblivious_kernel` from the fitted booster and
validate the emitted header against it. Safe load rejects a contradictory
wrapper value. Public ensemble saves derive the outer value from unanimous
fitted members, while public ensemble load requires the outer wrapper, every
member wrapper, and every member booster to agree. Private schema-3 constructor
binding remains unchanged and complete.

Class-safe ensemble sampling continues to require every observed label on both
sampled and OOB sides. With sample weights, positive mass on both sides is
required only for labels whose full-data mass is positive, matching the
ordinary validation-split rule.

Calibration preserves the original C-contiguous generated input for its
canonical dataset fingerprint and primary builder argument. Its histogram and
leaf-routing arguments now share the same Fortran view, exactly matching the
production multithreaded scalar layout. Both lanes use the identical repaired
layout.

## Scientific invariance and authority

Every scientific and operational clause in
[`fused_lane_dispatch_calibration_protocol.md`](fused_lane_dispatch_calibration_protocol.md),
its [`v2 successor`](fused_lane_dispatch_calibration_v2_protocol.md), and the
[`v3 gate-repair protocol`](fused_lane_dispatch_calibration_v3_protocol.md)
remains binding except where the parent capability and production routing
layout above are stricter. V4 changes no generator, coordinate, seed, lane,
warmup count, repeat order, threshold candidate, tie rule, acceptance limit,
or downstream authority. It carries forward the exact v3 worker environments.

V4 uses execution identity `calibration_v4` and unique create-only
authorization/raw/terminal/analysis paths. Freezing it is not authorization to
run it. A qualifying calibration still permits only a separately committed
threshold artifact, new candidate source pin, and separately frozen validation
execution. Failure still closes Wave 4; B remains closed, Q remains sequenced
after any retained dispatch, and the next mechanism slot remains quality-first.
