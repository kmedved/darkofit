# Wave 1 G-M portfolio decision

_Published 2026-07-20 after M1, Q0, and M3a._

## Decision

Fund **one private Track-B ensemble-v3 mechanism prototype**. Close the
quantized-histogram prototype. Do not run both sequentially.

This funds B0 plus the smallest separable B1/B2 attribution prototype:
without-replacement member sampling and a named member policy. It does not
fund a default change, a public API, parallel members, a broad-panel
campaign, or fresh/lockbox access.

The current DarkoFit bootstrap ensembles remain closed as quality routes and
preserved only as explicit opt-ins. The funded candidate is a new mechanism,
not a larger or retuned rerun of the failed group8 candidate.

## Evidence used

The decision binds to:

- M1 [`m1_wave1.json`](m1_wave1.json), SHA-256
  `74fd4c9c85948a4c19664a57534e19be3efb0483c78c13767c2521194626eb7a`;
- Q0 [`q0_wave1_profile.json`](q0_wave1_profile.json), SHA-256
  `9111f14ae4d0d89e122f541b53f85c76c6bd5e76f4fa781c69039c1020c04e1c`;
  and
- M3a [`m3a_wave1.json`](m3a_wave1.json), SHA-256
  `c811c8b04cbbaff6edb8226d7e8f5dbac3f9229adf18c3f8b658129ba7fc459a`.

All three use the clean post-H1 DarkoFit package pin
`726e5d8e6131c580bce948db833a5007d0692dca` and the exact ChimeraBoost
source `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`.

## Q disposition: close

Q0 found a credible DarkoFit hotspot: its frozen conservative projection was
`0.867242`, implying 13.28% lower fit time and clearing the 10% local screen.
M1 did not establish the conjunctive donor signal: ChimeraBoost
quantized/float fit was `0.903595`, narrowly missing the predeclared `0.90`
materiality bar. Integrity and timing stability passed.

The rule is not relaxed for a near miss. Q is
**close / do not fund**. Re-entry requires a new pinned donor
characterization that clears the declared materiality boundary or a distinct
DarkoFit-specific causal result that justifies a newly frozen profile and
budget. The current Q0 result alone is insufficient.

## B disposition: fund a different mechanism, not the failed implementation

M3a's frozen DarkoFit group8 gate failed decisively and correctly stopped
repeat timing:

- player-disjoint RMSE ratio `1.025482`;
- season-clustered p95 `1.032391`;
- held-team ratio `1.016048`;
- cold-player ratio `1.015661`; and
- all three season aggregates worse.

Every shipped DarkoFit ensemble diagnostic was worse than its single:
row5 `1.023115`, row8 `1.018335`, group5 `1.032361`, and group8 `1.025482`
on player-disjoint sports RMSE. DarkoFit row8 also lost the selected general
medium slice at `1.019556`, winning only 2/6 cells. These paths stay closed;
no threshold, member count, or spent coordinate is retuned.

The predeclared ChimeraBoost primary arm supplied a separate, unusually
consistent donor signal:

- ensemble8/single player-disjoint sports ratio `0.950230`;
- every one of the nine sports cells improved, with worst ratio `0.996298`;
- all three season aggregates improved; clustered p95 `0.962517`;
- held-team and cold-player ratios `0.977973` and `0.977935`; and
- selected general medium-slice ratio `0.947797`, with 6/6 wins.

The float ensemble reproduced the sports gain (`0.950577`), so the result is
not a quantization artifact. ChimeraBoost's quality mechanism differs
materially from DarkoFit's current implementation: 0.8 row subsampling
without replacement plus ensemble-specific member defaults. Those are
exactly the mechanisms B0/B1/B2 were written to isolate.

This is not a post-hoc rescue of group8. M3a's formal disposition remains
**close / preserve current opt-in**, its failed gate remains immutable, and
no repeats are run. G-M is the plan's portfolio judgment across all
predeclared arms. It funds a new private mechanism whose candidate, protocol,
and evidence identity must be separate.

## Funded scope

The next authorized work is:

1. Finish B0's compatibility contract for explicit row and group
   subsampling without replacement, a declared fraction, group-disjoint OOB,
   and a named member policy whose explicit user parameters always win.
2. Build the smallest private, sequential B1/B2 attribution prototype.
   Separate sampling-only, member-policy-only, and combined arms. Add the
   group-safe analog for entity workloads.
3. Prove deterministic sampling, group/OOB disjointness, weights,
   serialization, fitted metadata, failure behavior, and existing-bootstrap
   non-regression before comparative scoring.
4. Use mechanism-specific synthetics and the frozen M5 invariants first.
   M6 v3 remains non-ranking after its terminal backtest failure. Any sports
   or selected general development result is spent and descriptive.
5. Freeze M3b before inspecting prototype outcomes. Preserve the current
   public ensemble semantics and expose nothing publicly unless the
   attributed quality mechanism survives.

B3 parallel members are explicitly deferred. ChimeraBoost's parallel
ensemble used `2.82x` single-model fit time, `4.01x` prediction time,
`5.90x` model bytes, and `6.16x` sampled aggregate process-tree RSS on the
M3a workload. Quality attribution comes first; parallelism gets a separate
fixed-total-CPU and memory decision only after a quality winner exists.

## Portfolio consequences

- **Funded now:** B0 plus one private sequential B1/B2 attribution
  prototype.
- **Closed now:** Q prototype and extensions of the current DarkoFit
  bootstrap/member policy.
- **Deferred:** B3 parallelism, M2, M4, TabArena, fresh confirmation, and
  every default change.
- **Unaffected:** M5 remains a non-ranking drift sentinel; M6 remains
  non-ranking; C, X, and the T7b quality levers remain in Track I behind the
  one funded prototype.

No default, public option, release claim, or cross-season generalization is
authorized by this decision.
