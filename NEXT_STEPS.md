# NEXT_STEPS — gate retraction, ensemble-v3 revisit, and gate reform

> **Status:** binding owner decision record, adopted 2026-07-21 at
> `671f2e0`, rebased onto the completed
> dispatch outcome (calibration v4 closed `qualifies=false`; the owner then
> promoted the selected threshold by separate product decision — see
> [`benchmarks/fused_lane_dispatch_owner_promotion_20260721.md`](benchmarks/fused_lane_dispatch_owner_promotion_20260721.md)).
> This draft was authored across that boundary and was untracked while the
> calibration ran, so nothing in it claims prospective, outcome-blind
> authority; §4.7 is a forward governance proposal only.
> **Binding scope:** the adoption commit enacts the §1–§2 gate
> retraction and disposition supersessions and the §4.9 design rules.
> Everything executable is authorized only per the §6 matrix — nothing
> else is authorized by prose elsewhere in this document. Frozen records
> named below stay immutable; this document changes forward decisions
> only. The create-only
> [`owner adoption note`](benchmarks/gate_reform_owner_adoption_20260721.md)
> clarifies that the adoption commit's “planning-only” wording described
> its code-free diff; it did not revoke the future work explicitly authorized
> by §6.

## 1. Owner decision: the archive-size gate class is retracted

Saved-model disk size is not a shipping concern for this library unless it
reaches absurd absolute magnitudes (order hundreds of megabytes per model).
The objectives are quality, training speed, inference speed, and peak
runtime memory. The `median_archive_to_single_at_most: 4.0` gate in the M3b
campaign — and the ratio denomination of size gates generally — was a
design mistake:

- The models in question are **50–300 KB (single)** and **0.35–1.6 MB
  (8-member ensemble)**. Median: single 91 KB, prototype ensemble 515 KB.
- The ensemble DarkoFit **already ships** has a median archive of 791 KB
  (~8.7× a single). The gate blocked an additive candidate that is **35%
  smaller** than the existing ensemble surface it would sit beside.
- No user-visible harm exists at these magnitudes. The gate enforced a
  ratio, not a harm.

**Supersession, not amendment.** The M3b r3 and B-archive v1/v2 frozen
records stay unchanged *as history* — they were executed correctly under
the rules as written, and rewriting them would spend the credibility every
future gate depends on. But their forward-looking dispositions are
**superseded by this decision**, specifically:

- `close_b1_b2_preserve_existing_opt_in` → the combined B1/B2 mechanism is
  **reopened and promoted to a public ship path** (§4.2);
- "no canonical serializer authorized" → the size-model result is
  reclassified from failed prerequisite to optional future work (§2 —
  note it was a size *simulation*, not an implemented serializer);
- "do not continue B3" → the stop rule is lifted; B3 becomes **eligible
  for a fresh authorization decision** (§4.3). Its deferral rested
  primarily on quality-first sequencing and aggregate RSS — bytes was only
  one cited cost — and the quality prerequisite is now satisfied.

Every piece of work rejected because of this gate is reopened by this
document. Nothing rejected on quality, power, or stability grounds is.

## 2. What the mistaken gate actually decided (full inventory)

A sweep of every campaign record and contract found the archive-ratio gate
bound in exactly the B family. Nothing else in the program's history closed
on a size gate (all other closures were quality, power, stability, or speed
bars — see §3).

| Decision | Role of the gate | Revised disposition |
| --- | --- | --- |
| **M3b r3: combined B1/B2 closed** (`close_b1_b2_preserve_existing_opt_in`) | Sole failed check. Quality (13/13 wins vs single, `0.9655` pooled, `0.9611` player-disjoint sports cold), predict, RSS (`1.069×`), and value all passed. | **Reopen as a product decision:** ship as an explicit, additive Tier-E opt-in (§4.2) alongside the existing bootstrap surface. Tier-E policy requires correctness + honest characterization, not binary bars. |
| **Wave 3 B-archive campaign** (exact-factoring size model, `6.03×→4.15×`, closed) | The whole campaign existed only to serve the 4.0× gate. | Moot as a blocker. The result was a **non-loadable size simulation**: the ~31% exact shrink is real as a bound, but an actual serializer still needs a new format and correctness contract. Optional, unscheduled Track I entry. |
| **Planned member-count-reduction campaign (5/6 members)** | Proposed solely to fit under the gate. | **Cancelled as a size measure.** Eight members is the **only evaluated** combined-v3 count (not "measured-best"); member-count exploration may return later purely as a quality/speed question. |
| **B3 parallel members: "do not continue B3" stop rule** | Deferral cited `5.90×` bytes among four cost figures; primary rationale was quality-first sequencing plus `6.16×` aggregate RSS. | **Stop rule lifted; fresh authorization required** (§4.3): a new fixed-CPU / process-tree-memory decision, not a mechanical reopen. |
| **Track I entries** (B-archive serialization; "B revival requires a distinct mechanism") | Written under the gate's authority. | Annotate: serialization entry demoted to optional future work; the combined-arm revival happens via §4.2, not via a new size mechanism. |
| **Future campaign templates** carrying "model-size budgets" | Habit inherited from M3b; `SHIPPING_POLICY.md` itself mandates only fit/predict/RSS budgets. | Size leaves the gate vocabulary entirely: corruption detection uses **schema-derived checks** (arrays validated against fitted metadata at save/load), any absolute ceiling must be independently harm-justified (the owner's stated harm begins near hundreds of MB), and size ratios are **telemetry, never gates**. |

## 3. What this does NOT reopen

The retraction is narrow. These closures stand because they failed on
quality, power, stability, or speed — not size:

- B1-alone and B2-alone arms: under this document's own regime they remain
  unselected because the **combined arm Pareto-dominates them** (better
  quality at comparable cost) — not because the old binary bars retain
  force. (Historically they also failed frozen value checks; only the
  combined arm was size-blocked.)
- The shipped bootstrap ensembles (group8/row8/group5/row5 — lost to the
  single on quality in M3a).
- Q / quantization as previously ruled (its funding rule is amended
  prospectively in §4.6; the historical closure stands).
- Panel 3 (power), the linear-leaves selector, C2 native ordinal, the S
  early-stopping candidate, E2 (all pre-existing non-size closures).
- The fused-dispatch calibration campaign verdict (`qualifies=false`) — the
  owner promotion is a separate, already-recorded product decision; this
  document neither revisits nor re-authorizes it.
- No fresh-confirmation or lockbox access is implied by anything here.
  M4 TabArena-Lite access is governed solely by the §6 authorization
  matrix (explicit owner sign-off at release time), not by its appearance
  in §4.5's planning text.

## 3b. Adjacent gate families reviewed (completeness)

The gate audit also covered two historical near-misses and the gates that
earned their keep, so this reform's scope is bounded on both sides.

**Healed by the two-tier reform — nothing to reopen, one measurement to
schedule:**

- **E2 large-n certification** (measured `1.2793×` against a frozen `1.30×`
  bar): a 27.9% speedup denied a certificate for wanting 30%. The claim
  class no longer exists (Tier-E publishes measurements with dispersion,
  never pass/fail certificates). Wave 1's M1 is the **newer current-pin
  answer** — different comparator, thread count, and machine, so not
  directly comparable — and no rerun of the old lane is warranted. Closed;
  nothing to do.
- **P2 predict certification** (failed two timing-stability checks **and**
  the minimum-duration gate, despite `0.805–0.987` no-slower medians): the
  certification stays closed. Prediction against 0.18 is not literally
  unmeasured — M1, the 0.18 sports diagnostic, and M3a all carry narrow or
  workload-specific current measurements. The dedicated repeat-series grid is
  now published under §4.5 as characterization rather than certification; its
  post-run audit keeps the small-batch duration misses explicit.

**Gates and disciplines affirmed at current strictness — this reform does
not touch** (the governing text is
[`benchmarks/SHIPPING_POLICY.md`](benchmarks/SHIPPING_POLICY.md), which is
incorporated here by reference; this list highlights, it does not replace):

- Panel 3's 80% power floor and design-time power/uncertainty requirements;
- Tier-D quality harm bounds, leave-one-out concentration, and the full
  Tier-D path for any default or automatic policy;
- contamination controls, fresh-confirmation boundaries, and sealed-lockbox
  discipline;
- behavior-exactness invariants and integrity/stability preflights;
- the no-rerun rule and create-only artifact discipline;
- source/runtime/hardware pins, paired fresh-worker timing, and
  equal-resource accounting;
- peak-RSS discipline — real memory is a real harm — with the hybrid
  denomination defined operationally in §4.3.

## 4. Sequenced next steps

Execution order (rationale in the subsections): **(0)** commit this
document as the supersession/authorization record → **(1)** freeze the
public ensemble-v3 API and archive-schema contract (§4.2), with the M6
successor build running in parallel (§4.8) → **(2)** implement and
validate sequential ensemble-v3 (§4.2) → **(3a, complete)** run the dedicated
prediction characterization (§4.5) → **(3b, separately gated)** run milestone
M2/M4 evidence only after its owner sign-off →
**(4)** ship v0.11 with the already-promoted dispatch and honestly
characterized ensemble-v3 → then **three distinct mechanism slots, one at
a time, each with its own entry and exit**: **(5)** T7b quality levers
(§4.4), **(6)** B3 as its own fixed-CPU/memory campaign (§4.3), **(7)** Q,
funded only by a measured local microprototype (§4.6).

**4.0 — Dispatch status (resolved; nothing to wait for).** Calibration v4
closed `qualifies=false` (30/30 behavior-exact cells, worst
selected/current-fused ratio `1.0`, but `0.973846` geomean against the
frozen `<=0.970000` and six unstable cells). The owner then promoted the
`scan_work` threshold `1048576` into `oblivious_kernel="auto"` on the
measured macOS-arm64 envelope, with explicit non-claims and the
`"fused"` override as rollback, under
[`fused_lane_dispatch_owner_promotion_20260721.md`](benchmarks/fused_lane_dispatch_owner_promotion_20260721.md).
That record is the promotion's sole authority; this document adds nothing
to it and takes nothing from it.

**4.1 — Land the decision record.** Commit this document (the §6 matrix
defines exactly what that commit authorizes), then one dated, create-only
owner decision note in `benchmarks/` stating §1–§3, plus the matching
edits: plan status line, B ledger row (Closed → reopened for shipping
under Tier-E), Track I annotations, and the §4.9 gate-design rules into
the plan's evidence-discipline section.

**4.2 — Prepare ensemble-v3 as an explicit opt-in candidate (target v0.11;
public ship separately gated).**
1. **Freeze the public B0 contract first — the private contract is not
   ship-ready.** It cannot distinguish omitted constructor parameters from
   explicitly supplied defaults, and it excludes presets, automatic tree
   mode, callbacks, external eval sets, refit, auto-learning-rate probing,
   and distributional losses. The public contract must define: explicit
   parameter-precedence semantics (sentinel defaults so explicit-user-wins
   is decidable), a full support matrix (each excluded surface either
   supported or documented-unsupported with a loud error), sklearn
   clone/`get_params` behavior, and a public serialization schema.
2. Build the release-candidate implementation behind a private, non-exported
   integration surface with the correctness suite already proven privately:
   deterministic sampling, group/OOB disjointness, weights, serialization
   round-trips (including the uneven-group fix from M3b attempt 2), failure
   propagation, and existing-bootstrap non-regression. Do not add public
   constructor parameters, exports, or public documentation until the public-
   ship row in §6 receives its separate owner sign-off.
3. **Completed:** publish the honest characterization **with uncertainty, per Tier-E
   policy** — point estimates and win counts alone are insufficient.
   Quality from the immutable r3 readout (13/13 vs single; sports view
   labeled player-disjoint cold-player within held teams; general view
   labeled seeded 75/25), with **season-clustered descriptive uncertainty
   for the nine sports cells** (three season clusters, cluster-bootstrap
   dispersion as in M3a) and dispersion plus leave-one-case-out
   sensitivity for the four general cases; never imply 13 independent
   datasets. The historical M3b attribution costs were fit `0.56×` the then-
   current ensemble, RSS `1.07×` single, and archive ~5.5× single (~0.5 MB
   typical) with the gate-retraction note; the current-source checkpoint below
   supersedes those figures for release planning. Additive surface; existing `n_ensembles`
   bootstrap semantics unchanged. No default change; no sports-safe claim
   beyond the panel's scope; eight members documented as the only
   evaluated recipe.
4. Docs page + CHANGELOG entry.

**Characterization checkpoint (2026-07-21):** the separately authorized
report-only phase is prospectively frozen as
`ensemble-v3-characterization-v1` against published DarkoFit `c5e66ef` and
the exact ChimeraBoost 0.18 pin `f14be60`. Its one complete run reproduced the
immutable M3b quality point estimates, added the predeclared clustered/general
uncertainty analysis, and measured the four-task by four-batch prediction grid
plus current fit/process-tree-RSS/safe-NPZ telemetry. Quality remained 13/13;
the candidate cost `6.14x` single fit time, `1.14x` process-tree peak RSS,
`8.13x` safe-NPZ bytes, and `6.21x` single prediction time. Nine small-batch
single-model intervals missed the duration target and remain disclosed; no
rerun or subset replaced them. A post-run audit supplies the protocol's omitted
peak-minus-start aggregate (`3.262867x` v3/single), marks the full prediction
grid non-certifying and non-timing-decision-eligible, and retires the v1 harness
without changing any frozen artifact. This checkpoint does not authorize items 4.2.4
or any §6 ship, milestone, or release row.

**4.3 — B3: a fresh authorization, operationalized.** B3 was deferred for
quality-first sequencing and aggregate memory, not only bytes; the quality
prerequisite is now met, so it gets its own small frozen campaign — new
decision, not a mechanical reopen. Speed is the objective. Required
accounting, defined so the tiny-denominator problem cannot recur:

- **scope:** aggregate parent-plus-worker process-tree peak RSS, fixed
  worker topology, fixed total CPU, deterministic member seeds,
  worker-failure propagation, equal-total-CPU timing vs the sequential
  control;
- **memory rule:** the candidate fails on memory only if it exceeds a hard
  absolute ceiling (declared in GB for this machine class), **or** exceeds
  both the declared ratio allowance and the declared absolute-delta
  allowance versus the sequential control. Either allowance alone cannot
  bind on a trivial base.
- Include ensemble predict-path characterization; batched member
  prediction is a candidate follow-on for inference speed.

**4.4 — Wave 5 quality slot: T7b levers.** Unchanged: `l2_leaf_reg` and
the samples-per-feature depth policy remain the nominated quality
mechanisms, per the anti-drift guard and the promotion record's own
"next slot remains quality-first" note.

**4.5 — v0.11 milestone measurements.** M2 (first broad-panel reading
against ChimeraBoost 0.18 — the calibrated-yardstick test) and the first
M4 TabArena-Lite placement ride the release, as already planned. Added by
this document, the **dedicated prediction characterization is complete**
against the exact 0.18 pin for DarkoFit single and the private ensemble-v3
candidate. It is a Tier-E repeat-series measurement, not a certificate: nine
8,192-row DarkoFit-single intervals missed the duration floor, so the full-grid
aggregate is descriptive and not timing-decision eligible; every 65,536-row-and-
larger interval met the floor. The create-only post-run audit retires that v1
harness and records the limitation without a favorable rerun or subset. M2 and
M4 remain independently owner-gated milestone work; this completed prediction
grid does not authorize either one.

**4.6 — Q funding revisit: strike the donor prong.** Q closed on a
conjunctive rule whose second prong conditioned DarkoFit's funding on
ChimeraBoost's implementation gain (`quantized/float ≤ 0.90`; observed
`0.9036` — a 0.0036 miss on someone else's codebase), while DarkoFit's own
Q0 profile *cleared* the pre-declared 10% worthwhileness budget with a
13.28% conservative projection. The rival-conditioned prong is retracted
prospectively as a design error: donor evidence may inform sizing, never
bind funding. Action, sequenced after the ship work as its own mechanism
slot (because the promoted dispatch changes the baseline): build a
**measured private local causal microprototype** — a packed-histogram
kernel at the Q0 hotspot — and evaluate it against the **post-dispatch
baseline**. Q1 funding requires that *measured* evidence to clear the
frozen worthwhileness budget; the stale 13.28% screening projection (a
`1.30×` assumed-prior calculation, not a measurement) may size the effort
but cannot fund it, and no donor condition applies. The frozen Q closure
stands as history; this supersedes its forward effect exactly as §1 does
for the archive gate. (§3's "as previously ruled" defers to this section.)

**4.7 — Behavior-exact acceptance: a forward governance proposal (no
retroactive force).** The dispatch calibration realized the predicted
pattern — a `0.0038` geomean near-miss on a frozen `0.970` bar — but its
original framing here ("if integrity and stability pass") did **not**
match what occurred: six cells were unstable, and the owner resolved the
situation by a documented risk-accepting promotion, which stands on its
own record (§4.0). Prospectively, for future behavior-exact mechanisms
(Tier-E engineering only — Tier-D automatic policies remain governed in
full by `SHIPPING_POLICY.md`):

- **no arbitrary minimum-speedup bar** — behavior-exactness removes the
  quality risk that materiality ratios exist to offset; but
- acceptance still requires: proven exactness on the declared envelope,
  bounded resource cost and implementation complexity, **stable direction**
  (instability is handled by narrowing scope or stability remediation,
  never by a bigger speedup number), a defined platform/shape envelope
  with loud fallback outside it, recorded per-fit resolution metadata, and
  an explicit user rollback.

**4.8 — Fund the M6 successor (start in parallel with §4.2).** The M6
backtest gate worked — it caught the analyzer mis-killing a known-advance
mechanism — but its terminal closure left the pipeline's cheap ranking
rung missing, taxing every future mechanism (T7b next) with earlier
sports/milestone evidence needs. Commission backtest v2 under a new
contract identity: a newly declared verdict subset containing at least one
known-advance and one known-kill, every replay executable within current
machine limits (the failed round's packed replay hard-required 18 threads
on a 14-thread host), and the same create-only artifact rules. Until it
passes, M6 stays non-ranking; nothing else changes.

**Completion note (2026-07-21):** `m6-quality-successor-v1` reproduced the
predeclared known advance and known kill, but a pre-activation audit found a
self-referential analyzer hash and missing repeat attestation. V1 remains
non-ranking and will not be rebound or rerun. Structurally corrected v2 keeps
the thresholds/subset unchanged, binds an exact repeat-attested runner, and
passed its own clean outcome-known backtest. Quality-development ranking is
eligible only under v2; M6 v3 remains terminal and no successor grants
shipping, default, speed-ranking, or evidence-access authority.

**4.9 — Standing gate-design rules (add to the plan's evidence
discipline):**
1. Every resource gate names the user-visible harm it prevents at
   plausible absolute magnitudes before it may bind. Prefer absolute
   budgets where users feel absolutes (disk, bytes); reserve ratios for
   quantities where the ratio is the product story (fit/predict time);
   peak RSS is declared hybrid per the §4.3 operational rule. A gate that
   cannot name its harm is telemetry, not a gate.
2. Bars on continuous measures state why the materiality band sits where
   it does; a bar whose only defense is being a round number is not
   defensible on a near-miss.
3. Conjunctive funding rules require each prong independently
   harm-justified, and no prong may condition on rival-implementation
   performance.
4. Behavior-exact Tier-E mechanisms follow the §4.7 acceptance rule: no
   arbitrary minimum speedup, with exactness, bounded resources and
   complexity, stable direction, defined scope, and rollback all still
   required. Tier-D automatic policies are out of scope for this rule.

## 5. Open questions for the owner

1. Public API naming for the v3 mode and member policy, to be settled in
   the §4.2 public contract (explicit-params-win semantics are fixed; the
   sentinel-default mechanism is the contract's job).
2. Default member count for the opt-in recipe: 8 is the only evaluated
   count — document as such, or fund a small member-count quality/speed
   screen first?
3. The archive size-model follow-on: fund an actual serializer (new format
   + correctness contract — a real project, not a free rider) or leave the
   ~31% bound on the shelf?
4. v0.11 scope: promoted dispatch + ensemble-v3 together (per the §4
   execution order), or staggered releases?

## 6. Authorization matrix

The single binding checkpoint is the commit of this document. It enacts
exactly the rows marked "this document"; every other row requires its own
later authorization, and nothing in §4's planning prose overrides this
table.

| Item | Authorized by | Notes |
| --- | --- | --- |
| §1–§2 gate retraction and disposition supersessions | **This document, on commit** | The supersession list in §1 is exhaustive. |
| §4.9 standing gate-design rules | **This document, on commit** | To be copied into the plan's evidence-discipline section per §4.1. |
| Ensemble-v3 **public contract freeze** (§4.2.1) | **This document, on commit** | Contract work only; no public code ships from it. |
| Ensemble-v3 **implementation + correctness suite** (§4.2.2) | **This document, on commit** | Private and non-exported until the public-ship row; no constructor/API surface is authorized here. |
| Ensemble-v3 **public ship** (§4.2.3–4) | Owner sign-off at the v0.11 release checkpoint | The §4.2.3 characterization prerequisite is published, including its post-run correction; owner ship sign-off is still required. |
| M6 successor **build + backtest** (§4.8) | **This document, on commit** | Ranking eligibility only via a passed backtest, never by fiat. |
| M2 broad panel | Owner sign-off at the v0.11 milestone | Frozen protocol per the plan; not started earlier. |
| **M4 TabArena-Lite placement** | Owner sign-off at the v0.11 release, explicitly | **Not authorized by this document.** Resolves the §3/§4.5 tension: §4.5 schedules it; only this row authorizes it. |
| v0.11 release itself | Owner sign-off at the release checkpoint | Scope per §5 Q4. |
| B3 campaign (§4.3) | Separate owner authorization after the ship work | Needs its own frozen contract with the §4.3 memory rule. |
| Q local microprototype (§4.6) | Separate owner authorization at its mechanism slot | Development-screen evidence only. |
| Q1 prototype funding | Separate owner decision on the microprototype's **measured** result | The 13.28% screening projection cannot fund it. |
| Fresh confirmation, CTR23 or any lockbox | **Nothing in this document** | Unchanged discipline. |
