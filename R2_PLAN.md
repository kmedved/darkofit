# R2_PLAN — depth confirmation, catcross completion, and the successor mechanisms

> **Status:** owner direction, 2026-07-23. Execution instruction for Codex.
> Supersedes the R1 sequencing in
> [`BEAT_CHIMERABOOST_PLAN.md`](BEAT_CHIMERABOOST_PLAN.md) (which remains
> the governing strategy doc); all discipline unchanged:
> [`benchmarks/SHIPPING_POLICY.md`](benchmarks/SHIPPING_POLICY.md),
> [`NEXT_STEPS.md`](NEXT_STEPS.md) §4.9 gate rules, [`AGENTS.md`](AGENTS.md),
> create-only artifacts, TESTING_LOG entries, one mechanism slot at a time.
> The §7 authorization matrix defines exactly what this document enacts.

Every item below maps to a measured frontier deficit or an earned candidate:

| Item | Frontier target | State |
| --- | --- | --- |
| P1 Depth fresh Tier-D run | Broad + sports quality; first automatic default | V3 registry, power, combined execution freeze, and data-free preflight complete; awaiting owner one-shot authorization |
| P2 Catcross attribution → path to ship | Diamonds/healthcare (0.20's categorical blow) | M6 `advance`; holds the current mechanism slot |
| P3 Depth opt-in exposure | Immediate user value (−5% sports, spent) | Candidate validated, private |
| P4 B3-v2 activation-gated parallelism | 6.1× ensemble fit optic; rescue the 3.84× | Successor to a clean kill |
| P5 Selector-v3 principled margin | Smooth-data quality (protein) | Successor to a clean kill |
| P6 Member-policy retune | Ensemble broad quality (+1.5% vs their +3.5%) | Queued (R1-5) |

---

## P1 — Automatic-depth fresh Tier-D confirmation

**Outcome of execution identities v1/v2 (2026-07-23):** closed before
launch, correctly. The v2 registry preflight could not fill frozen slot
`high_density_numeric_02` — no eligible, uncontaminated identity existed —
and the frozen `all_32_slots_required` rule closed the campaign with **the
fresh inspection unspent** (`fresh_inspection_spent: false`), no
substitution, no partial read. The 99.8% power result does not transfer to
a recomposed panel. The candidate remains private, unpromoted, and — 
critically — still holds its one shot. Land the closure branch
(`codex/t7b-fresh-tier-d-20260723`) on `main` as the record.

**Root cause:** the design froze abstract slots first and checked
fillability last. The successor inverts that order.

### P1-v3 — fillability-first redesign (authorized)

**Enumeration complete 2026-07-23:** 32/40 concrete identities are verified
before design freeze: 17 depth-4 lineages (9 numeric, 8
categorical-or-grouped), 15 depth-8 lineages (5 numeric, 10
categorical-or-grouped), and three group-safe lineages. Eight identities were
rejected for concrete schema, target-validity, or OpenML-binding reasons; no
model ran and the fresh inspection remains unspent. The dated pre-design note
and hash-bound JSON are the only permitted inputs to the P1-v3 power
recalculation.

**As-built power qualified 2026-07-23:** the exact 32-identity, 17/15
depth-branch panel passed the unchanged primary design at `0.998000`
simulated power with one-sided Wilson lower `0.996657`. All downstream
authority remains false pending the combined design/execution freeze review.

**Freeze review ready 2026-07-23:** the combined contract/harness is published
and its data-free 32-lineage preflight passed. The harness requires a separate
hash-bound owner record and cannot launch without it. No model has run and the
fresh inspection remains unspent.

1. **Concrete registry enumeration before anything freezes:** for every
   proposed slot, name the exact dataset/lineage identity, verify it loads
   in the frozen worker environment (the `autogluon.common` lesson), and
   attest contamination status against the full campaign history. Slots
   without a verified identity do not exist. Publish this enumeration as a
   dated pre-design note.
2. **Recompute power prospectively on the as-built panel** (the frozen
   simulation method; ≥80% Wilson lower bound required as before). The
   depth effect sizes are large — a slightly smaller verified panel very
   plausibly still clears the bar; the analysis, not hope, decides.
3. **Freeze design + execution contract together** over the verified
   registry; owner gives final sign-off at the freeze review.
4. **One-shot fresh run**, create-only, testing-log entry.
5. **On GO:** automatic depth becomes the public default in v0.12 — the
   first evidence-confirmed automatic default. **On NO-GO:** close for
   defaults, keep the P3 opt-in, record the transfer failure.

**New standing rule (add to the plan's evidence discipline alongside
§4.9):** *no design freezes over unverified resources* — every frozen
slot, dataset, dependency, and environment import must be attested
fillable and loadable **before** the freeze that binds it. A design that
discovers its world at execution preflight has frozen a wish, not a plan.

## P2 — Catcross: finish attribution, then the ship path
(current mechanism slot)

### P2a. Mechanism-specific spent attribution (next step, authorized)

Mirror the protein-attribution pattern on the categorical targets. Frozen
three-arm design per dataset, on `diamonds` and
`healthcare_insurance_expenses` at the three registered M2 coordinates:

- **constant** (current public default, no crosses);
- **automatic** (the private candidate `c3f2608c`'s guarded engagement);
- **forced** (crosses unconditionally on).

Gates, frozen before execution: automatic/constant aggregate ≤ 1.000 per
dataset; harm bound ≤ 1.02 per coordinate; and the behavior rule stated
with the selector lesson applied — **engagement completeness is evaluated
against the arm's own margin rule, and a decline-with-value-left counts as
a calibration finding for the successor, not an automatic identity kill,
unless the frozen contract explicitly says otherwise.** Write the rule
either way *before* seeing outcomes; do not leave it implicit.

### P2b. Implementation spec (the candidate surface, for the record and
for the eventual public contract)

**What it is:** group-centered categorical cross features, ported in
design (not code-copied without attribution — NOTICE applies if any
ChimeraBoost code is adapted) from ChimeraBoost 0.20's CATCROSS.

- **Candidate generation:** ordered pairs of declared-or-detected
  categorical columns, capped by a deterministic budget
  (`max_cross_candidates`, default from the development contract; ties
  broken by column index). No triples in v1.
- **Encoding:** each selected pair becomes one synthetic column encoded
  with the existing ordered-target-statistics machinery in
  `darkofit/target_encoding.py`, with **group centering**: when `groups`
  is supplied, the target statistic is computed on group-aggregated
  residuals (subtract the group mean target before encoding) so a
  cross cannot memorize entity identity — this is the generic-abstraction
  form of the fix for the old categorical-combinations donor, which was
  killed for −0.091 cold-player R². Without `groups`, plain ordered
  statistics apply.
- **Audition:** each candidate cross is auditioned on the fit's internal
  validation split: fit-with vs fit-without on a capped-iteration probe,
  keep if the validation gain clears the engagement margin. The margin is
  **noise-derived, not a constant**: engage when
  `gain > k × SE(gain)` with `k` frozen prospectively in the development
  contract (the selector-v1 lesson: a fixed 0.03 constant is a knob that
  kills identities). Audition cost is bounded by a declared fraction of
  total fit time; borrow ChimeraBoost 0.20's audition train-loss skip
  idea (triaged from their changelog) to keep probes cheap.
- **Determinism and observability:** fixed-seed audition order; selected
  pairs, per-pair validation gains, margins, and decline reasons persisted
  in fitted metadata (`catcross_` state); safe-NPZ round-trips the
  selected pairs and their encodings exactly; prediction-time
  reconstruction from stored encodings only (no target access).
- **Support matrix (loud pre-fit errors, ensemble-v3 pattern):**
  regression and binary first; multiclass deferred; incompatible with
  `linear_leaves` in v1 (interaction untested); weights supported
  (weighted target statistics); missing categorical levels at predict
  fall back to the pair's prior, recorded.
- **SHAP:** synthetic columns appear as their own features with a
  documented mapping back to the source pair; no silent attribution to
  parents in v1.
- **Invariants/tests:** `catcross=off` byte-identical to current engine;
  fixed-seed repeatability; save/load prediction identity; group-centered
  encoding proven leak-free on a synthetic entity-memorization test
  (a generator where identity memorization is the only signal — the
  mechanism must score ~zero); audition budget respected.

### P2c. Evidence path after attribution

Sports guardrail replay (cold-player view — the failure mode that killed
the old donor), then: **opt-in ship** under Tier-E with honest
characterization at the next release, and **automatic-default candidacy**
through a powered fresh panel (same registry infrastructure as P1; its own
campaign identity and power analysis — diamonds-sized effects are large,
so power should be achievable).

## P3 — Depth `"auto"` opt-in exposure (product work, authorized)

Independent of P1's outcome. The ensemble-v3 exposure pattern, smaller:
public parameter with sentinel semantics per the established precedence
design; resolved depth recorded in fitted metadata and round-tripped;
support matrix with loud errors for untested combinations; docs disclose
the spent-evidence numbers with labels (−0.7% general with the diabetes
harm case named, −5.0% sports cold-player, both spent/dev) and that
default-on is pending fresh confirmation. Ships in the next release
regardless of P1.

## P4 — B3-v2: activation-gated parallel members (Track I, next speed slot)

Successor to the killed 7×2 topology, designed to be un-killable by the
same case:

- **Deterministic activation rule, dispatch precedent:** parallelize
  member fitting only when a pre-fit work estimate clears a frozen bound —
  `member_work = rows_after_sampling × active_features × planned_iterations`
  (exact formula frozen in the contract), with the bound calibrated the
  same outcome-blind way as the dispatch `scan_work` threshold. Below the
  bound: sequential path, byte-identical to today. The killing case
  (startup overhead on short fits) is excluded *by construction*, not by
  hope.
- Same member seeds ⇒ identical models either path (behavior-exact at the
  model level); acceptance per `NEXT_STEPS.md` §4.7: no materiality bar,
  no regression where not engaged, stable measured win where engaged.
- Memory per the §4.3 operational rule: process-tree peak RSS, fixed
  topology and total CPU, hard absolute ceiling plus
  ratio-or-absolute-delta allowance vs sequential control.
- Warm-worker lifecycles stay out of v2 (a third identity if ever needed).

## P5 — Selector-v3: principled margin (Track I, behind P2/P4)

The mechanism is validated (0.9686 protein aggregate, zero harm; 0.951/0.955
where engaged); only the engagement margin failed. Successor rules:

- **Anti-grind (binding):** the new margin may not be a constant chosen to
  capture the known coordinate-1 miss (margin 0.0252 vs old 0.03). Use the
  same noise-derived form as P2b (`gain > k × SE(gain)`, `k` frozen
  prospectively), calibrated on spent data *excluding* the protein
  coordinates that produced the kill.
- Confirmation only on the powered fresh panel (P1 registry
  infrastructure, separate campaign identity).
- Sequenced after P2 and P4 — same one-slot discipline.

## P6 — Member-policy retune (after P2 resolves)

M6 v3 development comparing member recipes on the broad slice: current
recipe vs the rival's blessed member defaults (lr 0.15 / colsample 0.85,
public) vs one intermediate; goal is closing the ensemble broad-quality
wedge (+1.5% ours vs +3.5% theirs over respective singles). Winner feeds
the ensemble opt-in's documented recipe; any default change is Tier-D as
always.

## Standing items

- **Docs dossier: keep.** Commit the README/docs/mkdocs + TabArena
  dossier working-tree changes as their own docs checkpoint (owner
  decision: keep). The dossier's joint depth+L2 rows-per-feature idea is
  a natural Track I entry if P1 confirms.
- Rival-changelog triage continues each milestone (0.21 triage exists;
  0.22+ when it lands).
- Re-fork tripwires unchanged; next milestone frontier ladder re-runs
  against the rival's then-current release with the same compute-ladder
  protocol.
- Next release (v0.12) assembles: P1 outcome (default or not), P3 opt-in,
  catcross opt-in if P2c reaches it, updated characterizations, milestone
  ladder, and — if the owner separately signs off — the deferred first
  TabArena-Lite placement.

## 7. Authorization matrix

| Item | Authorized by this document | Still owner-gated later |
| --- | --- | --- |
| P1-v3 fillability enumeration, power recompute, design+execution freeze | ✔ | The fresh run itself (final sign-off at freeze review) |
| P1 v1/v2 closure branch landed on `main` | ✔ | — |
| P1 default promotion on GO | — | ✔ release-time sign-off |
| P2a attribution runs | ✔ | — |
| P2b candidate development (private) | ✔ | Public opt-in ship (release sign-off) |
| P2c powered fresh candidacy | — | ✔ own campaign authorization |
| P3 opt-in exposure | ✔ | Rides release sign-off |
| P4 B3-v2 development + calibration | ✔ (after P2a completes; one slot) | Merge/ship at release sign-off |
| P5 selector-v3 | — | ✔ at its slot |
| P6 member retune | — | ✔ at its slot |
| Docs dossier checkpoint | ✔ | — |
| TabArena, fresh data beyond P1's registry, lockbox | — | ✔ explicit, named, per-access |
