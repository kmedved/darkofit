# R2_PLAN — depth confirmation, catcross completion, and the successor mechanisms

> **Status:** owner direction, 2026-07-23. Execution instruction for Codex.
> **Regime note:** [`SHIP_RULES.md`](SHIP_RULES.md) governs. Any remaining
> reference in this document (or its ancestors) to Tier-D, powered fresh
> campaigns, one-shots, campaign identities, contamination ledgers, or
> authorization rows reads as: *dev suite → holdout ship-check → revertible
> flag → owner release sign-off.* One quality-changing automatic default
> per release; behavior-exact changes and opt-ins may ride alongside.
> Supersedes the R1 sequencing in
> [`BEAT_CHIMERABOOST_PLAN.md`](BEAT_CHIMERABOOST_PLAN.md) (which remains
> the governing strategy doc). Current process is defined by
> [`SHIP_RULES.md`](SHIP_RULES.md) and [`AGENTS.md`](AGENTS.md);
> [`benchmarks/SHIPPING_POLICY.md`](benchmarks/SHIPPING_POLICY.md),
> [`NEXT_STEPS.md`](NEXT_STEPS.md), and older campaign records are historical
> evidence, not active gates. Exact source pins, fixed seeds, exclusive-machine
> timing, TESTING_LOG notes, and one mechanism slot at a time remain.
> Section 7 lists the remaining owner decision points.

Every item below maps to a measured frontier deficit or an earned candidate:

| Item | Frontier target | State |
| --- | --- | --- |
| P1 Automatic-depth paired development benchmark → ship-check | Broad + sports quality; first automatic default | Fix harness, then rerun 32 lineages; consult holdout only if development clearly improves |
| P2 Catcross attribution → path to ship | Diamonds/healthcare (0.20's categorical blow) | M6 `advance`; holds the current mechanism slot |
| P3 Depth opt-in exposure | Immediate user value (−5% sports, spent) | Candidate validated, private |
| P4 B3-v2 activation-gated parallelism | 6.1× ensemble fit optic; rescue the 3.84× | Successor to a clean kill |
| P5 Selector-v3 principled margin | Smooth-data quality (protein) | Successor to a clean kill |
| P6 Member-policy retune | Ensemble broad quality (+1.5% vs their +3.5%) | Queued (R1-5) |

---

## P1 — Automatic-depth paired development benchmark and ship-check

The v1–v3 material below is retained only to explain the historical harness
failures. The current action begins at **Current SHIP_RULES path**.

**Outcome of execution identities v1/v2 (2026-07-23):** closed before
launch, correctly. The v2 registry preflight could not fill frozen slot
`high_density_numeric_02` — no eligible, uncontaminated identity existed —
and the frozen `all_32_slots_required` rule closed the campaign with **the
fresh inspection unspent** (`fresh_inspection_spent: false`), no
substitution, no partial read. The 99.8% power result does not transfer to
a recomposed panel. At that point the candidate remained private,
unpromoted, and still held its one shot. The closure branch's patch was
landed on `main` as `d55e8b6` (patch-equivalent to branch commit
`660bc90`).

**Root cause:** the design froze abstract slots first and checked
fillability last. The successor inverts that order.

### P1-v3 — fillability-first redesign (historical authorization; superseded)

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

**P1-v3 execution closed 2026-07-23:** the owner authorized the exact
one-shot and the launch manifest spent the fresh inspection. The first
candidate worker then failed the frozen branch-integrity check before a
control arm or paired comparison: the registry classified the outer
training split as depth 8 (`23,373 / 9 = 2,597` rows per feature), while
the actual automatic policy saw the post-validation fit population
(`19,867 / 9 = 2,207.444444`) and correctly resolved depth 6. The runner
published no raw/result artifact and forbids a rerun. This is a
prospective-design failure, not a quality verdict. Automatic depth remains
private and unpromoted; P3 is unchanged. See the dated terminal note and
create-only launch/failure records.

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

### P1-v3 outcome (2026-07-23): one-shot spent on a design defect —
candidate untainted

The v3 one-shot closed without a quality verdict at the first worker: the
registry hand-derived branch expectations from **outer training rows**,
while the policy resolves depth on **post-validation-reservation rows**
(15% smaller); lineage 1's rows-per-feature crossed the 2,500 branch
threshold between the two bases (2,597 vs 2,207), the integrity check
correctly fired, and the frozen rules closed the campaign. One row
completed, unpublished and unread; no control ran; the record states the
outcome is evidence about nothing. Costs: the v3 inspection accounting is
spent, and `airlines_departure_delay_10m` is contaminated for future
fresh use. The candidate behaved exactly per its code contract.

### Current SHIP_RULES path (owner decision, 2026-07-23)

The owner retired the preregistration apparatus — see
[`SHIP_RULES.md`](SHIP_RULES.md). The pragmatic path replaces P1-v4
entirely:

1. **Fix the harness bug** (branch expectations computed on the wrong row
   basis — or better, drop the hand-derived expectation check and trust
   the pinned policy's own deterministic resolution, recorded per fit).
2. **Rerun all 32 lineages as a paired development benchmark.** No
   one-shot semantics, no contamination ledger — excluding Airlines over
   one unpaired result would preserve the bookkeeping we abolished. Keep
   the useful integrity checks (deterministic policy resolution recorded,
   serialization, splits, candidate/control pairing); drop the
   hand-derived expectation gate. If a bug is found mid-run, fix it and
   rerun.
3. **Read the result like an engineer:** if automatic depth is clearly
   better on the panel and not worse on the holdout ship-check
   (SHIP_RULES), it becomes the v0.12 default with a documented revert
   flag. If not, it stays an opt-in and we say so in the docs.

The section below is retained for historical context only.

### P1-v4 — superseded (historical)

v4 is a **new campaign identity**, not a rerun: outcome-blindness is
intact (only one deterministic branch-resolution fact was revealed), the
defect was in the harness's hand-derived expectation, and the candidate is
unchanged. Opening it requires a dated owner decision record saying
exactly that. Conditions, all binding:

1. **Drop the contaminated lineage**; re-verify the remaining 31 and
   recompute power prospectively (v3's headroom was enormous — 0.998
   against 0.80 — so 31 slots very likely still qualify; the simulation
   decides).
2. **Re-derive every branch assignment by executing the pinned
   candidate's actual resolution code on the exact fit-time inputs**
   (post-reservation rows). Any lineage whose rows-per-feature falls in
   the flip band (threshold ÷ 0.85) gets special scrutiny. Re-stratify
   and re-power from the executed truth, never from hand math.
3. **Mandatory full rehearsal before freeze:** a data-free execution of
   the complete worker path — imports, warmup, environment, policy
   resolution, integrity checks — against the frozen registry in the
   frozen environment. This one stage would have caught all three
   failures (v1's import defect, v2's unfillable slot, v3's branch
   mismatch).
4. **Expectations by execution, never by hand (standing rule, add
   everywhere §4.9 lives):** a frozen expectation about candidate
   behavior must be *generated by running the pinned candidate code* on
   the frozen inputs. Hand-derived expectations are how a correct
   candidate fails an incorrect contract. The integrity check itself
   stays — stratification and power depend on the engagement mix — but
   its reference values come from execution.

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
the old donor), then per SHIP_RULES: **opt-in ship** with honest
characterization at the next release, and **automatic-default candidacy**
via the standard ship-check (clearly better on dev, not worse on the
holdout, revertible flag), respecting one-automatic-default-per-release.

## P3 — Depth `"auto"` opt-in exposure (product work, authorized)

Independent of P1's outcome. The ensemble-v3 exposure pattern, smaller:
public parameter with sentinel semantics per the established precedence
design; resolved depth recorded in fitted metadata and round-tripped;
support matrix with loud errors for untested combinations; docs disclose
the spent-evidence numbers with labels (−0.7% general with the diabetes
harm case named, −5.0% sports cold-player, both dev-labeled) and that
default-on is pending its SHIP_RULES ship-check. Ships in the next
release regardless of P1.

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
- Default candidacy via the SHIP_RULES ship-check (dev + holdout +
  revertible flag), never by re-testing against the coordinates that
  produced the kill.
- Sequenced after P2 and P4 — same one-slot discipline.

## P6 — Member-policy retune (after P2 resolves)

M6 v3 development comparing member recipes on the broad slice: current
recipe vs the rival's blessed member defaults (lr 0.15 / colsample 0.85,
public) vs one intermediate; goal is closing the ensemble broad-quality
wedge (+1.5% ours vs +3.5% theirs over respective singles). Winner feeds
the ensemble opt-in's documented recipe; any default change goes through
the SHIP_RULES ship-check.

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

## P7 — Process-kill re-triage, corrected (owner + Codex reconciliation, 2026-07-23)

The archive supports **two clean rescues, several narrower follow-ups, and
one integration project** — not a mass resurrection. (Accounting: the
Phase F audit's reproducible tally is 23 valid / 13 healed / 1
infrastructure / 4 forward actions: 3 mechanism re-adjudications plus the
M6 tripwire. An independent two-reader re-scoring under SHIP_RULES graded
more kills as apparatus, but several of its calls were corrected on review
— notably T5, which *was* later measured on Panel 3's calibration
coordinates: `0.97156x` aggregate, `1.01859x` worst dataset, 4W/2L/7T —
promising value with tail harm, not "never measured." Do not quote a
papercut headcount as the audit's finding.)

**Rescue now (clean procedural casualties, evidence unusually strong):**

1. **B3 parallel members** — behavior-exact, memory-safe, 3.84x faster
   warm; killed by one cold short-fit case. Add the deterministic
   minimum-work activation threshold with sequential fallback; ship when
   exactness, memory, and engaged-speed tests pass.
2. **Linear-leaf auto-selector** — 0.9686 protein / 0.9627 replay with
   harm gates passing; killed on a 0.0252-vs-0.03 margin technicality.
   Rebuild the margin from measured noise and operating cost, not a
   hand-chosen constant.

**Narrower follow-ups (re-scoped, not blanket revivals):**

3. **Declared-ordinal selector** — the 17.3% safe-ordinal win is real but
   covered only the datasets with externally declared order, and its
   `1.265x` causal inference cost was a valid predeclared measurement.
   Re-scope: a native-vs-ordinal selector engaging **only on externally
   declared order** (never inferred order), with targeted inference
   optimization — exactly what the original record recommended.
4. **One-hot-255 donor probe** — the 15.2% T7b result came from modifying
   **CatBoost**, not DarkoFit, and carried dataset regressions. It is a
   donor hypothesis: run a small DarkoFit-local probe before believing it
   transfers. Decide after catcross lands which categorical mechanism
   (this or the declared-ordinal selector) goes next.
5. **Q reprofile** — last, against the post-dispatch engine.

**Accuracy-v2 integration project (the capstone, not a revival list):**
rebuild the accuracy rung with component ablations and a true no-op
fallback, retaining A10 as the fallback preset. The 10k horizon (already
shipping inside `preset="accuracy"`; +0.45% with ~13% train / ~11%
inference cost and two slightly-losing splits — investigate an adaptive
early-stopping ceiling instead of a raw cap) and the guarded numeric
crosses (positive aggregate with tail harm per the Panel 3 measurement)
belong **inside this project's ablations**, not as standalone defaults.

**Documentation only (not mechanisms):** `hybrid` stays documented as
experimental; `target_ordered_cat_codes` exposes deliberately leaky
research behavior (`leaky_full`) and is documented truthfully as such.
Neither enters the mechanism queue.

**Stay dead (ran and lost on the merits):** S1 robust heads, S2 entity
ensemble, auto-LR+refit, basketball cross-feature donor,
categorical-combinations donor (catcross is the fixed successor), T10
automatic OOB5, C2 native ordinal (1.32x worst task), binary temperature
scaling, both `random_strength` variants, float32 histograms, target-stat
permutations, safe one-hot (Diamonds concentration), linear residual,
unconditional packed router, auto tree mode as default (rides accuracy-v2
where its cost is the point), T7b automatic-L2.

**Retained hypothesis (generic, not sports code):** three mechanisms
improved cold players while losing primary folds — but all on the same
basketball dataset and cold subset, so treat as one suggestive
observation, not three confirmations. If pursued, it enters as a generic
new-entity/group-shift mechanism through the normal pipeline.

## Execution queue

1. Fix the automatic-depth harness and rerun all 32 lineages as a paired
   development benchmark.
2. **Only if depth is clearly better in development**, consult CTR23 and the
   newest untouched sports season once as the release ship-check. Otherwise
   keep depth opt-in and do not spend the holdout.
3. Finish catcross attribution and its resulting development or opt-in path.
4. Build B3's deterministic minimum-work threshold with sequential fallback.
5. Rebuild the linear-leaf selector with a noise- and cost-aware margin.
6. After catcross, choose one next categorical mechanism: the declared-order
   selector or the small DarkoFit-local one-hot donor probe.
7. Retune the ensemble member policy.
8. Build accuracy-v2 with component ablations and A10 as the fallback.
9. Reprofile Q against the post-dispatch engine.

Depth opt-in exposure and other release/documentation work do not consume a
mechanism slot.

## 7. Owner decision points (SHIP_RULES regime)

The original authorization matrix is retired with the apparatus it
governed (see git history). Under SHIP_RULES the owner's decision points
reduce to:

- **Release sign-off** (each release): which automatic default ships (at
  most one quality-changing default per release), which opt-ins and
  behavior-exact changes ride along, and the release scope.
- **Holdout consultations**: each ship-check against CTR23 / the newest
  untouched season is deliberate and logged; after first use, CTR23 is
  relabeled per SHIP_RULES.
- **TabArena placement**: still deferred by owner decision; revisit at a
  release of the owner's choosing.
- Everything else — development, benchmarks, harness fixes, reruns,
  characterization, documentation — proceeds without ceremony, one
  mechanism slot at a time.
