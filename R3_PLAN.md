# R3_PLAN — the foundation gate, then the overhead war (v0.13 → v0.14)

> **Status:** owner direction, 2026-07-24, **revised same day after external
> review** (three independent reviews + synthesis; see §0.5). Execution
> instruction for Codex. Supersedes [`R2_PLAN.md`](R2_PLAN.md): its v0.12
> release work shipped or was honestly closed, while its funded Q1 mechanism
> is carried forward here (resequenced — see Q1). [`SHIP_RULES.md`](SHIP_RULES.md)
> governs process; [`AGENTS.md`](AGENTS.md) governs working discipline.
> One quality-changing automatic default per release; behavior-exact
> engineering and opt-ins ride alongside freely.
>
> **Revision note:** the 2026-07-24 external review confirmed the §0
> decomposition but found two real defects in this plan's first draft
> (cross-member preprocessing sharing was statistically invalid as
> written; capped auditions were mislabeled decision-identical) and one
> sequencing error (a second automatic selector before selector-cost
> amortization). Those are fixed below. It also moved the
> salvage-vs-re-fork question from "settled by the 2026-07-22 tripwires"
> to "decided by a bounded experiment": the tripwires watched the engine
> and the feature backlog, and the v0.12 loss came from neither — it came
> from orchestration overhead the tripwires never measured. §0.5 is that
> experiment. **P2's default flip, Q1, and P3's sharing work are frozen
> until the gate reports.**

## 0. Where we stand (v0.12 ladder, 2026-07-24, dev slice)

Source: [`benchmarks/v012_compute_ladder_20260724_result.md`](benchmarks/v012_compute_ladder_20260724_result.md)
(DarkoFit v0.12.0 `a9eb4db` vs ChimeraBoost v0.23.0 `6667843`, 13 M2
regression datasets, ratios DarkoFit/ChimeraBoost, lower better):

| Contrast | Quality [95%] | Fit | Predict/call | Peak RSS | W-L |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0/M0 | 1.0097 [1.0032, 1.0159] | 2.60x | 3.27x | 1.02x | 8-5 |
| DA/MA | 0.9881 [0.9746, 1.0037] | 1.25x | 3.35x | 0.92x | 6-7 |
| D8/M8 | 1.0363 [1.0338, 1.0388] | 3.57x | 1.82x | 0.39x | 2-11 |

**Strict Pareto victory: no.** The per-dataset decomposition
([`per_dataset.csv`](benchmarks/v012_compute_ladder_20260724_per_dataset.csv))
shows the deficit is concentrated and named:

1. **Diamonds is ~all of the default quality deficit.** D0/M0 loses
   1.3825x there; ex-diamonds the D0 aggregate is **0.9836 — a win**.
   Diamonds is where their group-centered crosses auto-engage and ours
   shipped opt-in. At DA/MA diamonds is 1.0032 (deep trees learn the
   interaction) — a representation-policy gap, not an engine gap.
   Caveat (review): ex-diamonds **D8/M8 is still 1.0134** — the cross
   default cannot close the ensemble gap by itself, and whether v3
   members even reach the cross audition is unverified (gate item b).
2. **The predict loss is per-call fixed overhead, not kernel
   throughput.** We win large batches (protein 0.61x, diamonds 0.95x)
   and lose small ones up to 10.5x (QSAR_fish). Their 0.21–0.23
   releases were a systematic sweep of exactly this overhead; named
   suspects in our code are listed under P1.
3. **The fit loss is small-n fixed overhead plus audition tax, stacking
   in the ensemble.** D0 fit went 1.38x → 2.60x across v0.11 → v0.12
   while quality improved — the selector's ~2.20x audition cost landing
   at ladder scale. On tiny sets D8 fit reaches 17–18x (airfoil,
   concrete): sequential sub-threshold members, each plausibly paying
   the audition race (verify: gate item b). Where B3 engages, ensemble
   fit is already at donor parity or better (protein 0.60x, QSAR-TID
   0.89x) — the deficit is the small-n tail, not the parallel
   architecture.
4. **Their ensemble lift is structurally better and unexplained.** M8
   improves on M0 by 3.54% (13/13) at 0.57x its default's fit; our v3
   gains ~1.0% at 0.78x. Member recipes are *nearly identical* across
   the two stacks (both ~0.8n without-replacement subagging, colsample
   0.85, OOB-complement early stopping; the recipe transplant already
   measured worse on our engine) — so the wedge is baseline variance,
   member horizons, or orchestration, and the gate's matched-member
   experiment (item e) is the instrument that separates those.

What v0.12 bought: airfoil 0.953 and concrete 0.888 at D0
(selector-era strongholds), a near-win accuracy point at 1.25x fit, and
the 0.39x ensemble RSS (partly an artifact of sequential small-set
members — do not sell it as durable). The external reviews' shared
verdict on "did anything help": several additions created real
capability, quality, memory, or local speed value; none has yet
compounded into whole-curve product dominance. The blocker is
amortization and engagement, not mechanism quality.

## 0.5 The foundation gate (3 days, dev data only, owner go required)

> **VERDICT (2026-07-24): gate complete — salvage confirmed on all three
> pre-stated readings.** Result:
> `benchmarks/r3_foundation_gate_result_20260724.md` on branch
> `codex/r3-foundation-gate-20260724` at `4f70938`. Diamonds closed at
> 1.0045x vs donor under real defaults with healthcare's bit-exact
> decline intact; the ledger resolved the cost deficits into named
> policy/dispatch (default retains 1,000 trees where the donor stops at
> 43–63; B3's threshold leaves a 4.6x forced-parallel win unengaged;
> members pay no selector race); matched members collapsed 95.6% of the
> ensemble lift *wedge* before forcing horizons, and forcing donor
> horizons closed the wedge entirely (0.99909).
> **Read that number precisely:** 0.99909 is the *averaging-lift* wedge
> — how much each stack gains from ensembling over its own single
> model — **not** overall quality parity. At fully matched members and
> horizons DarkoFit's RMSE is still **1.00343x** the donor's (1.00767x
> at stage 1). The wedge result retires the "their ensemble
> architecture is better" hypothesis; a ~0.3% matched-configuration
> quality residual remains open and is neither explained nor claimed
> away. The donor-core proof is **not
> funded**; the re-fork question is closed on evidence. §2 sequencing
> below is updated with the post-gate facts. Corrections to this plan's
> own assumptions, recorded honestly: v3 members do NOT run the
> audition races (the airfoil-17x arithmetic was wrong — the cost is
> the B3 threshold plus horizon policy), so the catcross flip carries
> no 8x ensemble cost multiplier and is naturally scoped to
> single-model fits; and the predict deficit was roughly half
> removable overhead (now removed: 1-row −52% default, −68% ensemble8,
> 24/24 bit-identical fingerprints) and half retained-tree policy.

**Question:** are the v0.12 deficits removable orchestration costs
(salvage: continue this plan in place), or is the product execution
graph structurally inferior (trigger the re-fork program)? The
2026-07-22 no-re-fork decision stands but its tripwires didn't cover
this failure mode; the gate closes that hole with measurement instead
of presumption.

**Frozen until the gate reports:** the P2 default *ship* (its dev run is
gate work), Q1, and all P3 sharing/caching work. P1's bounded predict
fast paths are NOT frozen — they are gate instruments and keepable in
every world.

### The five gate items

- **(a) The microsecond ledger.** py-spy + targeted timers on: one
  QSAR_fish-shaped predict call, one healthcare-shaped default fit, and
  the airfoil 1-vs-8-member marginal cost. Attribute wall time to
  {input conversion, validation, thread-mask switch, categorical remap,
  binning, forest walk, audition fits, param resolution, allocation}.
  Test the thread-switch hypothesis first (a ~1 ms OpenMP re-team on a
  ~100–300 µs call would *be* the 10x; skip-when-ambient-equal is the
  fix if so). **Paired donor reference arm (Codex amendment 1):** run
  the same ledger on pinned ChimeraBoost `6667843` — called directly
  from the proof harness, no renamed branch — in the same session, so
  the comparison is attribution-vs-attribution (where their ~36 µs goes
  vs where our ~300 µs goes), not fresh numbers against last week's
  ladder. All gate timing cells carry this pinned reference arm; the
  pin does not refresh mid-gate even if the donor releases.
- **(b) Member-path facts.** Establish by execution: do v3 members run
  the linear-selector audition per member (the airfoil 17x arithmetic
  says probably yes)? Does the catcross audition reach members at all
  (CHANGELOG scoping says probably no)? Both facts gate P2's honest
  scope and P3's amortization design.
- **(c) `salvage-p1` branch — the bounded predict fast paths.** Exactly
  five, all behavior-exact, none new-policy: one input
  conversion/validation per public call; shared per-call bag context
  (conversion + canonical factorization shared, member-local fitted
  maps untouched); thread-mask hygiene (restore parent after member
  fits; switch once per bag call; skip when ambient == target); serial
  twins on the routes that lack them (levelwise `add_predict`,
  class-major routes, binning — scalar oblivious already gates at
  8192; also fix `flat_predict_preferred` falling to a per-tree loop at
  `thread_count=1`); gdiff cross-block allocation removal (reuse the
  numeric block, write into the destination). Exactness suite on all;
  NPZ must not capture any cache. Measure the six proof cells before/
  after; stand up the 1-row M5 serving sentinel (donor number: 36 µs).
- **(d) Catcross-auto dev run.** Flip the audition to automatic for
  eligible fits on the gate branch and run the 13-dataset D0 slice
  (three coordinates). Tests the diamonds close under *real* defaults —
  the attribution ran against a constant control, and the interaction
  with the live linear selector is untested — plus healthcare's
  bit-exact guard decline. This doubles as P2's dev evidence if salvage
  wins. Measure D8 on diamonds too if (b) says members audition:
  quality gain vs stacked-audition fit cost, both reported.
- **(e) Matched-member ensemble decomposition, two stages (Codex
  amendment 2).** Benchmark-only index shims on both stacks: inject
  identical member train/OOB indices, seeds, and parameters, selectors
  off, on the six proof cells (QSAR_fish, healthcare, diamonds,
  protein, airfoil, concrete). **Stage 1:** each engine chooses its own
  stopping horizon per member — measures the stopping/orchestration
  contribution to the lift wedge. **Stage 2:** additionally force
  identical horizons (the donor's stage-1 selected horizons as the
  common reference) — isolates preprocessing/tree-engine differences.
  Forcing horizons immediately would erase one of the two leading wedge
  hypotheses before measuring it; the stage-1−stage-2 difference IS the
  horizon effect. Lift that disappears across the stages is portable
  policy; lift that survives stage 2 is foundation. Record member
  retained rounds, member-prediction correlation, and per-member wall
  time on both sides.

Deliberately **not** built in the gate: a renamed donor-core product
branch. Its decision-relevant facts are covered cheaper — the donor's
overhead numbers are already measured (the ladder *is* the donor
benchmark), the engines are byte-identical at matched config on the one
diagnosed coordinate (RSSI), and (e) covers the ensemble question. The
donor branch gets built as step one of the re-fork program *if the gate
triggers it*, with the port bill priced honestly first.

### Decision rules at gate close (owner decides; pre-stated readings)

**Salvage confirmed** — continue this plan — if ALL of:

1. diamonds D0 ≤ 1.05 under real defaults with guard declines intact;
2. the ledger attributes the dominant share of small-call/small-fit
   overhead to the named removable items (conversion, remap, thread
   switches, missing serial paths, audition amortization) rather than
   smearing it in sub-10% slices across the wrapper stack; and
3. matched members collapse most of the ensemble lift wedge (their
   advantage is portable policy/baseline property, not foundation).

**Donor-core proof funded** if ANY of: the ledger shows kernels at
parity with the gap smeared across the abstraction stack (architecture,
not policy); the lift wedge survives stage 2 of the matched-member
decomposition; or diamonds fails to close under real defaults.

**Wording that matters (Codex):** those are the gate's only two
outcomes — *salvage confirmed* or *donor-core proof funded*. The gate
cannot itself select ChimeraBoost: no cutover decision exists until a
donor-core branch has carried DarkoFit's required contracts (group
safety, rare-class safety, sample-weight semantics, safe NPZ, and one
representative extension — the 2-SE selector) **without losing the
donor's cost advantage**. The funded proof is that build: two branches
at pinned `a9eb4db` vs `6667843` per the external reviews, with the
donor's **mandatory pre-adoption fixes** (its bagging is row-based —
verified: members take an explicit row-complement OOB `eval_set`,
bypassing the group-aware splitter, so one player's rows can sit on
both sides; and its rare-class bagged-OOB scoring can misencode absent
classes), the layered migration architecture (`policy/`, `heads/`,
`io/`, `groups/` over a vendored core; fitted policy off the hot
path), and the honest 6–10 week bill to sports-pipeline parity
including revalidation of the distributional/conformal/NPZ stack the
live pipeline depends on.

**Two-tier tie rule (Codex amendment 3, reconciling the reviews):**

- *This screening gate:* ties go to incumbent DarkoFit. An ambiguous
  reading does not justify paying six-to-ten weeks of migration and
  revalidation while the rival keeps shipping.
- *The funded donor-core proof, if it runs:* ties go to ChimeraBoost —
  but only once the required contracts above are in place on that
  foundation without erasing its advantage. At that point the port
  cost is sunk into the proof itself, and at equal performance the
  dramatically smaller foundation rationally wins.

Both rules are fixed now, before any measurement, so neither verdict
can be argued backward from a preference.

### Gate hygiene

Dev data only; no holdout consultation; no new percentage gates —
readings 1–3 are judgment calls made on the published numbers, by the
owner, with my recommendation attached. Time-box: three working days of
machine time; instrumentation runs don't need the exclusive machine,
the before/after timings do. Execution uses clean worktrees at the
tagged pins (`a9eb4db`, `6667843`); the uncommitted Fresh Eyes repair
set in the local tree stays untouched and out of the gate branches.

## 1. The R3 mechanisms (amended; execution order set by the gate)

### P1 — Predict fast-path (behavior-exact; ships on exactness tests)

The gate's item (c) list IS P1's first wave, and it ships in every
world. Second wave (post-gate, ledger-ordered): the
`_codes_for_transform` path (per-call pandas `Series` construction +
`get_indexer` when pandas is imported, per-row Python fallback when
not — replace with a NumPy/Numba canonical factorization cache with
unique-value remapping, the donor's 0.21 design that bought ~9x at
1-row); preallocated output blocks; dead-init removal; multiclass and
SHAP-path reuse. Targets on the ladder slice: worst per-call ≤1.5x
(from 10.5x), aggregate D0 predict ≤1.3x (from 3.27x), keep the
large-batch wins, and get the 1-row sentinel within 2x of the donor's
36 µs now, parity later.

### P2 — Catcross auto-engagement: the one quality default of v0.13

Unchanged destination, corrected path: the gate's item (d) supplies the
dev evidence under real defaults. The ship decision additionally waits
on **(i)** the gate verdict and **(ii)** the P3 amortization question:
if members pay the race (gate item b), the flip must ride the
parent-level selector architecture (below) or ship scoped to
single-model fits with the release claim scoped to match — an automatic
default that improves D0 while worsening the dominant D8 fit deficit
is exactly the whole-curve failure this plan exists to end. Ship path
per SHIP_RULES once unblocked: dev suite → CTR23 release-validation +
newest unused sports season (expect bit-exact declines on sports) →
`categorical_crosses=False` rollback → CHANGELOG. Consumes v0.13's
single quality-default slot.

### P3 — Fit-overhead war (amended per review blockers B1/B2)

Profile-first from the gate ledger, then, in order:

- **Parent-level selector architecture** (the centerpiece): selection
  races run once at the parent — group-safe, on the parent's split —
  and members receive resolved *identities* (engage linear leaves
  yes/no; selected cross-pair list), then fit **member-local** state
  from scratch. The race is paid once, not eight times. This is a
  quality-policy change for ensembles (member decisions were previously
  member-local where they existed at all): it goes through dev +
  holdout as part of a release, not behind a behavior-exact label.
- **Sharing boundary (corrected).** Shareable across members: raw input
  conversion, schema validation, immutable column metadata, canonical
  raw-value factorization, parent-selected pair identities. **Never
  shared:** fitted borders, ordered target statistics, category maps,
  group-centered means, binned matrices, OOB scoring state — these are
  sample- and weight-dependent fitted state; sharing them lets a
  member's OOB rows shape its representation, contaminates the OOB
  early-stopping signal, and violates the zero-weight-rows contract.
  Fitted-preprocessing reuse is permitted only among auditions and the
  winner refit *inside one member* when rows, weights, and split are
  identical (the donor's intra-fit cache pattern — its measured 17–32%
  small-fit win).
- **Audition cost control (honest version).** Two lanes only: (1)
  reuse that provably preserves the full decision rule
  (shared prep + completed-fit reuse within a member; early
  termination only where the original stopping rule's decision is
  already information-complete); (2) capped/raced auditions as a
  **declared quality policy** through dev + holdout — the donor's own
  `selection_rounds` cap occasionally flips decisions with 0.5–1.5%
  regressions, so "capped but decision-identical" is not a claimable
  property. No third lane.
- **Per-fit fixed costs**: lazy imports, deferred allocations, cheaper
  small-n split machinery — ledger-ordered.

Targets: worst small-n D0 fit ≤2x (from 8x), D8 worst ≤4x (from 18x),
aggregate D0 fit ≤1.8x (from 2.60x).

### P4 — Ensemble lift (gate item e answers the core question)

The matched-member decomposition replaces the original instrumentation
plan. Post-gate, whatever the wedge turns out to be (baseline variance,
member horizons, engagement policy) is specced as v0.14's quality
default with its falsifier stated. The member-recipe route stays dead.
Residual watch item: protein D0 1.0156.

### Q1 — resequenced to last, in both worlds

New fact (verified in donor source): **quantized-gradient histograms
have been ChimeraBoost's default since 0.18.0** (`quantize_gradients`,
~15-bit, leaf values from unquantized floats). Q1 is therefore
catch-up on an axis where we are already near parity, not
differentiation. It stays funded; it runs after P1/P2/P3 in the salvage
world and is inherited rather than reimplemented in a re-fork world.
Unchanged technical scope otherwise (lane + byte-identical float
fallback + seed-exact repeatability + quality characterization + the
1M predict-ratio anomaly).

## 2. Sequencing and releases (post-gate, 2026-07-24)

The gate reordered the work. The dominant remaining deficits are, in
measured order: **default horizon/stopping policy** (1,000 trees built
and walked where the donor retains 43–63 — the largest single lever on
both fit and predict at small n, and a quality-changing default),
**B3's engagement threshold** (behavior-exact 4.58x wall-time win on
tiny ensembles), and the residual predict tail after P1.

**Strength of the horizon evidence, stated honestly:** the tree-count
gap (1,000 vs 43–63 retained) is a structural fact of the two default
policies and is not in question. The claim that it is the *dominant*
cost driver rests on one non-formal, loaded-machine healthcare cell in
which the two engines also trained on different row counts (the donor
holds out its validation fraction) — so "we lose ~7x purely on volume"
is stronger than the evidence supports. The horizon experiment below
is what converts a well-supported hypothesis into a measurement.

- **Integration first:** review and merge the gate branch (P1
  behavior-exact commits, the selector-composition fix, harnesses,
  sentinel). Verified: `categorical_crosses=False` remains the product
  default on the branch; the flip stays out of main until its
  ship-check passes. **This is now urgent for a second reason:** the
  docs commit `5d86beb` is on `main` and pushed, and it links
  `benchmarks/r3_foundation_gate_result_20260724.md`, which exists
  only on the gate branch — `main` currently cites evidence it does
  not contain. Integrating the branch repairs the reference; until
  then the link is known-broken, not a missing file.
- **Horizon dev measurement immediately (Opus amendment, 2026-07-24):**
  the public default is `early_stopping=False, iterations=1000`
  (sklearn_api.py:10230/13121) while every internal surface — members,
  auditions — sets `early_stopping=True`; on small data we build each
  tree ~31% cheaper than the donor and lose ~7x by building ~9–23x as
  many. Because every later measurement (B3 calibration, catcross cost
  profile, Q1 payoff, the serving sentinel) is polluted while the
  baseline builds 1,000 trees, the horizon question is measured FIRST:
  three arms on the dev slice, **all using existing code**: (1) the
  current default; (2) `early_stopping=True` alone; (3)
  `early_stopping=True, refit=True`, which already selects the horizon
  on validation and then retrains on all rows
  (`sklearn_api.py:11690`) — recovering the row forfeit the donor's
  policy pays permanently. *Correction (Codex, 2026-07-24): an earlier
  draft of this bullet said the refit variant needed building. It does
  not; `refit` is implemented and only needs measuring.*
  **Slot rule, pre-stated:** quality-neutral-or-better on the dev
  slice → horizon policy takes v0.13's single quality-default slot and
  catcross moves to v0.14; a real quality cost → catcross ships v0.13
  as planned and horizon work continues for v0.14. **Refit caveat:**
  `refit=True` currently makes automatic catcross ineligible
  (`sklearn_api.py:10536` returns `"refit"`), so if arm 3 wins, the
  refit–catcross composition must be fixed and catcross re-measured
  before the two can ship together — the same class of collateral
  suppression the gate already caught between catcross and the linear
  selector. Owner signs the slot assignment either way.
- **Thread-topology hypothesis (unproven; test before planning around
  it).** *Correction, 2026-07-24: two earlier per-tree figures in this
  plan and in review — a "26x cheaper at one thread" donor comparison
  and a "0.325 → 0.071 ms/tree" within-engine one — were both computed
  by dividing **concurrent multi-worker wall time** by **aggregate**
  tree count, which understates per-tree cost by roughly the worker
  count. Both are withdrawn.* What the ledger actually supports: B3's
  forced-parallel route is **4.58x faster in wall time** on tiny
  ensembles, and that win is mostly **member-level concurrency**, not
  per-tree thread efficiency. Correcting for the critical path (the
  917-tree member defines the forced-parallel wall), per-tree cost at
  2 threads vs 14 is roughly 0.26 vs 0.33 ms — a ~1.3x hint, well
  inside the confounds (different member row counts and
  hyperparameters, loaded machine, non-formal timing). Whether
  single-model small-n fits are mis-threaded is therefore **an open
  question, not a finding**: settle it with a controlled sweep at 1, 2,
  and 14 threads on identical model, data, and horizon before any
  design work. **Exactness requirement if it proceeds:** identical tree
  counts across routes is NOT proof — histogram reduction order is the
  classic float trap — so thread-invariant bit-exact fits must be
  verified in the exactness suite per lane; where bit-exactness fails,
  the lane is a measured quality-neutral dispatch change, not
  behavior-exact.
- **What does NOT wait for the horizon result (Codex, 2026-07-24):**
  P1's exactness is independent of horizon policy; B3's members
  already early-stop on OOB rows, so their tree counts and the
  threshold work are unaffected; and Q1's causal microbenchmark stays
  valid (only its product-level payoff shrinks if the default builds
  far fewer trees). Only the *quality-slot assignment* and the
  re-baselining of P1/Q1 economics depend on the horizon measurement.
- **v0.13** = integrated P1 + **B3 threshold reshape** (behavior-exact,
  measured envelope, memory guard, rollback) + the slot-rule winner as
  the release's one quality default (the loser leads v0.14). Catcross,
  whichever release it lands in, ships scoped to single-model fits
  with the release claim scoped to match, and the release notes state
  explicitly that ensemble diamonds (1.35x) remains open until
  parent-to-member decision passing exists. Ladder at release against
  the then-current donor pin.
- **v0.14** = the slot-rule loser as its quality default; member
  horizon policy rides the same coherent flag as the default horizon
  change if it can honestly be one policy.
- **v0.15 candidate** = parent-level decision passing to members
  (extends catcross + linear leaves to ensembles; fixes D8 diamonds),
  plus Q1 — **re-scoped after the horizon change lands**: quantization
  accelerates tree construction, and a default that builds an order of
  magnitude fewer trees shrinks Q1's absolute payoff; re-measure before
  funding the integration.
- **Timing hygiene note:** the gate's fit/ensemble ledgers ran at load
  average ~5.5 with `formal_timing_evidence: false`. Tree counts and
  route facts are structural and stand; the ms-per-tree ratios are
  directional only and get re-measured on the exclusive machine before
  appearing in any result doc.
- v0.13 tracking targets (not continuation gates): D0/M0 quality ≤1.00
  with diamonds ≤1.05; D0 predict ≤2x aggregate on registered batches
  (horizon-limited until v0.14); D8 worst small-set fit ≤4x. The v0.13
  ladder still carries the stop criteria: rival velocity reopening
  feature gaps, or regressions against these targets, convene the
  strategy question again — but the re-fork presumption is now closed
  and would need new evidence, not a stale reading.

## 3. Discipline notes

- The 13-dataset slice is development data, fixed for comparability,
  never a tuning set. Ship-checks consult the holdout per SHIP_RULES.
  The gate consults no holdout at all.
- One quality default per release. A parent-level selector change and
  the catcross flip in the same release must be presented as the single
  coherent default change they are, or split across releases.
- **The overhead rule (new, permanent, from the review):** an automatic
  feature's disabled or ineligible state performs no work beyond a
  constant branch — no preprocessing, no allocation, no policy
  resolution on the hot path; an engaged feature pays only its inherent
  transform/model cost; selection never recomputes unchanged base
  preprocessing. This is what makes automation-first compatible with
  frontier dominance, and every future selector ships against it.
- Classification remains out of R3 scope by owner priority (noted: a
  donor-core world would inherit their classifier work; a salvage world
  leaves the blind spot open — revisit at v0.14). TabArena stays
  deferred.
- Rival triage: 0.21–0.23 added no product surface (no backlog growth,
  no old-tripwire fire), but the *new* lesson is that cross-cutting
  performance polish is itself a competitive attack the backlog metric
  misses — the gate exists because of it.

## 4. Standing owner decision points

1. **Gate go/no-go.** The two-tier tie rule (§0.5) is part of the
   authorization — a single yes adopts both.
2. Gate verdict sign-off (salvage vs re-fork program).
3. Release sign-off on v0.13 (catcross default scope).
4. P4's mechanism choice before it becomes v0.14's default.
5. Any holdout consultation outside a scheduled ship-check (logged).
6. Classification/TabArena scope changes — owner-initiated only.
