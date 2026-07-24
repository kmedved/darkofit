# R3_PLAN — the overhead war and the catcross default (v0.13 → v0.14)

> **Status:** owner direction, 2026-07-24. Execution instruction for Codex.
> Supersedes [`R2_PLAN.md`](R2_PLAN.md) (complete: everything it queued
> shipped in v0.12.0 or was honestly closed). [`SHIP_RULES.md`](SHIP_RULES.md)
> governs process; [`AGENTS.md`](AGENTS.md) governs working discipline.
> One quality-changing automatic default per release; behavior-exact
> engineering and opt-ins ride alongside freely.

## 0. Where we stand (v0.12 ladder, 2026-07-24, dev slice)

Source: [`benchmarks/v012_compute_ladder_20260724_result.md`](benchmarks/v012_compute_ladder_20260724_result.md)
(DarkoFit v0.12.0 `a9eb4db` vs ChimeraBoost v0.23.0 `6667843`, 13 M2
regression datasets, ratios DarkoFit/ChimeraBoost, lower better):

| Contrast | Quality [95%] | Fit | Predict/call | Peak RSS | W-L |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0/M0 | 1.0097 [1.0032, 1.0159] | 2.60x | 3.27x | 1.02x | 8-5 |
| DA/MA | 0.9881 [0.9746, 1.0037] | 1.25x | 3.35x | 0.92x | 6-7 |
| D8/M8 | 1.0363 [1.0338, 1.0388] | 3.57x | 1.82x | 0.39x | 2-11 |

**Strict Pareto victory: no.** But the per-dataset decomposition
([`per_dataset.csv`](benchmarks/v012_compute_ladder_20260724_per_dataset.csv))
shows the deficit is **concentrated, named, and mechanistically attackable**,
not diffuse:

1. **Diamonds is ~all of the default quality deficit.** D0/M0 loses
   1.3825x on diamonds; that single dataset contributes ~+2.5pp to the
   13-dataset geomean. Excluding it, D0 quality ≈ **0.985 — a win**. At
   D8/M8 diamonds is 1.3545x (~+2.4pp of the 3.6% deficit). Diamonds is
   exactly the dataset where ChimeraBoost's group-centered categorical
   crosses auto-engage at default — and **our identical mechanism shipped
   in v0.12 as an opt-in**. At the accuracy points (DA/MA) diamonds is
   1.0032 — deep trees learn the interaction unaided, confirming this is a
   feature-representation gap, not an engine gap.
2. **The predict loss is per-call fixed overhead, not kernel throughput.**
   We *win* the large batches (protein 0.61x, diamonds 0.95x) and lose the
   small ones catastrophically (QSAR_fish 10.5x, Fiat-500 6.6x,
   healthcare 6.3x). ChimeraBoost's 0.21–0.23 releases were a systematic
   sweep of exactly this overhead (pandas removal, single input
   conversion, shared bagged-member transforms, serial kernel twins for
   ≤4-row batches; their 1-row numeric predict is now 36 µs). Ours has
   had no equivalent pass.
3. **The fit loss is small-n fixed overhead, and it stacks 8x in the
   ensemble.** D0/M0 fit: healthcare 7.98x, Fiat-500 6.40x, QSAR-TID
   4.74x — versus airfoil 0.78x, protein 1.04x, concrete 1.07x at scale.
   D8/M8 fit on tiny sets: airfoil 17.1x, concrete 18.1x — eight members
   each paying the full per-fit overhead, while their bagged ensemble
   fits in **0.57x their own default's time**.
4. **Their ensemble lift is architecturally better.** M8 improves on M0 by
   3.54% winning 13/13 datasets, at 0.57x the fit cost. Our v3 improves
   on D0 by ~1.0% at 0.78x. The member-recipe transplant was already
   falsified (their recipe on our engine: 1.0088 quality at 2.11x cost),
   so the wedge is structural — how members are budgeted, diversified,
   and early-stopped — not the hyperparameters.

What the v0.12 defaults bought: airfoil 0.953 and concrete 0.888 at D0
(selector-era strongholds), accuracy-point quality parity-or-better at
1.25x fit, and the 0.39x ensemble RSS. The losses that remain each have a
name. That is what this plan attacks. Watch item: protein D0 1.0156 —
both engines engage linear leaves there; small residual gap worth one
diagnostic look during P4, not a campaign.

## 1. The R3 mechanisms, in priority order

### P1 — Predict fast-path (behavior-exact; ships on exactness tests)

One profiling pass, then a fixed-overhead kill list. Known suspects, in
likely order of yield (each mirrors a measured ChimeraBoost win, so these
are proven-yield donor ideas, implemented generically):

- **Single input conversion per call**: validate/convert once, reuse the
  same array for checks and prediction (theirs removed a full duplicate
  `to_numpy` materialization).
- **Per-call allocation and revalidation audit**: cache fitted-transform
  lookups (categorical/ordinal remaps — we already reuse exact pandas
  codes; extend to the generic path), preallocate output blocks, skip
  dead inits.
- **Serial kernel twins for tiny batches**: parallel binning/forest-walk
  kernels pay the OpenMP fork/join on 1-row calls; dispatch to serial
  twins below the measured crossover (~5 rows for them). Bit-identical by
  construction; `warmup()` compiles both sides. This is the predict-side
  mirror of the fused-lane dispatch we shipped fit-side.
- **Ensemble predict de-duplication**: members share one input
  conversion/validation and one transform cache per call (their 0.21
  change: 8-member 50k predict 1.21 s → 0.40 s).

Acceptance (SHIP_RULES behavior-exact): bit-identical predictions on the
exactness suite, defined envelope, rollback flag per lane where dispatch
is involved. Targets, measured on the ladder slice: worst-case per-call
ratio ≤1.5x (from 10.5x), aggregate D0 predict ≤1.3x (from 3.27x), keep
the large-batch wins. Add a standing **1-row serving micro** (sports
game-state shape) to the M5 sentinels; their number to beat is 36 µs.

### P2 — Catcross auto-engagement: the one quality default of v0.13

The mechanism, audition guard, eligibility floor, provenance, and NPZ
round-trip all shipped in v0.12 as `categorical_crosses=True`. R3 flips
eligible scalar-RMSE fits to auto-audition (mirroring the selector's
pattern: guarded, deterministic, exact fallback, decision recorded in
fitted metadata).

- Dev expectation: diamonds D0 1.38 → ~1.0; D0/M0 aggregate quality from
  1.0097 to ≈0.985. The audition guard already declines ineligible/
  harmful cases (healthcare sits below the 2,353-row floor; forced probes
  there measured harmful — the guard must keep declining it).
- Ship path per SHIP_RULES: clearly better on dev suite → not worse on
  the holdout (CTR23 release-validation set + newest unused sports
  season; sports check should show bit-exact declines — no categorical
  cross candidates in the sports schema) → `categorical_crosses=False`
  as the documented rollback → CHANGELOG.
- This consumes v0.13's single quality-default slot. Depth stays opt-in;
  no other quality default rides this release.

### Q1 — Packed-histogram gradient quantization (already funded)

Unchanged scope from the funding note (`03ae4a4`): bounded engine
prototype → integration behind a lane with byte-identical float fallback,
seed-exact repeatability, quality characterization (stochastic
quantization is not behavior-exact by construction), and the 1M-row
predict-ratio anomaly (1.114) measured properly. Expected yield: ~17%
large-n fit — attacks the axis where we are already near parity (protein
1.04x, concrete 1.07x) and converts them to wins. Q1 is an engine
improvement; it is explicitly **not** the answer to small-n overhead (P3)
or ensemble lift (P4), per the ladder result's own closing note.

### P3 — The fit-overhead war (small-n default + ensemble stacking)

Profile-first: measure the fixed cost of one eligible small fit
end-to-end (input validation, auto-param resolution, binning, validation
-split machinery, selector auditions, callback plumbing) on a ~1–3k-row
dataset, and the marginal cost of ensemble members 2–8. Then kill in
measured order. Known structural candidates:

- **Share immutable work across ensemble members**: binning/borders,
  input validation, preprocessing — compute once, reuse per member
  (their bagged fit shares the categorical transform; ours refits
  everything per member).
- **Audition cost control at small n**: the selector and (post-P2)
  catcross auditions each fit extra models; verify their cost shows up in
  the profile and, if material, bound audition budgets at small n
  without changing decisions (decision-identical or it doesn't ship).
- **Per-fit fixed costs**: lazy imports, deferred allocations, cheaper
  split-machinery setup on the small-n path.

Acceptance: behavior-exact where claimed (bit-identical fits) or
decision-identical with recorded provenance. Targets on the ladder
slice: worst small-n D0 fit ≤2x (from 8x), D8 worst ≤4x (from 18x),
aggregate D0 fit ≤1.8x (from 2.60x, with Q1 compounding).

### P4 — Ensemble-lift diagnosis (the v0.14 quality mechanism)

Question to answer, with instrumentation rather than speculation: **why
does their 8-member bag beat their default by 3.5% on 13/13 datasets at
0.57x its fit cost, while our v3 gains ~1.0% at 0.78x?** Decompose on a
handful of datasets:

- Member budget accounting: their effective per-member iterations /
  early-stop behavior vs ours (do members train far shorter than the
  default single model?).
- Diversity source: their bagging (row sampling, seeds) vs our
  deterministic 80% without-replacement + colsample 0.85 — measure
  member-prediction correlation in both stacks.
- Baseline headroom: how much of their lift is variance harvesting off a
  higher-variance default (vs our lower-variance default having less to
  harvest)? Compare single-model vs member variance profiles.
- Structural sharing: what their members share (binning? transforms?)
  that keeps M8 at 0.57x.

Output: a mechanism candidate (member budget division, diversity policy,
shared-preprocessing parallel members, or a combination) specced for the
v0.14 quality-default slot, with the falsifier stated up front. The
member-*recipe* route is already dead; do not re-litigate it.

## 2. Sequencing and releases

Order: **P1 → P2 → Q1 → P3**, with **P4 diagnosis running whenever the
timed machine is busy elsewhere** (it is mostly instrumentation and
reading, not timed benchmarking). P1 before P2 so the predict fast-path
is settled before the catcross ship-check re-times anything.

- **v0.13** = P1 (behavior-exact predict pass) + P2 (catcross default —
  the release's one quality default) + Q1 lane + whatever P3 items have
  landed on exactness. Rerun the ladder at release against
  ChimeraBoost's then-current release (pin refresh; they will have
  moved past 0.23.0 — verify at worker zero, same as this run).
- **v0.14** = the P4-derived ensemble mechanism as its quality default,
  plus remaining P3 items.
- Success criteria for the v0.13 ladder (interim, honest): D0/M0 quality
  ≤1.00 with diamonds ≤1.05; D0 predict ≤1.3x aggregate and ≤1.5x worst;
  D0 fit ≤1.8x; D8/M8 quality ≤1.02. **Full strict dominance is a
  two-release program** — v0.13 closes default-point quality and the
  overhead axes; v0.14 closes ensemble lift.

## 3. Discipline notes (unchanged, restated for this cycle)

- The 13-dataset ladder slice is **development data** and stays fixed for
  cross-release comparability; it is never a tuning set for any P-item.
  Ship-checks consult the holdout (CTR23 release-validation set + newest
  unused sports season) exactly as SHIP_RULES prescribes.
- One quality default per release: P2 in v0.13, P4's mechanism in v0.14.
  Everything else ships behavior-exact or opt-in.
- Classification remains unmeasured on the ladder and undeveloped as a
  product story; it is a growing blind spot (their classifier keeps
  improving), but it stays out of R3 scope by owner priority. TabArena
  first placement stays deferred (owner decision).
- Rival triage at next milestone: their velocity is currently
  polish-over-features (0.22/0.23 added no new features), so the
  unported-feature backlog did not grow — no re-fork tripwire. The one
  standing post-fork feature gap is cat×cat combinations; if ever
  ported, port the 0.23 pair-coded design, not the repudiated string
  scheme.

## 4. Standing owner decision points

1. Release sign-off on v0.13 (defaults flip: catcross auto).
2. P4's mechanism choice before it becomes v0.14's default.
3. Any holdout consultation outside a scheduled ship-check (logged).
4. Classification/TabArena scope changes — owner-initiated only.
