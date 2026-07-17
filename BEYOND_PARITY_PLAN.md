# Beyond Parity: the ceiling program

*Drafted 2026-07-17 at `main` = `4f4a1c0`, after independent verification of the
closed best-of-both execution (see `benchmarks/best_of_both_completion_audit.md`).
Status then: engine fit parity with ChimeraBoost 0.15.0 proven byte-identical
(fit ratio 0.975 matched-core), basketball quality tied (−0.00023 R², cold-player
edge +0.0096 to DarkoFit), remaining gaps = predict throughput (1.83× matched),
ES policy (1.31× product fit), smooth-data and categorical quality classes,
and the deferred cleanup. ChimeraBoost is the floor. This program is about the
ceiling.*

## Execution ledger

Execution began from published `main` at `ab86269`.

- Prerequisite review chips are complete: categorical validation no longer
  scans every numeric object cell (`a280315`); empty prediction batches and
  the infinity/duplicate-category compatibility policy are explicit
  (`10d0fc3`); background warmup is single-flight, conventional falsy values
  disable it, and unsafe Numba workqueue concurrency is guarded (`ab86269`).
- Wave 1 is complete. New campaign protocols use the forward-only shared
  paired-ratio stability primitive in `basketball_campaign_harness.py`;
  `basketball_harness.py`, historical frozen runners, and their bound support
  manifests remain unchanged.
- `validation_strategy="group"` and the basketball robust-head protocol are
  the first Wave-1 implementation units. S3 and I1 pass the complete suite
  (`1,647 passed, 23 skipped`).
- S1 is closed from clean source `f10b449`: Student-t location lost `0.007982`
  mean creator-fold R² and MAE lost `0.008461`; both also failed the
  leave-one-fold-out and held-team gates despite small cold-player gains. The
  fatal first block stopped the campaign before timing confirmation, and no
  downstream evidence was consumed.
- P1's matched throughput protocol is implemented for reciprocal fresh-worker
  blocks, numeric and mixed inputs, cold/warm/public/binning/core phases, and
  8k-to-2M batches. The clean `27ff54e` run found public median ratios of
  `0.84-1.03x` ChimeraBoost, but three 8k/64k paired-ratio stability gates
  failed, so the all-case target is not formally certified. The packed core is
  generally faster; binning is the selected P2 target at a median `+33.8%`
  component excess. Peak RSS was `0.969x`.
- E4's fixed 50k-row phase-attribution profile now covers scalar control,
  binary, per-class and shared-vector multiclass, Gaussian, and Student-t. The
  clean `447190e` run finds every path tree-build-limited: `73.9%` for binary,
  `90.7-92.6%` for multiclass/distributional, versus at most `3.5%` for
  gradient/Hessian work. Gaussian LightGBM is the selected drill-down.
  Four-class per-class CatBoost was also materially faster per round than the
  shared-vector LightGBM path on this workload; this is diagnostic only.
- Z1's required cheap `random_strength` screen found evidence to retain the
  knob: `0.5` gained `+0.002124` mean creator-fold R², passed every
  leave-one-fold-out check, and improved overlap-exposed held-team and
  cold-player R² by `+0.006087` and `+0.007300`. It stays opt-in because the
  one-shot wall cost was roughly 50% higher; promotion requires S4's fresh
  sports suite. `1.0` failed. The remaining Z1 parameters proceed to 0.10
  warning preparation without this one.
- Z1 deprecation preparation is implemented and documented: depthwise,
  low-level histogram/leaf controls, automatic-LR probes, Bayesian bootstrap,
  and weighted GOSS now emit caller-located migration warnings when selected.
  `sigma_calibration` retains its existing alias warning. The defended
  `random_strength` and `rho_*` controls, plus the L/C-track-dependent
  surfaces, do not warn. The complete Wave-1 suite is green at `1,673 passed,
  23 skipped`.
- P2's first mechanism preserves already-contiguous C/F float64 blocks instead
  of forcing a C-order copy before binning. The isolated 524k-row binner fell
  to `0.586x` old numeric and `0.547x` old mixed time with byte-identical bins.
  In the clean matched campaign, every public median was `0.71-1.015x`
  ChimeraBoost and RSS was `0.969x`; numeric binning became `0.79-0.99x`.
  The formal target remains open because four of eight public paired-ratio
  stability gates failed. The artifact is final. Next is a preregistered
  seconds-integrated timing protocol and the remaining mixed validation cost,
  not a packed-core rewrite. The post-mechanism complete suite is green at
  `1,680 passed, 23 skipped`.
- P2's seconds-integrated successor is also final: DarkoFit's public median
  beat ChimeraBoost in all eight cases (`0.805-0.987x`), six were stable, and
  RSS was `0.992x`. The conjunctive proof failed numeric 8k/524k stability at
  `0.10556`/`0.10032` and one `0.690s` interval missed the `0.75s` floor. The
  code optimization stays, but P2 certification remains unavailable. No third
  protocol or further packed/binner optimization will chase this gate.
- E1's first expansion is retained from clean source `1016e7e`: the
  variable-Hessian fused histogram/split lane is canonical-model-state exact
  for binary Logloss and weighted RMSE. Across reciprocal fresh-worker blocks,
  geometric-mean fit and tree-build ratios were `0.7870x` and `0.7666x`
  reference, every paired ratio was stable, and peak RSS was about `0.99x`.
  This is an internal exact engine gain, not an external ChimeraBoost claim.

## Standing constraints (inherited, non-negotiable)

- **Basketball first, fatal, for every model-behavior mechanism**: unchanged ten
  creator folds, overlap-exposed held-team view, 585-row cold-player subset,
  behavior fingerprints, reciprocal clean timing. Engine-exactness work needs
  only the exactness+timing boundary.
- **Closed candidates stay closed as shaped**: OOB-5 row-bootstrap ensemble,
  the linear-leaves random-split auto-selector, auto-LR ES+refit, the donor
  cross-features selector, `cat_combinations`, binary temperature scaling, the
  quantile conformal offset and Gaussian scalar calibration *as scoped*, and
  the forest-work predict router. Anything below that resembles one of these
  is a materially different mechanism and must say exactly how.
- **CTR23**: the 9 spent-confirmation tasks (auction_verification,
  grid_stability, video_transcoding, kin8nm, fps_benchmark, health_insurance,
  student_performance_por, cars, space_ga) are development-only. The lockbox
  (naval_propulsion_plant, wave_energy, sarcos, cps88wages, socmob, fifa,
  moneyball, energy_efficiency, forest_fires) stays sealed until a candidate
  has passed development + a genuinely fresh preregistered panel + a ≥80%
  simulated-power design.
- **Gate-design fix, adopted for all NEW protocols** (this is allowed; it is
  not retuning an old decision): stability gates bind the **paired
  ratio** between arms measured in the same block, never per-arm absolute
  IQR/median on millisecond-scale series. Two of the strongest candidates died
  on environment noise; new protocols must not repeat that. Absolute-time
  gates remain for wall-clock budgets only, at seconds scale.
- No frozen artifact is rewritten; historical runners stay as attested.

## Track S — Win the creator's benchmark outright

The scoreboard on ChimeraBoost's own gist: DarkoFit 0.5267, ChimeraBoost 0.5270,
CatBoost 0.5363, ChimeraBoost ens-5 0.5402. Target: **beat 0.5363, then 0.5402,
single-library**, with cold-player quality preserved. This is the trophy and it
plays to our differentiators (distributional heads, entity structure).

**S1. Robust-head screen (zero new code — run first).**
Basketball targets are noisy and heavy-tailed; RMSE chases tails. Arms:
`loss="StudentT"` location prediction (learned ν), `loss="MAE"`, and RMSE
default, all under the frozen creator protocol with paired-ratio gates.
StudentT is a mechanism neither ChimeraBoost nor CatBoost defaults own.
Success = mean + LOFO + held-team + cold-player all non-regressing and mean
gain ≥ +0.002 (the screen bar used throughout the closed program). Cost: one
campaign day; the heads and vector-leaf trees already exist. Even a null
result is cheap and informative (it bounds how much tail-robustness matters).

**S2. Entity-aware ensemble (materially different from the closed OOB-5).**
What's different, explicitly: (a) **grouped bootstrap by player** (resample
player identities, not rows — respects the entity structure that both
guardrails exist to protect) with per-member group-aware OOB early stopping;
(b) shared preprocessing/binning across members (the closed candidate paid
2.41× wall; shared binning should bring K=5 nearer ~2×; measure);
(c) stability gated on paired ratios per the new template. Quality bar:
mean ≥ +0.004 (the closed OOB-5 reproduced +0.0039, so an entity-aware
variant must at least match it), cold-player positive, LOFO nonnegative.
If it passes basketball, it proceeds to the sports suite (S4) — not to a
default; ensembles ship as an explicit `n_ensembles` API only after S4.

**S3. Group-aware validation as a first-class option.**
`validation_strategy` today is `random | weighted_stratified`. Add
`"group"` (GroupShuffleSplit semantics driven by the existing `groups=` fit
argument). This is infrastructure, not a default change: it unblocks every
future selector on entity data. The linear-leaves selector failed precisely
because its random internal split shares players between train and val; a
group split is the honest version. Gate: exactness (no behavior change when
unused) + a focused unit suite. Then any future selection mechanism (linear
leaves, ensembles, calibration) may specify group-aware selection as part of
its design.

**S4. The sports confirmation suite (durable asset).**
One frozen basketball dataset cannot support promotion claims. Build a
preregistered multi-target, multi-season DARKO-derived panel: 3+ box-score
targets × 3+ seasons, each with creator-style folds plus held-team,
seen-player, and cold-player guardrails, contamination-documented, with a
declared primary aggregate and power analysis *before* first use. This is the
confirmation bed for S1/S2 survivors and every future sports claim. It is
also the panel where "beats ChimeraBoost where the owner actually works"
becomes a certified sentence.

## Track L — The smooth-data campaign (linear leaves' real test)

The mechanism is shipped, default-off, exact in its fallbacks. Its motivating
class (kin8nm/grid_stability/space_ga; lockbox naval/wave_energy/sarcos) was
never tested — basketball cannot test it. This campaign does.

**L1. Panels.**
- *Mechanism-probe tier (optional but cheap)*: synthetic smooth suite with
  exact Bayes floors — SCM generators with continuous mechanisms and known
  local-linear structure, where linear leaves should win by construction and
  canaries (pure-noise targets) must stay flat.
- *Development tier*: the three dev-legal smooth CTR23 tasks (kin8nm,
  grid_stability, space_ga) + fresh OpenML smooth/simulation regressions not
  in CTR23 or the lockbox lineage: puma8NH, puma32H, bank8FM, bank32nh,
  ailerons, elevators, delta_ailerons, delta_elevators, cpu_act, pol,
  friedman variants (screen each against the contamination registry
  fingerprints first; lineage-check the -8/-32 pairs as atomic clusters).
- *Confirmation tier*: a fresh preregistered panel of ≥14 datasets (the
  CTR23 power lesson: 9 tasks can never pass a concentrated-effect gate),
  drawn during I3 registry construction, untouched during development.

**L2. Arms.**
1. `linear_leaves=True` fixed-on (mechanism ceiling, no selector);
2. margin-thresholded selection: pick linear only when internal validation
   improves by ≥δ (δ preregistered from the probe tier — the closed selector
   failed with margins 0.03–0.18 that didn't generalize on entity data, so δ
   must be validated on grouped data too);
3. group-aware selection (S3) where an entity column exists;
4. `linear_residual=True` as a comparison arm — this doubles as the
   **retire-or-keep review** the closed program deferred: if linear leaves
   dominates it everywhere, deprecate `linear_residual` in Track Z;
5. ChimeraBoost 0.15.0 (`851ab7f`) as external comparator, its own defaults.

**L3. Gate ladder.** Basketball exactness first (mechanism is default-off, so
the fatal screen is: selector arms must not regress basketball when they
decline to engage; fixed-on arm is research-only there). Then L1 development
tier with sign tests. A promotion claim (opt-in default `linear_leaves=None`
= selected) needs the fresh confirmation tier. The lockbox shot happens only
if the confirmed effect distribution simulates ≥80% pass probability — and it
should target the certified sentence "DarkoFit beats ChimeraBoost 0.15 on the
sealed smooth-simulation panel."

## Track P — Predict throughput (the last engine gap)

Matched-lane predict is 1.83× ChimeraBoost; product-lane 2.91×. The closed
router died on noise gates while beating their core on small batches (0.88–
1.04×), so the capability is demonstrably there.

**P1. A real throughput protocol.** Dedicated harness: batch sizes 8k / 64k /
512k / 2M rows on synthetic + dev-real matrices (numeric-only and categorical
variants), fresh workers, cold and warm phases separated, paired-ratio gates,
seconds-scale absolute budgets only. The tiny basketball folds are guardrails
here, not the measurement.

**P2. Mechanisms, in order:**
1. **Row-major packed predict consuming the binner's output directly** (their
   0.14.2 trick): eliminate the transpose/copy between binning and the packed
   kernel; keep per-sample tree walks. Bit-identity oracle against the
   per-tree loop.
2. **Pack at fit time** (kill routing entirely for oblivious scalar models):
   the packed arrays become part of the fitted model; predict has no
   dispatch decision to make. Serialization already stores packed manifests
   for the leafwise lane; extend to oblivious.
3. **uint8 advantage**: our adaptive uint8 bins (≤255) already halve bin-row
   bandwidth vs their fixed uint16 — make sure the packed kernels exploit it
   (no widening loads), then measure whether we can *beat* their predict at
   equal tree counts, not just match.
4. Product-lane extras: cache the packed forest across predict calls (their
   `_forest_` lazy cache), and skip re-validation cost via the existing
   `assume_finite` path once the Track Z validation decisions land.

Target: ≤1.30× ChimeraBoost matched-lane predict to close the plan's Phase-3
line; stretch goal ≤1.0×. All behavior-exact.

## Track E — Fit-engine ceiling (beat their engine, not just match it)

Parity is proven on the default constant-leaf regression lane. Two honest
advantages are available that ChimeraBoost structurally lacks:

**E1. Fused-lane expansion (also the deletion enabler).** The shipped fused
kernel covers only the unit-Hessian, full-row, full-feature float64 lane.
Extend stepwise, each step bit-identity-gated and each retiring its
superseded reference variants (keep exactly one oracle pair per family):
1. **Complete:** hessian-carrying fused kernel → binary classification and
   weighted RMSE joined the exact fast lane at `0.7870x` total fit and
   `0.7666x` tree-build geometric-mean reference time;
2. counts-carrying variant → `min_child_samples` lane;
3. feature-mask and row-subset support via runtime branches (measure branch
   cost; adopt only where free).
Expected: several thousand deletable lines with proofs, plus real user-facing
classification speedups.

**E2. Histogram subtraction × fused (large-n advantage).** ChimeraBoost
rescans every sample at every level; DarkoFit already owns level-subtraction
kernels. A fused+subtraction lane for the oblivious path can be *faster than
their engine* on large data (target: ≥1.3× their matched-core fit at ≥200k
rows). This is the cleanest "ceiling not floor" engine claim available.
Protocol: matched-core lane like `basketball_chimera_v015` but on large-n
dev data; exactness against our own reference kernels (subtraction is
float64-rounding-equivalent, not bit-exact — so gate it as its own lane with
the documented equivalence class, exactly as the existing subtraction lanes
are handled today).

**E3. Float32 histogram streams** (ROADMAP R5, opt-in, already implemented):
measure on the throughput harness; promote to a size-gated auto lane only
behind regret gates.

**E4. Profile the unprofiled**: multiclass and distributional fits have never
had an attribution profile. Measure before touching; their 2× class-minor
layouts (R6) may already be fine.

## Track C — The categorical program (bank the −18% class)

Safe-ordinal's quality effect was −17…−19% on categorical CTR23 tasks; the
frozen causal gate failed only on inference time (1.2652 > 1.25) because the
transform *added* columns. The materially different mechanism:

**C1. Native ordinal-at-binning.** Mark declared-ordinal categoricals and bin
their *codes* directly in the existing numeric pipeline — zero added columns,
zero extra TS blocks, so the closed failure mode (wider matrix → slower
predict) is structurally eliminated. The `target_ordered_cat_codes` plumbing
(ROADMAP R10) is the starting point; the new piece is a declared-ordinal API
(`ordinal_features={col: ordered_categories}`) plus a safe auto-detection
rule that only fires on integer-coded or lexicographically-ordered categories
(the resolver idea from the accuracy-shootout follow-ups).

**C2. Panels.** Dev tier: the four dev-legal categorical CTR23 tasks
(auction_verification, video_transcoding, fps_benchmark, cars) + spent
TabArena categorical sets (Diamonds class) as development-only + fresh
cat-heavy OpenML sets found during I3. Confirmation: the I3 fresh panel's
categorical stratum.

**C3. Ladder.** Basketball first (should be a no-op there — no declared
ordinals; gate is exactness + no-engagement proof), then dev tier, then
fresh confirmation. Success also unblocks re-running the deferred mode-mix
diagnostic (plan item 11): if oblivious+ordinal absorbs the Diamonds-class
wins, the hybrid/depthwise deletion evidence finally exists.

## Track Z — Cleanup to 1.0 (bundle every break loudly)

The package grew to ~24.4k lines because deletions were correctly blocked on
replacement proofs. The proofs now have a pipeline (E1, C3); the rest is
policy. Do this as a **deprecation release (0.10) → 1.0** cycle:

**Z1. Deprecate now, delete at 1.0** (warnings in 0.10, removal PR pre-staged):
- `depthwise` tree mode (absent from every selector and campaign; ~800 lines
  incl. serialization pack/unpack once removed);
- `histogram_dtype`, `leaf_dtype`, `histogram_parallelism` (concluded
  experiments — fix winning defaults; float32 streams live on via E3's lane,
  not a user knob);
- `auto_learning_rate_probe*` (three params; superseded);
- `bootstrap_type="bayesian"` + `bagging_temperature`, and the weighted-GOSS
  uniform-mass variants (evidence-free; keep MVS + plain GOSS);
- `random_strength` is retained default-off: the required basketball screen
  passed at `0.5`; S4 confirmation, not deprecation, is next;
- `sigma_calibration` (already warning-deprecated in favor of
  `dist_calibration` — finish it);
- `rho_*` multipliers are retained because focused distributional tests defend
  independent head scaling, metadata, and round-trip behavior;
- `linear_residual` **only if** the L2 comparison arm shows dominance.
- `hybrid` stays until C3 produces the mode-mix evidence. `target_ordered_
  cat_codes` stays until C1 replaces it properly.

**Z2. Kernel deletions ride E1**, never ahead of it. Bookkeep per-family:
fused expansion PR lands → superseded variants deleted in the same PR with
the oracle retained → goldens + suite green. Target trajectory 24.4k → ~15k
after E1 complete, ~11-12k after mode deletions (C3-gated) — honest numbers,
not the old ~9k guess, since linear leaves/SHAP/warmup/validation added
legitimate mass.

**Z3. Fold in the review findings** (chips already filed): the categorical
predict validation overhead fix; the inf/empty-batch/duplicate-cat
compatibility decision (whatever is chosen, CHANGELOG it as breaking with
migration notes — 1.0 is the moment); `DARKOFIT_WARMUP` falsy parsing and the
background thread-count race.

**Z4. Docs**: root planning docs (KALMAN_READINESS_PLAN,
LINEAR_RESIDUAL_BOOSTING_PLAN, DISTRIBUTIONAL_*SPEC, fable_supervisor_handoff,
BEST_OF_BOTH_PLAN once superseded) → `docs/archive/`; stand up an mkdocs site
(their docs/ layout is a good template: getting-started, parameters, concepts,
shap, faq); README gets the Pareto headline (I4).

**Z5. Tests**: mark campaign verifiers (`-m campaign`), split CI jobs
(library suite vs campaign suite), prove identical coverage before/after —
this was deferred pending exactly that proof; do it as its own PR.

**1.0 criteria**: Z1 deletions executed, Z3 decisions shipped and documented,
P1 target met, suite green on the version matrix, NOTICE current, CHANGELOG
in Keep-a-Changelog form with every break enumerated.

## Track I — Infrastructure (parallel, never consumes promotion evidence)

**I1. Paired-ratio gate template** codified in the forward-only
`benchmarks/basketball_campaign_harness.py` (the historical
`basketball_harness.py` is hash-bound into frozen artifacts) so every new
protocol inherits: paired per-block ratios for stability, seconds-scale
absolute budgets only, arm-order alternation, fresh-worker blocks — the lesson
from the OOB-5 and router closures, structural.

**I2. SynthGen port + ledger backtest.** Port their `benchmarks/synthgen/`
recipe (Apache-2.0, numpy-only) or rebuild: SCM-prior generators, harvested
marginals (exclude CTR23 lockbox lineage and TabArena at the source), exact
Bayes floors, earned canaries. Before it may gate anything, it must reproduce
≥7/9 of *our own* decision ledger (we now have 18 basketball decisions plus
the TabArena campaigns to backtest against — a better ledger than they had).
Use: mechanism-probe tier for L/C/S direction-finding, never promotion.

**I3. Fresh confirmation registry.** One contamination-screened, fingerprinted
registry of ~20 unused regression datasets stratified smooth / categorical /
noisy-tabular, with lineage clusters atomic (the CTR23 v3 methodology,
reused). This is the durable confirmation asset every track's ladder ends in,
and it gets built once, before anyone needs it in anger. Power-analyze at
freeze time.

**I4. Pareto + status.** `benchmarks/make_pareto.py` equivalent: blended
quality vs fit-slowdown vs ChimeraBoost 0.15 / CatBoost / LightGBM,
regenerated per release, README headline. Plus a `bench_status.py` that
prints the latest aggregate table after every campaign (their CLAUDE.md rule
is right).

## Sequencing

**Wave 1 (cheap, parallel, this week's Codex-sized chunks):**
S1 robust-head screen · S3 group validation option · I1 gate template ·
P1 throughput harness · Z1 deprecation release prep · Z3 chips · E4 profiles.

**Wave 2 (the two quality campaigns + engine pushes):**
S2 entity ensemble · L1/L2 smooth campaign (includes the linear_residual
verdict) · P2 predict mechanisms · E1 fused expansion (+ paired deletions) ·
I2 SynthGen · I3 registry construction.

**Wave 3 (confirmation + ceiling claims):**
S4 sports suite + S-track confirmation · L3 fresh-panel confirmation →
lockbox power simulation → (if ≥80%) the one lockbox shot · E2 large-n
beat-their-engine protocol · C1–C3 categorical program → mode-mix rerun →
C3-gated deletions · 1.0 release.

**Explicitly not in this program:** reopening any closed candidate as shaped;
touching the lockbox before its preconditions; TabArena as anything but
report-only; rewriting frozen artifacts; deleting ahead of proofs.

## What success looks like

1. A certified sports claim on a preregistered multi-season panel: DarkoFit
   (single model or explicit ensemble API) above CatBoost's 0.5363-class
   scores with cold-player quality intact.
2. A certified smooth-panel claim over ChimeraBoost 0.15 with linear leaves —
   optionally sealed with the lockbox.
3. Engine: predict ≤1.3× (stretch ≤1.0×) and large-n fit ≥1.3× *faster* than
   their matched core.
4. A 1.0 release: ~28-param constructor, ~12-15k-line package, every break
   documented, docs site, Pareto headline — the library their creator would
   recognize as disciplined, doing things theirs cannot (distributions,
   uncertainty, entity-aware sports modeling, versioned serialization).
