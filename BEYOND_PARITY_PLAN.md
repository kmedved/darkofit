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
  `basketball_harness.py` and bound support manifests remain unchanged, while
  attested run-time versions of later-hardened runners remain recoverable at
  their source commits and hashes.
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
- E1's subset expansion is closed from clean source `11a72a1`. All eight
  basketball-scale cells were model-state exact and materially faster, with
  subset geometric-mean fit/tree ratios of `0.5348x`/`0.5265x`, but weighted
  row sampling failed the frozen timing-stability gate at `0.2085` fit and
  `0.2060` tree-build IQR/median (limit `0.15`). Automatic subset dispatch was
  restored to the reference path. A count-carrying oblivious lane is
  inapplicable because `min_child_samples` intentionally belongs to leaf-wise
  and hybrid builders. E1 is closed as shaped and authorizes no kernel
  deletions.
- S2 is closed as shaped from clean source `d2c14ba`. The complete
  player-identity bootstrap with group-disjoint OOB early stopping and shared
  numeric preprocessing lost `0.004182` mean creator-fold R², lost 8/10
  folds, and failed every leave-one-fold-out check. It did improve the
  overlap-exposed held-team and cold-player views by `+0.002239` and
  `+0.014869`, but the fatal primary gate stopped the campaign after one block;
  there is no timing claim and no ensemble API work.
- L1/L2's 21-coordinate smooth development screen advances current-default
  fixed linear leaves to selector design. They improved equal-task RMSE by
  `7.97%` versus DarkoFit default, won all 21 splits, and beat ChimeraBoost's
  linear-only lane by `3.63%`; the matched-policy variant was slightly worse.
  ChimeraBoost product defaults remain `1.62%` better because their separate
  cross-feature selector contributed `5.46%`. Global `linear_residual`
  improved only `0.48%`, regressed on space_ga, and lost to local-linear
  leaves on every dataset, satisfying the planned deprecation criterion.
  This used folds 3–9 of spent development tasks only; the lockbox is sealed.
- The first selector safety gate passes from clean source `1dd1c36`. A frozen
  3% relative-improvement threshold over player-group-disjoint validation
  declined linear leaves on all ten basketball folds and the held-team
  guardrail. Final predictions, canonical model state, fold scores,
  held-team score, and cold-player score matched the constant-leaf control
  exactly. Reciprocal wall, fit, prediction, and RSS ratios were stable and
  within budget. This advances only to the spent smooth development
  coordinates; no API, automatic policy, default, or lockbox spend is
  authorized.
- The same frozen 3% selector passes its 21-coordinate spent smooth
  development gate from clean source `a81e874`. It selected linear leaves on
  20 coordinates, declined one, improved equal-task RMSE by `7.67%` versus
  default, won all three datasets, and retained `96.17%` of fixed-linear's
  benefit. It remains `1.96%` behind ChimeraBoost product defaults. This
  advances the exact selector only to I3 fresh-confirmation registry and
  power design; the public API, defaults, and CTR23 lockbox remain sealed.
- I3's target-blind fresh registry is frozen from clean source `8664f7c`: 20
  outcome-unseen lineages, stratified 14 smooth numeric / 3 categorical / 3
  noisy tabular, with 60 immutable coordinates. Every candidate passed
  pre-program DarkoFit, ChimeraBoost benchmark-universe, TabArena-name,
  exact-ID/name/fingerprint, and CTR23-v3 near-lineage checks. A 200,000-draw
  preregistered bootstrap gives `99.9965%` conditional pass probability under
  the spent development effect distribution. This authorizes one exact fresh
  confirmation run only; no candidate task has been scored and the CTR23
  lockbox remains sealed.
- A pre-score dtype audit supersedes I3's descriptive `smooth_numeric` label
  with registry v2's `smooth_process`: the 14-task primary stratum is 5
  numeric-only complete, 7 categorical complete, and 2 categorical with
  missing predictors. Tasks, lineages, coordinates, contamination decisions,
  and power are unchanged. All confirmation and later claims bind v2 and use
  the narrower smooth/process wording.
- The sole fresh selector confirmation is complete and closed from clean
  source `29bd30c`: selector/default was `0.9893x`, only 2/14 primary
  lineages won, and selector/ChimeraBoost was `1.1196x`. Categorical and noisy
  guardrails passed, but the promotion gates did not. No power calculation,
  default change, or lockbox spend is authorized.
- E2's matched large-n certification is complete and closed from clean source
  `d77dc8b`. DarkoFit was quality-neutral and faster at both sizes, but its
  equal-size geometric-mean speedup was `1.2793x`, below the frozen `1.30x`
  claim threshold. All other gates passed. The raw artifact remains immutable;
  the post-run analyzer audit reproduces the same failure decision.

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
- No frozen artifact is rewritten. Attested run-time code stays recoverable at
  its bound source commit and hash; later harness hardening must be labeled
  post-run and must reproduce the immutable artifact's decision.

## Track S — Win the creator's benchmark outright

The scoreboard on ChimeraBoost's own gist: DarkoFit 0.5267, ChimeraBoost 0.5270,
CatBoost 0.5363, ChimeraBoost ens-5 0.5402. Target: **beat 0.5363, then 0.5402,
single-library**, with cold-player quality preserved. This is the trophy and it
plays to our differentiators (distributional heads, entity structure).

**S1. Robust-head screen — closed.**
Student-t location and MAE both failed the fatal creator-fold,
leave-one-fold-out, and held-team gates despite small cold-player gains. No
robust-head default or follow-on confirmation is authorized from this screen.

**S2. Entity-aware ensemble — closed as shaped.**
The materially different player bootstrap, group-disjoint OOB validation, and
shared preprocessing implementation failed the fatal creator-fold primary
gate despite improving cold-player quality. No ensemble API is authorized.

**S3. Group-aware validation — complete.**
`validation_strategy="group"` now provides GroupShuffleSplit semantics through
the existing `groups=` fit argument. It is infrastructure, not a default
change, and preserves exact behavior when unused. Future entity-data selectors
may use it without reopening the closed selector shapes.

**S4. The sports confirmation suite (durable asset).**
One frozen basketball dataset cannot support promotion claims. Build a
preregistered multi-target, multi-season DARKO-derived panel: 3+ box-score
targets × 3+ seasons, each with creator-style folds plus held-team,
seen-player, and cold-player guardrails, contamination-documented, with a
declared primary aggregate and power analysis *before* first use. This is the
confirmation bed for materially new S-track candidates and every future sports
claim. It is
also the panel where "beats ChimeraBoost where the owner actually works"
becomes a certified sentence.

## Track L — The smooth-data campaign (linear leaves' real test)

**Status (2026-07-17): closed for policy promotion.** The frozen 3% selector
passed basketball and spent-development gates but failed fresh confirmation:
`0.9893×` default, only 2/14 primary lineage wins, and `1.1196×`
ChimeraBoost product. Fixed linear leaves were `1.0069×` default on the fresh
primary panel. See
[`benchmarks/fresh_selector_confirmation_result.md`](benchmarks/fresh_selector_confirmation_result.md).
Do not open the CTR23 lockbox or retune this selector on the fresh panel.

The mechanism is shipped, default-off, and exact in its fallbacks. Its
motivating class was tested through the development and fresh-confirmation
tiers below; it did not generalize strongly enough for policy promotion.

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

**L3. Gate ladder (completed).** Basketball exactness passed, the L1
development tier passed, and the sole fresh confirmation failed. Therefore no
promotion claim, power simulation, or lockbox shot follows from this selector.

## Track P — Predict throughput (the last engine gap)

**Status: optimization retained; certification closed as scoped.** The
program began at 1.83× ChimeraBoost matched-lane and 2.91× product-lane
prediction time. P1 established the 8k/64k/512k/2M reciprocal fresh-worker
protocol. P2 then removed an unnecessary contiguous-block copy before binning
with byte-identical results.

The seconds-integrated successor measured DarkoFit public medians at
`0.805-0.987×` ChimeraBoost in all eight cases, but failed the conjunctive
stability/absolute-duration proof. The optimization stays; the all-case
≤1.30× certification does not. The frozen result explicitly closes a third
protocol and further packed/binner work under this track. Any future predict
claim needs a materially new question and preregistered protocol, not another
retry of P1/P2.

## Track E — Fit-engine ceiling (beat their engine, not just match it)

Parity is proven on the default constant-leaf regression lane. The program
tested the following engine ceilings:

**E1. Fused-lane expansion — closed as shaped.** The shipped fused kernel
covers the unit- and variable-Hessian full-row/full-feature float64 lanes:
1. **Complete:** hessian-carrying fused kernel → binary classification and
   weighted RMSE joined the exact fast lane at `0.7870x` total fit and
   `0.7666x` tree-build geometric-mean reference time;
2. **Inapplicable:** counts-carrying oblivious variant → `min_child_samples`
   is a leaf-wise/hybrid semantic and cannot be added as an exact engine
   optimization;
3. **Closed:** feature-mask and row-subset fusion was exact and fast in all
   cells, but failed the frozen all-cell paired-timing stability requirement.
   Automatic subset dispatch is off and the selected reference kernels remain.

No kernel deletion is authorized by E1. See
[`benchmarks/fused_subset_oblivious_result.md`](benchmarks/fused_subset_oblivious_result.md).

**E2. Large-n advantage — closed at the frozen claim threshold.** Profiling
rejected fused+subtraction at the production thread count. The retained
fused/uint8/capped-border system reached a `1.2793x` equal-size geometric-mean
fit speedup over ChimeraBoost 0.15 on the matched 500k/1M numeric lane, with
quality neutrality and every non-speed gate passing. That misses the frozen
`1.30x` claim bar, so no large-n advantage claim or threshold retry is
authorized. See
[`benchmarks/large_n_engine_result.md`](benchmarks/large_n_engine_result.md).

**E3. Float32 histogram streams** (ROADMAP R5, opt-in, already implemented):
measure on the throughput harness; promote to a size-gated auto lane only
behind regret gates.

**E4. Multiclass/distributional attribution — complete.** The fixed 50k-row
profile found every measured path tree-build-limited. Gaussian LightGBM is the
only selected drill-down; no loss-kernel optimization is justified.

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
TabArena categorical sets (Diamonds class) and I3's now-spent categorical
stratum as development-only. Confirmation requires a new target-unseen,
contamination-screened categorical panel frozen before C1 outcome inspection;
I3 cannot serve as confirmation again.

**C3. Ladder.** Basketball first (should be a no-op there — no declared
ordinals; gate is exactness + no-engagement proof), then dev tier, then
fresh confirmation. Success also unblocks re-running the deferred mode-mix
diagnostic (plan item 11): if oblivious+ordinal absorbs the Diamonds-class
wins, the hybrid/depthwise deletion evidence finally exists.

## Track Z — Cleanup to 1.0 (bundle every break loudly)

The package grew to ~24.4k lines because deletions were correctly blocked on
replacement proofs. E1 did not authorize its planned kernel deletions; C3 is
the remaining mode-deletion proof. The rest is policy. Do this as a
**deprecation release (0.10) → 1.0** cycle:

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
- `linear_residual`: the L2 comparison arm established local-linear
  dominance, so the 0.10 warning cycle is active; delete at 1.0.
- `hybrid` stays until C3 produces the mode-mix evidence. `target_ordered_
  cat_codes` stays until C1 replaces it properly.

**Z2. E1 kernel deletions — closed without deletion.** The subset fused
candidate failed its frozen stability gate, so the selected reference kernels
remain production code rather than removable oracles. Any future size target
must come from the 1.0 API/mode removals after their own evidence, primarily
C3; the former E1-driven 24.4k → ~15k trajectory is withdrawn.

**Z3. Review findings — complete.** Categorical predict validation no longer
does a Python cell scan; infinity, empty prediction batch, and duplicate
categorical-index semantics are explicit and documented; warmup parses
conventional falsy values and prevents unsafe concurrent Numba workqueue use.

**Z4. Docs**: root planning docs (KALMAN_READINESS_PLAN,
LINEAR_RESIDUAL_BOOSTING_PLAN, DISTRIBUTIONAL_*SPEC, fable_supervisor_handoff,
BEST_OF_BOTH_PLAN once superseded) → `docs/archive/`; stand up an mkdocs site
(their docs/ layout is a good template: getting-started, parameters, concepts,
shap, faq); README gets the Pareto headline (I4).

**Z5. Tests**: mark campaign verifiers (`-m campaign`), split CI jobs
(library suite vs campaign suite), prove identical coverage before/after —
this was deferred pending exactly that proof; do it as its own PR.

**1.0 criteria**: Z1 deletions executed, Z3 decisions shipped and documented,
the retained P2 optimization and its non-certification stated accurately,
suite green on the version matrix, NOTICE current, and CHANGELOG in
Keep-a-Changelog form with every break enumerated.

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

**I3. Fresh confirmation registry — complete and spent for Track L.** The
contamination-screened, fingerprinted 20-lineage registry is frozen and
stratified smooth/process, categorical, and noisy-tabular. Its Track-L outcomes
are now spent; any reuse must follow the registry's declared evidence rules.

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
S4 sports suite + S-track confirmation · ~~L3 fresh-panel confirmation~~
(closed; no lockbox) · ~~E2 large-n beat-their-engine protocol~~ (closed below
the claim bar) · C1–C3 categorical program with a new confirmation panel →
mode-mix rerun → C3-gated deletions · 1.0 release.

**Explicitly not in this program:** reopening any closed candidate as shaped;
touching the lockbox before its preconditions; TabArena as anything but
report-only; rewriting frozen artifacts; deleting ahead of proofs.

## What success looks like

1. A certified sports claim on a preregistered multi-season panel: DarkoFit
   (single model or explicit ensemble API) above CatBoost's 0.5363-class
   scores with cold-player quality intact.
2. A certified smooth-panel claim over ChimeraBoost 0.15 from a materially new
   selector and fresh evidence; the closed 3% selector cannot supply it.
3. A future engine claim beyond the closed P2/E2 protocols; current evidence
   supports the retained exact optimizations but not the frozen all-case
   predict or ≥1.3× large-n certifications.
4. A 1.0 release: ~28-param constructor, ~12-15k-line package, every break
   documented, docs site, Pareto headline — the library their creator would
   recognize as disciplined, doing things theirs cannot (distributions,
   uncertainty, entity-aware sports modeling, versioned serialization).
