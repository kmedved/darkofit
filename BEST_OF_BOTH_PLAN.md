# DarkoFit 1.0: Best-of-Both-Worlds Plan

*Drafted 2026-07-16 from a review of ChimeraBoost @ `29602d3` (the frozen
basketball comparator; version 0.14.2 + ~40 commits) against DarkoFit @
`20e6ee8`. Reviewed and corrected against DarkoFit @ `3295f70` and the synced
ChimeraBoost local/origin/upstream heads, all still exactly `29602d3`. Numbers
described as DarkoFit evidence below come from frozen artifacts in
`benchmarks/`; claims attributed to ChimeraBoost's research remain upstream
claims until independently reproduced here.*

### Review corrections that govern execution

- Basketball is the mandatory fast screen for every candidate: unchanged ten
  folds, overlap-exposed team holdout, 585-row cold-player subset, prediction
  fingerprints, and reciprocal clean timing. A basketball failure stops the
  candidate before a larger panel.
- Basketball is not sufficient by itself to promote a universal default.
  Survivors still need broader and genuinely fresh evidence.
- Early stopping is a candidate mechanism, not a presumed new default. The
  current-auto-LR early-stop/exact-refit arm was only 11.7% faster and regressed
  mean and overlap-exposed basketball R².
- The safe-ordinal mechanism produced a large accuracy gain but formally
  failed its frozen causal inference-time gate. It is research evidence, not a
  verified product lane waiting to be switched on.
- The 243 confirmation coordinates not used in the minimal CTR23 run are now
  development-only: their task identities and neighboring outcomes are spent
  for final claims. They cannot serve as fresh confirmation.
- Line-count and API claims use the checked source: ChimeraBoost contains
  4,029 package lines (3,755 in the six principal modules) and its regressor
  has 27 constructor parameters; DarkoFit contains 22,057 package lines and
  its regressor has 58 parameters.

### Current execution status (2026-07-16)

- The first fused engine port is complete. DarkoFit now combines its existing
  unit-Hessian histogram builder and shared split scan in one feature-parallel
  launch for the narrow proven lane; all unsupported paths retain their exact
  previous kernels.
- Basketball remained the first fatal gate throughout. The final automatic
  promotion run matched every fold, held-team and cold-player prediction,
  feature importance, fitted metadata payload, and serialized model byte.
  Median fit time fell from 28.93s to 19.31s (33.2%) and steady wall time from
  29.46s to 19.83s (32.7%).
- Expanded exactness coverage passed for categorical RMSE, MAE, Quantile,
  callbacks, and early-stop/exact-refit fits. Weighted RMSE, binary
  classification, and one- or two-thread fits proved they remain on the
  reference fallbacks.
- This closes roughly one third of DarkoFit's basketball runtime, but it does
  not reach the aspirational 13-second target or ChimeraBoost's diagnostic
  7.52-second run. The next engine work must again pass basketball first.

## 0. Thesis

ChimeraBoost wins on **engine discipline and product defaults**; DarkoFit wins
on **feature depth** (distributional heads, tuning, serialization, multiple
tree shapes). The plan is to adopt the useful discipline — validation-selected
candidates, fused kernels with bit-identity oracles, and a fast decision-suite
R&D loop — while keeping our distributional/uncertainty product as the
differentiator. Default changes and deletions are earned outcomes, not inputs.

Target end state:

| | Today | Target |
| --- | ---: | ---: |
| `darkofit/` package lines | 22,057 | ~9,000 |
| DarkoRegressor constructor params | 58 | ~28, only if evidence supports deletion |
| Tree modes | 4 (catboost/lightgbm/hybrid/depthwise) | 2 (oblivious/leafwise) |
| Default regression posture | fixed 1000 rounds, no ES | unchanged until a candidate passes basketball and broader gates |
| Basketball steady fit (10 folds) | 27.2–28.3 s | aspirational ≤ 13 s only with quality preserved |
| Distributional heads | 5 losses + predict_dist/interval/sample | kept, hardened, promoted |

---

## 1. ChimeraBoost review (what they actually built)

### 1.1 Core: 4.0k lines, one tree shape, little waste

| Module | Lines | Content |
| --- | ---: | --- |
| `sklearn_api.py` | 1,537 | validation, ES auto-split, selection lanes, bagging, calibration |
| `tree.py` | 944 | oblivious trees incl. linear leaves, packed predict, exact SHAP |
| `booster.py` | 664 | scalar + multiclass boosters, MVS, LOO ordered step, callbacks |
| `preprocessing.py` | 255 | numeric + ordered-TS + cat-combos + cross-feature blocks |
| `losses.py` | 187 | RMSE, Logloss, MAE, Quantile, MultiSoftmax — nothing else |
| `binning.py` | 168 | quantile borders, uint16 bins, NaN bin, per-feature budgets |

These six principal modules total 3,755 lines. The complete package is 4,029
lines after including `target_encoding.py`, `warmup.py`, and `__init__.py`.

Engine facts that matter for us:

- **One fused `_build_and_split` kernel** per level (histogram build + split
  scan in a single parallel launch, cache-hot per feature), with an
  **active-leaf list** so empty leaves are neither zeroed nor scanned, per-feature
  `n_bins` bounds on every scan, and a transposed (leaf-outer) gain scan. The
  old readable kernels are retained purely as **exact-equality test oracles**.
  Result: small-n fit 1.2–1.35× faster, bit-identical across 17 configs.
- **One reusable histogram buffer** per fit, shape `(F, 2^depth, bins, 2)` with
  grad/hess interleaved so each scatter touches one cache line.
- **In-place leaf descent** (`leaf = (leaf<<1) + (x>t)`) — the numpy version of
  this was measured at ~⅓ of their fit time before they killed it.
- **Serial twins** below n=32,768 where parallel fork/join costs more than the
  pass; all dispatches bit-identical by construction.
- **Packed-forest predict**, parallel over samples, consuming the binner's
  row-major output directly (no transpose copy): 1.26 M rows/s on their bench,
  1.3× LightGBM, ~2.6 M rows/s constant-leaf. (CatBoost C++ remains ~10×.)
- **Linear leaves**: per-leaf hessian-weighted ridge over the tree's *own
  numeric split features* evaluated on standardized bin centers; leaves with
  `< 2(k+1)` rows fall back to constant; `< 1000` training rows disables the
  feature entirely; hand-rolled LU solve chosen specifically to avoid numba's
  LAPACK bindings (~25% of cold-start JIT).
- **Exact interventional TreeSHAP** — tractable because oblivious trees have
  ≤ depth distinct features per tree, so the Shapley game is enumerated
  exactly, *including linear-leaf slopes*, mapped back to original columns,
  averaged across bags. Compile-time and JIT hygiene are explicit design
  considerations throughout.

### 1.2 Product layer: the "it just works" posture

- **Early stopping ON by default** since 0.10: `n_estimators=2000` as a
  *ceiling*, auto 20% holdout (stratified / group-aware), patience 50,
  `learning_rate=None → 0.1` under ES. Their changelog: "out-of-box defaults
  match the benchmarked/Pareto configuration exactly." Degrades gracefully on
  tiny data (ES silently off rather than raising).
- **The validation-selected lane pattern** (their signature move): fit
  variants, keep whichever scores better on the *already-held-out* ES split.
  Applied to: `linear_leaves=None` (fixed-on was 16W/12L — a wash with
  casualties; selection banked 20W/9T/7L, −0.58% mean vs constant, OpenML
  one-shot 8W/7T/1L), `cross_features=None` (top-6-importance numeric pairs →
  diff/prod columns, refit, keep-if-better; Grinsztajn 51W/8L, +1.5% mean),
  temperature scaling (classifier `predict_proba`), and a **split-conformal
  quantile offset** (restores tail coverage that lr-shrunk quantile steps
  starve; Romano et al. 2019).
- **`n_ensembles` bagging** with a detail worth stealing verbatim: each
  bootstrap member early-stops on its **own OOB rows** (an auto-split of a
  bootstrap would contain ~57% duplicate rows → optimistic val loss → ~38%
  extra trees). Members parallelize across processes with numba threads
  divided among them.
- **Guard-rail culture**: every auto feature has explicit engagement guards
  (loss type, row count, numeric-feature count) and records its decision
  (`linear_leaves_selected_`, `cross_pairs_`). Size-adaptive
  `min_child_weight` for classification (veto fades 500→2000 rows).
- **`warmup()`** precompiles all default-path kernels (or loads the disk
  cache); `CHIMERABOOST_WARMUP=1` does it at import. First-fit 9.3 s → 0.10 s
  inside timed sections. This exists specifically because TabArena's cluster
  re-times every fold in a fresh worker.
- **Input validation & sklearn compliance** at reference quality: named errors
  for every malformed input, predict-time feature-name/order enforcement,
  pandas-nullable/pyarrow/polars handling, masked-array rejection,
  `assume_finite` escape hatch, `check_estimator` compliance with documented
  deviations.

### 1.3 The R&D machine (why their defaults are good)

- **Sealed holdout discipline**: TabArena (Lite and Full) is report-only and
  never influences a source change. Decisions run synthetic → dev panel →
  Grinsztajn-59 (3 seeds, sign tests) + a real high-cardinality suite → an
  independent OpenML one-shot gate. PMLB is for HP tuning only.
- **SynthGen** (frozen `syn:v2`): unlimited prior-sampled synthetic datasets
  (SCM recipes, marginals bootstrapped from 1,644 OpenML profiles with
  TabArena's 51 members excluded at the source), exact Bayes floors, ~10%
  *canary* datasets already at ceiling — a flag that "wins" on canaries is
  exposed as variance injection. The suite itself is gated: it must reproduce
  ≥ 7/9 verdicts from their historical decision ledger before it may gate
  anything.
- **Kill discipline**: 0.13.0 deleted eight default-off experimental flags in
  one release (constructor 36 → 24 params) after their research cascade found
  each null or net-negative. The PAYOFF program (classification Brier gap)
  closed with *nothing shipped* — including killing ensembles-as-default at
  the Grinsztajn gate. Features must re-earn their existence.
- **Bit-identity culture**: every speed refactor ships with exact-equality
  kernel tests and a golden-prediction suite (395+ tests).
- **Pareto north star**: blended strength vs slowdown chart is the README
  headline; ship only frontier-pushing changes. Current claim: blended 99.4
  vs CatBoost's 98.1 with cross_features on.

### 1.4 Their gaps (our openings)

- **No distributional regression.** Five point losses; no NLL heads, no
  variance/interval/sampling API. Quantile + conformal offset is their only
  uncertainty story.
- **No serialization format** — pickle or nothing. No versioning, no payload
  validation.
- **No tuning package.**
- **One tree shape.** Oblivious-only; our A10 evidence says non-oblivious
  modes win 54.5% of validation selections (Diamonds rejects oblivious 24/24).
  Their answer to that regime is cross_features/cat_combinations, which
  recover *some* of it.
- **Multiclass is a second-class citizen**: no linear leaves, no
  cross_features, no SHAP.
- **`n_ensembles` stays manual** (their own gate killed it as a default).
- Windows-first dev environment quirks; docs claim defaults are
  "Grinsztajn-tuned", so some overfit-to-suite risk is acknowledged.

---

## 2. Head-to-head evidence (all verified campaigns)

| Axis | Evidence | Verdict |
| --- | --- | --- |
| Default quality (TabArena 13-task) | D/M = 1.0125 (5/13 wins); D/CatBoost = 1.0538 (`tabarena_regression_same_machine_result.md`) | ChimeraBoost ahead on defaults |
| Ceiling quality | A10 (auto-mode+10k+l2=3) = 0.9756 vs live Chimera (−2.44%); mode selection is the lever (A10/B10 −2.78%, 11/2); but 2.57× inference | We can beat them at 2–3× cost; not shippable as-is |
| Unseen-data check (CTR23, 9 tasks) | A10/M point −5.80%, gate FAIL on power; wins are categorical tasks (−19…−23%), losses are smooth numerics (+2…+6%) where their linear leaves rule | Two named gaps: smooth data (theirs), categorical representation (ours to bank) |
| Categorical representation | Safe ordinal was −18.4% RMSE vs product default and +2.2% deployment-lane inference, but the causal `O/B` inference ratio was 1.265 versus a 1.25 ceiling | Strong mechanism; frozen policy decision failed |
| Speed (basketball steady, 10 folds) | Historical DarkoFit 28.30 s / Chimera 9.29 s / CatBoost 6.60 s; 99% in tree building. Fixed-LR ES+exact-refit reached 12.72 s but hurt quality; current-auto-LR ES+exact-refit reached 24.01 s median and also failed quality | Policy can buy speed, but no tested policy preserves the sports guardrail; engine work remains necessary |
| Small-noisy-data quality | Basketball: DarkoFit 0.5267 ≈ Chimera 0.5248; CatBoost 0.5363; Chimera ens-5 0.5402 | Bagging, not tuning, is the quality lever there |
| Engine numerics | 70/165 bit-identical splits vs Chimera in catboost mode | Same algorithm family; the fight is policy + features, not math |

Code-mass comparison:

| | DarkoFit | ChimeraBoost |
| --- | ---: | ---: |
| Core package | 22,057 lines | 4,029 lines (3,755 principal six) |
| Regressor params | 58 | 27 |
| Histogram/split kernels | ~100+ `@njit` variants (parallel/serial × rows-subset × selected-features × unit-hess × counts × subtraction-direction × class-layout × noise) | 1 fused kernel + 2 reference oracles + linear-leaf/predict/SHAP kernels |
| Tests | 34,442 lines / 31 files, 18 of which verify one-shot benchmark campaigns | 395+ focused tests incl. numerical-identity goldens |
| Benchmarks | 56,028 lines; five near-duplicate 2–5.5k-line campaign runners | one reusable harness + frozen suites + analysis scripts |

---

## 3. The plan

### Phase 0 — Safety net (no behavior change)

1. **Golden suite**: freeze stable prediction goldens for representative fits
   (regression/binary/multiclass/distributional × cat/numeric × each tree
   mode), plus exact-equality oracle tests for every kernel we intend to
   replace. This is the enabler for everything below; it is exactly how
   ChimeraBoost ships bit-identical refactors.
2. **Benchmark harness consolidation**: start with the live basketball data,
   split, score, fitted-metadata, prediction-hash, and reciprocal-timing
   boundary. Audit the larger campaign runners for copy drift, then extract
   only independently testable primitives. Do not rewrite or relocate frozen
   historical runners merely to reduce line count; their reproducibility takes
   precedence. Mark campaign verifiers separately only after proving CI and
   release-test behavior remains intact.

### Phase 1 — Evaluate product-layer candidates (lanes before defaults)

3. **Early-stopping policy candidates**: preserve today's default while
   testing isolated combinations of validation fraction, patience, round
   ceiling, learning-rate resolution, best-prefix selection, and refit. Keep
   group-aware/stratified auto-split and test graceful tiny-data degradation.
   Basketball is first and fatal: no candidate advances if mean folds,
   overlap-exposed holdout, or cold-player quality regresses. Only a candidate
   that survives sports/noisy data may enter broader regression,
   classification, weighted, and alternate-loss gates. A default flip is a
   later decision, not part of this implementation item.
4. **Generic validation-selection framework** in `sklearn_api`: fit variants,
   compare best validation loss, record `*_selected_`. First clients, in
   order of evidence strength:
   - **safe-ordinal categorical lane**, only after eliminating its failed
     causal inference overhead and expanding beyond two declared datasets,
   - **linear leaves** (once Phase 2 lands),
   - **cross_features** (port),
   - opt-in preset: `preset="accuracy"` = today's A10 (auto tree-mode at 2–3×
     cost) for users who ask.
5. **`n_ensembles` bagging with OOB early stopping** (port ~80 lines from
   their `_fit_bagged`). Basketball evidence says this is the quality ceiling
   on small noisy data (0.5402 vs our 0.5267).
   **Screened 2026-07-16:** an independent five-member DarkoFit prototype
   passed all five basketball quality gates (mean +0.003876 R²; cold-player
   +0.019349) but both arms failed the frozen timing-stability gate after a
   shared final-block slowdown. Formal decision: `advance_none`; preserve as
   promising evidence and do not add the API without a separately frozen,
   stable confirmation campaign.
6. **Calibration ports**: temperature scaling for `DarkoClassifier`
   (validation split, monotonic, predict unchanged) and the split-conformal
   quantile offset for `loss="Quantile"` — both natural fits for our
   uncertainty brand and nearly free. Evaluate the conformal idea against our
   distributional heads too (`predict_interval` + conformal correction).
7. **`darkofit.warmup()`** + `DARKOFIT_WARMUP=1`: three tiny synthetic fits
   covering default fit/predict kernels. Directly fixes the fresh-worker
   timing tax we've measured on TabArena-style harnesses.
8. **Input-validation/compliance layer**: compare their `_validate_fit_input`,
   `_check_predict_input`, feature-name enforcement, and nullable-dtype
   handling against ours; port or adapt only missing behavior with focused
   compatibility tests. Attribute substantial literal ports in `NOTICE`.

Gate for Phase 1: basketball first, including cold-player and clean timing.
Survivors proceed to the spent 13-task TabArena development panel and the 243
unused-but-spent CTR23 coordinates for development only, then to a genuinely
fresh preregistered panel for any promotion claim. Require sign tests and
inference within 1.10× of today's default unless a narrower feature-specific
gate is frozen in advance.

### Phase 2 — Port the two quality features that beat us

9. **Linear leaves** into the oblivious path: hessian-weighted per-leaf ridge
   over the tree's numeric split features on standardized bin centers,
   min-1000-rows guard, constant fallback, hand-rolled LU (JIT hygiene),
   packed linear-forest predict kernel. Default `None` = validation-selected
   for RMSE regression and binary classification, exactly like theirs.
   *This is the named fix for our smooth-data losses (kin8nm/grid/space_ga
   class — and the lockbox's naval/wave_energy/sarcos class).*
10. **cross_features + cat_combinations** in preprocessing (we already have
    the ordered-TS encoder and a `feature_map_`-equivalent need): diff/prod
    columns for top-6-importance numeric pairs, validation-selected;
    pairwise cat combos auto-on only for all-categorical data.
11. **Re-run the mode-mix diagnostic** (A10 selector shares) with linear
    leaves + ordinal lane active. Hypothesis: the 54.5% non-oblivious share
    collapses once oblivious trees get local slopes and honest categorical
    codes. This measurement decides Phase 4's mode deletions.

Gate for Phase 2: basketball first, then the 243 unused CTR23 confirmation
coordinates as development data under a new frozen protocol. Before any
lockbox shot, preregister a design whose simulated pass probability is at
least 80% under the confirmed development effect distribution and satisfy a
separate fresh-data gate. Only then may the one-shot lockbox be considered.

### Phase 3 — Engine consolidation (speed parity, bit-identity throughout)

12. **One fused build+split kernel** for the oblivious path (their design:
    active-leaf list, per-feature bin bounds, interleaved single buffer,
    transposed gain scan, serial twin below ~32k rows, in-place descent).
    Keep exactly one readable reference pair as the test oracle. Delete the
    current ~60-variant histogram/split kernel matrix; runtime branches
    replace compile-time forks wherever the branch is measurably free.
13. **Leafwise path**: keep the segment/subtraction design but collapse its
    variant axes the same way; port the packed row-major predict treatment so
    `tree_mode="lightgbm"` (and the accuracy preset) stops paying the 2.57×
    inference tax. `flat_model.py`'s empirical router dies; packing becomes
    unconditional per tree kind.
14. **Exact TreeSHAP** for oblivious (+ linear leaves) — port nearly verbatim;
    it depends only on the packed-forest layout. Ship as
    `model.shap_values(X)`; document leafwise as unsupported initially.
15. **Speed targets** (basketball steady harness, unchanged): a Phase 1
    candidate may target ≤ 13 s only while preserving every quality guardrail;
    Phase 3 targets ≤ ~10 s at equal tree counts and predict throughput within
    1.3× of ChimeraBoost's fused kernels.

### Phase 4 — Deletion sweep

With selector-share evidence from (11) in hand:

| Delete | Where | ~Lines | Rationale |
| --- | --- | ---: | --- |
| `depthwise`/levelwise mode | tree.py builders/classes, serialization pack/unpack, flat_model | ~800 | Not even in the A10 candidate set; no evidence it ever wins |
| `hybrid` mode | tree.py + selector slot | ~300 | Decide on post-linear-leaves selector share; expected to collapse |
| Kernel matrix | tree.py | ~2,500–3,000 | Replaced by fused kernels (Phase 3) |
| `random_strength` noise kernels | tree.py `_with_noise_py` ×5, booster plumbing | ~450 | Evidence-free; their cascade killed the analogous knobs |
| `bootstrap_type` bayesian + `bagging_temperature`, weighted-GOSS variants | booster.py | ~400 | Keep MVS + plain GOSS only |
| `linear_residual` (5 params, module) | linear_residual.py + sklearn_api | ~900 | Superseded by linear leaves (basketball +0.0004 mean; panel −1.07% with QSAR-TID-11 harm). Deprecate in the release that ships linear leaves, delete one release later |
| `histogram_dtype`/`leaf_dtype`/`histogram_parallelism` params | booster.py | ~200 | Concluded experiments; fix the winning defaults |
| `auto_learning_rate_probe*` (3 params) | sklearn_api/booster | ~250 | Delete only if a promoted policy proves it has no remaining unique value |
| `target_ordered_cat_codes` experiment param | booster/preprocessing | ~150 | Replaced by the ordinal validation lane |
| Root planning docs (KALMAN_READINESS_PLAN, LINEAR_RESIDUAL_BOOSTING_PLAN, DISTRIBUTIONAL_*SPEC, fable_supervisor_handoff) | repo root | n/a | Move to `docs/archive/`; root keeps README/CHANGELOG/ROADMAP/HANDOFF |
| Spent-campaign runner copies | benchmarks/ | ~15,000+ | Phase 0 harness extraction; archive frozen artifacts read-only |

Constructor target ≈ 28 params: iterations, learning_rate, depth, l2_leaf_reg,
max_bins, subsample, colsample, sampling(MVS/GOSS/uniform), cat_smoothing,
ts_permutations, cat_features, loss, alpha, dist_params, min_child_weight,
min_child_samples, num_leaves, tree_mode(oblivious/leafwise), linear_leaves,
linear_lambda, cross_features, cat_combinations, early_stopping,
validation_fraction, early_stopping_rounds, n_ensembles, thread_count,
random_state, verbose (+ refit, eval_metric under review).

**Explicitly kept** (our moat, hardened with dedicated goldens before any
engine work): the five distributional heads + CRPS eval +
`predict_dist/predict_variance/predict_interval/sample` + dist/sigma
calibration; `darkofit.tuning` (DarkoSearchCV/Stepwise, optuna backend);
versioned `serialization.py` (extend for linear-leaf fields; add a
ChimeraBoost-import shim only if ever useful); auto-LR for ES-off; ES→exact
refit (`get_refit_params`); callbacks + WallClockStopper; `groups` support;
`eval_metric`.

### Phase 5 — Adopt the R&D machine

16. **Decision suite**: port/reuse SynthGen's recipe (Apache-2.0) or build the
    DarkoFit equivalent — prior-sampled synthetic suites with Bayes floors and
    *earned* canaries, plus a real-data dev panel (Grinsztajn-59 regression
    subset + our CTR23-eligible spares), backtested against our own campaign
    ledger before it gates anything. This replaces multi-day frozen campaigns
    for *screening*; the frozen-campaign machinery remains for release gates
    and one-shot holdouts only.
17. **Pareto tracking**: a `benchmarks/make_pareto.py` equivalent (quality vs
    fit-slowdown vs CatBoost/LightGBM/ChimeraBoost) regenerated per release;
    it becomes the README headline and the ship/no-ship criterion.
18. **Kill calendar**: every opt-in param gets a review date; params that
    can't re-earn their place through the cascade get deleted (their 0.13.0
    move). CHANGELOG discipline in their Keep-a-Changelog style.

---

## 4. Risks and mitigations

- **Default flip breaks users and can hurt noisy sports data.** Do not schedule
  a flip until a candidate passes basketball, cold-player, broader regression,
  classification, weighted, alternate-loss, compatibility, and resource
  gates. If it ever earns promotion, ship a migration release with a loud
  CHANGELOG and an escape hatch restoring today's exact behavior. Decide
  `max_bins` 254→128 by measurement, not imitation.
- **Selection lanes double fit time when they engage.** Accept (they did):
  guards restrict to RMSE/binary, ≥1000/2000 rows; document
  `linear_leaves=True/False` to skip the double fit. Measure whether an
  accepted early-stopping candidate offsets the cost; do not assume it.
- **Mode deletion sacrifices real accuracy.** Keep leafwise permanently;
  delete hybrid/depthwise only after the Phase 2 selector-share measurement;
  the accuracy preset keeps auto-selection available regardless.
- **Distributional heads break during kernel consolidation.** They ride the
  vector-leaf paths — freeze goldens for all five heads first (Phase 0), and
  land Phase 3 behind exact-equality tests.
- **Benchmark overfitting** (their acknowledged risk). Our CTR23
  contamination registry + lockbox discipline already exceeds their rigor;
  keep TabArena sealed the way they do, and power-check every gate before
  running it (the CTR23 lesson).
- **License/attribution**: both repos are Apache-2.0. Ported code keeps
  headers where substantial; add a NOTICE entry crediting ChimeraBoost
  (bbstats) for linear leaves, fused-kernel design, SHAP, and validation
  patterns.

## 5. Suggested execution order (Codex-sized chunks)

1. Phase 0 goldens (1 PR) → basketball harness boundary (1 PR) → broader
   harness extraction only where duplication is proven (1–2 PRs).
2. Phase 1: ES candidate framework (1 PR), ordinal lane (1 PR),
   bagging + OOB-ES (1 PR), calibrations (1 PR), warmup + validation layer
   (1 PR).
3. Phase 2: linear leaves (2 PRs: kernels+booster, then selection+API),
   cross features (1 PR), mode-mix rerun (campaign config, cheap).
4. Development on the 243 unused-but-spent CTR23 coordinates → fresh-data and
   preregistered power gates → only then consider the lockbox one-shot.
5. Phase 3 kernels (3–4 PRs, each bit-identity-gated).
6. Phase 4 deletions (mechanical once 11's evidence is in).
7. Phase 5 R&D infra (parallel-izable with 3–6).
