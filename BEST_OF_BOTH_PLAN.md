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

### Current execution status (2026-07-17)

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
- The second engine port now dispatches leaf-ID descent to an exact serial
  twin below 32,768 rows. On the same basketball boundary it matched every
  model and sports guardrail byte-for-byte while reducing median fit time from
  20.00s to 11.01s and wall time from 20.57s to 11.59s. The strict suite
  produced 1,479 passes and 23 skips before timing.
- Together, the fused kernel and serial descent reduce median basketball fit
  time by about 62% from 28.93s. This clears the aspirational 13-second target
  but remains above ChimeraBoost's diagnostic 7.52-second run. Basketball stays
  the first fast fatal gate for subsequent engine or quality work.
- A source-frozen same-machine comparison against ChimeraBoost 0.15.0 now
  resolves that remaining diagnostic gap. With the same 1,000 constant-leaf
  trees and common core parameters, every fold and player-guardrail prediction
  was byte-identical; DarkoFit's median fit and wall ratios were 0.975 and
  0.976. Low-level default-tree fit parity is achieved. Under current product
  defaults DarkoFit took 1.312× as long because ChimeraBoost retained only 64–163 trees,
  but mean R² differed by just 0.000232 and DarkoFit led cold-player R² by
  0.00955. That is a policy trade-off, not permission to promote early stopping.
- Exact interventional TreeSHAP is shipped for the supported scalar-oblivious
  regression and binary-classification lanes. On the frozen basketball fold
  and cold-player guardrail it matched ChimeraBoost predictions, expected
  values, and attributions byte-for-byte at a 1.027x median runtime ratio.
- The separately frozen OOB-5 stable confirmation is complete and closes the
  ensemble attempt. It reproduced the original +0.003876 mean R² and +0.019349
  cold-player R² gains, exact behavior, stable wall timing, and a 2.414x median
  wall cost. However, default prediction IQR/median was 0.235 versus the frozen
  0.20 limit. The no-rerun rule therefore rejects API work despite every other
  gate passing; see `basketball_oob_ensemble_confirmation_result.md`.
- The shared input-validation and sklearn-compliance layer is shipped. Its
  frozen six-block basketball campaign reproduced creator-fold, held-team,
  and cold-player predictions, feature importance, fitted metadata, and the
  serialized model exactly in all 12 fresh workers. Validated prediction was
  1.025x the `assume_finite` arm by ratio of medians, and every timing and
  stability gate passed. Both wrappers also passed the frozen scikit-learn
  1.7.2 `check_estimator` gate with only the preregistered expected failure.
  This closes the compatibility item without changing a model default or
  spending CTR23/TabArena evidence; see
  `basketball_input_validation_result.md`.
- The donor's automatic numeric cross-feature selector was screened before
  any DarkoFit port. Against the same ChimeraBoost defaults, automatic crosses
  reduced mean basketball R² by 0.001042 and cold-player R² by 0.013881; the
  selector activated on four folds and improved only one. The slight
  +0.000476 overlap-exposed team-holdout gain does not rescue a sports
  candidate that fails unseen players. Do not port the automatic selector or
  spend broader evidence on it. `cat_combinations` remains a separate
  categorical research question; see
  `basketball_cross_features_donor_screen_result.md`.
- Binary temperature scaling is also closed before implementation. On the
  frozen starter target it improved log loss on only 5 of 10 creator folds,
  worsened pooled log loss/Brier/ECE, and worsened all three metrics on both
  held teams and the 585 cold-player rows. Runtime and exact monotonicity
  passed, but the external sports guards did their job: no classifier
  calibration API or broader panel is authorized; see
  `basketball_temperature_scaling_result.md`.

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
| Basketball steady fit (10 folds) | 11.59 s after two exact engine ports | achieved ≤ 13 s with quality preserved; next target is same-machine parity |
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
| Speed (basketball steady, 10 folds) | Two exact engine ports reduced DarkoFit from 29.46 s to ~12 s. In a frozen matched 1,000-tree lane, DarkoFit/Chimera fit and wall ratios were 0.975/0.976 with byte-identical predictions. Under product defaults DarkoFit took 1.312× as long because Chimera retained 64–163 trees. | Default-tree fit parity achieved; remaining product gap is policy, while packed prediction remains slower |
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
4. **Generic validation-selection framework** in `sklearn_api`: defer this
   abstraction until a concrete client proves that internal validation
   decisions generalize across the external basketball and cold-player
   boundaries. The linear-leaf selector chose its candidate on every fold but
   failed four of five quality gates; the donor's cross-feature selector chose
   four folds but improved only one and materially hurt cold players. Do not
   build a generic framework around those rejected policies. Remaining
   research is scoped separately:
   - **safe-ordinal categorical lane**, only after eliminating its failed
     causal inference overhead and expanding beyond two declared datasets;
   - opt-in `preset="accuracy"` = today's A10 (auto tree-mode at 2–3× cost)
     for users who explicitly request it.
5. **`n_ensembles` bagging with OOB early stopping** (port ~80 lines from
   their `_fit_bagged`). Basketball evidence says this is the quality ceiling
   on small noisy data (0.5402 vs our 0.5267).
   **Closed 2026-07-17:** the separately frozen six-block confirmation
   reproduced all prediction hashes and quality gains, passed wall stability,
   paired-ratio stability, and cost gates, but failed the absolute prediction
   stability gate because default IQR/median was 0.235 against a 0.20 limit.
   The preregistered decision is `close_oob_ensemble_attempt`: do not add the
   API or rerun this candidate. Preserve the result only as research evidence.
6. **Calibration ports**: temperature scaling for `DarkoClassifier`
   (validation split, monotonic, predict unchanged) and the split-conformal
   quantile offset for `loss="Quantile"` — both natural fits for our
   uncertainty brand and nearly free. Evaluate the conformal idea against our
   distributional heads too (`predict_interval` + conformal correction).
   **Quantile-offset candidate closed 2026-07-17:** the frozen basketball
   screen improved summed pinball loss on all 10 creator folds and repaired
   interval coverage on pooled, held-team, and cold-player views. It failed
   the fatal width budget on all three views, widening intervals by 35.7%,
   41.6%, and 43.7% against a 25% ceiling. Do not implement or retune this
   candidate, and do not spend broader evidence on it.
   **Classifier-temperature candidate closed 2026-07-17:** validation-fitted
   positive temperatures improved only 5 of 10 creator folds and worsened
   pooled, held-team, and cold-player log loss, Brier score, and ECE. The
   candidate also exceeded the frozen worst-fold limit. Exact class decisions,
   within-model ranking, runtime, and memory gates passed, but no product
   implementation or broader evidence is authorized. Distributional-interval
   calibration remains a separate, untested mechanism.
7. **`darkofit.warmup()`** + `DARKOFIT_WARMUP=1`: three tiny synthetic fits
   covering default fit/predict kernels. Directly fixes the fresh-worker
   timing tax we've measured on TabArena-style harnesses.
   **Shipped 2026-07-17:** a frozen six-block basketball campaign passed every
   correctness, isolation, stability, and timing gate. Array-exact
   creator-fold, held-team, and 585-row cold-player predictions were preserved
   across all 12 fresh workers. Median first fit fell from 3.1236 s to
   1.6067 s (0.5144x), and first prediction from 99.67 ms to 3.82 ms
   (0.0384x). Median explicit warmup cost 4.7186 s, so this is an opt-in API
   for reusable or overlap-capable workers, not a lower-work one-shot cold
   start. Ordinary imports remain cold; no model default changed; CTR23 was
   not used.
8. **Input-validation/compliance layer**: compare their `_validate_fit_input`,
   `_check_predict_input`, feature-name enforcement, and nullable-dtype
   handling against ours; port or adapt only missing behavior with focused
   compatibility tests. Attribute substantial literal ports in `NOTICE`.
   **Shipped 2026-07-17:** the shared wrapper/core boundary now rejects masked,
   complex, infinite, sparse, empty, and malformed inputs with compatible
   messages; handles nullable frame-like inputs and named categorical
   features; enforces fitted feature names; publishes sklearn tags; and
   preserves metadata across safe serialization. The frozen basketball gate
   passed all exactness and timing conditions across ordinary, held-team, and
   cold-player views. Median validated prediction was 2.6245 ms versus
   2.5600 ms with `assume_finite` (`1.0252x`); both wrappers passed the frozen
   scikit-learn 1.7.2 compliance gate. No model default changed and no broad
   quality claim is authorized.

Gate for Phase 1: basketball first, including cold-player and clean timing.
Survivors proceed to the spent 13-task TabArena development panel and the 243
unused-but-spent CTR23 coordinates for development only, then to a genuinely
fresh preregistered panel for any promotion claim. Require sign tests and
inference within 1.10× of today's default unless a narrower feature-specific
gate is frozen in advance.

### Phase 2 — Screen donor quality mechanisms before any port

9. **Linear leaves** into the oblivious path: hessian-weighted per-leaf ridge
   over the tree's numeric split features on standardized bin centers,
   min-1000-rows guard, constant fallback, hand-rolled LU (JIT hygiene),
   packed linear-forest predict kernel.
   **Closed for automatic use 2026-07-17:** the mechanism is implemented as
   explicit, default-off research, with safe persistence and prediction
   fallbacks. Its validation selector failed mean-fold, held-team, and
   cold-player basketball gates. Do not expose the selector, change a default,
   or advance it to the 243-coordinate development panel. A future smooth-data
   study would be a new research question, not continuation of this candidate.
10. **Treat numeric crosses and categorical combinations separately.**
    **Numeric automatic selector closed 2026-07-17:** an isolated screen of
    the donor's top-pair diff/prod implementation lost 0.001042 mean R² and
    0.013881 cold-player R² on basketball, with only one external improvement
    among four selected folds. Do not port or promote it.
    All-categorical `cat_combinations` remains unresolved and needs its own
    future categorical protocol; the numeric basketball screen supplies no
    evidence for or against that distinct mechanism.
11. **Mode-mix diagnostic deferred.** Do not rerun A10 selector shares with
    linear leaves or the ordinal lane while their automatic policies remain
    rejected. Reopen this diagnostic only if a materially different mechanism
    first passes basketball and the appropriate categorical or smooth-data
    confirmation gate.

Gate for Phase 2: basketball first, then the 243 unused CTR23 confirmation
coordinates as development data under a new frozen protocol. Before any
lockbox shot, preregister a design whose simulated pass probability is at
least 80% under the confirmed development effect distribution and satisfy a
separate fresh-data gate. Only then may the one-shot lockbox be considered.
Neither current Phase 2 mechanism is authorized to enter that larger panel.

### Phase 3 — Engine consolidation (speed parity, bit-identity throughout)

12. **One fused build+split kernel** for the oblivious path (their design:
    active-leaf list, per-feature bin bounds, interleaved single buffer,
    transposed gain scan, serial twin below ~32k rows, in-place descent).
    Keep exactly one readable reference pair as the test oracle. Delete the
    current ~60-variant histogram/split kernel matrix; runtime branches
    replace compile-time forks wherever the branch is measurably free.
    **Partially shipped 2026-07-16:** the proven unit-Hessian full-row/full-
    feature lane now fuses histogram construction and shared-split scanning,
    and leaf descent uses an exact serial twin below 32,768 rows. The two
    behavior-exact ports reduced basketball fit time by about 62% and cleared
    the 13-second target. Broader kernel unification and deletion remain
    unearned; unsupported lanes still retain their reference implementations.
    A subsequent matched-core comparison against ChimeraBoost 0.15.0 produced
    byte-identical predictions and slightly favored DarkoFit fit time. Stop
    optimizing this training path absent a newly measured regression; do not
    delete fallback kernels merely to reduce code size.
13. **Leafwise path**: keep the segment/subtraction design but collapse its
    variant axes the same way; port the packed row-major predict treatment so
    `tree_mode="lightgbm"` (and the accuracy preset) stops paying the 2.57×
    inference tax.
    **Bounded prediction route shipped 2026-07-17:** the scalar leafwise
    packed kernel passed the frozen basketball, cold-player, held-team,
    exactness, persistence, memory, and timing confirmation. Public prediction
    improved 1.44× on the reserved 524-row fold, 1.59× on 585 cold players,
    1.66× on 2,409 held-team rows, and 1.98× at 8,192 repeated rows. The route
    deliberately remains empirical and bounded to two resolved threads and at
    most 32,768 rows: direct packed execution was 1.127× and 1.210× the
    per-tree loop at 65,536 and 100,000 rows, while public fallback stayed
    within 1.021×. Do not make leafwise packing unconditional without a new
    protocol.
14. **Exact TreeSHAP** for oblivious (+ linear leaves) — port nearly verbatim;
    it depends only on the packed-forest layout. Ship as
    `model.shap_values(X)`; document leafwise as unsupported initially.
15. **Speed targets** (basketball steady harness, unchanged): a Phase 1
    candidate may target ≤ 13 s only while preserving every quality guardrail;
    Phase 3 targets ≤ ~10 s at equal tree counts and predict throughput within
    1.3× of ChimeraBoost's fused kernels.

### Phase 4 — Deletion sweep

This sweep is not currently authorized. Item 11's selector-share diagnostic
was deferred because both proposed automatic mechanisms failed basketball;
therefore none of the mechanism-dependent deletions below may proceed. Treat
the table as a review ledger, not an execution queue.

| Delete | Where | ~Lines | Rationale |
| --- | --- | ---: | --- |
| `depthwise`/levelwise mode | tree.py builders/classes, serialization pack/unpack, flat_model | ~800 | Not even in the A10 candidate set; no evidence it ever wins |
| `hybrid` mode | tree.py + selector slot | ~300 | Retain: the required selector-share evidence is unavailable and the linear-leaf selector was rejected |
| Kernel matrix | tree.py | ~2,500–3,000 | Replaced by fused kernels (Phase 3) |
| `random_strength` noise kernels | tree.py `_with_noise_py` ×5, booster plumbing | ~450 | Evidence-free; their cascade killed the analogous knobs |
| `bootstrap_type` bayesian + `bagging_temperature`, weighted-GOSS variants | booster.py | ~400 | Keep MVS + plain GOSS only |
| `linear_residual` (5 params, module) | linear_residual.py + sklearn_api | ~900 | Retain pending a separate review; rejected automatic linear leaves do not supersede it |
| `histogram_dtype`/`leaf_dtype`/`histogram_parallelism` params | booster.py | ~200 | Concluded experiments; fix the winning defaults |
| `auto_learning_rate_probe*` (3 params) | sklearn_api/booster | ~250 | Delete only if a promoted policy proves it has no remaining unique value |
| `target_ordered_cat_codes` experiment param | booster/preprocessing | ~150 | Replaced by the ordinal validation lane |
| Root planning docs (KALMAN_READINESS_PLAN, LINEAR_RESIDUAL_BOOSTING_PLAN, DISTRIBUTIONAL_*SPEC, fable_supervisor_handoff) | repo root | n/a | Move to `docs/archive/`; root keeps README/CHANGELOG/ROADMAP/HANDOFF |
| Spent-campaign runner copies | benchmarks/ | ~15,000+ | Phase 0 harness extraction; archive frozen artifacts read-only |

Constructor target ≈ 28 params: iterations, learning_rate, depth, l2_leaf_reg,
max_bins, subsample, colsample, sampling(MVS/GOSS/uniform), cat_smoothing,
ts_permutations, cat_features, loss, alpha, dist_params, min_child_weight,
min_child_samples, num_leaves, tree_mode(oblivious/leafwise), linear_leaves,
linear_lambda, cat_combinations, early_stopping,
validation_fraction, early_stopping_rounds, thread_count, random_state,
verbose (+ refit, eval_metric under review). Rejected OOB-ensemble and numeric
cross-feature parameters are not part of this target.

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
- **Selection lanes double fit time and can select the wrong sports model.**
  Do not add a generic automatic lane from the rejected linear-leaf or
  cross-feature policies. Keep `linear_leaves=True/False` explicit and
  default-off; require external basketball and cold-player proof before any
  new selector is considered.
- **Mode deletion sacrifices real accuracy.** Keep leafwise permanently;
  retain hybrid/depthwise while the Phase 2 selector-share measurement is
  deferred; the accuracy preset keeps opt-in auto-selection available.
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

1. **Completed:** Phase 0 goldens and the shared basketball boundary; exact
   fused/serial engine ports; same-machine engine parity; TreeSHAP; warmup;
   input validation and sklearn compliance.
2. **Closed:** current-auto-LR early-stop/refit, OOB-5 API work, automatic
   linear-leaf selection, automatic numeric cross features, and the dependent
   mode-mix rerun. Do not repeat or enlarge those campaigns.
3. **Closed isolated mechanism:** split-conformal quantile offsets failed the
   basketball interval-width gates despite improving coverage and pinball
   loss. Binary temperature scaling independently failed pooled, held-team,
   cold-player, fold-breadth, and worst-fold quality gates. Existing Gaussian
   scalar calibration improved NLL, CRPS, and 80% coverage on all basketball
   boundaries and won all 10 creator folds, but failed the frozen interval
   width cap by widening intervals roughly 2.1x. Do not retune or enlarge
   these campaigns, and do not promote a default from them.
4. **Basketball remains next:** every new mechanism starts on the unchanged
   creator folds, overlap-exposed held-team view, and 585-row cold-player
   subset. A failure on any sports guardrail stops it before broader data.
5. **Separate categorical track:** safe ordinal and all-categorical
   `cat_combinations` each require their own protocol; numeric basketball
   evidence cannot promote or reject them.
6. Only a basketball survivor may enter the 243 development coordinates,
   then a genuinely fresh preregistered gate with sufficient simulated power;
   only then consider the one-shot lockbox.
7. Resume kernel work only for a newly measured regression. Phase 4 deletions
   remain blocked on replacement behavior proof, not line-count targets.
8. Phase 5 R&D infrastructure can proceed independently where it does not
   consume promotion evidence.
