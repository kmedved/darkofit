# Changelog

## Unreleased

- Adopt the evidence-calibrated shipping policy: Tier-E opt-in APIs,
  behavior-exact engine work, recipes, and measurements ship on correctness
  plus honest scoped reporting; Tier-D defaults retain fresh preregistered
  confirmation with power, uncertainty, concentration, harm, and cost checks.
  Win counts and binary engineering-certification bars are no longer gates.
- Add the opt-in `preset="accuracy"` regression profile for the frozen A10
  managed fields, with fitted resolution metadata, concrete refit export, and
  safe model round trips. The historical development advantage and its
  Diamonds concentration/inference-cost caveats are documented together.
- Add opt-in `selection_rounds` for capped automatic tree-mode auditions,
  followed by a fresh full-budget fit of the selected mode. The default
  `None` path is unchanged.
- Add opt-in held-out split-conformal prediction intervals for Gaussian
  distributional regressors. Calibration rows are isolated from fitting,
  selection, early stopping, learning-rate probing, and distribution
  calibration; the finite-sample order statistic, provenance metadata, and
  safe NPZ round trips are covered. Parametric intervals remain the default,
  while weighted conformal calibration and full-data refit fail closed.
- Complete the frozen 75-coordinate distributional flagship against NGBoost,
  CatBoost uncertainty, and quantile LightGBM. Conformal DarkoFit has the
  smallest equal-dataset mean absolute 90%-coverage error (`0.0110`) and
  `0.9831×` parametric-DarkoFit width on this panel; the result remains
  descriptive and reports coverage beside width.
- Re-score the immutable SynthGen df1 artifact against the corrected,
  later-confirmed outcome ledger. Its agreement moves from 6/9 to 8/9 when
  exactly the random-strength and fixed-linear development labels are
  superseded. It is admitted only as a retrospective probe-tier direction
  finder and remains inadmissible as confirmation or promotion evidence.
- Close the five-member row-OOB ensemble as a sports automatic policy on the
  fresh 2014–2016 basketball panel. It passed the frozen 3× fit/predict/RSS
  budgets but worsened player-disjoint aggregate RMSE by 2.31%, lost eight of
  nine target-season lineages, and worsened cold-player RMSE by 2.87%.
  Current DarkoFit remained 2.81% better than ChimeraBoost 0.15.0 and 5.26%
  worse than CatBoost 1.2.10 in equal-lineage aggregate RMSE on the same
  coordinates; no global or sports-profile default changes.
- Complete T7b, the fixed-learning-rate, three-seed CatBoost attribution
  follow-on. None of the tested stochastic, sampling, one-hot, L2, or
  leaf-estimation directions explains the historical DarkoFit gap.
  `l2_leaf_reg=1` remains a promising development configuration; one-hot 255's
  large mean improvement failed its uncertainty and worst-task gates.
- Close Panel 3 at its preregistered spent-data power gate. The repaired
  117-result calibration reproduced the original substantive outcomes exactly.
  T5 reached `50.00%` simulated pass probability (one-sided Wilson lower bound
  `48.61%`) and guarded cross-features reached `10.64%` (`9.82%`), versus the
  required `80.00%`. The valid create-only decision retains zero candidates
  and authorizes no fresh target access, registry, confirmation run, default
  change, or product claim; the CTR23 lockbox remains sealed.
- Add the Tier-E `n_ensembles` API for deterministic row or group bootstrap
  bagging. Members select their horizons on OOB rows; regression averages
  predictions, classification soft-votes, SHAP values are averaged, and
  safe nested NPZ persistence remains pickle-free. Numeric-only fits share a
  target-free preprocessor, while categorical and ordinal fits deliberately
  use member-local preprocessing.
- Enable the already-tested behavior-exact fused histogram/split kernels for
  eligible selected-row and selected-feature oblivious fits. All old frozen
  gate outcomes remain immutable; the engine route ships under Tier-E because
  its predictions and canonical model state were exact in all eight cells.
- Generate a hash-bound Tier-E engineering measurements page from immutable
  same-machine artifacts, and add honest feature recipes for robust heads,
  random strength, linear leaves, declared ordinals, and the accuracy profile.
- Preserve zero- and one-category ordered pandas vocabularies under
  `ordinal_features="auto"` during stepwise cross-validation, and validate
  explicit evaluation-set width before ordinal transformation so malformed
  inputs raise the documented `ValueError` instead of a raw `IndexError`.
- Correct non-decision-changing prose errors in the frozen prediction and
  fused-subset benchmark reports without rewriting their hash-bound artifacts.
- Make the campaign collection hook composable with pytest and third-party
  ignore hooks by returning `None` for modules included in the selected
  partition.
- Make generated release benchmark status fail closed when any of its five
  frozen source artifacts changes, rather than merely recording a new hash in
  regenerated output.
- Keep all five optional TabArena adapter-integration tests collectable without
  AutoGluon, so the documented library/campaign partition remains stable
  across development and CI environments as coverage grows.
- Apply the exact `campaign` and `not campaign` pytest partitions before module
  import, preventing optional or Python-3.10-only campaign tooling from leaking
  skips and compatibility requirements into the library matrix.

## 0.10.0 - 2026-07-17

### Release infrastructure

- Split the full 1,913-test union into an explicit 767-test library lane and
  1,146 frozen campaign verifiers. CI runs the library lane across Python
  3.9, 3.11, and 3.13 and the campaign lane separately on Python 3.11.
- Correct SynthGen's inherited cross-platform golden contract. Immutable
  dataset-byte goldens now run only on their Darwin/arm64 reference platform;
  a separate canonical recipe/RNG-stream hash runs everywhere, so Linux CI
  still detects versioned recipe drift without mistaking floating reduction
  differences for changed synthetic data.
- Add a strict MkDocs user guide and archive superseded planning documents
  without presenting them as current API contracts.
- Add generated release benchmark status and Pareto tooling. It keeps the
  frozen 13-dataset general-regression and nine-cell sports panels separate,
  binds every source artifact by SHA-256, and records the honest conclusion:
  ship the deprecation/docs/infrastructure release without promoting a new
  quality policy or claiming a certified all-case engine win.
- Close native-ordinal C2 development without opening its target-unseen
  confirmation panel. The candidate improved equal-task RMSE by 0.72%, but a
  categorical FPS task regressed by 31.75% and its worst split by 143.59%;
  no default, mode mix, deletion, or lockbox action follows.
- Clarify an ambiguous frozen prediction-artifact counter: all eight public
  prediction medians were no slower than ChimeraBoost, while six of eight also
  passed the paired-ratio stability gate. The immutable artifact is preserved
  and current status tooling labels both counts.

### Product and engine changes

- Add `ordinal_features` to `DarkoRegressor.fit`,
  `DarkoClassifier.fit`, and `DarkoStepwiseSearchCV.fit`. Explicit mappings
  encode declared ranks as numeric features; `"auto"` recognizes ordered
  pandas categoricals and integer-coded categorical columns. The fitted
  resolution is recorded, unknown prediction categories fail closed, and safe
  model round trips preserve the declaration.
- Preserve already-contiguous C- and Fortran-order float64 feature blocks
  through preprocessing instead of forcing a row-major copy. The matched
  campaign retained byte-identical bins and predictions; its formal
  all-case throughput claim remained closed on timing stability.
- Add exact fused histogram/split construction for eligible variable-Hessian
  oblivious fits, including binary Logloss and weighted RMSE. Ineligible
  sampled, noisy, low-thread, and other lanes retain their reference kernels.
  The later generalized subset-lane experiment was not promoted.

### Input compatibility decisions

The shared input-validation layer intentionally changes some previously
accepted inputs. The final compatibility posture is:

1. **Infinity is rejected by default at fit, explicit evaluation, and
   prediction.** This is a breaking safety default: infinity usually signals
   an upstream arithmetic or data-cleaning error. Migrate by clipping it or
   replacing it with NaN. Trusted legacy pipelines may instead wrap fitting
   (including an explicit evaluation set) and prediction in
   `sklearn.config_context(assume_finite=True)`; this disables feature-matrix
   infinity checks at all three boundaries. Unchecked positive and negative
   infinity in numeric features use DarkoFit's existing missing-value bin
   rather than a finite extreme bin. The escape hatch does not relax
   finite-value requirements for targets, weights, or other non-feature
   inputs.
2. **Zero-row prediction batches are supported.** The temporary rejection has
   been removed: prediction, probability, staged, multiclass, distributional,
   categorical, and linear-residual paths return outputs with zero rows and
   their normal trailing dimensions. Fit and explicit evaluation sets remain
   non-empty, and empty prediction batches must still have the fitted feature
   count and column schema.
3. **Duplicate `cat_features` entries remain rejected.** This is an intentional
   breaking change because duplicates—especially a name and index resolving
   to the same column—often reveal caller mistakes. Migrate by fixing the
   caller or applying stable deduplication such as
   `list(dict.fromkeys(cat_features))` before fitting.

These policy decisions refine the behavior authorized by the frozen
[basketball input-validation campaign](benchmarks/basketball_input_validation_result.md).
They do not change its valid-input prediction, archive, or timing evidence.
The campaign's prediction-only `assume_finite` wording is superseded by the
consistent fit/evaluation/prediction escape hatch above.

### Warmup environment safety

`DARKOFIT_WARMUP` now has an explicit, case-insensitive grammar:
`0`/`false`/`off`/`no`/empty disable it; `1`/`true`/`on`/`yes` run blocking
warmup; and `background`/`thread`/`bg` request background warmup. Surrounding
whitespace is ignored. Unknown values warn and skip warmup instead of
accidentally doing expensive import work.

Background warmup is now single-flight. DarkoFit fit and prediction calls wait
before entering Numba until it completes, preventing the default non-threadsafe
Numba `workqueue` backend from terminating the process on concurrent parallel
execution. A programmatic background request made after `workqueue` is active
raises with migration guidance; the environment dispatcher warns and skips so
package import remains usable. Start background warmup during import/startup,
use blocking `warmup()` after Numba is active, or configure a thread-safe Numba
backend such as TBB before import. Numba thread masks are thread-local, so the
warmup thread's existing snapshot/restore does not mutate a caller thread.
This hardening preserves the frozen basketball campaign's blocking-warmup
timing and exactness evidence; its background completion check did not exercise
concurrent DarkoFit work.

### Group-aware automatic validation

Add explicit `validation_strategy="group"` support to the sklearn wrappers.
It requires `groups=` and records the requested strategy. Supplying `groups=`
with the historical default `validation_strategy="random"` remains grouped for
backward compatibility.

### Deprecations for 1.0

The following evidence-free or concluded experimental surfaces remain
functional during the 0.10 deprecation cycle and emit `FutureWarning` when a
non-default value selects them. They will be removed in DarkoFit 1.0:

- `tree_mode="depthwise"` and its `"levelwise"` alias. Migrate to
  `"catboost"` for symmetric trees or `"lightgbm"` for non-oblivious trees.
- `histogram_dtype`, `leaf_dtype`, and `histogram_parallelism`. Remove these
  arguments to use the supported float64 histogram, int64 leaf-ID, and
  automatically selected feature-parallel paths.
- `auto_learning_rate_probe`, `auto_learning_rate_probe_values`, and
  `auto_learning_rate_probe_iterations`. Use a validation-backed explicit
  learning-rate search outside the estimator instead.
- `bootstrap_type="bayesian"` and `bagging_temperature`. Use
  `bootstrap_type="none"` with uniform, GOSS, or MVS row sampling.
- `sampling="weighted_goss"`. Use plain `"goss"` or `"mvs"`.
- `linear_residual`, `linear_residual_alpha`, `linear_residual_features`,
  `linear_residual_fit_intercept`, and `linear_residual_standardize`. For
  scalar RMSE smooth data, use the more accurate local
  `linear_leaves=True`. For other supported losses, explicitly detrend the
  target before fitting and add the deterministic trend back to location
  predictions.

The older `sigma_calibration` alias continues to emit `DeprecationWarning`;
use `dist_calibration` before 1.0.

`random_strength` is deliberately **not** deprecated because its
preregistered retirement screen defended the public capability. The broader
three-target, three-season S4 confirmation did not validate `0.5` as a sports
recommendation: equal-cell R² changed by only `+0.000036`, 6 of 9 cells lost,
the leave-one-cell-out gate failed, and fit cost was `1.835x`. That candidate
is closed without retuning, and `random_strength=0.0` remains the global
default. The same S4 run put current DarkoFit defaults above ChimeraBoost
0.15.0 on all nine cells but below CatBoost 1.2.10 on 8 of 9. The two Gaussian
`rho_*` multipliers also remain: focused distributional tests defend their
independent head-scaling behavior and safe serialization. `hybrid` and
`target_ordered_cat_codes` remain pending their declared comparison campaigns.

The `linear_residual` deprecation follows the frozen smooth development
campaign: local linear leaves improved equal-task RMSE by 7.97% versus
DarkoFit default, beat global residual on all three datasets and 20 of 21
splits, while global residual improved only 0.48% and regressed on space_ga.
The comparison is development evidence and does not promote linear leaves as
a default.

### Detailed 0.10 implementation and campaign record

* Harden the shared wrapper/core input boundary: reject masked, complex,
  infinite-by-default, sparse, empty training/evaluation, and malformed inputs
  before lossy conversion while preserving shape-correct empty prediction;
  preserve nullable frame missing values; resolve named categorical columns;
  enforce fit-time feature names and order; publish NaN/sparse sklearn tags;
  and honor `assume_finite=True` for feature infinity checks at fit, explicit
  evaluation, and prediction.
  Full sklearn estimator checks retain only the documented sample-weight row-
  replication deviation. The layer adapts Apache-2.0 ChimeraBoost 0.15.0
  validation behavior, recorded in `NOTICE`.
* Add exact interventional TreeSHAP for scalar oblivious-tree regressors and
  binary classifiers, including constant and local-linear leaves, original-
  feature grouping for categorical expansions, caller-supplied or bounded
  deterministic fitted backgrounds, and safe `.npz` persistence. Unsupported
  multiclass, distributional, global-linear-residual, non-oblivious, and
  over-16-player cases fail explicitly. The independent-oracle tests and full
  suite pass. On the frozen basketball fold plus corrected cold-player
  guardrail, DarkoFit and ChimeraBoost 0.15.0 produced byte-identical
  attributions, expected values, and predictions. Median SHAP time was 33.581ms
  versus 32.686ms (1.027x), with both timing series stable. The implementation
  is adapted from Apache-2.0 ChimeraBoost commit
  `ff6f248d09f92d608ed8cc366463b61f1af04acc`, recorded in `NOTICE`; see the
  [basketball confirmation](benchmarks/basketball_tree_shap_result.md).
* Close the experimental five-member OOB-ensemble promotion attempt without
  adding an API or changing defaults. A frozen six-block basketball
  confirmation reproduced every prediction fingerprint and all quality gains:
  mean ten-fold R² improved by 0.003876, held-team R² by 0.006126, and the
  585-row cold-player R² by 0.019349. Wall timing and paired cost were stable at
  a 2.414x median wall ratio, but the default arm's summed prediction timing
  had IQR/median 0.235 against the preregistered 0.20 limit. The protocol's
  no-rerun rule therefore closes the attempt; see the
  [confirmation result](benchmarks/basketball_oob_ensemble_confirmation_result.md).
* Fuse unit-Hessian oblivious histogram construction and shared-split scanning
  into one feature-parallel launch for the proven full-row/full-feature lane
  at three or more threads. Weighted RMSE, classification, sampled rows or
  columns, injected root histograms, random split noise, and one- or two-thread
  fits retain their existing kernels. The readable implementation remains an
  exact-equality oracle and no public parameter or model format changed. On
  the frozen basketball creator folds plus held-team and cold-player
  guardrails, automatic dispatch produced byte-identical models and identical
  R² while reducing median fit time by 33.2% (28.93s to 19.31s) and steady
  wall time by 32.7% (29.46s to 19.83s), with stable reciprocal blocks and
  essentially flat RSS. Expanded archive-exact tests cover categorical RMSE,
  MAE, Quantile, callbacks, and exact refit; ineligible weighted and binary
  lanes prove fallback. The design is adapted from Apache-2.0 ChimeraBoost
  commit `a04430657fb82c806ee2a039506c99944a27accc`, recorded in `NOTICE`.
* Route oblivious-tree leaf-ID updates through an exact serial kernel below
  32,768 training rows, avoiding one parallel launch per tree level while
  retaining the previous parallel implementation at and above the cutoff.
  The frozen basketball campaign matched all ten folds, serialized models,
  fitted metadata, held-team predictions, and the 585-row cold-player subset
  byte-for-byte. Median fit time fell 45.0% (20.00s to 11.01s) and steady wall
  time fell 43.7% (20.57s to 11.59s), with stable reciprocal blocks and flat
  RSS. Combined with the preceding fused-kernel change, median basketball fit
  time is down about 62.0% from 28.93s without changing R² or any public API.
  The serial-twin design and cutoff are adapted from Apache-2.0 ChimeraBoost
  commit `a04430657fb82c806ee2a039506c99944a27accc`, recorded in `NOTICE`.
* Confirm the optimized constant-leaf training engine against synced
  ChimeraBoost 0.15.0 on the frozen basketball boundary. With identical
  1,000-tree core parameters, every fold and player-guardrail prediction was
  byte-identical; DarkoFit median fit and wall time were 2.5% and 2.4% lower,
  respectively, with 0.8% higher RSS. Current product defaults answer a
  different question: DarkoFit took 1.312× as long while ChimeraBoost retained 64–163 trees,
  while mean R² differed by only 0.000232 and DarkoFit led cold-player R² by
  0.00955. This is characterization evidence only and does not change
  DarkoFit's early-stopping, linear-leaf, or selection defaults.
* Add a bounded packed predictor for scalar LightGBM-mode leafwise forests.
  The route is limited to exactly two resolved threads, at least five trees,
  at least 32,768 row-tree work units, and no more than 32,768 rows; other
  leafwise, multiclass, distributional, hybrid, and large-batch calls retain
  the per-tree loop. The frozen basketball confirmation reproduced every
  fitted-state, archive, prediction, staged-output, dispatch, storage, and
  memory oracle exactly. Public prediction improved 1.44x on the reserved
  524-row fold, 1.59x on 585 genuinely cold players, 1.66x on 2,409 held-team
  rows, and 1.98x at 8,192 repeated rows. At 65,536 and 100,000 rows the
  deliberately retained fallback remained within 1.021x of the reference;
  see the
  [basketball confirmation](benchmarks/basketball_leafwise_packed_prediction_result.md).
* Add experimental, default-off `linear_leaves=True` for scalar RMSE
  CatBoost/oblivious fits. Each tree can attach Hessian-weighted local ridge
  models over its numeric split features without changing DarkoFit's split
  search; small leaves retain their exact constant Newton value. The feature
  includes a packed prediction forest, weighted feature standardization,
  deterministic small/all-categorical fallbacks, safe format-v4 `.npz`
  persistence with corruption checks, fit diagnostics, and Apache-2.0
  attribution for the adapted ChimeraBoost solver/design. Existing default
  archives keep their prior format and constructor payload. The mechanism is
  not an automatic policy. Its first frozen basketball selector screen failed
  mean, leave-one-fold-out, held-team, and cold-player quality gates, so it
  remains opt-in research and does not advance to broader development
  validation.
* Close ChimeraBoost 0.15.0's pairwise categorical-combinations donor
  mechanism before any DarkoFit port. In the frozen four-feature categorical
  basketball view, six pair columns reduced mean ten-fold R² by 0.001024,
  held-team R² by 0.046489, and genuinely cold-player R² by 0.091404. They
  also made held-team prediction 2.843x slower and cold-player prediction
  1.653x slower. Exact routing, repeatability, numeric non-engagement, timing
  stability, and memory gates passed, so this is a mechanism rejection rather
  than a harness failure. No API, default, donor code, broader campaign, or
  lockbox spend is authorized; see the
  [basketball result](benchmarks/basketball_categorical_combinations_result.md).
* Fix symmetric/shared split legality so an already-pure leaf's empty child
  contributes zero gain instead of vetoing a useful split for every other
  active leaf. Sparse non-empty children still obey `min_child_weight` and the
  hybrid shared trunk still enforces `min_child_samples`; per-leaf builders
  remain strict. The correction applies to serial, parallel, noisy, and
  count-aware shared searches, including `l2_leaf_reg=0`, and tracks the last
  positive Hessian bin so float32 cancellation cannot disguise a sparse child
  as structurally empty. On the 13-dataset
  TabArena regression check it reduced the geometric-mean RMSE gap to the
  unrelated ChimeraBoost 0.13 default from 5.14% to 1.25%. A fixed learning
  rate of 0.1 improved the aggregate by a further 0.40% but regressed four
  datasets. A subsequent staged multisplit check on four deliberately hard
  tasks selected `l2_leaf_reg=1`, `max_bins=128`, and `learning_rate=0.1` as a
  candidate; it improved six untouched outer splits by 1.25%, won 19 of 24
  paired comparisons, and narrowed the matched ChimeraBoost gap from 3.00% to
  1.57%. Because the tasks were selected from prior residual gaps, this
  advances the candidate to a broader dataset gate rather than changing the
  automatic defaults. See
  [the multisplit report](benchmarks/tabarena_regression_multisplit_ablation.md).
* Add public fit-time boosting callbacks, including a monotonic soft
  `WallClockStopper`, across scalar, multiclass, and distributional boosters.
  Fitted models now record requested, attempted, completed, retained, and best
  rounds plus their stop reason; the metadata survives safe `.npz` round trips
  without serializing callback objects. The TabArena adapter records the same
  diagnostics for every bag child so cap saturation and deadlines are directly
  auditable. Automatic tree-mode selection shares one callback/deadline across
  its CatBoost, LightGBM, and hybrid candidates and records each candidate's
  rounds, score, learning rate, stop reason, selection state, and deadline
  timing. A frozen 444-job, 3,552-child TabArena comparison then retained
  the 1,000-round scalar-regression horizon: 10,000 rounds improved
  equal-dataset test RMSE by 0.453% but missed the required 0.5% gate and added
  12.88% training time plus 10.65% inference time. See the
  [cap-horizon result](benchmarks/tabarena_regression_cap_horizon_result.md).
  A subsequent isolated 156-job, 1,248-child mechanism screen advanced only
  the source-declared safe ordinal representation: it improved equal-dataset
  test RMSE by 19.50% across Airfoil and Diamonds and won all six screen
  splits, but remains exploratory preprocessing evidence rather than a new
  default. Automatic tree-mode selection improved RMSE by 3.10% but failed
  the frozen inference-cost gate at 2.57x the control; four target-statistic
  permutations, generic safe one-hot, and linear-residual boosting also
  failed their predeclared gates. See the
  [follow-on screen result](benchmarks/tabarena_regression_followon_screen_result.md).
  A source-frozen mechanism replication on the remaining 33 Airfoil and
  Diamonds coordinates confirmed the ordinal accuracy signal: versus the
  identical fixed native policy it improved equal-dataset test RMSE by 17.29%,
  won all 13 repeat blocks and all 33 coordinates, and passed every accuracy
  gate. It nevertheless did not advance because its 1.265x inference-time
  ratio exceeded the predeclared 1.25x ceiling. These coordinates had already
  appeared in the cap-horizon campaign, so this is mechanism replication, not
  independent dataset generalization or support for generic ordinal inference.
  See the
  [ordinal confirmation result](benchmarks/tabarena_regression_ordinal_confirmation_result.md).
  A final source-frozen same-machine default characterization compared
  DarkoFit 0.9.0 with ChimeraBoost 0.14.1 and CatBoost 1.2.10 over 117 primary
  jobs, plus 18 separate Airfoil/Diamonds representation jobs, for 1,080 fitted
  children. Equal-dataset test RMSE put DarkoFit 1.25% behind ChimeraBoost
  (95% CI 0.82%-1.73% behind; five of 13 dataset wins) and 5.38% behind
  CatBoost (95% CI 5.11%-5.68% behind; one of 13 wins). DarkoFit trained 9.1%
  faster than ChimeraBoost and 62.7% faster than CatBoost, while inference was
  43.3% and 25.6% slower and incremental memory was 43.4% and 84.7% lower,
  respectively. All jobs completed without failure or imputation and with no
  known deadline/time-limit stops; 494 competitor-child stop reasons remained
  unresolved and are reported as unknown. The Airfoil diagnostic restores a
  numeric physical angle from OpenML's nominal/string task schema, so the
  separate representation lane remains descriptive and does not advance a
  generic ordinal policy. See the
  [same-machine result](benchmarks/tabarena_regression_same_machine_result.md).
  A subsequent source-frozen accuracy shootout evaluated a 10,000-round,
  fixed-0.1-LR, 128-bin, L2=3 profile with validation-selected
  CatBoost/LightGBM/hybrid tree mode. It passed every development gate:
  equal-dataset test RMSE was 2.44% lower than ChimeraBoost 0.14.1, 3.64%
  lower than the DarkoFit product default, and 2.78% lower than the identical
  fixed CatBoost-mode arm. Diamonds supplied 87.6% of the ChimeraBoost
  advantage; without it the edge was only 0.33%, so this is development parity
  rather than a broad superiority claim. The candidate remained 1.54% behind
  CatBoost 1.2.10 and won only two of 13 datasets. All 78 jobs and 624 selected
  children completed without failure, imputation, deadline, restart, recovery,
  or time-limit stop. Because the quality-only run allowed swap-in, timing and
  memory-performance evidence is inadmissible. The profile freezes unchanged
  for unseen confirmation and does not change defaults. See the
  [accuracy-shootout result](benchmarks/tabarena_regression_accuracy_shootout_result.md).

## 0.9.0 - 2026-07-12

Behavior-changing default improvements from a full-repo review, plus targeted
performance and robustness fixes. These are intentional clean cutovers; the
previous behaviors remain available through explicit parameters.

* Make `ordered_boosting="auto"` task-aware in CatBoost/depthwise modes: the
  ordered leave-one-out leaf update stays on for classification and turns off
  for scalar regression. Categorical regression continues to use ordered
  target-statistic preprocessing to prevent leakage. On the real-data
  guardrail matrix plain boosting improved three of four numeric case means;
  weighted Diabetes was effectively neutral. It also improved every Abalone
  split. On House Prices, ordered boosting was catastrophically unstable on
  one of three splits (2.6x test RMSE with a healthy-looking validation score)
  while mildly better on the other two — a tail-risk failure a small
  validation set cannot catch. The numeric result also closes ~93% of the
  QSAR TabArena gap in `docs/archive/HANDOFF.md`. `MAE`/`Quantile` recompute
  leaf values from residual statistics and never applied the ordered update;
  `"auto"` now resolves off for them and explicit `ordered_boosting=True`
  raises instead of being silently ignored. The resolved rule is recorded
  under `auto_params_["tree"]["ordered_boosting_rule"]`.
* Default `eval_train_loss=False` on the boosters and sklearn wrappers. The
  per-round training-loss pass is diagnostic-only (early stopping watches the
  eval set) and cost about 15% of multiclass fit time; `verbose=True` still
  forces it on, and `train_history_` is empty unless it is enabled.
* Change `DarkoSearchCV` refit defaults to `refit_rounds="median_best"` and
  `refit_learning_rate="fold_median"` so the final refit matches the
  early-stopped configuration CV actually scored, instead of retraining the
  full nominal iteration budget with early stopping disabled. `"preserve"`
  recovers the old semantics. Fold round counts of zero (folds with no legal
  split) participate in the median; missing fold metadata (e.g. resumed
  pre-0.9 studies) falls back to preserve semantics with a warning. Missing
  or invalid fold learning-rate metadata does the same, and both fallback
  sources are recorded in tuning metadata; non-finite fold learning rates are
  filtered before the median.
* Allocate small stepwise tuning budgets across phases by largest remainder
  instead of dumping rounding leftovers into the last (lowest-priority)
  phase, and warn when requested trials cannot be scheduled by the configured
  phases.
* Parallelize the Gaussian/StudentT/LogNormal/Poisson/NegativeBinomial NLL
  and Gaussian CRPS eval kernels with `prange`, keeping zero-weight rows
  unevaluated so extreme values cannot poison the metric with `0 * inf`.
  The multiclass softmax evaluators stay deliberately serial: their inner
  class loop trips numba's parfor analysis on Python 3.13 + numba 0.66,
  which is inside the declared support range. The NegativeBinomial
  dispersion refresh drops the `lgamma(y+1)` constant from its
  golden-section objective, removing redundant work from the
  profile-likelihood search.
* Speed up single-threaded leaf-wise fits: when the left child is smaller,
  the serial refill lane now scans only that child and derives the sibling
  by parent subtraction (mirroring the parallel lane) instead of rebuilding
  both children.
* Share fitted preprocessing between the selection fit and full-data refit
  for explicit-eval-set fits: the preprocessing cache key no longer includes
  the eval set (the cached artifacts never depend on it), `refit=True`
  enables the cache exactly when the refit trains on the same rows, and the
  cache is emptied at fit end so retained models no longer pin binned-matrix
  copies in memory.
* Avoid converting the whole feature matrix to object dtype on every predict
  when reading the `per_metric_affine` calibration column from NumPy inputs.
* Vectorize the Poisson/NegativeBinomial tuning scorers with
  `scipy.special.gammaln` instead of per-row Python `math.lgamma` loops.
* Report corrupt model archives with unknown losses or invalid constructor
  params as `invalid DarkoFit model` `ValueError`s instead of raw
  `KeyError`/`TypeError`.
* Remove the unused `_build_counts_into`/`_build_counts_rows_into` kernels
  and the dead `BIN_DTYPE` constant, and size the multiclass leaf-wise
  `changed_leaves` buffer to `max_leaves` to eliminate a latent
  out-of-bounds hazard on future full rescores.

## 0.8.0 - 2026-07-09

* Rename the distribution and import package from `chimeraboost` to `darkofit`.
  This is an intentional hard break while the project has no external users.
* Rename the public sklearn estimators to `DarkoRegressor` and
  `DarkoClassifier`, and the tuning helpers to `DarkoSearchCV` and
  `DarkoStepwiseSearchCV`.
* Point package metadata at the canonical DarkoFit repository and issue
  tracker.
* Reframe the package around tabular machine learning, with gradient boosting
  as its current core rather than a permanent package boundary.

## 0.7.0 - 2026-07-08

* Add opt-in sklearn-wrapper linear residual boosting for `RMSE`, `MAE`,
  `Quantile`, `Gaussian`, and `StudentT` regressors. The wrapper fits a
  weighted ridge trend on raw numeric columns, trains the booster on residuals,
  adds the trend back to point/location predictions, and persists the trend in
  the plain-array `.npz` archive format.
* Intentionally reject `linear_residual=True` for `LogNormal`, `Poisson`, and
  `NegativeBinomial` until those losses have distribution-specific offset
  protocols.
* Standardize Gaussian, StudentT, and LogNormal canonical targets internally
  during distributional training, then map public raw predictions, intervals,
  samples, and distribution parameters back to the original target scale. This
  fixes scale-dependent Newton behavior for raw-unit targets, but it changes
  numerical results for new Gaussian/StudentT/LogNormal fits and is therefore
  a minor-version release.
* Preserve backward compatibility for older distributional archives by loading
  them with the target transform disabled, and keep wrapperless scalar archives
  aligned with their saved fitted loss when loaded through
  `ChimeraBoostRegressor.load_model()`.
* Update calibrated distributional `SearchCV` refits so
  `refit_rounds="preserve"` uses the median fold-best horizon when calibration
  was learned at fold-best prefixes, and aggregate small-fold calibration
  warnings across all folds.
* Add a numeric ndarray/DataFrame fast path for the linear residual trend so
  selected numeric columns are sliced from one float matrix conversion instead
  of converting the whole design matrix to object dtype once per column.
* Refresh the public synthetic calibrated-Gaussian Chimera lane after target
  standardization in
  `benchmarks/distributional_standardization_check.md`. The private WNBA/DARKO
  artifacts remain pre-0.7 and should be rerun before production replay or
  release-performance claims that depend on those exact numbers.

## 0.6.0 - 2026-07-07

* Add native Gaussian distributional regression with
  `ChimeraBoostRegressor(loss="Gaussian", tree_mode="lightgbm")`, including
  `predict_dist`, `predict_interval`, `sample`, `.npz` save/load support, and
  Gaussian-specific guardrails for unsupported v1 training modes.
* Support Gaussian distributional fits with uniform row subsampling
  (`subsample < 1`) and column subsampling (`colsample < 1`) in LightGBM mode,
  including capped sampled-depth-zero retries so an unlucky empty/no-split
  sample does not stop the whole fit prematurely.
* Add Gaussian `eval_metric="crps"` validation/early-stopping support while
  keeping Gaussian NLL as the default validation objective.
* Add opt-in Gaussian `sigma_calibration="scalar"` on the sklearn regressor:
  the wrapper fits a validation-set global sigma scale at the selected best
  prefix, persists it through `.npz` save/load, and applies it to
  `predict_dist`, `predict_interval`, and `sample` without changing raw scores
  or point predictions. Fits with fewer than 200 effective calibration rows
  record a `small_sigma_calibration_fold` diagnostic warning.
* Generalize distributional calibration under `dist_calibration`, including
  global affine scale calibration and `dist_calibration="per_metric_affine"`
  for grouped affine scale maps keyed by `dist_calibration_feature` (for
  example `metric_code`). Grouped calibration is preserved through
  prediction APIs, SearchCV refits, and `.npz` save/load.
* Add WNBA DARKO real-data validation artifacts for per-metric affine Gaussian
  calibration plus a scalar Kalman shadow replay that injects
  `predict_variance()` as row-level `R_t` against the incumbent
  `sigma2 / sample_weight` heuristic.
* Enable `ChimeraBoostStepwiseSearchCV` for Gaussian regressors on the
  LightGBM lane with Gaussian NLL default scoring and Gaussian-safe
  sampling/regularization suggestions.
* Add `benchmarks/bench_distributional.py` for Gaussian NLL/CRPS/coverage
  comparisons against fixed-round and early-stopped Chimera Gaussian lanes,
  RMSE constant-sigma, quantile-pair, NGBoost, CatBoost uncertainty, and
  LightGBM twin-model baselines when optional packages are installed, including
  coverage binned by predicted sigma.
* Change sklearn estimator defaults to `l2_leaf_reg="auto"`; the resolver keeps
  CatBoost-mode fits near the historical `3.0` default while preserving the
  task/tree-mode-specific auto-structure metadata in `auto_params_`.
* Require `early_stopping` to be a Boolean on sklearn estimators, so string
  values such as `"auto"` or `"false"` no longer activate early stopping by
  truthiness.
* Replace the previous automatic learning-rate heuristic with a transparent
  CatBoost-form selector keyed to loss, resolved iteration budget, eval-set
  presence, and Kish effective sample size.
* Add an LR-only ChimeraBoost correction for materially weighted RMSE fits in
  CatBoost/oblivious-tree mode; unweighted and all-ones-weight fits keep the
  raw CatBoost-form learning rate.
* Damp unweighted LightGBM-mode automatic learning rates as a provisional
  tree-mode-specific correction; weighted projection fits stay on the hotter
  effective-sample-size corridor.
* Add a bounded high-dimensional automatic learning-rate shrinkage based on the
  post-preprocessing model feature count relative to Kish effective sample size;
  the multiplier is recorded in `auto_params_["learning_rate"]`.
* Add `use_best_model=True` default behavior for fits with validation data,
  keeping the best validation prefix even when early-stopping patience does not
  fire. Set `use_best_model=False` to keep every fitted tree.
* Resolve default early-stopping patience from the fitted learning rate when
  `early_stopping=True` and `early_stopping_rounds` is left unset, using
  `ceil(5 / lr)` clipped to `20..200`.
* Add `early_stopping_min_delta`, preserving the legacy `1e-9` tolerance by
  default while allowing explicit numeric tolerances or opt-in
  `early_stopping_min_delta="auto"` resolution from baseline validation loss.
  Min-delta controls patience resets; best-prefix selection uses the true
  validation argmin.
* Add opt-in `validation_fraction="auto"` and regression
  `validation_strategy="weighted_stratified"` for automatic validation splits
  that account for effective sample size and weighted target quantiles, with
  feasible-strata caps for small default validation fractions. The realized
  split policy is recorded separately from the requested strategy.
* Raise the default boosting budget to `iterations=1000` and the default
  numeric bin budget to `max_bins=254`.
* Expose `auto_params_` on fitted boosters and preserve it through `.npz`
  serialization. The metadata records resolved learning rate, effective sample
  size, feature counts, tree sizing, regularization, binning, early stopping,
  sampling, validation split policy, target statistics, and threading context
  from the last fit.
* Preserve scalar refit wrapper metadata through `.npz` save/load and mark the
  fold-selection model as intentionally non-persistent with
  `selection_model_persisted_=False` on loaded wrappers.
* Add `auto_params_["diagnostics"]` for low effective sample size and automatic
  learning-rate clipping, with throttled runtime warnings controlled by
  `diagnostic_warnings={"once","always","never"}`. Diagnostics also record
  weighted-binning activation, observed bin counts, feature expansion, and the
  best-prefix policy.
* Add auto structure defaults (`l2_leaf_reg` by default on sklearn estimators,
  and opt-in `depth`, `num_leaves`, `min_child_samples`, `min_child_weight`,
  and `cat_smoothing`) with resolved values recorded under
  `auto_params_["auto_structure"]`; refit helpers freeze these resolved values
  along with the resolved learning rate.
* Add opt-in sklearn-wrapper learning-rate probing with
  `auto_learning_rate_probe=True`, recording candidate scores, the selected
  explicit learning rate, and the final-budget automatic base rate under
  `auto_params_["learning_rate_probe"]`.
* Add opt-in CatBoost-like stochastic regularization: Bayesian bootstrap
  (`bootstrap_type="bayesian"` / `bagging_temperature`), MVS row sampling
  (`sampling="mvs"` / `mvs_reg`), and deterministic split-score noise
  (`random_strength`). The resolved settings and per-fit sampling diagnostics
  are recorded under `auto_params_["stochastic_regularization"]` and persisted
  through `.npz` save/load.
* Add `sampling="weighted_goss"` as an opt-in sample-weight-aware GOSS variant;
  existing `sampling="goss"` behavior is unchanged.
* Make numeric bin construction respect non-uniform `sample_weight` through
  weighted quantile borders. `sample_weight=None` and all-ones weights keep the
  previous unweighted behavior.
* Add sklearn-wrapper refit helpers: `get_refit_params()`, selected-round and
  resolved-learning-rate properties, and optional `refit=True` full-data refits
  after early-stopping selection.
