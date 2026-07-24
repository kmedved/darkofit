# TabArena tree and tree-adjacent model dossier

*Research updated: 2026-07-23. TabArena grades remain frozen at the
2026-07-21 snapshot named below. This is an idea-intake document, not a
promotion protocol or a claim of expected benchmark improvement.*

## Executive decision

The review produces five different answers, depending on what DarkoFit is
trying to buy:

1. **Best small quality probe:** couple the existing automatic depth and L2
   rules with an effective-rows-per-modeled-feature branch. DarkoFit already
   resolves both values independently; the missing idea is the *joint*
   CatBoost-derived policy, not another generic auto mode.
2. **Best new modeling mechanism:** add node-local categorical partition
   splits to the leaf-wise builder. Current target statistics and raw category
   codes are useful, but they are not a gradient/Hessian-aware category-set
   split.
3. **Best contained split-semantic improvement:** learn whether missing values
   route left or right at each split. DarkoFit currently assigns missing values
   to the highest bin, which always routes right for an ordinary threshold.
4. **Best future private-ensemble diagnostic:** if a new ensemble contract is
   separately authorized, first retain OOB predictions, coverage, member-error
   correlation, and OOB permutation importance. The public ensemble API already
   exists; learned weights and automatic/default promotion are not the next
   funded step.
5. **Best grouped-data hybrid probe:** when callers supply meaningful repeated
   entity or cluster metadata, test GPBoost's central idea that residual
   dependence should be modeled rather than treated as independent noise. The
   smallest DarkoFit-shaped experiment is a strongly shrunk group-residual
   component with explicit seen-group and cold-group behavior—not a Gaussian
   process or LightGBM port.

The highest-value product capability remains monotonic and interaction
constraints. Those should be funded for hard guarantees and domain-prior
control, not advertised as an automatic Elo gain.

Outside TabArena's strict `Tree-based` label, only two current models pass a
defensible tree-adjacent screen. xRFM has an explicit oblique routing tree with
kernel experts. iLTM's tree-enabled checkpoints fit GBDTs and turn every row's
leaf assignments into a learned second-stage representation. The best bounded
new iLTM probe is an honest leaf-address diagnostic or small sparse residual
head—not its meta-trained hypernetwork, dense neural ensemble, or retrieval
stack.

GPBoost is a different case. It has no grade in the already-frozen TabArena
snapshot and is included at the owner's request as an **ungraded structured
residual hybrid**. Its tree component is LightGBM-derived; the distinctive idea
is to learn a nonlinear boosted mean jointly with grouped random effects or a
Gaussian-process residual. That idea is potentially valuable for repeated
entities and spatial/temporal data, but it is not leaderboard evidence and does
not reorder the current mechanism campaign.

Do **not** reopen generic categorical combinations, generic cross features,
automatic linear-leaf selection, automatic ordinal discovery, or the current
OOB-ensemble default policy under a new donor name. The repository already has
direct evidence closing those particular routes.

## Scope and how to read the leaderboard

This dossier freezes the leaderboard view at TabArena Space commit
[`832ff68`](https://huggingface.co/spaces/TabArena/leaderboard/tree/832ff68cf9740a64088f6808dc922f6b2d2c8b6c).
It uses the page's default **all tasks / all datasets / all repeats /
include-imputed**
[CSV snapshot](https://huggingface.co/spaces/TabArena/leaderboard/blob/832ff68cf9740a64088f6808dc922f6b2d2c8b6c/data/imputation_yes/splits_all/tasks_all/datasets_all/website_leaderboard.csv).
The core inclusion rule is mechanical: take every row whose `TypeName` is
**Tree-based**, then collapse default, tuned, and tuned-plus-ensembled rows into
one dossier per underlying family. That produces eight families, not 24
allegedly distinct algorithms. A second screen admits a non-tree row only when
its fitted predictor uses an actual tree partition or consumes tree-derived
partition addresses. xRFM and iLTM pass that screen. High-scoring MLP,
nearest-neighbor, retrieval-transformer, and foundation-model rows do not
become tree models merely because they are local, nonlinear, or ensembled.
GPBoost is a research-only exception to the leaderboard inclusion rule: the
already-pinned CSV and model-registry review found no GPBoost row or adapter, so
this dossier assigns it no Elo, rank, or TabArena quality claim.

TabArena evaluates model-specific pipelines on 51 curated, small-to-medium,
IID tabular datasets. Binary classification uses ROC AUC, multiclass uses
log-loss, and regression uses RMSE; the main display aggregates those
dataset-level results with Elo. See the
[leaderboard methodology](https://huggingface.co/spaces/TabArena/leaderboard/blob/832ff68cf9740a64088f6808dc922f6b2d2c8b6c/website_texts.py)
and the [TabArena paper](https://arxiv.org/abs/2506.16791).

The model search-space audit uses TabArena source commit
[`3b2669d`](https://github.com/autogluon/tabarena/tree/3b2669da335eac08531bcef64c147c99e46cf595).
That matters because the tuned rows are pipelines, not Platonic versions of
the underlying algorithms.

Consequences:

- The rows measure preprocessing, configuration search, bagging, and
  aggregation as well as the core learner.
- TabArena's reproduction example describes 200 random configurations,
  eight-fold bagging, and a post-hoc weighted ensemble. The framework naming
  code exposes a default ensemble size of 40. Those are orchestration choices,
  not built-in properties of LightGBM or CatBoost. See the
  [reproduction example](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/examples/running_tabarena_models/run_tabarena_model.py)
  and
  [framework naming code](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/evaluation/framework_naming.py).
- Every strict-cohort tree row is verified, CPU, and reports 0% imputed tasks
  in this view. Elo can still move when the page's imputation toggle changes
  because the comparison pool changes.
- Overlapping Elo intervals are not evidence that neighboring families are
  decisively ordered. LightGBM and CatBoost are effectively a leading group in
  this snapshot, not a settled first and second place.
- `xRFM` is outside the core eight because TabArena labels it **Other**, even
  though its [paper](https://arxiv.org/abs/2508.10053) combines a balanced
  oblique partition tree with AGOP-adapted kernel Recursive Feature Machine
  leaf experts.
- `iLTM` is labeled **Foundation Model**, but its tree-enabled checkpoints in
  the [paper](https://arxiv.org/abs/2511.15941) fit XGBoost or CatBoost and
  one-hot encode the resulting leaf indices before the neural stages. It
  therefore receives a tree-adjacent dossier, not a ninth core-tree dossier;
  its robust-only checkpoints remain non-tree ablations.
- `GPBoost` has no row in this frozen snapshot. Its
  [paper](https://www.jmlr.org/papers/v23/20-322.html) combines a
  LightGBM-derived boosted predictor with grouped random effects and/or a
  Gaussian process. It is reviewed for mechanism relevance only.
- AutoGluon is also outside the eight because it is a multi-family reference
  pipeline.

## The eight families at a glance

Ranks are one-indexed global Elo positions. Times are the leaderboard's median
seconds per 1,000 rows for the best-ranked variant. Every family's best row is
its tuned-plus-ensembled row, which is evidence to study selection and
diversity—not proof that a fixed ensemble belongs in DarkoFit's defaults.

| Family | Default Elo | Tuned Elo | Tuned + ensemble Elo | Best global rank | Best-row train / predict | First DarkoFit conclusion |
|---|---:|---:|---:|---:|---:|---|
| LightGBM | 1180 | 1366 | **1408** | 14 | 417.05 / 2.639 | Native category-set splits are the largest missing modeling mechanism. |
| CatBoost | 1357 | 1385 | **1397** | 16 | 1346.21 / 0.344 | Most of the core is present; causal ordering and a joint depth/L2 rule remain. |
| XGBoost | 1208 | 1333 | **1355** | 24 | 693.49 / 1.689 | Learned missing routing and constraints are concrete semantic gaps. |
| ChimeraBoost | 1250 | 1292 | **1314** | 31 | 427.39 / 0.404 | Core parity is high; retain diverse tuned trials instead of copying knobs. |
| EBM | 1190 | 1220 | **1253** | 43 | 1711.25 / 0.141 | Its additive terms and interaction diagnostics are the valuable difference. |
| ExtraTrees | 1007 | 1179 | **1204** | 50 | 263.04 / 0.766 | Random cut points are distinct from noisy gain ordering. |
| RandomForest | 1000 | 1136 | **1172** | 56 | 373.24 / 0.771 | OOB exists locally; OOB diagnostics and aggregation do not. |
| PerpetualBooster | 931 | 1050 | **1085** | 65 | 185.31 / 0.600 | Budget UX, missing branches, drift, and refresh are product hypotheses. |

## Live duplicate audit

The repository audit was refreshed on 2026-07-23 at `974b5ee`, including the
automation-first product directive and current mechanism roadmap. The recently
landed private-ensemble provenance and fused-kernel evidence-gate changes
harden group-code persistence, authorization, and engagement accounting; they
do not add any xRFM, iLTM, or GPBoost mechanism recommended below.

The distinction among **implemented**, **partial**, **absent**, and **closed**
is important:

| Mechanism | Live DarkoFit status | Evidence and consequence |
|---|---|---|
| Histogram binning and histogram subtraction | **Implemented** | `darkofit/binning.py` learns bins; `darkofit/tree.py` contains level-wise, leaf-wise, and multiclass parent/sibling subtraction kernels. Do not port this from LightGBM. |
| Leaf-wise and hybrid growth | **Implemented** | `build_leafwise_tree()` and `build_hybrid_tree()` are first-class native builders. Do not relabel them as new XGBoost/LightGBM work. |
| GOSS and MVS | **Implemented** | The booster has uniform, GOSS, weighted-GOSS compatibility, and MVS sampling. Any new proposal needs different estimator semantics. |
| Ordered target statistics | **Implemented** | `OrderedTargetEncoder` uses one or more random permutations and leakage-safe prefix totals. It does not accept a causal time/block order. |
| Automatic depth and L2 | **Partial** | `_resolve_auto_structure_params()` resolves depth from effective-row buckets and L2 from tree mode plus weight concentration, independently. A coupled rows-per-feature rule is absent. |
| Per-feature bin budgets | **Absent** | `Binner` accepts one scalar `max_bins`; realized bin counts vary with cardinality, but users cannot allocate a larger border budget to selected features. |
| Native categorical partition splits | **Absent** | Categorical inputs become target-statistic and optional raw-code numeric columns. Tree payloads store numeric feature/threshold pairs, not category sets. |
| Missing-value semantics | **Partial** | Missing numeric values receive the highest bin and therefore route right. Missingness can be isolated, but left/right direction is not learned and no third child exists. |
| Feature subsampling | **Partial** | `colsample` chooses one feature subset per tree. There is no separate by-level or by-node subset. |
| Monotonic and interaction constraints | **Absent / named, unrated backlog** | No production constraint path exists; the planning entry is unauthorized, not active work. |
| DART, SGLB, random thresholds | **Absent** | `random_strength` adds deterministic noise to split *scores* after every legal threshold is generated. It is neither tree dropout nor random cut-point generation. |
| Sparse input and EFB | **Absent / named, unrated backlog** | Sparse input is rejected and the working binned matrix is dense. EFB is not a local tree-builder switch. |
| OOB bagged boosted models | **Implemented** | Public ensembles bootstrap rows or groups and early-stop each member on its OOB partition. The private v3 prototype also studies without-replacement sampling. |
| OOB prediction matrix and diagnostics | **Absent** | Private-v3 provenance retains sampling/OOB indices; public metadata retains counts, digests, and member summaries. Neither retains predictions, coverage, residual correlation, or permutation importance. |
| Learned ensemble weights / stacking | **Absent** | Regression uses an exact equal mean and classification uses an exact equal soft vote. The private-v3 provenance work preserves that contract. |
| Group labels as fitted predictive structure | **Partial plumbing only** | `groups=` can keep validation and ensemble bootstrap/OOB partitions intact. The fitted mean does not estimate a group random effect, and ordinary prediction accepts no group identity. |
| Group-specific distributional scale | **Implemented, narrower purpose** | `dist_calibration="per_metric_affine"` can fit feature-value-specific scale maps on held-out data. It changes marginal scale, not the conditional mean or covariance between rows. |
| Grouped random effects / Gaussian-process residual | **Absent** | There is no variance-component fit, posterior group correction, coordinate kernel, or latent residual model. |
| Covariance-aware gradients or GLS leaf values | **Absent** | Losses emit per-row gradients and diagonal Hessians, optionally multiplied by sample weights. No fitted cross-row precision operator enters boosting or leaf correction. |
| Cross-row predictive covariance | **Absent** | Distributional heads return per-row parameters, variances, intervals, and draws. They do not represent covariance between two prediction rows or the variance of an aggregate under shared latent effects. |
| EBM/GA2M additive terms | **Absent** | TreeSHAP exists only on supported scalar-oblivious lanes; there is no cyclic one-feature learner, explicit term table, FAST screen, or purified interaction representation. |
| Single model `budget`, continual update, drift monitor | **Absent** | Wall-clock callbacks, tuning budgets, refit recipes, and benchmark drift checks are different concepts; estimators expose no Perpetual-style model budget or continued-fit state. |
| Generic categorical pairs, cross features, automatic linear leaves, automatic ordinal treatment | **Closed as tested** | Existing frozen result files reject those particular automatic mechanisms. A future idea must be causally and operationally different. |

## Priority order after the duplicate audit

| Priority | Hypothesis | Donor families | Kind | Why it is here |
|---|---|---|---|---|
| P0 | Coupled effective-rows-per-feature depth/L2 policy | CatBoost | Small quality policy | Uses existing resolver plumbing and the only positive local attribution signal. |
| P1 | Native category-set split in leaf-wise trees | LightGBM, XGBoost | New model mechanism | Fixes arbitrary numeric thresholds over category IDs and is not the closed combination route. |
| P2a | Learned missing direction | XGBoost, Perpetual, EBM | Split semantics | Contained, testable, and currently fixed-right. |
| P2b | Monotonic and interaction constraints | XGBoost, LightGBM, CatBoost, Perpetual | Product capability | High domain value, but success is a guarantee rather than leaderboard lift. |
| P3 | OOB prediction, coverage, correlation, and permutation diagnostics | RandomForest | Future private instrumentation | High information if a separate private diagnostic contract is authorized; it does not change the existing public API or default aggregation. |
| P4 | Causal target statistics, per-feature bins, or learned aggregation | CatBoost, ChimeraBoost, RandomForest | Medium research | Distinct mechanisms, but each requires new leakage or selection contracts. |
| P5 | Random thresholds, SGLB, DART, EBM lane, sparse/EFB, continual refresh | ExtraTrees and others | Larger exploration | Wider stability, API, or representation blast radius. |

P0 means “fund the next bounded development probe,” not “change a default.”
Any default or automatic policy still needs the preregistered fresh-data,
worst-case-harm, concentration, and cost evidence required by
[`benchmarks/SHIPPING_POLICY.md`](https://github.com/kmedved/darkofit/blob/main/benchmarks/SHIPPING_POLICY.md).

The xRFM and iLTM work below is a separate adjacent-model track. GPBoost is an
ungraded structured-residual track. External comparators and attribution
diagnostics come first; none changes the P0–P5 ordering above or reopens a
closed default/ensemble route. Under the current one-mechanism-at-a-time
roadmap, GPBoost is a queued research hypothesis, not the next authorized
implementation slot.

## Dossiers

### 1. LightGBM — efficient histograms, best-first growth, and category sets

#### How the model differs

LightGBM quantizes continuous values and builds gradient/Hessian histograms.
After the `O(rows)` histogram pass, candidate thresholds cost `O(bins)` rather
than another scan of every row. It derives one sibling histogram from the
parent and the other sibling, and grows the leaf with the largest available
loss reduction. At fixed leaf count this concentrates capacity where it helps,
but the same concentration can overfit small/noisy data, which is why leaf
count, depth, and child-mass constraints matter.

Its categorical algorithm is more important here than the marketing label
“native categoricals.” At a node, LightGBM accumulates statistics by category,
sorts categories by a regularized objective statistic such as accumulated
gradient over Hessian, and scans contiguous partitions in that ordering. This
reduces an exponential set-partition search to roughly `O(k log k)` for `k`
categories. The
[official feature guide](https://lightgbm.readthedocs.io/en/stable/Features.html)
describes histogram subtraction, best-first growth, and category ordering; the
[advanced guide](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html)
documents unseen-category treatment and rare-category controls.

The other signature ideas are GOSS—retain large-gradient rows and reweight a
sample of small-gradient rows—and Exclusive Feature Bundling, which compresses
mostly mutually exclusive sparse columns. The
[NeurIPS paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
is the primary source.

#### What TabArena actually searches

TabArena's
[LightGBM space](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/lightgbm/hpo.py)
uses 200 random configurations over learning rate, row and feature fractions,
leaf count, minimum leaf size, L1/L2, and `extra_trees`. It also searches four
categorical controls: `min_data_per_group`, `cat_l2`, `cat_smooth`, and
`max_cat_to_onehot`. The 228-Elo default-to-tuned gap therefore cannot be
attributed to one split trick; capacity, regularization, sampling, and
categorical behavior all move together.

#### DarkoFit overlap

| LightGBM mechanism | DarkoFit status | Intake decision |
|---|---|---|
| Histograms and sibling subtraction | Implemented, with multiple scalar/multiclass kernels | No work. |
| Best-first leaf-wise growth | Implemented in `tree_mode="lightgbm"` | No work. |
| GOSS | Implemented, including weighted compatibility | No work. |
| MVS | Implemented even though it is not a LightGBM signature | No work. |
| Native category-set split | Absent | Highest-value LightGBM prototype. |
| Node-local category regularization | Absent because node-local category splits are absent | Design with the partition prototype. |
| `extra_trees` random cut points | Absent | Route to the ExtraTrees dossier. |
| EFB/sparse histogram path | Absent | Defer until a named sparse workload exists. |
| Cost-efficient feature penalties | Absent | Useful opt-in product experiment when feature acquisition has a real cost. |

#### Steal A: native category-set splits — P1

This is not target encoding. Keep original categorical codes available to the
leaf-wise builder, build a category histogram for the active node, shrink the
per-category statistic toward the node/global prior, sort valid categories,
and scan prefix partitions. The fitted split must store the chosen category
set, not a numeric threshold over arbitrary codes.

The smallest honest implementation seam is:

- preprocessing retains a stable category-code block and category metadata
  separately from target-statistic numeric features;
- the scalar leaf-wise split scorer gains a categorical candidate path;
- `NonObliviousTree` stores split kind plus category-set payload;
- prediction declares missing and unseen-category routing;
- safe serialization validates category-set offsets and feature identity;
- feature importance maps the split back to the original input feature; and
- TreeSHAP is either implemented for category-set paths or rejected explicitly
  until its path semantics are proven.

Start with scalar RMSE and binary log-loss in leaf-wise mode. Do not begin with
the symmetric or vector-tree lanes: one categorical split shared across many
leaves and classes expands the proof surface before the basic mechanism is
known to help.

Minimum discriminating experiment:

1. Current target statistics only.
2. Current target statistics plus raw codes.
3. Native partition plus target statistics.
4. Native partition without target statistics, as an attribution arm.

Use low-, medium-, and high-cardinality categoricals; rare levels; weights;
missing levels; prediction-only unseen levels; and group/future splits. Report
quality, calibration, split stability, tree size, fit time, and the fraction of
splits using category sets. A random IID win with poor unseen-level behavior is
not a pass.

#### Steal B: category-specific regularization

LightGBM's `min_data_per_group`, `cat_l2`, `cat_smooth`, and one-hot threshold
are not duplicates of DarkoFit's current `cat_smoothing`. The local parameter
smooths preprocessing target statistics; LightGBM's controls regularize the
*node-local category partition*. Implement them only with Steal A:

- pool or suppress categories below a minimum node mass;
- shrink each category's ordering statistic before sorting;
- require minimum Hessian/count on both sides of the proposed set split; and
- use a small-cardinality one-vs-rest path only if it is cheaper or more stable
  than partition search.

The experiment should sweep regularization only after an unregularized semantic
prototype passes exactness tests. Otherwise, a large search can hide a broken
category representation.

#### Steal C: cost-efficient feature penalties — conditional product value

LightGBM's CEGB subtracts penalties for a split, first use of a feature in a
model, or first use for an individual row. DarkoFit has no feature-cost surface.
A narrow version could accept user-declared nonnegative acquisition costs and
subtract a scaled cost from true split gain while retaining both raw and
penalized gain in diagnostics.

This is worthwhile only when a feature has a real latency, licensing, or
availability cost. The minimum test is a two-tier feature set with an explicit
quality-at-inference-cost frontier. It should not become an automatic
regularizer with invented costs.

#### Steal D: EFB and sparse storage — defer, but scope it correctly

EFB matters when wide sparse columns are mutually exclusive. DarkoFit rejects
sparse input and its binned matrix, histogram buffers, feature map,
serialization, and importances all assume a dense layout. A real project needs:

- CSR/CSC or bundled-column input semantics;
- collision detection and a reversible bundle map;
- zero/missing semantics that cannot alias;
- original-feature importance and explanation recovery; and
- a named memory/throughput gate at several sparsity levels.

Do not call dense histogram subtraction “EFB”; the former is already present.

### 2. CatBoost — ordered learning, symmetric trees, and causal preprocessing

#### How the model differs

CatBoost's central contribution is reducing prediction shift from target
leakage. It computes categorical statistics from rows preceding the current
row in a permutation and uses ordered approximations during boosting, rather
than letting a row's target influence the feature or residual used to predict
that same row. Its symmetric trees apply one split at each level to every
active leaf, giving compact models and fast branchless-style prediction. See
[CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
and the
[training-stage documentation](https://catboost.ai/docs/en/concepts/algorithm-main-stages).

CatBoost layers extensive regularization around that core: row-weight
bootstrap, split-score noise, L2 leaf shrinkage, multiple leaf-estimation
steps, categorical combinations/CTRs, and different tree growth policies. It
also exposes a `has_time` mode that preserves input order instead of generating
random permutations, targeted border files for “golden” numeric features, and
CPU Stochastic Gradient Langevin Boosting.

#### What TabArena actually searches

TabArena's
[CatBoost space](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/catboost/hpo.py)
uses Bernoulli sampling and searches learning rate, subsample,
`SymmetricTree` versus `Depthwise`, depth 4–8, `colsample_bylevel`, L2,
1–20 leaf-estimation iterations, one-hot threshold, `model_size_reg`, and CTR
complexity. It pins plain boosting and 254 bins for comparability. CatBoost's
small 42-Elo default-to-tuned improvement, compared with LightGBM's much larger
gap, suggests its defaults are already strong in this benchmark regime.

#### DarkoFit overlap

| CatBoost mechanism | DarkoFit status | Intake decision |
|---|---|---|
| Symmetric/oblivious trees | Implemented and default | No work. |
| Ordered target statistics | Implemented with repeated random permutations | No generic port. |
| Ordered boosting | Implemented on supported task/tree lanes | No generic port. |
| Split-score noise | Implemented with fixed-amplitude deterministic noise | Keep opt-in; the tested sports default is closed. |
| Bayesian bootstrap | Implemented but deprecated | Do not build new work on the compatibility surface. |
| Automatic depth and L2 | Implemented independently | Test only a genuinely joint rule. |
| Causal/time-ordered statistics | Absent | Strong sports/time-series candidate. |
| Per-feature border budget | Absent | Bounded quality experiment. |
| SGLB/posterior sampling | Absent / named, unrated backlog | Distinct diversity experiment, not authorized work. |
| Generic categorical combinations | No live route; historical external donor screen closed as tested | Do not reopen unchanged. |

#### Steal A: couple depth and L2 through rows per modeled feature — P0

The current automatic resolver already does two useful things:

- symmetric-tree depth is chosen from effective-row buckets; and
- L2 starts from a tree-mode base and increases when sample weights are
  concentrated.

What is absent is a joint branch using effective rows per *modeled* feature.
The frozen CatBoost attribution found `l2_leaf_reg=1` to be the only promising
follow-on configuration, while an earlier CatBoost screen retained a
samples-per-feature depth policy as a research candidate. The same screen also
warned that blindly transferring CatBoost's assembled depth rule widened the
historical DarkoFit/CatBoost gap. The correct steal is therefore an auditable
candidate arm, not the literal CatBoost rule.

A bounded implementation would:

1. compute `n_eff / p_model` after preprocessing dimensionality is known;
2. define a very small number of predeclared depth/L2 pairs;
3. leave current defaults untouched outside the experimental policy;
4. record the ratio, selected branch, and rule version in `auto_params_`; and
5. preserve exact explicit-parameter precedence and refit serialization.

The minimum experiment compares current independent auto, fixed depth/L2,
and the joint policy on the eligible spent-development slice. Report
leave-one-task-out concentration and worst-task harm; do not assemble a policy
after seeing a fresh panel.

#### Steal B: causal block-ordered target statistics — P4

`OrderedTargetEncoder.fit_transform()` currently creates random row
permutations. That is leakage-safe in an exchangeable IID sense but not causal
for a future-prediction problem. CatBoost's
[internal-order option](https://catboost.ai/docs/en/concepts/parameter-tuning#internal-dataset-order)
suggests a separate mode that consumes a caller-supplied order.

The stronger DarkoFit version should operate on *time blocks*, not simply row
position:

- sort by a declared timestamp/order key before computing target statistics;
- give all rows in the same timestamp/block the same prior history so peers do
  not leak into one another;
- accumulate weights and category totals only after the block is encoded;
- reject a non-monotone or shape-mismatched order vector; and
- preserve full-training totals only for genuinely future prediction.

The minimum discriminator is a rolling-origin panel with repeated entities,
new categories, and same-day batches. Compare random-permutation ordered TS,
K-fold TS, block-causal TS, and numeric/no-category controls. Include an
invariant proving that each encoded row depends only on earlier blocks.

#### Steal C: per-feature “golden” border budgets — P4

CatBoost's
[border-count guidance](https://catboost.ai/docs/en/concepts/parameter-tuning#border-count)
allows a very small number of important numeric features to receive more
borders instead of raising the global budget. DarkoFit's `Binner` already
stores per-feature border arrays and realized counts, but accepts one global
maximum.

A prototype could accept an internal mapping from original numeric feature to
maximum bins, with the global value as fallback. Selection must be predeclared
or cross-fitted; choosing “golden” features on the evaluation set would be
leakage disguised as binning. Good synthetic controls contain a narrow signal
transition in one feature and smooth/noisy distractors elsewhere.

Measure quality, border stability, preprocessing time, histogram-buffer size,
and model dtype. Because buffer width follows the maximum realized bin count,
one 1,024-bin feature can increase memory for every feature unless storage is
also made ragged; that systems cost belongs in the decision.

#### Steal D: SGLB/posterior-sampling diversity — P5

CatBoost's
[official parameters](https://catboost.ai/docs/en/references/training-parameters/common#langevin)
define a CPU Langevin mode, diffusion temperature, and a posterior-sampling
convenience configuration. This is a different source of diversity from
bootstrap rows or noisy split ranking: it perturbs the boosting dynamics so
independently seeded models can approximate posterior-like samples.

Start only as an ensemble research lane. Require:

- a mathematically documented noise and shrinkage schedule;
- exact seed replay and deterministic serialization;
- identical compute/member budgets versus row-bootstrap ensembles;
- member residual correlation and calibration/disagreement readouts; and
- no uncertainty claim until empirical coverage is validated.

Do not combine SGLB with learned ensemble weights in the first experiment; the
attribution would be unreadable.

#### Explicit non-recommendations

- Do not retry all-pairs categorical combinations. The local cold-player
  result closed that tested mechanism.
- Do not call `random_strength` new. A fixed `0.5` sports recommendation was
  already tested and closed; a decay schedule would need a new causal case.
- Do not port a larger one-hot threshold as an automatic policy from the
  heterogeneous CatBoost screen.
- Do not add more leaf-estimation/backtracking knobs before the existing
  attribution evidence is reconciled.

### 3. XGBoost — sparse-aware split semantics, constraints, and dropout

#### How the model differs

XGBoost formalized a regularized second-order boosting objective and supplied
systems machinery for exact, approximate, histogram, sparse, and distributed
training. Its sparsity-aware algorithm learns a default direction for missing
or absent values at each split. Modern XGBoost adds native categorical
partitioning, monotonic constraints, path-level interaction constraints,
depth-wise or loss-guided growth, and DART tree dropout. The
[KDD paper](https://arxiv.org/abs/1603.02754) remains the main algorithmic
source.

#### What TabArena actually searches

TabArena's
[XGBoost space](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/xgboost/hpo.py)
searches learning rate, depth, child Hessian, row fraction,
`colsample_bylevel`, `colsample_bynode`, L1/L2, depth-wise versus loss-guided
growth, leaf count, and the category one-hot/partition boundary. It explicitly
enables categorical handling. This is a broader per-node randomness and
growth-policy search than DarkoFit's single per-tree `colsample` control.

#### DarkoFit overlap

| XGBoost mechanism | DarkoFit status | Intake decision |
|---|---|---|
| Second-order Newton leaves and child-Hessian controls | Implemented | No generic port. |
| Histograms and row/column sampling | Implemented | No generic port. |
| Loss-guided growth | Implemented as leaf-wise mode | No relabeling. |
| Weighted numeric borders | Implemented from sample weights | Do not claim full XGBoost sparse/sketch parity. |
| Learned missing direction | Absent; missing is fixed-right | Best contained XGBoost mechanism. |
| Native category partitions | Absent | Share the LightGBM implementation. |
| Monotonic and interaction constraints | Absent / named, unrated backlog | High-value capability, but not authorized work. |
| By-level/by-node feature sampling | Absent | Small, distinct diversity surface. |
| DART | Absent | Lower-priority opt-in experiment. |
| Sparse block/external-memory engine | Absent | Large systems project. |

#### Steal A: learn missing direction per split — P2a

DarkoFit gives non-finite values the highest bin. Numeric routing is
`bin <= threshold` left, otherwise right, so missing values always go right.
The tree can isolate missingness at the last finite threshold, but it cannot
decide that missing values belong with the left side of an otherwise useful
split.

For each candidate threshold, compute finite left/right gradient, Hessian, and
count totals once, then score both missing assignments. Record a `default_left`
bit with the winning split. The first implementation should be scalar
leaf-wise trees, where the bit belongs naturally to each node.

Required seams:

- split scorers must avoid double-counting the dedicated missing bin;
- `NonObliviousTree` needs one missing-direction value per internal node;
- predict, staged predict, SHAP, feature importance, and serialization must
  share the same routing rule;
- old archives need an explicit fixed-right default; and
- missing-direction gain should remain inspectable separately from raw gain.

Minimum experiment: MCAR, MAR, informative-missingness, and shifted-missingness
synthetics plus real tasks with meaningful missing rates. Compare fixed-right,
learned binary direction, and simple training-only imputation. A win only on
informative missingness with a reversal under shifted collection practice is a
warning, not a default case.

#### Steal B: monotonic constraints — P2b

The
[XGBoost monotonic guide](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html)
shows the product contract: a user declares increasing, decreasing, or
unconstrained response by feature. In a leaf-wise scalar prototype, propagate
feasible lower/upper output bounds down the tree and either clip/recompute leaf
weights or reject splits that cannot satisfy the direction.

This is a hard semantic feature, so tests outrank average score:

- exhaustive prediction checks over each constrained feature's binned domain;
- adversarial correlated proxies that can otherwise reintroduce a violation;
- weights, missing values, and unseen categories;
- save/load and staged-prediction invariants; and
- an explicit policy for target-stat-expanded categorical features.

Constraints must be declared in original feature space. Do not silently apply
a numeric direction to an encoded categorical statistic. Symmetric-tree
support should be a later proof because one shared split constrains multiple
leaves simultaneously.

#### Steal C: interaction constraints — P2b

XGBoost defines an interaction by features appearing together on a root-to-leaf
path. Its
[interaction-constraint guide](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html)
lets users declare which feature groups may coexist on a path.

For leaf-wise trees, carry the allowable feature set with each active node and
intersect it after each split. Tests should inspect serialized paths directly,
not infer compliance from predictions. A planted allowed/forbidden-interaction
dataset plus a group/future split is the minimum useful quality case.

Again, symmetric trees are harder: the same level feature extends every active
path. A blanket union can violate some paths and a blanket intersection can be
too restrictive, so do not claim support without a defined global rule.

#### Steal D: by-level or by-node feature subsampling

DarkoFit samples one feature set per tree. XGBoost/TabArena separately search
feature subsets at a level and at a node. This can reduce member correlation
more aggressively and prevent one strong feature from monopolizing every
split, but it can also omit a rare essential feature exactly where needed.

A small experiment should add only one new granularity at a time:

- per-level masks for symmetric trees; or
- per-node masks for leaf-wise trees.

Use a counter-based seed keyed by tree/level/node so parallel execution cannot
change the sample. Compare equal expected feature evaluations, not merely equal
tree counts. Report quality, member correlation, feature-use stability, and
fit time.

#### Steal E: DART only for a declared overfit regime — P5

[DART](https://xgboost.readthedocs.io/en/stable/tutorials/dart.html) drops a
random subset of prior trees when fitting a new one and renormalizes dropped
and new contributions. XGBoost's own documentation warns that prediction
buffer reuse is lost, training slows, and early stopping can become unstable.

DarkoFit would need new training-state, staged-prediction, normalization,
importance, SHAP, and archive semantics. Test it only on a preregistered
high-overfit slice under a fixed wall-time budget. Reject the route if seed
variance or early-stop instability dominates its quality gain.

#### Defer or reject

- Native category partitioning is one shared feature with LightGBM, not two
  implementations.
- A Hessian-weighted sketch is interesting only for nonconstant-Hessian losses
  after current weighted quantile bins are shown inadequate.
- Ranking, custom objectives, GPU, and external memory need a product owner and
  workload; leaderboard admiration is not enough.

### 4. ChimeraBoost — the closest comparator and the least fertile donor

#### How the model differs

ChimeraBoost is the closest architectural relative: a Python/Numba GBDT
combining CatBoost-like ordered categoricals and oblivious trees,
XGBoost-style Newton leaves and regularization, and LightGBM-like histograms.
It also exposes bagging with member-owned OOB early stopping, local linear
leaves, MVS, exact oblivious TreeSHAP, and optional cross features. See the
[fixed source snapshot](https://github.com/bbstats/chimeraboost/tree/6a76586dfdff90275e7e816f25e35c927b8527fb).

The donor value is therefore mostly orchestration and product discipline. A
wholesale port would duplicate DarkoFit.

#### What TabArena actually searches

The
[ChimeraBoost space](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/chimeraboost/hpo.py)
searches learning rate, depth, L2, child Hessian, row/column fractions,
leaf-estimation iterations, 128/254 bins, linear leaves plus ridge strength,
categorical smoothing and permutation count, and ordered boosting. It
explicitly excludes raw categorical combinations because bypassing the
automatic cardinality guard can create a resource explosion.

#### DarkoFit overlap

| ChimeraBoost mechanism | DarkoFit status | Intake decision |
|---|---|---|
| Oblivious trees, ordered TS/boosting, MVS | Implemented | No work. |
| OOB member early stopping | Implemented for row/group ensembles | No work. |
| Local linear leaves | Implemented; automatic promotion rejected | Retain explicit only. |
| Cross features / categorical combinations | Research paths exist; tested automatic routes closed | No unchanged retry. |
| Exact TreeSHAP and safe persistence | Implemented on supported lanes | No work. |
| Optuna stepwise/joint tuning | Implemented | Do not propose “add HPO.” |
| Deployable retention of several diverse tuned configurations | Absent; study history retains trial parameters and scores | Possible portfolio extension under a new contract. |
| Learned portfolio weights | Absent | Shares the RandomForest/AutoGluon aggregation prerequisite. |
| Classifier temperature scaling | Absent, but closed as tested | Do not reopen the terminal local result unchanged. |

#### Steal A: turn existing tuner output into a diverse portfolio — P4

DarkoFit's tuner retains trial parameters and scores in the study/CV results,
then chooses and refits one winner. `tree_mode="auto"` also selects one lane.
Public ensembles repeat one configuration and average members. None of those
builds a deployable portfolio from several materially different trials.

An opt-in accuracy portfolio could reuse existing CV trials:

1. collect fold predictions from eligible top trials across CatBoost,
   LightGBM, and hybrid lanes;
2. remove dominated and near-duplicate trials by validation loss, latency, and
   residual correlation;
3. compare the best single trial, equal average, greedy forward selection, and
   regularized nonnegative weights;
4. refit only the selected bases with each trial's existing refit semantics;
5. persist every base configuration, fold provenance, weight, and cost; and
6. expose the portfolio as a separate artifact, never as a single-booster
   quality claim.

The aggregation fit must be nested or outer-OOF. Reusing the same fold scores
both to search hundreds of trials and to fit unconstrained weights will
overfit. Start with a small hard member cap and simplex/ridge regularization.

#### Closed locally: classification temperature scaling

ChimeraBoost includes temperature calibration for probabilities, and DarkoFit
does not expose a classifier-logit temperature surface. That absence is not a
new opportunity: `benchmarks/basketball_temperature_scaling_result.md`
terminally stopped the evaluated route before product implementation and did
not authorize a broader campaign. Do not repackage it as an untried Chimera
steal. A future proposal would need a materially different task regime and a
new prospective contract, including disjoint calibration/selection data.

#### Steal C: hard search-space admissibility, not more knobs

The useful lesson in ChimeraBoost's TabArena configuration is that an HPO
space is also a safety contract: it excludes a known high-cardinality
combination path even though the model supports it. DarkoFit should preserve
the same principle when extending its tuner:

- new mechanisms enter only after semantic tests;
- resource-risk options need dataset-dependent guards before search;
- closed mechanisms are not silently reintroduced by a broad categorical
  distribution; and
- trial metadata records which guard admitted or rejected a parameter.

This is a discipline to preserve, not a missing module to implement.

#### Do not steal again

MVS, OOB early stopping, float32 histogram experiments, local linear leaves,
automatic cross features, and exact supported-lane TreeSHAP already exist or
have explicit local decisions. The recently landed private-ensemble work is
provenance hardening, not an invitation to reopen its default-policy gate.

### 5. EBM — additive structure, explicit interactions, and stable shapes

#### How the model differs

An Explainable Boosting Machine is a tree-based generalized additive model,
not an unrestricted GBDT. It learns a function for one feature at a time,
cycles across features at a low learning rate, and optionally adds a bounded
set of explicit interaction terms:

`link(E[y]) = intercept + sum(main_effect_j) + sum(interaction_jk)`.

Each term becomes a binned lookup table. Prediction is therefore table lookup
plus addition, which explains EBM's very low TabArena prediction time despite
its expensive training. The
[official overview](https://interpret.ml/docs/ebm.html) describes cyclic
boosting, bagging, and automatic interaction detection; the
[GA2M paper](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf) is the
primary interaction source.

Modern InterpretML mixes greedy and cyclic steps, uses heavily regularized
initial smoothing rounds, separates main-effect and interaction bin
resolutions, and outer-bags models to stabilize graphs and estimate error
bounds.

#### What TabArena actually searches

TabArena's
[EBM space](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/ebm/hpo.py)
searches 2–3 leaves, smoothing rounds, learning rate, interaction count,
interaction smoothing, minimum Hessian/leaf count, gain scale, categorical
regularization, and four missing-value modes. The wide interaction multiplier
range is a reminder that EBM capacity is controlled mainly by how many
explicit terms are admitted, not by deep trees.

#### DarkoFit overlap

| EBM mechanism | DarkoFit status | Intake decision |
|---|---|---|
| Histogrammed shallow trees | Loose primitive overlap | Not an EBM implementation. |
| Cyclic one-feature additive terms | Absent | Possible companion lane, not a tree-mode rename. |
| Explicit pair term tables | Absent | Potential narrow interaction model. |
| FAST interaction ranking | Absent | Use externally before implementing. |
| Purified/identifiable interactions | Absent | Valuable only with explicit term models. |
| Outer-bag term uncertainty | Absent | Useful interpretability diagnostic. |
| Exact TreeSHAP | Implemented on supported scalar-oblivious lanes | Different guarantee; not global additivity. |

#### Steal A: external EBM as a glass-box comparator now

The cheapest high-information action is not a core dependency. Add an optional
benchmark adapter that fits EBM on the same training-only feature contract and
records:

- main-effect curves and their outer-bag standard deviations;
- selected pair interactions and strengths;
- test quality and calibration;
- prediction latency and model size; and
- stability of term shapes across group/time folds.

Use this to answer whether DarkoFit's gain comes from stable nonlinear main
effects or fragile high-order interactions. It is a diagnostic baseline, not a
claim that EBM should replace DarkoFit.

#### Steal B: FAST-style residual interaction screening

InterpretML's
[interaction utility](https://interpret.ml/docs/python/api/measure_interactions.html)
ranks pairs with the FAST algorithm and can start from an existing model score.
That makes it suitable for screening interactions in residual space.

This proposal is **not** “generate generic crosses and feed them to the GBDT,”
which is locally closed. The non-duplicative experiment is:

1. generate base predictions strictly out of fold;
2. rank pairs inside each training fold only;
3. fit a bounded two-dimensional additive lookup term for a small stable set;
4. add those terms to the base score without letting an unrestricted tree
   create higher-order crosses; and
5. require pair-selection stability across folds.

Use planted-interaction and correlated-but-additive synthetics. If correlated
main effects are repeatedly misranked as interactions, stop.

#### Steal C: a cyclic additive residual companion

A small companion model could reuse DarkoFit's bins and loss gradients but
restrict each boosting step to one original feature. The result would be an
additive correction with inspectable term tables, not another general tree.
Start with regression main effects only, shallow 2–3 leaf updates, and no
interactions.

Compare on two opposing controls:

- nonlinear additive truth, where the companion should help or match; and
- interaction-dominated truth, where it should correctly lose to the base
  GBDT rather than invent misleading main effects.

If the external EBM baseline supplies the same diagnostic value, there is no
reason to maintain a native implementation.

#### Steal D: purified interactions and separate resolutions

InterpretML's
[purification operation](https://interpret.ml/docs/python/api/purify.html)
moves row/column marginal mass out of a pair tensor so the remaining weighted
row and column sums are zero. That makes “main effect” and “interaction”
identifiable and prevents a pair term from hiding a duplicated main effect.

If explicit pair terms are ever implemented, purification plus separate bin
budgets is worth stealing:

- fine bins for one-dimensional main effects;
- coarser bins for two-dimensional interactions; and
- exact prediction conservation before/after purification.

This is explanation stability and memory control first, not a guaranteed
quality gain.

#### Do not confuse with existing features

DarkoFit's exact TreeSHAP is local attribution for its supported
scalar-oblivious lanes; unrestricted and other unsupported tree modes are
rejected rather than silently approximated. EBM term plots are the model
itself and are globally additive. Likewise, an EBM pair table is not the same
as a raw cross feature that downstream trees can combine arbitrarily.

### 6. ExtraTrees — random cut points as a controlled diversity source

#### How the model differs

ExtraTrees strongly randomizes both feature choice and cut-point choice at
each node. A tree evaluates a small random candidate set instead of finding the
globally best cut over every feature/threshold. The resulting individual trees
are weaker and more biased, but cheap and less correlated; averaging can
recover quality. The
[original paper](https://link.springer.com/article/10.1007/s10994-006-6226-1)
explicitly frames the method as a tunable bias/variance and computational
trade-off.

#### What TabArena actually searches

TabArena's
[ExtraTrees space](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/extra_trees/hpo.py)
uses 50 trees, fixes `bootstrap=False`, and searches feature fraction,
minimum split size, and impurity-decrease regularization. The no-bootstrap
choice separates cut-point/feature randomness from row-resampling randomness;
it also means ordinary OOB validation is unavailable.

#### DarkoFit overlap

| ExtraTrees mechanism | DarkoFit status | Intake decision |
|---|---|---|
| Feature subset per boosted tree | Implemented | Partial diversity overlap. |
| Split-score noise | Implemented | Not random cut points. |
| Random threshold candidate generation | Absent | Main ExtraTrees experiment. |
| Feature subset per node/level | Absent | Share XGBoost granularity work. |
| Full-sample, no-bootstrap randomized members | Absent as a public ensemble lane | Requires external/nested validation. |

#### Steal A: random threshold candidates in an ensemble-only lane — P5

For each eligible feature at a node, sample `K` legal nonempty histogram
boundaries and choose the best true gain among only those candidates. `K=all`
must recover the greedy reference. This differs cleanly from
`random_strength`, which scores every threshold and then perturbs the ordering.

Implementation details that keep the experiment honest:

- sample from legal thresholds after child-mass/count checks, or define and
  test how invalid samples are replaced;
- key the RNG by tree, node/level, feature, and seed;
- retain true unnoised gain and sampled-candidate count in diagnostics;
- use only inside an explicit ensemble/diversity mode initially; and
- never silently fall back to greedy search when all sampled candidates fail.

#### Steal B: separate threshold and feature randomness

Run a factorial experiment:

1. greedy thresholds, full features;
2. greedy thresholds, node/level feature subsets;
3. random thresholds, full features; and
4. random thresholds plus feature subsets.

This distinguishes which randomization lowers correlation and which merely
weakens members. Match either split-evaluation count or wall time, state which,
and report member error, pairwise residual correlation, ensemble error,
calibration, and worst-group behavior.

Use both smooth and narrow/discontinuous synthetic signals. Random candidates
should be expected to struggle when a very specific cut point is essential.

#### Steal C: full-sample randomized members as a comparator

Because TabArena fixes `bootstrap=False`, test whether randomness alone gives
more useful diversity than spending rows on bootstrap/OOB partitions. This
arm cannot use member OOB early stopping; give all arms the same external or
nested validation contract. Otherwise the comparison confounds randomization
with better validation data.

#### Stop rule

ExtraTrees is far below the leading GBDTs in the strict leaderboard cohort.
Promote random thresholds only if they improve an equal-compute blend through
lower residual correlation. A weaker standalone model with no marginal blend
value is not a win.

### 7. RandomForest — strength, correlation, and honest OOB information

#### How the model differs

Random forests average independently randomized trees trained on bootstrapped
rows and random feature subsets. Breiman's key result is qualitative and still
useful: ensemble error depends on both individual-tree strength and correlation
among trees. The same bootstrap supplies out-of-bag predictions, internal
error estimates, and variable-importance estimates. See the
[original paper](https://doi.org/10.1023/A:1010933404324).

#### What TabArena actually searches

TabArena's
[RandomForest space](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/random_forest/hpo.py)
uses 50 trees and searches feature fraction, row fraction, bootstrap on/off,
minimum split size, and impurity decrease. It disables child OOF use in the
AutoGluon wrapper, so its tuned/ensembled result should not be mistaken for a
demonstration of DarkoFit-style member-owned OOB early stopping.

#### DarkoFit overlap

| RandomForest mechanism | DarkoFit status | Intake decision |
|---|---|---|
| Row/group bootstrap | Implemented for boosted-model ensembles | No generic port. |
| Member-owned OOB early stopping | Implemented | Already deeper than a basic forest OOB score. |
| Equal mean / soft vote | Implemented | Current supported aggregation. |
| OOB prediction retention and coverage | Absent | First new work. |
| OOB permutation importance | Absent | High-information diagnostic. |
| Member strength/correlation curves | Absent | Add before adaptive member counts. |
| Learned OOB/OOF weights | Absent | Test only after diagnostic substrate. |

The private-ensemble-v3 provenance work persists normalized group codes and
validates sampling/OOB provenance on load. It does **not** retain predictions,
learn weights, or change equal aggregation. The ideas below are intake for a
future separately authorized private contract, not permission to change the
existing public API, learned-weight policy, or default aggregation.

#### Steal A: OOB prediction and coverage artifact — P3

For each member, evaluate only rows absent from that member's training sample
and retain:

- row index and group identity;
- raw margin and transformed prediction;
- target and optional sample weight;
- member seed/configuration and best iteration; and
- an explicit OOB-eligibility bit.

Aggregate a per-row equal OOB prediction only from eligible members and expose
coverage counts. Do not fill missing member/row cells with in-sample
predictions. For group bootstrap, assert group disjointness before accepting a
record.

First use this artifact for diagnostics, not a new predictor. Compare OOB loss
with an external group/future holdout and measure whether the gap changes with
ensemble size or sampling unit.

#### Steal B: OOB member strength and correlation diagnostics

On overlapping OOB rows, compute pairwise residual correlation and each
member's OOB loss. Also report the ensemble's marginal OOB improvement as
members are added in a fixed seed order. This makes “more members” falsifiable:
a new member that is weak and highly correlated is unlikely to justify its
inference cost.

Only after repeated curves are stable should DarkoFit consider an automatic
member-count stop. A fixed five-member policy has already been tested/closed in
one sports route; diagnostics are new, but a default-count retry is not.

#### Steal C: OOB permutation importance

For each member and original feature, permute that feature only within the
member's OOB rows, recompute loss, and aggregate the loss increase with
coverage/uncertainty. Grouped or temporal data may require within-block
permutations to avoid creating impossible examples.

This complements gain importance and TreeSHAP:

- it evaluates held-out predictive dependence rather than training split use;
- it can reveal high-cardinality gain bias; and
- it naturally reports when importance is unstable across members/groups.

The minimum contract includes deterministic permutations, feature-name
mapping, weights, categoricals, and a null-noise feature expected to have no
importance.

#### Steal D: regularized cross-fitted aggregation weights — P4

Once coverage is understood, compare:

1. best member;
2. equal member mean/soft vote;
3. sparse OOB-based nonnegative weights; and
4. proper group/time-fold OOF weights.

Weights should lie on a simplex or use strong ridge shrinkage toward equal
weights. The proper OOF arm is the cleaner reference because every base has a
prediction for every validation row. A sparse OOB matrix can otherwise reward
members merely because they cover easier rows.

Serialize the aggregation rule and reproduce SHAP/expected-value aggregation
only after prediction semantics are stable. This is a new layer, not a retune
of `n_ensembles`.

### 8. PerpetualBooster — budget UX, missing branches, and lifecycle features

#### What is established and what is only claimed

PerpetualBooster is a Rust histogram GBDT whose public product abstraction is a
single `budget`. Its current architecture documentation says learning rate is
set deterministically as `10^-budget` and stopping follows an internal
generalization criterion. The project also advertises native categoricals,
learnable missing splits, monotonic/interaction constraints, continual
learning, drift monitoring, calibration, and columnar zero-copy input. See the
[documentation](https://perpetual-ml.github.io/perpetual/) and
[architecture](https://perpetual-ml.github.io/perpetual/architecture.html).

Its generalization-algorithm paper is still described as work in progress.
Treat the model's public behavior and API as source material; do not present an
undocumented algorithm as independently established science.

#### What TabArena actually searches

TabArena's entire
[PerpetualBooster space](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/perpetual_booster/hpo.py)
is `budget` in `{0.1, 0.2, 1.0, 1.5, 2.0}`, with manual/default `0.5`. That is
the purest example in the eight of tuning a product abstraction rather than a
large conventional parameter space.

#### DarkoFit overlap

| Perpetual mechanism | DarkoFit status | Intake decision |
|---|---|---|
| Histogram GBDT and automatic parameters | Implemented generally | Not Perpetual's one-budget contract. |
| Time limit and tuning budgets | Implemented | Different from model complexity budget. |
| Learned missing-left/right | Absent | Implement the XGBoost binary version first. |
| Separate third missing branch | Absent | Test only if binary routing is insufficient. |
| Monotonic/interaction constraints | Absent | Share the XGBoost design. |
| Continued fit / leaf refresh | Absent | Rolling-origin product experiment. |
| Model-native data/routing drift monitor | Absent | Low-risk diagnostics project. |
| Zero-copy Polars/Arrow path | Absent | Profile before funding. |

#### Steal A: a transparent budget-to-resource contract — P5

Do not copy the claim “no tuning required.” A useful DarkoFit budget would be a
versioned convenience facade that resolves to explicit existing controls:

- maximum boosting rounds and early-stopping patience;
- tree capacity or auto-selection audition spend;
- tuning trials and/or portfolio member cap;
- wall-clock safety margin; and
- a declared memory/inference-cost ceiling where enforceable.

The complete mapping should appear in `auto_params_` and be reproducible with
explicit constructor parameters or `get_refit_params()`. Manual parameters
must have clear precedence over the facade.

Test three budget levels across fresh datasets. Resource use should be
monotone and deterministic even if predictive quality is not. A budget that
occasionally spends less at a larger setting or silently changes validation
semantics is not a trustworthy UX.

#### Steal B: a third missing branch only after learned binary routing

Perpetual's
[API](https://perpetual-ml.github.io/perpetual/api.html) distinguishes a
learned missing split and an optional separate missing branch. A ternary node
can model “missing” independently rather than forcing it to share a finite
side, but it expands tree storage, traversal, gain legality, SHAP paths, and
overfit risk for rare missingness.

Sequence the work:

1. fixed-right reference;
2. XGBoost-style learned binary direction; then
3. ternary missing branch only on datasets where binary routing leaves a
   measured residual gap.

Require a minimum missing count/Hessian before the third branch is legal and
test collection-shift scenarios. This should not be bundled into the first
missing-direction patch.

#### Steal C: model-native drift diagnostics

Current repository “drift” checks protect benchmark artifacts and source
identity; they are not fitted-model data-drift monitoring. A read-only model
diagnostic could compare new data with fit-time references using:

- per-feature bin occupancy and missing rate;
- unknown-category rate;
- tree/leaf routing occupancy;
- raw-margin and probability distribution; and
- group-specific versions of the same summaries.

Persist compact reference counts, not training rows. Validate on no-shift,
covariate-shift, missingness-shift, and concept-shift simulations. Covariate
drift does not imply quality loss, so the API should report distances and
threshold provenance rather than declare retraining automatically.

#### Steal D: fixed-structure refresh / continual update

Perpetual exposes continued training; XGBoost also supports model refresh-like
workflows. The safest first DarkoFit experiment is narrower than
`partial_fit()`:

- freeze all split structures and preprocessing;
- update only leaf values, or append a bounded number of new trees, on a new
  time block;
- retain the prior model as a recoverable artifact; and
- compare with both a frozen model and a full cumulative refit.

Use rolling origins with gradual drift and structural concept drift. Refresh
should be faster under gradual drift and should correctly lose when old splits
are obsolete. Define sample-weight aging, category evolution, early stopping,
and calibration refresh before exposing a public method.

#### Steal E: zero-copy columnar ingestion only after profiling

Perpetual's Rust/PyO3 layer can read columnar buffers directly. That is a
systems idea, not a quality mechanism. DarkoFit should fund Arrow/Polars or
zero-copy work only if profiling shows input coercion is a material fraction of
end-to-end fit/predict time on a named workload. Optional dependency weight,
layout constraints, and categorical metadata transfer all count against the
benefit.

## Recommended experiment sequence

The sequence below minimizes representation churn and avoids stacking several
unproven mechanisms into one result.

1. **Run the P0 coupled depth/L2 development probe.** It touches existing
   resolver plumbing, has local attribution support, and does not require a new
   tree payload.
2. **Prototype P2a learned missing direction.** It is a contained new split
   semantic and forces prediction/serialization/SHAP contracts to become
   explicit before category sets add another split kind.
3. **Prototype P1 categorical partitions in scalar leaf-wise mode.** Build
   semantic and unseen-category tests before spending a broad quality panel.
4. **Treat P2b constraints as a capability program.** Acceptance is exact
   monotonic/path compliance and domain utility first, aggregate score second.
5. **If a new private-ensemble contract is authorized, build P3 diagnostics.**
   Retain OOB predictions, coverage, member loss/correlation, and permutation
   importance without changing fitted predictions. The current B3 boundary and
   default-promotion decision stay unchanged.
6. **Choose one authorized P4 preprocessing or aggregation experiment.**
   Causal target statistics, per-feature bins, or learned OOF weights should
   not share the first protocol; weighting also depends on the P3 contract.
7. **Fund only one P5 diversity mechanism at a time.** Random thresholds,
   SGLB, or DART should compete at identical compute with explicit seed and
   correlation readouts.
8. **Keep EBM, EFB/sparse, budget/continual, and zero-copy work product-led.**
   Each needs a named user/workload rather than a generic leaderboard mandate.

Use the existing focused tests and benchmark harnesses first: LightGBM
comparison, categorical behavior, ensemble API/provenance, staged prediction,
safe persistence, TreeSHAP, tuning/refit semantics, and the documented
[development validation partitions](development.md). Any default-facing result
must remain separate from spent artifacts and sealed confirmation data.

## Out-of-scope but important: the reference-pipeline lesson

AutoGluon is not a ninth tree-model dossier. TabArena labels it a reference
pipeline spanning multiple model types and evaluates it outside the
model-specific protocol. It is still the clearest ceiling signal.

Its useful mechanisms are cross-validation bagging, a heterogeneous model
portfolio, multi-layer stacking, and weighted-ensemble selection. The
appropriate translation is not “make DarkoFit AutoGluon.” It is:

1. retain leakage-safe OOF predictions from DarkoFit's own tuned modes;
2. compare best single model, equal average, and regularized nonnegative
   weights under one deployment-cost budget;
3. use nested group/time folds when trial selection and weight fitting share a
   dataset;
4. add external CatBoost/LightGBM/EBM bases only after the DarkoFit-only
   portfolio proves marginal value; and
5. separate portfolio quality, calibration, latency, and artifact size claims.

See AutoGluon's
[how-it-works guide](https://auto.gluon.ai/stable/tutorials/tabular/how-it-works.html)
and [paper](https://arxiv.org/abs/2003.06505).

## Extended dossier: xRFM

TabArena labels xRFM **Other**, so it is excluded from the strict eight. That
taxonomy should not be read as a quality dismissal. Its best row sits between
XGBoost and ChimeraBoost in the pinned all-task leaderboard.

The acronym is easy to misread: RFM means **Recursive Feature Machine**, not
random-feature model. xRFM is not a random forest, a random Fourier-feature
model, or a boosted tree with a slightly richer leaf. It is better understood
as a balanced oblique routing tree whose terminal regions contain local,
feature-learning kernel-ridge experts. See the current
[paper](https://arxiv.org/html/2508.10053v3), the authors'
[v0.4.5 source](https://github.com/dmbeaglehole/xRFM/tree/v0.4.5), and their
[algorithm notes](https://github.com/dmbeaglehole/xRFM/blob/v0.4.5/ALGORITHM.md).

### What the Recursive Feature Machine learns

For a fitted predictor `f`, the Average Gradient Outer Product is

`M = (1 / n) * sum_i J_f(x_i)^T J_f(x_i)`.

For a scalar target, `J_f(x_i)` is the input gradient. For multiple outputs,
the construction aggregates across output dimensions. `M` is an uncentered
gradient second moment, interpretable as a supervised sensitivity covariance:

- large diagonal entries identify coordinates along which the predictor
  changes strongly;
- leading eigenvectors identify joint directions of predictive change;
- a diagonal `M` acts like learned coordinate reweighting; and
- a full `M` learns a rotation/Mahalanobis geometry that can combine features.

A leaf RFM starts from an identity metric, fits kernel ridge regression,
computes the fitted predictor's AGOP, updates the kernel geometry with that
matrix, and repeats. Validation chooses the retained iteration. The paper's
kernel family can be written as

`K_pq(x, z) = exp(-||x - z||_p^q / L^q)`, with `0 < q <= p <= 2`.

Consequently, each leaf learns a nonlinear predictor, its own feature metric,
and optionally its own bandwidth. It does not learn a finite vector of random
features, and it is materially richer than a ridge-linear leaf.

### How the routing tree is built

The tree exists to provide local feature learning and to cap the size of each
kernel problem:

1. At a node larger than the maximum leaf size, sample local rows.
2. Fit a lightweight split RFM and compute its AGOP.
3. Take the AGOP's leading eigenvector `v`.
4. Project every node row onto `v^T x`.
5. Split at the median projection and recurse.
6. After the routing tree is fixed, train a full leaf RFM in every terminal
   region.

The direction is supervised and oblique; the threshold is selected for
balance, not immediate loss reduction. This creates a nearly balanced tree,
unlike CART or DarkoFit's gain-optimized axis-aligned builders. With a fixed
maximum leaf size, routing depth grows logarithmically with the total row
count.

The default maximum leaf size is 60,000 rows, adjusted by the released package
for available device memory. If a training fold is smaller than that, the
default model may never split: the experiment is then a global RFM experiment,
not evidence for the tree. Any DarkoFit comparison must report the realized
split and leaf counts.

Hard inference follows the oblique path and evaluates the selected leaf's
kernel predictor against that leaf's stored centers. The paper describes
inference as logarithmic in total sample count because leaf size is treated as
fixed. More precisely, routing is logarithmic; the local kernel evaluation can
still cost `O(leaf_size * features)`, plus a full-matrix transform when `M` is
dense.

### Soft routing is not classifier temperature scaling

xRFM can replace a hard split with an IQR-scaled logistic gate. A leaf's weight
is the product of its branch probabilities, and the prediction is a normalized
mixture of several leaf experts. This is an optional Appendix-B extension to
the paper's hard-routed core model. It tunes one routing temperature on
validation data and retains at most eight leaves per row. The released v0.4.5
implementation uses 20 positive candidates plus hard routing, retains enough
leaves to cover 99% of routing mass, and caps the default at 12. See the
[paper discussion](https://arxiv.org/html/2508.10053v3#Sx1) and
[released implementation](https://github.com/dmbeaglehole/xRFM/blob/v0.4.5/xrfm/xrfm.py#L1028-L1120).

This is not the classifier-logit temperature scaling that DarkoFit tested and
closed. Routing temperature changes which local models contribute to the
prediction. It can smooth an arbitrary region boundary, but it also evaluates
multiple expensive kernel experts and changes the model representation.

### What the pinned leaderboard actually says

All three rows are verified, non-imputed, and GPU-evaluated in this snapshot.
The CSV's `#` field is a zero-based leaderboard row index, not the
dataset-average `Rank` statistic.

| xRFM row | Elo | 95% interval | Leaderboard row `#` | Train / predict seconds per 1,000 rows |
|---|---:|---:|---:|---:|
| Default | 1038 | +55 / -66 | 67 | 3.23 / 0.919 |
| Tuned | 1286 | +45 / -38 | 35 | 846.89 / 0.130 |
| Tuned + ensemble | **1331** | +45 / -40 | **28** | 846.89 / 2.549 |

The 248-Elo default-to-tuned jump and another 45 Elo from post-hoc ensembling
make xRFM one of the clearest examples of orchestration dominating a default
row. The tuned-plus-ensemble row is below XGBoost at 1355 and above
ChimeraBoost at 1314, but its interval overlaps nearby methods.

TabArena's pinned
[xRFM search space](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/xrfm/hpo.py)
uses 200 random configurations and searches:

- kernel bandwidth from 0.5 to 200 on a log scale;
- diagonal versus full AGOP geometry;
- kernel exponent from 0.7 to 1.4 and an interpolation controlling `p`;
- an 80/20 mixture of accelerated `lpq_kermac` and `l2` kernels; and
- ridge regularization from `1e-6` to `10`.

It fixes constant bandwidth mode, prevalence classification coding, direct
solves, unstandardized one-hot categorical columns, and an early-stop
multiplier of 1.1. The wrapper standardizes numeric features, densely one-hot
encodes categoricals, mean-imputes numerics with missing indicators, and
standardizes regression targets. This is a full pipeline result, not an
isolated AGOP result.

The paper's strongest headline is regression: it reports the best aggregate
results across 100 TALENT regression datasets and third-place competitive
classification across 200 datasets. Those are author-reported benchmark
results, not a local reproduction. Its TabArena experiment also uses
TabArena-Lite, with far fewer outer folds than the full leaderboard. The fair
claim is “promising regression model and competitive classifier,” not
“universally beats GBDTs.”

### DarkoFit overlap

| xRFM mechanism | Live DarkoFit status | Intake decision |
|---|---|---|
| Nonlinear learned-metric kernel leaf expert | Absent | Genuine external comparator; not a local-linear retry. |
| Balanced median split on a supervised oblique direction | Absent | New top-level representation, not another `tree_mode`. |
| Full/diagonal AGOP feature geometry | Absent | Potential heterogeneity diagnostic before any model port. |
| Hard routing to one local expert | Loose conceptual overlap only | DarkoFit sums many boosted trees rather than selecting one regional model. |
| IQR-scaled soft routing over local experts | Absent | Distinct from closed probability calibration; high implementation blast radius. |
| Local ridge-linear leaves | Implemented in a narrower form | Automatic promotion remains closed as tested. |
| One-hot/kernel categorical treatment | Absent by design | Not a categorical donor; DarkoFit's ordered statistics are a better local fit. |
| Self-contained safe persistence | DarkoFit implemented; xRFM does not match it | Do not weaken the archive contract. |

DarkoFit's local-linear path fits a Hessian-weighted ridge update separately
inside each oblivious-tree leaf, using only numeric features selected by that
tree. It remains one update inside an additive boosting ensemble. xRFM instead
fits a nonlinear kernel expansion against local training centers, learns a
full or diagonal feature metric, and uses a separate routing tree to choose the
expert. The local basketball result closes the tested automatic linear-leaf
selector; it does not make these architectures equivalent.

Likewise, AGOP and TreeSHAP answer different questions. AGOP describes local
input sensitivity and learned directions. TreeSHAP decomposes a prediction
relative to a background distribution on DarkoFit's supported
scalar-oblivious lanes. Neither is causal, and AGOP should not be advertised
as a contribution decomposition.

### Steal A: run xRFM as an external comparator first

The highest-information action is an optional benchmark adapter, not a kernel
implementation inside DarkoFit. A minimum attribution panel is:

1. DarkoFit constant leaves.
2. DarkoFit explicit local-linear leaves, without reopening the automatic
   selector.
3. A global RFM or xRFM with a leaf cap larger than the fold.
4. Hard-routed xRFM with realized splits.
5. Soft-routed xRFM only after the hard model is measured.

If every real fold is below 60,000 rows, add a separately labeled forced-split
arm with a smaller leaf cap. Do not replace the official/default arm with that
choice after seeing outcomes.

Use the real group/rolling-origin gate, training-only preprocessing, and a
disjoint validation source for RFM iteration/bandwidth/routing-temperature
selection. Report RMSE or task loss, calibration, cold-group/future harm,
fit/predict time, CPU/GPU peak memory, realized leaves, evaluated experts per
row, support-center count, and artifact size. A tuned GPU win without its HPO
and deployment costs is not comparable to a single DarkoFit fit.

### Steal B: use AGOP as a regime-heterogeneity diagnostic

Before changing DarkoFit's predictor, inspect an external xRFM's leaf AGOPs:

- diagonal mass by original feature;
- leading eigenvalues and eigenvectors;
- stability of those quantities across group/time folds; and
- differences among leaves that persist on future data.

This can reveal whether different subpopulations genuinely use different
features. Full one-hot AGOPs must be mapped back to original categoricals, and
eigenvector signs/rotations need alignment before cross-fold comparison. A
visually appealing direction that is unstable across folds is not a feature
discovery result.

Do not try to compute ordinary derivatives of DarkoFit's piecewise-constant
predictions and call the result AGOP; those gradients are zero almost
everywhere. Use the smooth external model as the diagnostic or define a
separate, validated finite-difference object.

### Steal C: oblique top-level routing only after an external win

If xRFM beats both global RFM and DarkoFit on a prospective shifted-data gate,
the most separable donor mechanism is its router:

- learn a small number of supervised oblique directions from training data;
- split at predeclared balanced thresholds;
- fit an ordinary DarkoFit expert in each region; and
- compare hard routing with a single global DarkoFit model at matched compute.

This would be a mixture-of-experts product, not a leaf-wise builder option and
not permission to change the current ensemble default or learned-aggregation
policy. It needs a separate authorization contract, expert-level group/time
validation, minimum region mass, missing/unseen-category routing, and a
fallback for regions with too little data.

An axis-aligned constant-leaf control is essential. If oblique routing adds no
value without the kernel experts, then the transferable idea is the local
kernel model—not the tree.

### Steal D: soften only a measured boundary problem

Hard oblique routing can create a prediction discontinuity around a median
boundary. If boundary-local residual plots show that problem, test an
IQR-scaled logistic gate at the top-level router and cap the number of experts
evaluated per row. Compare:

1. hard routing;
2. one globally tuned routing temperature; and
3. no router / one global expert.

Temperature must be selected on predictions not already used to select the
router or expert hyperparameters unless the nesting is explicit. Reject soft
routing if its gain disappears at matched inference cost. Do not describe it
as calibration: it can change rank, class decisions, and every predicted
value.

### A full xRFM port has a large and incompatible proof surface

The released package is a PyTorch model with optional CUDA-accelerated kernels.
A native port would need much more than a new leaf-value array:

- oblique vectors and thresholds in a new routing payload;
- kernel centers, coefficients, bandwidth, exponent, and full/diagonal `M`
  for every leaf;
- bounded support-center and archive-size policies;
- classification target coding and probability normalization;
- one-hot categorical block identity, missing-value behavior, and feature-name
  recovery;
- explanation semantics for both routing and leaf sensitivities;
- deterministic CPU/GPU parity and device-independent loading; and
- validation ownership for per-leaf RFM iterations and soft-routing
  temperature.

DarkoFit's safe `.npz` archives are self-contained and strictly validated.
xRFM v0.4.5's
[state restoration](https://github.com/dmbeaglehole/xRFM/blob/v0.4.5/xrfm/xrfm.py#L1438-L1564)
requires the original training features to reconstruct kernel centers, and the
global routing-temperature configuration is not fully represented in that
state dictionary. That is acceptable research-package context, but it is not
a persistence contract to copy. A DarkoFit implementation would need to store
or distill centers explicitly, with artifact-size and training-data privacy
treated as first-class gates.

### Minimum discriminator and stop rule

Start with a synthetic target containing both an oblique regime boundary and
different local feature sets. For example, one side of `u^T x > 0` depends
smoothly on features 1–2 while the other side depends on features 3–4. The
experiment should establish:

- whether global RFM finds all features but loses the regime decomposition;
- whether xRFM recovers the boundary and leaf-specific directions;
- whether hard routing creates boundary harm that soft routing repairs; and
- whether DarkoFit's boosted axis-aligned trees already solve the task more
  cheaply.

Then run the same arms on one prospective real group/time protocol. Stop the
route if xRFM does not beat both its global-RFM ablation and DarkoFit, if the
gain concentrates in one task, or if it requires unacceptable GPU memory,
support-center storage, or multi-leaf inference.

### Do not steal

- Do not start a random Fourier-feature project: xRFM supplies no such donor
  evidence.
- Do not transplant forced median splits into the booster; balance serves the
  kernel-scaling architecture, not gradient boosting's immediate objective.
- Do not reopen automatic local-linear leaves under the xRFM name.
- Do not call routing temperature classifier calibration.
- Do not use AGOP sensitivity as a causal or SHAP-equivalent explanation.
- Do not import dense one-hot preprocessing as DarkoFit's categorical policy.
- Do not repeat the paper's asymptotic scaling claim without measuring the
  local kernel, full-matrix, GPU-memory, and persistence constants.

The decision is therefore stronger than “watchlist,” but narrower than “port
it”: xRFM should become a serious external regression comparator and a source
of regime-discovery experiments. Its router or soft boundary may become a
separate product line only after the external model proves that the local
feature-learning mechanism survives DarkoFit's real shifted-data gates.

## Extended dossier: iLTM

TabArena labels iLTM **Foundation Model**, but this is not a superficial
tree-adjacency judgment. Its tree-enabled checkpoints first fit XGBoost or
CatBoost, record the leaf reached by each row in every tree, and use those
partition addresses as input to the neural stages. Robust-only checkpoints are
also part of the tuned search and supply the critical non-tree ablation. In a
tree-enabled lane, the GBDT is not the final router and its leaf values are not
the final local predictors. The tree ensemble supplies a supervised,
discontinuous representation; a generated MLP, optional retrieval,
fine-tuning, and internal ensembling make the final prediction.

iLTM means **Integrated Large Tabular Model**. The current
[paper](https://arxiv.org/html/2511.15941),
[source at `11c69c7`](https://github.com/AI-sandbox/iLTM/tree/11c69c79701bdfa1dcbf7ca70f9fcfcb2d11b060),
and pinned
[TabArena adapter](https://github.com/autogluon/tabarena/tree/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/iltm)
all support that classification.

### The model is a staged neural-tree hybrid

The full path has several separable mechanisms:

1. **Initial representation.** A checkpoint selects an XGBoost leaf embedding,
   a CatBoost leaf embedding, robust-preprocessed raw features, or a
   concatenation of tree and raw lanes.
2. **Leaf-address encoding.** For every fitted tree, a row reaches exactly one
   leaf. iLTM one-hot encodes that leaf index and concatenates the blocks across
   trees. The result is a supervised, high-dimensional partition fingerprint.
3. **Fixed-size bridge.** A fixed random ReLU expansion approximates an
   arc-cosine kernel; PCA reduces the expansion to a fixed dimension; and the
   resulting columns are normalized. This lets one hypernetwork accept tables
   with different original feature counts and different leaf vocabularies.
4. **Dataset-conditioned weight generation.** A hypernetwork meta-trained on
   1,806 real classification datasets pools a labeled generation subset and
   emits the weights of a dataset-specific three-layer, 512-unit MLP.
5. **Optional task adaptation.** The generated MLP can be fine-tuned on the new
   table. Regression uses the classification-pretrained hypernetwork but needs
   fine-tuning to adapt the generated network to a continuous target.
6. **Optional retrieval.** The model computes cosine or Euclidean similarity in
   its penultimate representation, forms a temperature-weighted label average,
   and interpolates that result with the MLP output.
7. **Optional internal ensemble.** Different generation subsets or feature
   bags produce several generated predictors whose outputs are averaged.

This is much more than “GBDT plus an MLP.” The expensive pretraining amortizes
weight initialization across datasets; the random-feature/PCA bridge solves a
variable-schema problem; and retrieval preserves labeled training examples as
an inference-time context. Conversely, the GBDT is still a real fitted,
target-aware part of each tree-enabled checkpoint, not merely a tree-shaped
synthetic pretraining prior.

### What a leaf address contains—and what it loses

If tree `t` sends a row to leaf `L_t(x)`, the complete, unpruned one-hot
representation contains one active coordinate for `(t, L_t(x))`. Across `T`
trees, that full address has `T` active cells. The released frequency-selected
8,192-column representation can retain fewer active cells, and an unseen leaf
block is ignored. The address encodes several useful facts:

- every split interaction along a path has already been resolved into a region;
- numeric scale and arbitrary monotone transformations matter only when they
  change a learned split;
- mixed types and missingness inherit the donor GBDT's handling; and
- two rows' fraction of shared leaves defines a supervised tree-proximity
  measure.

It does **not** explicitly encode which individual split was decisive, path
order, distance from a threshold, or the GBDT's leaf prediction as downstream
features. The retained fitted tree can still recover a leaf's path and value
from `(tree_id, leaf_id)`. A rare one-hot cell can mean a useful interaction, a
tiny overfit region, or simple sampling noise. The address is target-informed
because the tree was fit on labels. It must not be described as unsupervised
embedding or be generated from the whole dataset before a prospective split.

The released implementation obtains XGBoost leaves with `pred_leaf=True` and
CatBoost leaves with `calc_leaf_indexes`, then fits a one-hot encoder. Its
[current tree-embedding code](https://github.com/AI-sandbox/iLTM/blob/11c69c79701bdfa1dcbf7ca70f9fcfcb2d11b060/iltm/tree_embedding.py#L595-L736)
uses a dense one-hot output and can retain a frequency-selected subset within
an 8,192-column budget. That is an implementation fact to improve upon, not a
storage design to copy into DarkoFit.

### The fixed-size projection is an interoperability layer

The random-ReLU/PCA stage is easy to mistake for the main predictive idea. Its
primary architectural job is to turn a table-specific input dimension into a
fixed dimension suitable for a shared hypernetwork. The random expansion adds
a nonlinear arc-cosine-kernel approximation before PCA selects directions of
variance; normalization then conditions the generated MLP's input.

DarkoFit does not need that bridge merely to fit one task-specific ridge or
logistic head. A sparse linear head can consume `(tree, leaf)` columns directly
and retain their identity. Random features become relevant only if a nonlinear
second stage has already shown incremental value and a fixed interface is
required. PCA components, means, random matrices, normalization statistics,
and original column/block identity would all become new fitted state.

### The hypernetwork and retrieval solve different problems

The hypernetwork learns a mapping from a labeled dataset summary to a useful
MLP initialization. At deployment, its weights are frozen; the new table
generates a main network, which can then be fine-tuned. This is meta-learning,
not ordinary hyperparameter tuning and not a generic `auto` rule.

Retrieval is a separate prediction path. It operates in the generated MLP's
penultimate space, not in raw feature space and not directly in leaf-address
Hamming distance. The final output mixes the main-network and neighbor-label
signals with a learned or tuned coefficient. A model can use tree embeddings
without retrieval, robust raw features with retrieval, or neither. Those
factorizations are essential when interpreting the leaderboard.

The paper's own ablation is cautionary: GBDT embeddings improve initial and
few-shot accuracy, but robust-only preprocessing can match or exceed the tree
lanes after full fine-tuning. Fine-tuning is generally the largest accuracy
increment. The paper therefore does not establish that leaf addresses explain
the full model's final score.

### What the pinned leaderboard actually says

All three standard iLTM rows are verified, non-imputed, and GPU-evaluated. The
`#` values below reproduce the CSV's zero-based row index.

| iLTM row | Elo | 95% interval | Leaderboard row `#` | Train / predict seconds per 1,000 rows |
|---|---:|---:|---:|---:|
| Default | 1089 | +46 / -48 | 60 | 296.64 / 68.173 |
| Tuned | 1281 | +35 / -32 | 37 | 12,685.08 / 62.130 |
| Tuned + ensemble | **1383** | +41 / -40 | **18** | 12,685.08 / 464.370 |

The 192-Elo default-to-tuned jump and further 102 Elo from TabArena's post-hoc
ensemble are the dominant empirical fact. The best row is competitive with the
leading tree group—below tuned-plus-ensembled LightGBM at 1408 and CatBoost at
1397, and essentially level with tuned CatBoost at 1385—but it is not a cheap
single-fit result. TabArena's tuned training-time field is the cumulative
search/portfolio cost, not the latency of one fitted iLTM: it reports more than
12,000 seconds per 1,000 rows for the tuned protocol. The top post-hoc ensemble
then reports 464 seconds of prediction time per 1,000 rows. Overlapping
intervals also prevent a decisive ordering against nearby models.

TabArena's pinned
[iLTM search-space generator](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/iltm/hpo.py)
defaults to 200 configurations, but the module's checked-in suite emitter
requests 25 random configurations plus the manual default. Those configurations
span:

- eight checkpoints: raw-plus-XGBoost, raw-plus-CatBoost, two robust-only
  variants, XGBoost-only, CatBoost-only, robust-plus-retrieval, and
  CatBoost-plus-retrieval;
- internal ensemble sizes from 4 to 64 generated predictors;
- fine-tuning optimizer, learning rate, dropout, batch size, and 2,048 or 4,096
  maximum steps;
- tree family settings including 100–300 trees, depth 4–6, learning rate,
  minimum leaf size, row/feature sampling, regularization, and tree/main data
  split;
- retrieval on/off, mixture weight, temperature, and cosine versus Euclidean
  distance; and
- prediction clipping, scheduler floor, and correlation-based feature count.

The tuned row can therefore select different representations and large
internal ensembles on different datasets. The tuned-plus-ensemble row then
adds TabArena's cross-configuration aggregation. Neither row identifies a
causal “leaf embedding effect.” Any DarkoFit intake must recreate the relevant
ablation rather than citing Elo 1383 as proof for one component.

### The tree/main split is useful but not automatically honest

iLTM can fit the tree and neural stages on all rows or separate them. In the
released `dynamic` policy, tables below 2,000 rows reuse all rows for both
stages; tables from 2,000 through 200,000 use a 50/50 split; and larger tables
fit the tree on 100,000 rows and use the remainder for the main stage. See the
[fixed source](https://github.com/AI-sandbox/iLTM/blob/11c69c79701bdfa1dcbf7ca70f9fcfcb2d11b060/iltm/inference_interface.py#L1316-L1358).

That is a useful attribution control because the second stage can be trained on
rows the GBDT did not fit. It is not a universal leakage proof: the small-data
branch deliberately reuses rows, random splitting is not a valid future/group
boundary, and downstream HPO still needs an outer selection gate. DarkoFit
should use chronological/group-honest stage splits first and compare a rotated
cross-fit ensemble only if the single honest split is promising.

### DarkoFit overlap

| iLTM mechanism | Live DarkoFit status | Intake decision |
|---|---|---|
| Fit target-aware boosted trees | Implemented | The donor is the downstream use of partitions, not tree fitting itself. |
| Per-tree leaf assignment | Internal primitive only | Native tree objects implement `apply()`, and boosting uses it internally; estimators expose no supported leaf-transform API. |
| Sparse/dense one-hot leaf-address representation | Absent | Genuine new representation; do not materialize the released dense design. |
| Global second-stage model on leaf addresses | Absent | Distinct from fitting a local linear update inside each tree leaf. |
| Raw-feature plus leaf-address lanes | Absent | Useful factorial experiment, not the closed generic cross-feature route. |
| Honest or cross-fitted leaf stack | Absent | Required proof mechanism before any second-stage quality claim. |
| Dataset-conditioned pretrained hypernetwork | Absent | Very large external research program; not an `auto_params` extension. |
| Learned-representation neighbor retrieval | Absent | Adjacent control, with temporal/entity leakage risk. |
| Internal 4–64-model generated-neural ensemble | No matching mechanism | Public bagged boosted ensembles already exist; a generated-member/feature-bag path is different, and automatic/default promotion remains closed. |
| Self-contained safe archive | Implemented locally, not supplied by the paper's architecture | Any local hybrid would need an explicit artifact and training-context contract. |

The critical distinction from local-linear leaves is scope. DarkoFit's linear
leaf computes a Hessian-weighted ridge update inside one fitted tree region and
adds it to the boosting ensemble. An iLTM-style address concatenates regions
from many trees and lets one global downstream model learn across their
co-occurrence patterns. Closing automatic local-linear promotion does not close
this representation, but it raises the burden for a more expensive successor.

### Steal A: require a checkpoint attribution panel first

The highest-information first action is an external iLTM adapter with a small,
predeclared checkpoint panel:

1. robust raw features only;
2. XGBoost leaf addresses only;
3. robust raw plus XGBoost addresses;
4. the corresponding CatBoost lanes if the first tree comparison is positive;
5. retrieval off versus on for the winning representation; and
6. one fixed internal ensemble size before any broad search.

Run the panel on the real rolling-origin/group gate and hold fine-tuning steps,
ensemble size, and compute budget fixed across the key raw-only/tree-only/concat
comparison. Report the exact checkpoint, tree/main split, generated member
count, retrieval settings, fit/predict time, GPU memory, and artifact/context
size. If tree-only and concatenated checkpoints do not beat robust-only after
matched fine-tuning, iLTM is evidence for meta-initialization or retrieval—not
for a DarkoFit leaf-address project.

### Steal B: test a sparse leaf-address residual head

The smallest native model experiment does not need PyTorch, random features,
PCA, or a hypernetwork:

1. Fit a frozen DarkoFit model on an earlier/group-disjoint stage-A sample.
2. Transform stage-B rows to integer `(tree_id, leaf_id)` pairs.
3. Build a CSR or feature-hashed address matrix with one active value per tree.
4. For regression, fit a strongly regularized ridge head to **out-of-stage
   residuals**. For classification, fit a regularized logistic correction on
   the original labels with the frozen base logit as an offset/input; do not
   regress probability or logit residuals as ordinary targets.
5. Evaluate the frozen base plus shrunk correction on untouched stage C.
6. Rotate A/B only after the single honest split wins, keeping each fold's leaf
   vocabulary and head separate and averaging final predictions.

Required controls are the frozen DarkoFit prediction, a one-feature recalibration
head, an equally regularized raw-feature head, leaf-only, and raw-plus-leaf. Use
one fixed leaf hash/budget and regularization grid selected inside development
data. This establishes whether partition identity adds information beyond the
base score rather than merely refitting it.

Do not concatenate fold-specific leaf IDs into one fake common vocabulary.
Tree 7, leaf 3 from two separately fitted models is not the same region. Do not
fit the first-stage tree on the confirmation rows, and do not let a direct head
memorize tiny leaves. Minimum cell support, coefficient shrinkage, feature
budget, and archive size belong in the protocol.

### Steal C: use tree proximity for a causal residual smoother

The address also defines a low-cost local-model experiment without a neural
embedding. For two rows,

`proximity(i, j) = (1 / T) * sum_t 1[L_t(i) == L_t(j)]`.

Use one frozen stage-A router to encode both the stage-B neighbor pool and
stage-C queries. For a new row, retrieve only eligible past/group-safe stage-B
rows with high proximity, softmax their **out-of-stage** residuals, shrink the
correction toward zero as effective neighbor count falls, and add it to the
frozen base prediction. If folds are used, keep each fold's router, residual
pool, and smoother isolated and average only final predictions; never mix
residuals from fold-specific models in one leaf vocabulary. Start with
regression; probability-space corrections need a separate logit/calibration
contract.

This is not already implemented and is different from local-linear leaves: it
borrows information across training examples that repeatedly share ensemble
regions. It also carries severe leakage risk. Neighbor pools must obey the same
time/entity boundary as the model, and no row may contribute an in-sample
residual. Compare against raw-feature kNN, uniform eligible-row subsampling, and
no correction. ModernNCA and TabDPT are useful external controls precisely here:
they test whether learned/raw locality works without tree addresses.

Stop if the effect disappears against uniform context, is concentrated in
near-duplicate entities, harms future/cold groups, or requires an unbounded
training-row index at inference.

### Steal D: make the representation ablation the product decision

iLTM's most transferable discipline is keeping representation lanes separable.
The decisive local factorial is:

| Arm | Frozen base score | Raw features | Leaf addresses | Question |
|---|---:|---:|---:|---|
| Base | Yes | No | No | Current quality and cost. |
| Score head | Yes | No | No | Can a trivial refit explain the gain? |
| Raw residual head | Yes | Yes | No | Does a cheap smooth correction suffice? |
| Leaf residual head | Yes | No | Yes | Do learned partitions add incremental structure? |
| Raw + leaf head | Yes | Yes | Yes | Is their interaction worth the larger artifact? |

Keep the same stage split, regularization budget, and selection data across
arms. A concat win without a leaf-only advantage can still be valuable, but it
does not justify claiming that the tree address was the mechanism. A result
that exists only when the base trees and head share the same fitting rows is a
stop, not a reason to relax the protocol.

### Steal E: leaf-address drift before continual refitting

A prediction-free diagnostic is cheaper still. Record per-tree leaf occupancy,
effective occupied leaves, and co-leaf collision rates on training and future
windows. Large, stable changes in those distributions can expose regime drift
even when aggregate raw-feature marginals move little. Relate the drift score
to future residual deterioration before proposing an alert or refresh policy.

This is a concrete form of the otherwise broad Perpetual-style drift idea. It
does not require a hypernetwork or stored labels at inference. Reject it if the
statistic is dominated by sample size, random seed, or expected arrival of new
categories, or if it adds no warning beyond monitored loss and raw-feature
drift.

### A full iLTM port is not a bounded DarkoFit feature

A native reproduction would introduce several independent systems:

- XGBoost/CatBoost fitting and their categorical/missing semantics;
- leaf vocabularies, one-hot selection, raw/tree concatenation identity, and
  unseen-leaf behavior;
- random-feature matrices, PCA state, and normalization state;
- downloaded pretrained hypernetwork weights and a generated PyTorch MLP;
- fine-tuning optimizer/scheduler state and validation ownership;
- retrieval embeddings, labels, privacy/retention policy, and neighbor index;
- internal model generation and aggregation; and
- device-independent loading, deterministic CPU/GPU behavior, and dependency
  isolation.

DarkoFit's safe archive would need to store and validate every fitted component
or explicitly mark the artifact as an external composite. Retaining retrieval
context can retain transformed training examples and labels, which is a privacy
and artifact-size decision, not ordinary model metadata. None of this belongs
in the current estimator merely because the top leaderboard row is strong.

### Minimum discriminator and stop rule

First prove representation plumbing on a development-only interaction problem:
fixed shallow trees should create useful partition addresses; a sparse leaf
head should beat an equally sized raw linear head; and it should round-trip
without a dense matrix. That smoke test is not quality evidence.

The real discriminator is the three-stage prospective panel in Steals B–D.
Stop the native route if any of the following holds:

- the external tree-only/concat checkpoints fail to beat robust-only at matched
  fine-tuning and internal ensemble size;
- the sparse leaf head fails to beat the base-score and raw-feature controls;
- the win disappears under honest group/time separation or one fold/task
  supplies most of the gain;
- same-row tree/head fitting is required;
- hashed/CSR state, prediction latency, or archive size exceeds the declared
  budget; or
- a non-tree residual smoother wins, showing that locality rather than tree
  partitions is the transferable mechanism.

### Do not steal

- Do not import the 1,806-dataset hypernetwork as the first experiment.
- Do not treat Elo 1383 as attribution to leaf embeddings; the search mixes
  checkpoints, fine-tuning, retrieval, internal ensembles, and post-hoc
  ensembling.
- Do not materialize dense one-hot leaf matrices inside DarkoFit.
- Do not assume leaf IDs from separately fitted models share meaning.
- Do not use in-sample residuals or random splits where the real gate is
  temporal/entity-based.
- Do not reopen automatic local-linear promotion or change ensemble defaults
  under the iLTM name; the existing public ensemble API is not absent.
- Do not retain training examples for retrieval without an explicit privacy,
  size, persistence, and deletion contract.
- Do not call a random-feature/PCA bridge necessary until a fixed-dimensional
  nonlinear downstream model has earned it.

The intake decision is therefore: iLTM is a serious external comparator and
the strongest donor for a bounded **leaf-address representation** experiment.
Its leaderboard result is not evidence to reproduce the foundation model. The
native gate is whether sparse, honest partition fingerprints add prospective
value beyond DarkoFit's own score and cheap raw-feature controls.

## Extended dossier: GPBoost

### Bottom line: steal the residual structure, not the engine

**Qualified yes.** GPBoost contains an idea DarkoFit does not currently have
and that is unusually relevant to repeated-entity, longitudinal, spatial, and
other clustered data: split prediction into

`nonlinear mean from trees + structured latent residual`.

The best first donor is the grouped random-intercept lane, not a full Gaussian
process. A small, generic, strongly shrunk group-residual component could test
whether repeated entities retain predictable residual signal after DarkoFit's
features and trees. If that simple lane cannot establish honest prospective
value, covariance-aware tree fitting and Gaussian-process machinery should
stop.

This is not a recommendation to copy GPBoost's tree builder. Its official
[implementation](https://github.com/fabsig/GPBoost) uses LightGBM for tree
growth plus a C++/Eigen random-effects system. DarkoFit's NumPy/Numba/sklearn
dependency contract and existing native tree engine make that the wrong unit of
transfer.

### The statistical model changes the independence assumption

For Gaussian regression, the core model is

`y = F(X) + Zb + epsilon`,

with `b ~ N(0, Sigma)` and `epsilon ~ N(0, sigma^2 I)`. `F(X)` is the boosted
nonlinear fixed-effect mean. `Zb` can be a grouped random intercept, one or more
random slopes, a Gaussian process evaluated at supplied coordinates, or a
combination. After integrating over `b`,

`y ~ N(F(X), Psi)`, where `Psi = Z Sigma Z^T + sigma^2 I`.

Ordinary squared-error boosting is the special independent case in which
`Psi` is proportional to the identity. GPBoost instead minimizes the marginal
negative log-likelihood

`0.5 * (y - F)^T Psi^-1 (y - F) + 0.5 * log(det(Psi)) + constant`.

That distinction has two consequences:

- repeated or nearby observations do not count as independent copies of the
  same evidence; and
- the fitted covariance model supplies conditional random-effect corrections
  and probabilistic predictions, not merely a training-time weight.

The [JMLR paper](https://www.jmlr.org/papers/v23/20-322.html) describes an
alternating procedure. At each boosting iteration it updates covariance
parameters `theta` conditional on the current `F`, then adds a tree using a
functional gradient, Newton step, or hybrid update. With covariance fixed, the
gradient pseudo-response is proportional to `Psi^-1 (y - F)`. In the hybrid
tree case, split structure can be learned from the gradient step and terminal
values updated by generalized least squares:

`gamma = (H^T Psi^-1 H)^-1 H^T Psi^-1 (y - F)`,

where `H` maps rows to the new tree's leaves. This is more than fitting a tree
and smoothing its residuals afterward: residual dependence changes what the
tree learns.

### GPBoost contains several separable mechanisms

| Mechanism | What it buys | DarkoFit relevance |
|---|---|---|
| Grouped random intercepts | Partial pooling for repeated levels; residuals from small groups shrink more strongly toward zero. | Strongest first donor when a semantically meaningful entity or cluster ID is supplied. |
| Grouped random slopes | Lets the effect of a numeric covariate vary by group. | Potential follow-up only after intercept heterogeneity is proven. |
| Crossed or nested group effects | Represents multiple grouping structures without flattening every combination into one category. | Useful in principle, but expands API, fitting, persistence, and identifiability scope. |
| Gaussian-process residual | Smooths residual signal over user-supplied spatial, temporal, or other meaningful coordinates. | High upside on genuine geometry; inappropriate for arbitrary tabular columns. |
| Joint tree/covariance training | Uses residual precision in boosting and re-estimates covariance parameters as the mean improves. | The most distinctive algorithmic idea and the largest native implementation. |
| Predictive variance and covariance | Returns uncertainty for individual rows and correlated sets of predictions. | Distinct from DarkoFit's current marginal distributional heads. |
| Large-data approximations | Vecchia, inducing-point/full-scale, tapering, and iterative methods reduce exact GP cost. | Necessary for a broad GP feature, but not for the first group-intercept probe. |
| Non-Gaussian latent models | Extends the latent Gaussian approach to binary, count, heavy-tailed, and other likelihoods. | Do not start here; Gaussian regression provides the cleanest discriminator. |

The official
[`GPModel` API](https://gpboost.readthedocs.io/en/stable/pythonapi/gpboost.GPModel.html)
accepts group labels, random-coefficient data, GP coordinates, covariance
functions, independent-cluster IDs, and approximation controls. Prediction can
return a posterior mean, per-row variance, a covariance matrix, or posterior
samples. Exact GP algebra is not cheap: without an approximation, the paper
gives cubic time and quadratic memory in the latent process size. The official
[efficiency guidance](https://gpboost.readthedocs.io/en/stable/Computational_efficiency.html)
generally recommends Vecchia and exposes accuracy/runtime controls such as
neighbor count.

### Evidence boundary: there is no TabArena grade

GPBoost does **not** appear in the dossier's already-frozen TabArena CSV or
model-registry review. It receives no Elo, rank, or “grades well” claim here.
The paper reports simulation, grouped-data, spatial-data, and UCI results, but
those author experiments are neither TabArena nor a DarkoFit reproduction.

The workspace contains v1 and v2 protocol records
(`benchmarks/gpboost_basketball_v1_protocol.md` and
`benchmarks/gpboost_basketball_v2_protocol.md`) plus
`benchmarks/bench_gpboost_basketball.py`. Both designs are deliberately
**feature-only**:

- they instantiate `GPBoostRegressor` without a `GPModel`;
- they pass no entity ID, group effect, coordinate, or residual covariance
  structure; and
- they compare public defaults and a roughly aligned tree budget.

The official
[`GPBoostRegressor.fit()` contract](https://gpboost.readthedocs.io/en/stable/pythonapi/gpboost.GPBoostRegressor.html)
uses independent tree boosting when `gp_model=None`. Consequently, that draft
can characterize the GPBoost/LightGBM-derived tree engine against DarkoFit, but
it cannot test GPBoost's defining mechanism. The four-thread v1 attempt failed
its repeatability check before summary construction and emitted no raw artifact;
the v2 record changes only to a single-thread deterministic envelope. A future
valid feature-only result—positive or negative—must not be used to accept or
reject grouped random effects.

### DarkoFit overlap and the real gap

| GPBoost concept | Live DarkoFit status | Consequence |
|---|---|---|
| Nonlinear boosted mean `F(X)` | Implemented | Do not replace DarkoFit's tree engine. |
| `groups=` during fitting | Split/sampling metadata only | Groups keep automatic validation and group-bootstrap OOB partitions intact; they are not learned predictive effects. |
| Ordered target statistics for categoricals | Implemented, different semantics | Target encoding supplies a leakage-controlled feature to trees. It is not a residual random effect, does not induce cross-row covariance, and need not isolate the group contribution. |
| Per-feature-value distributional scale calibration | Implemented, narrower | It can adjust held-out marginal scale by a calibration feature; it does not estimate group mean corrections or shared latent covariance. |
| Per-row `sample_weight` | Implemented, diagonal only | Reliability weights rescale row losses. They cannot represent off-diagonal dependence. |
| Grouped random intercept/slope | Absent | No variance component, posterior shrinkage map, seen-group correction, or cold-group variance contract exists. |
| GP residual over coordinates | Absent | No coordinate metadata, kernel covariance, inducing points, Vecchia graph, or posterior context exists. |
| `Psi^-1` gradient / GLS leaf update | Absent | Current losses expose rowwise gradients and diagonal Hessians. Cross-row curvature is outside the tree-builder contract. |
| Marginal predictive distributions | Implemented | Gaussian, Student-t, LogNormal, Poisson, and Negative Binomial heads already cover per-row distributions. |
| Cross-row predictive covariance | Absent | `predict_variance()`, intervals, and sampling are marginal by row; shared group/process uncertainty is not represented. |
| Safe structured-residual archive | Absent | A native implementation would need variance parameters, group dictionaries or process state, training-context policy, and strict load validation. |

A nearby idea exists in
`docs/archive/DARKOFIT_PRACTICAL_WISHLIST.md`: grouped empirical-residual
shrinkage was marked “revisit on demand” and dropped from that committed
roadmap pending a real consumer and acceptance protocol. It was not implemented
and was not experimentally disproven. It also targeted a grouped **residual
distribution** wrapper, whereas GPBoost estimates a latent residual mean and
covariance and can let that covariance alter tree fitting. GPBoost therefore
supplies a concrete donor and discriminator, not proof that the archived design
should silently be revived.

### Steal A: run the mechanism attribution externally first

When a separate run is authorized, the most informative experiment is not
“GPBoost versus DarkoFit.” It is the within-GPBoost contrast that isolates the
`GPModel`, with DarkoFit added as the product reference:

| Arm | Tree predictor | Structured component | Question |
|---|---|---|---|
| DarkoFit base | DarkoFit | None | Product quality and cost reference. |
| GPBoost tree-only | GPBoost/LightGBM-derived | None | Engine and wrapper control. |
| GPBoost + group intercept | Same GPBoost tree settings | One grouped random intercept | Does residual group dependence add value beyond that engine? |
| GPBoost + meaningful coordinates | Same GPBoost tree settings | One prespecified GP | Only if the dataset has genuine coordinate semantics. |
| DarkoFit + residual sidecar | Frozen DarkoFit | Bounded prototype from Steal B | Is a small native approximation enough? |

The GPBoost tree-only and GPBoost-plus-`GPModel` arms must share features, tree
configuration, selection budget, and outer split. Otherwise a difference
cannot be attributed to residual structure.

Use at least two prospective views:

1. **Seen-group future:** training contains earlier observations for an entity,
   and evaluation contains later observations. This is where a posterior group
   correction can legitimately help.
2. **Cold group:** evaluation groups are absent from training. This exposes
   whether the tree mean remains portable and whether uncertainty honestly
   expands instead of leaking identity.

In the current basketball draft, player-disjoint primary folds are a cold-player
point-prediction view. The held-team `seen_player` slice is the only named view
in which a player effect learned on primary rows could contribute to the
conditional mean. A future mechanism protocol must report primary/cold,
held/seen, and held/cold separately rather than collapsing them into one score.

Report row-level RMSE plus negative log-likelihood or CRPS, calibration by group
support, and uncertainty for prespecified aggregates. Report fit time, predict
time, peak memory, retained group/process state, and the fraction of evaluation
rows assigned to seen groups. Random row splits that mix future observations
of the same entity into training are not acceptable evidence. DarkoFit's
current `groups=` path protects an automatically created validation split, but
an explicit `eval_set` has no companion `eval_groups` disjointness check; a
mechanism runner must validate that boundary itself.

Stop the path if the `GPModel` increment:

- disappears on the seen-group prospective view;
- exists only under random-row leakage;
- harms cold groups beyond the declared bound;
- estimates a variance component at a numerical boundary or changes materially
  across folds/seeds;
- adds no aggregate-uncertainty value over marginal DarkoFit distributions; or
- requires cost or retained context outside the declared product envelope.

No such run is authorized by this dossier.

### Steal B: a bounded shrunken group-residual component

The smallest native experiment can be written without a dense covariance
matrix. For regression, freeze a DarkoFit base on stage A. On disjoint,
prospectively eligible stage-B rows, compute residuals
`r_i = y_i - F_A(x_i)`. For group `g`, define

`S_g = sum_i w_i r_i`, `W_g = sum_i w_i`,

estimate positive random-effect and error variances `tau^2` and `sigma^2`, and
predict the weighted posterior-mean correction

`u_g = tau^2 * S_g / (sigma^2 + tau^2 * W_g)`

or, equivalently, `u_g = S_g / (W_g + sigma^2 / tau^2)`. Estimate the variance
components from honest cross-fitted or out-of-stage residuals with a
prespecified marginal-likelihood procedure; do not hand-pick the shrinkage from
the confirmation result. Prediction is `F_A(x) + u_g` for a known group and
`F_A(x)` for an unseen group.

This weighted formula treats `w_i` as inverse-noise precision. DarkoFit's
current `sample_weight` contract can also express modeling importance, so the
prototype must either start unweighted or require an explicit precision-weight
interpretation. It must not silently reinterpret every existing sample weight
as an observation-variance model.

Required controls are:

- frozen DarkoFit with no correction;
- one global intercept/recalibration correction;
- group ID supplied as an ordinary categorical feature using the existing
  preprocessing path;
- unshrunk group residual means;
- a support-only rule with no target residual; and
- external GPBoost with a true grouped random effect.

This is not “target encoding again.” It corrects what the fitted feature model
systematically leaves behind, has an explicit zero-mean cold-group fallback,
and isolates the group contribution outside tree interactions. It is also not
full GPBoost: the base tree never sees `Psi^-1`, covariance parameters are not
jointly updated, and a three-stage fit sacrifices data unless a later honest
refit design is established.

The first prototype should remain regression-only. Classification needs a
penalized random intercept on the link scale and a proper likelihood; averaging
probability residuals is not an acceptable shortcut. Any artifact must define
whether raw group labels are retained, how unseen values map to zero, how
deletion works, and whether the group dictionary is sensitive data.

Before quality work, synthetic invariants should cover `tau^2 = 0` recovery,
row-order and group-label-renaming invariance, global weight-scale invariance
under the declared weight contract, zero-weight-row invariance, unseen-group
zero mean, monotone shrinkage with group support, and exact safe-archive
round-trip.

### Steal C: a covariance-aware frozen-tree refit

If Steal B and external full GPBoost both win, the next experiment should move
one step closer to the actual algorithm without rewriting split search.
Freeze one tree structure and refit only its leaf values under a single grouped
random-intercept covariance.

For an unweighted group of size `n_g`,

`Psi_g = sigma^2 I + tau^2 1 1^T`.

Its inverse can be applied blockwise with the Sherman-Morrison identity:

`Psi_g^-1 r = r / sigma^2 - [tau^2 / (sigma^2 * (sigma^2 + n_g tau^2))] * 1 * sum(r)`.

That makes the precision operation linear in rows and avoids an `n x n`
matrix. A correctness prototype should:

1. estimate or fix positive `sigma^2` and `tau^2` on development data;
2. compare the blockwise operator against `numpy.linalg.solve` on small
   synthetic groups;
3. refit a frozen leaf-membership matrix with the GLS formula;
4. prove exact fallback to ordinary least squares as `tau^2 -> 0`; and
5. measure seen-group, cold-group, and cost effects without changing splits.

This probe asks whether covariance-aware terminal values matter. It does not
show that covariance-aware split selection or per-iteration variance-component
updates are worth implementing. A full GPBoost-style gradient changes the
pseudo-response for every row, while a full Newton treatment has non-diagonal
curvature that DarkoFit's current rowwise Hessian interface cannot express.

### Steal D: covariance-aware uncertainty for aggregates

DarkoFit's distributional heads already provide strong marginal machinery.
The distinct GPBoost lesson is that two predictions can share uncertainty. For
a random intercept, two rows in the same group have positive latent covariance;
for a GP, covariance varies with coordinate distance. This changes the
uncertainty of sums and averages even when every row's marginal variance is
unchanged.

A bounded first interface should expose the posterior variance of the grouped
effect before returning a dense covariance matrix. For the model above,

`Var(b_g | data) = tau^2 * sigma^2 / (sigma^2 + tau^2 * W_g)`,

while a new group retains prior group-effect variance `tau^2`. Candidate
research operations are:

- sample one shared latent group effect plus independent row noise;
- return covariance blocks for explicitly requested groups; or
- compute `Var(a^T y)` for a supplied sparse aggregation vector `a`.

The decision metric is calibration and sharpness for prespecified aggregates,
not TabArena Elo or point RMSE. Current split-conformal intervals establish
marginal coverage only; they do not imply coverage for a sum of correlated
rows. This uncertainty route needs its own consumer and acceptance protocol,
consistent with the archived decision to defer generic joint-output work.

### Do not start with a Gaussian-process feature

GP residual surfaces are attractive, but they are a poor first port:

- coordinate columns must have genuine metric semantics; arbitrary standardized
  tabular features do not automatically define a trustworthy distance;
- exact GP work scales cubically in latent locations and quadratically in
  memory;
- Vecchia orderings, neighbor graphs, inducing points, covariance kernels,
  iterative tolerances, and predictive approximation types all become fitted
  behavior;
- prediction may need retained training context or a process-specific state;
  and
- a broad kernel API would be a second modeling library inside DarkoFit.

If a grouped intercept proves useful and a concrete spatial or temporal
consumer then appears, the caller should declare coordinate semantics while
the estimator automatically chooses a bounded approximation inside a documented
envelope. DarkoFit should not guess that an arbitrary pair of numeric columns
is “space.”

### Automation-first destination

The end state cannot be a maze of `use_random_effect`, covariance-kernel, and
approximation knobs. A product-quality path would be:

1. the caller supplies semantic metadata—at fit and prediction time through
   distinct research surfaces such as `effect_groups` and
   `effect_groups_pred`, and much later through explicit coordinates;
2. an explicit research/opt-in component is characterized under Tier E;
3. an automatic selector chooses **none versus grouped residual** only from
   leakage-safe development evidence and records its reason, variance estimate,
   expected seen-group coverage, and fallback behavior; and
4. default automatic engagement takes the full Tier-D path with prospective
   power, harm bounds, concentration checks, and no rerun-to-improve.

Two useful development diagnostics are cross-fitted residual intraclass
correlation and the marginal-likelihood gain of a one-component group model over
an iid residual model. They are selector inputs, not acceptance criteria: both
must be computed inside the development boundary, and unstable or negligible
evidence should select **none**. The eventual product decision must still be
made by the prospective Tier-D outcome and harm contract.

Supplying effect-group identity is data schema, not asking users to hand-tune a
model. The model should decide whether that structure is useful. Conversely,
silently changing the meaning of today's `groups=` argument would be an API
break: currently it controls split and sampling integrity, not the prediction
function. `groups` and future `effect_groups` can be equal, but they answer
different questions and must remain explicit. A predictive group component
needs an explicit research contract and versioned persistence before
automation.

### Do not steal

- Do not port GPBoost's LightGBM tree engine or C++/Eigen stack.
- Do not call the existing feature-only draft a test of Gaussian-process
  boosting.
- Do not infer group effects from every high-cardinality categorical column;
  exchangeable residual identity is a stronger assumption than “categorical.”
- Do not use random-row validation when the same entity can appear on both
  sides of the split.
- Do not fit group corrections from in-sample residuals and report them on the
  same rows.
- Do not build a dense covariance matrix for the first random-intercept
  experiment.
- Do not treat sample weights as a substitute for off-diagonal covariance.
- Do not claim that marginal distributional calibration already covers
  correlated aggregates.
- Do not begin with crossed effects, random slopes, non-Gaussian Laplace
  machinery, ARD kernels, or Vecchia.
- Do not persist raw entity IDs or training coordinates without an explicit
  privacy, size, deletion, and safe-load contract.
- Do not describe GPBoost as TabArena-proven.

The intake decision is therefore: **steal one idea, conditionally**. First use a
true external `GPModel` arm to prove that structured residual dependence adds
value beyond GPBoost's own tree-only control. If it does, prototype the generic
shrunken group-residual component in Steal B. Only a stable positive result from
both should fund covariance-aware leaf refitting. A native GP, full joint
boosting rewrite, or broad mixed-effects API is not justified now.

## Boundary screen: high scorers that are not tree-adjacent

The same pinned leaderboard contains several excellent models whose local,
ensemble, or foundation-model behavior can look tree-like from a distance.
They fail the fitted-tree/partition-address rule and should not receive full
tree dossiers.

| Model | Best pinned all-task row | Why it is tempting | Boundary decision |
|---|---:|---|---|
| ModernNCA | Elo 1368, row `#20` | Learns a nonlinear metric and predicts by a soft label-weighted neighborhood. | No tree, threshold, hierarchy, or finite leaf expert. Use as the non-tree locality control for a leaf-proximity smoother. |
| TabDPT | Elo 1437: Turbo default row `#10` (unverified) and standard tuned-plus-ensemble row `#11` (verified) | Standard v1.1 retrieves selected rows as labeled context; Turbo uses uniform context subsampling. | Both are row-token in-context Transformers, not partition models. The tied Turbo row makes uniform context a mandatory control before crediting expensive retrieval. |
| LimiX | Elo 1344, row `#24` | Its synthetic SCM generator can use decision-tree edge functions, and inference includes attention-guided retrieval. | The fitted predictor is a two-axis Transformer; trees generate some pretraining tasks but are not its inference representation. |
| TabM | Elo 1425, row `#12` | Efficiently trains a weight-sharing ensemble of MLP branches. | Ensembling alone is not tree adjacency. DarkoFit already has public bagged ensembles; learned aggregation or default-policy changes remain separately gated. |
| RealMLP | Elo 1481, row `#9` | Strong irregular-function baseline with tuning and ensembling. | A tuned MLP pipeline with no tree-derived partition; useful broad comparator, not a tree donor. |
| SAP-RPT-OSS / ConTextTab | Elo 1268, row `#41`, unverified | The package name can be misread as “random projection tree.” | The linked method is a semantics-aware table-native ICL Transformer with alternating row/column attention. It has no projection-tree mechanism. |

ModernNCA's exact mechanism is still informative for the proposed control. It
applies PLR-lite numerical encoding and one-hot categoricals, learns an
embedding, and softmax-weights candidate labels by negative Euclidean distance.
During training it samples a fraction of candidate neighbors; at inference it
uses the full training set. TabArena searches that sampling rate and the neural
embedding over 200 configurations. Its 1368 top row therefore says learned
locality can be competitive, not that tree regions are unnecessary or that a
training-row index fits DarkoFit's deployment contract. See the
[paper](https://arxiv.org/html/2407.03257) and
[TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/modernnca/hpo.py).

TabDPT is the stronger context control. It pretrains a row-based Transformer on
real tables by choosing a target column, constructing classification or
regression pseudo-tasks, and dropping/shuffling columns. Standard v1.1 selects
neighbors in preprocessed and optionally reduced feature space; the tied
TabDPT-Turbo default instead uniformly subsamples context. Both perform forward
passes without dataset-specific gradient fitting. That tie strengthens the
uniform-context control: expensive local retrieval is not necessary to reach
the displayed Elo in this snapshot. Neither row's 1437 Elo should be credited
to a tree. See the standard [paper](https://arxiv.org/html/2410.18164),
[inference source](https://github.com/layer6ai-labs/TabDPT-inference), and
[Turbo paper](https://openreview.net/pdf?id=Y00pwFyrHR).

## Architecture watchlist: neural-tree and tree-inspired models without a current TabArena grade

Three prominent models have a much closer relationship to ordinary trees than
ModernNCA or TabDPT. GRANDE and NODE are differentiable tree architectures;
DOFEN is tree-inspired but does not learn finite leaf routes. None appears in
the pinned leaderboard CSV or the pinned
[TabArena model registry](https://github.com/autogluon/tabarena/tree/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models).
They must not be described as current TabArena leaders.

| Model | Tree relationship | Distinct idea worth retaining | Why it is not promoted here |
|---|---|---|---|
| [GRANDE](https://arxiv.org/html/2309.17130) | Dense hard axis-aligned trees trained end-to-end with straight-through gradients. | A reached leaf produces both a prediction and an instance-specific tree weight; weights are normalized across trees. | No current TabArena row or pinned HPO/pipeline evidence; dense full-tree cost grows exponentially with depth. |
| [NODE](https://arxiv.org/abs/1909.06312) | Differentiable oblivious trees use sparse feature selection and soft branch probabilities; later layers consume raw inputs plus earlier tree outputs. | Data-aware initialization places learned thresholds/scales in active-gradient regions. | No current row; only the symmetric base-tree structure overlaps DarkoFit, while NODE's explicit feature stacking is distinct and carries a broad GPU-memory/proof surface. |
| [DOFEN](https://arxiv.org/abs/2412.16534) | Fixed random permutations group learned per-column sigmoid conditions into relaxed-ODT embeddings; there is no learned `2^depth` leaf route. | Row-specific weighting over condition-group embeddings, sampled into forests and averaged across a two-level ensemble. | No current row, and no finite leaf payload that can be transferred into DarkoFit's existing oblivious-tree representation. |

One GRANDE-derived experiment is sufficiently distinct to record, but not to
queue from this leaderboard review: freeze a base ensemble, transform a
disjoint calibration/OOB slice into that fixed model's `(tree, leaf)`
vocabulary, learn a strongly shrunk log-weight over its tree predictions, and
normalize weights across trees for each row. Fold-local gates may be used for
honest evaluation, but their leaf identities must remain separate; a deployable
refit needs a newly frozen full-data base followed by its own untouched
calibration gate. Cap any member's maximum contribution, regularize toward the
current additive aggregation, monitor weight entropy, and keep the experiment
behind the current ensemble authorization boundary. Without a TabArena result
for GRANDE, this is an architecture hypothesis—not leaderboard-backed
priority.

The watchlist promotion trigger is a verified row on the same pinned TabArena
snapshot—or a documented later refresh—an inspectable model-specific
pipeline/HPO space, and an advantage that
survives comparison with DarkoFit's existing hard oblivious and leaf-wise
builders. Until then, iLTM is the only additional current high-scoring model
that merits a full tree-adjacent dossier.

## Source ledger

### Benchmark and pipeline sources

- [Pinned TabArena leaderboard CSV](https://huggingface.co/spaces/TabArena/leaderboard/blob/832ff68cf9740a64088f6808dc922f6b2d2c8b6c/data/imputation_yes/splits_all/tasks_all/datasets_all/website_leaderboard.csv)
- [TabArena leaderboard methodology](https://huggingface.co/spaces/TabArena/leaderboard/blob/832ff68cf9740a64088f6808dc922f6b2d2c8b6c/website_texts.py)
- [TabArena paper](https://arxiv.org/abs/2506.16791)
- [TabArena reproduction example](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/examples/running_tabarena_models/run_tabarena_model.py)
- [Pinned TabArena model registry](https://github.com/autogluon/tabarena/tree/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models)
- [AutoGluon how it works](https://auto.gluon.ai/stable/tutorials/tabular/how-it-works.html)

### Per-model primary and official sources

- LightGBM: [paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [features](https://lightgbm.readthedocs.io/en/stable/Features.html), [advanced topics](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html), and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/lightgbm/hpo.py)
- CatBoost: [paper](https://arxiv.org/abs/1706.09516), [categorical features](https://catboost.ai/docs/en/features/categorical-features), [parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning), [SGLB controls](https://catboost.ai/docs/en/references/training-parameters/common#langevin), and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/catboost/hpo.py)
- XGBoost: [paper](https://arxiv.org/abs/1603.02754), [categoricals](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html), [monotonic constraints](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html), [interaction constraints](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html), [DART](https://xgboost.readthedocs.io/en/stable/tutorials/dart.html), and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/xgboost/hpo.py)
- ChimeraBoost: [fixed source](https://github.com/bbstats/chimeraboost/tree/6a76586dfdff90275e7e816f25e35c927b8527fb) and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/chimeraboost/hpo.py)
- EBM: [overview](https://interpret.ml/docs/ebm.html), [hyperparameters](https://interpret.ml/docs/hyperparameters.html), [FAST interaction ranking](https://interpret.ml/docs/python/api/measure_interactions.html), [purification](https://interpret.ml/docs/python/api/purify.html), [GA2M paper](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf), and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/ebm/hpo.py)
- ExtraTrees: [paper](https://link.springer.com/article/10.1007/s10994-006-6226-1) and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/extra_trees/hpo.py)
- RandomForest: [paper](https://doi.org/10.1023/A:1010933404324) and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/random_forest/hpo.py)
- PerpetualBooster: [documentation](https://perpetual-ml.github.io/perpetual/), [architecture](https://perpetual-ml.github.io/perpetual/architecture.html), [API](https://perpetual-ml.github.io/perpetual/api.html), and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/perpetual_booster/hpo.py)
- xRFM: [paper v3](https://arxiv.org/html/2508.10053v3), [source v0.4.5](https://github.com/dmbeaglehole/xRFM/tree/v0.4.5), [algorithm notes](https://github.com/dmbeaglehole/xRFM/blob/v0.4.5/ALGORITHM.md), and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/xrfm/hpo.py)
- iLTM: [paper](https://arxiv.org/html/2511.15941), [source at `11c69c7`](https://github.com/AI-sandbox/iLTM/tree/11c69c79701bdfa1dcbf7ca70f9fcfcb2d11b060), [leaf embedding](https://github.com/AI-sandbox/iLTM/blob/11c69c79701bdfa1dcbf7ca70f9fcfcb2d11b060/iltm/tree_embedding.py#L595-L736), [tree/main split](https://github.com/AI-sandbox/iLTM/blob/11c69c79701bdfa1dcbf7ca70f9fcfcb2d11b060/iltm/inference_interface.py#L1316-L1358), [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/iltm/hpo.py), and [TabArena wrapper](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/iltm/model.py)
- GPBoost: [JMLR paper](https://www.jmlr.org/papers/v23/20-322.html), [official source](https://github.com/fabsig/GPBoost), [`GPModel` API](https://gpboost.readthedocs.io/en/stable/pythonapi/gpboost.GPModel.html), [`GPBoostRegressor` API](https://gpboost.readthedocs.io/en/stable/pythonapi/gpboost.GPBoostRegressor.html), [main parameters](https://gpboost.readthedocs.io/en/stable/Main_parameters.html), and [computational-efficiency guidance](https://gpboost.readthedocs.io/en/stable/Computational_efficiency.html)
- Adjacent controls: ModernNCA [paper](https://arxiv.org/html/2407.03257) and [TabArena HPO](https://github.com/autogluon/tabarena/blob/3b2669da335eac08531bcef64c147c99e46cf595/packages/tabarena/src/tabarena/models/modernnca/hpo.py); TabDPT [paper](https://arxiv.org/html/2410.18164), [Turbo paper](https://openreview.net/pdf?id=Y00pwFyrHR), and [inference source](https://github.com/layer6ai-labs/TabDPT-inference); LimiX [paper](https://arxiv.org/html/2509.03505); TabM [paper/source](https://github.com/yandex-research/tabm); RealMLP [paper](https://arxiv.org/abs/2407.04491); and ConTextTab/SAP-RPT-OSS [paper](https://arxiv.org/html/2506.10707) and [source](https://github.com/SAP-samples/sap-rpt-1-oss)
- Ungraded tree-related watchlist: GRANDE [paper](https://arxiv.org/html/2309.17130) and [source](https://github.com/s-marton/GRANDE); NODE [paper](https://arxiv.org/abs/1909.06312) and [source](https://github.com/Qwicen/node); and DOFEN [paper](https://arxiv.org/abs/2412.16534) and [source](https://github.com/Sinopac-Digital-Technology-Division/DOFEN)

### DarkoFit evidence consulted

- `README.md`, `docs/concepts.md`, `docs/parameters.md`,
  `darkofit/auto_params.py`, `darkofit/binning.py`,
  `darkofit/target_encoding.py`, `darkofit/preprocessing.py`,
  `darkofit/tree.py`, `darkofit/booster.py`, `darkofit/sklearn_api.py`, and
  `darkofit/tuning/` for current contracts and implementation overlap.
- The recently landed changes in `darkofit/booster.py`, `darkofit/tree.py`,
  `darkofit/sklearn_api.py`, `darkofit/serialization.py`, and their tests were
  checked so fused-dispatch evidence gates and private-ensemble provenance
  work were not misreported as new modeling mechanisms.
- A targeted search of the estimator, tree, tests, docs, and benchmark runners
  found internal per-tree `apply()` methods but no supported estimator-level
  leaf-address transform, one-hot/hash representation, global downstream leaf
  head, leaf-proximity residual retrieval, or learned neighbor correction.
- The same audit found that `groups=` controls validation and ensemble
  sampling integrity but does not enter the fitted prediction function. Current
  distributional heads are marginal by row; there is no grouped random effect,
  GP residual, covariance-aware gradient/leaf refit, or cross-row predictive
  covariance.
- `benchmarks/gpboost_basketball_v1_protocol.md`,
  `benchmarks/gpboost_basketball_v2_protocol.md`,
  `benchmarks/bench_gpboost_basketball.py`, and the corresponding
  `benchmarks/TESTING_LOG.md` entry define the current feature-only GPBoost
  comparator boundary. The invalid v1 repeatability attempt emitted no result,
  and neither protocol supplies a `GPModel`.
- `docs/archive/DARKOFIT_PRACTICAL_WISHLIST.md` records grouped empirical
  residual shrinkage as a deferred, evidence-dependent follow-up, not a live
  implementation or a negative experimental verdict.
- `benchmarks/t7_catboost_attribution_result.md` and
  `benchmarks/t7b_catboost_gap_attribution_result.md` define the P0 evidence
  boundary.
- `benchmarks/basketball_categorical_combinations_result.md`,
  `benchmarks/basketball_cross_features_donor_screen_result.md`,
  `benchmarks/fresh_selector_confirmation_result.md`,
  `benchmarks/basketball_temperature_scaling_result.md`, the OOB ensemble
  result, and the current testing/plan ledger define routes already closed as
  tested.
- `benchmarks/tabarena_regression_same_machine_result.md` and
  `benchmarks/bench_vs_lightgbm.py` provide comparator and evaluation context.
