# COUNTERPUNCH_PLAN — evidence-led response to ChimeraBoost 0.18

> **Status:** draft reviewed 2026-07-19; not authorized for execution.
> Nothing in this document freezes a protocol, starts a benchmark, changes a
> public API, opens fresh data, or authorizes a default change.

DarkoFit comparison pin: `v0.10.0` at
`ec66a64654becaf948592588a047bfb8205decc8`.
ChimeraBoost comparison pin:
`f14be606b641f1bf0dc92bb14b3951f1fe631c6b`
(`v0.18.0-6-gf14be60`).

Governing rules:
[`benchmarks/SHIPPING_POLICY.md`](benchmarks/SHIPPING_POLICY.md).
The current 0.18 sports characterization is
[`benchmarks/basketball_chimera_v018_diagnostic.md`](benchmarks/basketball_chimera_v018_diagnostic.md).
Every material run must append a 12-field entry to
[`benchmarks/TESTING_LOG.md`](benchmarks/TESTING_LOG.md).
Frozen artifacts are immutable.

## 1. Executive decision

The strategy is sound, but the original proposal advanced several mechanisms
faster than the evidence permits. The correct first move is a measurement and
feasibility gate, not an immediate implementation program.

1. Establish a clean documentation checkpoint and audit the specific
   post-0.18 hygiene concerns without assuming ChimeraBoost bugs exist in
   DarkoFit.
2. Refresh three stale comparisons: large-n engine throughput, the broad
   13-dataset panel, and shipped ensemble behavior.
3. Publish one owner-facing decision after those measurements:
   continue, defer, or close quantization and ensemble-v3 work.
4. Only surviving mechanisms enter opt-in implementation.
5. Any automatic resolver or default change follows the complete Tier-D path
   on prospectively frozen fresh evidence.

Quantized histograms are the leading engine candidate to investigate, not a
predetermined adoption. ChimeraBoost's result does not establish that the same
design will materially help DarkoFit's already-fused histogram lanes.

## 2. Verified evidence and open questions

| Topic | Current evidence | Status |
| --- | --- | --- |
| Basketball single-model quality | On the spent nine-lineage player-disjoint panel, DarkoFit had 2.89% lower primary RMSE and 1.82% lower cold-player RMSE than ChimeraBoost 0.18. | Current but panel-scoped; descriptive Tier-E. |
| Basketball speed | On that panel ChimeraBoost fit 2.77× faster and predicted 1.42× faster. On the creator's overlap-permitting folds, DarkoFit's single-model median wall time was 24.70% lower. | Current but workload-specific. |
| ChimeraBoost ensemble | Its eight-member ensemble reached R² 0.5433 on overlap-permitting creator folds, versus 0.5267 for DarkoFit's single model, at 5.63× DarkoFit's wall time. | Current; player-disjoint value unknown. |
| Quantization on sports | ChimeraBoost's default quantization mildly worsened its player-disjoint sports quality and did not help the one small-workload timing diagnostic. | Current only for this small, noisy workload. |
| Large-n engine comparison | DarkoFit's historical 1.28× fit advantage was measured against an older ChimeraBoost float engine. | Stale; M1 must refresh it. |
| Broad 13-dataset comparison | The historical DarkoFit/ChimeraBoost RMSE ratio was 1.0125 and training-time ratio was 0.9093 against ChimeraBoost 0.14.1. | Historically valid for its pins; stale for 0.18. |
| Prediction throughput | Prior medians were not measured against ChimeraBoost 0.18. | Unknown; do not claim they held. |
| Product breadth and conformal result | ChimeraBoost 0.18 did not directly test these DarkoFit capabilities. | Unchallenged, not re-won or newly improved. |
| Sports early stopping + exact refit | The prior candidate lost 7/10 creator folds, worsened the overlap-exposed holdout, and was only 11.7% faster. | Closed predecessor; any successor needs a distinct causal hypothesis. |
| Panel 3 | Power was 50.00% for T5 and 10.64% for guarded cross features versus the required 80%; no candidate was retained and the lockbox stayed sealed. | Closed and immutable. A new guard is a new campaign. |
| CatBoost quality gap | CatBoost remains ahead on the relevant broad-tabular and sports records. | Open research problem. |

The current evidence does **not** support a blanket statement that everything
else held or improved against 0.18. Only the spent basketball diagnostic has
been refreshed.

## 3. Decision ledger

| Track | State | Evidence class | Entry condition | Exit artifact | Stop rule |
| --- | --- | --- | --- | --- | --- |
| H — hygiene | Ready for audit after owner authorization | Tier-E engineering | Clean documentation checkpoint | Audit chips and, only for confirmed gaps, tested fixes | Close each item as fixed or not present |
| M — measurements | Ready after owner authorization | Tier-E descriptive | Exact pins, new dated protocols, exclusive machine access | M1, M2, and M3a results plus testing-log entries | Publish once; no rerun to improve a result |
| Q — quantization | Conditional | Q0 engineering, Q1/Q2 Tier-E opt-in, Q3 Tier-D automatic | M1 shows a material opportunity and Q0 shows a credible DarkoFit hotspot | Opt-in result or a documented closure; separate default-policy result if nominated | Close if the profile or prototype cannot plausibly meet the declared speed budget |
| B — ensemble v3 | Conditional | Tier-E explicit opt-in; Tier-D for any changed default | M3a shows useful quality/cost room and B0 produces a compatible API | Explicit v3 mode and M3b attribution, or closure | Preserve current behavior if no robust Pareto improvement |
| X — cross features | Conditional | Tier-E explicit research opt-in; Tier-D for automatic engagement | X0 defines separate force and guarded semantics plus full product obligations | Narrow opt-in result or deferral | No general safety claim from the three-task spent result |
| S — sports speed | Conditional successor to a failed candidate | Tier-D if automatic | One justified, preregistered group-safe candidate | Spent screen, then fresh result only if powered | Close automatic route on a failed spent screen |
| P — harm-bounded composite | New campaign only | Tier-D | New exact candidate and new protocol identity | Published power GO/NO-GO; fresh result only after GO and owner authorization | No fresh access below the preregistered 80% power bar |
| C — CatBoost gap | Research backlog | Development screen, then Tier-D if automatic | Higher-priority M/Q/B work no longer blocks it | Hypothesis result with bounded claim | Do not tune repeatedly on the same spent outcomes |
| Z — 1.0 cleanup | Conditional | Engineering/API | Surviving surfaces stable and deprecation inventory approved | Compatibility report, removal diff, release notes | No deletion for a line-count target |

## 4. Non-goals

- No CTR23 or other sealed lockbox access without a new, published,
  power-qualified authorization and explicit owner approval.
- No sports or noisy-data default justified by overlap-permitting folds.
- No automatic policy from spent evidence alone.
- No multiclass cross-feature program in this cycle.
- No new grow-kernel program unless profiling contradicts the current
  approximately 1% opportunity estimate.
- No silent reinterpretation of existing ensemble behavior.
- No open-ended hyperparameter sweep on previously inspected outcomes.
- No PyPI work. The owner has chosen a GitHub-only distribution posture unless
  that decision is explicitly reversed later.

---

## 5. Gate M — refresh the stale measurements

These are new dated characterizations that reuse declared coordinates. They
do not amend or masquerade as reruns of frozen protocols under changed source.
All timed campaigns run exclusively on the machine; M1, M2, and M3a may not
overlap each other or unrelated heavy work.

### M0 — documentation checkpoint

After owner authorization, land the reviewed documentation already pending in
the worktree, including this plan, the testing log, the 0.18 diagnostic, and
their pointer updates. Record the exact clean source pins used by subsequent
protocols. This is a provenance step, not feature work.

### M1 — large-n matched engine characterization

The primary estimand is matched-core engine throughput, not product-default
performance. Freeze one common configuration matching the historical
large-n comparison and compare:

- DarkoFit 0.10 under the common configuration;
- ChimeraBoost 0.18 quantized under the same configuration; and
- ChimeraBoost 0.18 with `quantize_gradients=False`.

Reuse the explicit workload and matched capacity in
[`benchmarks/large_n_engine_protocol.md`](benchmarks/large_n_engine_protocol.md):
24 numeric features, 500k/1M training rows, a following 100k-row holdout, 300
trees, learning rate 0.1, depth 6, L2 1, 128 bins, full rows/features,
minimum child weight 1, ordered boosting and early stopping off, random state
4, and 18 threads. Preserve the protocol's product-specific controls,
including `tree_mode="catboost"`/minimum child samples 1 for DarkoFit and
disabled product selectors for ChimeraBoost.

Run six balanced order blocks using all permutations of the three primary
arms at each size. Every fresh timed worker performs its own same-arm 5k-row,
three-round JIT/cache warmup before the measured fit. Optional product-default
arms may be reported separately but may not be used for matched-core
attribution. Report:

- total fit wall time and phase-level timing where both products expose it;
- prediction wall time on a declared batch;
- peak RSS, archive size, CPU/thread policy, and resolved model metadata;
- paired median ratios and the full repeat series; and
- quality and prediction fingerprints sufficient to detect accidental
  configuration drift.

M1 answers whether the historical large-n crown survived and whether
ChimeraBoost's quantized lane, rather than unrelated 0.18 changes, explains
the movement. It does not prove that DarkoFit should implement quantization.

### M2 — current-version broad characterization

Create a new dated protocol that reuses the exact 13 datasets and
`r0f0/r1f1/r2f2` coordinates from the historical panel while preserving all
old artifacts. Primary arms:

- DarkoFit 0.10 default;
- ChimeraBoost 0.18 default; and
- CatBoost 1.2.10 default.

Add ChimeraBoost 0.18 float as a clearly labeled diagnostic arm for
quantization attribution. Keep shared splits, preprocessing boundaries,
thread budgets, early-stopping rules, fresh-worker execution, equal-dataset
aggregation, full per-coordinate rows, fit/predict time, and RSS. Publish
current-version quality and cost; do not use this spent panel to authorize a
new default.

### M3a — shipped ensemble comparison

Use only public behavior already present at the pinned sources.

Primary arms:

- DarkoFit single and group-bootstrap `n_ensembles=8` with exact player
  groups supplied; and
- ChimeraBoost 0.18 single default and its current eight-member ensemble.

Diagnostic arms:

- DarkoFit row-bootstrap `n_ensembles=5/8` and group-bootstrap
  `n_ensembles=5`; and
- ChimeraBoost float single and, if supported by the public surface, a float
  eight-member ensemble.

Primary scoring is the spent player-disjoint sports panel, including
held-team and cold-player views. Creator folds are a secondary,
overlap-permitting diagnostic. Report the quality/cost Pareto: fit, predict,
peak RSS, model bytes, and available OOB/member telemetry.

Label DarkoFit's shipped row-bootstrap member selection as player-overlap
exposed and its group-bootstrap selection as player-disjoint. Apply the same
disclosure to ChimeraBoost's row-subagged member/OOB policy. A player-disjoint
outer test fold does not make an overlap-exposed inner OOB selector
group-safe.

For the four primary arms, use at least three fresh-worker repeats in rotating
order. Each worker performs a same-arm warmup before timing; report median and
full-series fit/predict wall time plus peak RSS and model bytes. Diagnostic-arm
costs may be single warmed runs only if they are labeled descriptive and
excluded from timing decisions.

The proposed DarkoFit 0.8 subagging and member-tuned policy do **not** belong
in M3a because they do not yet exist. They move to B and M3b.

### G-M — owner-facing decision

Publish one short decision after M1–M3a:

- **Q0 fund** only if M1/M2 show a material current engine opportunity;
  otherwise defer or close Q. Q0—not G-M—later decides whether Q1 is
  justified.
- **B continue** only if ensemble quality survives the player-disjoint view
  at an acceptable quality/cost position; otherwise preserve the current
  opt-in without a v3 program.
- M3a may inform a documented recipe. It cannot change an ensemble default.

---

## 6. Gate F — feasibility before public implementation

### Q0 — quantization attribution and prototype

Profile DarkoFit's current scalar training path at the M1 workload shapes.
Measure the fraction attributable to gradient/Hessian preparation, histogram
construction, sibling subtraction, split search, leaf values, and
preprocessing. Build only the smallest private scalar prototype needed to
test the projected benefit.

Advance to Q1 only if:

- histogram bandwidth is a material current bottleneck;
- the prototype shows a repeatable gain large enough to justify the added
  arithmetic and maintenance surface; and
- a written arithmetic design can prove safe accumulation and fallback.

If the attainable upper bound cannot meet the subsequently declared speed
budget, record the profile and close Q. Do not implement a public option to
match a competitor's architecture when DarkoFit's bottleneck is elsewhere.

### B0 — ensemble compatibility and sampling design

Specify a new explicit ensemble-v3 mode without changing the current
full-size bootstrap semantics. The design must cover:

- row subsampling without replacement for ordinary tabular data;
- group subsampling without replacement for entity/player workloads, with
  group-disjoint OOB selection;
- a named member policy rather than overloading global parameters whose
  existing defaults cannot distinguish omitted from explicit values;
- sequential versus parallel semantics under a fixed total CPU budget; and
- fitted metadata, failure propagation, memory, serialization, and model-size
  obligations.

Row-level OOB is not a valid sports selector when the same player can occur in
both member training and OOB rows.

### X0 — cross-feature product contract

The prior T6b mechanism exists in benchmark/research code, not as a completed
DarkoFit public surface. Before implementation, define distinct semantics for:

- **force:** construct and use the declared cross features; and
- **guarded:** fit/evaluate the candidate and decline under a declared
  validation rule.

Do not make `cross_features=True` ambiguously mean both. The contract must
cover train-only ranking, validation-only engagement, exact decline behavior,
groups and weights, categorical and missing inputs, refit/early stopping,
prediction-time reconstruction, serialization, feature names, SHAP/importance
semantics, added columns, memory, and the cost of the extra fit.

The existing 5% guard was selected post hoc on three spent smooth datasets.
It is development evidence, not proof of general no-harm behavior.

---

## 7. Track Q — quantized-gradient histograms

Q is conditional on G-M and Q0.

### Q1 — explicit scalar opt-in

Start with the unweighted, constant-Hessian scalar RMSE lane
(`sample_weight is None`). Weighted fits remain on a recorded float fallback
until Q2. Preserve the current engine as the default and as an explicit
override. The final implementation may use packed integer histograms and
stochastic rounding, but Q0—not this plan—must choose the exact
representation.

Required invariants and tests:

- explicit float is byte-identical to the current engine;
- fixed-seed, fixed-thread runs repeat exactly; the rounding stream itself is
  independent of work scheduling;
- any counter key includes a stream/lane identifier in addition to seed,
  tree, and row identity;
- signed-gradient and nonnegative-Hessian encodings cannot overflow or carry
  into one another, proved for the declared row and quantized-value bounds;
- zero scales, degenerate nodes, split ties, and `min_child_weight`
  boundaries have defined behavior;
- weighted and extreme-weight cases exercise the recorded float fallback
  rather than packed accumulation;
- unsafe or unsupported workloads fail loudly or use a recorded float
  fallback—never silent arithmetic corruption;
- packed kernels match a slow unpacked-integer oracle;
- leaf values are computed from unquantized gradients unless a separate
  tested design supersedes this;
- cloning, fitted metadata, serialization, warmup, and public disclosure all
  preserve the selected lane and reason.

Do not promise full-model identity across different thread counts unless it is
actually proven; the baseline contract is exact repeatability at a fixed
supported configuration.

Attribution must reflect the implementation source. Call it an independent
implementation of a published technique only if no ChimeraBoost code is
adapted; otherwise preserve the applicable license and NOTICE history.

Before Q1 ships, publish a dated spent-data characterization of quality and
cost with uncertainty: aggregate and per-dataset/split effects, worst observed
harm, fit/predict/RSS dispersion, resolver/fallback counts, and the exact
supported scope. This is honest Tier-E disclosure for a deliberately
non-behavior-exact opt-in, not a retrofitted binary certification gate.

### Q2 — additional explicit lanes

Only after Q1 has a useful Pareto result, consider variable-Hessian binary and
weighted scalar lanes, then selected-feature fused lanes. Distributional and
multi-output heads remain outside the first version. Each new lane needs its
own arithmetic bounds, fallback contract, and measured result; Q1 does not
authorize them automatically.

### Q3 — narrowly scoped automatic resolver

Any size/workload-gated automatic policy is a Tier-D default change. Row count
alone is unlikely to be a sufficient resolver; the crossover may depend on
features, bins, depth, thread count, and selected-feature fraction.

Spent data may choose the workload descriptor, crossover threshold, candidate
scope, proposed numerical gates, and power assumptions. Before fresh access,
freeze:

- the exact eligible lane, initially no broader than scalar RMSE;
- the resolver and explicit override;
- exact no-engagement equivalence below the threshold;
- the point aggregate and bootstrap upper-bound rule;
- leave-one-dataset-out concentration;
- the per-dataset or selector-based harm route;
- fit, predict, peak-RSS, and model-size budgets;
- a design-time power analysis; and
- a no-rerun rule.

The binding quality and cost gates must be evaluated on a prospectively frozen
fresh panel. A single informal fresh check is insufficient. Candidate values
such as a 1.002 aggregate upper bound, 1.01 worst-dataset bound, and 15% speed
gain must be justified by the claim and powered design before they become
binding; otherwise use the canonical shipping-policy bars. If Q3 fails, a
successful Q1/Q2 opt-in may remain.

---

## 8. Track B — explicit ensemble-v3 mode

B is conditional on M3a and B0. It adds capability without silently changing
the meaning of existing `n_ensembles` configurations.

1. **B1 — sampling:** add explicit row and group subsampling without
   replacement, with a declared fraction. Preserve the existing bootstrap
   mode.
2. **B2 — member policy:** add a named, explicit policy for member-level
   learning rate, column sampling, and related automatic choices. Explicit
   user parameters win, and all resolutions are persisted.
3. **B3 — parallel members:** add `ensemble_n_jobs` only with deterministic
   member seeds, nested-parallelism prevention, equal-total-CPU timing,
   worker-failure propagation, and peak-memory accounting. Test model
   equivalence at the same per-member thread count separately from throughput
   under divided CPU budgets.
4. **M3b — attribution:** compare sampling, member policy, and parallelism in
   separable arms on the spent player-disjoint panel and a small broad-tabular
   development set. Include fit, predict, RSS, archive bytes, and OOB
   telemetry.

M3b may nominate a documented opt-in recipe or explicit v3 preset. A general
default change requires a separate Tier-D campaign across numeric,
categorical, classification, weighted, and relevant grouped workloads. The
T10 sports automatic-ensemble refutation remains in force.

---

## 9. Track X — cross-feature research opt-ins

After X0, a force-on research surface may ship under Tier-E once its product
correctness obligations are met. A guarded surface may also be offered
explicitly, but its documentation must say:

- the 5% rule was selected from three spent smooth datasets;
- the observed exact declines and no-harm record are bounded to those
  coordinates;
- the guarded Panel 3 candidate had only 10.64% simulated pass probability;
  and
- neither mode is a generally validated automatic default.

Persist selected pairs, added-column count, validation metrics, engagement
reason, memory/cost telemetry, and all information required to reconstruct
prediction exactly. Do not claim that this generally closes ChimeraBoost's
smooth-data moat. Automatic composite engagement belongs only in a new Track
P campaign.

---

## 10. Track S — a genuinely new sports-speed candidate

The prior current-auto-LR + early-stopping + exact-refit candidate is closed:
it lost 7/10 creator folds, worsened the overlap-exposed holdout, and delivered
only an 11.7% wall-time improvement. Exact refit can erase the apparent
selection-stage saving.

A successor is justified only by the new causal hypothesis that group-safe
inner validation plus a specifically justified auto-LR floor changes that
failure mode. Do not run a generic patience sweep on the same spent folds.
Predeclare one candidate, or a very small finite development set with
separate held-back spent development-holdout coordinates.

The spent screen must include:

- candidate/control RMSE ratio on player-disjoint primary, held-team, and
  cold-player views;
- aggregate uncertainty, leave-one-lineage-out concentration, and a declared
  harm rule;
- total fit time including selection and exact refit;
- selected LR, selected iteration, stop reason, group hashes, and final tree
  count; and
- a scope limited to calls with valid groups.

Use directionally explicit gates; for example, candidate/control RMSE
`<= 1.002`, with separate guardrail harm bounds, rather than “quality
`>= 0.998×`.” A speed target such as `>= 1.5×` must include all selection and
refit work.

If the spent screen fails, close the automatic route. A measured opt-in recipe
may remain with its exact tradeoff, but it must not be called sports-safe. If
the screen passes, freeze and power genuinely unseen seasons or lineages
before any default consideration.

---

## 11. Track P — new harm-bounded composite campaign

Panel 3 is closed. Its exact candidates, power transport, registry, and
decision artifacts remain immutable. A new harm-bounding guard creates a new
candidate and therefore requires a new campaign identity (provisionally
`P-next`, not “Panel 3 resumed”).

Panel 3 used 39 spent coordinates across 13 tasks and 117 model jobs across
three arms. Those outcomes may inform development, but they do not
automatically instantiate or authorize a newly selected guard. If they help
select the guard, account for that selection when constructing a separate
power calibration.

Sequence:

1. Use explicitly declared spent development evidence to design the exact
   per-dataset engagement/decline mechanism.
2. Before executing or inspecting the exact-policy calibration outputs,
   freeze the candidate, calibration protocol, transport/exchangeability
   assumptions, gates, simulation method, power-retention rule, exposure
   catalog, contamination review, registry rules, harm route, and no-rerun
   rule.
3. Run or reconstruct the exact policy on the declared calibration evidence,
   preserving per-coordinate outputs. The calibration evidence must be
   disjoint from guard selection, or the frozen simulation must explicitly
   model the selection step.
4. Publish the plain-language simulated pass probability and one-sided Wilson
   lower bound against the required 80%.
5. On NO-GO, stop with no fresh access.
6. On GO, re-attest that the proposed sealed lineages remain eligible under
   the new campaign. Only then may the owner authorize a one-shot fresh run.

The old lockbox remains physically sealed, but its Panel 3 eligibility and
allowlist do not automatically transfer to P-next.

---

## 12. Track C — CatBoost-gap research

The T7b observations are hypotheses for DarkoFit, not demonstrated DarkoFit
wins:

- `l2_leaf_reg=1` was promising in the attribution work; and
- a samples-per-feature depth policy improved CatBoost itself.

C1 and C2 are development screens on declared spent evidence. If the same
outcomes shape a policy, do not call a later view of those tasks
outcome-unseen. Reserve separate spent development-holdout coordinates within
the screen where possible. Any automatic DarkoFit policy that survives still
needs a fresh Tier-D confirmation campaign. Keep this track behind the M/Q/B
decision unless new evidence changes the priority.

---

## 13. Track H — hygiene and documentation

### H0 — clean documentation checkpoint

When authorized, commit the reviewed pending documentation as a standalone
checkpoint before benchmark protocols bind new source hashes. Do not mix it
with feature implementation.

### H1 — audit, then fix only confirmed gaps

ChimeraBoost's audit findings are prompts, not evidence that DarkoFit has the
same bugs:

- **Thread state:** test call-local thread-count restoration at fit and
  predict, including subprocess concurrency boundaries. Numba thread masks
  are thread-local; do not describe an unverified process-global leak.
  Preserve the existing decision to reject unsafe background warmup.
- **Serialization:** inspect whether saved models redundantly include
  rebuildable predictor caches; retain safe, bit-identical round trips.
- **Loud failures and parameter resolution:** cover unseen classifier
  `eval_set` labels, positional `sample_weight` misuse around categorical
  arguments, NumPy integer `cat_features`, and documented handling of
  `None`-valued constructor parameters.

For each item, publish one of: confirmed and fixed with a named regression
test; not present with a reproducer; or intentionally different with a
compatibility note. Do not rewrite already-correct input-hardening or warmup
behavior.

### H2 — measurement documentation

M1 and M2 add new dated records to the measurements and testing-log surfaces.
Older records keep their source/version boundaries and are never overwritten.

---

## 14. Track Z — conditional 1.0 cleanup

Do not use an arbitrary 12–15k-line target. Start from the deprecations already
announced in `CHANGELOG.md`, build an API/serialization compatibility
inventory, and measure the actual removal. Delete code because an approved
deprecation matured and coverage proves the replacement—not to hit a line
count.

Quantization and ensemble v3 are not prerequisites for 1.0 and must not be
rushed into the release to “break once.” They ship only if their own tracks
pass; otherwise 1.0 may proceed without them or wait for an explicit product
decision.

Distribution remains GitHub-only. The eventual marketing pass must keep claims
bounded:

- one current panel-scoped player-disjoint sports advantage, with
  ChimeraBoost's speed advantage and CatBoost's quality gap adjacent;
- the conformal result scoped to its marginal-coverage and width protocol;
- version-bound engine and prediction measurements; and
- opt-in product breadth without converting development findings into default
  claims.

---

## 15. Execution dependencies and rough effort

These are planning ranges, not delivery promises.

| Wave | Work | Approximate effort | Decision |
| --- | --- | --- | --- |
| 0 | Owner reviews this plan; H0 documentation checkpoint | Hours | Authorize or revise Gate M |
| 1 | H1 audit; M1, M2, M3a run sequentially on an otherwise idle machine | Roughly 1–3 machine-days plus analysis | Publish G-M continue/defer/close |
| 2 | Q0, B0, X0 feasibility/contracts | Roughly 2–5 engineering days | Approve only justified prototypes |
| 3 | Surviving Q1, B1–B3/M3b, and X implementation | Roughly 1–3 weeks depending on survivors | Ship explicit opt-ins, defer, or close |
| 4 | Q3, S, or P-next Tier-D campaigns | Separately estimated and power-gated | One fresh campaign per qualified automatic policy |
| Later | C research and Z cleanup | Independent backlog | No coupling to unfinished speculative tracks |

Benchmark waves use fresh workers and exclusive machine access. Parallelize
code review and analysis, not timed model jobs that would contend for CPU,
memory bandwidth, or cache.

## 16. Evidence and stopping discipline

- New source pins require new dated characterization artifacts; do not call a
  changed-source run a frozen-protocol rerun.
- Spent data may rank, debug, size, and power a candidate. It cannot confirm a
  new automatic policy.
- Tier-D claims freeze the candidate, point estimand, bootstrap rule,
  concentration check, harm route, cost budget, power calculation, and
  no-rerun rule before fresh access.
- Exact declines and selector no-engagement must be tested against the current
  control, not inferred.
- Timing claims use complete warmup separation, fresh workers, equal resource
  budgets, full repeat series, and paired ratios.
- Every material run has one immutable raw artifact, one analyzer, one result,
  and one testing-log entry.
- A failed gate is recorded and closed. It is not relaxed or retried on the
  same evidence.

The program reaches a clean stopping point when M1–M3a have a published
current-version answer; every downstream track is explicitly shipped,
deferred, or closed; no default claim rests on spent evidence; the testing log
and public claim surfaces agree; and any unused lockbox remains sealed.
