# COUNTERPUNCH_PLAN — evidence-led response to ChimeraBoost 0.18

> **Status:** draft reviewed 2026-07-19; reframed 2026-07-20 to the
> strongest-library goal; revised 2026-07-20 to mechanism-led sequencing;
> Wave 1 authorized by the owner 2026-07-20 and completed 2026-07-20;
> G-M published 2026-07-20; Wave 2 completed and closed 2026-07-20;
> Wave 3 B-archive feasibility completed and closed 2026-07-21;
> Wave 4 fused-lane-dispatch design contract frozen 2026-07-21.
> Wave 1 authorization covers the H1 audit and confirmed fixes, M1 plus the
> Q0 profiling half, M3a, and construction of M5/M6 infrastructure. It does
> not authorize a public prototype, default change, fresh confirmation
> access, or sealed-lockbox access.
> G-M closed Q and the shipped DarkoFit ensemble route, then funded only B0
> plus one private sequential B1/B2 mechanism-attribution prototype. M3b
> closed that prototype after no arm cleared every frozen gate. B3
> parallelism, public surfaces, default changes, M2, and fresh access remain
> unauthorized. B-archive v2 then showed that exact canonical-preprocessor
> factoring reaches `4.152525×`, missing the unchanged `4.0×` archive gate;
> no serializer prototype is authorized, and behavior-exact fused-lane
> dispatch is the next nominated Track I mechanism.

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

## 0. Purpose and framing

DarkoFit's north star is a mechanism-led, general-purpose fitting library —
optimizing the accuracy–speed–memory–reliability Pareto frontier — with
sports as the home workload, not the design target and not a leaderboard
entry. ChimeraBoost is the friendly rival and primary source of competitive
ideas; CatBoost, LightGBM, and others are additional idea donors and
ceilings. CatBoost remains the quality ceiling on record — it leads DarkoFit
on the broad 13-dataset record and dominates it on the sports panel — but
the CatBoost gap is a backlog of candidate mechanisms, not a campaign or a
strategic center (owner decision 2026-07-20).

Four standing constraints follow:

1. **No sports overfit, and no sports-specific code.** Sports evidence
   alone cannot justify a default. Defaults need broad-panel plus
   sports-panel joint evidence under
   [`benchmarks/SHIPPING_POLICY.md`](benchmarks/SHIPPING_POLICY.md), and
   sports-only wins ship as explicit opt-ins. Candidate generation must not
   be exclusively sports-sourced. Sports needs enter the library only as
   reusable generic abstractions — `groups`, group-safe validation and OOB,
   weights, deterministic sampling — never as basketball-specific branches
   or defaults.
2. **TabArena is a thermometer, not a target — ChimeraBoost is the
   calibrated yardstick.** ChimeraBoost maintains a strong published
   TabArena-Lite position at pinned versions, and its own sealed-holdout
   discipline keeps that position honest. The cheap standing test of
   general quality is therefore beating a pinned ChimeraBoost on the
   internal broad panel at its milestone cadence (M2); the rival's TabArena
   spend calibrates the yardstick at no cost to this repo. Routine breadth
   protection between M2 milestones comes from the M5 diversity sentinels
   plus an eligible M6 successor or mechanism-specific spent development
   evidence, not from TabArena or the broad panel. M6 v3 is terminal and
   cannot fill that role. TabArena-Lite itself runs at most once per minor
   release as a descriptive drift check (M4) that validates this proxy rather
   than replacing it; no benchmaxing, and the CTR23 lockbox discipline is
   unchanged. The proxy only covers what the internal panels cover: today
   that is regression-weighted evidence, and classification — where
   ChimeraBoost earned much of its TabArena position — remains outside the
   broad proxy until a comparative classification slice exists. M5 plus an
   eligible successor comparison reduce that risk between milestones but do
   not erase it.
3. **Absorption with bounded complexity.** Ideas adopted from other
   libraries enter through the normal Tier-E/Tier-D machinery. An absorbed
   surface should displace or consolidate existing code where possible; a
   genuinely net-new capability may instead carry an explicit complexity
   budget, maintenance owner, and review date (see Track I and Track Z).
4. **One mechanism at a time.** The unit of development is a mechanism
   moving through a fixed pipeline: profile → smallest private prototype →
   correctness invariants → M5 diversity sentinels → an eligible M6
   successor or mechanism-specific spent development evidence → sports panel
   → M2 broad checkpoint only if it survives. Mechanism-specific synthetic
   tests and profilers establish causal behavior; M5 detects drift; only a
   backtest-qualified successor or a prospectively frozen mechanism contract
   may rank or kill development candidates. Sports and M2 characterize
   progressively broader product value. Tier-D rigor is unchanged for
   defaults; explicit Tier-E capabilities may move faster once correct and
   honestly characterized.

## 1. Executive decision

The strategy is sound, but the original proposal advanced several mechanisms
faster than the evidence permits. The correct first move is a measurement and
feasibility gate, not an immediate implementation program.

1. Establish a clean documentation checkpoint and audit the specific
   post-0.18 hygiene concerns without assuming ChimeraBoost bugs exist in
   DarkoFit.
2. Refresh the stale comparisons that gate funding decisions: large-n
   engine throughput (M1, with the cheap Q0 profile alongside it) and
   shipped ensemble behavior (M3a). The broad 13-dataset panel (M2) is a
   periodic milestone — run it after a mechanism survives the cheaper
   stages or before a meaningful release, never as an upfront tollbooth.
3. Publish one owner-facing decision after M1, the Q0 profile, and M3a:
   fund the quantization prototype, ensemble v3, neither, or both in
   sequence — whichever shows the more convincing general Pareto gain.
   CatBoost-gap and cross-feature mechanisms compete through the Track I
   backlog rather than as pre-specified campaigns.
4. Only surviving mechanisms enter opt-in implementation.
5. Any automatic resolver or default change follows the complete Tier-D path
   on prospectively frozen fresh evidence.

Before Wave 1, quantized histograms were the leading engine candidate to
investigate, not a predetermined adoption. Wave 1 then closed that funding
route under its conjunctive rule; ChimeraBoost's result did not establish
enough material donor value for DarkoFit's already-fused histogram lanes.

**Wave 1 outcome (2026-07-20):** Q0 found a plausible DarkoFit hotspot, but
M1 missed Q's conjunctive material-donor threshold, so Q is closed. Every
shipped DarkoFit ensemble lost M3a quality, so that implementation route is
also closed and repeat timing was forbidden. The predeclared ChimeraBoost
ensemble8 arm, however, improved all nine player-disjoint sports cells and
all six selected general cells by roughly 5% in aggregate. G-M therefore
funds one different private B0/B1/B2 mechanism prototype: isolate
without-replacement sampling and a named member policy, sequentially first.
This is not authorization to retune the failed bootstrap arms or to ship an
ensemble surface. The binding record is
[`benchmarks/wave1_gm_decision.md`](benchmarks/wave1_gm_decision.md).

**Wave 2 outcome (2026-07-20):** B0 and the private sequential B1/B2
prototype completed, including row- and group-safe without-replacement
sampling, named member-policy resolution with explicit-user precedence,
OOB metadata, failure behavior, and safe serialization. In the frozen M3b
attribution all three candidates earned timing, but none cleared every final
gate. The combined arm improved aggregate loss to `0.979638` and fit time to
`0.557873` of the frozen control while keeping peak RSS at `1.069201` of a
single model; its archive remained `5.534767` times a single model against
the frozen `4.0` ceiling. The binding disposition therefore closes B1/B2,
preserves the existing opt-in, and authorizes no B3, public/default change,
fresh confirmation, TabArena, or lockbox access. See
[`benchmarks/m3b_ensemble_v3_r3_result.md`](benchmarks/m3b_ensemble_v3_r3_result.md).

**Wave 3 outcome (2026-07-21):** the dated matched-single readout established
that the M3b combined arm beat the matched single on all 13 spent cases while
failing the archive gate. B-archive then tested only complete, byte-identical
numeric target-free preprocessing sections across the same representative
portfolio. Attempt 1 terminal-failed before publishing a row because its
harness incorrectly required optional `feature_names_in` for NumPy inputs;
attempt 2 preserved that terminal lineage under a new identity and changed no
case, source, runtime, model arm, aggregation, or threshold. All 13 v2 rows
passed the frozen round-trip and component invariants, but median effective
archive/single was `4.152525` against `<= 4.0` (`6.032405` current). B-archive
therefore closes with no serializer implementation or retention authorization.
The binding machine-readable result is
[`benchmarks/barchive_v2_result.json`](benchmarks/barchive_v2_result.json);
the create-only Markdown note's inherited v1 heading is corrected without
rewriting it by
[`benchmarks/barchive_v2_result_heading_erratum_20260721.md`](benchmarks/barchive_v2_result_heading_erratum_20260721.md).

## 2. Verified evidence and open questions

| Topic | Current evidence | Status |
| --- | --- | --- |
| Basketball single-model quality | On the spent nine-lineage player-disjoint panel, DarkoFit had 2.89% lower primary RMSE and 1.82% lower cold-player RMSE than ChimeraBoost 0.18. | Current but panel-scoped; descriptive Tier-E. |
| Basketball speed | On that panel ChimeraBoost fit 2.77× faster and predicted 1.42× faster. On the creator's overlap-permitting folds, DarkoFit's single-model median wall time was 24.70% lower. | Current but workload-specific. |
| ChimeraBoost ensemble | M3a ensemble8/single was `0.950230` on player-disjoint sports RMSE with 9/9 cell wins, `0.977935` on cold players, and `0.947797` on six selected general medium cells with 6/6 wins. | Current Tier-E donor signal at `f14be60`; three spent sports seasons and a small general slice, not a default claim. |
| DarkoFit ensembles | M3a group8/single was `1.025482` on player-disjoint sports RMSE; row5, row8, and group5 were also worse. Row8/single was `1.019556` on the selected general slice. | Current Tier-E closure of the shipped bootstrap/member policy; no repeat timing. |
| Private ensemble-v3 attribution | M3b combined/control was `0.979638` for aggregate loss and `0.557873` for fit time, but combined/single archive bytes were `5.534767` against the frozen `4.0` ceiling. B1 and B2 also failed their value and archive/single checks. | Current spent private evidence; B1/B2 closed with no retained arm and no shipping authorization. |
| B-archive exact-factoring feasibility | Across the same 13-case portfolio, current combined/single archive bytes had median `6.032405`; exact canonical numeric-preprocessor factoring produced an effective median `4.152525` against the unchanged `4.0` limit. Eleven numeric cases were eligible; two member-local categorical cases remained unchanged. | Current spent Tier-E size evidence; B-archive closed, no serializer prototype authorized, fused-lane dispatch nominated next. |
| Quantization on sports | ChimeraBoost's default quantization mildly worsened its player-disjoint sports quality. M3a float/quantized ratios were `1.001949` for singles and `1.002315` for ensembles. | Current only for this small, noisy workload. |
| Large-n engine comparison | M1 DarkoFit/current-quantized-Chimera fit ratio was `0.844722`; Chimera quantized/float was `0.903595`, missing the frozen `0.90` donor bar. | Current Tier-E matched-capacity result; Q closed under its conjunctive rule. |
| Broad 13-dataset comparison | The historical DarkoFit/ChimeraBoost RMSE ratio was 1.0125 and training-time ratio was 0.9093 against ChimeraBoost 0.14.1. | Historically valid for its pins; stale for 0.18. |
| Prediction throughput | M3a supplies single descriptive sports-workload costs; no repeat series was allowed after DarkoFit group8 failed quality. | Current but not timing-decision eligible. |
| Product breadth and conformal result | ChimeraBoost 0.18 did not directly test these DarkoFit capabilities. | Unchallenged, not re-won or newly improved. |
| Sports early stopping + exact refit | The prior candidate lost 7/10 creator folds, worsened the overlap-exposed holdout, and was only 11.7% faster. | Closed predecessor; any successor needs a distinct causal hypothesis. |
| Panel 3 | Power was 50.00% for T5 and 10.64% for guarded cross features versus the required 80%; no candidate was retained and the lockbox stayed sealed. | Closed and immutable. A new guard is a new campaign. |
| CatBoost quality gap | CatBoost remains ahead on the relevant broad-tabular and sports records. | Quality ceiling on record; mechanisms feed the Track I backlog. |

The current evidence does **not** support a blanket statement that everything
else held or improved against 0.18. Only the spent basketball diagnostic has
been refreshed.

## 3. Decision ledger

| Track | State | Evidence class | Entry condition | Exit artifact | Stop rule |
| --- | --- | --- | --- | --- | --- |
| H — hygiene | Complete for Wave 1 | Tier-E engineering | Clean documentation checkpoint | Published H1 audit and tested fixes | Reopen only for a newly confirmed gap |
| M — measurements | Wave 1 complete | Tier-E descriptive | Exact pins, new dated protocols, exclusive machine access | M1, Q0, M3a, M5, M6, testing-log entries, and G-M published; M2 remains milestone cadence | Published once; no rerun to improve a result |
| Q — quantization | Closed at G-M | Q0 engineering, Q1/Q2 Tier-E opt-in, Q3 Tier-D automatic | Re-entry requires a new material donor result or distinct DarkoFit-specific causal case | [`benchmarks/wave1_gm_decision.md`](benchmarks/wave1_gm_decision.md) | Do not relax M1's `0.90` near miss |
| B — ensemble v3 | Closed at B-archive v2 | Spent private B0/B1/B2 attribution plus Tier-E exact-factoring feasibility; Tier-D for any changed default | Re-entry requires a distinct mechanism and new contract identity | [`benchmarks/barchive_v2_result.json`](benchmarks/barchive_v2_result.json) | Exact canonical factoring still missed `4.0×`; preserve existing opt-in, implement no serializer, and do not continue B3 |
| X — cross features | Rolling backlog via Track I | Tier-E explicit research opt-in; Tier-D for automatic engagement | Promotion from the backlog; X0 then defines separate force and guarded semantics plus full product obligations | Narrow opt-in result or deferral | No general safety claim from the three-task spent result |
| S — sports speed | Conditional successor to a failed candidate | Tier-D if automatic | One justified, preregistered group-safe candidate | Spent screen, then fresh result only if powered | Close automatic route on a failed spent screen |
| P — harm-bounded composite | New campaign only | Tier-D | New exact candidate and new protocol identity | Published power GO/NO-GO; fresh result only after GO and owner authorization | No fresh access below the preregistered 80% power bar |
| C — CatBoost gap | Rolling mechanism backlog via Track I | Development screen, then Tier-D if automatic | A C mechanism reaches the top of the Track I backlog | Hypothesis result with bounded claim | Do not tune repeatedly on the same spent outcomes |
| I — idea intake | Standing backlog | Scouting notes; normal tiers on adoption | An external idea with a stated primary Pareto axis and expected value on an eligible general-development slice and sports | Rated two-shortlist entry, then a normal track on adoption | Drop entries without either consolidation or a bounded-complexity case |
| Z — 1.0 cleanup | Conditional | Engineering/API | Surviving surfaces stable and deprecation inventory approved | Compatibility report, removal diff, release notes | No deletion for a line-count target |

## 4. Non-goals

- No CTR23 or other sealed lockbox access without a new, published,
  power-qualified authorization and explicit owner approval.
- No sports or noisy-data default justified by overlap-permitting folds.
- No default justified by sports evidence alone; joint broad-panel evidence
  is required.
- No TabArena benchmaxing. TabArena-Lite is a per-release descriptive drift
  check (M4), never a per-decision input or an optimization target.
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
overlap each other or unrelated heavy work. Every ChimeraBoost arm in this
gate resolves to the header pin `f14be60` (`v0.18.0-6-gf14be60`) — six
commits past the `v0.18.0` release tag, not the tag itself — and results
must be labeled with that exact pin.

### M0 — documentation checkpoint

The initial checkpoint landed at `f9b642c` (this plan) and `2dcb1c1` (the
testing log, the 0.18 diagnostic, and pointer updates). Land subsequent
documentation revisions, including this reframed plan, the same way: as
standalone documentation commits before any benchmark protocol binds new
source pins. Record the exact clean source pins used by subsequent
protocols. This is a provenance step, not feature work.

### M1 — large-n matched engine characterization

The primary estimand is matched-capacity product-path throughput, not
product-default performance and not isolated "matched-core" throughput: the
inherited workload keeps each product's own public border construction
(DarkoFit's 200k border sample versus ChimeraBoost's full-data borders), so
preprocessing above 200k rows is matched in capacity, not byte-identical.
Claims must be worded accordingly. Freeze one common configuration matching
the historical large-n comparison and compare:

- DarkoFit 0.10 under the common configuration;
- ChimeraBoost 0.18 quantized under the same configuration; and
- ChimeraBoost 0.18 with `quantize_gradients=False`.

An optional ChimeraBoost 0.15-era float arm under the same configuration may
be added as a labeled diagnostic outside the primary blocks; without it, no
claim may attribute movement since the historical comparison to 0.18
specifically rather than to accumulated changes since the old pin.

Reuse the explicit workload and matched capacity in
[`benchmarks/large_n_engine_protocol.md`](benchmarks/large_n_engine_protocol.md):
24 numeric features, 500k/1M training rows, a following 100k-row holdout, 300
trees, learning rate 0.1, depth 6, L2 1, 128 bins, full rows/features,
minimum child weight 1, ordered boosting and early stopping off, random state
4, and a fixed equal thread budget supported by the execution machine. The
2026-07-20 pre-freeze feasibility check found 14 logical CPUs and rejected
the inherited 18-thread value because TBB silently capped it. Preserve the
protocol's product-specific controls,
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

M1 answers the direction of the current same-machine large-n comparison, and
the quantized-versus-float pair isolates the flag's effect within the current
0.18 source. Attribution of the total movement since the historical
comparison requires the optional 0.15-era diagnostic arm and the original
18-logical-CPU machine. M1 does not prove that DarkoFit should implement
quantization.

Wave 1's dated executable contract is
[`benchmarks/m1_q0_wave1_protocol.md`](benchmarks/m1_q0_wave1_protocol.md).
It binds the clean post-H1 DarkoFit package source
`726e5d8e6131c580bce948db833a5007d0692dca`, the exact ChimeraBoost header
pin, all six primary-arm permutations, and a non-certifying material-donor
rule before any current outcome is inspected.

M1 completed on 2026-07-20 with all integrity checks passing. DarkoFit was
15.53% faster than current quantized ChimeraBoost in the equal-size
geometric mean. ChimeraBoost quantization improved its own fit time by 9.64%
with neutral quality, narrowly missing the frozen 10% material-donor rule.
The result is terminal and will not be rerun to cross the threshold; see
[`benchmarks/m1_wave1_result.md`](benchmarks/m1_wave1_result.md).

### M2 — current-version broad characterization (periodic milestone)

M2 is a periodic milestone, not an upfront prerequisite: run it after a
mechanism survives M5, an eligible M6 successor or prospectively frozen
mechanism-specific development evidence, and the sports stage of the
pipeline, or before a meaningful release. It is the broad checkpoint of the
mechanism pipeline and the operative test of the calibrated-yardstick proxy
in §0. G-M does not wait for it.

Create a new dated protocol that reuses the exact 13 datasets and
`r0f0/r1f1/r2f2` coordinates from the historical panel while preserving all
old artifacts. Primary arms:

- the exact frozen pre-mechanism DarkoFit default control;
- the candidate DarkoFit source and nominated public configuration (or the
  release-candidate default for a release milestone);
- ChimeraBoost 0.18 default; and
- CatBoost 1.2.10 default.

Add ChimeraBoost 0.18 float as a clearly labeled diagnostic arm for
quantization attribution. The protocol must bind both DarkoFit source hashes
and distinguish candidate-versus-control attribution from
candidate-versus-rival positioning. Keep shared splits, preprocessing
boundaries, thread budgets, early-stopping rules, fresh-worker execution,
equal-dataset aggregation, full per-coordinate rows, fit/predict time, and
RSS. Publish current-version quality and cost; do not use this spent panel
to authorize a new default.

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
overlap-permitting diagnostic. The panel's nine cells are three targets
across three seasons with shared players and overlapping rows; they are not
nine independent lineages. Aggregate uncertainty must use paired,
season-clustered inference, with held-team and cold-player views as
declared guardrails, and cross-season generalization claims require
genuinely unseen seasons. Report the quality/cost Pareto: fit, predict,
peak RSS, model bytes, and available OOB/member telemetry.

M3a is quality-first: run the quality scoring pass before committing to
repeated timing. If group-safe ensemble value does not survive the
player-disjoint view, record single descriptive timings only and skip the
repeat series. Supplement the sports scoring with selected non-sports M6
cells run with `n_ensembles` as descriptive context, while M5 checks
invariants and drift. This keeps an ensemble verdict from being sports-only
without turning the sentinel suite into a scoreboard.

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

M3a completed once under the frozen contract at harness `dae36ac`; combined
artifact SHA-256
`c811c8b04cbbaff6edb8226d7e8f5dbac3f9229adf18c3f8b658129ba7fc459a`.
All integrity checks passed. DarkoFit group8/single was `1.025482` on
player-disjoint RMSE, with clustered p95 `1.032391`, held-team `1.016048`,
and cold-player `1.015661`; repeat timing was therefore forbidden. The
row5, row8, and group5 diagnostics also lost. ChimeraBoost ensemble8/single
was `0.950230` on player-disjoint sports with all nine cells improving and
`0.947797` on the selected general medium cells with 6/6 wins. The M3a
terminal disposition closes the current DarkoFit ensemble implementation;
the distinct donor mechanism feeds the G-M portfolio judgment below.

### M4 — TabArena-Lite drift check (release cadence, optional)

A descriptive Tier-E TabArena-Lite run placing DarkoFit on the same scale
already published for ChimeraBoost and CatBoost. At most once per minor
release, never per-decision, and never an input to any gate. Its job is to
validate the calibrated-yardstick proxy in §0: confirm that the internal
panels' verdict against a pinned ChimeraBoost still transfers to TabArena
position, so routine general-quality tracking can stay on the cheap
internal comparison. It is not to be optimized. M4 may run after G-M, must
not contend with M1–M3a machine time, and touches no sealed lockbox.

### M5 — standing diversity sentinel suite

A small, fixed, fast regression-detection suite — minutes, not hours —
built largely from SynthGen generators plus a few small pinned real
datasets. It is the routine drift guard between M2 milestones, not a quality
ranking panel. Reserve *canary* for SynthGen datasets whose fixed verifier
has earned the known floor; earned canaries may be components of M5.

Fixed coverage domains:

- grouped/entity regression through a generic group-bearing generator;
- smooth numeric regression;
- noisy numeric regression;
- categorical and missing-value regression;
- high-row-count numeric fitting and prediction;
- binary classification;
- multiclass classification;
- weighted regression; and
- weighted classification.

Each domain pins datasets, seeds, configurations, correctness invariants,
behavior fingerprints, and expected quality ranges. Hard failures are
crashes, invalid/non-finite outputs, exactness or serialization breaks, and
violations of a frozen known-floor invariant. Other quality drift triggers
investigation and blocks advancement until explained, but is not itself an
acceptance score; no mechanism may be tuned to M5. Performance sentinels use
paired ratios against a pinned control in same-machine blocks with a hardware
fingerprint, never portable absolute-second ranges. M5's classification and
weighted domains reduce the current proxy blind spot but do not establish
comparative classification strength.

Building M5 is standing Tier-E infrastructure work authorized in Wave 1.
Its v1 executable contract predeclares 19 cells across all nine domains, the
exact post-H1 control, task-normalized quality ranges, two earned canary
floors, serialization/prediction invariants, and paired resource reporting.
Its 38-row behavior-identical baseline completed without failure and is
create-only and hash-bound; M5 v1 is frozen for future non-ranking drift
checks.

### M6 — fast general development slice

M6 was designed as the missing quality-development rung: a small, pinned,
explicitly spent comparative slice that could rank or kill mechanisms after
backtest qualification but could not support a shipping or default claim.
Its initial dataset contract reuses the existing
`benchmarks/benchmark_adapters.py` builders and weight modes rather than
creating a second data layer: numeric and categorical regression, binary and
multiclass classification, missing-value coverage where supported, and
unweighted plus stress-weighted cases at fixed small/medium sizes and seeds.

Every mechanism evaluation includes the exact frozen pre-mechanism DarkoFit
control and the candidate source. Pinned ChimeraBoost and CatBoost anchors
are established when M6 is frozen and refreshed at release cadence, not
rerun during every inner development iteration. Report task-appropriate
quality, fit/predict time, peak RSS where practical, failures, and resolved
model metadata. Only a backtest-qualified M6 successor may serve as the
standing cheap panel that influences quality-oriented backlog ranking;
tuning directly to its individual cells is prohibited, and repeated
inspection makes all outcomes spent. Every material full run receives a
stable mechanism id and a monotonically increasing, one-based inspection
index in both its manifest and testing-log
entry. Assign the index before launch; failed attempts consume it. Missing,
reset, or selectively omitted indices invalidate the mechanism's M6 audit.

The executable contract must refuse a frozen state until medium-size cells
and exact pinned ChimeraBoost and CatBoost release anchors are present; the
2,500-row draft is infrastructure smoke, not a size decision. The historical
backtest subset is committed before replay and includes both positive and
negative verdicts, including a quality-negative selector. Changing that
subset after inspection requires a new contract version and a fresh backtest.
Draft v3 includes both sizes, worker peak RSS, exact external source pins,
and machine-readable replay cases and gates. The complete 240-row
release-anchor artifact is now create-only and hash-bound, so the M6 contract
is frozen. Its replay executor binds the exact historical fused and packed
runners plus the source-pinned six-cell selector adapter before outcome
access. The first outcome-bearing replay terminal-failed: fused disagreed
with its known positive verdict, and the exact 18-thread packed runner was
unexecutable on the current 14-thread machine before model access. The
selector was not opened. The failure is hash-bound, reruns are closed, and
M6 remains non-ranking under v3.

Building and backtesting the M6 contract is Tier-E infrastructure work
authorized in Wave 1. It must reproduce a declared subset of prior mechanism
verdicts before it can rank new work.

The 2026-07-20 historical backtest failed terminally — the frozen analyzer
classified a known-advance mechanism as kill — so M6 v3 is non-ranking
(see [`benchmarks/m6_historical_backtest_result.md`](benchmarks/m6_historical_backtest_result.md)).
Rehabilitation, if ever pursued, requires a new contract identity with a
newly declared verdict subset, never a relax or rerun of the failed replay,
and every replay in the new subset must be executable within the current
machine's limits: the failed round's packed replay hard-required 18 Numba
threads on a 14-thread machine and could therefore only record
`lacks_power`.

### G-M — owner-facing decision

G-M is an owner portfolio judgment informed by declared descriptive
measurements. It is not itself an evidence gate: no Tier-E measurement
becomes a pass/fail certification through it, and the binding evidence
rules live at Tier-D. Each of M1 and M3a may publish its result and a
provisional per-track disposition as it completes rather than waiting for
the other; G-M is the single short portfolio note that sets priority
afterward. G-M does not wait for M2. Out of G-M, fund at most one private
engineering prototype at a time.

Publish one short decision after M1, the Q0 profile, and M3a:

- **Q prototype fund** only if M1 and the Q0 profile show a material
  current engine opportunity; otherwise defer or close Q. Q0's prototype
  result—not G-M—later decides whether Q1 is justified.
- **B continue** only if ensemble quality survives the player-disjoint view
  at an acceptable quality/cost position; otherwise preserve the current
  opt-in without a v3 program.
- **C and X mechanisms** are not G-M peers (owner decision 2026-07-20):
  they compete through the Track I backlog. If neither Q nor B earns
  funding, the top backlog mechanism — currently the T7b-derived quality
  levers — is the natural next bet.
- At G-M, **close** is the default outcome for a track that does not earn
  its slot; **defer** requires a named re-entry condition.
- M3a may inform a documented recipe. It cannot change an ensemble default.

G-M published 2026-07-20 in
[`benchmarks/wave1_gm_decision.md`](benchmarks/wave1_gm_decision.md):

- **Q closes.** Q0 passed its local projection, but M1's
  quantized/float donor ratio `0.903595` missed the frozen `0.90` materiality
  threshold. The near miss is not relaxed.
- **The current DarkoFit ensemble route closes.** All shipped row/group arms
  lost quality and group8 did not earn timing repeats.
- **One private B mechanism prototype is funded.** The separate predeclared
  ChimeraBoost arm produced consistent roughly 5% sports and general quality
  gains. B0 plus a sequential B1/B2 attribution prototype may isolate
  without-replacement sampling and a named member policy under a new
  candidate identity. B3 parallelism is deferred because the donor's
  sampled aggregate RSS was `6.16x` its single.
- No public surface, default, M2, fresh data, or lockbox access is authorized.

---

## 6. Gate F — feasibility before public implementation

### Q0 — quantization attribution and prototype

Q0 has two halves with different costs. The profiling half is cheap and
runs alongside M1, before G-M; its output feeds the G-M funding call. The
prototype half is the fundable private engineering bet and is built only
if G-M funds it.

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

Declare Q0's speed budget — the minimum end-to-end fit improvement that
would justify Q1's added arithmetic and maintenance surface — in the Q0
protocol before profiling begins, not after inspecting the profile. If the
attainable upper bound cannot meet that pre-declared budget, record the
profile and close Q. Do not implement a public option to match a
competitor's architecture when DarkoFit's bottleneck is elsewhere.

The frozen Wave 1 budget is 10% lower end-to-end fit time. The cheap profile
uses only the current fused production path for its funding projection and
screens it with a conservative 1.30x eligible-kernel prior; the forced
unfused path decomposes histogram construction from split search but cannot
contribute a production share. Full equations, stability rules, and the
close-before-prototype disposition are fixed in
[`benchmarks/m1_q0_wave1_protocol.md`](benchmarks/m1_q0_wave1_protocol.md).

The profiling half completed on 2026-07-20 with all integrity checks passing.
Its preregistered projection was a 13.28% end-to-end reduction, so Q remains
eligible for G-M rather than being closed before prototype. The behavior-exact
unfused diagnostic was unexpectedly faster on the current 14-logical-CPU
machine; that hardware-dispatch signal is recorded but did not enter the
frozen quantization projection. See
[`benchmarks/q0_wave1_profile_result.md`](benchmarks/q0_wave1_profile_result.md).

Taken together, M1 and Q0 do not satisfy Q's conjunctive G-M funding
condition: Q0 passed its local projection screen, while M1 did not establish
the predeclared material donor opportunity. The provisional Q disposition is
therefore close/do-not-fund; the final portfolio record belongs in G-M after
M3a.

### B0 — ensemble compatibility and sampling design

Completed 2026-07-20 as the private contract in
[`benchmarks/b0_ensemble_v3_contract.md`](benchmarks/b0_ensemble_v3_contract.md),
without changing the current full-size bootstrap semantics. The design
covers:

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

Q closed at G-M on 2026-07-20. The design below remains a re-entry contract,
not authorized work. Re-entry requires a new pinned donor result that clears
the material threshold or a distinct DarkoFit-specific causal case.

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

B0 plus one private sequential B1/B2 attribution prototype were funded at
G-M on 2026-07-20 and completed the same day. The work used a new private
candidate identity without silently changing the meaning of existing
`n_ensembles` configurations. M3b retained no arm, so B3, a public option,
and any default are closed rather than merely conditional. The current
bootstrap member policy also remains closed as a quality route.

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
4. **M3b — attribution:** compare sampling-only, member-policy-only, and
   combined arms on the spent player-disjoint panel and a small
   broad-tabular development set. Freeze the M3b protocol before inspecting
   any prototype outcome. Complete the paired weighted-holdout quality pass
   for every arm before repeated timing; only quality-eligible arms proceed
   to repeated fit, predict, RSS, archive-byte, and OOB measurement. The new
   contract must fix the worker environment before interpreter startup,
   record and validate fitted thread masks plus the Numba ceiling, bind exact
   split/weight fingerprints and implementation paths, and strictly validate
   fitted model and classification-probability metadata. The funded M3b
   excludes a parallelism arm; it joins only if B3 is later unlocked by a
   separate decision.

M3a's v1 runner, analyzer, freezer, and contract remain byte-preserved
historical evidence. M3b must use a new contract identity and must not reuse
M3a's `--contract` option with modified contract contents. The draft
`paired-evidence-v1` execution foundation is non-ranking and does not make
M6 v3 eligible; M3b must bind it (or a stricter successor) into M3b's own
prospectively frozen contract.

M3b may nominate a documented opt-in recipe or explicit v3 preset. A general
default change requires a separate Tier-D campaign across numeric,
categorical, classification, weighted, and relevant grouped workloads. The
T10 sports automatic-ensemble refutation remains in force.

### M3b disposition — closed 2026-07-20

The prospective campaign preserved three immutable attempt identities:

- Attempt 1 terminated before model fit with zero completed rows when the
  inherited process-tree RSS sampler hit a macOS sandbox denial. Its terminal
  artifact and failure record are preserved; no model outcome was opened.
- Attempt 2 replaced only the RSS measurement with self-worker-process RSS,
  then terminated during safe reload of the first group-bootstrap control.
  One completed row was discarded and not published or inspected. The
  failure exposed a real loader defect: group bootstrap can legitimately
  sample a row count different from the input row count. The implementation
  fix and uneven-group regression test landed under a new source pin.
- Attempt 3 bound both failures, the corrected source, the unchanged cases,
  arms, ordering, and decision rules. It completed the 65-row quality grid,
  frozen gate, and 130-row repeated-timing grid. All three candidates were
  quality-eligible, but none survived the final rules.

The binding attempt-3 result is
[`benchmarks/m3b_ensemble_v3_r3_result.json`](benchmarks/m3b_ensemble_v3_r3_result.json),
with the human-readable record in
[`benchmarks/m3b_ensemble_v3_r3_result.md`](benchmarks/m3b_ensemble_v3_r3_result.md).
B1 sampling failed both its value and archive/single checks; B2 member policy
failed both; the combined arm passed its quality/value, fit, predict, RSS,
and archive/control checks but failed the predeclared archive/single limit
(`5.534767` observed versus `4.0` maximum). The deterministic disposition is
`close_b1_b2_preserve_existing_opt_in`: no retained private arm, B3, public
or default surface, fresh confirmation, TabArena, or lockbox access.

A post-close contract audit found two implementation gaps outside the frozen
result: bootstrap classification had enforced full class coverage only on the
training draw, and private safe-load could validate only the syntax of index
digests. The follow-up fix requires every class, with positive mass when
weighted, on both training and OOB sides and binds private archive digests to
stored index payloads. Public ensemble archives remain format 1. These are
post-campaign correctness fixes; they neither amend attempt 3's source pin nor
reopen its outcome or disposition.

A second post-close review confirmed three further live implementation gaps;
its weighted-classification and digest concerns were already closed by the
preceding hardening. The private prototype now rejects a non-`None` preset
before sampling so a fit-time profile cannot override B2 or explicit-`None`
precedence, enforces the physical group-count/row-count bounds on safe load,
and uses private metadata schema 3. Schema 3 records the canonical base wrapper
constructor, every member wrapper constructor after only the frozen mechanical
and B2 overrides, and every fitted booster's original constructor inputs. Safe
load binds all three maps and rejects obsolete private schema 2; automatic
tree-mode selection and the deprecated automatic learning-rate probe are
outside this constructor-bound private identity. The fitted numeric learning
rate is stored separately from the constructor input. These corrections use a
new implementation pin and do not rewrite or reinterpret the immutable r3
artifacts.

The schema-3 correctness implementation is commit `e44de0f`. The focused
private and serialization/ensemble regressions passed before checkpointing;
the historical r3 source-pin guards remain intentionally bound to their old
implementation rather than being rewritten.

---

## 9. Track X — cross-feature research opt-ins

Tier note, requiring owner ratification: the shipping policy assigns
"automatic policies" to Tier-D. This plan reads that as governing
default-on automation. A guarded selector that runs only when the user
explicitly invokes it is treated as a Tier-E opt-in surface whose internal
automation ships with mandatory disclosure; any default-on engagement of
the same mechanism is Tier-D. If the owner rejects this reading, guarded X
moves to Tier-D or is dropped.

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
needs a fresh Tier-D confirmation campaign.

C is a rolling mechanism backlog managed through Track I, not a standing
campaign and not a G-M peer (owner decision 2026-07-20): the CatBoost gap
is a source of candidate mechanisms and a quality ceiling, not the
strategic center. The T7b-derived mechanisms stay near the top of the
backlog because they are the only current quality levers with attribution
evidence, and CatBoost's sports-panel dominance — better quality at lower
fit time — marks quality-plus-sports-speed as the weakest region of
DarkoFit's Pareto frontier. Source at least part of the C candidate set
from an eligible M6 successor or mechanism-specific general-development
evidence and the broad panel's worst datasets, not only from sports, so
candidate generation stays unbiased.

---

## 13. Track I — external idea intake

A standing, origin-agnostic scouting backlog. Any library or paper is a
valid source; ChimeraBoost holds no privileged position beyond the existing
NOTICE-based porting practice. Each entry records the mechanism, its
source, its primary Pareto axis, expected value on an eligible general
development slice and the sports panel, estimated implementation surface,
and the code it would displace or consolidate. A genuinely net-new capability
instead records a bounded complexity budget, maintenance owner, and review
date. Entries without either a consolidation story or a justified
bounded-complexity case are dropped.

Track C's quality mechanisms and Track X's cross-feature surfaces compete
through this backlog (owner decision 2026-07-20). Ranking must credit
quality-Pareto movement explicitly: a mechanism-led pipeline naturally
favors speed mechanisms, whose value appears in a profiler, over quality
mechanisms, whose value only shows up on panels — the backlog must not
drift all-speed. Maintain separate quality/reliability/capability and
speed/memory/maintainability shortlists. Every portfolio decision compares
the best evidence-adjusted candidate from each shortlist before funding the
single next prototype; profiler evidence alone cannot make a speed candidate
win by default.

Initial backlog, unrated and unauthorized:

- **Monotonic and interaction constraints** (CatBoost/LightGBM/XGBoost):
  mainstream general-tabular capability, direct sports value as domain
  priors, absent from DarkoFit today. Treat as a high-value capability gap,
  with rank still to be earned against an eligible general-development slice
  and implementation cost.
- **Exclusive feature bundling** (LightGBM): sparse and high-cardinality
  fit speed, orthogonal to quantization.
- **Langevin boosting / SGLB** (CatBoost): cheap ensemble diversity; a
  potential Track B interaction.
- CatBoost's `l2_leaf_reg` and samples-per-feature depth heuristics:
  already Track C candidates via the T7b attribution.
- **Closed — B-archive shared-component ensemble serialization** (internal,
  from Wave 2 M3b): the dated matched-single readout confirmed that the r3
  combined B1/B2 arm improved on the single reference in all 13 development
  cases (pooled primary geomean `0.9655`) while missing the frozen archive
  gate at `5.534767×`. A one-case exploratory census found only seven complete
  exact `prep__*`/`bin__*` arrays and their preprocessing header, with enough
  first-order savings to justify one representative feasibility campaign but
  not an implementation.

  B-archive v1 terminal-failed before publishing a row because its harness
  required optional `feature_names_in` even though every frozen case supplies
  NumPy input. V2 used a new identity, hash-bound that failure, and preserved
  the source, runtime, 13 cases/fingerprints, arms, median aggregation, and
  `<= 4.0` limit. All 13 rows passed current safe-NPZ prediction/probability,
  feature-schema, metadata, deterministic-resave, thread, provenance, and
  component invariants. Eleven numeric target-free cases factored only the
  exact seven-array, three-header-field canonical section; the two categorical
  member-local cases stayed at current bytes. The current median was
  `6.032405×`; the effective median was `4.152525×`. Because it missed the
  prospectively frozen limit, B-archive is closed. No canonical serializer,
  generalized delta format, B3 work, public/default change, fresh evidence,
  or lockbox access is authorized. The M3b sports view remains player-disjoint
  cold-player evidence within held teams; the seeded 75/25 split applies only
  to the general weighted view.
- **Next nominated — behavior-exact fused-lane dispatch** (internal, from Q0): the
  forced-unfused reference was behavior-exact and faster than the fused
  production lane on the current 14-CPU machine (paired fit ratios `0.901`
  at 500k rows, `0.981` at 1M) against a hotspot worth 52–63% of fit time.
  A hardware/shape-aware dispatch is a cheap Tier-E engineering candidate
  in the sampled-fused-kernels tradition. It precedes and re-baselines any
  Q re-entry work: quantization must beat the post-dispatch engine, not the
  current one. B-archive's frozen close promotes this to the next mechanism
  slot, but implementation still requires a new prospective dispatch contract
  with exact behavior and bounded speed/resource acceptance rules. That design
  contract is now frozen in
  [`benchmarks/fused_lane_dispatch_v1_contract.md`](benchmarks/fused_lane_dispatch_v1_contract.md):
  it permits one macOS-arm64, shape-aware scalar threshold between the existing
  proven fused and unfused paths, with generic synthetic calibration separated
  from six outcome-unseen validation cells. It authorizes only the private
  selector and invariants; calibration and validation still require separate
  create-only execution freezes.
- **Q re-entry microbenchmark** (internal): a DarkoFit-specific private
  histogram-bandwidth prototype at the Q0 hotspot (frozen conservative
  projection `0.867242`), pursued only as the distinct causal case the Q
  re-entry contract requires — no donor dependence. Competes here; it does
  not reopen Q by itself, and it is measured against the post-dispatch
  baseline if the dispatch mechanism lands first.

Adoption is never authorized from this backlog directly. A promoted entry
becomes a normal track with its own evidence class, gates, and stop rule.

---

## 14. Track H — hygiene and documentation

### H0 — clean documentation checkpoint

The original checkpoint has landed (see M0). When authorized, commit any
subsequently revised documentation the same way: as a standalone checkpoint
before benchmark protocols bind new source hashes. Do not mix it with
feature implementation.

### H1 — audit, then fix only confirmed gaps

ChimeraBoost's audit findings are prompts, not evidence that DarkoFit has the
same bugs. **Closed 2026-07-20** on the clean post-hygiene code pin
`726e5d8e6131c580bce948db833a5007d0692dca`; the complete per-item
dispositions and verification are in
[`benchmarks/h1_hygiene_audit_result.md`](benchmarks/h1_hygiene_audit_result.md).

- **Thread state — fixed 2026-07-20:** the confirmed same-thread gap was
  closed with nested-safe call-local save/restore around scalar,
  multiclass, and distributional fit/predict operations. The fitted
  `n_threads_` mask still governs kernels; the caller's thread-local ambient
  mask is restored afterward, including predict-during-fit and staged
  resumptions. Named regression coverage lives in
  `tests/test_thread_state_restoration.py`; the existing thread-local warmup
  coverage remains unchanged. Do not describe this as a process-global leak.
- **Serialization — not present:** safe NPZ serialization never included the
  rebuildable flat predictor cache. Named coverage proves byte-identical
  archives before/after cache construction and bit-identical lazy rebuilds.
- **Loud failures and parameter resolution — closed:** unseen classifier
  `eval_set` labels and NumPy integer `cat_features` were already correct.
  The positional-weight failure now names the required `sample_weight=w`
  keyword. `None` semantics are explicitly documented and depth resolutions
  have named coverage.
- **Adjacent compatibility — fixed:** the scikit-learn 1.0–1.5 tag fallback
  now preserves `allow_nan`, two-dimensional-only input, and `requires_y`;
  newer structured tags remain unchanged.

For each item, publish one of: confirmed and fixed with a named regression
test; not present with a reproducer; or intentionally different with a
compatibility note. Do not rewrite already-correct input-hardening or warmup
behavior. Complete H1 before freezing the M1/Q0 protocol, then commit and
record one clean post-hygiene DarkoFit source pin for all Wave 1
measurements. This avoids measuring a pre-audit pin or churning a frozen
protocol when a confirmed hygiene fix lands.

### H2 — measurement documentation

M1 and M2 add new dated records to the measurements and testing-log surfaces.
Older records keep their source/version boundaries and are never overwritten.

---

## 15. Track Z — conditional 1.0 cleanup

Do not use an arbitrary 12–15k-line target. Start from the deprecations already
announced in `CHANGELOG.md`, build an API/serialization compatibility
inventory, and measure the actual removal. Delete code because an approved
deprecation matured and coverage proves the replacement—not to hit a line
count.

Under the strongest-library goal this track is a competitive feature, not
housekeeping: a high shipped-to-carried ratio is part of what "best
available" means, and it is the counterweight to Track I's absorption. Z
stays independent of the speculative tracks and may proceed whenever the
deprecation inventory is approved.

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

## 16. Execution dependencies and rough effort

These are planning ranges, not delivery promises.

| Wave | Work | Approximate effort | Decision |
| --- | --- | --- | --- |
| 0 | Owner reviews this plan; H0 documentation checkpoint | Hours | Authorize or revise Gate M |
| 1 | **Complete:** H1 audit; M1/Q0; M3a quality-first; M5 baseline; M6 infrastructure/backtest; G-M | Completed 2026-07-20 | Q and current ensembles closed; one private B0/B1/B2 mechanism prototype funded |
| 2 | **Complete:** B0 contract, private sequential B1/B2 prototype and invariants, prospectively frozen M3b attribution | Completed 2026-07-20 | No arm cleared every final gate; close B1/B2 and preserve the existing opt-in |
| 3 | **Complete:** M3b matched-single readout, B-archive component feasibility, v1 terminal lineage, and corrected frozen v2 campaign | Completed 2026-07-21 | Exact canonical factoring missed `4.0×`; close B-archive with no serializer and nominate fused-lane dispatch |
| 4 | **In progress:** behavior-exact fused-lane dispatch; design contract frozen before implementation, with calibration and validation execution freezes still required. M2 waits for a survivor; Q1 remains sequenced after the post-dispatch baseline | Contract frozen 2026-07-21; implementation separately estimated | Retain exact dispatch, defer, or close |
| 5 | Q3, S, or P-next Tier-D campaigns | Separately estimated and power-gated | One fresh campaign per qualified automatic policy |
| Later | Track I backlog (C and X mechanisms included), M4 drift check, and Z cleanup | Independent backlog | No coupling to unfinished speculative tracks |

Benchmark waves use fresh workers and exclusive machine access. Parallelize
code review and analysis, not timed model jobs that would contend for CPU,
memory bandwidth, or cache.

## 17. Evidence and stopping discipline

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

The program reaches a clean stopping point when M1, the Q0 profile, and M3a
have published current-version answers; every downstream track is explicitly
shipped, deferred, or closed; any M2 milestone that became due is published;
no default claim rests on spent evidence; the testing log and public claim
surfaces agree; and any unused lockbox remains sealed.

Wave 1 reached that stopping point on 2026-07-20. M2 did not become due:
the current candidates did not survive their sports gates, and M6 v3 never
became ranking-eligible after its terminal replay failure. The newly funded
B mechanism starts a new private prototype cycle. The lockbox remains sealed.
