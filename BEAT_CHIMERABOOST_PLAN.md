# BEAT_CHIMERABOOST_PLAN — the pathway to a clean win over the rival

> **Status:** owner direction, 2026-07-22. Execution instruction for Codex.
> Authorization: Phase A is signed off by the owner's adoption of this
> document; each later phase names its own gate. Governing discipline is
> unchanged: [`benchmarks/SHIPPING_POLICY.md`](benchmarks/SHIPPING_POLICY.md),
> the gate-design rules in [`NEXT_STEPS.md`](NEXT_STEPS.md) §4.9, create-only
> artifacts, TESTING_LOG entries, exclusive machine for timed runs, no
> fresh-confirmation or lockbox access anywhere in this plan.

## 0. What "beat ChimeraBoost" means (owner definition, 2026-07-22)

**Strict Pareto dominance against the moving target.** The rival pin is
the **current ChimeraBoost release, refreshed at each milestone** — they
are active, and beating a stale ghost does not count. Victory is defined
on the quality-versus-compute frontier: **at any given level of compute,
DarkoFit must deliver equal or better performance.** Concretely, at each
milestone the scoreboard evaluates a compute ladder per engine — its
default, its accuracy-oriented configuration, its ensemble — and
DarkoFit's quality-vs-fit-time curve must sit on or above the rival's
curve at every evaluated budget, with the memory lead retained and the
prediction curves reported the same way. A win at one budget cannot excuse
a loss at another; dominance means the whole curve.

Today's known frontier positions vs the 0.18 pin: DarkoFit is ahead on
sports quality (−2.9%), training speed (0.81×), memory (0.84×), and
grid-shape prediction (0.49×, 16/16); behind on broad singles quality
(1.0174, W-L 6-7) and panel-shape prediction (1.32×); the ensemble
comparison becomes measurable once v0.11 ships.

The two structural facts this plan exploits: the broad-quality deficit is
**concentrated** — near-parity on 11/13 datasets, with `airfoil_self_noise`
(+7.1%) and `physiochemical_protein` (+8.8%) supplying about 68% of the
summed per-dataset log gap —
and the machinery to fix it already exists in this repo.

**Product philosophy (owner directive): automation-first.** The product's
job is to decide what's optimal automatically — auto parameters, auto
selectors, auto dispatch, auto compute-budget configuration. Manual knobs
exist as escape hatches and research surfaces, not as the product story:
a library where everything must be configured by hand is too complicated
to be the best library. Consequence for this program: quality mechanisms
target **automatic engagement** as their end state, which means the
Tier-D path (powered panels, harm-bounded selectors) is core product
work, not an optional extra. This directive is recorded durably in
[`AGENTS.md`](AGENTS.md).

---

## Phase A — Ship what is built: v0.11 (authorized now)

The evidence conditions gated on are met (reproduction clean, stop
conditions clear). Execute:

1. **Expose ensemble-v3 publicly** exactly per the frozen
   [`ensemble_v3_public_contract.md`](benchmarks/ensemble_v3_public_contract.md):
   constructor parameters, exports, loud-error support matrix, public
   safe-NPZ schema. No default changes.
2. **Docs + CHANGELOG.** Use the final v2 characterization numbers, not the
   superseded v1 checkpoint: 13/13 vs the matched DarkoFit single, `0.965513×`
   pooled; sports `0.961077×` with season-cluster interval
   `[0.958861, 0.962867]`; general `0.975569×` with case-bootstrap interval
   `[0.963303, 0.987718]`; fit `5.030×` single; process-tree peak RSS
   `1.090×` and peak-minus-start RSS `3.539×`; archive `6.181×` with
   per-case medians about 0.37–1.78 MB (telemetry, not a gate).
   **Prediction is presented as the full shape-dependent record, never one
   favorable number:** single-model predict is faster than ChimeraBoost on
   the dedicated grid (`0.478×`, 16/16) and slower on M2 panel shapes
   (`1.317×`); v3 takes `6.251×` the DarkoFit single's prediction time on the
   dedicated grid, while taking `0.126×` the pinned ChimeraBoost ensemble8's
   time there. These are descriptive measurements on different workloads,
   not a cross-panel composite or a certification.
3. **M4 TabArena-Lite: DEFERRED (owner decision, 2026-07-22).** The first
   placement is postponed — not because the thermometer is wrong, but
   because it would currently read a library whose classification side has
   never been comparatively developed, and a first number sticks.
   Re-entry: owner sign-off at a later release, at earliest after the
   first quality mechanism lands; a comparative classification slice
   remains the standing prerequisite consideration. No TabArena access of
   any kind in this release.
4. **Release v0.11**: version bump, annotated tag, GitHub release. Claims
   bounded per the plan's marketing constraints (panel-scoped, versioned,
   with costs adjacent). The promoted dispatch rides this release as
   already recorded.
5. The unrelated README/site/dossier edits remain a separate checkpoint.

**Deliverable:** public v0.11, the first release where a user can opt into an
ensemble that beat the matched DarkoFit single on all 13 fixed development
cases, while retaining the single-model speed/memory posture as a separate
compute point. No cross-engine quality claim is made until Phase E measures
the compute ladder directly. Estimated effort: half a day; M4 is deferred.

---

## Phase B — Close the smooth-data moat (the concentrated 2-dataset gap)

**B-1. Verify the mechanism (minutes; do this first).** Inspect the M2 raw
rows' fitted-model metadata for the ChimeraBoost arm on
`airfoil_self_noise` and `physiochemical_protein`: did its linear-leaf /
cross-feature / categorical-combination selectors engage? (Protein's
fit-time ratio 0.215 — ChimeraBoost spending 4.7× our time — is the
expected signature.) Publish a one-page dated note with the finding either
way. If selectors did *not* engage, Phase B stops and the gap cause is
re-investigated before any selector work is funded.

**B-2. Re-adjudicate the linear-leaves selector under the reformed rules
(the next mechanism slot, on owner confirmation).** History, stated
plainly in the campaign doc: the 3% smooth-data selector was killed in
fresh confirmation with aggregate **0.9893**, worst case **1.0000** (zero
harm), because it won only "2/14" — a verdict produced by the win-count
gate that `SHIPPING_POLICY.md` §1 later abolished *citing this exact case*
as the count metric's structural failure. This is a mistaken-gate casualty
in the same class as the archive gate, discovered late. Rules:

- **New campaign identity.** The historical closure stands as history; the
  frozen artifacts are untouched; this is a new candidate under modern
  harm-bounded evaluation (aggregate, bootstrap upper bound, LOO
  concentration, explicit worst-case harm — no win counts).
- **M6 caution (binding):** M6 v2's rule module encodes the historical
  "kill" verdict of this very selector as a backtest expectation. The
  revived selector must therefore run as its own campaign and **must not
  be pre-filtered by M6's kill rule** — otherwise the abolished gate kills
  it a second time from inside the new rig. M6 may still supply
  descriptive development context.
- Development on spent data (airfoil/protein are now spent and may be used
  to develop, never to confirm); any default-on engagement takes the full
  Tier-D path on prospectively frozen fresh evidence with a **design-time
  power analysis ≥ 80%** (Panel 3's lesson: qualify the panel first).
- Fit-time budget exists: we run 0.81× the rival's fit time; a selector
  that spends some of that on smooth data is following the rival's own
  proven playbook.

**B-3. No manual-switch product surface (owner decision, 2026-07-22).**
The automation-first directive replaces the earlier idea of shipping
manual selector toggles. The selector campaign's end state is **automatic
guarded engagement** — the model decides per-dataset whether linear
leaves / cross features help, under a harm-bounding guard — shipped
through the full Tier-D path (powered panel, harm bounds, no-rerun).
During development, force/guarded switches exist only as private research
surfaces, never as the shipped story. The X0-style product obligations
(persisted engagement reasons, serialization, SHAP semantics) apply to
the automatic surface.

**Deliverable:** the concentrated deficit attacked two ways — verified,
then re-adjudicated with automatic engagement as the target. Setting both
current outlier ratios to parity, with the other 11 fixed, would move the M2
equal-dataset aggregate from `1.0174×` to about `1.0056×`; diffuse gains are
still required for full-curve dominance. Estimated: B-1 minutes; B-2 a
standard mechanism campaign plus a Tier-D confirmation once the panel is
powered and fresh-evidence access is separately authorized.

---

## Phase C — T7b quality levers (the following mechanism slot)

Unchanged from the standing plan, now with a working cheap rung:
`l2_leaf_reg` and the samples-per-feature depth policy, developed through
**M6 v2** (its first legitimate customers; numbered, create-only, spent
inspections), then the sports panel, then the milestone M2 check. These
are diffuse small gains (~fractions of a percent each) that chip the broad
base against both rivals. Any default change: full Tier-D.

**Ordering (owner decision, 2026-07-22): B-2 → C.** The selector campaign
runs first because it targets the measured, concentrated deficit; the T7b
levers follow as the next slot. One mechanism at a time remains binding.

---

## Phase D — Prediction on panel shapes (speed slot after B/C)

The last axis where the rival leads at product level, and the evidence is
already split in our favor on grid shapes (0.49×, 16/16) while against us
on M2 panel shapes (1.32×). Sequence:

1. **Reconcile the discrepancy** (analysis, not benchmarking): identify
   which shapes/batch profiles flip the sign — the answer localizes the
   bottleneck (likely small-batch or per-dataset-shape overhead vs the
   packed large-batch path).
2. **One mechanism**, chosen from the localized evidence — candidates
   already on record: the deferred paged/fused leaf-wise prediction
   levers, batch-shape dispatch in the sampled-fused-kernels tradition,
   and batched member prediction for the ensemble path (its own predict
   multiplier is `6.251×` on the dedicated grid and per-member batching is
   the obvious lever).
3. Behavior-exact acceptance per the reformed rule: no materiality bar;
   exactness, stable direction, bounded complexity, defined envelope,
   rollback.

---

## Phase E — Standing cadence (how the win is verified and kept)

1. **Milestone frontier characterization at each release** against the
   refreshed rival pin: the M2 panel extended to a compute ladder per
   engine — default, accuracy-oriented configuration, ensemble — with
   per-dataset rows, dispersion, and the head-to-head win-rate supplement.
   §0's Pareto-dominance condition is evaluated here, on the whole curve,
   never on development panels. This is a larger overnight run than
   singles-only M2 and runs at release cadence only.
2. **M4 TabArena-Lite: deferred** per Phase A item 3; when re-authorized,
   once per release, descriptive, next to the rival's published position.
3. **M5 sentinels** guard drift between milestones; **M6 v2** ranks
   quality-development candidates (subject to the B-2 caution).
4. **Pin policy:** every campaign declares its rival pin explicitly; the
   local ChimeraBoost checkout's HEAD is never an implicit comparator. When
   they ship a release that plausibly moves an axis (their CHANGELOG is
   public), note it in the testing log and refresh the pin at the next
   milestone — beating a stale pin is not the goal.
5. **Scoreboard note per milestone:** one short dated table against §0's
   victory conditions, so "are we ahead" always has a current, honest,
   one-page answer.

---

## Phase F — Verdict audit: revisit rules that killed work prematurely
(owner mandate, 2026-07-22)

A one-time systematic sweep, then a standing rule:

1. Enumerate every historical closure in the program's ledger — S1, S2,
   OOB-5/T10, the linear-leaves selector, the categorical-combinations
   donor, `random_strength`, C2 native ordinal, the T5 composite, P2, E2,
   Panel 3's guarded cross features, Q, the B family, and anything else
   surfaced by `TESTING_LOG.md` and the plan ledgers.
2. For each: name the exact rule that killed it, and classify that rule
   under current policy — **still valid** (harm bounds, power floors,
   integrity/stability, genuine quality failures) or **since abolished or
   reformed** (win counts, size ratios, certification bars,
   rival-conditioned prongs, unjustified round-number materiality).
3. Casualties of abolished rules go on a dated re-adjudication backlog,
   each requiring a new campaign identity. The linear-leaves selector is
   casualty #1 (Phase B); Q's donor prong is already reopened.
4. Publish the audit as one dated create-only note. Frozen records stay
   immutable; forward effects supersede, exactly as `NEXT_STEPS.md` §1.
5. **Standing rule:** whenever a gate class is retracted or reformed, the
   retraction note must include the sweep of that gate's historical
   casualties. A retracted rule with unexamined victims is an unfinished
   retraction.

M6 note: M6 v2's backtest encodes the count-gate-era kill of the
linear-leaves selector as an expectation. M6 v2 stays frozen and usable
under its existing caution; any future M6 v3 must draw its backtest
expectations only from verdicts that survive this audit.

## Sequencing summary

| Order | Work | Gate |
| --- | --- | --- |
| A | Ship v0.11: ensemble-v3 public, docs, release (**M4 deferred**) | Authorized by this document |
| B-1 | Selector-engagement verification note | Authorized by this document |
| F | Premature-kill verdict audit note | Authorized by this document |
| B-2 | Selector re-adjudication campaign; **automatic engagement** is the target surface | Owner-confirmed next mechanism slot |
| C | T7b levers through M6 v2 | Follows B-2 (owner-confirmed order) |
| D | Panel-shape prediction mechanism | Owner confirms at its slot |
| E | Milestone frontier ladder + scoreboard each release | Standing, per release sign-off |

The one-sentence version: **ship the ensemble now; verify, then take back
the two smooth-data datasets using the selector your own abolished gate
killed — rebuilt as an automatic feature, because automation is the
product; audit every other verdict the dead rules produced; keep the
leads you hold; and grade yourself at every release on the whole
quality-versus-compute curve against whatever ChimeraBoost has become.**

---

## Revision R1 — post-ladder reprioritization (owner decision, 2026-07-22)

The v0.11 compute ladder against ChimeraBoost 0.20 did not achieve
dominance. The owner reviewed the decomposition and decided: **the fork
stands — no re-fork, no rebase.** The measured deficits are feature-level
with named fixes, while the engine wins at scale (fit ratios 0.23–0.76 on
the three largest datasets); re-forking would trade ~25k lines of
differentiated work and the test suite's guarantees for four portable
features. This section supersedes the earlier sequencing table.

**Ladder attribution (from the per-dataset rows):**

- The singles aggregate deficit (1.0145, W-L 7-6 in DarkoFit's favor) is
  carried by **diamonds (1.3865)** and healthcare — the categorical
  datasets — flipped by 0.20's group-centered categorical crosses.
  Excluding diamonds alone, DarkoFit leads the other twelve by ~1.2%.
- The default fit aggregate (1.38×) is a **small-dataset overhead**
  artifact: DarkoFit is 1.3–4.3× faster on every large dataset and
  4.5–9.4× slower on the three tiny ones.
- The ensemble fit gap (6.10×) is principally **B3 parallelism**
  (their members fit in parallel; ours sequentially) plus member-recipe
  cost; the ensemble quality gap (1.0363) is the diamonds deficit flowing
  through plus their broad-tuned member policy (+3.5% over their default
  vs our +1.5%).

**Revised mechanism order (one at a time, unchanged discipline):**

| # | Work | State / gate |
| --- | --- | --- |
| R1-1 | **Finish the selector campaign**: Protein attribution attempt 2 (loader preflighted after the env-failed attempt 1), guardrail replay, then close as `ready_for_powered_fresh_design` or killed | **Closed 2026-07-22: killed.** Protein improved `0.968638×` in aggregate and passed both harm gates, but coordinate 1's `0.025179` validation margin missed the frozen `0.03` engagement rule, so automatic mode declined while explicit linear helped. The artifact-only historical replay remained harm-free (`0.962739×` across 21 dependent lineages; worst lineage/split `1.0×`) but cannot reverse the terminal Protein invariant failure. No rerun, merge, fresh campaign, default, or claim. |
| R1-2 | **Categorical-crosses port/build** — the 0.20 counter, targeting diamonds/healthcare; Apache-licensed public design; develop via M6 v3 | **General development complete:** exact private candidate `c3f2608c` advanced at aggregate `0.992606×`, worst group/LOO `1.0×`; eligible only for separately frozen spent attribution and remains unmerged |
| R1-3 | **B3 parallel members** — tests whether member parallelism can remove the 6.10× ensemble-fit optic under the §4.3 memory rules | **Closed 2026-07-23: killed.** The exact private `7x2` candidate was behavior-exact, passed memory, improved the cold equal-case aggregate to `0.684187×`, and improved steady fit to `0.260379×`, but the cold Friedman case regressed to `1.075049×` and failed the frozen all-case direction gate. No rerun or merge. |
| R1-4 | **Powered fresh Tier-D panel design and execution** — one shared design effort (≥80% power, Panel 3's lesson) serving the queued T7b automatic-depth policy (`eligible_for_fresh_tier_d_design`); the killed selector is excluded | **Design complete and power-qualified 2026-07-23:** frozen 32-lineage template passed at `0.998000` simulated power with one-sided Wilson lower `0.996657` under the preregistered 20%-retained log-effect scenario. **Owner-authorized for registry freeze, contamination review, target preflight, and one one-shot fresh confirmation on 2026-07-23.** A terminal GO authorizes public-default promotion for v0.12; a NO-GO closes the default candidate while leaving P3 untouched. Release publication remains separately gated. |
| R1-5 | **Member-policy retune on broad data** via M6 v3 (their blessed member recipe is public; ours was sports-attributed and transferred at only +1.5% broad) | Track I, next after R1-3 |
| R1-6 | **Small-dataset fit fast-path** (the 4.5–9.4× tiny-data overhead) | Track I, engineering |
| R1-7 | T7b automatic-L2 | Closed in M6; stays dead |

**Standing additions:**

- **Rival-changelog triage at every milestone**: read their CHANGELOG,
  triage each shipped idea into Track I with an expected-value note.
  (0.21.0 is already out: bagged-member categorical-transform sharing at
  predict and a parent-thread-restore fix — the predict items are
  directly relevant to R1-3/R1-5.)
- **Re-fork tripwires (falsifiable, evaluated at each milestone):**
  reopen the re-fork question only if (a) a matched-config large-n
  engine comparison against their latest shows their engine ahead at
  scale, or (b) the unported-feature backlog grows across two
  consecutive rival releases despite triage. Neither is true today.
