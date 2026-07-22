# v0.11 evidence phase — owner authorization and end-to-end instruction

_Dated 2026-07-21. Create-only once committed. Issued pursuant to the §6
authorization matrix in [`NEXT_STEPS.md`](../NEXT_STEPS.md); this note is
the owner sign-off for Phases 0–2 below and explicitly does **not**
authorize the rows in Phase 3, which remain separate owner decisions._

Governing records: [`NEXT_STEPS.md`](../NEXT_STEPS.md) (binding decision
record), [`ensemble_v3_public_contract.md`](ensemble_v3_public_contract.md)
(frozen public contract),
[`fused_lane_dispatch_owner_promotion_20260721.md`](fused_lane_dispatch_owner_promotion_20260721.md)
(promoted dispatch),
[`m6_quality_successor_v2_backtest_result.md`](m6_quality_successor_v2_backtest_result.md)
(active quality-ranking rung), and
[`SHIPPING_POLICY.md`](SHIPPING_POLICY.md) (incorporated by reference).

Standing discipline applies to every phase: exclusive machine access for
timed work, fresh workers with same-arm warmup, paired ratios with
dispersion, exact source pins recorded, create-only raw artifacts, one
12-field [`TESTING_LOG.md`](TESTING_LOG.md) entry per material run, no
rerun-to-improve, no fresh-confirmation or lockbox access anywhere in this
document.

---

## Phase 0 — Publication and hygiene (authorized now; ~30 minutes)

1. **Fix the stale ChimeraBoost-pin test.** Identify the full-suite test
   that fails when the local ChimeraBoost checkout is not at the pinned
   hash, and convert it to skip-with-reason on pin mismatch (the same
   guard pattern as the M3a plumbing smoke). The suite must read
   green-or-skipped, never red-for-environment.
2. **Push `main`** (the seven completed commits, the test fix, and this
   note) so every Phase 1–2 record binds published hashes.
3. **Do not include** the unrelated working-tree changes (`README.md`,
   `docs/index.md`, `mkdocs.yml`, `docs/tabarena_tree_model_dossier.md`).
   They remain someone else's separate checkpoint.

## Phase 1 — Ensemble-v3 characterization (authorized now; private
surface; roughly 2–5 hours)

All work runs against the **private** release candidate through its
private helper. Nothing in this phase exposes constructor parameters,
exports, or public docs.

**1a. Reproduction check — the only quality stop rule.** Freeze, before
any execution, a tolerance under which the RC's fixed-seed per-case
results must reproduce the immutable M3b r3 combined-arm readout (declare
the exact band in the protocol — it should be near-exact, since this is
the same mechanism behind a public-semantics wrapper). Failure here means
**implementation divergence**: stop, fix, re-run. This is a correctness
check, not a quality gate.

**1b. Quality characterization with uncertainty (Tier-E, descriptive).**
From the reproduction run: pooled and per-case primary ratios vs the
matched single; **season-clustered bootstrap dispersion for the nine
sports cells** (three season clusters, the M3a method; sports view labeled
player-disjoint cold-player within held teams); dispersion plus
leave-one-case-out sensitivity for the four general cells (labeled seeded
75/25). Never present the 13 cells as independent datasets. Eight members
documented as the only evaluated recipe.

**1c. Cost characterization (telemetry, not gates).** Fit wall time vs the
single and vs the existing bootstrap ensemble; aggregate peak RSS; archive
bytes with the gate-retraction note (~5.5× single, ~0.5 MB typical).

**1d. Dedicated prediction-throughput grid.** The first repeat-series,
grid-wide inference characterization. Arms: DarkoFit single, DarkoFit v3
ensemble (private helper), ChimeraBoost single and its 8-member ensemble
at the **pinned** source, and **CatBoost 1.2.10 single** — CatBoost is
included because it, not ChimeraBoost, led DarkoFit on predict in the
historical broad panel, and inference speed is a named owner priority.
Declare the row-count × batch-size grid and thread budget in the frozen
protocol; fresh workers, warmup separation, paired medians with full
repeat series and dispersion. Tier-E measurement only — no certification,
no pass/fail.

**Pins for Phase 1:** DarkoFit = the Phase 0 pushed head. ChimeraBoost =
`f14be606b641f1bf0dc92bb14b3951f1fe631c6b` (`v0.18.0-6-gf14be60`),
declared explicitly — **never the local checkout's current HEAD**, which
has moved past the pin. CatBoost = 1.2.10.

**Pre-declared exposure stop conditions (complete list).** Public exposure
is later blocked only by: (a) a correctness failure, or (b) an unresolved
1a reproduction failure. Every other result — dispersion, predict
numbers, costs — is *disclosed*, not gated. No new bars may be invented
after seeing numbers; that is the trapdoor this program just closed.

## Phase 2 — M2 broad comparison panel (authorized now; overnight run)

The first current-version calibrated-yardstick reading.

- **Scope:** the 13 historical datasets at the exact `r0f0/r1f1/r2f2`
  coordinates, under a **new dated frozen protocol**; all prior artifacts
  preserved untouched.
- **Arms — defaults only, singles only:** DarkoFit default (pushed head:
  note this now includes the promoted dispatch on eligible fits, which is
  the shipped default being measured), ChimeraBoost default at the pinned
  `f14be60`, CatBoost 1.2.10 default. **The ensemble arm is excluded** —
  its costs live in Phase 1; this keeps M2 to one night.
- **Measurement:** shared splits and preprocessing boundaries, equal
  thread budgets, early-stopping rules as historically declared, fresh
  workers, fit and predict wall time, peak RSS, equal-dataset
  aggregation, full per-coordinate rows.
- **Reporting:** paired ratios with dispersion; per-dataset rows; a
  descriptive head-to-head win-rate supplement (Elo-style position is
  driven by head-to-head, not aggregate ratio). This spent panel
  characterizes; it cannot authorize a default change.
- **Scheduling:** must not overlap Phase 1 timed work; run overnight on
  the otherwise-idle machine; expect 4–12 hours plus ~1 hour of analysis
  and record-writing.

**Phase 2 deliverable:** one dated result answering, for the first time
against 0.18: does DarkoFit beat the pinned ChimeraBoost on the internal
broad panel (the NEXT_STEPS §0-yardstick test), and where does each
engine sit on quality/fit/predict/memory vs CatBoost.

## Phase 3 — Gated decisions (NOT authorized by this note)

Each row below requires its own explicit owner sign-off, taken with
Phases 1–2 evidence in hand:

1. **Public exposure of ensemble-v3.** Contingent only on the Phase 1
   stop conditions being clear. Mechanics per the frozen public contract:
   constructor parameters, exports, docs page, CHANGELOG.
2. **M4 TabArena-Lite placement.** One-shot, descriptive, release-cadence;
   DarkoFit's first-ever placement. Explicit authorization required —
   scheduling text elsewhere does not authorize it.
3. **v0.11 release.** Scope decision (promoted dispatch + ensemble-v3
   together, or staggered), version bump, tag, GitHub release with
   claims bounded per the plan's marketing constraints.

## Phase 4 — Post-v0.11 mechanism slots (sequenced; each separately
authorized at its turn)

One mechanism at a time, in this order, each through the restored
pipeline (synthetics/profile → M5 sentinels → **M6 v2 quality rung where
applicable** → sports → milestone):

1. **T7b quality levers** (`l2_leaf_reg`, samples-per-feature depth
   policy) — the promised quality-first slot, and the first mechanisms to
   use M6 v2's ranking (numbered, create-only, spent inspections;
   quality-only scope).
2. **B3 parallel members** — its own frozen training-speed campaign under
   the NEXT_STEPS §4.3 operational memory rule (process-tree scope, fixed
   topology and total CPU, hard absolute ceiling, ratio-or-absolute-delta
   allowance).
3. **Q local causal microprototype** — a measured packed-histogram
   microprototype at the Q0 hotspot against the **post-dispatch**
   baseline. The 13.28% screening projection may size the effort; only
   measured microprototype evidence can fund Q1. No donor condition.

## Effort summary

| Phase | Wall time | Machine posture |
| --- | --- | --- |
| 0 | ~30 minutes | Any |
| 1 | ~2–5 hours | Exclusive for timed sections |
| 2 | Overnight (4–12 h) + ~1 h analysis | Exclusive, idle machine |
| 3 | Owner decisions + ~half-day release mechanics | Any |
| 4 | Separately estimated per slot | Per-campaign |

Total evidence phase: one to two days, mostly unattended.
