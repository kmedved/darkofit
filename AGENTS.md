# AGENTS.md — working rules for agents in this repository

DarkoFit is a mechanism-led, general-purpose gradient-boosting library
(NumPy/Numba/sklearn only) in a friendly rivalry with ChimeraBoost
(`bbstats/chimeraboost`), from which it forked in 2026-05. Sports
(basketball) is the home workload; the library must stay general.

## Product philosophy: automation-first (owner directive, 2026-07-22)

The product's job is to **decide what's optimal automatically** — auto
parameters, auto selectors (linear leaves, cross features), auto kernel
dispatch, auto compute-budget configuration. Manual knobs exist as escape
hatches and research surfaces, not as the product story: a library where
everything must be configured by hand is too complicated to be the best
library.

Consequences:

- Quality mechanisms target **automatic engagement** as their end state.
  Automatic policies and defaults take the full Tier-D path
  (prospectively frozen protocol, design-time power ≥ 80%, harm bounds,
  no-rerun); that path is core product work, not overhead.
- Explicit opt-ins ship earlier under Tier-E (correctness + honest
  characterization with uncertainty; measurements, never pass/fail
  certificates), but the roadmap's destination is always the automatic
  surface.
- Behavior-exact engineering (dispatch, kernels) needs no materiality
  bar: exactness, stable direction, bounded complexity, defined
  envelope, rollback (`NEXT_STEPS.md` §4.7/§4.9).

## The goal: strict Pareto dominance over ChimeraBoost, moving target

Victory is defined on the quality-versus-compute frontier: **at any given
compute level, equal or better performance** than the *current*
ChimeraBoost release (pin refreshed per milestone), with memory and
prediction reported the same way. The whole curve must dominate; a win at
one budget does not excuse a loss at another. The scoreboard is the
milestone frontier characterization in `BEAT_CHIMERABOOST_PLAN.md` §E.
CatBoost is the quality ceiling and an idea donor, not the campaign.
TabArena is a thermometer, never a target; its first placement is
deferred by owner decision.

## Governing documents, in precedence order

1. [`benchmarks/SHIPPING_POLICY.md`](benchmarks/SHIPPING_POLICY.md) —
   claim tiers and the Tier-D decision rule.
2. [`NEXT_STEPS.md`](NEXT_STEPS.md) — gate reform: retracted gate
   classes, standing gate-design rules (§4.9), authorization-matrix
   pattern (§6).
3. [`BEAT_CHIMERABOOST_PLAN.md`](BEAT_CHIMERABOOST_PLAN.md) — current
   roadmap, victory conditions, sequencing.
4. [`COUNTERPUNCH_PLAN.md`](COUNTERPUNCH_PLAN.md) — campaign history and
   track definitions (largely executed; consult for context and frozen
   records).

## Non-negotiable discipline

- **Frozen artifacts are immutable.** Wrong decisions are superseded by
  new dated create-only records, never by editing history.
- **One mechanism at a time**, through the pipeline: profile → smallest
  private prototype → correctness invariants → M5 sentinels → M6 v2
  (quality ranking only, see caution below) → sports panel → milestone.
- **Every material run**: exact source pins (never a checkout's implicit
  HEAD — the rival repo moves), fresh workers, exclusive machine for
  timed work, create-only raw artifacts, a 12-field `TESTING_LOG.md`
  entry, no rerun-to-improve.
- **Gate design** (`NEXT_STEPS.md` §4.9): every gate names the
  user-visible harm it prevents at plausible absolute magnitudes;
  no round-number materiality bars; no rival-conditioned funding prongs;
  size ratios are telemetry, never gates.
- **Retraction completeness**: when a rule is retracted, sweep its
  historical casualties for re-adjudication (a retracted rule with
  unexamined victims is an unfinished retraction).
- **No sports-specific code**: sports needs enter only as generic
  abstractions (groups, group-safe validation/OOB, weights,
  deterministic sampling).
- **No fresh-confirmation, TabArena, or lockbox access** without
  explicit owner authorization naming the access.

## Known tripwires

- **M6 v2's backtest encodes an abolished verdict**: its known-kill
  replay is the linear-leaves selector, killed by the retracted win-count
  gate. Never pre-filter that selector's revival through M6's kill rule;
  any M6 v3 must draw expectations only from verdicts surviving the
  Phase F audit.
- **Prediction speed is shape-dependent**: DarkoFit is faster than the
  rival on the dedicated grid (0.49×) and slower on M2 panel shapes
  (1.32×). Always report both; never collapse to one number.
- **Historical protocols may pin 18 threads**; the current machine has
  14. Machine-infeasible replays record `lacks_power`, they are not
  silently skipped.
