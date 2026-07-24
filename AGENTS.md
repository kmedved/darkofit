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
  Automatic defaults ship via the SHIP_RULES check: clearly better on
  dev, not worse on the holdout, revertible flag — at most one
  quality-changing automatic default per release.
- Explicit opt-ins ship on correctness + honest characterization
  (measurements with labels, never pass/fail certificates); the
  roadmap's destination is always the automatic surface.
- Behavior-exact engineering (dispatch, kernels, parallelism) needs no
  materiality bar: exactness, stable direction, bounded complexity,
  defined envelope, rollback.
- **The overhead rule (2026-07-24):** an automatic feature's disabled or
  ineligible state performs no work beyond a constant branch — no
  preprocessing, allocation, or policy resolution on the hot path; an
  engaged feature pays only its inherent transform/model cost; selection
  never recomputes unchanged base preprocessing, and in ensembles a
  selection race is paid once at the parent, not once per member.
  Automation that taxes every fit and call is how a quality win becomes
  a frontier loss (v0.12's selector: quality shipped, fit 1.38x→2.60x).

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

1. [`SHIP_RULES.md`](SHIP_RULES.md) — **the current regime** (owner,
   2026-07-23): the ship rule, the holdout, the scoreboard. The
   preregistration/Tier-D apparatus is retired; prior campaign records
   stand as history.
2. [`BEAT_CHIMERABOOST_PLAN.md`](BEAT_CHIMERABOOST_PLAN.md) +
   [`R3_PLAN.md`](R3_PLAN.md) — roadmap, victory definition, current
   sequencing ([`R2_PLAN.md`](R2_PLAN.md) is complete history).
3. [`NEXT_STEPS.md`](NEXT_STEPS.md), [`COUNTERPUNCH_PLAN.md`](COUNTERPUNCH_PLAN.md),
   [`benchmarks/SHIPPING_POLICY.md`](benchmarks/SHIPPING_POLICY.md) —
   historical context for how the evidence to date was produced.

## Working discipline (the light version)

- **Defaults ship** when clearly better on dev, not worse on the holdout,
  and revertible via a documented flag. **Opt-ins ship** on correctness +
  honest docs. Behavior-exact engineering ships on exactness tests.
- **Never tune against the holdout** (SHIP_RULES). Label dev vs holdout
  numbers everywhere they appear.
- **Benchmarks are normal software**: fix bugs and rerun; keep source
  pins (never a checkout's implicit HEAD — the rival repo moves) and
  fixed seeds; note material runs in `TESTING_LOG.md` as a lab notebook.
- **Exclusive machine for timed runs**; M5 sentinels stay on as cheap
  regression detection.
- **No sports-specific code**: sports needs enter only as generic
  abstractions (groups, group-safe validation/OOB, weights,
  deterministic sampling).
- **TabArena and the holdout are consulted deliberately** (ship-checks
  and releases), not casually — they stay meaningful by staying rare.
- **Don't rebuild the bureaucracy.** When a check fails, fix the bug and
  rerun; do not add a new governance layer. Complexity in the evidence
  machinery is a cost, not a virtue.

## Known tripwires

- **M6 v2's backtest encodes an abolished verdict**: its known-kill
  replay is the linear-leaves selector, killed by the retracted win-count
  gate. Never pre-filter that selector's revival through M6's kill rule;
  any M6 v3 must draw expectations only from verdicts surviving the
  Phase F audit.
- **Prediction speed is shape-dependent**: as of the v0.12 ladder,
  DarkoFit wins large batches (protein 0.61×) and loses small ones on
  per-call fixed overhead (up to 10.5×). Always report both ends; never
  collapse to one number.
- **Historical protocols may pin 18 threads**; the current machine has
  14. Machine-infeasible replays record `lacks_power`, they are not
  silently skipped.
- **Expectations by execution, never by hand:** any frozen expectation
  about candidate behavior (branch choices, engagement decisions,
  resolved parameters) must be generated by executing the pinned
  candidate code on the exact frozen inputs at design time — and every
  campaign freeze requires a data-free rehearsal of the full worker path
  (imports, warmup, policy resolution, integrity checks) first. The
  P1-v3 one-shot was spent by a hand-derived expectation that used the
  wrong row basis; a correct candidate failed an incorrect contract.
