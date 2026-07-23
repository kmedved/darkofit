# SHIP_RULES — how changes ship (owner regime, 2026-07-23)

This replaces the preregistration/Tier-D apparatus for internal decisions.
The goal is a library that **performs well out of sample**, pursued with
normal ML hygiene — not trial preregistration. Prior campaign records
remain valid history; their machinery is retired.

## The ship rule

A **default or automatic behavior** ships when:

1. it is clearly better on the development suite;
2. it is not worse on the **holdout** (below); and
3. it is revertible via a documented flag, noted in the CHANGELOG.

An **opt-in** ships on correctness tests plus honest documentation.
Behavior-exact engineering ships on exactness tests. That's the whole rule.

## The holdout

One fixed set, never tuned against, consulted only at ship-checks:

- the **CTR23 lockbox** (finally opened for its actual purpose);
- the newest available sports season not used in development;
- refreshed over time as genuinely new data arrives (new seasons, new
  benchmark tasks). When a holdout set has been consulted enough times to
  feel familiar, rotate it into development and designate fresh holdout.

Dev-suite numbers and holdout numbers are labeled as such wherever they
appear. That labeling habit is retained in full.

Holdout hygiene: **after its first ship-check, CTR23 is relabeled an
observed release-validation set** — still useful, no longer pristine.
Never tune a failed candidate against a holdout result and immediately
retest; new tasks and new seasons replenish the genuinely untouched
layer. And at most **one quality-changing automatic default ships per
release**, so each holdout reading stays interpretable; behavior-exact
changes and opt-ins ride alongside freely.

## The scoreboard

The milestone compute ladder against ChimeraBoost's **current release**
(plus real usage) is the arbiter of "are we winning." The rival is an
independently moving release comparator, which makes the ladder harder to
game, but it is still not a tuning set. Run it at each release; publish
the curves.

## Benchmarks are normal software

- A bug in a harness gets fixed, and the benchmark gets rerun. Note
  material reruns in the log; no identities, no spent/fresh ledgers.
- Keep source pins and fixed seeds (cheap, and they make reruns
  meaningful). Keep the M5 sentinels as fast regression detection.
- Keep the testing log as a lab notebook, not a legal record.

## Retired

One-shot inspections, contamination bookkeeping, design-time power
analyses for internal decisions, frozen hand-derived behavior
expectations, campaign identities and supersession chains, and the
Tier-D confirmation path. `SHIPPING_POLICY.md` and the campaign records
stand as history of how the evidence to date was produced.

## What keeps us honest now

Three things: the holdout we do not tune against, the independently moving
rival used only as a release scoreboard, and labels that say what was
measured. Everything else is engineering.
