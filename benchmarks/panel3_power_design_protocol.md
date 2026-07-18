# Panel 3 authorization-grade power and candidate retention

> **Prospective source only.** This protocol and its implementation are frozen
> before the spent exact-policy calibration summary exists. They do not
> authorize target access, a fresh fit, promotion, or a product claim by
> themselves.

## Purpose

The spent 13-task calibration preserves each candidate's exact three-coordinate
RMSE-ratio vector. This design consumes that complete census without filtering
wins, losses, ties, disengagements, or candidate-selected lineages. It answers
one narrow question: is the already-frozen 12-task panel likely enough to pass
the already-frozen statistical confirmation gates?

The executable contract is
`benchmarks/panel3_power_design_contract.json`. The output, created only after
the calibration raw and summary artifacts exist and validate, is
`benchmarks/panel3_power_design_decision.json`.

## Frozen transport map

The calibration-to-stratum assignment is fixed from subject matter and feature
structure, not candidate results:

- smooth numeric: Airfoil, Concrete, Protein, Superconductivity;
- mixed categorical: Fiat 500, Diamonds, Food Delivery, Healthcare Expenses;
- applied/noisy: Houses, Miami Housing, QSAR-TID-11, QSAR Fish Toxicity, Wine
  Quality.

Each calibration dataset remains one indivisible ordered triplet
(`r0f0`, `r1f1`, `r2f2`). No coordinate is pooled, independently redrawn, or
recombined with another source task. The source mapping also freezes the
expected T5 size-gate applicability count (zero or three) for every spent
calibration task and a three-coordinate applicability vector for every
prospective slot. Registry construction derives that vector from the final
frozen outer-training sizes and fails closed on any coordinate drift.

The prospective panel is exactly four smooth, four mixed, and four noisy
lineages. T5's seven known below-2,000-row slots—Energy Efficiency, Forest
Fires, Garment Productivity, Sensory, Wheat, Colleges, and Baseball
Hitter—are fixed to exact `[1, 1, 1]` decline triplets.
Every applicable T5 slot samples only fully applicable calibration triplets
from its stratum. Guarded cross-features has no T5 size gate and samples all
triplets in the matching stratum. At least two distinct source tasks must
support every stratum and every applicable T5 stratum, or authorization fails.

This is an explicit exchangeability assumption: within semantic stratum and
matching T5 applicability, the spent panel is treated as an empirical proxy for
the prospective panel. It is not evidence that the assumption is true, and the
power result is not a quality claim.

## Simulation is the decision rule, not a z approximation

For each of 5,000 deterministic outer simulations:

1. sample complete calibration triplets with replacement into the twelve
   frozen prospective slots;
2. calculate the equal-dataset geometric mean, least-favorable
   leave-one-favorable-dataset-out ratio, and worst-dataset ratio;
3. run the confirmation analyzer's hierarchical bootstrap resampling
   construction:
   sample twelve lineages with replacement, then three coordinates with
   replacement within every sampled lineage occurrence; and
4. apply the same geometric-mean, bootstrap-upper, concentration, and harm
   bars as the fresh decision.

The bootstrap uses the confirmation seed, 100,000 replicates, 10,000-replicate
random-draw batches, 50-panel matrix batches, and NumPy's linear percentile.
The implementation losslessly converts the task-then-fold draw identities into
a 100,000 by 36 count matrix. Matrix multiplication evaluates the same
resample means, but its floating-point reduction order can differ from the
fresh analyzer in the last representable bit. The frozen count-matrix
arithmetic is binding for this power calculation; the fresh campaign remains
bound to its direct analyzer. This never substitutes a normal standard error
or a dataset-level z upper bound.

Power is the fraction of simulated panels passing all four statistical gates.
Both its point estimate and a preregistered one-sided Wilson lower bound must be
at least 0.80. Operational cost, completeness, integrity, and no-deviation
gates remain mandatory in the real campaign but are not claimed to be
predictable from this quality-only calibration.

## Candidate retention and multiplicity

The two candidates are first evaluated separately with the frozen
two-candidate Bonferroni gate: one-sided alpha 0.025, a 97.5th-percentile
hierarchical-bootstrap upper bound, and a one-sided 97.5% Wilson lower bound on
power.

- A candidate below 0.80 on either the power point estimate or lower bound is
  removed.
- If neither survives, stop. A singleton calculation cannot rescue one.
- If both survive, retain both. No joint-both-pass power probability is
  required; each hypothesis is independently powered and Bonferroni controls
  the fresh family.
- If exactly one survives, recompute only that already-surviving candidate
  under the prospectively frozen singleton fallback: one-sided alpha 0.05,
  95th-percentile hierarchical bootstrap, and a one-sided 95% Wilson lower
  bound. It remains only if both power measures still clear 0.80.

There is no ranking, outcome-dependent tie-break, per-dataset tuning, candidate
substitution, or discretion after calibration. If primary-task target
preflight substitutes a reserve, the exact-panel power authorization no longer
applies and the campaign stops.

## Immutable handoff

The builder revalidates the calibration raw, spool, source freeze, summary,
triplets, task identities, applicability states, exact runtime, and every
source digest. It then publishes the design decision with create-only
semantics and a canonical self-hash. The decision embeds the complete
three-row pre-H1 target-statistic exclusion ledger. Prospective validation
requires those bytes to match the current declarations; historical validation
uses only the embedded ledger and still rejects an exposed task or lineage, or
a missing same-stratum replacement. Historical registry validation separately
enforces the excluded dataset IDs against its embedded task rows.

The decision artifact may set `target_preflight_authorized=true` only when at
least one preregistered candidate survives. It always leaves
`registry_build_authorized`, `confirmation_run_authorized`,
`default_promotion_authorized`, and `product_claim_authorized` false. The
target preflight, registry builder, confirmation runner, and historical
analyzer must independently validate the decision and its hashes. Absence,
schema drift, source drift, summary drift, an unsupported candidate set, or
power below 0.80 fails closed.
