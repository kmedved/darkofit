# Shared Tier-D power design for T7b automatic depth

_Frozen before the canonical power simulation. This is design infrastructure,
not authorization to identify, load, inspect, or fit fresh data._

Contract identity:
`t7b-automatic-depth-shared-tier-d-power-v1-20260723`.

## Purpose and boundary

The exact private automatic-depth candidate `41e948f0` is the first DarkoFit
quality mechanism to reach `eligible_for_fresh_tier_d_design`. This protocol
sizes a reusable confirmation-panel shape and implements its statistical
decision rule. It does not name prospective datasets, build a registry, open
targets, run a fresh model, merge the candidate, change a default, or authorize
a release or claim.

The simulator is intentionally generic: a future automatic candidate can use
the same lineage-cluster bootstrap and Tier-D decision engine under a new
candidate-specific contract. No sports-specific code or dataset assumption is
embedded in the engine.

## Panel template

The proposed fresh panel contains 32 independent lineage clusters, eight in
each of four coverage strata:

- low-density numeric (`depth_4`);
- low-density categorical or grouped (`depth_4`);
- high-density numeric (`depth_8`); and
- high-density categorical or grouped (`depth_8`).

Each lineage supplies three nested coordinates, for 96 paired comparisons and
192 model fits across control and candidate. A lineage—not a coordinate—is the
independent statistical unit. At least four low-density lineages must exercise
group-safe validation, and ordinary plus nonuniform-weight coordinates must be
represented in every stratum. These are generic abstractions that admit sports
workloads without tailoring the library or the statistical code to sports.

Middle-density scalar RMSE resolves to depth 6 in both revisions;
classification and other ineligible lanes are also exact no-ops. Those routes
remain byte/exactness invariants rather than diluting quality power with known
ties. The fresh quality claim is therefore scoped to calls where the automatic
policy actually changes depth.

Dataset identities are deliberately absent. A later owner-authorized contract
must create and freeze an eligible, contamination-reviewed registry before
target access. If the final registry cannot fill this exact template, this
power decision does not transfer.

## Plausible effect used only for sizing

The sizing inputs are the complete changed-lane profiles already spent in T7b:
the two changed general regression groups and three spent sports season
clusters, including every nested coordinate. They selected the candidate and
are not independent confirmation evidence.

To account for that selection, the primary alternative scales every spent log
effect and within-lineage deviation to 20% of its observed magnitude,
shrinking the complete profile 80% toward no effect. This weakens both the
benefits and the observed harmful deviations; it is a plausible structured
alternative for sizing, **not** a one-sided conservative bound, estimate, or
promise.
Sensitivity scenarios retain 10%, 15%, and 25%.

The DarkoFit sizing profiles exercised the low-density depth-4 branch. Using
the same structured alternative for depth 8 is an explicit prospective
planning assumption, not DarkoFit evidence of a high-density win. The frozen
branch-direction gate exists so a depth-4 gain cannot carry a harmful depth-8
result through confirmation.

The alternative is conditional on true lineage ratios no worse than 1.015.
That does not waive the observed worst-lineage gate of 1.02; it states the
harm-compatible population for which power is being measured. If the real
candidate has systematic harmful lineages, the confirmation should fail
rather than be redesigned to pass.

## Exact simulated decision

For each of 5,000 deterministic outer panels, the engine:

1. samples one true log effect for every lineage from the shrunken parametric
   alternative and applies the frozen 1.015 true-effect cap;
2. samples three nested log-ratio coordinates per lineage;
3. aggregates coordinates inside each lineage;
4. applies the Tier-D equal-lineage geometric-mean, 95th-percentile
   lineage-cluster bootstrap upper bound, least-favorable
   leave-one-favorable-lineage-out, and worst-lineage gates; and
5. additionally requires both changed branches (`depth_4`, `depth_8`) to have
   non-harmful aggregate direction, so one branch cannot carry the other.

The lineage bootstrap uses 5,000 multinomial count draws with seed `20260724`.
The outer simulation uses seed `20260723`. The primary design qualifies only
if both its pass fraction and one-sided 95% Wilson lower bound are at least
0.80. Sensitivity scenarios are disclosure and cannot rescue a failed primary
design.

## What a power-qualified result means

`design_power_qualified` means only that this frozen panel template has at
least 80% simulated probability of passing the exact statistical gates under
the declared plausible alternative. It is not evidence that the candidate's
effect is real, that the transport assumptions are correct, or that costs are
acceptable.

Before any fresh run, a separate owner-authorized execution contract must
freeze the exact registry, contamination review, source pins, group-safe
splits, paired worker environment, harm-justified fit/predict/process-tree-RSS
budgets, complete analyzer, create-only artifacts, and no-rerun rule. Power
does not grant fresh access or any product authority.
