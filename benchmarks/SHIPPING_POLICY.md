# Shipping policy: how DarkoFit decides what ships

*Adopted 2026-07-17. Supersedes ad-hoc per-campaign gate design. Companion to
`PRODUCT_OFFENSE_PLAN.md` §2 (claim tiers). Applies prospectively: it governs
new campaigns and new claims; it does not retroactively promote candidates
closed under prior frozen protocols (see §5).*

## 1. Why win-count gates are abolished

Win counts ("≥9 of 14 lineages") were a proxy for two legitimate fears:
(a) one dataset carrying the aggregate while most cases regress, and
(b) shipping noise. They fail at both, measurably:

- **They are not a significance test.** Under a fair coin, P(≥9 wins of 14)
  ≈ 21%. A real one-sided sign test at α=0.05 needs 11/14. So 9/14 neither
  certifies signal nor tracks the aggregate — it is an arbitrary number with
  the costume of rigor.
- **Ties are scored as failures.** The fresh linear-leaves selector went
  2W–0L–12T: a *perfect record on decided cases* plus twelve exact no-ops.
  The count gate read that as "2/14 — fail". A safe, selective mechanism is
  precisely what a selector is supposed to be, and the count metric is
  structurally incapable of crediting it.
- **They ignore magnitude in both directions.** +0.1% on thirteen datasets
  and −3% on one scores 13/14 (pass); −25% on one dataset and exact ties
  elsewhere scores 1/14 (fail). The first is a harm we'd regret; the second
  is free money.

The legitimate fears get real statistics instead (§3): concentration is
checked by leave-one-out aggregates, noise by uncertainty intervals, and
harm by explicit worst-case bounds.

## 2. Claim tiers

- **Tier-E (opt-in APIs, presets, recipes, engineering measurements):**
  ships on exactness/correctness tests + truthful documentation of measured
  effects with uncertainty. No quality panels, no binary bars. Engineering
  facts (speed/memory ratios) are published as measurements with dispersion,
  never as pass/fail certifications.
- **Tier-D (defaults and automatic policies):** preregistered frozen
  protocol, the decision rule in §3, power analysis at design time, no-rerun
  discipline, fresh confirmation data. Unchanged in spirit; recalibrated in
  thresholds (§4).

## 3. The Tier-D decision rule

A default/automatic candidate SHIPS when all of the following hold on the
preregistered confirmation panel (equal-weight per dataset/lineage,
geometric aggregation of ratios; candidate/control < 1 is improvement):

1. **Helps overall:** point aggregate ratio ≤ the preregistered bar
   (default bar 0.995; a bar of 1.000 is allowed when the candidate's value
   is cost, robustness, or capability rather than headline quality).
2. **Confidently:** the 95% bootstrap upper bound of the aggregate ratio
   ≤ 1.002 — "we might be shipping nothing" is acceptable; "we might be
   shipping harm" is not.
3. **Not concentrated:** the aggregate still meets the bar (with a
   preregistered slack, default +0.003) after removing the single most
   favorable dataset (leave-one-out).
4. **Harm bounded**, by either route:
   - *unguarded mechanism:* worst-dataset ratio ≤ 1.02; or
   - *selection-guarded policy:* the evaluated unit is the selector, and its
     regression profile on the panel is clean (no dataset worse than 1.005);
     ties/declines are correct behavior, never failures.
5. **Costs declared:** fit/predict/RSS budgets preregistered and met, with
   stability measured on paired same-block ratios at seconds scale only.

No win counts anywhere. Sign tests may be *reported* (on decided,
non-tied cases only) as descriptive context, never as gates.

## 4. Calibration of bars

- Quality bars are set from the competitive shipping standard (ChimeraBoost
  ships default policies at ≈ −0.5% aggregate with a no-regression selection
  profile), not from round numbers. A stricter bar requires a written
  justification of why this candidate's risk profile demands it.
- Every Tier-D protocol includes a design-time power simulation against a
  *plausible* effect (the CTR23 lesson: 9 tasks could never pass a
  concentrated-effect gate; the panel, not the mechanism, failed).
- Engineering thresholds (e.g. "1.30× faster") are abolished as claim gates:
  measurements ship as measurements (Tier-E). A number like 1.28× is stated
  as 1.28× with its interval.

## 5. Transition rules (honesty constraints)

- Closed candidates are not retroactively promoted from their old artifacts:
  choosing a friendlier rule *after seeing outcomes* is the exact failure
  mode the no-rerun discipline exists to prevent.
- Closed artifacts MAY be used to *prioritize* new campaigns: a mechanism
  whose old evidence would satisfy §3 is a preferred nominee for a new
  protocol on fresh confirmation data (normally as part of the composite
  tabular candidate, `PRODUCT_OFFENSE_PLAN.md` T5).
- Tier-E reclassification is immediate: opt-in surfaces and engineering
  facts blocked only by old binary bars ship now with honest docs.
- The lockbox, contamination registries, exactness culture, and frozen
  artifacts remain untouched by this policy.
