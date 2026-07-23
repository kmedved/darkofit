# T7b automatic-depth shared Tier-D power-design result

The canonical design-time simulation ran on 2026-07-23 under frozen contract
`t7b-automatic-depth-shared-tier-d-power-v1-20260723` from clean, published
source `f895e480fcd2ffc117dc85fd9bd0b9bf0d492414`.

This was a sizing calculation over spent evidence and a synthetic effect
model. It did not identify, load, inspect, or fit any fresh dataset or target.

## Frozen panel template

The template requires 32 independent lineage clusters, with eight in each of
four strata: low-density numeric, low-density categorical/grouped,
high-density numeric, and high-density categorical/grouped. Each lineage has
three nested coordinates. The two changed branches (`depth_4`, `depth_8`) are
balanced, and each must have non-harmful aggregate direction.

Middle-density and classification routes remain exact no-op invariants rather
than known ties in the quality calculation. Dataset identities are not part of
this artifact; a later registry must fill the exact anonymous template or the
power result does not transfer.

## Power result

The five complete spent sizing clusters had geometric-mean ratio `0.956174`.
The primary scenario scaled every spent log effect and within-lineage
deviation to 20% of its observed magnitude, producing an implied true
geometric-mean ratio of `0.991077`. This joint shrinkage is a structured
sizing alternative, not a conservative bound or quality estimate.

| Measure | Result | Required | Status |
| --- | ---: | ---: | --- |
| Simulated pass probability | `0.998000` (4,990/5,000) | `>=0.800000` | pass |
| One-sided 95% Wilson lower bound | `0.996657` | `>=0.800000` | pass |

Component pass probabilities were `0.9984` for the point aggregate, `1.0000`
for the cluster-bootstrap upper bound, `1.0000` for leave-one-favorable-out,
`0.9996` for worst-lineage harm, and `1.0000` for both branch-direction gates.

The primary scenario's median / 95th-percentile metrics were:

| Gate metric | Median | p95 | Frozen limit |
| --- | ---: | ---: | ---: |
| Equal-lineage point ratio | `0.991060` | `0.993247` | `0.995000` |
| Lineage-bootstrap upper ratio | `0.993184` | `0.995452` | `1.002000` |
| Leave-one-favorable-out ratio | `0.991561` | `0.993732` | `0.998000` |
| Worst-lineage ratio | `1.006225` | `1.013095` | `1.020000` |

## Sensitivity and limits

| Retained spent log profile | Implied true ratio | Power | Wilson lower | Power-qualified |
| ---: | ---: | ---: | ---: | --- |
| 10% | `0.995528` | `0.217600` | `0.208156` | no |
| 15% | `0.993300` | `0.957600` | `0.952660` | yes |
| 20% (primary) | `0.991077` | `0.998000` | `0.996657` | yes |
| 25% | `0.988859` | `0.991800` | `0.989420` | yes |

The panel is therefore demonstrably powered for the preregistered primary
alternative and for the 15% sensitivity scenario, but not for effects as weak
as the 10% scenario. The high-density depth-8 assumption is an extrapolation:
the DarkoFit sizing profiles exercised depth 4. This is why the final design
requires depth 8 to pass its own direction gate rather than allowing depth 4
to carry it.

An initial direct-file command failed at Python import resolution before the
contract or simulator loaded and before any result existed. The unchanged
published source then ran successfully through its supported module entry
point: `python -m benchmarks.tier_d_fresh_power_design`. The JSON below is the
only design outcome artifact.

## Create-only artifact

- [`t7b_automatic_depth_fresh_tier_d_power_design_result_20260723.json`](t7b_automatic_depth_fresh_tier_d_power_design_result_20260723.json),
  file SHA-256
  `5b767ce0a27e09d479bb18d6314d9adce3bbac78380aeff481639b13152714ad`;
  canonical result self-hash
  `735604d24828f6294e60e023ceda053caf272095c50ae83310593833ccdd07d1`.

## Decision

The disposition is `design_power_qualified`. This completes R1-4's design
work only. Fresh access, registry construction, a confirmation run, candidate
merge, default change, release, and lockbox access all remain false.

The next possible action is an owner decision on whether to authorize a
separate execution-contract and registry freeze. That later contract must add
eligible lineage identities, contamination review, exact group-safe splits,
paired-worker execution, and harm-justified fit/predict/process-tree-RSS
budgets before any target access. The private candidate remains unmerged.
