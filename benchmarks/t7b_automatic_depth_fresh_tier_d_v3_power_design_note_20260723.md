# P1-v3 as-built automatic-depth power result

_Dated design note, 2026-07-23. This is prospective panel-sizing evidence,
not candidate-quality evidence and not fresh-run authorization._

Contract:
`t7b-automatic-depth-fresh-tier-d-v3-power-v1-20260723`.

Create-only result:
`t7b_automatic_depth_fresh_tier_d_v3_power_design_result_20260723.json`,
file SHA-256
`d6d572e47c672262b007c436cc048b6259a753097e860357523bcec033085ba8`,
self-hash
`78e74a48e060edfe09e371a4d1b5355a684847c4c2dba16e3966ae5c6ac858c1`.

## Disposition

`design_power_qualified`.

The exact verified 32-lineage panel (17 depth 4, 15 depth 8) produced:

- primary 20%-retained scenario: `0.998000` pass probability;
- one-sided 95% Wilson lower bound: `0.996657`;
- required floor for both: `0.800000`.

Sensitivity:

| Retained spent log effect | Pass probability | Wilson lower | Floor passes |
| ---: | ---: | ---: | --- |
| 10% | 0.217600 | 0.208156 | no |
| 15% | 0.957600 | 0.952660 | yes |
| 20% primary | 0.998000 | 0.996657 | yes |
| 25% | 0.991800 | 0.989420 | yes |

The non-monotonic 20%/25% pass fraction is expected from the binding
worst-lineage harm gate: increasing the alternative strengthens aggregate
benefit but also the simulated between-lineage dispersion. Sensitivities are
disclosure and do not rescue or overturn the primary decision.

## Interpretation and authority

The numerical headline equals the old abstract-template result because the
as-built panel retains 32 independent lineages and the same simulation
method. This result is nevertheless a new, valid design decision: its exact
identities, 9/8/5/10 stratum census, 17/15 branch split, and three group-safe
lineages were all loaded and verified before the power contract froze.

The effect inputs selected the candidate, and depth-8 behavior remains a
planning transport assumption. Qualification says only that this concrete
panel has adequate simulated power under the declared alternative and exact
gates. It does not show that automatic depth works.

Every downstream authority remains false. The next step is to prepare a
single design/execution freeze over this exact registry for owner review. The
fresh one-shot, candidate merge, default promotion, and release remain
separately gated.
