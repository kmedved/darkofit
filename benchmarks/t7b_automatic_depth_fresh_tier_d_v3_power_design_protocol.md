# P1-v3 as-built automatic-depth power design

_Prospective power contract over the verified fillability-first registry. This
does not authorize a fresh confirmation run._

Contract identity:
`t7b-automatic-depth-fresh-tier-d-v3-power-v1-20260723`.

## Bound panel

The design binds the complete 32-identity eligible census from
`t7b-automatic-depth-fresh-tier-d-v3-enumeration-v2-20260723`, not an
anonymous slot template:

- 9 low-density numeric and 8 low-density categorical/grouped lineages;
- 5 high-density numeric and 10 high-density categorical/grouped lineages;
- 17 depth-4 and 15 depth-8 lineages;
- three group-safe lineages; and
- three already-attested coordinates per lineage.

The enumeration artifact, dated pre-design note, exact eligible lineage list,
and their file hashes are part of the contract. No identity may be added,
removed, reclassified, or replaced by the power result or a later execution
freeze.

## Power method

The calculation deliberately reuses the v1 method without tuning:

- the same five complete spent development clusters;
- 20% retained log effect and dispersion as the primary sizing scenario;
- 10%, 15%, and 25% retained-effect sensitivities;
- the same 1.015 true-lineage cap;
- the same equal-lineage, cluster-bootstrap, leave-one-favorable-lineage-out,
  worst-lineage, and per-branch gates;
- 5,000 outer panels and 5,000 lineage-bootstrap draws with the same seeds; and
- qualification only when both the point estimate and one-sided 95% Wilson
  lower bound are at least 80%.

Only the verified 17/15 branch composition changes. The spent effect still
comes from development that selected the candidate and exercised depth 4.
Transport to depth 8 remains a planning assumption protected by the binding
per-branch direction gate, never a quality claim.

## Boundary

This power simulation reads no prospective target or model outcome. A
qualified result permits preparation of a combined design/execution freeze
for owner review; it does not itself authorize that freeze, fresh access,
model fitting, candidate merge, default promotion, release, TabArena, CTR23,
or any lockbox.
