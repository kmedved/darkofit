# T7b automatic-depth fresh Tier-D v3 fillability enumeration

_Authorized by `R2_PLAN.md` P1-v3. This is a pre-design resource audit, not a
frozen confirmation panel and not quality evidence._

Enumeration identity:
`t7b-automatic-depth-fresh-tier-d-v3-enumeration-v1-20260723`.

## Purpose and boundary

Execution identities v1/v2 froze a 32-slot power design before proving that
the registry could fill it. V2 correctly closed before launch when
`high_density_numeric_02` had no eligible identity. No fresh inspection,
model fit, partial read, or quality result occurred.

This successor inverts the order. It evaluates every one of the 40 concrete
OpenML identities already declared in the v1 registry independently. It does
not assign replacements to abstract slots and does not require any stratum
count. For each identity it verifies:

- exact OpenML task/dataset/name/target binding;
- successful feature/target load in the frozen `darko311` environment;
- numeric finite target without computing target statistics;
- feature-family and declared group-column feasibility;
- exact/near-lineage contamination against the published fresh and spent
  fingerprint catalogs;
- repository-history references at exact DarkoFit and ChimeraBoost pins,
  excluding only the P1 registry/enumeration disclosure files that name the
  proposed resources but contain no model outcomes;
- all three frozen row/group split coordinates and the realized automatic
  depth branch; and
- deterministic full-view and split digests.

The failed v2 preflight may have loaded some proposed identities solely for
these same value-free checks. That contact created no candidate/control
outcome and is classified as resource enumeration, not quality exposure. It
is disclosed rather than silently treated as pristine.

## Execution discipline

The harness and this protocol must be committed and published before the
enumeration runs. The run requires a clean DarkoFit checkout at that published
commit and records the exact ChimeraBoost revision. An unexpected harness,
network, or environment error fails the enumeration as a whole; it is not
misreported as a dataset rejection. Only prospective `EligibilityError`
conditions produce an ineligible identity.

The complete create-only JSON artifact may publish per-identity eligibility
and rejection reasons because this is the declared purpose of the resource
audit. It publishes no target values, target moments, model scores, or
candidate/control measurements. TabArena, CTR23 execution, every lockbox, and
all confirmation outcomes remain unopened.

## What may follow

The dated pre-design note will name only the identities that pass this audit.
A new power design may then be simulated prospectively on that exact as-built
panel. It must retain a one-sided Wilson lower bound of at least 80%; otherwise
P1-v3 stops. Design and execution may freeze together only after the owner
reviews the verified registry and power result. The fresh one-shot itself
remains separately owner-gated.
