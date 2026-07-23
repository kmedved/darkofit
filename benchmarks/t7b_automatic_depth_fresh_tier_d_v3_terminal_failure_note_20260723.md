# P1-v3 automatic-depth one-shot terminal failure

_Terminal record, 2026-07-23. This note does not authorize a rerun._

## Disposition

The P1-v3 one-shot is **closed without a quality verdict**. The published
launch manifest spent the sole fresh inspection. The first fresh worker then
failed the frozen branch-integrity check, before a control arm or any paired
comparison completed. The runner published no raw or result artifact, and its
terminal attestation records one completed row as unpublished and unread.

No quality, fit-time, prediction-time, RSS, or archive gate was evaluated.
Automatic depth remains private and unpromoted. The existing P3 explicit
opt-in evidence is unaffected.

## Exact failure

The verified registry bound `airlines_departure_delay_10m` to the
`depth_8` branch using the outer training split:

- effective outer-training rows: `23,373`;
- input features: `9`;
- registry rows per feature: `2,597`, above the frozen `2,500` threshold.

The actual candidate applies its automatic policy after reserving the frozen
15% internal validation set:

- fit-time effective rows: `19,867`;
- input features: `9`;
- fit-time rows per feature: `2,207.444444`;
- resolved branch: `middle_density`;
- fitted depth: `6`.

The frozen worker required the registry's `depth_8` realization, correctly
marked the row `integrity_failed`, and stopped. This is a prospective-design
error: branch assignment was verified against the outer split rather than
the exact rows seen by the automatic policy under the frozen fit semantics.
It is not evidence for or against the candidate's predictive quality.

## Immutable artifacts

- Authorization SHA-256:
  `775cdd0d3ff2f7913470e2d2badc35cbcd1b78ce72630ed6e8be4df60baf5bda`
- Launch manifest SHA-256:
  `cb0198d3bf42224ef1ca7c2e7feed9e2145ca72d9c8f85b43544a2e6203f1b54`
- Terminal failure SHA-256:
  `10b0b225c16a3f8c1039ada13fbb4884379d4ae7c982fc3c4963f1d72c17aeae`
- Harness commit:
  `37bf561a1415cef072c767a2a5240d10849f905d`
- Candidate commit:
  `41e948f0c53b1d124e16071a7fa66eba47d084d3`
- Control commit:
  `e23d2b164f10374b1c0e02521c33fc96d48980da`

## Forward rule

There is no repair or rerun under this identity. Any successor requires a new
campaign identity, new explicit owner authorization, a contamination review,
and a genuinely fresh inspection. Before freezing, its registry must resolve
every branch from the exact post-validation fit population used by the
candidate, not from an outer-split proxy. The contaminated first lineage
cannot serve as fresh confirmation evidence again.
