# v0.11 private ensemble evidence protocol — successor v2

_Prospectively frozen before any formal v0.11 release-candidate outcome is opened._

Contract identity: `v011-private-ensemble-evidence-v2`.

This protocol inherits every source pin, case, arm, order, reproduction band,
uncertainty rule, timing rule, claim boundary, and no-rerun rule from
`v011_ensemble_evidence_protocol.md` without change.

The v1 contract was retired before execution. A pre-outcome synthetic runtime
smoke showed that ChimeraBoost's shipped eight-member fit emits its documented
member-default warning during the unmeasured warmup. The v1 parent stores
worker stderr and its analyzer correctly rejects any nonempty stderr, so v1
could not publish a valid row. No v1 formal worker ran, no model metric or
timing outcome was opened, and no v1 raw or terminal artifact exists.

The sole v2 amendment is that warnings raised during same-case/same-arm warmup
are captured and discarded inside the worker. Warmup remains outside all
measurement. Warnings raised by the formal fit remain captured in the row and
disclosed exactly as v1 requires. Unexpected process stderr remains forbidden.

The v1 frozen contract and all v1 source files are hash-bound into v2. Any
other difference from v1 is invalid. The complete later public-exposure stop
list remains correctness failure or unresolved reproduction failure only;
performance, cost, and dispersion remain disclosures rather than gates.
