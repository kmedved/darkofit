# v0.11 M2 broad-panel protocol v3

Status: **draft harness-corrected successor; no outcome may be inspected until
the machine-readable v3 contract is committed and published.**

The v2 scientific protocol remains unchanged. Its dry run passed, but formal
execution stopped after writing only its manifest and terminal attestation:
the shared comparator-warmup module retained a historical constant requiring
18 threads even though v2 consistently froze and resolved 14. Zero workers and
zero fits ran. The create-only v2 preflight record is bound into this successor.

V3 makes one harness correction: throughout parent execution, fresh workers,
completion validation, and analysis, it binds the warmup module's thread
constant to the already-frozen common 14-CPU/thread budget. This changes no
dataset, coordinate, arm, seed, order, model setting, time limit, measurement,
analysis, framework pin, or decision scope from v2.

The pushed v3 contract commit must be the direct child of the v3 harness freeze
and add only the v3 machine-readable contract. The panel remains spent,
descriptive evidence: it cannot select a default, expose the private ensemble,
authorize TabArena-Lite, authorize a release, or access fresh/lockbox evidence.
