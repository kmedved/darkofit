# v0.11 M2 broad-panel protocol v2

Status: **draft successor source contract; no outcome may be inspected until
the machine-readable v2 contract is committed and published.**

This protocol supersedes only the resource count in v1. The v1 dry run stopped
before output creation, warmup, `run_jobs`, fitting, or outcome inspection
because that contract required 18 CPUs while the execution host exposes 14
logical, physical, and active CPUs. The create-only v1 preflight record is bound
into this successor.

The common per-job CPU and thread budget is prospectively fixed at the live host
maximum of 14 for every arm. This preserves equal-resource comparison without
pretending the current host has the historical machine's 18 cores.

Everything else is unchanged from v1: the 13 datasets and three registered
coordinates; DarkoFit 0.10.1, pinned ChimeraBoost `f14be60`, and CatBoost 1.2.10
official-default single-model arms; frozen TabArena and AutoGluon sources; the
117-job continuous balanced order; eight sequential bag folds; one-hour job
limit; one new same-arm-warmed process per outer job; exact provenance and raw
artifact binding; no resume or favorable rerun; and the equal-dataset paired
ratio, bootstrap-dispersion, per-coordinate, per-dataset, and descriptive
head-to-head reporting contract.

The pushed v2 contract commit must be the direct child of the v2 harness freeze
and add only the v2 machine-readable contract. The panel remains spent,
descriptive evidence: it cannot select a default, expose the private ensemble,
authorize TabArena-Lite, authorize a release, or access fresh/lockbox evidence.
