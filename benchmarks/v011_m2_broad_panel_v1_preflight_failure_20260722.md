# v0.11 M2 broad-panel v1 preflight failure

Status: **terminal before execution; zero fits and zero campaign artifacts.**

On 2026-07-22, the published v1 contract at `301c1e0` was invoked with
`--dry-run` from its exact clean source, pinned ChimeraBoost checkout, and exact
TabArena/AutoGluon/CatBoost framework environment. The preflight stopped while
resolving resources because the v1 protocol required 18 CPUs while this host
exposes 14 logical, 14 physical, and 14 active CPUs. AutoGluon likewise resolved
14.

The failure occurred before the output directory was created, before warmup,
before `run_jobs`, and before any model fit or outcome inspection.
The v1 identity will not be retried. Its successor may change only the common
per-job CPU/thread budget to the live host maximum of 14; arms, cases, order,
seeds, time limits, measurements, analysis, and decision scope remain fixed.
