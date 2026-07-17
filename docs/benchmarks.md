# Benchmarks and evidence

DarkoFit separates diagnostic benchmarks, spent development panels, and
target-unseen confirmation evidence. A failed preregistered gate is recorded
as a failure; thresholds and frozen artifacts are not rewritten afterward.

The current high-level record is in
[Benchmark notes](https://github.com/kmedved/darkofit/blob/main/BENCHMARK_NOTES.md).
The active ceiling ledger is
[Beyond parity](https://github.com/kmedved/darkofit/blob/main/BEYOND_PARITY_PLAN.md).
The release-level same-machine frontier is regenerated in
[Benchmark status](https://github.com/kmedved/darkofit/blob/main/benchmarks/benchmark_status.md).

Key current conclusions:

- DarkoFit's sports defaults beat ChimeraBoost 0.15 on all nine frozen
  multi-season cells, but CatBoost remains the quality ceiling on eight of
  nine.
- Matched large-n fit was faster than ChimeraBoost, but the frozen `1.30x`
  claim threshold was missed (`1.2793x`).
- Public prediction medians beat ChimeraBoost in the seconds-integrated
  harness, but the conjunctive stability proof failed.
- Smooth-selector and native-ordinal promotion campaigns failed their frozen
  confirmation/development gates; neither changed defaults.

Benchmark artifacts under `benchmarks/` are provenance records. Historical
timings from different machines are directional only; same-machine paired
results are required for speed claims.
