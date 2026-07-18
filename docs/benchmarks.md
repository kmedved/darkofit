# Benchmarks and evidence

DarkoFit separates diagnostic benchmarks, spent development panels, and
target-unseen confirmation evidence. A failed preregistered gate is recorded
as a failure; thresholds and frozen artifacts are not rewritten afterward.

The binding prospective decision rule is the
[shipping policy](https://github.com/kmedved/darkofit/blob/main/benchmarks/SHIPPING_POLICY.md).
Opt-in APIs and behavior-exact engine work are Tier-E; defaults and automatic
modeling policies are Tier-D. Tier-D uses uncertainty, leave-one-dataset-out
concentration, explicit harm bounds, and declared cost budgets—not win counts.

The current high-level record is in
[Benchmark notes](https://github.com/kmedved/darkofit/blob/main/BENCHMARK_NOTES.md).
The active ceiling ledger is
[Beyond parity](https://github.com/kmedved/darkofit/blob/main/BEYOND_PARITY_PLAN.md).
The release-level same-machine frontier is regenerated in
[Benchmark status](https://github.com/kmedved/darkofit/blob/main/benchmarks/benchmark_status.md).
Descriptive same-machine performance is published separately in
[Engineering measurements](measurements.md).

Key current conclusions:

- On the newer
  [player-disjoint 2014–2016 sports panel](https://github.com/kmedved/darkofit/blob/main/benchmarks/basketball_sports_panel_v2_result.md),
  DarkoFit's aggregate RMSE was 2.81% better than ChimeraBoost 0.15 across
  nine target-season lineages, with six lineage wins and three losses.
  DarkoFit's RMSE was 5.26% higher than CatBoost's, and CatBoost won all nine
  lineages. The older overlap-permitting S4 panel had produced DarkoFit wins
  over ChimeraBoost in all nine cells.
- Matched large-n fit was faster than ChimeraBoost, but the frozen `1.30x`
  claim threshold was missed (`1.2793x`).
- Public prediction medians beat ChimeraBoost in the seconds-integrated
  harness, but the conjunctive stability proof failed.
- Smooth-selector and native-ordinal promotion campaigns failed their frozen
  confirmation/development gates; neither changed defaults.

Benchmark artifacts under `benchmarks/` are provenance records. Historical
timings from different machines are directional only; same-machine paired
results are required for speed claims.
