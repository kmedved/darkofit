# Ensemble-v3 member-policy retune

This is a normal, rerunnable development benchmark under `SHIP_RULES.md`.
It uses the already-spent M6-v3 general slice and does not inspect a holdout.

The original roadmap phrasing became stale when v0.11 shipped the rival-derived
member policy: DarkoFit's current recipe and ChimeraBoost's public
`learning_rate=0.15`, `colsample=0.85` recipe are now the same arm. The useful
comparison is therefore:

1. `current`: policy values `0.15` and `0.85`;
2. `legacy_auto`: the former automatic learning rate with `colsample=1.0`;
3. `intermediate`: explicit `0.125` and `0.925`; and
4. `single`: a constant-leaf single-model reference, not a recipe candidate.

Every arm uses the same 10 medium M6-v3 datasets, seeds 0--2, ordinary and
stress-weighted views, 1,500-round maximum, 50-round patience, and four
threads. The eight-member ensembles use public v3 without-replacement
sampling, OOB stopping, and sequential scheduling; scheduling is
behavior-exact and fixed here so timing telemetry is interpretable. Arm order
rotates by cell.

The selected recipe is the lowest equal-cell primary-loss ratio among the
current recipe and strict improvements over it that also preserve M6-v3's
dataset-level harm (`<=1.02`) and leave-one-dataset-out concentration
(`<=1.003`) bounds. There is no win-count or arbitrary minimum-effect gate.
Fit time, prediction time, archive size, per-dataset ratios, and improvement
over the single reference are reported alongside quality.

This benchmark may update the existing explicit ensemble-v3 recipe with honest
development labeling. It cannot establish an automatic default, a holdout
result, or a claim about arbitrary unseen datasets.
