# Benchmark status

This release status is generated from frozen same-machine artifacts. Panels remain separate; it does not average unrelated datasets or reuse timings from another machine.

## General regression Pareto

Scope: 13 TabArena regression datasets, r0f0/r1f1/r2f2. Ratios use CatBoost 1.2.10 as 1.0; lower is better.

| Engine | Test RMSE | Fit | Predict | Incremental memory | Pareto |
|---|---:|---:|---:|---:|:---:|
| DarkoFit 0.9.0 | 1.0538× | 0.3729× | 1.2561× | 0.1526× | yes |
| ChimeraBoost 0.14.1 | 1.0408× | 0.4101× | 0.8765× | 0.2696× | yes |
| CatBoost 1.2.10 | 1.0000× | 1.0000× | 1.0000× | 1.0000× | yes |

All three engines remain on this four-axis frontier: CatBoost has the best quality, DarkoFit the lowest fit time and incremental memory, and ChimeraBoost the best prediction time.

## Sports Pareto

Scope: nine target-season basketball cells plus cold-player guardrail. Timing ratios use ChimeraBoost 0.15.0 as 1.0.

| Engine | Equal-cell R² | Cold-player R² | Fit | Predict | Pareto |
|---|---:|---:|---:|---:|:---:|
| DarkoFit 0.9.0 | 0.786569 | 0.793729 | 5.923× | 1.478× | no |
| ChimeraBoost 0.15.0 | 0.761923 | 0.787290 | 1.000× | 1.000× | yes |
| CatBoost 1.2.10 | 0.803272 | 0.816725 | 2.345× | 1.013× | yes |

DarkoFit beats ChimeraBoost on sports quality, but CatBoost is both more accurate and faster than DarkoFit on this panel. The failed `random_strength=0.5` candidate is excluded from the product frontier.

## Engine tracks

| Track | Observed result | Formal status |
|---|---|---|
| Large-n matched core | Darko/Chimera fit 0.7817× (1.2793× speedup); RMSE 0.99998–1.00085× | Not certified: missed the frozen 1.30× speedup threshold |
| Public prediction | 8/8 median wins, 6/8 also stable; ratios 0.805–0.987× | Not certified: two stability gates and one minimum-interval gate failed |
| Native ordinal C2 | Candidate/default RMSE 0.9928×, fit 1.0082×, predict 1.0407× | Closed in development; confirmation remained sealed |

The historical integrated-prediction JSON field `stretch_public_cases_at_or_below_chimera` counts cases that were both stable and no slower (6), despite its broader name. The eight raw median ratios show 8/8 no-slower medians. This report preserves the immutable artifact and labels both counts explicitly.

## Release conclusion

Ship the deprecation/docs/infrastructure release; do not promote a new quality policy or claim a certified all-case engine win.
