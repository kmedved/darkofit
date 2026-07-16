# Basketball creator benchmark: initial DarkoFit baseline

## Result

DarkoFit is at practical parity with the current single-model ChimeraBoost on
this score, but it does not yet match CatBoost or the five-member ChimeraBoost
ensemble.

| Arm | Mean 10-fold R² | DarkoFit gap | DarkoFit fold wins | Author-lane wall |
|---|---:|---:|---:|---:|
| DarkoFit default | **0.5267495** | — | — | 3.81s |
| ChimeraBoost default | 0.5248259 | +0.0019236 | 5 / 10 | 5.87s |
| ChimeraBoost ensemble 5 | **0.5401588** | -0.0134093 | 1 / 10 | 7.10s |
| CatBoost default | 0.5363082 | -0.0095587 | 1 / 10 | 2.73s |

The single-model mean favors DarkoFit, but the fold evidence is a 5-5 split and
the median fold gap is only +0.000188 R². Treat this as parity, not an
established win. CatBoost and ChimeraBoost ensemble 5 each beat DarkoFit on 9
of 10 folds.

## Relation to the creator screenshot

| Arm | Creator screenshot | Frozen local run | Difference |
|---|---:|---:|---:|
| ChimeraBoost default | 0.5280233 | 0.5248259 | -0.0031974 |
| ChimeraBoost ensemble 5 | 0.5388819 | 0.5401588 | +0.0012769 |
| CatBoost default | 0.5363082 | 0.5363082 | +0.000000004 |

CatBoost's near-exact reproduction strongly supports the reconstructed
data, folds, and score. The ChimeraBoost differences reflect a different code
state and/or runtime: the frozen local comparator is upstream commit
`29602d3452b1754042006ad2b14bca320c94b4b7`, which still reports version
0.14.2 but is 40 commits beyond tag `v0.14.2`. The current commit, not the
screenshot value, is the comparator for future DarkoFit work.

Against the screenshot's single-model number, DarkoFit is 0.001274 R² behind.

## Same-machine steady timing

The steady lane warms one full fold outside timing, evaluates folds
sequentially, and gives each model all 18 logical CPUs.

| Arm | Mean 10-fold R² | Timed wall | Relative to DarkoFit |
|---|---:|---:|---:|
| DarkoFit default | 0.5267495 | 28.30s | 1.00x |
| ChimeraBoost default | 0.5248259 | 9.29s | DarkoFit is 3.05x slower |
| ChimeraBoost ensemble 5 | 0.5401588 | 52.15s | DarkoFit is 1.84x faster |
| CatBoost default | 0.5363082 | 6.60s | DarkoFit is 4.28x slower |

The parallel author-lane wall time includes Loky worker startup, imports, and
Numba cache loading, so it is useful for end-to-end throughput but not as a
single-fit speed claim. The steady lane is the better optimization diagnostic.

## Fixed target

The next optimization work should use this unchanged harness and pursue two
separate targets:

1. Quality: reach CatBoost's 0.536308 R² first (+0.009559), then the current
   ChimeraBoost ensemble's 0.540159 (+0.013409).
2. Speed: reduce DarkoFit's 28.30-second steady wall toward ChimeraBoost's
   9.29 seconds without sacrificing R².

Do not tune to individual folds or alter the dataset, feature list, split,
random seed, scoring, comparator commit, or execution controls. Candidate
selection on this one toy dataset remains exploratory; any promoted product
default still needs confirmation on unseen data.

## Artifacts

- `basketball_creator_benchmark_baseline.json`: baseline-eligible author lane,
  runner source `9b2127eac92d07cec56a0c44806eda89bf2fa51c`.
- `basketball_creator_benchmark_steady.json`: diagnostic steady lane, runner
  source `6e0863d`.
- `basketball_creator_benchmark_protocol.md`: frozen protocol and limitations.

Both runs used ChimeraBoost
`29602d3452b1754042006ad2b14bca320c94b4b7`, CatBoost 1.2.10, Python 3.12.13,
scikit-learn 1.9.0, and the pinned dataset/fold fingerprints recorded in the
JSON artifacts.

## Follow-up

The frozen five-arm DarkoFit diagnostic is recorded in
`basketball_darkofit_ablation_result.md`. No tested candidate passed its
quality, fold-breadth, held-team, and runtime gates, so the current default was
left unchanged.
