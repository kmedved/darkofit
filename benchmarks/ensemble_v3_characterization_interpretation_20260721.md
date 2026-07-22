# Ensemble-v3 characterization interpretation

_Dated, create-only interpretation of the frozen v1 result. This note does
not amend the raw artifact, generated result, contract, or claim boundary._

## Bottom line

The ensemble-v3 quality mechanism remains convincing within its spent
development evidence, and its cost is now concrete. It is a genuine explicit
quality/compute tradeoff, not a free Pareto improvement:

- quality beat matched single on all 13 historical cases, with a `0.965513x`
  equal-case loss ratio;
- current general-case fit took `6.142053x` single-model time;
- current process-tree peak RSS was `1.135581x` single;
- current safe-NPZ archives were `8.125239x` single; and
- current public prediction took `6.207940x` DarkoFit-single time and
  `3.013607x` pinned ChimeraBoost-single time across the 16-coordinate grid.

That supports only the planned additive, explicit opt-in positioning. It does
not support a default, an unqualified speed claim, or a release decision by
itself.

## Quality uncertainty and concentration

The point estimates reproduce the immutable M3b r3 readout exactly:

| View | Equal-case ratio | Wins |
| --- | ---: | ---: |
| All 13 cases | `0.965513x` | 13/13 |
| Nine player-disjoint cold-player sports cells | `0.961077x` | 9/9 |
| Four seeded 75/25 general cases | `0.975569x` | 4/4 |

The three season-cluster ratios were `0.961507x`, `0.962867x`, and
`0.958861x`. The frozen 100,000-draw cluster bootstrap was
`[0.958861x, 0.962867x]` at the 2.5th/97.5th descriptive percentiles;
leave-one-season-out results ranged from `0.960183x` to `0.962187x`.

The four-case general descriptive bootstrap was
`[0.963303x, 0.987718x]`; leave-one-case-out results ranged from
`0.970189x` to `0.981160x`. These narrow observed ranges are encouraging,
but three spent seasons and four fixed cases do not become population-level
uncertainty or 13 independent datasets.

## Current fit, process-tree memory, and archive size

Three fresh balanced blocks measured the private release candidate against
DarkoFit single on four frozen medium general tasks:

| Case | Fit ratio | Peak RSS ratio | V3 peak-RSS delta | Safe-NPZ ratio | V3 archive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Categorical multiclass | `4.051x` | `1.062x` | `22.0 MB` | `7.528x` | `1.022 MB` |
| Categorical regression | `8.584x` | `1.098x` | `14.3 MB` | `9.742x` | `0.762 MB` |
| Friedman numeric | `8.329x` | `1.121x` | `25.2 MB` | `9.849x` | `0.900 MB` |
| Numeric binary | `4.913x` | `1.273x` | `67.6 MB` | `6.035x` | `1.783 MB` |

Ratios are medians of paired blocks per case; aggregate values use an
equal-case geometric mean. RSS is the worker plus recursive children during
formal fit. The absolute delta is the median process-tree peak minus its
pre-fit start, so the peak-RSS ratio is not allowed to hide the additional
memory behind a large interpreter baseline.

The older 13-case M3b record remains useful but has a narrower self-worker RSS
scope. Recomputed v3/single equal-case ratios there were `4.658893x` fit,
`1.531222x` predict, `1.074015x` RSS, and `5.780731x` archive. Against the
then-current eight-member bootstrap control, the stored combined-arm ratios
were `0.557873x` fit, `0.753550x` predict, `0.976841x` RSS, and `0.708856x`
archive. These are adjacent workload-scoped facts, not interchangeable
denominations.

## Prediction throughput

The current grid covers four tasks by four batch sizes from 8,192 through
2,000,000 rows. Each cell has three position-balanced worker observations.

| Comparison | Equal-coordinate time ratio | At/below parity | Best | Worst |
| --- | ---: | ---: | ---: | ---: |
| DarkoFit single / ChimeraBoost 0.18 single | `0.485145x` | 16/16 | `0.309461x` | `0.742340x` |
| Ensemble-v3 / ChimeraBoost 0.18 single | `3.013607x` | 0/16 | `1.735856x` | `4.506601x` |
| Ensemble-v3 / DarkoFit single | `6.207940x` | 0/16 | `4.699394x` | `8.366665x` |

Forty-seven of 48 paired-ratio series had IQR/median at or below `0.10`; the
one exception was DarkoFit-single/ChimeraBoost on numeric-binary at 65,536
rows (`0.108104`). Nine of 144 formal intervals missed the declared `0.75 s`
duration: all were DarkoFit single at the 8,192-row batch, spanning three
tasks and all three blocks. Every 65,536-row-and-larger interval met the
minimum. The frozen aggregate includes the short intervals and is not
recomputed on a favorable subset.

This is therefore a useful repeat-series characterization, not a prediction
certificate. The small-batch DarkoFit-single ratios carry an explicit
duration limitation, while the ensemble's several-fold inference cost is
large, directionally consistent, and visible at every measured coordinate.

## Immutable evidence

- Contract SHA-256:
  `f8f7b780c6dc915926a33262e24545696754221ef310d76c01da6f9df3b00103`.
- Raw artifact SHA-256:
  `005c50a89a06e100aa95cb6a776dd7f67026786de6f261470e808a39f9310a9b`.
- Generated result JSON SHA-256:
  `5cfd7b40382187aebed43798715017e1e2867744c5c40f66a00e935f6acefeed`.
- DarkoFit source: `c5e66ef7e6bdcf5665b55b81c6b870f42d76237b`.
- ChimeraBoost source: `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`.
- Fresh or lockbox data: none.
