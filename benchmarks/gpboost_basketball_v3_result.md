# GPBoost versus DarkoFit on spent basketball data: v3 result

## Decision

On the frozen player-disjoint 2014--2016 basketball panel, **DarkoFit has the
better aggregate RMSE in both measured lanes**. GPBoost's public default is
far cheaper because it retains only 100 trees rather than DarkoFit's 1,000,
but it is modestly worse on primary RMSE and worse on both held-team views.
At the declared near-matched maximum tree budget, GPBoost is materially worse
on all three quality views, predicts more slowly, and uses more peak RSS.

This is a spent Tier-E descriptive characterization. It is not a default,
release, Pareto, or GPBoost-random-effects claim.

## Quality

All values are the equal-lineage geometric mean of
`GPBoost RMSE / DarkoFit RMSE`; below 1.0 favors GPBoost. Every one of the
three fresh-process repeats gave the same displayed value and win/loss/tie
counts. The raw artifact retains the three separate prediction fingerprints.

| Lane | Primary | Held team | Cold player | Lineage context |
| --- | ---: | ---: | ---: | --- |
| Public defaults | 1.006029× | 1.028571× | 1.020030× | GPBoost won 5/9 primary lineages, but DarkoFit won the aggregate; DarkoFit won 6/9 held and cold lineages. |
| Near-matched tree budget | 1.056729× | 1.087318× | 1.074953× | DarkoFit won 8/9 primary and cold lineages, and all 9 held-team lineages. |

The public-default comparison is necessarily a compute-policy comparison:
GPBoost retained 100 trees, while DarkoFit retained 1,000. In the capacity
lane, both were allowed at most 1,000 trees with no validation set; GPBoost
retained 1,000 in every fit and DarkoFit retained 516--721 depending on the
fold because it stopped adding useful splits. The libraries also retain
different tree-growth and minimum-Hessian semantics, so that lane is not
prediction-exact engine parity.

## One-thread cost telemetry

All cost ratios are median `GPBoost / DarkoFit`; below 1.0 favors GPBoost.
They are single-thread measurements only.

| Lane | Fit | Predict | Steady wall | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Public defaults | 0.009887× | 0.126731× | 0.011050× | 0.666369× |
| Near-matched tree budget | 0.955725× | 1.503465× | 0.957989× | 1.661392× |

The default lane's large speed difference is therefore not evidence that one
1,000-tree engine is a hundred times faster: it is mainly the 100-versus-1,000
retained-tree policy difference. At the declared capacity alignment, GPBoost
fit/wall time was about 4% lower, but its prediction total was about 50% higher
and peak RSS about 66% higher.

## Evidence boundary

- DarkoFit: clean archive of commit `b666d7d5c6583f6629adb8ae43795286c1260d43`.
- GPBoost: installed PyPI wheel 1.7.1.1.
- Data: already-spent player-disjoint sports panel 2; nine target-season
  lineages, frozen player-disjoint folds, and held-team/seen/cold views.
- Features: the same 15 numeric columns in both arms. GPBoost received no
  `GPModel`, player/team group, coordinate, or other extra input.
- Repeats: three reciprocal fresh-worker blocks, one native thread, one
  warmup outside timing. Model/split/fitted-tree structure was identical by
  arm across repeats; full prediction hashes differed, but all reported
  aggregate metrics were identical across the three blocks.

## Artifacts

- Protocol: [`gpboost_basketball_v3_protocol.md`](gpboost_basketball_v3_protocol.md), SHA-256 `4d6cd4879016119562f025815f8cd65d199863f71a6ce9abab97fb6a5acfb87c`.
- Runner: [`bench_gpboost_basketball.py`](bench_gpboost_basketball.py), SHA-256 `f366fbe3f8068e1a0f2afa5f2e558b223b92345b1b75cfa127285436a26afb1f`.
- Raw repeat-level artifact: [`gpboost_basketball_v3_raw_20260723.json`](gpboost_basketball_v3_raw_20260723.json), SHA-256 `5bd3770decce28b7e8c9f9a106ba0357d1e1002bca82500d1dd5c1c51879a44e`.

The raw JSON retains the runner's historical generic schema name
`gpboost_vs_darkofit_basketball_v1`; its bound v3 protocol and hashes above
are the controlling execution identity. The v1 and v2 executions produced no
raw artifact and remain invalid exact-prediction attempts in the testing log.
