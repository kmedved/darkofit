# Smooth cross-feature development result

**Outcome:** the diff/product mechanism is implemented correctly and is large
enough to nominate for T5, but its raw validation selector is not safe enough.
A development-derived 5% validation-improvement engagement margin removes the
observed tail harm while retaining a 4.09% equal-dataset improvement. The
margin remains a nominee, not a default; it must face fresh T5 confirmation.

## Evidence bindings

- Frozen raw file SHA-256:
  `b5544bc598601862c443c237214124007ae49b72b11e2cf2888f03112450d30c`
- Frozen margin-analysis file SHA-256:
  `0e115192938137c8bb713f5ca533c84bad315460aae12a900b32603970ad0190`
- Original raw-run source commit:
  `f6d7983f537f9995739dbfd327b31e97f28cd747`
- Original run-time runner SHA-256:
  `473f82baaf25d692ffc298f563b5a290b6f38089c5705d268044a79e9570f308`
- Original margin-analyzer source commit:
  `da5e2d313e522fb9da0abe3c93853bbc8a052512`
- Original margin-analyzer SHA-256:
  `0c370e3380857bd86e2c632a2c26632f1c36f7e4893b6be9f865b76d02617e85`
- Current hardened source commit:
  `816101476bb65cf5a0e2f59cd11edaf96f46a1cc`
- Current hardened runner/verifier SHA-256:
  `4e831fa0ff26f7c64b4e130259d1a3fcb565b51c0310ab5ee0bd2a8da7a248eb`
- Current hardened margin-analyzer SHA-256:
  `dbb233f6d4d9776881bdfeaa839480c8b3e353a75a8ae60233c7da1b5a1463da`

The original hashes identify the code bytes that produced the frozen raw
screen and its margin derivation. The current hashes identify later hardened
copies used to revalidate those existing artifacts; no benchmark outcome was
rerun or replaced.

## Evidence boundary

- Three spent CTR23 confirmation tasks: grid stability, kin8nm, and space_ga.
- Repeat 0, folds 3–9, sample 0: 21 coordinates.
- No lockbox or fresh data.
- Full-budget constant/linear selection followed by a top-six numeric
  diff/product challenger.
- Raw artifact:
  [`smooth_cross_features.json`](smooth_cross_features.json).
- Margin derivation:
  [`smooth_cross_margin_analysis.json`](smooth_cross_margin_analysis.json).
- Invalid attempts and their corrections:
  [`smooth_cross_features_invalid_attempt.md`](smooth_cross_features_invalid_attempt.md).

Protocol erratum: the frozen protocol's final sentence says the analyzer
reports RSS. The runner did not record RSS, so no RSS measurement or claim
exists for this screen. The protocol bytes remain unchanged to preserve the
raw artifact's recorded hash; this correction does not affect the decision.

## Implementation result

DarkoFit's target-free external augmentation is exact to ChimeraBoost 0.15's
native full-budget cross implementation on all 21 coordinates at the common
best-validation prefix:

- same constant/linear lane;
- same cross decision and ordered selected pairs;
- same borders and complete validation history;
- same trees through the best round; and
- byte-identical best-prefix predictions.

One coordinate reached the 2,000-round cap with its best score at round 1,990.
DarkoFit honored `use_best_model=True`; ChimeraBoost retained all 2,000 trees.
Its actual product RMSE was 1.000064× the common best-prefix RMSE. This wrapper
retention difference is recorded separately and is not treated as an engine
failure.

## Raw selector

| Dataset | Selected/base RMSE | Cross engagements | Worst split |
|---|---:|---:|---:|
| Grid stability | 0.943106 | 7/7 | 0.985272 |
| kin8nm | 0.926864 | 7/7 | 0.986411 |
| space_ga | 0.979391 | 6/7 | **1.070763** |
| Equal-dataset aggregate | **0.949535** | 20/21 | **1.070763** |

The aggregate improves by 5.05%, every dataset aggregate improves, and the
effect is not concentrated: the least favorable leave-one-dataset-out
aggregate is 0.961077. But a space_ga split regresses 7.08%, so the raw
validation selector fails the governing guarded-policy harm standard.

## Scoped nominee

The deterministic margin analyzer evaluated whole-percentage engagement
thresholds from 0% through 10%. It selected the smallest grid value with no
observed split regression:

```text
engage crosses only when
(base validation RMSE - crossed validation RMSE) / base validation RMSE >= 5%
```

| Measure | 5% nominee |
|---|---:|
| Engaged coordinates | 11/21 |
| Equal-dataset RMSE ratio | **0.959136** |
| Worst dataset ratio | **0.973684** |
| Worst split ratio | **1.000000** |
| Least favorable leave-one-out ratio | **0.971909** |

Declines are exact base-model ties, not failures. The threshold is
cost-aware—cross selection pays for a third full model fit—and sits above the
largest validation gain among the two observed harmful crossed splits
(4.63%).

## Decision

- Nominate the exact top-six diff/product policy with a 5% validation margin
  inside the fresh T5 composite.
- Keep full-budget constant/linear and cross comparisons for the T5 candidate;
  the RSSI diagnosis already showed that a 100-round linear audition can pick
  the wrong lane.
- Use capped auditions only for the existing tree-mode race unless a separate
  spent-data study establishes ordering preservation for another race.
- Declare the roughly three-fit selection cost and augmented-matrix memory
  cost in the T5 protocol.
- Do not expose or promote an automatic cross-feature default from this
  three-dataset result.
