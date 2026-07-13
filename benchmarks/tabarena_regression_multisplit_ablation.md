# TabArena regression multisplit default-policy ablation

_Final status: exploratory screen and untouched held-out confirmation complete;
the frozen candidate advances to a broader dataset gate, while public defaults
remain unchanged. Run date: 2026-07-12._

## Status and decision

The three-split screen selected one global DarkoFit candidate for held-out
confirmation:

```text
l2_leaf_reg=1.0
max_bins=128
learning_rate=0.1
ts_permutations=1
```

All other settings remain at the corrected 0.9.0 defaults. In particular, the
candidate uses CatBoost/symmetric trees, depth 6, `min_child_weight=1`, the
corrected empty-child split semantics, a 1,000-round cap, validation early
stopping, and eight AutoGluon bagging folds.

The candidate improved test RMSE on all 12 exploratory dataset-splits and
reduced equal-dataset-weight geometric-mean RMSE by 1.67% relative to the
corrected default. It then improved the six untouched splits by 1.25%, won
19 of 24 paired comparisons, and improved all three repeat blocks. Its worst
dataset-level held-out movement was +0.12% on Diamonds, and its worst individual
split was +1.57%, both within the frozen regret guardrails.

`ts_permutations=4` was rejected from the frozen candidate. It produced a
large, repeatable Airfoil gain, but when added to the selected core candidate
it worsened Diamonds test RMSE by 0.94% on average and lost on two of three
Diamonds screen splits. That violates the categorical no-material-regret gate,
and the evidence covers only two categorical datasets.

**The candidate passes this stress-panel gate, but no package default changes
are justified yet.** These four tasks were selected from the largest residual
gaps, so the next gate must use the broader, unselected regression panel.

## Provenance and environment

The screen used the exact DarkoFit source at commit
`b89faada8d1fdae4898be09f1a57b9ad457d2b26` (`b89faad`, "Fix
empty-child handling in shared split search").

| Component | Version / setting |
| --- | --- |
| DarkoFit | 0.9.0 working-tree code at `b89faad` |
| Python | 3.12.13 |
| Platform | macOS 26.5.2, arm64 |
| NumPy | 2.4.6 |
| Numba | 0.66.0 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| AutoGluon core/common/tabular | 1.5.1b20260712 |
| TabArena | 0.0.1 source checkout at `4cd1d252` |
| OpenML | 0.15.1 |
| CPU resources | 18 detected CPUs |
| Bagging | 8 inner folds, 1 bag set, matched seeds |
| Fit policy | 1,000-round cap with validation early stopping |

The local diagnostic artifacts live under
`.cache/tabarena-regression-factorial-0.9.0-20260712/`. Core screen jobs,
core-interaction jobs, and categorical TS jobs used isolated Numba caches under
`.cache/numba-tabarena-factorial` and
`.cache/numba-tabarena-interactions`. These are local research artifacts, not
an official TabArena submission.

The committed, tidy per-split results are in
[`tabarena_regression_multisplit_ablation.csv`](tabarena_regression_multisplit_ablation.csv).
It contains all 156 local DarkoFit jobs plus the 36 matched, registered
ChimeraBoost reference rows; large prediction and model artifacts remain local.

## Tasks

The four tasks were selected because they retained some of the largest gaps
after correcting symmetric-tree empty-child legality. They are therefore a
stress-test panel, not a representative random sample of tabular problems.

| Dataset | OpenML task ID | Rows | Predictors | Categorical predictors |
| --- | ---: | ---: | ---: | --- |
| Airfoil self noise | 363612 | 1,503 | 5 | `attack-angle` (27 levels) |
| Diamonds | 363631 | 53,940 | 9 | `cut` (5), `color` (7), `clarity` (8) |
| Physiochemical protein | 363693 | 45,730 | 9 | None |
| Superconductivity | 363705 | 21,263 | 81 | None |

For reference, the corrected automatic learning rate resolved near 0.04987 on
Airfoil, 0.08751 on Diamonds, 0.08527 on protein, and 0.07561 on
superconductivity. The exact value can differ in the sixth decimal because
inner bag folds occasionally differ by one training row.

## Staged outer-split protocol

Every configuration is compared on matched OpenML outer splits. The screen and
confirmation partitions were fixed before held-out results were examined:

| Stage | Repeat/fold coordinates | TabArena split IDs | Use |
| --- | --- | --- | --- |
| Exploratory screen | `r0f0`, `r1f1`, `r2f2` | 0, 4, 8 | Select and freeze one candidate |
| Held-out confirmation | `r0f1`, `r0f2`, `r1f0`, `r1f2`, `r2f0`, `r2f1` | 1, 2, 3, 5, 6, 7 | Confirm frozen candidate only |

The screen used test RMSE for research selection. The held-out split results
must not change the candidate. If they do, those six splits become exploratory
and a new confirmation set is required.

The primary estimand is the paired log RMSE ratio
`log(RMSE_config / RMSE_default)` on the same outer split. Reported percentage
changes are its back-transform, `100 * (exp(mean(log ratio)) - 1)`. Each
dataset receives equal weight regardless of row count. Validation RMSE is a
diagnostic secondary endpoint; training and inference time are operational
secondary endpoints and must be compared only on the same machine.

## Configuration matrix

The corrected default is `l2_leaf_reg=3`, `max_bins=254`, automatic learning
rate, and one target-statistic permutation. Blank cells below retain that
default.

| Configuration | L2 | Bins | LR | TS permutations | Scope |
| --- | ---: | ---: | ---: | ---: | --- |
| Corrected default | 3 | 254 | auto | 1 | All four datasets |
| L2 only | 1 | 254 | auto | 1 | All four datasets |
| Bins only | 3 | 128 | auto | 1 | All four datasets |
| LR only | 3 | 254 | 0.1 | 1 | All four datasets |
| L2 + LR | 1 | 254 | 0.1 | 1 | All four datasets |
| Bins + LR | 3 | 128 | 0.1 | 1 | All four datasets |
| Core candidate | 1 | 128 | 0.1 | 1 | All four datasets |
| TS only | 3 | 254 | auto | 4 | Airfoil and Diamonds only |
| Bins + TS | 3 | 128 | auto | 4 | Airfoil and Diamonds only |
| LR + TS | 3 | 254 | 0.1 | 4 | Airfoil and Diamonds only |
| Core candidate + TS | 1 | 128 | 0.1 | 4 | Airfoil and Diamonds only |

This produced 108 completed outer jobs in the screen: seven core
configurations across four datasets and three splits (84 jobs), plus four TS
configurations across two categorical datasets and three splits (24 jobs).
Each outer job fit eight inner bag models.

The staged matrix prioritized interactions with a direct modeling mechanism:
learning rate with L2 regularization, learning rate with binning, TS features
with binning, and TS features with learning rate. It did not spend screen jobs
on the lower-priority L2-by-bins or L2-by-TS pairs.

## Exploratory screen results

Negative percentages are better than the corrected default. Dataset columns
are geometric means across the three matched screen splits; the aggregate
weights the four datasets equally.

| Configuration | Airfoil | Diamonds | Protein | Superconductivity | Aggregate | Test wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 only | -1.72% | -0.11% | -0.27% | -0.40% | -0.63% | 10 / 12 |
| Bins only | -2.10% | -0.19% | -0.04% | +0.25% | -0.52% | 7 / 12 |
| LR only | -1.49% | +0.25% | -0.94% | -1.12% | -0.83% | 10 / 12 |
| L2 + LR | +0.20% | +0.14% | -1.29% | -1.61% | -0.64% | 10 / 12 |
| Bins + LR | -2.43% | -0.56% | -1.04% | -1.06% | -1.27% | 12 / 12 |
| **Core candidate** | **-3.04%** | **-0.62%** | **-1.37%** | **-1.64%** | **-1.67%** | **12 / 12** |

The candidate's absolute geometric-mean screen RMSE was:

| Dataset | Corrected default | Core candidate | Change | Wins |
| --- | ---: | ---: | ---: | ---: |
| Airfoil self noise | 1.745899 | 1.692810 | -3.04% | 3 / 3 |
| Diamonds | 701.903956 | 697.585804 | -0.62% | 3 / 3 |
| Physiochemical protein | 3.772800 | 3.721227 | -1.37% | 3 / 3 |
| Superconductivity | 9.682473 | 9.524002 | -1.64% | 3 / 3 |

Validation movement broadly agreed with test movement:

| Configuration | Test RMSE change | Validation RMSE change | Test wins |
| --- | ---: | ---: | ---: |
| L2 only | -0.63% | -0.36% | 10 / 12 |
| Bins only | -0.52% | -0.66% | 7 / 12 |
| LR only | -0.83% | -0.69% | 10 / 12 |
| L2 + LR | -0.64% | -0.34% | 10 / 12 |
| Bins + LR | -1.27% | -1.18% | 12 / 12 |
| **Core candidate** | **-1.67%** | **-1.36%** | **12 / 12** |

### Categorical TS screen

The first table compares each TS configuration with the corrected default on
the six categorical dataset-splits.

| Configuration | Airfoil | Diamonds | Categorical aggregate | Wins |
| --- | ---: | ---: | ---: | ---: |
| TS only | -2.46% | -0.03% | -1.25% | 4 / 6 |
| Bins + TS | -4.27% | +0.23% | -2.05% | 4 / 6 |
| LR + TS | -3.08% | +0.17% | -1.47% | 4 / 6 |
| Core candidate + TS | -6.03% | +0.32% | -2.91% | 4 / 6 |

The decision-relevant comparison is incremental TS at the selected core
background:

| Incremental effect of TS=4 | Airfoil | Diamonds | Categorical aggregate |
| --- | ---: | ---: | ---: |
| Test RMSE change vs core candidate | -3.08% (3 / 3 wins) | +0.94% (1 / 3 wins) | -1.09% (4 / 6 wins) |
| Validation RMSE change vs core candidate | -2.22% | -0.24% | -1.23% |

The aggregate gain is driven by Airfoil. Diamonds fails the predeclared
dataset-level regret guardrail and shows a validation/test mismatch: validation
slightly favors TS while held-out screen test RMSE worsens. With only two
categorical datasets, this is not enough evidence for a categorical default
change. The frozen candidate therefore retains `ts_permutations=1`.

## Held-out confirmation

All 48 confirmation jobs completed successfully with zero failures or
imputation. Negative changes favor the frozen candidate.

| Dataset | Default geometric-mean RMSE | Candidate geometric-mean RMSE | Paired change | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| Airfoil self noise | 1.815343 | 1.779908 | -1.95% | 5 / 6 |
| Diamonds | 680.955645 | 681.740185 | +0.12% | 2 / 6 |
| Physiochemical protein | 3.760646 | 3.709162 | -1.37% | 6 / 6 |
| Superconductivity | 9.689146 | 9.516209 | -1.78% | 6 / 6 |
| **Equal-dataset aggregate** | — | — | **-1.25%** | **19 / 24** |

Validation moved in the same aggregate direction:

| Dataset | Test RMSE change | Validation RMSE change |
| --- | ---: | ---: |
| Airfoil self noise | -1.95% | -1.24% |
| Diamonds | +0.12% | +0.44% |
| Physiochemical protein | -1.37% | -1.04% |
| Superconductivity | -1.78% | -1.08% |
| **Equal-dataset aggregate** | **-1.25%** | **-0.73%** |

The direction was stable across the three repeat blocks:

| Repeat | Equal-dataset RMSE change | Candidate wins |
| --- | ---: | ---: |
| 0 | -1.39% | 7 / 8 |
| 1 | -1.33% | 6 / 8 |
| 2 | -1.03% | 6 / 8 |

The worst individual regression was +1.57% on Diamonds `r0f2`; the only other
regressions above 0.5% were Diamonds `r2f1` (+0.80%) and Airfoil `r1f2`
(+0.61%). No split breached the 2% guardrail.

| Held-out operational metric | Corrected default | Frozen candidate | Change |
| --- | ---: | ---: | ---: |
| Paired geometric-mean training time | 24.160 s | 19.458 s | -19.46% |
| Paired geometric-mean inference time | 0.0852 s | 0.0764 s | -10.32% |
| Mean observed peak CPU memory | 505.53 MiB | 505.31 MiB | -0.04% |
| Failed or imputed jobs | 0 | 0 | No change |

The timing and memory comparisons above are same-machine diagnostics. The
reported peak is process-level and includes the Python/AutoGluon baseline, so
it is useful as a no-regression check rather than a precise retained-model-size
measurement.

## Matched ChimeraBoost comparison

TabArena's registered `CHIMERA (default)` result contains non-imputed quality
measurements for the same nine outer splits on all four tasks. These rows come
from the external `tabarena-2026-06-30` suite, so RMSE is directly matched by
split but its timings are not a same-machine comparison and are excluded here.
The table uses all nine splits, combining screen and confirmation.

| Dataset | Corrected DarkoFit | Frozen candidate | ChimeraBoost | Candidate vs ChimeraBoost |
| --- | ---: | ---: | ---: | ---: |
| Airfoil self noise | 1.791893 | 1.750389 | 1.709823 | +2.37% |
| Diamonds | 687.868011 | 686.981656 | 693.492009 | -0.94% |
| Physiochemical protein | 3.764693 | 3.713179 | 3.560657 | +4.28% |
| Superconductivity | 9.686921 | 9.518806 | 9.460087 | +0.62% |
| **Equal-dataset RMSE gap** | **+3.00%** | **+1.57%** | **Reference** | **+1.57%** |

The frozen candidate narrows the four-dataset geometric-mean gap by 1.43
percentage points, but remains behind ChimeraBoost on three of four dataset
means. Both corrected default and candidate win 6 of 36 individual splits
against ChimeraBoost, all on Diamonds. The candidate is a meaningful
improvement, not regression parity.

## Decision gates

The six-split confirmation advances the candidate to a broader default-policy
gate. It cannot by itself justify a package-wide default change, because these
four datasets were selected from the largest residual gaps.

The candidate passed the frozen advancement gates:

1. All 48 held-out jobs completed with zero failures or imputation.
2. Equal-dataset held-out RMSE improved 1.25%, above the 0.5% threshold.
3. The worst dataset mean was +0.12% and the worst split was +1.57%, inside the
   0.5% and 2% regret limits.
4. Every repeat block improved by at least 1.03%.
5. Same-machine training and inference became faster, while observed peak CPU
   memory was flat.

Before any global default change ships:

1. The frozen policy must pass an external-dataset gate on the remaining,
   unselected regression tasks, with numeric and categorical coverage.
2. Classification, weighted RMSE, other losses, and other tree modes must be
   tested, or the implementation change must be narrowly scoped to the exact
   unweighted-RMSE symmetric-tree regime supported here.
3. `ts_permutations=4` requires broader categorical evidence and must remain
   opt-in based on this screen.
4. The complete Python test suite, serialization/`auto_params_` tests, and the
   TabArena smoke gate must pass.
5. Explicit parameters must continue to recover the prior behavior.

## Limitations

- The task panel is deliberately biased toward prior hard cases. Held-out
  outer splits control within-task candidate-selection bias, not task-selection
  bias.
- Only three OpenML repeats exist for Diamonds, protein, and superconductivity.
  Fold-level observations within a repeat share data and are correlated.
- Only Airfoil and Diamonds exercise categorical target statistics.
- The screen selected a three-factor combination. The staged interaction
  matrix identifies the most plausible pairwise mechanisms but is not a full
  four-factor response surface.
- The matched ChimeraBoost RMSE rows cover the exact outer splits, but their
  timings are external historical measurements and remain directional unless
  rerun on the same machine.
