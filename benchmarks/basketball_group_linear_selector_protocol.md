# Basketball group-aware linear-leaf selector gate

## Question

Can a conservative, group-aware validation selector safely decline local
linear leaves on noisy player data while preserving the exact current
basketball model?

This is the basketball-first gate for the selector authorized by the smooth
development screen. It does not add a public automatic policy.

## Frozen selector

- Source before implementation: clean, published `main` at `7241f12`.
- For every external training set, make one deterministic 20% group shuffle
  split using exact source `Player` identities and random state 4.
- Fit constant- and linear-leaf variants on the same selection rows with
  current DarkoFit defaults, OOB group rows as an explicit evaluation set,
  early stopping, no refit, and identical parameters except
  `linear_leaves`.
- Compute relative validation improvement:

  ```text
  (constant RMSE - linear RMSE) / constant RMSE
  ```

- Select linear leaves only when that value is at least `0.03`; ties and
  smaller improvements select constant leaves.
- Refit only the selected leaf type on the complete external training set
  using current defaults: 1,000 rounds and the full-data automatic learning
  rate.

The 3% margin is a development choice. On the measured design data it cleanly
separates all ten group-disjoint basketball margins (maximum 1.95%) from 20 of
21 smooth-task margins (the useful cluster begins at 3.06%). It must therefore
face this frozen basketball gate and later fresh confirmation; it cannot be
presented as an unbiased effect estimate.

`Player` is validation metadata only and is never a model feature. Selection
train and validation player sets must be disjoint.

## Fatal exactness gate

The first fresh-worker block runs current default followed by the selector on
the unchanged ten creator folds and held-team/cold-player views. The selector
passes only if:

1. it declines linear leaves on all ten folds and the held-team training fit;
2. every creator-fold and held-team prediction hash exactly matches control;
3. canonical serialized final-model state exactly matches control;
4. mean, leave-one-fold-out, held-team, seen-player, and cold-player scores
   therefore do not regress; and
5. every selection split is player-disjoint and both selection fits stop on
   the declared explicit validation set.

Failure stops immediately.

## Reciprocal cost gate

An exact survivor receives two more fresh-worker blocks:

```text
control, group_margin_selector
group_margin_selector, control
control, group_margin_selector
```

Wall, summed-fit, and summed-prediction candidate/control ratios require
paired `IQR / median <=0.10`. The selector's median wall ratio must be at most
`3.5x`, prediction ratio at most `1.25x`, and peak RSS ratio at most `2.0x`.
Warmup remains outside timing.

A pass advances the selector only to the spent smooth development
coordinates. It does not authorize a default, lockbox use, or a public
`linear_leaves=None` API.
