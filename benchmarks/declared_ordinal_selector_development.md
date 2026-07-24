# Declared-order selector development panel

## Purpose

Test whether the native-vs-ordinal selector preserves the historical
source-declared ordinal benefit without applying it when its paired inner
validation evidence is weak. This is spent development evidence, not a
holdout or a claim about arbitrary categorical data.

## Data and splits

The panel reuses the two historical mechanism domains:

- UCI Airfoil Self-Noise: the physical attack-angle levels are represented as
  one externally ordered categorical feature;
- ggplot2 Diamonds: `cut`, `color`, and `clarity` use their published semantic
  orders.

The source files are hash-pinned in the runner. Seeds 4, 17, and 29 each make
an 80/20 development/test split, then an 80/20 train/validation split within
development. The automatic selector and early stopping see only the inner
validation rows. The outer test rows score all arms.

## Arms

All arms use the same fixed 1,000-round CatBoost-mode DarkoFit policy with
learning rate 0.1, L2 3, 128 bins, one target-stat permutation, explicit
constant leaves, and 14 requested threads:

- native categorical;
- forced declared ordinal;
- `ordinal_features="select"`.

The selector requires positive paired MSE gain at two standard errors. After
selection it refits on the complete development rows. Its final prediction
must be bit-identical to the separately fitted forced arm when it engages, or
to native when it declines.

The primary descriptive readout is the equal-dataset selector/native outer
test RMSE ratio. Per-dataset ratios, the worst coordinate, engagement count,
fit/predict telemetry, and the forced-ordinal contrast are all retained.
