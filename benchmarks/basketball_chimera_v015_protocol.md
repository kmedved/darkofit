# Basketball DarkoFit versus ChimeraBoost 0.15.0 characterization protocol

## Question and decision boundary

This campaign measures the remaining same-machine basketball gap after
DarkoFit's fused histogram/split and small-row serial-descent ports. It keeps
two questions separate:

1. How do the current public regression defaults compare on the creator folds
   and corrected player guardrails?
2. At the same 1,000-tree budget and common core hyperparameters, is there a
   material remaining fit-engine gap?

This is a characterization, not a product-policy or default-promotion gate.
It cannot authorize automatic early stopping, linear leaves, cross features,
an ensemble, or any other default change. Basketball remains the first fast
screen for a separately frozen candidate, and no CTR23 data is used.

## Frozen source and data

- DarkoFit source before this protocol:
  `041101187fbc68f5648df63724fe901fabdd42f1`.
- ChimeraBoost 0.15.0 source:
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d` (`v0.15.0`).
- Both repositories must be clean, on `main` or the committed campaign branch,
  and the ChimeraBoost checkout must equal its local `origin/main` and
  `upstream/main` when those refs exist.
- The creator CSV, 5,241-row training view, 15 features, unshuffled ten folds,
  and fingerprints remain those enforced by `basketball_harness.py`.
- The overlap-exposed 2,409-row held-team view, 1,824 seen-player subset, and
  585 genuinely cold-player subset are scored separately. They are never
  blended into creator-fold R².

Both arms run at 18 threads, random state 4, no sample weights, and no outer
parallel cross-validation. Every model receives the same train/test indices.

## Lane A: current product defaults

Only execution and reproducibility controls are explicit:

- DarkoFit: `DarkoRegressor(random_state=4, thread_count=18)`;
- ChimeraBoost: `ChimeraBoostRegressor(random_state=4, thread_count=18)`.

DarkoFit therefore retains its fixed 1,000-round, auto-learning-rate,
no-early-stopping policy. ChimeraBoost retains its 2,000-round ceiling,
automatic validation split and early stopping, 100-round candidate auditions,
validation-selected linear leaves and cross features, and current defaults.
Tree counts and selection outcomes must be recorded per fitted model.

The product-default lane reports mean and every fold R², fold wins, held-team,
seen-player and cold-player R², fit/predict/wall time, tree counts, selected
features/lanes, and peak RSS. For descriptive classification only, product
quality is called broadly comparable when absolute mean-fold R² differs by at
most 0.002 and both held-team and cold-player R² differ by at most 0.01. These
bands do not promote either policy.

## Lane B: matched constant-leaf engine

The matched lane disables policy differences and fits the same pure
constant-leaf oblivious-tree configuration:

```text
rounds=1000
learning_rate=0.1
depth=6
l2_leaf_reg=1
max_bins=128
subsample=1
colsample=1
min_child_weight=1
ordered_boosting=false
early_stopping=false
linear_leaves=false
cross_features=false
cat_combinations=false
```

DarkoFit additionally fixes `tree_mode="catboost"`,
`min_child_samples=1`, and disables diagnostic warnings. ChimeraBoost has no
separate minimum-child-sample parameter. Both models must retain exactly 1,000
trees on every creator fold and the held-team fit.

The exploratory pre-protocol check produced identical ten-fold R² values. The
formal lane therefore requires array-exact fold and guardrail predictions,
identical prediction hashes, and equal mean/fold R² before an engine-parity
claim is allowed. Feature-importance and internal archive layouts are reported
but are not cross-library equality gates because the products expose different
formats.

Engine parity requires:

- median summed fit-time ratio DarkoFit/ChimeraBoost at most 1.10;
- median steady wall-time ratio at most 1.10;
- both arms' wall max/min ratio at most 1.20;
- median fresh-worker peak-RSS ratio at most 1.10; and
- exact matched-lane predictions and 1,000 retained trees throughout.

Prediction time is diagnostic: the matched configuration reaches different
public predictor implementations even when its learned outputs are exact.

## Execution and evidence

Each lane runs three reciprocal fresh-worker blocks:

1. DarkoFit, ChimeraBoost;
2. ChimeraBoost, DarkoFit; and
3. DarkoFit, ChimeraBoost.

Each worker performs one complete first-fold fit and prediction outside the
timer, then fits and predicts the ten creator folds sequentially. Held-team
and player-subset scoring occurs after the steady ten-fold timer. Imports and
dataset loading are outside timing. Import-time ChimeraBoost warmup is forced
off so both packages use the same explicit full-fold warmup.

Within each lane and arm, all repeat behavior fingerprints must be identical.
The artifact records predictions, hashes, fold indices, fitted metadata,
source states, environment, peak RSS, reciprocal order, and raw timings.
Source state is rechecked between every worker. Publication is atomic and
create-only.

The formal runner must bind this protocol, its own normalized whole-repository
content manifest, the exact ChimeraBoost head, and both clean source states.
Any source, behavior, tree-count, prediction-exactness, or stability failure
stops the relevant claim rather than relaxing a threshold after observation.

## Stop/continue rule

- If matched-engine parity passes, stop low-level default-tree optimization as
  the explanation for the product-default gap. Re-profile only if a later
  concrete engine regression appears; pursue isolated product mechanisms for
  quality/speed trade-offs instead.
- If matched-engine parity fails, profile both matched fits at basketball
  scale and select exactly one behavior-exact hotspot with opportunity score
  at least 2.0.
- Regardless of the result, do not promote ChimeraBoost's early-stopping or
  selection policy into DarkoFit from this campaign. Prior DarkoFit basketball
  evidence remains controlling: a policy must independently preserve ordinary,
  held-team, and cold-player quality.
