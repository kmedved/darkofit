# Basketball donor cross-feature screen: stop before porting

## Decision

Do not port ChimeraBoost's automatic numeric cross-feature selector or adopt
its default policy. The donor implementation failed DarkoFit's recurring first
screen: mean ten-fold basketball R² fell by `0.001042`, and cold-player R²
fell by `0.013881`. The small gain on the player-overlap-exposed team holdout
does not rescue a sports candidate that is worse on genuinely unseen players.

This is an exploratory, post-observation characterization rather than a
preregistered promotion experiment. It is sufficient to stop this port because
basketball was already designated as the primary and fatal development screen.
It is not evidence that numeric cross features are universally harmful.

## Isolated donor comparison

The screen used the synced ChimeraBoost 0.15.0 implementation at
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`. Both arms used
`ChimeraBoostRegressor(random_state=4, thread_count=18)` and all of its
product defaults, including validation-selected linear leaves. The only
change was:

- control: `cross_features=False`;
- candidate: `cross_features=None`, ChimeraBoost's automatic selector.

The dataset, `MP > 500` filter, 15 numeric features, `MPG` target, unchanged
unshuffled ten folds, alphabetical team holdout, and 585-row cold-player
subset all came from the existing basketball harness.

| View | Crosses off R² | Automatic crosses R² | Candidate delta |
| --- | ---: | ---: | ---: |
| Mean of ten creator folds | **0.528023** | 0.526981 | **-0.001042** |
| Overlap-exposed team holdout | 0.533946 | **0.534423** | +0.000476 |
| Seen-player subset | 0.532460 | **0.537525** | +0.005065 |
| Cold-player subset | **0.504762** | 0.490881 | **-0.013881** |

The automatic selector activated on four of ten folds. Only one of those four
external folds improved. The selected-fold deltas were `-0.010885`,
`-0.005850`, `+0.014121`, and `-0.007806`. This is the same failure pattern
seen with validation-selected linear leaves: a random internal validation
choice does not reliably protect the player-shifted sports boundary.

## Runtime interpretation

The sequential observations were 7.69 seconds with crosses disabled and 8.41
seconds with automatic crosses, a directional ratio of `1.093x`. There was
only one run per arm, without reciprocal ordering or fresh-process timing, so
this is not a formal runtime claim and did not affect the decision.

## Scope

- No ChimeraBoost code was copied, so no new Apache-2.0 attribution is needed.
- Do not implement the automatic numeric selector in DarkoFit.
- Do not spend a broader development panel or any lockbox evidence on this
  candidate.
- This all-numeric basketball screen did not resolve `cat_combinations`. The
  separate all-categorical mechanism was subsequently tested and closed in
  [`basketball_categorical_combinations_result.md`](basketball_categorical_combinations_result.md).
- Continue to use basketball first for the next isolated candidate, with mean
  folds, held-team, and cold-player outcomes all treated as fatal gates.

The complete recorded values, source hashes, data hashes, environment, and
limitations are in
[`basketball_cross_features_donor_screen.json`](basketball_cross_features_donor_screen.json).
The candidate arm is also preserved in the earlier source-bound
[`basketball_chimera_v015.json`](basketball_chimera_v015.json) artifact; an
independent review replayed the newly added crosses-off control and reproduced
all reported fold and player-view scores exactly.
