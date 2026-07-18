# Invalid smooth cross-feature attempt

The first execution of `smooth_cross_features_protocol.md` at DarkoFit commit
`8298b21` stopped before writing an artifact on grid-stability fold 5.

The v1 exactness requirement incorrectly treated fitted-tree retention as an
engine property. Both libraries produced:

- identical ordered cross pairs;
- identical selected constant/cross lanes;
- identical borders;
- an identical 2,000-round validation history;
- the same best validation RMSE, at zero-based round 1,989; and
- identical trees through that best round.

DarkoFit honored `use_best_model=True` and retained 1,990 trees when the
2,000-round iteration limit was reached. ChimeraBoost retained all 2,000 trees
because its wrapper truncates only when patience stops the booster. Their
actual predictions therefore differed by at most `0.000120588`, with test RMSE
`0.006537596` for DarkoFit's best prefix versus `0.006538016` for
ChimeraBoost's retained 2,000-tree product.

The attempt exposed a wrapper-policy difference, not a cross-feature or tree
engine difference. No result artifact was created and no campaign conclusion
was drawn. The amended protocol compares the common best-iteration prefix for
engine exactness and separately records the actual retained product prediction.

A second execution of the amended protocol reached the last coordinate,
space_ga fold 9, and stopped with no fingerprint mismatches. Both products
declined the crossed challenger. The runner incorrectly compared
ChimeraBoost's empty **selected** pair list with DarkoFit's 30 **candidate**
pairs. The schema was corrected to record candidate and selected pairs
separately and exactness now compares selected pairs, as the protocol states.
Again, no artifact was written.
