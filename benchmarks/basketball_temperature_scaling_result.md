# Basketball binary temperature-scaling result

## Decision

**Stop before product implementation.**

The frozen validation-fitted temperature candidate worsened probability
quality on the pooled creator folds, overlap-exposed held teams, and genuinely
cold players. It also missed the required creator-fold breadth and worst-fold
guards. No classifier calibration parameter, automatic policy, serialization
field, prediction branch, or broader campaign is authorized.

## Result

| View | Rows | Log loss, control → candidate | Brier, control → candidate | ECE, control → candidate |
| --- | ---: | ---: | ---: | ---: |
| Pooled creator folds | 5,241 | 0.541680 → 0.542589 (+0.168%) | 0.183786 → 0.184123 (+0.183%) | 0.010536 → 0.011248 (+6.76%) |
| Held teams | 2,409 | 0.557386 → 0.557779 (+0.071%) | 0.190097 → 0.190217 (+0.063%) | 0.032906 → 0.036362 (+10.50%) |
| Cold players | 585 | 0.544544 → 0.544835 (+0.053%) | 0.184273 → 0.184312 (+0.022%) | 0.056698 → 0.057781 (+1.91%) |

The candidate lowered log loss on 5 of 10 creator folds; the frozen gate
required at least 6. Its worst fold was 1.02734x control, exceeding the 1.02
limit. Accuracy, per-model ranking, ties, and held-model ROC AUC were exactly
preserved as expected from a positive scalar transform. The pooled cross-fold
ROC AUC is diagnostic only because each fold fitted a different temperature.

The operational checks passed but cannot rescue the quality failure. The
candidate/control median prediction ratio was 1.09859 against a 1.10 ceiling;
both arms met the 1.20 timing-stability limit, and the candidate added zero
maximum traced bytes in the frozen measurement.

## Interpretation

The internal calibration objectives improved, but their fitted temperatures
did not generalize across the external sports boundaries. This is the exact
failure the held-team and cold-player guards were intended to catch. The
preregistered no-rerun rule forbids changing the split fraction, bounds,
metric bins, or thresholds after observing the result.

## Provenance

- Formal runtime: 26.79 seconds.
- Source: clean `main` at
  `d81e7d1e3418feef06b6db4c363fecf4937102ab`, equal to `origin/main` when
  run.
- Artifact: `basketball_temperature_scaling.json`.
- Artifact SHA-256:
  `60f18a875525b0645cb5a792472f6f511d50e502935e1bad2d07cd0dc991a8a8`.
- All JSON numeric values were independently verified finite.
- No product source changed and no ChimeraBoost code was copied.

Basketball remains the primary fatal development screen for mechanisms that
can activate on its numeric sports features. This result supplied no evidence
about all-categorical `cat_combinations`; that separate mechanism was
subsequently tested and closed in
[`basketball_categorical_combinations_result.md`](basketball_categorical_combinations_result.md).
