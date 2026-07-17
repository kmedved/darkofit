# Basketball quantile-calibration result

## Decision

**Stop before product implementation.**

The frozen split-conformal constant-offset candidate improved calibration and
pinball loss broadly, but it failed the fatal interval-width budget on all
three decision views. No product parameter, automatic policy, serialization
field, or default change is authorized. The broader panel was not run.

## Result

| View | Rows | Coverage, control → candidate | Summed pinball, control → candidate | Mean-width ratio | Crossings, control → candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled creator folds | 5,241 | 0.6171 → 0.7790 | 2.0541 → 1.8910 | **1.3568** | 121 → 26 |
| Held teams | 2,409 | 0.6343 → 0.8244 | 2.0118 → 1.8754 | **1.4161** | 46 → 10 |
| Cold players | 585 | 0.6376 → 0.8103 | 2.0246 → 1.8802 | **1.4371** | 18 → 5 |

The candidate lowered summed pinball loss on all 10 creator folds. Its worst
fold candidate/control ratio was 0.981923, and every preregistered coverage,
pinball, crossing, finiteness, and fold-breadth gate passed. The three width
ratios exceeded the frozen maximum of 1.25, so the overall gate failed.

The result is useful but not ambiguous: a constant residual-rank correction
can repair the undercoverage of the raw quantile pair, including on genuinely
cold players, only by producing intervals that are too wide for the accepted
sports-data trade-off. The protocol forbids retuning the calibration fraction,
quantiles, thresholds, or offsets after seeing this result.

## Provenance

- Formal runtime: 33.35 seconds.
- Source: clean `main` at `ad22ef41ffd1e6df668ca864dd3be8e07bfd60e7`,
  equal to `origin/main` when run.
- Artifact: `basketball_quantile_calibration.json`.
- Artifact SHA-256:
  `a68e082efe5ea3a49d6d5a5c6acfef247fa5213a5617af52d49d1d2f0730faed`.
- All JSON numeric values were independently verified finite.
- No ChimeraBoost code was copied.

Basketball remains the primary fatal development gate for subsequent
candidates. Passing it would still be necessary, not sufficient, for a
universal default change.
