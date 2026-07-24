# Automatic linear-selector v3 noise calibration result

Source `ca2f29f` completed all 24 declared non-Protein M6 regression cells.
Six resampled-diabetes cells were below the selector's minimum sample count;
18 cells were eligible.

The predeclared one-standard-error rule engaged three cells:

- `wide_numeric_reg`, seed 1, unweighted: z `1.523139935`;
- `categorical_reg`, seed 0, unweighted: z `1.164168924`; and
- `categorical_reg`, seed 2, unweighted: z `1.974423049`.

The same dataset families declined on their other seeds and on every stressed
weighting. This is seed-fragile engagement, not a stable mechanism signature.
Per the calibration note's predeclared contingency, the selector threshold is
tightened to `2.0` standard errors before any Protein or holdout result is
read. The 2-SE rule would engage none of these calibration cells; that must be
verified by a separately named normal-software rerun.

The create-only JSON artifact has SHA-256
`626814e8522415d52085779c61935669efa3122c71313fbae8d5578735070b9f`.
This calibration records engagement statistics only. It contains no quality
outcome and supports no default or shipping claim.

## Two-standard-error verification

Source `b3c006f` reran the same 24 cells under the revised rule. All 18
eligible cells declined, all six small diabetes cells retained their exact
fallback, and the maximum z score was unchanged at `1.974423049`. Artifact
`automatic_linear_selector_v3_noise_calibration_2se_20260723.json` has
SHA-256
`d179aa46ac3787be1f50f8b6fc11ede52745e0ddae4d7ba9ede37e901db1c7da`.

The calibrated 2-SE rule is ready for spent Protein development evaluation.
No Protein or holdout outcome was used to choose it.
