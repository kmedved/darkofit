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
