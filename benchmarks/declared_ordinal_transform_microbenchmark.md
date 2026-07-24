# Declared-ordinal transform microbenchmark

## Purpose

Measure whether directly reusing pandas categorical codes removes overhead
from the public `ordinal_features` prediction path without changing any
prediction. This is a development mechanism check, not general benchmark or
quality evidence.

## Comparison

The same fitted `DarkoRegressor` predicts the same rows in two equivalent
representations:

- ordered pandas categoricals whose category sequence exactly matches the
  fitted declaration (the new fast path);
- object columns containing the same values (the established generic mapping
  path).

The fixed grid uses 128, 4,096, and 65,536 rows; four numeric columns; three
ordinal columns of cardinality 16, 8, and 4; seed `20260723`; one model thread;
and alternating paired timing order. Predictions must be bit-identical at
every shape. A result is favorable only if the fast route is faster at every
shape and its equal-shape geomean ratio is below one.

The runner requires an exact source commit and a clean tree and refuses to
overwrite either artifact. Results are descriptive for the recorded machine.
