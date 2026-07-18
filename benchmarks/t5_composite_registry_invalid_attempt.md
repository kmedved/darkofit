# T5 registry invalid attempt

The first execution of the committed T5 registry builder (`f83cf03`) stopped
without creating `t5_composite_registry.json`.

OpenML task `168887` (`CD4`) declares `Future_CD4` as the task target, but its
dataset metadata declares no default target. The shared contamination builder
requires those two independent metadata fields to agree before it will create
an opaque target-marked fingerprint:

```text
RuntimeError: task 168887 target 'Future_CD4' differs from the dataset
default target 'None'
```

This was a target-blind metadata failure. No estimator was constructed, no
prediction or metric was computed, no target statistic was inspected, and no
registry artifact was written.

The amendment replaces CD4 with the already metadata-screened OpenML task
`363204` (`dataFTR`, target `time`, dataset `46149`, R package `RISCA`). It
retains the frozen panel size, coordinate count, `smooth_numeric` allocation,
power source, candidate policy, and every decision gate.

## Second target-blind attempt

The amended builder again stopped without writing an artifact. Four nominees
had contamination reasons:

- `UCC` and `child` were repository-literal false positives. `UCC` occurred
  inside encoded evidence and `child` is a common implementation word. Neither
  had an ID, catalog-name, or semantic-fingerprint collision.
- `avocado_sales` was explicitly named by ChimeraBoost's
  `benchmarks/HIGHCARD_PLAN.md`.
- `fifa` was catalog-known and triggered a schema-deletion near-lineage alarm
  against spent task `361272`.

Repository literals shorter than six normalized alphanumeric characters are
now non-discriminating, matching the pre-existing conservative-name threshold.
This repairs the two mechanical false positives without weakening exact ID,
catalog-name, or fingerprint screening. The two genuine exposures were
replaced target-blind:

- `avocado_sales` → `std` (`363210`, KMsurv), preserving
  `mixed_categorical`; and
- `fifa` → `colrec` (`363200`, relsurv), preserving `smooth_numeric`.

No model, prediction, metric, target statistic, or registry artifact existed
at this point. Panel size, strata, coordinates, power source, candidate, and
decision rule remain unchanged.
