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
