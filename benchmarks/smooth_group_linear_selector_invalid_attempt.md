# Invalid smooth group-linear-selector attempts

Three launches stopped without writing a result artifact:

1. The first used the shell's Python 3.13 interpreter, which did not have the
   optional OpenML dependency. It failed before loading a dataset.
2. The second used the declared `tabarena-darko312` environment and completed
   the default wave, but the selector's warmup failed an over-strict harness
   assertion. Early stopping and best-model retention were enabled correctly;
   a smooth candidate legitimately reached the frozen 1,000-round limit.
3. The first correction used the descriptive name `max_iterations` instead
   of DarkoFit's emitted stop-reason value, `iteration_limit`, so the complete
   restart failed at the same warmup boundary.

No launch wrote an artifact, changed a threshold, consumed a new coordinate,
or supports a result claim. In-memory output from completed default waves was
discarded. The corrected runner accepts only the two actual valid stop
reasons, `early_stopping` and `iteration_limit`, and the full four-wave
campaign must be rerun from clean committed source.
