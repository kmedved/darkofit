# Invalid smooth group-linear-selector attempts

Two launches stopped without writing a result artifact:

1. The first used the shell's Python 3.13 interpreter, which did not have the
   optional OpenML dependency. It failed before loading a dataset.
2. The second used the declared `tabarena-darko312` environment and completed
   the default wave, but the selector's warmup failed an over-strict harness
   assertion. Early stopping and best-model retention were enabled correctly;
   a smooth candidate legitimately reached the frozen 1,000-round limit. The
   protocol requires the early-stopping configuration, not that every
   candidate must stop before the cap.

Neither launch wrote an artifact, changed a threshold, consumed a new
coordinate, or supports a result claim. In-memory output from the completed
default wave was discarded. The corrected runner accepts only the two valid
stop reasons, `early_stopping` and `max_iterations`, and the full four-wave
campaign must be rerun from clean committed source.
