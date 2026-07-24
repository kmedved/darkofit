# Automatic linear-selector v3 noise calibration

This is spent-development calibration under `SHIP_RULES.md`, not a frozen
campaign or shipping gate. It measures the selector's paired per-row MSE-gain
z statistic on the 24 regression cells of the M6-v3 synthetic grid.

The three Protein coordinates that exposed the old `0.03` cutoff are excluded
from calibration. The runtime rule is declared before this measurement:
engage linear leaves only when their paired MSE gain is positive and at least
`1.0` standard error above zero. The run records every margin, gain, standard
error, z score, engagement decision, and fallback reason. It records no
quality outcome and cannot support a default or shipping claim.

After calibration, the rule proceeds unchanged if the non-Protein grid shows
no obviously noisy positive engagement. Otherwise the implementation is
revised from the calibration evidence before any Protein or holdout result is
read. Any later quality benchmark is a normal rerunnable benchmark, with dev
and holdout results labeled separately.
