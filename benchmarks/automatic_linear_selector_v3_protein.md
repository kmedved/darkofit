# Automatic linear-selector v3 Protein development check

This normal, rerunnable benchmark compares constant leaves, the automatic
2-SE selector, and explicit linear leaves on the same three spent Protein
coordinates used by the v0.11 release ladder. The direct OpenML 0.15.1 loader
must reproduce the historical split fingerprints.

This is development evidence, not a holdout or shipping certificate. The
selector is ready for the holdout ship-check when its automatic arm is better
than constant leaves in aggregate, no coordinate is worse, and every final
model is exact to the arm the selector recorded. Fit time, prediction time,
RSS, margins, and z scores are telemetry.

Protein was excluded from the preceding noise calibration. The 2-SE rule is
fixed for this run.
