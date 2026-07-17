# Basketball small-row serial leaf descent: promote the internal router

## Decision

Promote the private small-row leaf-descent router. The frozen campaign's
recommendation is `promote_internal_serial_descent`.

For fewer than 32,768 training rows, DarkoFit now updates oblivious-tree leaf
IDs with the exact serial kernel instead of paying for one parallel launch at
every tree level. At and above the cutoff it retains the prior parallel
kernel. This changes no public parameter, estimator default, prediction path,
model field, or archive format.

## Final basketball result

| Measurement | Forced parallel | Automatic serial | Automatic / reference |
|---|---:|---:|---:|
| Mean 10-fold R² | 0.526749518388 | 0.526749518388 | exact |
| Median fit time | 20.005s | **11.006s** | **0.550** |
| Median steady wall | 20.569s | **11.590s** | **0.563** |
| Diagnostic prediction time | 20.042ms | 20.375ms | 1.017 |
| Median fresh-worker RSS | 258.5MB | 258.7MB | 1.001 |
| Leaf-update microkernel | parallel | serial | **0.0169** |

Median fit time improved by 45.0% and steady wall time by 43.7%. Both arms
were stable in all three reciprocal blocks: reference wall max/min was 1.083
and candidate max/min was 1.130, below the frozen 1.20 limit. Prediction time
was diagnostic only because this is a training-only change; the 0.33ms median
difference has no causal prediction-path mechanism.

Relative to the already-promoted fused-kernel campaign, the two engine changes
together reduced median basketball fit time from 28.93s to 11.01s and steady
wall time from 29.46s to 11.59s, approximately 62.0% and 60.7% respectively.
The earlier synced ChimeraBoost 0.15.0 diagnostic was about 7.5s, so this
closes most, but not all, of the basketball runtime gap.

## Exactness and sports guardrails

Every creator-fold prediction, R² value, feature-importance vector, fitted
metadata payload, behavior fingerprint, serialized model byte, and archive
size matched. Mean R² remained exactly 0.526749518388.

The corrected sports guardrails also matched byte-for-byte:

| Guardrail | Rows | R² in both arms |
|---|---:|---:|
| Overlap-exposed held-team view | 2,409 | 0.531269116869 |
| Seen-player subset | 1,824 | 0.530247445885 |
| Cold-player subset | 585 | 0.500433606122 |

The canonical forced-reference worker recorded 66,000 parallel calls and zero
serial calls. The automatic worker recorded 66,000 serial calls and zero
parallel calls. Candidate instrumentation wrapped the production router's
actual kernels rather than duplicating its row-count decision.

## Verification and evidence boundary

The committed clean source first ran strict prediction goldens and the complete
test suite with inherited pytest selectors removed and third-party plugin
autoload disabled. The run produced 1,479 passes, 23 skips, and 14 warnings in
101.72s. A separate collection step proved the readable oblivious oracle,
prediction goldens, serial-kernel tests, and campaign tests were present.

The campaign then ran three reciprocal fresh-worker blocks over all ten folds
and both player guardrails. Exactness was fatal and completed before timing
confirmation. The artifact binds the protocol, package and test Git trees,
support helpers, `NOTICE`, and a normalized content manifest of every tracked
repository file. Publication was atomic and create-only. No CTR23 development
or lockbox data was used.

The evidence source was clean commit
`b44aae37497e9e45f4520307184e303735c56774` with 18 threads. The serial-twin
design and cutoff are adapted from Apache-2.0 ChimeraBoost commit
`a04430657fb82c806ee2a039506c99944a27accc`; attribution is recorded in
`NOTICE`.

## Artifacts

- `basketball_serial_leaf_descent.json`: complete clean-source evidence and
  promotion decision.
- `basketball_serial_leaf_descent_protocol.md`: frozen pre-run rules.
- `run_basketball_serial_leaf_descent.py`: attested runner and analyzer.
- `test_serial_leaf_descent.py`: direct kernel, archive, callback, refit, and
  hybrid exactness coverage.
- `test_basketball_serial_leaf_descent.py`: campaign binding, dispatch,
  prerequisite, exactness, and runtime-gate coverage.
