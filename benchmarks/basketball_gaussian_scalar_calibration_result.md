# Basketball Gaussian scalar-calibration result

## Decision

**Stop this mechanism at the basketball boundary.**

The existing opt-in Gaussian scalar calibration improved every probabilistic
quality gate, including genuinely cold players, but widened central 80%
intervals by roughly 2.1 times. That exceeds the preregistered 1.25 maximum.
The result therefore does not authorize a broader panel, affine or grouped
calibration, or any default-policy change. The existing explicit API remains
available.

## Result

| View | Rows | Gaussian NLL, control → candidate | Gaussian CRPS, control → candidate | 80% coverage, control → candidate | Width ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled creator folds | 5,241 | 4.31533 → 3.16191 (-26.73%) | 3.15664 → 2.98350 (-5.49%) | 47.85% → 82.69% | 2.105x |
| Held teams | 2,409 | 4.16197 → 3.15787 (-24.13%) | 3.15681 → 3.01208 (-4.58%) | 48.98% → 84.10% | 2.148x |
| Cold players | 585 | 4.04340 → 3.10508 (-23.21%) | 3.16842 → 2.98030 (-5.94%) | 47.52% → 82.74% | 2.148x |
| Seen players | 1,824 | 4.20000 → 3.17480 (-24.41%) | 3.15309 → 3.02227 (-4.15%) | 49.45% → 84.54% | 2.148x |

The scale multiplier ranged from 2.0026 to 2.1671 across creator folds and was
2.1482 for the held-team model. Candidate NLL was strictly lower on all 10
creator folds; the worst candidate/control NLL ratio was 0.80982 against a
maximum of 1.02. Point RMSE, mean predictions, and raw scores were array-exact,
and all interval crossing counts were zero.

The public `predict_dist` runtime and transient-memory checks passed. The
candidate/control median runtime ratio was 0.99809, both timing spreads were
below 1.004, and the candidate added zero maximum traced bytes. Public
distribution parameters, variance, point predictions, and interval bounds
matched the independent reconstruction array-exactly.

## Interpretation

The raw Gaussian model is severely under-dispersed on this sports task. A
single validation-fitted scale fixes most of that calibration defect and
generalizes cleanly to the external creator folds, held teams, and cold
players. However, the frozen campaign also required the correction to stay
within a 25% width increase. The learned correction needs approximately a
105–115% increase, so it fails that explicit sharpness guard.

That distinction matters: the evidence does not say scalar calibration is
ineffective. It says this scalar correction cannot meet the project's
predeclared quality-versus-sharpness contract on basketball. Changing the
width threshold after observing the result would invalidate the one-shot
screen. No retuning or rerun is permitted.

## Provenance

- Formal runtime: 90.57 seconds.
- Source: clean `main` at
  `768609b702cdf5415ab5c702a75051fdef52c114`, equal to `origin/main` when
  run.
- Frozen `darkofit/` package tree:
  `1a60b529c5f5d09920d81338406b491fb7275e3a`.
- Artifact: `basketball_gaussian_scalar_calibration.json`.
- Artifact SHA-256:
  `9972278f4afa7c94b4dce3f1b4acef52084e688fcb4ac37a1d2b697d7eb8ffd1`.
- Full verification before the formal run: 1,602 passed, 24 skipped.
- All fitted numeric state and artifact numeric values were verified finite.
- No product source changed and no ChimeraBoost code was copied.
