# Group-centered categorical crosses v1: basketball guardrail

Date: 2026-07-23  
Harness: `6d76dfae694c621a7dbf05861755d6f0a4638a94`  
Candidate: `c3f2608cd3033cfc00aa0737897a92ed868b5865`

## Result

The private automatic selector passed the spent mixed-feature basketball
guardrail:

| View | Automatic/control RMSE | Result |
| --- | ---: | --- |
| Ten creator folds, equal-fold geomean | `0.996016` | pass |
| Worst creator fold | `1.010136` | pass vs `1.020000` |
| All held-team rows | `0.996891` | pass |
| Seen-player held-team rows | `0.997861` | descriptive improvement |
| Cold-player held-team rows | `0.993971` | pass |

Automatic was eligible and selected 12 centered pairs on all 11 coordinates.
It won 8 of 10 creator folds; the two losses were `1.009995` and `1.010136`.
On the established 2,409-row held-team view, it improved both the 1,824
seen-player rows and the 585 genuinely cold-player rows.

## Scope

This benchmark used the already-spent creator data: the established 15 numeric
features plus the natural categorical fields `Pos`, categorical `Age`, `Tm`,
and derived `starter`. Player identity was used only for group-aware internal
validation and the cold-player boundary; it was never a model feature.

The result supports an honestly scoped explicit opt-in. It is not holdout,
default, fresh-confirmation, or release evidence. Together with the prior
attribution, the honest characterization is:

- large eligible categorical data can benefit materially (Diamonds
  `0.724496x`);
- the mixed sports view improves modestly, including cold players;
- the 892-row healthcare splits remain exact automatic fallback, and forcing
  the mechanism there was slightly harmful (`1.008072x`).

## Cost and integrity

All 22 workers completed from clean, hash-pinned source trees; all resolved
14 Numba threads and restored the caller's ambient mask. Automatic fit
telemetry was `1.597123x` control across coordinates because selection adds
two auditions before the final fit. The `0.860144x` single-call prediction
ratio is noisy telemetry, not a throughput claim.

## Immutable artifacts

| Artifact | SHA-256 |
| --- | --- |
| manifest | `6343e8aff8042efa7cd0be108fadba96002047bba78e7bc8a52da679982f9bae` |
| launch | `af0d1c90b21d97fa8bf24d76d238d80a58114eca114089419381fdefef7ffc40` |
| raw | `7405b6a827caf296693003705fc6c6d155dbdd18bf1f0c5bb028986af60a40c1` |
| result | `b7c0b76f32f7a66294b29497415dc533eeab67e7fcce9cbf705b3409621a7359` |

The raw artifact re-analyzes exactly to the committed result.
