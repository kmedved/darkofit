# Group-centered categorical crosses v1: spent attribution

Date: 2026-07-23  
Harness: `1b2f6b6f81bcf0a7ad6c9ca593cf18684c6c1e27`  
Candidate: `c3f2608cd3033cfc00aa0737897a92ed868b5865`

## Result

The private v1 automatic selector is a large, repeatable win on Diamonds:
the automatic and forced arms were identical on all three coordinates and
achieved an equal-coordinate RMSE ratio of `0.724496` versus the constant
control. The selector engaged on all three Diamonds coordinates.

The healthcare task is below the candidate's automatic-eligibility floor:
each official M2 split has 892 training rows, versus the approximately
2,353 rows required to leave 2,000 rows after the selector's internal
validation reservation. Automatic therefore took its exact constant fallback
on all three coordinates (`1.000000`). A separately labeled forced probe,
using pairs derived from the same coordinate's full-train constant-model
importance, was slightly harmful in aggregate (`1.008072`), with coordinate
ratios `1.018144`, `0.992974`, and `1.013275`.

Across all six coordinates, automatic/control was `0.851173` and
forced/control was `0.854602`. Those pooled numbers are descriptive only:
they combine a large eligible Diamonds effect with exact healthcare fallback.

## Declared checks

- automatic/control geomean no worse than `1.000` on each dataset: **pass**;
- worst automatic/control coordinate no worse than `1.02`: **pass**
  (`1.000000`);
- automatic eligible on every target coordinate: **fail** (3/6 eligible).

The recorded disposition is
`attribution_requires_selector_successor`. In product terms, this does not
erase the Diamonds result: under `SHIP_RULES.md`, it supports continuing to
the cold-player sports guardrail for an honestly scoped opt-in, while the
automatic roadmap still needs a small-data successor before claiming coverage
of the healthcare target.

## Cost and integrity

All 6 workers and 18 arms completed from clean, hash-pinned source trees.
Every arm resolved 14 Numba threads and restored the caller's ambient mask.
Automatic fit time was `2.440972x` control on eligible Diamonds because it
includes two auditions plus the final fit. Timing is telemetry only; no
prediction-throughput, memory, archive, holdout, release-ladder, or default
claim is made.

## Immutable artifacts

| Artifact | SHA-256 |
| --- | --- |
| manifest | `3c4897e2165c769ab4cb2df8f65f515149370781d55846a1fa22c4a8f8150819` |
| launch | `4f024f8bf0fce8e5378e37602f18c5995146c7f83558990821ac16bd7c28a2df` |
| raw | `7679133a3740f6998067366a0b1205a7c19a4ff84d546bf864c450e8913dc5d4` |
| result | `c9d7f4268a3018aeb518cf215b7b1fa39532a8e268fbd33786c3cd95eeb851f4` |

The raw artifact re-analyzes exactly to the committed result.
