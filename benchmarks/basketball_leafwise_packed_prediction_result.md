# Basketball leafwise packed-prediction result

_Formal result for
[`basketball_leafwise_packed_prediction_protocol.md`](basketball_leafwise_packed_prediction_protocol.md)._

## Decision

**Retain the bounded scalar leafwise packed-prediction route.**

The clean-main confirmation passed every frozen source, behavior, routing,
timing, stability, storage, and memory gate. The authorization remains narrow:
scalar `tree_mode="lightgbm"` prediction, exactly two resolved Numba threads,
at least five trees, at least 32,768 row-tree work units, and no more than
32,768 rows. It does not authorize a default model-policy change, an
unconditional packed route, or a broader A10 claim.

The immutable artifact is
[`basketball_leafwise_packed_prediction.json`](basketball_leafwise_packed_prediction.json).

## Performance

| View | Rows | Packed/reference core | Packed/reference public | Public speedup |
| --- | ---: | ---: | ---: | ---: |
| Small basketball control | 127 | 0.4811x | 0.5007x | 2.00x |
| Reserved creator fold 1 | 524 | 0.6793x | 0.6943x | 1.44x |
| Genuinely cold players | 585 | 0.6139x | 0.6286x | 1.59x |
| Overlap-exposed held teams | 2,409 | 0.5950x | 0.6020x | 1.66x |
| Repeated basketball rows | 8,192 | 0.5004x | 0.5060x | 1.98x |
| Repeated basketball rows | 32,768 | 0.9379x | 0.9368x | 1.07x |
| Fallback control | 65,536 | 1.1273x direct | 1.0206x fallback | 0.98x |
| Fallback control | 100,000 | 1.2101x direct | 1.0051x fallback | 0.99x |

The last two rows explain the bounded policy. Direct packed execution becomes
slower on large two-thread batches, while the selected public fallback remains
within the frozen 1.10 non-regression limit. Packing therefore must not become
unconditional for this tree kind.

All gated timing series were stable; the largest IQR/median was 0.0349 against
the 0.20 limit. The 32,768-row core ratio was 0.9379 against the 0.98 limit.

## Behavior and persistence

- Public candidate, independently executed per-tree loop, direct packed core,
  and final staged predictions were array-exact for every real and repeated
  view.
- Outer-selector and scalar-kernel instrumentation observed every frozen
  boundary coordinate, including large-batch fallback.
- The fitted state, 606,056-byte serialized archive, and full-training
  prediction reproduced their pre-change SHA-256 oracles exactly.
- The packed representation reproduced its pinned 2,146,616-byte manifest,
  retained cache identity, added zero persistent bytes, and added zero maximum
  traced transient bytes.
- The complete strict Python 3.12 suite passed with 1,614 tests and 24 skips;
  the Python 3.13 suite passed with 1,613 tests and 25 skips before the formal
  run.

## Provenance

- Formal source: clean `main` at
  `82ce33ad177047ebd02b5bd8304d4b245bad4685`, equal to `origin/main`.
- Frozen candidate package tree:
  `1b93ea2f52bb563e81cefc9102f4a5b0ad29308b`.
- Runtime: Python 3.12.13 on Apple M5 Max; fitted model resolved two threads.
- Frozen fit: 1,000 depth-six `NonObliviousTree` instances, learning rate 0.1,
  `l2_leaf_reg=1`, 128 bins, no selection, refit, or linear lane.
- Artifact SHA-256:
  `a5ce8df81682b374d86324c579571658a8f3069a24217ef0669d0d066a414c7b`.
- No CTR23 or lockbox data was used and no ChimeraBoost source was copied.
