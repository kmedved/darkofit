# Declared-ordinal transform microbenchmark

This development microbenchmark compares the direct pandas categorical
code path with the behavior-equivalent generic object-mapping path.
It is a mechanism check, not general quality or hardware evidence.

- Source: `7738b0af27daa9d79ad7f7e833e029d58600ac99`
- Raw SHA-256: `788314a4f0f55dcae08223c9bd9ec58668754bfe4bbcd007d5b5b479b552b165`
- Predictions bit-exact: `true`
- Equal-shape geomean fast/generic prediction ratio: `0.480737`
- Faster at every measured shape: `true`

| Rows | Fast seconds | Generic seconds | Fast / generic |
| ---: | ---: | ---: | ---: |
| 128 | 0.000306875 | 0.000566028 | 0.542156 |
| 4096 | 0.001318725 | 0.002862175 | 0.460742 |
| 65536 | 0.017344833 | 0.038996791 | 0.444776 |

The generic route is retained as the correctness fallback whenever
the incoming categorical dtype does not exactly match the fitted
declared category order.
