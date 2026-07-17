# Fused subset oblivious-tree result

_Run 2026-07-17 from clean `main` at `11a72a1`, under the frozen
[`fused_subset_oblivious_protocol.md`](fused_subset_oblivious_protocol.md)._

## Decision

Do not promote fused automatic dispatch for selected-feature, selected-row, or
combined subset lanes. Restore the automatic dispatch boundary to the already
shipped full-row/full-feature lanes.

The candidate was behavior-exact and materially faster in every cell, but it
failed the preregistered paired-timing stability requirement. The protocol
made that gate conjunctive and authorized no threshold change or repeat.

## What passed

- All 48 fresh workers completed.
- Candidate engagement was positive and reference engagement was zero in all
  eight cells.
- Predictions, behavior fingerprints, and canonical serialized model state
  were identical in every reference/candidate/block comparison.
- No fit or tree-build cell regressed; all eight improved materially.
- Subset-lane geometric-mean ratios were `0.5348x` fit and `0.5265x`
  tree-build time.
- Full-lane control geometric-mean ratios were `0.5088x` fit and `0.5048x`
  tree-build time.
- Every median peak-RSS ratio was below `1.05x`.

## Failed gate

The frozen stability limit was `IQR / median <= 0.15` for every paired ratio.

| Cell | Metric | Paired ratios | Median | IQR / median |
|---|---|---|---:|---:|
| Weighted RMSE, rows | Fit | 0.4529, 0.6868, 0.5607 | 0.5607 | **0.2085** |
| Weighted RMSE, rows | Tree build | 0.4472, 0.6739, 0.5502 | 0.5502 | **0.2060** |
| Weighted RMSE, both | Fit | 0.5110, 0.5986, 0.4441 | 0.5110 | **0.1512** |

Weighted-RMSE/both tree build remained inside the gate at `0.1434`. All other
fit, tree-build, and RSS stability values also passed.

This is measurement variability, not evidence of a quality or correctness
defect. It nevertheless fails the frozen promotion rule, so the subset
dispatch remains off.

## E1 disposition

- The direct subset fused kernels and exactness tests remain as bounded
  research evidence.
- Public `subsample` and `colsample` fits continue to use their existing
  selected histogram builders followed by `_best_split`.
- The previously certified full-row/full-feature unit- and variable-Hessian
  fused lanes remain enabled.
- A count-carrying oblivious variant is not pursued: `min_child_samples`
  belongs to DarkoFit's leaf-wise and hybrid builders, and adding it to
  oblivious trees would change model semantics.
- No reference histogram family can be deleted on this evidence.

E1 is closed as shaped. Reopening it requires a materially new performance
question and a new preregistered protocol, not a relaxed threshold or a repeat
of this campaign.

## Evidence

- Raw artifact:
  [`fused_subset_oblivious.json`](fused_subset_oblivious.json), SHA-256
  `ed45820d74733ebcc6fca3ed1524a49eb9d73ae7ef22925bec41e7dea22d9d01`.
- Protocol SHA-256:
  `02ff43675572a208ddca9d962fe4d6290906d841ab784cad32504b90ac64b413`.
- Runner SHA-256:
  `bf5a488124b1316236ccf263143b6532088a29d0d0f77c0bbdf66f3f7f952334`.
