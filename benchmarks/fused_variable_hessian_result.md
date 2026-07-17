# Fused variable-Hessian oblivious-tree result

_Run 2026-07-17 from clean `main` at `1016e7e`, under the frozen
[`fused_variable_hessian_protocol.md`](fused_variable_hessian_protocol.md)._

## Decision

Retain the internal fused variable-Hessian lane for eligible binary Logloss
and weighted RMSE fits. It changes no public parameter or model policy.

Both cases produced identical public prediction hashes and canonical
serialized model-state hashes across all reference and candidate workers.
The candidate engaged on every eligible tree level; the reference arm never
engaged.

## Performance

| Case | Fit ratio | Tree-build ratio | Peak-RSS ratio | Exact | Stable |
|---|---:|---:|---:|---:|---:|
| Binary Logloss | 0.7871× | 0.7673× | 0.9908× | Yes | Yes |
| Weighted RMSE | 0.7869× | 0.7659× | 0.9901× | Yes | Yes |
| Geometric mean | **0.7870×** | **0.7666×** | — | Yes | Yes |

The retained lane reduced total fit time by about 21.3% and tree-build time by
about 23.3% on the frozen 50,000-row, 24-feature, 300-round workload. All fit,
tree-build, and peak-RSS paired-ratio IQR/median values were below 0.032,
comfortably inside the preregistered 0.10 stability limit.

## Scope

This is a behavior-exact internal engine result against DarkoFit's prior
variable-Hessian dispatch. It is not an external speed claim against
ChimeraBoost, does not consume basketball or CTR23 evidence, and does not
authorize default or API changes.

The raw NPZ file hash is diagnostic only because ZIP timestamps and serialized
phase timing differ between fresh timed workers. The binding model-state hash
covers every serialized key, dtype, shape, and array byte after replacing only
the observational `header.timing` field with `null`.

## Evidence

- Raw artifact:
  [`fused_variable_hessian.json`](fused_variable_hessian.json), SHA-256
  `8af5c94a8561013c94be5d1a9997dbb7b84b9e19733a9dcd4988a1dfc4b4a6cf`.
- Protocol SHA-256:
  `4028c9b40175c1e6cf3b25c4d8211820879410b9cf5e45af71788016ea79ce03`.
- Runner SHA-256:
  `bed0ce17628500d1cebc0ebdecfb652e4c1ecba700842ec45fabb82d80a7dc22`.
