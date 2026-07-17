# Basketball exact TreeSHAP result: promote the supported API

_Formal result for
[`basketball_tree_shap_protocol.md`](basketball_tree_shap_protocol.md)._

## Decision

Promote exact interventional TreeSHAP for the protocol's supported DarkoFit
models. Every frozen correctness, provenance, storage, sports-noise, and timing
gate passed. This is a product-capability decision only: it changes no fitting,
prediction, learning-rate, tree-selection, or other modeling default.

Basketball was the complete development and confirmation boundary for this
feature. The formal lane used creator fold 1 for matched correctness and speed,
then the corrected genuinely cold-player subset as the noisy sports-data
guardrail. It did not inspect a broader tabular panel or any CTR23 coordinate.

The immutable evidence is
[`basketball_tree_shap.json`](basketball_tree_shap.json). It binds DarkoFit
commit `daa2c1ee777f3273bf3b66fbcc4058888875b5f4`, synced ChimeraBoost 0.15.0
commit `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`, the protocol and runner hashes,
the DarkoFit package manifest, the shared basketball helpers, and both clean
source states.

## Correctness and sports guardrails

- DarkoFit and ChimeraBoost retained the same 1,000 constant-leaf depth-six
  trees and produced array-exact predictions on the 524-row confirmation fold,
  2,409-row held-team view, and 585-row cold-player subset.
- Attributions were byte-identical for both the eight-row fold request and the
  eight-row cold-player request. Expected values were exactly equal and the
  maximum attribution difference was zero.
- DarkoFit's maximum Shapley-efficiency error was `1.60e-13` on the fold and
  `1.35e-13` on cold players, well below the frozen `1e-9` limit. Repeated
  DarkoFit calls were array-exact.
- The default empirical background also passed efficiency (`1.42e-13`). Its
  stored binned payload was a bounded 200 by 15 `uint8` array, or 3,000 bytes.
- Independent brute-force oracles, grouped categorical features, local-linear
  slopes, MAE/Quantile, binary margins, serialization corruption checks, and
  explicit unsupported-mode failures are covered by the focused tests. The
  complete suite passed with 1,498 tests and 23 skips.

## Same-machine timing

The warmed eight-row, 32-background-row request ran in 11 reciprocal blocks
with five calls per block at 18 threads:

| Implementation | Median per call | IQR / median |
|---|---:|---:|
| ChimeraBoost 0.15.0 | 32.686 ms | 0.110 |
| DarkoFit | 33.581 ms | 0.069 |

DarkoFit was `1.027x` ChimeraBoost time, comfortably inside the predeclared
`1.50x` parity limit. Both stability ratios were below the `0.30` ceiling.

## Supported boundary and provenance

`DarkoRegressor.shap_values` supports retained scalar oblivious trees with
constant or local-linear leaves. `DarkoClassifier.shap_values` supports binary
raw-log-odds explanations. Both report contributions in original input-feature
space and set `expected_value_`. The deterministic fitted background survives
safe `.npz` round trips, while legacy archives can use a caller-supplied
background.

Multiclass and distributional models, active global linear residuals,
non-oblivious retained trees, and trees using more than 16 original coalition
players remain explicitly unsupported. This result does not authorize those
extensions or any default-policy change.

The algorithm is adapted from Nathan Walker's Apache-2.0 ChimeraBoost commit
`ff6f248d09f92d608ed8cc366463b61f1af04acc`. The module and repository
`NOTICE` preserve that attribution.
