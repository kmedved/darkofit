# Large-n matched-core engine result

_Run 2026-07-17 from clean DarkoFit `main` at `d77dc8b`, with clean
ChimeraBoost 0.15.0 at `851ab7f`, under the frozen
[`large_n_engine_protocol.md`](large_n_engine_protocol.md)._

## Decision

Do not claim that DarkoFit is at least 1.30× faster than ChimeraBoost on this
matched 500k–1M numeric lane. The measured equal-size geometric-mean speedup
was **1.2793×**, just below the frozen 1.30× requirement. Protocol failure
closes this certification and does not authorize a weaker threshold or another
optimization round.

| Train rows | DarkoFit / ChimeraBoost fit | Speedup | RMSE ratio | RSS ratio |
|---:|---:|---:|---:|---:|
| 500,000 | 0.8038× | 1.2441× | 0.99998× | 0.8432× |
| 1,000,000 | 0.7601× | 1.3155× | 1.00085× | 0.7129× |
| Equal-size geometric mean | **0.7817×** | **1.2793×** | — | — |

Every other frozen gate passed: both size-specific fit ratios were below
0.85, quality was within the 1.002 RMSE boundary, fit and RSS paired ratios
were stable, peak RSS was lower, behavior fingerprints were stable, and the
fused lane engaged in every DarkoFit worker. All 12 workers completed with no
stderr.

The pre-protocol forced sibling-subtraction profile remains rejected. It was
slower at the production thread count, so the formal run measured the retained
fused/uint8/capped-border system rather than adding an unearned mechanism.

## Fresh-eyes verification

The post-run analyzer audit made coordinate pairing explicit, checks each
reciprocal block by key, binds the generated-data probe across both arms, and
enforces the protocol's no-stderr gate. Re-analyzing the immutable raw results
with those stronger checks reproduced the same fit ratios and the same
failure decision.

## Evidence

- Raw artifact: [`large_n_engine.json`](large_n_engine.json), SHA-256
  `ac9e6e9f136117b7b1db7488b38f660561195f86b29ae2f87868a5d293c62508`.
- Protocol SHA-256:
  `d78273f73d513beffaf5cbd44f76c1709419f1a08fc2bbe161d0b2e63b2d11bc`.
- Attested run-time runner SHA-256:
  `3f5411b03c58c9a56cd1549510b702cadfd4b27a319010e1dabad7871363ab26`.
- DarkoFit source:
  `d77dc8b6ede6e87c84aaebfb3c9d03447c48cbb9`.
- ChimeraBoost source:
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`.
