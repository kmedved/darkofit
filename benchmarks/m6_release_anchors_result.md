# M6 release-anchor establishment

Run 2026-07-20 from clean harness commit `d509111`, under draft-v3 of the
standing-evidence contract.

## Result

The exact M6 ChimeraBoost and CatBoost anchors are established. All 240
create-only rows completed: 120 per product over ten datasets, small and
medium sizes, three seeds, and unweighted plus stress-weighted fits. Every
matched product pair used the same data fingerprint. There were no worker
failures and no worker stderr.

- ChimeraBoost: commit
  `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`
  (`v0.18.0-6-gf14be60`).
- CatBoost: version 1.2.10; installed-wheel `RECORD` SHA-256
  `9c20fb35750d9ff814309323b225e836b538c1496745f357c8fd50187e7824ed`.
- Raw artifact: [`m6_release_anchors.json`](m6_release_anchors.json),
  SHA-256
  `59747bc08d48a2ddad9b3cec05c965ecbd9edf21025c537f17dc58d816385409`.

The hash was embedded in the executable contract, allowing M6 to be marked
`contract_frozen`. The separately predeclared historical backtest later
failed terminally, so v3 never became ranking-eligible; see
[`m6_historical_backtest_result.md`](m6_historical_backtest_result.md).

A same-day post-publication harness audit verified all 50 hashed CatBoost
wheel files against that pinned `RECORD` and strengthened future anchor runs
to assert the imported module path and recheck installation state after
execution. It did not rerun or alter the 240-row artifact.

## Descriptive context

Across the 120 paired cells, CatBoost/ChimeraBoost primary-loss geometric mean
was `0.841814`. By task it was `0.920872` for regression, `0.930470` for
binary classification, and `0.675699` for multiclass classification. CatBoost
fit-time geometric mean was `3.343943x` ChimeraBoost. These are spent
release-anchor descriptions, not gates, portable speed claims, or evidence
for a DarkoFit default.

Each row ran in a fresh worker after a same-product three-tree warmup. Product
defaults were preserved except for the fixed four-thread budget and random
seed. Peak RSS, resolved tree count, quality metrics, fit/predict time,
prediction fingerprints, source identities, and environment metadata are in
the raw artifact.
