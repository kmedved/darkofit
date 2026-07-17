# Basketball packed-prediction routing confirmation protocol

## Decision and scope

This is an isolated confirmation of one prediction-only routing change for
constant-leaf oblivious forests. It does not change fitting, tree contents,
learning-rate policy, early stopping, categorical handling, linear leaves, or
any public default. It uses only the frozen basketball data; no CTR23 or
TabArena task is opened.

The prior same-machine ChimeraBoost 0.15 characterization established that
matched 1,000-tree models produce byte-identical predictions, while DarkoFit's
small-batch public prediction was slower. An exploratory first-fold profile
localized the cause: DarkoFit's fixed 8,192-row boundary forced the serial
packed kernel on a 525-row basketball fold. Serial versus parallel packed-core
medians were 3.29 ms versus 0.90 ms at that fold size. At 8,192 through 100,000
repeated basketball rows, DarkoFit was already at parity with or faster than
ChimeraBoost.

The single candidate therefore replaces the fixed boundary for constant-leaf
oblivious forests with a conservative forest-work rule:

```text
parallel rows = min(8192, max(128, ceil(131072 / fitted_tree_count)))
```

The 128-row floor remains above the exploratory scheduler crossover. The
8,192-row cap preserves the existing behavior for forests of 16 trees or
fewer, including the five-tree benchmark warmups. Linear-oblivious and
explicit-node predictors retain their existing routes because this campaign
does not measure them.

## Frozen sources, data, and model

- DarkoFit must be a clean committed checkout whose tracked `darkofit/`
  content manifest is exactly
  `6e80c24202ef503d43f6655ea66e866d7cb52ff670df8054fbf962483b8e9846`.
  The three shared basketball data/split helpers are separately content-pinned;
  the exact head, package-tree hash, and runner hash are also recorded.
- ChimeraBoost must be the clean synced `v0.15.0` checkout at
  `851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`.
- The immutable creator CSV, processed feature/target fingerprints, folds, and
  cold-player mask are enforced by `basketball_harness.py`.
- Exploratory work used creator fold 0. Confirmation uses creator fold 1 only.
- Both libraries fit the exact matched constant-leaf configuration from
  `basketball_chimera_v015_protocol.md`: 1,000 depth-six trees, learning rate
  0.1, L2 1, 128 bins, no ordered boosting, no early stopping, no linear
  leaves, no cross features, seed 4, and 18 threads.

Both fitted models must retain 1,000 trees. Their public and packed-core
predictions must be array-exact on the confirmation fold, the full held-team
view, its 585-row cold-player subset, and every repeated-row throughput case.
The runner independently constructs both estimators, hashes their complete
pre-fit `get_params(deep=False)` mappings, and verifies every resolved
structural setting, all 1,000 fitted depths, and the absence of linear leaves.

## Frozen prediction cases

The runner evaluates:

- 127 confirmation-fold rows, which must remain serial;
- the untouched fold-1 test batch;
- the 585 genuinely cold-player rows;
- all 2,409 held-team rows;
- 8,192 repeated fold-1 rows; and
- 100,000 repeated fold-1 rows.

Repeating rows changes no values or labels and is used only to measure
throughput at realistic larger batch sizes. It is not accuracy evidence.

The pre-change DarkoFit reference uses the same packed arrays and kernels but
retains the old row-only 8,192 boundary. The candidate calls the public fitted
router. ChimeraBoost calls its packed row-major predictor. Timing excludes
fitting and binning for the core comparison; a separate public comparison
includes input validation and binning.

All kernels, packed caches, and public paths are warmed before measurement.
Cases use 11 alternating timing blocks and enough inner calls to keep the
small-batch measurements stable. Medians and IQR fractions are recorded.
The selected DarkoFit serial or parallel kernel is instrumented and observed
once per case outside timing; route gates never infer dispatch from the desired
formula. ChimeraBoost's packed kernel is imported and bound before any timed
call.

## Promotion gates

The routing change advances only if all of the following hold:

- all public and packed-core predictions are array-exact across DarkoFit
  candidate, DarkoFit legacy route, and matched ChimeraBoost;
- the fitted tree counts, resolved learning rate, and resolved thread counts
  remain the frozen values;
- 127 rows select serial, while the fold, cold-player, held-team, and larger
  cases select parallel;
- at both the confirmation-fold and cold-player batch sizes, the packed-core
  candidate is at least 2.0 times faster than the legacy DarkoFit route;
- at those two sizes, DarkoFit candidate packed-core time is at most 1.15
  times ChimeraBoost and DarkoFit public time is at most 1.20 times
  ChimeraBoost;
- at 8,192 and 100,000 rows, candidate packed-core time is no more than 10%
  slower than the already-parallel legacy route and no more than 20% slower
  than ChimeraBoost;
- every gated timing series has IQR/median at most 0.30; and
- candidate and legacy use the same fitted packed arrays and allocate the same
  output shape and dtype. No new fitted-model storage is permitted.

A failed gate rejects this cutoff; it does not authorize tuning the threshold
on fold 1. A pass promotes only this constant-leaf routing rule. The broad
best-of-both objective remains active, with product-policy and linear-leaf
work separate.

The frozen command is:

```bash
PYTHONPATH=. NUMBA_CACHE_DIR=.cache/numba-basketball-packed \
  .venv/bin/python benchmarks/run_basketball_packed_prediction.py \
  --threads 18 \
  --chimeraboost-repo /Users/kmedved/Code/GitHub/chimeraboost \
  --output benchmarks/basketball_packed_prediction.json
```
