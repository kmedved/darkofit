# SynthGen — DarkoFit mechanism-probe suite

DarkoFit’s `df1` SynthGen suite is a contamination-safe, prior-sampled
synthetic probe. It is used to test whether the generator can reproduce
already-known DarkoFit mechanism decisions. It cannot promote a model policy
or support a release claim by itself.

This implementation is modified from the Apache-2.0 SynthGen implementation
in ChimeraBoost 0.15.0, commit `851ab7f`. DarkoFit changed the generator
version, calibration corpus, exclusion rules, canary verifier, suites,
goldens, tests, and decision ledger. See `NOTICE` and
`benchmarks/synthgen_darkofit_protocol.md`.

## Contamination boundary

`harvest_metadata.py` refreshes active OpenML metadata and removes, before any
deduplication or calibration reduction:

- every member of TabArena study 457;
- all 35 frozen CTR23 datasets, by both OpenML dataset id and normalized name.

The refresh fails closed if either exclusion set cannot be resolved. The
committed `corpus_marginals.json` stores source and identity hashes, observed
counts, and the exact exclusions. The raw OpenML response is a re-downloadable
cache and is not committed.

No TabArena or CTR23 feature values, labels, benchmark outcomes, or fitted
model results are inputs to this generator.

## Generator

Dataset content is a pure function of `(VERSION, dataset_id)` using
`SeedSequence([VERSION_SEED, dataset_id])`. Keys such as `syn:df1/086` carry
the generator version. Any recipe or random-stream change requires a new
version, a complete refreeze, and new golden hashes; goldens must never be
silently re-pinned under the same version.

The structural prior covers numeric and categorical tabular data, layered
random DAGs, linear and nonlinear node functions, interactions, missingness,
irrelevant features, class imbalance, entity categoricals, and deterministic
saturated rules. Regression records expose the known noise floor;
classification records expose a sum-form Brier floor.

Freeze-time quality gates reject degenerate, unlearnable, or intractable
datasets. Canary membership is earned: a fixed DarkoFit verifier must reach
the known floor across three predeclared splits. Canary membership is frozen
in `suites.py`; it is not inferred at benchmark runtime.

## Frozen `df1` suites

- smoke: 6 datasets;
- screen: 145 datasets / 400,036 rows;
- full: 240 datasets / 1,600,125 rows.

The suites are nested. The screen has 48 regression datasets, 51 datasets
with categorical features, and four categorical canaries. Its four frozen
decision slices contain 46, 8, 11, and 17 datasets respectively. The full
suite has six verified canaries. Exact ids and goldens live in `suites.py` and
`tests/golden_synthgen.json`.

## Reproduction

Refresh the contamination-safe corpus:

```bash
.venv/bin/python benchmarks/synthgen/harvest_metadata.py \
  --refresh --max-rows 2500
```

Refreeze and persist all scan records:

```bash
PYTHONPATH=benchmarks .venv/bin/python benchmarks/synthgen/freeze.py \
  --count 1000 --output benchmarks/synthgen_df1_freeze.json
```

Run generator contract tests:

```bash
.venv/bin/python -m pytest -q \
  tests/test_synthgen.py tests/test_synthgen_harvest.py
```

The formal nine-decision DarkoFit ledger is defined, including slices,
splits, budgets, pass criteria, and prohibitions, in
`benchmarks/synthgen_darkofit_protocol.md`. Formal results must be produced
only by the frozen DarkoFit ledger runner from a clean source commit.

## Files

`recipe.py` samples structural metadata; `scm.py` builds the latent graph;
`emit.py` emits the observed dataset and floor; `api.py` owns keys, caching,
and hashing; `calibration.py` samples the committed corpus;
`harvest_metadata.py` refreshes that corpus; `filters.py` applies freeze-time
quality and canary gates; `freeze.py` scans and selects; `suites.py` freezes
membership.
