# T7b automatic scalar-RMSE L2 development contract

_Frozen on 2026-07-22 before candidate implementation and before any new
candidate quality output was inspected._

Contract identity: `t7b-automatic-scalar-rmse-l2-v1-20260722`.

## Question and authority

Can the CatBoost-derived `l2_leaf_reg=1` observation improve DarkoFit's broad
quality without bundling a second tuning change or creating a manual-switch
product?

This is the next one-mechanism quality project authorized by
[`BEAT_CHIMERABOOST_PLAN.md`](../BEAT_CHIMERABOOST_PLAN.md), after the automatic
linear-selector campaign reached its terminal disposition. The exact
pre-mechanism control is published commit
`370b8924c034de0332a4b990817972cf0e876f3e`. This contract authorizes a private
prototype and spent-development evidence only. It does not authorize a public
default change, a release, TabArena, fresh confirmation, or lockbox access.

The historical CatBoost attribution and DarkoFit regression ablation are
candidate-generation evidence, not confirmation. In particular:

- CatBoost's `l2_leaf_reg=1` arm had a promising aggregate result in T7b;
- DarkoFit's old four-dataset stress screen found a `0.9937` aggregate ratio
  for the L2-only arm; and
- the later broad-panel failure belongs to a combined L2/bin/learning-rate
  candidate and cannot identify L2's own broad effect.

No new TabArena coordinate may be opened to resolve that ambiguity here.

## Exact one-mechanism candidate

The candidate changes only the existing automatic L2 resolver:

1. when `l2_leaf_reg="auto"`, the fitted loss is scalar `RMSE`, and the
   normalized tree mode is `catboost`, use an unweighted base of `1.0` instead
   of `3.0`;
2. retain the existing effective-sample-size concentration multiplier and
   clipping to `[base, 20.0]`;
3. record a candidate-specific stable rule name and the resolved base in
   `auto_params_`; and
4. leave every other path byte-for-byte behavior-equivalent where feasible:
   explicit L2 values, classifiers, MAE/Quantile, distributional losses,
   LightGBM, hybrid, depthwise, constructor/get-params semantics, and the
   public parameter default remain unchanged.

There is no L2 audition, validation-set selection, or outcome-dependent
branch. The product surface remains the existing automatic parameter: callers
who omit L2 get the fitted task/mode/weight policy, while explicit callers
always win. The samples-per-feature depth idea is a separate mechanism and is
forbidden in this candidate.

## Invariants before quality evidence

The candidate is ineligible for a quality run until tests establish:

1. unweighted scalar-RMSE CatBoost auto-L2 resolves to exactly `1.0` and the
   fitted metadata names the new rule;
2. stress weights retain the old concentration formula with base `1.0` and
   the existing upper clip;
3. explicit L2 values still resolve exactly as supplied;
4. classification, non-RMSE scalar losses, and non-CatBoost tree modes retain
   their control resolutions and fitted metadata;
5. safe-NPZ and pickle round trips preserve the requested `"auto"` policy,
   its resolved value, fitted metadata, and exact predictions;
6. clone/get-params, refit parameters, feature names, categorical inputs,
   sample weights, and empty prediction batches remain valid;
7. fit and predict restore the caller's ambient thread-local Numba mask; and
8. the focused suite plus the full local test suite (apart from explicitly
   recorded unavailable-environment prerequisites) passes.

A control-versus-candidate invariant probe must additionally show exact
predictions and fitted-state equality on representative explicit-L2,
classification, MAE, LightGBM, and hybrid cases. This probe is correctness
evidence, not quality selection.

## Frozen spent-development sequence

### 1. M5 non-ranking sentinels

Run the frozen M5 v1 sentinel check from clean committed sources. M5 uses an
explicit L2 of `3.0`, so it is deliberately a no-op check: all 19 paired cells
must preserve behavior fingerprints, serialization, probability, known-floor,
metadata, and thread invariants. M5 timings and ratios are telemetry only.

### 2. M6 v3 inspection 1

If the invariants and M5 pass, run exactly one quality inspection through the
mechanism-specific `run_t7b_automatic_l2_v1.py` wrapper. The wrapper binds this
contract, the clean harness/control/candidate commits, the candidate-only file
allowlist, the exclusive-machine audit, the exact output paths, and the
underlying `run_m6_quality_successor_v3.py` command in a create-only launch
manifest **before** starting the first quality worker. The underlying command
uses:

- mechanism id `t7b_automatic_scalar_rmse_l2_v1`;
- inspection index `1`;
- the exact 60-cell medium grid, three seeds, unweighted and stress-weighted,
  three repeats, and four threads; and
- public defaults in both arms, using the pinned control and one clean
  candidate commit.

The immutable M6-v3 disposition is the only general-development stop rule:
aggregate ratio at most `1.000`, worst dataset at most `1.020`, and worst
leave-one-dataset-out ratio at most `1.003`. There is no win-count or minimum
effect gate. The six classification datasets are expected exact no-ops and
remain in the aggregate; they may not be removed after inspection. Every
per-cell ratio, per-dataset ratio, and leave-one-out ratio is reported.

Any failure after the launch manifest exists consumes inspection 1 and closes
this identity, including runner, source, coverage, metadata, or create-only
failures before the first successful fit. A pre-manifest preflight rejection
does not spend the attempt because the wrapper proves no worker could have
started and no quality output could exist. Any successor still requires a new
contract identity; there is no favorable rerun under v1.

### 3. Sports and Tier-D boundary

An M6 `kill` is terminal. An M6 `advance` means only that the exact candidate
may receive a separately frozen, player-disjoint sports development protocol
using the already-spent sports-panel data. That successor must preserve the
nine target-season lineages, season-clustered uncertainty, lineage harm and
leave-one-season-out disclosure, and create-only/no-rerun discipline. It must
be frozen before candidate sports outcomes are opened.

Even a sports advance cannot change the default. A public automatic-policy
decision needs separate owner authorization for a design-time-powered Tier-D
campaign with prospective harm bounds and eligible fresh evidence. M2 and
TabArena remain owner-gated milestone evidence.

## Reporting and terminal dispositions

Every material run uses clean committed sources, fresh workers, exact source
hashes, create-only raw/result/manifest artifacts, and a 12-field
[`TESTING_LOG.md`](TESTING_LOG.md) entry. Timed telemetry requires the usual
exclusive-machine preflight. No inspected gate may be relaxed, no cell may be
retuned, and a candidate change creates a new identity.

The only v1 dispositions are:

- `closed_in_invariants`;
- `closed_in_m5`;
- `closed_in_m6`;
- `eligible_for_spent_sports_design`; or
- after a separately frozen sports successor, its recorded terminal outcome.

None means `ship` or `change_default`.
