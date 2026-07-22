# T7b automatic scalar-RMSE depth development contract

_Frozen on 2026-07-22 before candidate implementation and before any new
candidate quality output was inspected._

Contract identity: `t7b-automatic-scalar-rmse-depth-v1-20260722`.

## Question and authority

Can a samples-per-feature depth policy improve DarkoFit's broad scalar-RMSE
quality without bundling L2 or another tuning change?

This is the second and separate T7b mechanism under the quality slot in
[`BEAT_CHIMERABOOST_PLAN.md`](../BEAT_CHIMERABOOST_PLAN.md). Automatic
scalar-RMSE L2 v1 is already terminal and remains closed. The exact
pre-mechanism control is published commit
`e23d2b164f10374b1c0e02521c33fc96d48980da`. This contract authorizes a
private prototype and spent-development evidence only. It does not authorize
a public default change, a release, sports evidence, TabArena, fresh
confirmation, or lockbox access.

The candidate-generation evidence is donor-engine evidence, not a DarkoFit
outcome. T7 predeclared a CatBoost policy using inner-fit rows per feature:
depth 4 below 100, depth 8 at or above 2,500, and default depth 6 otherwise.
On eight spent development tasks that assembled policy had equal-dataset test
ratio `0.962248`, three wins, no losses, five exact defaults, and worst-task
ratio `1.000000`. That is enough to fund one bounded DarkoFit test, but no
CatBoost result is imputed to DarkoFit.

## Exact one-mechanism candidate

The candidate changes only the fitted interpretation of the existing public
`depth=None` default when all of the following are true:

- the fitted loss is scalar `RMSE`;
- normalized `tree_mode` is `catboost`; and
- the caller left `depth` at `None`.

For that lane only:

1. compute effective sample size with the existing weight-aware helper;
2. divide it by the validated input feature count;
3. resolve depth 4 below 100 effective rows per feature, depth 8 at or above
   2,500, and depth 6 otherwise; and
4. record the rule version, effective rows, input feature count, density,
   thresholds, selected branch, and resolved depth in `auto_params_`.

The thresholds and branches are frozen from the donor policy; they may not be
retuned after DarkoFit outcomes. Explicit numeric depth always wins. The
literal `depth="auto"` keeps its existing effective-row buckets unchanged.
Classification, MAE/Quantile, distributional losses, LightGBM, hybrid,
depthwise, constructor/get-params semantics, and every L2 resolution remain
unchanged. In particular, the candidate must not contain the closed L2 v1
change or any joint depth/L2 rule.

This is a private default-policy candidate, not an opt-in/manual-switch
product. Even a development advance remains ineligible for public exposure
without the full Tier-D automatic-policy path.

## Invariants before quality evidence

The candidate is ineligible for quality evidence until tests establish:

1. unweighted scalar-RMSE CatBoost defaults resolve to depth 4, 6, and 8 at
   the three frozen density branches, including exact boundary tests at 100
   and 2,500;
2. stress weights affect only the density through the existing effective
   sample-size calculation;
3. fitted metadata records all declared rule inputs and the selected branch;
4. explicit numeric depth and literal `depth="auto"` keep their exact control
   resolutions and metadata;
5. classification, non-RMSE scalar losses, and non-CatBoost tree modes retain
   their control resolutions and fitted metadata;
6. L2 resolutions are unchanged in engaged and no-op cases;
7. safe-NPZ and pickle round trips preserve the requested `None` policy, its
   resolved depth, fitted metadata, and exact predictions;
8. clone/get-params, refit parameters, feature names, categorical inputs,
   sample weights, and empty prediction batches remain valid;
9. fit and predict restore the caller's ambient thread-local Numba mask; and
10. the focused suite plus the full local test suite (apart from explicitly
    recorded unavailable-environment prerequisites) passes.

A control-versus-candidate invariant probe must additionally show exact
predictions and fitted-state equality on representative explicit-depth,
literal-auto-depth, classification, MAE, LightGBM, hybrid, and depthwise
cases. It must separately show the three intended candidate branches and
unchanged L2. This probe is correctness evidence, not quality selection.

## Frozen spent-development sequence

### 1. M5 non-ranking sentinels

Run the frozen M5 v1 sentinel check from clean committed sources. M5 fixes
depth at `6` and L2 at `3.0`, so it is deliberately a no-op check: all 19
paired cells must preserve behavior fingerprints, serialization, probability,
known-floor, metadata, and thread invariants. M5 timings and ratios are
telemetry only.

### 2. M6 v3 inspection 1

If the invariants and M5 pass, run exactly one quality inspection through the
mechanism-specific `run_t7b_automatic_depth_v1.py` wrapper. The wrapper binds
this contract, a single clean contract commit directly above the pinned
control with an exact five-file harness allowlist, the clean candidate commit
and its candidate-only file allowlist, and the exact create-only invariant and
M5 artifacts. It
validates that both preconditions belong to the same candidate/harness and
bind current runner hashes before performing the exclusive-machine audit. The
launch manifest records those artifact hashes, exact output paths, and the
underlying `run_m6_quality_successor_v3.py` command **before** starting the
first quality worker. The underlying command uses:

- mechanism id `t7b_automatic_scalar_rmse_depth_v1`;
- inspection index `1`;
- the exact 60-cell medium grid, three seeds, unweighted and stress-weighted,
  three repeats, and four threads; and
- public defaults in both arms, using the pinned control and one clean
  candidate commit.

The immutable M6-v3 disposition is the only general-development stop rule:
aggregate ratio at most `1.000`, worst dataset at most `1.020`, and worst
leave-one-dataset-out ratio at most `1.003`. There is no win-count or minimum
effect gate. The six classification datasets are exact no-op obligations and
remain in the aggregate. Every per-cell ratio, per-dataset ratio, and
leave-one-out ratio is reported.

Any failure after launch-manifest creation consumes inspection 1 and closes
this identity, including runner, source, coverage, metadata, or create-only
failures before the first successful fit. A pre-manifest rejection does not
spend the attempt because no worker or quality output can exist. Any successor
requires a new contract identity; there is no favorable rerun under v1.

### 3. Sports and Tier-D boundary

An M6 `kill` is terminal. An M6 `advance` means only that the exact candidate
may receive a separately frozen, player-disjoint sports development protocol
using already-spent sports-panel data. That successor must preserve the nine
target-season lineages, season-clustered uncertainty, lineage harm and
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
