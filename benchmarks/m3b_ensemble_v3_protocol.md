# Wave 2 M3b private ensemble-v3 attribution protocol

_Prospective protocol. It becomes frozen only when the create-only M3b
machine contract binds this file, the implementation, runner, analyzer,
source pin, exact case fingerprints, and decision rules._

## Claim and evidence boundary

M3b is a spent-data, Tier-E private attribution. It compares the current
sequential eight-member ensemble against the B1 sampling mechanism, the B2
member-policy mechanism, and their combination. It may recommend continued
private work or a later documented opt-in proposal. It cannot authorize B3,
a constructor/public/default change, fresh confirmation, TabArena, a release
claim, or lockbox access.

M3a's runner, analyzer, freezer, contract, and result remain historical and
byte-preserved. M3b has a new contract identity and imports only M3a's frozen
sports-panel loader/view construction. The strict `paired-evidence-v1` worker
environment is required before DarkoFit import.

## Fixed arms

All ensemble arms have eight members, seed 4, four threads per sequential
member, 600 maximum rounds, patience 30, shared numeric preprocessing, and
OOB early stopping. The exact arms are:

1. `single_reference`: one normal DarkoFit estimator, used only for the B0
   archive/RSS targets;
2. `control`: full-size bootstrap sampling and no member policy;
3. `b1_sampling`: 0.8 sampling without replacement and no member policy;
4. `b2_member_policy`: full-size bootstrap sampling plus
   `donor_balanced_v1`; and
5. `b1_b2_combined`: 0.8 sampling without replacement plus
   `donor_balanced_v1`.

Sports arms sample groups; general arms sample rows. `single_reference` uses
group-aware automatic validation for sports and task-appropriate automatic
validation for general cases. The four ensembles use the private B0 entry
point. Control prediction parity with the public bootstrap path is an
invariant, not an M3b outcome.

## Fixed quality cases

### Spent sports view

Reuse all nine target-season cells from the frozen sports-panel-v2 data:
2014, 2015, and 2016 crossed with `minutes_per_game`, `game_score`, and
`box_plus_minus`. Fit once on each frozen non-held-team primary set, passing
`bref_id` groups. Score the frozen held-team set. The primary loss is RMSE on
cold-player held-team rows; all-held-team RMSE is the secondary guard. This
uses the player-disjoint part of the already-spent held-team view without
rerunning the ten outer player folds.

### General weighted view

Use the frozen M6 adapter at medium size (10,000 rows), seed 0, and stress
weights for exactly four cells:

- `friedman_numeric` (regression);
- `categorical_reg` (regression);
- `numeric_binary` (binary classification); and
- `categorical_multiclass` (multiclass classification).

Use one deterministic 75/25 holdout split with seed 20260720, stratified for
classification. Fit and score weights are aligned from the full deterministic
stress-weight vector. Primary loss is weighted RMSE for regression and
weighted log loss for classification; the unweighted counterpart is
secondary. These medium cells close the small-data and classification
blindspots; they are spent development evidence, not M6 rehabilitation.

The freezer computes and binds exact case, dataset, split, weight, sports
manifest/cache, and implementation fingerprints before any model outcome.

## Quality-first execution

Every arm/case runs in a fresh interpreter. The parent scrubs inherited Python,
Numba, OpenMP, BLAS, and related execution overrides, then fixes the
four-thread `paired-evidence-v1` environment. The worker attests the Numba
ceiling/current mask and threading backend before importing DarkoFit, imports
the exact clean source pin, performs an arm-matched two-member/two-round
synthetic warmup outside measurement, then runs the formal fit.

The quality artifact is complete and create-only before any repeat timing.
Each row records exact input fingerprints, implementation path, fitted thread
counts, member/sampling/policy/OOB metadata, prediction/probability hashes,
primary and secondary losses, fit/predict wall time, aggregate process-tree
peak RSS and sampler errors, safe-NPZ bytes, fitted per-member OOB scores,
warnings, and exact safe-load prediction parity. OOB scores are descriptive
telemetry and do not add an undeclared quality gate.

For each candidate, compute paired candidate/control loss ratios. A candidate
is quality-eligible for repeat timing only when all of these hold:

- all-case primary-loss geometric mean at most `1.005`;
- general primary-loss geometric mean at most `1.005`;
- sports cold-player geometric mean at most `1.005`;
- sports all-held-team geometric mean at most `1.010`; and
- worst individual primary-loss ratio at most `1.030`.

The frozen analyzer creates a gate artifact from the complete quality
artifact. Failed candidates receive no repeat timing. If no candidate is
eligible, repeat timing is skipped entirely. Otherwise `single_reference`,
`control`, and eligible candidates receive two additional fresh-worker runs
in contract-fixed rotating order. Failed attempts are terminal; no rerun may
replace an unfavorable or failed row. Once preflight passes, any worker or
execution failure creates a distinct create-only terminal failure artifact;
completed in-memory rows are discarded rather than published as partial
outcomes.

## Final attribution rules

For timing/resource summaries, take each case/arm median over the quality run
and its two repeats, then aggregate paired case ratios geometrically. Archive
and RSS ratios against `single_reference` use the median of the 13 paired case
ratios, matching B0's stated design targets.

Every final survivor must remain quality-eligible and satisfy:

- prediction-time ratio to control at most `1.10`;
- safe-NPZ-byte ratio to control at most `1.05`;
- peak-RSS ratio to control at most `1.10`;
- median safe-NPZ bytes at most `4.0` times matched single; and
- median peak RSS at most `2.0` times matched single.

The mechanism-specific value rules are:

- `b1_sampling`: all-case primary ratio at most `1.002` and fit-time ratio at
  most `0.90`;
- `b2_member_policy`: all-case primary ratio at most `0.995` and fit-time
  ratio at most `1.10`; and
- `b1_b2_combined`: either all-case primary ratio at most `0.995`, or
  all-case primary ratio at most `1.002` with fit-time ratio at most `0.90`.

The final private disposition is deterministic:

1. prefer `b1_b2_combined` when it survives;
2. otherwise retain each surviving causal component separately;
3. otherwise close B1/B2 and preserve the existing opt-in unchanged.

No result from this campaign is a public shipping claim. Any later public
surface requires a separate owner decision, API contract, documentation, and
the evidence tier appropriate to its proposed behavior.
