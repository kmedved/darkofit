# T7b automatic depth spent-sports successor protocol

_Prospective protocol authored on 2026-07-22 before the candidate was fit on
any sports case. It becomes frozen only when the create-only machine contract
binds this file, its harness, the exact sources, the already-spent panel, and
the decision rules._

Contract identity: `t7b-automatic-depth-spent-sports-v1-20260722`.

## Question and authority

The exact private automatic-depth candidate `41e948f0c53b1d124e16071a7fa66eba47d084d3`
advanced through its single M6-v3 general-development inspection. This
successor asks whether that unchanged candidate also stays within prospective
quality, uncertainty, concentration, and harm bounds on DarkoFit's
already-spent player-disjoint sports panel.

This is Tier-E private development evidence. It can only make the candidate
eligible for the design of a later, owner-authorized Tier-D campaign on fresh
confirmation data. It cannot merge the candidate, change a default, expose a
public API, authorize fresh data, TabArena, M2, a release, or lockbox access.

## Exact sources and cases

The comparison is clean control
`e23d2b164f10374b1c0e02521c33fc96d48980da` versus clean candidate
`41e948f0c53b1d124e16071a7fa66eba47d084d3`. The candidate tree and its
three-file change allowlist remain those inspected by M6. The contract binds
the advancing M6 launch, raw, result, manifest, and terminal-attestation
hashes; no other candidate may inherit that result.

Use exactly nine cases: seasons 2014, 2015, and 2016 crossed with
`minutes_per_game`, `game_score`, and `box_plus_minus`. Reuse M3b's immutable
sports-panel loader and exact case fingerprints. Each case fits on the
non-held-team primary rows with `bref_id` groups supplied to group-aware
validation. It scores the frozen held-team rows. Primary loss is RMSE on
cold-player held-team rows; all-held-team RMSE is the secondary guard.

The nine rows are not independent datasets. Targets are geometrically pooled
inside each season first. Uncertainty resamples the three season clusters
100,000 times with replacement using seed `20260722`. The one-sided 95th
percentile, all three season ratios, all leave-one-season-out aggregates, and
all nine lineage ratios are reported.

## Fixed execution

Each arm/case runs once in a fresh `darko311` worker under the four-thread
`paired-evidence-v1` environment. Arm order alternates by case. Every worker
performs a same-source, same-policy two-round synthetic warmup outside the
measurement, then fits one `DarkoRegressor` with:

- 600 maximum iterations;
- early stopping with patience 30, `use_best_model=True`, and `refit=False`;
- 0.15 group-aware validation;
- random state 4; and
- four requested threads.

The runner verifies exact source imports, input fingerprints, fitted thread
counts, ambient thread-mask restoration, safe-NPZ prediction parity, and the
candidate's declared low-density automatic-depth metadata. Fit time,
prediction time, peak worker-plus-child RSS, and archive bytes are single-run
telemetry only. They are neither speed claims nor gates.

The exclusive-machine preflight and all prerequisites occur before a
create-only launch manifest. Manifest creation spends the only inspection.
Any subsequent source drift, worker failure, integrity failure, analysis
failure, or output collision is terminal; completed in-memory rows are
discarded and no rerun is authorized under v1.

## Frozen spent-development gates

The exact candidate is `eligible_for_fresh_tier_d_design` only if every gate
passes:

1. cold-player equal-lineage geometric-mean ratio is at most `1.000`;
2. all-held-team equal-lineage geometric-mean ratio is at most `1.010`;
3. the cold-player season-cluster bootstrap 95th percentile is at most
   `1.010`;
4. no season-level cold-player ratio exceeds `1.020`;
5. no individual cold-player lineage ratio exceeds `1.030`; and
6. the worst leave-one-season-out cold-player ratio is at most `1.003`.

There is no win-count or minimum-effect gate. Failure means
`closed_after_spent_sports`; gates are never relaxed after inspection.

These are development triage gates, not the shipping rule. A later automatic
default campaign would still need separate owner authorization, design-time
power, preregistered costs, eligible fresh confirmation, and every Tier-D
requirement in `SHIPPING_POLICY.md`—normally aggregate `<=0.995`, bootstrap
upper `<=1.002`, concentration and harm guards, and stable paired costs.

## Artifacts and reporting

The machine contract, launch manifest, raw rows, analyzed result, and terminal
attestation are create-only and hash-bound. The result records every passed
and failed gate, uncertainty, concentration, harm, telemetry, limitations,
and non-claims. A 12-field `TESTING_LOG.md` record is required before closure.
