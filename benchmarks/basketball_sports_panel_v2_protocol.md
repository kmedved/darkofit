# Basketball sports automatic-policy confirmation, panel 2

## Decision and candidate

This is T10's fresh Tier-D confirmation boundary for the named `sports`
profile. It asks whether a five-member row-bootstrap OOB ensemble should
become an automatic sports policy:

- control: `DarkoRegressor(random_state=4)`;
- candidate: the same estimator with `n_ensembles=5`,
  `ensemble_bootstrap="rows"`, and
  `ensemble_shared_preprocessing=True`.

Every ensemble member draws a training-sized row bootstrap, uses its exact
out-of-bag complement for early stopping, and contributes equally to the mean
prediction. The candidate is the public T2 implementation of the earlier
row-OOB mechanism whose quality reproduced but whose API was closed under a
superseded millisecond timing-stability rule. The group-bootstrap mechanism
and `random_strength=0.5` both failed sports quality and are excluded. There
is no composite retuning on the spent S4 panel.

Passing authorizes only a named sports-profile automatic policy. It does not
change the global default or silently enable ensembles for ordinary
`DarkoRegressor` construction.

## Fresh data boundary

The raw source is the same pre-existing Basketball Reference game-log export
attested by S4:

```text
bytes   214366516
sha256  96e0efffb09e27f64cee395faa1783b025757c88efb74f5fb98cbd82c583d826
```

Panel 2 uses complete seasons 2014, 2015, and 2016. S4 spent 2017–2019; none
of panel 2's target-season cells has previously received a DarkoFit,
ChimeraBoost, or CatBoost score. The aggregation, 500-minute eligibility
threshold, canonical ordering, 15 predictors, and three targets are inherited
unchanged from the source-attested S4 builder:

```text
features:
age, games, start_rate, ts_pct, efg_pct,
orb_pct, drb_pct, trb_pct, ast_pct, stl_pct,
blk_pct, tov_pct, usg_pct, offensive_rating, defensive_rating

targets:
minutes_per_game, game_score, box_plus_minus
```

Schema, identities, row counts, missingness, team counts, and the split
construction may be inspected before the freeze. No target/model association,
prediction, or score may be inspected until the protocol, builder, runner,
analyzer, tests, processed manifest, and power calculation are committed and
pushed.

## Player-disjoint folds and guardrails

Each target-season pair is one equally weighted lineage, for nine lineages.
For each season, sort its 30 team labels and hold out the middle third
(positions 10–19). This differs from S4's first-third holdout and is resolved
separately by season so historical franchise-code changes cannot alter the
boundary.

The remaining 20-team rows form the primary sample. Ten-fold `GroupKFold`
uses exact `bref_id` player IDs, so a player never appears in both training
and test within a fold. Fold indices and identities are frozen in the
manifest. This replaces S4's overlap-permitting row K-fold with the
user-relevant cold-player estimand.

For each lineage, an additional model fits all primary rows and predicts the
held-team rows. These are reported as:

- all held-team rows;
- seen players, whose ID occurs in primary rows that season; and
- cold players, whose ID does not.

Primary and guardrail RMSEs are pooled over their complete prediction vectors,
not averaged across fold metrics. R² is descriptive only.

## Frozen Tier-D decision

For each lineage let `r = RMSE(candidate) / RMSE(control)`. Aggregate ratios
are geometric means over the nine lineages. The candidate passes only if:

1. the equal-lineage aggregate ratio is at most `1.000`; the candidate's
   declared value is robustness, so the constitution permits the no-harm bar;
2. the 95th percentile of 100,000 lineage-bootstrap aggregates, seed
   `20260718`, is at most `1.002`;
3. after removing the single most favorable lineage, the aggregate is at
   most `1.003`;
4. the worst lineage ratio is at most `1.020`;
5. the aggregate all-held-team and cold-player ratios are each at most
   `1.005`, and no individual guardrail lineage exceeds `1.020`;
6. same-block median candidate/control total-fit, total-predict, and peak-RSS
   ratios are each at most `3.0`; and
7. the paired fit and prediction ratio series each have IQR/median at most
   `0.20`.

The 3× budget is declared from the reproduced five-member measurements
(2.414× wall and 2.259× prediction) and represents the maximum acceptable
cost for an automatic accuracy/robustness sports policy. Timing stability is
evaluated only on complete multi-lineage workloads measured in seconds.
Behavior fingerprints must reproduce exactly across all blocks.

## Design-time power

The immutable OOB-ensemble confirmation artifact supplies ten paired fold
R² values. Because both arms share each fold's target vector, each RMSE ratio
is exactly:

```text
sqrt((1 - R2_candidate) / (1 - R2_control))
```

The observed geometric mean is approximately `0.99673`; log-ratio sample SD
is approximately `0.01048`. The preregistered simulation uses seed
`20260718`, 20,000 panels, nine lineages, ten folds per lineage, conservative
between-lineage log-effect SD `0.005`, and 2,000 fixed bootstrap resamples per
simulated panel. It applies gates 1–4 exactly. The design proceeds only if
the simulated pass probability is at least 80%. Guardrail and cost gates are
not assigned invented probability models.

## Execution and external context

DarkoFit control/candidate, ChimeraBoost 0.15.0 at clean commit `851ab7f`,
and CatBoost 1.2.10 run on byte-identical matrices, folds, and holdouts with
random state 4 and 18 threads. Three fresh-worker reciprocal/rotating blocks
run all four arms. Each worker performs one complete first-fold warmup outside
the timer. The raw create-only artifact records predictions, fitted metadata,
hashes, timings, RSS, source state, versions, and environment. A separate
frozen analyzer owns all decision arithmetic and requires unambiguous raw,
JSON-output, and report paths.

External arms are descriptive same-machine context and cannot rescue a failed
candidate. The first formal run spends panel 2. No rerun, task removal, gate
change, or candidate adjustment is allowed after any outcome is observed.
