# Wave 1 M3a shipped-ensemble protocol

_Frozen before M3a outcome access on 2026-07-20._

## Purpose and evidence boundary

M3a is the quality-first shipped-ensemble comparison authorized in
`COUNTERPUNCH_PLAN.md`. It asks whether DarkoFit's existing group-bootstrap
ensemble has enough player-disjoint quality value at an acceptable cost to
fund a private ensemble-v3 program. It also places the current public
ChimeraBoost eight-member ensemble on the same spent sports panel and adds a
small non-sports development-slice check.

This is Tier-E descriptive evidence on an already spent panel. It may close
or continue Track B in the owner-facing G-M portfolio decision. It cannot
change a default, certify a release, make a cross-season generalization
claim, or authorize any public ensemble-v3 behavior.

The campaign uses only public behavior at these exact package pins:

- DarkoFit `726e5d8e6131c580bce948db833a5007d0692dca`, the single clean
  post-H1 package-source pin;
- ChimeraBoost `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`, version 0.18.0;
- the frozen player-disjoint sports-panel-v2 manifest; and
- the frozen M6 adapter source for the selected general cells.

The contract manifest binds this protocol, the runner, analyzer, source
pins, sports manifest, M6 adapter, fold construction, arms, orders, and
decision thresholds by SHA-256. Formal workers import the packages from
clean exact-source trees. Numba, Python, joblib, and plotting caches live
outside those trees.

## Sports scoring views

The primary view is the spent sports-panel-v2 player-disjoint design:
three complete seasons (2014-2016), three targets, ten exact `bref_id`
`GroupKFold` folds per target-season cell, and the frozen middle-third
held-team split. Each cell reports pooled player-disjoint RMSE plus
all-held-team, seen-player, and cold-player RMSE.

A secondary creator-style diagnostic uses ten-fold shuffled row K-fold
within the same primary rows. Its seed is `20260720 + season`; its exact
fold-index hashes are frozen in the M3a contract. Because players may appear
on both sides, this view is explicitly player-overlap exposed.

The nine sports cells share players and overlapping rows. They are not nine
independent lineages. Paired aggregation first takes the mean log RMSE ratio
across the three targets within each season. Uncertainty then resamples the
three season clusters, with replacement, 100,000 times using seed
`20260720`. The report must show the three season ratios and the 2.5th,
50th, and 95th/97.5th percentiles. With only three spent seasons, these are
descriptive clustered intervals, not evidence of transfer to unseen seasons.

## Arms and overlap disclosures

Primary arms:

1. `darkofit_single`: one DarkoFit model, with exact player groups supplied
   so its internal validation split is group-aware;
2. `darkofit_group_ensemble8`: `n_ensembles=8`,
   `ensemble_bootstrap="groups"`, shared numeric target-free preprocessing,
   and exact player groups;
3. `chimeraboost_single`: current quantized ChimeraBoost single, with exact
   player groups supplied to its internal validation split; and
4. `chimeraboost_ensemble8`: current quantized eight-member 0.8 row
   subagging ensemble.

Diagnostic arms:

- DarkoFit row-bootstrap ensembles with five and eight members;
- DarkoFit group-bootstrap with five members;
- ChimeraBoost float single; and
- ChimeraBoost float eight-member ensemble.

DarkoFit group-bootstrap member selection is player-disjoint. DarkoFit
row-bootstrap member selection and ChimeraBoost row subagging are
player-overlap exposed even when the outer fold is player-disjoint.
Forwarding groups to ChimeraBoost makes its internal validation group-aware;
it does not make its row sampling group-safe.

All arms use seed 4, the public defaults at their pinned sources, and a
14-thread current-machine budget. ChimeraBoost's public ensemble scheduler
may divide that budget across member processes. No proposed 0.8 DarkoFit
subagging, member tuning, or unshipped ensemble-v3 behavior is included.

## Quality-first execution

Every arm runs in a fresh worker and performs a same-arm first-fold warmup
outside measured work. The first formal phase runs each primary arm once.
Only after its quality artifact is complete may the frozen analyzer decide
whether additional primary timing blocks are allowed.

DarkoFit group-ensemble8 survives only when all of these predeclared checks
hold against `darkofit_single`:

1. all integrity checks pass;
2. equal-cell player-disjoint geometric-mean RMSE ratio is at most `0.995`;
3. the season-clustered 95th percentile is at most `1.000`;
4. all-held-team and cold-player geometric-mean ratios are each at most
   `1.005`;
5. no season-level player-disjoint ratio exceeds `1.010`, and no individual
   player-disjoint cell exceeds `1.030`;
6. measured player-disjoint-plus-held fit and prediction ratios are each at
   most `9.0`;
7. median held-team fitted-model bytes are at most `9.0` times single; and
8. aggregate process-tree peak RSS is at most `4.0` times single.

The 0.5% aggregate quality requirement is the minimum worthwhile signal for
funding another private ensemble program; simple no-harm at roughly eight
models is not enough. The cost bars are bounded-complexity guards, not claims
of a Pareto win.

If any check fails, record only the first-pass descriptive timing and skip
the repeat series. If all checks pass, run two more fresh-worker blocks in
rotating order, giving each primary arm three total observations. Report
the full series and medians. Diagnostic-arm costs are one warmed run and are
excluded from timing decisions.

## General development-slice context

The general check reuses the frozen M6 adapter without treating M5 as a
scoreboard. It fixes the medium 10,000-row, unweighted, seed-0 and seed-1
cells for:

- `friedman_numeric`;
- `wide_numeric_reg`; and
- `categorical_reg`.

It compares DarkoFit single versus its shipped row-ensemble8 and
ChimeraBoost single versus its current ensemble8. These cells have no entity
groups, so the ensemble comparisons are explicitly row-sampling diagnostics.
They can show whether a sports finding is obviously sports-only; they cannot
rescue a failed group-safe sports result or rank new mechanisms because M6
v3's historical backtest terminal-failed.

The already frozen 38-row M5 baseline remains the invariant/drift evidence.
No M5 quality coordinate is ranked or tuned here.

## Resource and artifact contract

Each worker records pooled scores, prediction/target hashes, fold-level
scores, fitted tree/thread/member and OOB telemetry where exposed, captured
warnings, fit/predict wall time by view, pickle bytes, and a sampled
aggregate RSS peak across the worker plus recursive child processes.
The ChimeraBoost aggregate RSS sample is best-effort at a 10 ms interval and
is labeled accordingly.

Formal phase shards are create-only temporary artifacts. The frozen analyzer
combines them once into one create-only JSON artifact and one result note,
recording every shard hash and raw result. Source and harness hashes must be
unchanged across every worker. Integrity failures are published rather than
rerun to improve an outcome.

M3a's final Track-B disposition is:

- **continue** only when group-ensemble8 survives every frozen check; or
- **close / preserve opt-in** otherwise.

The later G-M note combines that disposition with the already published M1
and Q0 results.
