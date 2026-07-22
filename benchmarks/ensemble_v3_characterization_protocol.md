# Ensemble-v3 release-candidate characterization protocol

_Prospectively frozen before any current-source performance measurement._

Contract identity: `ensemble-v3-characterization-v1`.

## Purpose and evidence boundary

This Tier-E characterization supports the separately gated ensemble-v3 public
ship decision in `NEXT_STEPS.md` section 4.2. It describes the private release
candidate honestly; it is not M2, M4, a shipping certificate, or authority to
expose the public API, change a default, cut a release, use fresh data, or open
a lockbox.

The campaign has two evidence sources:

1. immutable M3b r3 artifacts provide the already-spent 13-case quality and
   historical cost record; and
2. one new current-source campaign measures fit time, aggregate process-tree
   RSS, safe-NPZ bytes, and integrated public prediction throughput on four
   frozen medium general cases.

The historical quality result is not rerun. A named exactness test proves that
the private release-candidate helper uses the same sampling and member-policy
mechanics as the historical combined B1+B2 arm; its different metadata marker
does not alter predictions.

## Source and input pins

- DarkoFit model source: clean published commit
  `c5e66ef7e6bdcf5665b55b81c6b870f42d76237b`.
- ChimeraBoost comparator: clean commit
  `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`, the plan's pinned 0.18 source
  (`v0.18.0-6-gf14be60`), not a moving branch or later release.
- M3b r3 result, quality, timing, and dated versus-single readout are bound by
  exact SHA-256 in the generated contract.
- The M3b case builder, M6 adapter, public ensemble-v3 contract, private
  release-candidate implementation, and named tests are likewise hash-bound.

Formal workers import models only from separate clean detached source trees at
the two exact commits. The harness itself also runs from a clean committed
checkout and verifies all bound files before and after execution. Caches and
create-only outputs live outside all three checkouts.

## Quality uncertainty from immutable evidence

The point estimate is the combined B1+B2 versus-single readout already recorded
by M3b r3: 13/13 wins, with nine sports and four general cases. The analysis
must reproduce the stored all-case, sports, and general geometric means before
adding uncertainty.

Sports uncertainty respects the dependence structure. For each of the three
seasons, first take the geometric mean across its three targets. Then draw
100,000 samples of three season clusters with replacement using seed
`20260720`; report the three season ratios, the bootstrap 2.5th/50th/97.5th
percentiles, and all three leave-one-season-out geometric means. This is
descriptive clustered dispersion over three spent seasons, not an inferential
claim about unseen seasons or nine independent datasets.

General uncertainty uses the four fixed medium cases. Draw 100,000 samples of
four cases with replacement using seed `20260721`; report the descriptive
2.5th/50th/97.5th percentiles, sample standard deviation of log ratios, and
all four leave-one-case-out geometric means. The seeded 75/25 cases are not
presented as four population-random datasets.

## Current performance grid

The frozen cases reuse M3b's exact medium, seed-0, stress-weighted general
views and split seed `20260720`:

- `general_friedman_numeric` (regression);
- `general_categorical_reg` (categorical regression);
- `general_numeric_binary` (binary classification); and
- `general_categorical_multiclass` (categorical multiclass).

The three arms are:

1. `darkofit_single`;
2. `darkofit_ensemble_v3`, the private eight-member release candidate with
   fixed 0.8 without-replacement row sampling and `donor_balanced_v1`; and
3. `chimeraboost_0_18_single`, its shipped quantized single model.

All arms use seed 4, 14 total threads, 600 maximum iterations/estimators,
30-round early stopping, and a 0.15 random validation fraction where the
single-model API owns validation. DarkoFit v3 uses its OOB complement instead.
No ensemble member parallelism is enabled. Each formal case/arm/block runs in
a fresh interpreter after a same-task, same-arm two-iteration warmup outside
measurement.

There are three complete blocks. Arm order rotates by case and block so each
arm occupies each position. No failed worker is silently retried and no partial
raw result is published as a completed characterization.

### Fit, RSS, and archive telemetry

For every worker, a 10 ms sampler records the sum of resident bytes for the
worker and all recursive child processes during formal fit. The row records
the pre-fit process-tree RSS, peak fit RSS, peak-minus-start delta, sampler
count/errors, fit wall time, fitted member/tree/thread metadata, and warnings.

DarkoFit arms are saved through the safe-NPZ path, loaded, and required to
produce array-identical predictions before archive bytes are accepted.
ChimeraBoost pickle bytes are telemetry only and are not compared with safe
NPZ as if the formats were equivalent.

The report shows per-case three-block series and equal-case geometric means
for v3/single fit time, absolute peak RSS, positive peak-RSS delta, and
safe-NPZ bytes. RSS ratios and absolute deltas are both shown; no tiny
denominator is allowed to stand in for absolute memory harm. The historical
13-case self-worker RSS and cost record remains labeled with its narrower
scope rather than being relabeled process-tree evidence.

### Integrated public prediction throughput

Each fitted model predicts deterministic repeated test rows at exactly:

```text
8,192 / 65,536 / 524,288 / 2,000,000 rows
```

For each batch, input construction, one complete public `predict` warm call,
and hashing are outside timing. The warm-call duration selects a deterministic
loop count for that worker/batch:

```text
calls = clamp(ceil(1.0 second / warm-call seconds), 2, 8192)
```

The formal timer surrounds that many complete public `predict` calls. The
interval must last at least 0.75 seconds. Per-call seconds and rows/second are
reported; the final output must be array-identical to the warm output. The
chosen call count, warm duration, full interval, input hash, output hash, and
method (`predict`) are retained.

For every case/batch coordinate, analysis pairs arms within block and reports
median ratios plus IQR/median for:

- DarkoFit single / ChimeraBoost 0.18 single;
- DarkoFit v3 / ChimeraBoost 0.18 single; and
- DarkoFit v3 / DarkoFit single.

It also reports equal-coordinate geometric means, throughput medians, and
counts above/below parity. These are characterization statistics, not pass/fail
gates. Instability, interval-duration misses, and outliers remain visible and
cannot trigger a favorable rerun.

## Artifact discipline and claims

The freezer writes one create-only JSON contract that binds this protocol,
runner, analyzer, tests, inputs, source pins, grid, environment, and
aggregation rules. The parent writes one create-only raw artifact or, on a
post-preflight failure, one create-only terminal record. The analyzer writes
one create-only result JSON and one result note whose hashes are recorded in
the testing log.

The result may say only what these spent/frozen cases and this hardware show.
It may not call ensemble-v3 generally superior, sports-safe outside the panel,
prediction-certified, released, or public. Eight members remain the only
evaluated recipe, not an optimized default.
