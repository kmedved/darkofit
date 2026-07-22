# v0.11 private ensemble evidence protocol

_Prospectively frozen before any v0.11 release-candidate outcome is opened._

Contract identity: `v011-private-ensemble-evidence-v1`.

## Authority and boundary

This Tier-E campaign executes Phase 1 of
`benchmarks/v011_evidence_phase_instruction_20260721.md`. It characterizes the
still-private eight-member ensemble-v3 release candidate. It does not expose a
public API, change a default, authorize M4, cut v0.11, use fresh-confirmation or
lockbox data, or begin a post-release mechanism.

The complete list of conditions that may block later public exposure is:

1. a correctness failure; or
2. unresolved failure to reproduce the immutable M3b r3 combined-arm result.

Every timing, memory, archive-size, dispersion, and quality-uncertainty result
is disclosure, not a gate. No post-outcome acceptance bar may be added.

## Pins and isolation

- DarkoFit model source: published commit
  `543604dd9860a28c30912f914b2cfccfcb99d783`.
- ChimeraBoost: commit
  `f14be606b641f1bf0dc92bb14b3951f1fe631c6b`
  (`v0.18.0-6-gf14be60`), never a moving checkout.
- CatBoost: exact distribution version `1.2.10`.
- Hardware: one exclusive machine; 14 total threads per worker.

The formal harness, DarkoFit model source, and ChimeraBoost source are separate
clean detached worktrees. Formal artifacts are written create-only outside all
three worktrees. Source state and the CatBoost distribution version are checked
before every worker and again before publication. Workers are fresh Python
processes and set the same BLAS, OpenMP, and Numba thread ceilings.

## Part A: reproduction, quality, and cost

The case grid is exactly the immutable M3b r3 grid:

- nine sports cells: seasons 2014, 2015, and 2016 crossed with box plus/minus,
  game score, and minutes per game; and
- four fixed medium general cells: Friedman numeric regression, categorical
  regression, numeric binary classification, and categorical multiclass.

The sports view is player-disjoint cold-player scoring within held teams and
uses group-disjoint ensemble sampling. The four general cells retain their
seeded 75/25 splits and stress weights. Case manifests and hashes are frozen.

Three complete fresh-worker blocks run these arms:

1. `darkofit_single`;
2. `darkofit_existing_bootstrap8`, the existing eight-member row/group
   bootstrap ensemble with no member policy; and
3. `darkofit_ensemble_v3`, the private release candidate: eight sequential
   members, 0.8 without-replacement row/group sampling, and
   `donor_balanced_v1` member policy.

All use 600 maximum iterations, 30-round early stopping, seed 4, and 14 total
threads. Singles own a 0.15 validation split; ensembles use each member's OOB
complement. Same-task/same-arm two-iteration warmup is outside measurement.
Arm order rotates prospectively by case and block.

### Reproduction stop rule

For every case and every block, compute the release-candidate primary-loss
ratio to the matched single. The immutable expected ratios are read from
`m3b_ensemble_v3_r3_vs_single_readout_20260721.json`. The frozen tolerance is:

```text
abs(current_ratio - immutable_ratio) <= 1e-10
```

This is an absolute ratio band, not a quality threshold. The same band applies
to the pooled, sports-only, and general-only geometric means. It is deliberately
near exact because the promoted dispatch is behavior-exact and the release
candidate only wraps the same fixed-seed mechanism. A miss is implementation
divergence: the campaign writes a terminal record and stops. A successor run is
allowed only after a named implementation or harness correction under a new
contract identity; the tolerance and immutable expected values may not change.

### Quality uncertainty

Point estimates are equal-cell geometric means of release-candidate/single
primary-loss ratios. The nine sports cells are not treated as independent:
within each season, first take the geometric mean over its three targets, then
run 100,000 seeded (`20260720`) bootstrap resamples of the three season
clusters. Report percentiles 2.5/50/97.5 and all three leave-one-season-out
values.

For the four general cells, run 100,000 seeded (`20260721`) case bootstrap
resamples and report percentiles 2.5/50/97.5, log-ratio dispersion, and all four
leave-one-case-out values. The report must call these four fixed seeded cases,
not population-random independent datasets. Eight members is the only evaluated
recipe.

### Cost telemetry

Each row records fit wall time, safe-NPZ bytes, fitted tree/member/thread
metadata, and process-tree RSS sampled every 10 ms. RSS scope is the worker plus
recursive children. Report absolute peak, start, positive peak-minus-start,
and sampler errors. Ratios never replace the absolute values. Safe-NPZ load
must preserve predictions and probabilities array-exactly before archive bytes
are accepted.

Report three-block paired series and dispersion for v3/single and v3/existing
bootstrap. Archive size is telemetry under the retracted size gate: roughly
5.5x single and roughly 0.5 MB was historically typical, but neither number is
a current pass/fail bar.

## Part B: dedicated prediction-throughput grid

The frozen training-row dimension is the four M3b medium general cases, whose
exact fit-row counts and hashes are stored in the contract. Each fitted model
predicts repeated copies of its own held-out input at exactly:

```text
8,192 / 65,536 / 524,288 / 2,000,000 rows
```

The five arms are:

1. `darkofit_single`;
2. `darkofit_ensemble_v3` through the private helper;
3. `chimeraboost_0_18_single`;
4. `chimeraboost_0_18_ensemble8`; and
5. `catboost_1_2_10_single`.

DarkoFit and ChimeraBoost use 600 maximum trees, 30-round early stopping, seed
4, 0.15 internal validation where the single API owns validation, and 14 total
threads. DarkoFit v3 uses OOB validation. ChimeraBoost's eight-member shipped
ensemble uses its normal 0.8 member sampling while respecting the same total
thread budget. CatBoost uses 600 maximum trees, seed 4, 14 threads, and a fixed
seed-4 stratified/non-stratified 0.15 validation split with 30-round early
stopping and best-model retention. These validation details are disclosed;
this grid characterizes fitted product paths and is not a quality comparison.

There are three complete blocks. Case-specific cyclic arm orders spread the
five arms across positions; the exact order table is frozen. Each case/arm/block
uses a fresh worker after a same-task/same-arm two-iteration warmup.

For each batch, construction and hashing are outside timing. Timing is:

1. one full public `predict` warm call;
2. five additional untimed pilot calls;
3. the median pilot duration chooses
   `calls = clamp(ceil(2.0 seconds / pilot_median), 3, 65536)`; and
4. one formal interval containing those complete public `predict` calls.

The formal interval must last at least 1.0 second. A shorter interval is a
measurement-integrity failure, not an unfavorable result and not rankable
data. The harness fails closed and requires a new successor identity rather
than adjusting the floor after inspection. Warm, pilot, and final outputs must
be array-identical. Store every pilot duration, call count, full formal
interval, seconds/call, rows/second, input/output hashes, and warnings.

Analysis pairs arms within case, batch, and block. It reports the full series,
median paired ratios, median absolute throughput, IQR/median, equal-coordinate
geometric means, and parity counts for at least: each engine single versus
DarkoFit single; both eight-member ensembles versus DarkoFit v3; and DarkoFit
v3 versus DarkoFit single. No value is a certificate or pass/fail bar.

## Artifact and rerun discipline

The freezer creates one contract binding this protocol, runners/analyzer/tests,
the authorization, relevant implementation/data files, the immutable readout,
source pins, case manifests, order tables, timing rules, uncertainty rules,
claims, and installed comparator version. The parent writes either one complete
create-only raw artifact or one create-only terminal record. A failed worker is
never silently retried; partial rows are discarded and counted in the terminal
record. Analysis writes create-only JSON and Markdown results. Every material
run receives one 12-field `TESTING_LOG.md` entry.

Outcomes may support only hardware- and grid-scoped descriptive statements.
They may not be described as M2/M4, a public API, a default change, a release,
fresh confirmation, lockbox evidence, or general superiority.
