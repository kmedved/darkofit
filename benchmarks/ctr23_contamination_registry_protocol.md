# CTR23 contamination and panel-freeze protocol

_Status: registry-construction protocol. This process may inventory data and
official split indices, but it must not fit a model or read any CTR23 benchmark
score, prediction, or result artifact._

## Scope

OpenML suite 353 contains 35 regression tasks. The intended later experiment
compares DarkoFit with ChimeraBoost, so independence is assessed against the
union of known exposure in both development histories. A task exposed to
either side cannot provide neutral confirmation of their comparison.

The source-reviewed declaration contains:

- 12 hard exclusions from DarkoFit `main` history;
- four additional hard exclusions from the separate historical ChimeraBoost
  comparator lineage;
- one Puma-family ambiguity from the comparator lineage, excluded fail-closed;
- 18 eligible tasks before fingerprint alarms.

The historical ChimeraBoost commits used for the five comparator-only
judgments are not ancestors of DarkoFit `main`; the declarations label that
model scope explicitly. No other current CTR23 task was found in either known
history.

## Frozen inputs and outputs

[`ctr23_contamination_sources.json`](ctr23_contamination_sources.json) pins the
suite ID, all 35 task IDs and normalized names, manual lineage judgments,
expected statuses, target-blind near-match thresholds, registry scope, and
allocation tie seed. It also freezes the exact Python, OpenML, NumPy, pandas,
SciPy, PyArrow, and liac-arff versions; the builder refuses a different runtime
because matching plus Parquet and split-ARFF decoding participate in the signed
result. Runtime validation precedes OpenML fetching as well as task, evidence,
or matching work.

[`ctr23_manual_evidence_catalog.json`](ctr23_manual_evidence_catalog.json)
binds every manual judgment to either a fetched OpenML source-task record, a
SHA-256-pinned repository file and exact claim literal, or a self-contained
historical Git snapshot. Historical snapshots include full commit and blob
identities plus raw commit/tree membership proofs, so verification does not
depend on a local branch, object database, or network fetch.

[`build_ctr23_contamination_registry.py`](build_ctr23_contamination_registry.py)
emits three separately hashed files:

1. `ctr23_suite_snapshot.json`: task/dataset versions, roles, source metadata,
   ingestion-artifact and OpenML hashes, full semantic fingerprints, and every
   official split-coordinate digest.
2. `ctr23_contamination_registry.json`: one status and evidence trail per task.
3. `ctr23_partition.json`: the exact confirmation/lockbox task assignment,
   allocation metadata, objective, diagnostics, and bundle hash.

The builder refuses to overwrite any artifact. `--verify-existing` rebuilds
all three and requires byte equality.

## Fingerprinting boundary

The builder downloads features and targets only inside the isolated
fingerprint function. A target contributes two opaque digests: its value
multiset and the target-marked full-table digest. No target mean, variance,
skew, quantile, missingness, or other marginal is returned, printed, committed,
or accepted by the allocator.

Feature fingerprints include:

- the OpenML-declared ARFF MD5 and the SHA-256 of the actual cached ingestion
  artifact;
- normalized provenance URL and task/dataset metadata hashes;
- a full-row feature-table digest and a target-marked table digest;
- per-column full value-multiset digests;
- cardinality-aware bottom-k/KMV value sketches, a joint unlabeled-row sketch,
  and a three-lane commutative full/one-feature-deletion row-sketch deck with
  no-false-negative Bloom membership witnesses, used only to raise
  near-lineage alarms.

The semantic feature hash is invariant to row order, global feature order, and
feature renaming, while preserving duplicate rows. Numeric canonicalization is
lossless for integers and uses exact binary rational tokens for finite floats
through float64; exactly representable integer/float values share a token
without collapsing adjacent integers above the float64 precision limit.
Complex and wider floating dtypes are rejected rather than downcast. Tied
column marginals are ordered using target-blind row anchors; any remaining nonidentical tie fails
closed instead of being hashed as exact identity. Target transforms leave the
feature-only digest unchanged and change the target-marked digest. Near-match
columns use a common textual representation across bool/categorical/string
dtypes and stable maximum bipartite matching. Low-cardinality numeric columns
remain eligible for exact-multiset matches, while the joint row sketch keeps
binary matrices and row subsets visible. Equal-width schemas compare their
full-schema views; when schemas differ by exactly one feature, the smaller
full-schema view is checked against every deletion view of the larger schema.
The lower unique-row-cardinality view's deterministic sample is queried
against the higher-cardinality view's Bloom membership witness; equal unique
cardinalities require containment in both directions. Total-row ratio remains
the separate policy gate. Thus an exact unique-row subset cannot evade the
alarm by omitting the parent's bottom-k sample or by duplicate-expanding back
to the parent's total row count. Bloom false positives can only fail closed.
The maximum and all tied winners are recorded without a positional choice.
`schema_row_comparison_supported` records full-view or deletion-view coverage;
`schema_deletion_supported` is true only for the one-feature-difference case.
Equal-width feature replacement and wider schema drift remain explicitly
unsupported by that alarm. Because unlabeled low-cardinality
sketches can conservatively collide on unrelated binary tables, a positive
near-match is never
evidence of cleanliness; it produces `ambiguous`, which is ineligible until a
reviewed declaration resolves it.

Manual lineage remains authoritative where exact hashes are expected to miss:
curated subsets (Miami), transformed targets (California/Houses), concatenated
components (red and white Wine), and different-target arrangements
(Geographical Origin of Music).

## Official split inventory

The builder uses the task objects and split ARFFs, not suite prose. The live
inventory has 800 official coordinates:

- 30 tasks use one repeat of ten-fold cross-validation (10 coordinates each);
- tasks 361617, 361618, 361619, 361621, and 361622 use ten repeats of ten-fold
  cross-validation (100 coordinates each).

The suite description's claim that large tasks use a 33% holdout is stale and
must not drive execution. For every coordinate, the snapshot verifies that
train and test indices are disjoint and cover all rows once. Within every
cross-validation repeat, test folds must partition all rows exactly once.

This registry freezes the complete inventory. The later execution protocol
must separately freeze an exact, compute-feasible coordinate subset before the
confirmation panel opens. No coordinate may be selected from model outcomes.

## Target-blind panel assignment

Every eligible task has a source-reviewed lineage declaration independent of
the exclusion list. The allocator treats those clusters atomically and
exhaustively enumerates every size-balanced confirmation assignment. It rejects assignments
whose per-resampling-regime task counts differ by more than one. Among feasible
assignments it minimizes, lexicographically:

1. official-coordinate-count imbalance;
2. maximum standardized imbalance across declared metadata;
3. sum of squared standardized metadata imbalances;
4. a SHA-256 tie breaker over the sorted confirmation task IDs.

The declared metadata is limited to log row count, log raw-predictor count,
categorical presence, predictor-missingness presence, and a compute proxy based
on rows, predictors, and official coordinate count. The allocation function
does not accept a target.

## Lockbox enforcement required later

The registry alone does not authorize a CTR23 run. A later runner must bind the
frozen model profile, exact coordinates, comparator versions, all three
registry hashes, runner/analyzer sources, dependency lock, hardware/resources,
and ordered job grid in a zero-start manifest.

The lockbox job enumerator must additionally require a passing confirmation
attestation bound to those exact hashes. Any profile, representation, resource,
registry, coordinate, runner, analyzer, or dependency change invalidates the
attestation and keeps the lockbox closed.
