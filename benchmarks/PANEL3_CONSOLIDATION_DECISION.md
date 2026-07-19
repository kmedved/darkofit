# Panel 3 pre-freeze consolidation decision

**Status:** safeguard-retention review complete; evidence freeze not yet
created.

This record closes the pre-freeze review before Panel 3 spends any calibration
or lockbox evidence. It does **not** claim that Panel 3 became smaller overall.
No spent-data calibration result, target preflight, registry, or fresh outcome
was accessed while making these changes.

## Measured result

The directive arrived when the uncommitted Panel 3 scope was reported at about
19,300 lines. The same production-and-test scope is now **24,320 lines**:
roughly **26.0% larger**, not slimmer. Most of that growth
preceded the measured consolidation slices and added publication, resume,
source-closure, and security checks. The decision here is that the retained
checks are worth their weight, not that the requested global slimming was
achieved. Here, "same scope" means every benchmark or test Python file whose
basename contains `panel3`, plus `benchmarks/campaign_lib/*.py`.

A narrower, reproducible comparison uses commit `62d8f52` as the clean
pre-consolidation baseline. Relative to that commit, the pre-H1 source is:

- benchmark production code: **139 net lines removed**;
- tests: **1,434 net lines added**; and
- combined same-scope delta: **1,295 net lines added**.

The earlier consolidation slices themselves removed 176 net lines. Subsequent
blocking reviews required order-sensitive task binding, provenance repair, and
the committed differential census. The final stopping-rule review then exposed
two more fail-closed defects: net Git diffs could hide an intermediate source
change followed by a revert, and the immutable power decision did not bind the
required owner-facing `GO` or `NO-GO` sentence. Those fixes and their tests are
included in the final numbers above rather than hidden behind an earlier
checkpoint.

The first H2 creation attempt then stopped before publication because the
source-freeze builder compared raw OpenML categorical declarations with a
historical AutoGluon child-visible ledger. Healthcare, Miami, and Wine retain
binary categorical columns in the raw task view that AutoGluon had converted
before its child adapter. The unpublished freeze schema now names and binds the
raw task-view categorical map explicitly and shares the execution resolver.

The test growth is deliberate. Differential review found real acceptance gaps
in the formerly duplicated fitted-metadata validators. The committed mutation
census binds exact integer, numeric, timing, pair-count, ratio-tolerance, arm,
and T5-applicability semantics after the duplicate validator is removed.

## Accepted cuts

1. **Shared provenance primitives.** Canonical JSON, SHA-256, Git, and
   repository-relative path helpers now live in
   `benchmarks/campaign_lib/provenance.py`. Frozen historical runners were not
   edited.
2. **One fitted-metadata authority.** The spent calibration runner retains only
   its calibration-coordinate T5 applicability boundary and delegates policy
   metadata to the confirmation analyzer's strict validator. This removed 367
   production lines. A 2,367-mutation differential found zero cases accepted
   by the new path that the old effective path rejected; all six canonical
   producer variants remained accepted.
3. **Mechanical test-fixture reuse.** Only identical full coordinates, source
   attestations, integrity refreshes, behavior fingerprints, preflight setup,
   and decision-mutation scaffolding were shared. Distinct phase and security
   fixtures remain separate. These substitutions removed 129 test lines.
4. **Complete executable source closure.** The freeze now binds
   `tabarena_adapter.py` and `tabarena_screen_adapters.py`, the two repo-local
   modules dynamically imported by already-bound TabArena sources. A recursive
   import audit reports zero missing repo-local modules in the 69-file closure.

## Why the retained safeguards stay

Every retained mechanism either caught a concrete campaign-invalidating defect
in this preparation window or closes a direct one-shot evidence failure:

- **Atomic attempts, claims, invalidations, and partial publication resume.**
  Review reproduced an early comparator crash before a worker claim and a crash
  after the first create-only publication. Without durable state, the former
  strands a coordinate and the latter makes an otherwise complete one-shot
  analysis irrecoverable. Malformed, symlinked, and raced claims were also
  exercised explicitly.
- **Immutable source, registry, and publication boundaries.** The source-closure
  audit found two dynamically imported adapter modules missing from the freeze.
  Historical-artifact validation separately caught a derived report whose
  original analyzer hash had been overwritten by the hardened analyzer hash.
  Final review also proved that a source-changing commit followed by a revert
  disappeared from a net tree diff. All three create-only campaign boundaries
  now enumerate every intervening commit and reject that history.
- **Immutable owner decision.** The machine-readable power arithmetic and
  authorization flag were already fail-closed, but the required owner-facing
  sentence was not part of the published contract. The existing create-only,
  self-hashed JSON now binds and prints the exact candidate probabilities,
  Wilson lower bounds, 0.80 floor, and `GO` or `NO-GO` verdict; no second,
  partially publishable artifact was added.
- **Order-sensitive task-view binding.** Final review demonstrated that the
  contamination fingerprint is intentionally row-order invariant: a joint
  `X`/`y` permutation preserved it while changing which observations positional
  splits selected. The preflight, registry, worker result, and analyzer now bind
  the exact ordered task view.
- **Strict fitted metadata plus differential coverage.** Differential review
  found real type, range, timing, pair-count, and applicability acceptance gaps.
  The single authority is protected by a repeatable mutation census rather than
  an uncommitted one-off result.
- **Target preflight and exclusion provenance.** Earlier preparation found
  non-finite targets, three benchmark-exposed lineages, and accidental target
  footer exposure. The target preflight checks finiteness without publishing
  target statistics, while the exclusion ledger prevents those lineages from
  re-entering the lockbox.
- **Raw task-view binding.** The attempted H2 freeze caught a category-boundary
  mismatch before writing an artifact: a historical AutoGluon child ledger had
  been applied to raw OpenML frames. The corrected contract distinguishes the
  two representations, binds all raw declared or nonnumeric categorical
  columns, and uses the same categorical resolver in freeze and execution.
- **Runtime and machine binding.** Timing evidence is valid only when all arms
  share the declared interpreter, package set, thread policy, and machine. The
  binding prevents mixed-runtime or mixed-machine measurements from being
  aggregated as a paired campaign.
- **Private diagnostics.** Review found host paths and exception text flowing
  toward durable artifacts. Worker diagnostics are now fixed codes and hashes;
  machine-local details remain private.
- **Historical validation.** Frozen artifacts must remain verifiable after
  hardening. That path exposed the analyzer-hash provenance error above and is
  therefore retained separately from live prospective validation.

These mechanisms do add code. Their justification is evidence integrity, not
reuse or aesthetic consolidation. No further safeguard or schema work is
authorized before H1 unless a red stopping-rule check forces it.

## Rejected cuts

- **Merge `build_panel3_registry.py` and `panel3_registry_common.py`.** Their
  measured `SequenceMatcher` ratio over stripped nonempty line sequences, with
  auto-junk disabled, is about 8.5%. Their normalized distinct-line
  intersection is about 11.4% of the smaller file's distinct-line set
  (Jaccard intersection over union is about 4.8%). No substantive duplicate
  function was found.
- **Phase-switch calibration and confirmation runners or analyzers.**
  Calibration is rerunnable spent-data execution. Confirmation is a one-shot
  fresh campaign with durable attempt/claim/invalidation semantics and
  nonbinding comparator failures. Combining them would hide different evidence
  boundaries behind flags.
- **Remove archival validation, machine/runtime binding, atomic claims, or
  strict selector metadata.** These are safeguards, not unused product
  features; they have already caught campaign-invalidating defects.
- **Create speculative `environment`, `spool`, `execution`, `fitters`, and
  `stats` libraries.** An exact environment-helper probe increased production
  code by 19 lines and total code with its test by 67 lines, so it was reverted.
  The other proposed modules do not yet have three isomorphic live clients.
- **Collapse tests to one cross-phase fixture.** Confirmation, spent
  calibration, power design, and registry fixtures encode different contracts.
  Sharing those objects would reduce independent evidence-boundary coverage.
- **Reduce the freeze builder to a thin provenance wrapper.** Its source
  snapshot, runtime, ancestry, and immutable-publication checks are substantive
  authorization logic.

## Freeze boundary

The commits in this review are source preparation, not an evidence freeze.
The next valid order is:

1. run the complete test and review gates;
2. commit and publish the clean source head as `H1`;
3. create the source-freeze artifact from exactly `H1`;
4. commit only that artifact as `H2`;
5. run the spent-data calibration from the clean allowed `H1..H2` history;
6. publish the preregistered power decision; and
7. access fresh Panel 3 targets only if that decision explicitly authorizes
   the one-shot campaign.
