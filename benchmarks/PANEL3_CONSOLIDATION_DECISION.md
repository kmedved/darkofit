# Panel 3 pre-freeze consolidation decision

**Status:** consolidation complete; evidence freeze not yet created.

This record closes the infrastructure-slimming review before Panel 3 spends
any calibration or lockbox evidence. No spent-data calibration result, target
preflight, registry, or fresh outcome was accessed while making these changes.

## Measured result

The review used commit `62d8f52` as the clean pre-consolidation baseline.
Relative to that commit, the accepted slices changed 16 benchmark and test
files by 703 insertions and 879 deletions:

- benchmark production code: **381 net lines removed**;
- tests: **205 net lines added**; and
- combined tracked delta: **176 net lines removed**.

The test growth is deliberate. Differential review found real acceptance gaps
in the formerly duplicated fitted-metadata validators. The added mutations bind
exact integer, numeric, timing, pair-count, ratio-tolerance, arm, and T5
applicability semantics before the duplicate validator is removed.

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
   import audit reports zero missing repo-local modules in the 68-file closure.

## Rejected cuts

- **Merge `build_panel3_registry.py` and `panel3_registry_common.py`.** Their
  measured normalized line-sequence similarity is 8.4%, distinct-line overlap
  is 11.3%, and no substantive duplicate function was found.
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
