# Basketball entity-aware ensemble result

_Run 2026-07-17 from clean `main` at `d2c14ba`, under the frozen
[`basketball_entity_ensemble_protocol.md`](basketball_entity_ensemble_protocol.md)._

## Decision

Close the five-member entity-aware ensemble as shaped. Do not run the
remaining timing blocks, add an ensemble API, or change a default.

The candidate resampled exact player identities, used group-disjoint OOB
early stopping, shared one full-external-training numeric preprocessor across
members, and never used `Player` as a model feature. It therefore tested the
materially different S2 mechanism rather than rerunning the closed row
bootstrap.

## Quality

| Metric | Control | Entity ensemble | Delta |
|---|---:|---:|---:|
| Mean creator-fold R² | 0.526750 | 0.522568 | **−0.004182** |
| Fold record | — | 2 wins / 8 losses | Fail |
| Held-team R² | 0.531269 | 0.533508 | +0.002239 |
| Cold-player R² | 0.500434 | 0.515302 | **+0.014869** |
| Seen-player R² | 0.530247 | 0.528512 | −0.001736 |

Every leave-one-fold-out mean delta was negative, ranging from `−0.001190` to
`−0.008074`. The primary mean and broad-robustness gates therefore failed,
even though the held-team and cold-player guardrails improved.

## Cost and scope

The single descriptive block took 13.96 seconds for the candidate and 10.28
seconds for control, but one block cannot establish a stable ratio. The
fatal-quality design correctly skipped the two remaining reciprocal blocks,
so this result makes no timing or memory claim.

The result suggests player-level resampling trades ordinary in-distribution
accuracy for unseen-player robustness on this dataset. That may be useful in
a differently specified research objective, but it does not beat the creator
benchmark and cannot advance in this program.

## Evidence

- Raw artifact:
  [`basketball_entity_ensemble.json`](basketball_entity_ensemble.json),
  SHA-256
  `7088170060de9124d6508f1096df7af737332ab9e3b95ed11c1058b51e79ba35`.
- Protocol SHA-256:
  `992e2ae4ab292a38531c06c4b8f8307453b72b46074bd35ca12508f1b09ba3ca`.
- Runner SHA-256:
  `d97ce9513e4d6f6216c5a07f5fc9b422e7ffaa21456f6ab8ab3395791947d6d3`.
- The artifact binds the clean source, unchanged fold and guardrail
  fingerprints, exact player draws and OOB groups, shared-preprocessor state,
  member fitted metadata, and prediction hashes. No lockbox data was used.
