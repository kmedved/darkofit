# Basketball OOB ensemble: quality passes, timing is inadmissible

## Decision

Do not add `n_ensembles` to the public API from this run. The five-member OOB
ensemble passed every preregistered basketball quality gate and stayed well
inside the feature-specific median runtime budgets, but both arms failed the
predeclared timing-stability bound. The formal recommendation is therefore
`advance_none`.

This is an evidence-quality stop, not a quality rejection. The result is
promising enough to retest under a separately frozen, thermally stable timing
campaign or on another machine. It does not authorize discarding the slow
block, weakening the stability gate after the fact, or promoting ensembles as
a default.

## Quality

| Arm | Mean 10-fold R² | Fold wins | Overlap-exposed team R² | Cold-player R² |
|---|---:|---:|---:|---:|
| Current default | 0.526750 | — | 0.531269 | 0.500434 |
| OOB ensemble 5 | **0.530625** | **6 / 10** | **0.537395** | **0.519783** |
| Candidate delta | **+0.003876** | — | **+0.006126** | **+0.019349** |

All five quality gates passed. The candidate won exactly 6 of 10 folds. Its
leave-one-fold-out mean deltas remained positive in every case, from +0.001652
to +0.005624, so the aggregate gain does not depend on any one fold. It also
improved both the player-overlap-exposed team holdout and the stricter 585-row,
210-unseen-player cold subset.

## Position against the frozen comparators

The OOB ensemble closes part of DarkoFit's quality gap, but it does not reach
the current ChimeraBoost ensemble or CatBoost on this dataset:

| Frozen arm | Mean 10-fold R² | DarkoFit OOB-5 delta | Fold record |
|---|---:|---:|---:|
| ChimeraBoost default | 0.524826 | **+0.005799** | 6W / 4L |
| DarkoFit OOB ensemble 5 | **0.530625** | — | — |
| CatBoost default | 0.536308 | -0.005683 | 3W / 7L |
| ChimeraBoost ensemble 5 | **0.540159** | -0.009534 | 1W / 9L |

Thus the mechanism recovers about 29% of the R² gap from the current DarkoFit
default to ChimeraBoost ensemble 5. It is useful, but not the mechanism that
gets DarkoFit to the benchmark target by itself.

## Fitted behavior

- Five deterministic member seeds were derived from seed 4 for every external
  fit; each member drew one training-sized bootstrap and used only its exact
  OOB complement for validation.
- External-fold OOB sets contained 1,681–1,750 rows. Bootstrap and OOB index
  hashes, member prediction hashes, averaged predictions, and fitted metadata
  were identical across all three fresh-process repetitions.
- Every member resolved automatic learning rate to 0.064960–0.064962 and
  stopped on OOB validation. Retained tree counts ranged from 201 to 814.
- No member refit, no public estimator parameter changed, and no lockbox data
  was used.

## Timing

| Arm | Three steady times | Median | Max / min | Gate |
|---|---|---:|---:|---:|
| Current default | 28.27s, 29.69s, 35.93s | 29.69s | 1.271 | fail |
| OOB ensemble 5 | 65.23s, 67.30s, 80.90s | 67.30s | 1.240 | fail |

Both arms slowed together in the final reciprocal block. Their median ratio
was 2.267×, comfortably below the feature-specific 4.0× wall-time budget.
Measured prediction time was stable and the candidate/default median ratio was
2.303×, below the 6.0× budget. Nevertheless, the protocol requires each arm's
steady max/min ratio to be at most 1.20, so the timing evidence is
inadmissible and the full gate fails.

The prior failed attempt was not used: it revealed that
`warmup_seconds_outside_timing` was incorrectly included in behavior hashes.
That shared harness bug was fixed and tested in commit `aa8f71b`; this artifact
was then regenerated from that clean committed source.

## Next step

Do not implement the API yet. If this mechanism is revisited, freeze a second
complete timing campaign in advance and run it after controlling machine load
and thermal state (or on an independent machine). Preserve this run as the
first campaign rather than replacing it. Advance to API work only if the new
campaign is stable and reproduces the same quality hashes and guardrail gains.

Regardless of a future ensemble retest, the remaining benchmark gap still
requires a single-model quality mechanism—linear leaves/cross features—and
tree-building speed work. An opt-in ensemble cannot substitute for those.

## Artifacts

- `basketball_oob_ensemble.json`: full clean-source results, prediction and
  bootstrap/OOB hashes, fitted metadata, all six reciprocal worker runs, and
  every gate.
- `basketball_oob_ensemble_protocol.md`: frozen pre-run decision rules.
- `run_basketball_oob_ensemble.py`: shared-harness runner.
- `test_basketball_oob_ensemble.py`: deterministic bootstrap, OOB isolation,
  metadata, decision, and dirty-source fail-closed tests.
