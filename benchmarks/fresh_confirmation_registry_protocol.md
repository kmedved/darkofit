# Fresh confirmation registry and power-freeze protocol

## Purpose

Freeze a genuinely outcome-unseen real-data confirmation panel for the exact
3% smooth linear-leaf selector before fitting any candidate model on it.
Registry construction may read metadata, features, targets only for opaque
semantic fingerprints, and official split indices. It must not fit a model,
compute target statistics, or read a benchmark result involving a candidate
task.

## Candidate selection

The declaration contains 20 primary OpenML supervised-regression tasks from
20 source-reviewed lineage clusters:

- 14 `smooth_numeric`;
- 3 `categorical`; and
- 3 `noisy_tabular`.

Selection used task metadata, source descriptions, row/feature scale, missing
and categorical presence, and compute feasibility only. Related tasks from
the same source family are declared but not counted as independent votes.
Exactly repeat 0, folds 0–2, sample 0 are reserved for confirmation: 60
coordinates.

The 14-task smooth stratum is the primary selector panel. The categorical and
noisy strata are policy guardrails. Equal-lineage weighting is binding.

## Contamination boundary

The builder must bind and audit:

1. DarkoFit immediately before this registry, commit `a3c55d6`;
2. clean ChimeraBoost 0.15.0, commit `851ab7f`;
3. ChimeraBoost's OpenML, Grinsztajn, PMLB, high-cardinality, and TabArena-name
   benchmark universes;
4. all candidate and source semantic fingerprints in the frozen CTR23-v3
   registry; and
5. exact OpenML dataset IDs, normalized names, and conservative name
   containment with a six-character minimum.

The builder reuses CTR23-v3's row/feature-order-invariant fingerprint and
near-lineage alarm. Any exact ID/name/fingerprint match, repository source
reference, conservative name containment, near-match alarm, ambiguous feature
canonicalization, task-name drift, non-regression task, non-10-fold procedure,
or split-integrity failure excludes the task and fails the 20-task freeze
closed.

Metadata harvesting by SynthGen is not model-result exposure: its checked-in
corpus intentionally contains no dataset names, targets, or benchmark scores.
It is therefore recorded but is not an exclusion.

## Power design

Power is computed before confirmation from the already-spent 21 smooth
development ratios only. With a fixed seed and 200,000 simulations, draw 14
smooth-lineage effects with replacement from the 21 observed selector/default
split ratios. A simulated confirmation passes when:

1. equal-lineage geometric-mean RMSE ratio is at most `0.98`;
2. at least 9 of 14 lineages improve; and
3. no lineage ratio exceeds `1.02`.

The confirmation run is authorized only if simulated pass probability is at
least 80%. This empirical bootstrap is deliberately labeled conditional on
the three spent source lineages; it is not evidence of generalization.

## Non-authorization

Freezing this registry and passing its power calculation do not open the
CTR23 lockbox and do not promote a selector. The exact profile must first pass
the fresh primary and guardrail gates. Only then may the observed fresh effect
distribution be used in a separately frozen lockbox power calculation.
