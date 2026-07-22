# Automatic linear-selector v2 M6 engagement companion

_Frozen before M6 inspection 1 and before any M6 v3 quality outcome exists._

Identity: `automatic-linear-selector-v2-m6-engagement-20260722`.

The immutable M6 quality-successor-v3 runner records strict generic fitted
metadata but predates `automatic_linear_selector_`; its raw CSV therefore
cannot report the engagement reasons required by the selector development
contract. This companion closes only that observability gap. It does not edit
the frozen M6 runner, paired-execution foundation, decision rule, backtest, or
historical artifacts.

Before inspection 1, the companion runs the candidate arm once on each of the
exact 60 M6 v3 dataset/size/seed/weight cells. It reuses the same builders,
splits, public-default policy, four-thread worker environment, implementation
path checks, prediction/probability validation, and fitted-model attestation.
One repeat is sufficient because selector decisions and fitted state are
deterministic for each frozen seed; timing is neither retained nor ranked.

The create-only output contains cell identity, data/split/weight hashes, and
the complete fitted automatic-selector record for regression. Classification
cells must have no selector state and are recorded as
`classification_not_applicable`. The output deliberately excludes primary
quality metrics, predictions, benchmark fit/predict timings, RSS, and any
acceptance decision. The selector record's own audition-cost fields remain as
required provenance. The companion cannot rank, kill, ship, change a default,
or access fresh/TabArena/lockbox evidence.

Both candidate and harness must be clean committed sources and remain
unchanged. Any failed companion attempt may be fixed before M6 because it
reveals no quality outcome, but no M6 inspection may launch until one complete
60-cell companion artifact exists. Once M6 inspection 1 launches, its usual
no-rerun and terminal rules apply unchanged.
