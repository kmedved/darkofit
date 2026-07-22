# Automatic linear-selector v2 development contract

_Frozen on 2026-07-22 after the Phase B-1 engagement check and before any v2
implementation or outcome inspection._

Contract identity: `automatic-linear-selector-v2-development-20260722`.

## Question and evidence boundary

Can DarkoFit turn the previously validated 3% local-linear selector into a
safe automatic per-dataset policy that closes the verified Protein gap without
changing unrelated datasets?

This campaign uses only spent development evidence. Its control source is
published commit `b11f013f7ba926e533c38db8261f1a569ebce6c6` (the full commit is
verified by the execution harness). It may build and rank an isolated
candidate, but it cannot merge a default change, ship a public `"auto"`
surface, access fresh confirmation, TabArena, or a lockbox. A later default-on
decision requires separate owner authorization and a prospectively frozen,
design-time-powered Tier-D confirmation.

The old `fresh_selector_confirmation.json` is now spent. It may size and test
the development rule, but it is not reused as new confirmation. The obsolete
win-count and minimum-effect verdict do not apply. M6 v3 is the only eligible
general development-ranking rule.

## Exact candidate policy

The private candidate extends regression `linear_leaves` to accept
`False`, `True`, or `"auto"` and uses `"auto"` as the candidate branch's
default. This is a research branch, not a manual interim product. Explicit
`False` and `True` always win and retain their v0.11 behavior.

For eligible automatic fits:

1. when the caller did not supply `eval_set`, use one deterministic 20%
   selection holdout;
2. for that automatic holdout, keep groups disjoint when groups are supplied,
   otherwise use DarkoFit's
   weighted target-stratified regression split (including sample weights);
3. fit constant- and linear-leaf auditions on identical selection rows,
   validation rows, weights, feature declarations, and constructor policy;
4. when the caller supplied an `eval_set`, use those rows instead of creating
   the automatic holdout;
5. select linear leaves only when
   `(constant RMSE - linear RMSE) / constant RMSE >= 0.03`; ties and smaller
   gains select constant leaves; and
6. fit the selected leaf family from scratch with the caller's original
   full-fit semantics.

The selector is eligible only for a single scalar RMSE CatBoost-mode regressor
without another automatic model-family audition, distributional calibration,
interval calibration, a linear residual, callbacks, or an ensemble. An
ineligible automatic request resolves to exact constant-leaf behavior and
records a stable reason; because automatic mode is the candidate default, it
must not turn an otherwise valid fit into an error. Existing explicit
`linear_leaves=True` errors and fallbacks remain unchanged.

Every automatic fit records the resolved boolean, reason, split provenance,
constant and linear validation scores, relative margin, threshold, selection
fit horizons, and total selection cost. Serialization must preserve that
record and the requested `"auto"` policy. `get_refit_params()` resolves the
already-selected boolean rather than auditioning again.

## Invariants before evidence

The implementation is ineligible for any quality run until tests establish:

1. explicit `False` and `True` are behavior-identical to the control source;
2. a selected automatic final fit is prediction- and state-identical to a
   direct explicit fit of the selected boolean on the same full data;
3. declining the mechanism is identical to `linear_leaves=False`;
4. selection rows and validation rows are disjoint, group-disjoint when
   groups exist, deterministic for the fitted seed, and correctly subset
   sample/eval weights;
5. unsupported modes fall back loudly in fitted metadata without failing;
6. selection failure restores a previously fitted estimator transactionally;
7. fit and predict restore the caller's ambient thread-local Numba mask,
   including the two nested audition fits;
8. clone/get-params semantics, safe-NPZ round trips, repeated round trips,
   corruption rejection, feature names, categorical/ordinal inputs, and
   empty prediction batches remain valid; and
9. no classification or ensemble behavior changes.

## Frozen spent-development sequence

1. **Invariants and mechanism synthetics.** Implement the smallest candidate
   and pass the invariants above plus the relevant M5 correctness sentinels.
2. **M6 v3 inspection 1.** Run the exact 60-cell medium paired grid from
   `m6-quality-successor-v3`, control versus candidate default, with mechanism
   id `automatic_linear_selector_v2` and inspection index 1. Its aggregate,
   worst-dataset, and leave-one-dataset-out rule is the only general ranking
   stop. All cells and engagement reasons are reported; no win count applies.
3. **Protein attribution.** Only after M6 advances, run the three already-spent
   release-ladder Protein coordinates with constant, automatic, and explicit
   linear DarkoFit arms. The automatic arm must select linear on every
   coordinate and be behavior-identical to the explicit-linear final fit.
   Report quality, fit, prediction, RSS, selection margin, and worst
   coordinate. There is no minimum-effect gate; aggregate or coordinate harm
   above the standing Tier-D 1.02 bound closes this exact candidate.
4. **Historical guardrail replay.** Recompute—not rerun—the modern aggregate,
   worst-lineage, split, and leave-one-lineage-out readout from the old smooth,
   categorical, noisy-tabular, and group-safe sports artifacts. This is a
   consistency check and must disclose dependence and prior outcome knowledge.

M6 or Protein failure is terminal for this identity. If all spent-development
steps pass, the terminal disposition is `ready_for_powered_fresh_design`, not
`ship`: the candidate branch and artifacts are retained, main stays unchanged,
and fresh access remains a separate owner decision.

## No-rerun and reporting rules

Every material execution uses clean committed control/candidate sources,
fresh workers, exact source hashes, create-only raw/result/manifest artifacts,
and a 12-field `TESTING_LOG.md` entry. Failed or inspected attempts consume
their inspection index. Timed work requires the exclusive-machine preflight.
No failed gate may be relaxed after inspection. Costs are adjacent telemetry,
not ratio gates. Airfoil is excluded from the selector causal claim.
