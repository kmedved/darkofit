# Basketball fused-oblivious-kernel protocol

## Decision being tested

This campaign asks whether one fused histogram-build and shared-split kernel
can materially reduce DarkoFit's basketball fit time while preserving the
current model exactly. It is an engine refactor, not a quality candidate:
every split, fitted value, prediction byte, and sports guardrail score must
remain unchanged.

Basketball is the first fatal gate because it is fast, directly represents the
user's primary noisy sports-data regime, and currently spends about 99% of fit
time in tree construction. A passing result advances the kernel to broader
behavior coverage; it does not authorize deletion of the reference kernels or
claim universal speedup.

## Comparator refresh

The synced ChimeraBoost checkout is clean at version 0.15.0, commit
`851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d`. On the unchanged creator folds,
its default produced mean R² 0.526981 in 7.52 seconds after warmup. Current
DarkoFit produced mean R² 0.526750 in 26.71 seconds. The quality difference is
only +0.000232 R² (ChimeraBoost won 6 of 10 folds), but DarkoFit was 3.55 times
slower.

A diagnostic ChimeraBoost ablation found that disabling cross features
improved its basketball mean to 0.528023, while disabling both cross features
and linear leaves reduced it to 0.519069 but completed the folds in 2.46
seconds. Cross features are therefore not the next sports candidate. Linear
leaves explain the comparator's quality, but DarkoFit's independent linear
leaf selector already failed the frozen team and cold-player gates. The
remaining actionable basketball gap is engine speed.

These comparator timings are diagnostic rather than a new creator-baseline
claim: the existing runner correctly labels the one-arm steady run ineligible
for its frozen 0.14.2 author-lane baseline.

## Candidate scope

The candidate fuses the two existing parallel operations used by the default
basketball lane:

1. `_build_histograms_unit_hess_into`; and
2. `_best_split`.

The fused kernel uses one feature-parallel launch per tree level. Within each
feature, it builds that feature's histogram and immediately executes the
current split scan over the same buffers. The split scan must preserve the
current loop order, empty-child legality, tie policy, `min_child_weight`
semantics, and floating-point operation order.

Initial eligibility is deliberately narrow:

- oblivious/`tree_mode="catboost"` scalar trees;
- more than one Numba thread;
- constant Hessian;
- all rows and all features;
- no root-histogram injection;
- no level histogram subtraction;
- no row-parallel histogram buffers; and
- `random_strength == 0`.

Every other lane continues to call the existing kernels. The campaign toggles
the candidate through a private benchmark-only builder argument; it adds no
public estimator parameter and changes no saved-model format. The readable
and current optimized kernels remain exact-equality oracles.

The small LU/linear-leaf attribution already recorded in `NOTICE` is separate.
If the fused implementation substantially adapts ChimeraBoost source or
design, `NOTICE` must be extended with the exact Apache-2.0 commit provenance
before shipping.

## Behavior gates

Before timing:

- direct fused-versus-current kernel tests must compare selected feature,
  threshold, gain, active histogram cells, split routing, leaf totals, and
  predictions exactly across depths, feature masks, varied per-feature bin
  counts, empty leaves, `l2=0`, and boundary `min_child_weight` cases;
- `DARKOFIT_STRICT_GOLDENS=1` must pass the complete prediction-golden suite;
- the standalone readable oblivious oracle must pass;
- the complete test suite must pass; and
- unsupported/ineligible lanes must prove that the fused function was not
  called.

The clean basketball campaign then runs the unchanged ten folds plus the
overlap-exposed team holdout and 585-row cold-player subset. Candidate and
default must have identical prediction hashes on every fold and both
guardrail views. Mean R², every fold R², team R², cold-player R², selected tree
mode, resolved learning rate, fitted tree count, stop reason, split importance,
and serialized model bytes must be identical. Any mismatch is a fatal failure
and skips timing confirmation.

## Timing and resource gates

After an untimed complete-fold warmup for each fresh worker, run three
reciprocal blocks:

1. default, fused;
2. fused, default; and
3. default, fused.

All non-timing behavior fingerprints must remain identical within and across
arms. Each arm's steady max/min wall-time ratio must be at most 1.20. The fused
arm advances only if:

- median summed fit time is at most 0.85 times default (at least 15% faster);
- median steady wall time is at most 0.85 times default;
- median prediction time is at most 1.02 times default;
- serialized model bytes are exactly equal; and
- median fresh-worker peak RSS is at most 1.10 times default.

A missing measurement fails closed. The long-run Phase 3 target remains about
10 seconds for the ten folds at equal tree counts; this isolated kernel need
not reach that target alone, but a smaller or unstable gain does not justify
adding another production dispatch path.

## Advance path

If the candidate fails exactness or the basketball speed gate, stop and remove
the fused production path while preserving the protocol/result as research
evidence. If it passes, expand exact tests to weighted RMSE, categorical
preprocessing, binary classification, alternate scalar losses, callbacks,
early stopping/refit, and every supported thread count before making the fused
lane automatic. Only after those gates may the superseded optimized kernel be
demoted to a test oracle; no kernel-matrix deletion is part of this campaign.

No CTR23 development coordinates or lockbox data are used in this phase.
