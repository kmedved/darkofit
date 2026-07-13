# TabArena regression default check

_Run date: 2026-07-12. Environment: Python 3.12.13, TabArena/AutoGluon
development snapshots installed in `tabarena-darko312`, 18 detected CPUs._

> **Default-policy status:** This one-split diagnostic is superseded for
> default-policy decisions by the staged
> [TabArena regression multisplit ablation](tabarena_regression_multisplit_ablation.md).
> The root-cause analysis and one-split comparisons below are retained as
> historical provenance.

## Decision

Keep `learning_rate="auto"`. Fix the shared-split empty-child bug.

The split fix is the dominant quality correction: it reduces DarkoFit's
geometric-mean RMSE gap to the unrelated ChimeraBoost 0.13 default from 5.14%
to 1.25%. A fixed learning rate of 0.1 makes a smaller aggregate improvement,
to a 0.84% gap, but regresses four of thirteen datasets and reduces the number
of head-to-head wins from five to four. One TabArena outer split per dataset is
not sufficient evidence to remove the broader data-dependent automatic policy.

Users targeting this particular unweighted RMSE regime can still set
`learning_rate=0.1` explicitly. A public policy change should wait for a
multi-split numeric-regression guardrail showing that the gain survives across
splits without the categorical regressions seen here.

## Root cause and correction

DarkoFit's symmetric split search previously rejected a threshold when any
active leaf put all its rows on one side. In a shared/oblivious level, that is
normal: an already-pure leaf contributes zero gain while other leaves can still
benefit from the split. Treating the empty child as a minimum-child-weight
violation prematurely capped nominal depth-six trees near depth three or four.

The corrected rule is applied consistently to the parallel, serial, noisy, and
count-aware hybrid shared-trunk split searches. Empty children contribute zero;
sparse non-empty children still obey `min_child_weight` and, where applicable,
`min_child_samples`. Per-leaf builders retain strict empty-child rejection.

For unweighted RMSE, `min_child_weight=0` is an exact proxy for the corrected
default semantics because every non-empty child has integer Hessian mass of at
least one. The actual patched `min_child_weight=1` run matched that proxy on all
thirteen datasets in test RMSE, validation RMSE, and test predictions.

## Controlled result

All lanes use the same TabArena task split, eight inner bagging folds, model
random seed, 1,000-round cap, early stopping, and DarkoFit adapter. The two
post-fix lanes differ only in learning-rate policy. Percentages are RMSE gaps
relative to the unrelated ChimeraBoost 0.13 TabArena default; negative is
better.

| Dataset | Before fix | Corrected auto LR | Corrected LR 0.1 |
|---|---:|---:|---:|
| Healthcare expenses | -1.53% | -0.03% | -0.19% |
| Used Fiat 500 | +0.40% | -0.21% | +0.15% |
| Food delivery | +0.40% | -0.19% | -0.15% |
| QSAR fish toxicity | +1.31% | -0.11% | +0.26% |
| Miami housing | +2.35% | +1.45% | +1.09% |
| Diamonds | +3.06% | +1.70% | +3.13% |
| Concrete strength | +3.17% | -2.04% | -2.85% |
| Wine quality | +3.28% | +1.24% | +1.11% |
| Houses | +6.23% | +0.70% | +0.17% |
| Superconductivity | +7.93% | +2.21% | +0.86% |
| QSAR-TID-11 | +8.63% | +0.48% | -0.74% |
| Physiochemical protein | +9.66% | +5.59% | +4.71% |
| Airfoil noise | +24.29% | +5.75% | +3.64% |

| Aggregate | Before fix | Corrected auto LR | Corrected LR 0.1 |
|---|---:|---:|---:|
| Wins / losses vs ChimeraBoost | 1 / 12 | 5 / 8 | 4 / 9 |
| Median RMSE gap | +3.17% | +0.70% | +0.26% |
| Geometric-mean RMSE gap | +5.14% | +1.25% | +0.84% |
| Arithmetic-mean RMSE gap | — | +1.27% | +0.86% |

Fixed 0.1 beats corrected auto on nine datasets and loses on four. Its
geometric-mean RMSE is 0.40% lower, but the losses include Diamonds (+1.42%
relative to corrected auto), QSAR fish toxicity (+0.37%), and Used Fiat 500
(+0.37%). This is useful tuning evidence, not a safe global-default result.

## Local artifacts

- Corrected auto and fixed-0.1 run:
  `.cache/tabarena-regression-fixed-0.9.0-20260712/`
- Empty-child proxy run:
  `.cache/tabarena-regression-empty-child-proxy-0.9.0-20260712/`
- Pre-fix DarkoFit and ChimeraBoost comparison:
  `.cache/tabarena-regression-lite-0.9.0-20260712/`

These are local diagnostic artifacts, not an official TabArena submission.
