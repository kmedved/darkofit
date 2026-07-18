# T9 SynthGen corrected-ledger re-backtest protocol

**Frozen:** 2026-07-17 before executing the T9 analyzer. This is the
predeclared re-backtest in `PRODUCT_OFFENSE_PLAN.md` T9.

## Boundary

T9 does not generate data, fit a model, rerun a SynthGen coordinate, or alter
an old artifact. It re-scores the immutable 1,464-coordinate SynthGen df1
artifact against the most authoritative real-data outcome now available for
each of its nine questions.

The original backtest scored 6/9 and closed adoption. Later fresh confirmation
superseded two development labels:

1. `random_strength=0.5` advanced on one basketball dataset but failed the
   later frozen nine-cell sports panel.
2. Fixed local linear leaves won on three spent smooth tasks but regressed
   overall on the later frozen 14-lineage smooth/process panel.

Those are decisions 3 and 5. They are the only labels this protocol may
supersede. Decision 6 remains tied to its direct global-versus-local
development comparison because the fresh panel did not run global residual
boosting.

This is retrospective and not statistically independent: both the synthetic
scorecard and the corrected real outcomes were known when T9 was placed on the
agenda. Passing can only admit SynthGen as a cheap probe-tier direction finder.
It cannot gate, confirm, promote, or justify any product behavior.

## Frozen inputs

The analyzer hardcodes and verifies SHA-256 for:

- the raw SynthGen ledger;
- the original runner, analyzer, and protocol;
- all nine binding outcome-source reports.

The immutable raw SHA-256 is
`fd8f93ec4c0e1cbd6889200d0d79f235e7e880732f0597e09a2bd025f09af7eb`.
The T9 analyzer is
`benchmarks/analyze_t9_synthgen_corrected_ledger.py`, SHA-256
`ae60cbec0544840e8ab0003480f6b75c828c50c2ff8ed16d586568fc7805d9ae`.

The original analyzer must independently reproduce the original 6/9 pattern
`yes, yes, no, yes, no, no, yes, yes, yes` before any correction is applied.

## Frozen corrected rules

- Decisions 1, 2, 4, 6, 7, 8, and 9 retain their original rule and outcome.
- Decision 3 agrees when SynthGen does **not** meet the superseded positive
  random-strength rule (`ratio <= 0.999` and wins > losses).
- Decision 5 agrees when SynthGen does **not** meet the superseded positive
  fixed-linear rule (`ratio <= 0.99` and wins on at least two thirds of
  datasets).
- All original raw-boundary, slice-size, canary-floor, canary-no-variance, and
  protected-source gates must still pass.
- The adoption threshold remains the original **at least 7/9**.

No other arm, slice, seed, threshold, synthetic measurement, canary, or
integrity rule may change. The analyzer must explicitly say why each
superseded label changed and why decision 6 did not.

## Disposition

If every integrity gate passes and corrected agreement is at least 7/9,
SynthGen df1 is adopted only as a probe-tier direction finder:

- it may rank inexpensive development ideas;
- it may not kill an idea by itself;
- it may not substitute for real data;
- it may not provide confirmation, power, promotion, release, or marketing
  evidence;
- every product decision still requires its own frozen real-data protocol.

Otherwise SynthGen remains not adopted.

## Exact command

```bash
PYTHONHASHSEED=0 PYTHONPATH=. \
python benchmarks/analyze_t9_synthgen_corrected_ledger.py
```
