# Contiguous-layout prediction-throughput confirmation

## Question

Does preserving already-contiguous Fortran-order preprocessing blocks remove
the measured binning copy without changing predictions, and does the resulting
matched prediction lane formally meet the `<=1.30x` ChimeraBoost 0.15 target?

This is the P2 successor to
[`predict_throughput_protocol.md`](predict_throughput_protocol.md). It neither
changes model behavior nor consumes any quality, CTR23, or lockbox evidence.

## Frozen mechanism

- Pre-mechanism DarkoFit source: clean `main` at `61c1d44`.
- A float64 block entering `Binner.transform_blocks` is reused when it is
  C-contiguous or F-contiguous. A genuinely strided block is copied to C order.
- The binning kernel, learned borders, output allocation/layout/dtype, packed
  forest, validation semantics, and public API are unchanged.
- Focused tests require byte-identical bins for C-, F-, and strided layouts,
  and prove the two contiguous inputs are not copied.

## Matched campaign

The source, comparator, fitted models, numeric/mixed matrices, batch sizes,
18-thread fresh workers, reciprocal three-block ordering, warm repeats,
phases, exactness checks, and paired-ratio stability definition are exactly
those in the predecessor protocol.

The historical worker runner
`run_predict_throughput.py` is reused unchanged and hash-bound separately.
This successor parent only creates a new artifact and never rewrites the
predecessor artifact or runner.

## Gates

The mechanism passes P2's current target only if:

1. all within-library public/packed exactness checks pass;
2. each library has one stable behavior fingerprint across all three blocks;
3. every numeric and mixed warm-public case has
   `IQR(paired ratio) / median(paired ratio) <= 0.10`;
4. every warm-public DarkoFit/ChimeraBoost median ratio is `<=1.30`; and
5. the paired peak-RSS median ratio is `<=1.05`.

Warm-public ratios `<=1.00` are reported as the stretch target. Binning and
packed-core ratios remain diagnostic because the product boundary is public
prediction.

Passing closes P2's `<=1.30x` engine target. Failure keeps the target open and
selects the largest remaining component; no default or router change is
authorized either way.
