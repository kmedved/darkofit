# Fused-lane dispatch v2 realized-bin-width erratum — 2026-07-21

This create-only erratum was recorded during selector invariant development,
before calibration execution, validation execution, threshold selection, or
any campaign outcome access.

It binds:

- v1 design contract SHA-256
  `68d0dd6ef42f29d164943ef16e766821c5bd53319840b22a59b1bd449191cf1a`;
- v2 design contract SHA-256
  `ed032758dfa5829766ae324bdde54b9a1724ed0063d3997f55f3d72f7907240e`;
  and
- v2 contract commit
  `bee460b992ffebac82b042428f4d937d36b5bf58`.

The v1 envelope says `64--254 maximum realized bins`, while its calibration
and validation tables use `Bins` as the public configured `max_bins` value.
DarkoFit reserves one additional histogram slot for missing/non-finite values:
continuous data configured with `max_bins=64`, `128`, `192`, or `254` therefore
has maximum realized histogram widths `65`, `129`, `193`, or `255`.

The binding correction is:

- the table/configuration range remains `max_bins=64--254` unchanged;
- the selector metadata continues to record the actual
  `max(n_bins_per_feature)` histogram width; and
- the automatic realized-width envelope is `65--255`, not `64--254`.

This is a representation correction, not an outcome-responsive gate change.
It changes no dataset, seed, size, feature count, thread count, depth, Hessian
case, timing repetition, threshold-selection rule, acceptance limit, stop rule,
or downstream authority. Without it, the frozen default-`254` validation cells
would be structurally unable to engage the candidate they are meant to test.
