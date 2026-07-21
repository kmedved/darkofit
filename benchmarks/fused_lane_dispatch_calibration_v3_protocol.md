# Fused-lane dispatch calibration execution protocol v3

_Prospective, outcome-blind successor to the v2 execution freeze. No formal
calibration or validation worker ran and no timing outcome was opened under
v2._

## Reason for supersession

The published v2 execution contract has SHA-256
`b2075f9c45df3b3fb674c74fe0b47cd9ddd1ec3bae790f5379308e15a327061a`.
An independent pre-authorization review found four evidence-contract defects:

1. the directly callable calibration and validation worker subcommands did not
   enforce the frozen contract and owner authorization themselves;
2. the parent created per-coordinate Numba cache directories instead of using
   the exact per-thread environments serialized in the contract;
3. product dispatch metadata inferred lane engagement after the builder
   returned instead of using the builder's actual fused/unfused counters; and
4. private group-ensemble archives did not retain the normalized input group
   partition needed to recompute plausible group-count provenance.

The same review also found that validation analysis accepted a free threshold
integer instead of reloading and checking the hash-bound threshold artifact.
These are gate and provenance defects, not campaign outcomes. V2 had no owner
authorization, raw, terminal, analysis, threshold, or validation artifact, and
no formal worker started. V2 is therefore preserved and superseded without
opening or responding to timing evidence.

## Gate repairs

V3 requires every worker process, including a directly invoked subcommand, to
perform all of the following before importing or timing DarkoFit:

- load the canonical frozen execution contract;
- load the canonical owner-authorization artifact;
- verify source, phase, execution identity, and the complete frozen generator
  coordinate;
- for validation, verify the arm and block against the frozen block orders and
  reload the exact hash-bound threshold artifact; and
- compare every recorded worker-environment field, including the absolute
  `NUMBA_CACHE_DIR`, with the per-thread environment frozen in the contract.

The v3 freezer carries forward the exact v2 per-thread environment records;
the repair changes enforcement, not the cache layout or runtime declaration.

Worker rows repeat the source, execution identity, contract hash,
authorization hash, and (for validation) threshold-artifact hash. Analysis
must rebind every row to its raw envelope and must derive the validation
threshold only from the contract-bound artifact.

Product dispatch metadata is finalized only from counters incremented inside
the actual builder lane. A functionally eligible fit fails closed if the
selected lane has no engagement or the opposite lane reports engagement.

Private group-ensemble persistence moves to private ensemble archive format 3
and private metadata version 4. It stores canonical contiguous int64 group
codes for all original input rows, binds their SHA-256 and group count, and
recomputes complete-group sampling, draw multiplicities, selected groups, OOB
groups, and disjointness on safe load. Public format-1 ensemble behavior is
unchanged.

## Scientific invariance and authority

Every scientific and operational clause in
[`fused_lane_dispatch_calibration_protocol.md`](fused_lane_dispatch_calibration_protocol.md)
and its
[`v2 successor`](fused_lane_dispatch_calibration_v2_protocol.md) remains
binding. V3 changes no generator, coordinate, seed, lane, timing region,
warmup, repeat order, threshold candidate, tie rule, acceptance limit, or
downstream authority.

V3 uses execution identity `calibration_v3` and unique create-only
authorization/raw/terminal/analysis paths. Freezing it is not authorization to
run it. A qualifying calibration still permits only a separately committed
threshold artifact, new candidate source pin, and separately frozen validation
execution. Failure still closes Wave 4; B remains closed, Q remains sequenced
after any retained dispatch, and the next mechanism slot remains
quality-first.
