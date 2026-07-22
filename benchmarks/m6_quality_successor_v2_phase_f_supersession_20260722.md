# M6 v2 Phase F forward-authority supersession

_Create-only record, 2026-07-22._

`m6-quality-successor-v2` and every v1/v2 contract, runner, test, and result
artifact remain immutable. V2 correctly implemented and backtested its frozen
rule. This note changes only the rule's authority for future mechanism ranking.

Phase F found that v2's 60% win-count gate and known-negative selector replay
encode a verdict that no longer survives current policy. The 3% smooth-data
selector had aggregate ratio `0.989264`, zero lineage and split losses, and
twelve intentional no-ops. Its old `kill` depended on the abolished win-count
and minimum-effect gates. Continuing to use v2 would silently make those gates
part of every future quality-development decision.

Therefore:

- v2 retains historical reproducibility but has **no forward ranking
  authority** after this record;
- no v2 outcome is relabeled, deleted, or reused as new evidence;
- `m6-quality-successor-v3` is the only prospective M6 quality-ranking path,
  and only if its create-only three-replay backtest passes; and
- v3 remains development triage only. It cannot authorize shipping, a default,
  fresh/lockbox evidence, or the selector campaign itself.

The complete casualty sweep and new backtest selection are recorded in
[`premature_kill_audit_20260722.json`](premature_kill_audit_20260722.json) and
[`premature_kill_audit_20260722.md`](premature_kill_audit_20260722.md).
