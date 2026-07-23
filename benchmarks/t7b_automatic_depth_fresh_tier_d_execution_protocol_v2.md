# T7b automatic-depth fresh Tier-D execution v2

_Prospective successor to
`t7b-automatic-depth-fresh-tier-d-execution-v1-20260723`. No feature matrix,
target value, model fit, partial result, or one-shot launch existed when this
successor was frozen._

Contract identity:
`t7b-automatic-depth-fresh-tier-d-execution-v2-20260723`.

The v1 preflight stopped before its first OpenML lineage load because Python
resolved the generic `benchmarks` package to another local repository. The
create-only failure record is
`t7b_automatic_depth_fresh_tier_d_preflight_failure_v1_20260723.json`.

V2 changes only harness import resolution: the runner and analyzer place this
published DarkoFit checkout at the head of `sys.path` before importing local
`benchmarks` helpers. V2 reuses the immutable v1 contamination registry and
all substantive terms of
`t7b_automatic_depth_fresh_tier_d_execution_protocol.md` unchanged:
candidate and control pins, eligible identities and reserve order, power
assumptions, splits, coordinates, model settings, quality and cost gates,
one-shot/no-rerun discipline, terminal GO/NO-GO rule, and authorization
boundary.

Preflight under the v2 identity is permitted only after this protocol, the v2
contract, and the repaired hash-bound harness are committed and published.
The sole fresh inspection remains unspent until the launch manifest is
created.
