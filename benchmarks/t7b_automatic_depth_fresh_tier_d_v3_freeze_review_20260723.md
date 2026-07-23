# P1-v3 automatic-depth freeze review

_Owner decision requested after this package is committed and published._

## Exact package

- Execution contract:
  `t7b-automatic-depth-fresh-tier-d-v3-execution-v1-20260723`
- Contract file SHA-256:
  `12ff0db7553b2748eaa75b2e0f0610fa423abc3112df79fb061bb4b59a4dc34d`
- Verified enumeration SHA-256:
  `c7c76259823d6ee4d3ce6202b127d4bed984493e6153775dfb0f05a105b22851`
- Qualified power-result SHA-256:
  `d6d572e47c672262b007c436cc048b6259a753097e860357523bcec033085ba8`
- Candidate:
  `41e948f0c53b1d124e16071a7fa66eba47d084d3`
- Control:
  `e23d2b164f10374b1c0e02521c33fc96d48980da`
- Panel: 32 verified independent lineages, three coordinates each, 192
  arm rows; 17 depth 4 and 15 depth 8; no substitution.
- Power: `0.998000`, one-sided Wilson lower `0.996657`, required `0.800000`.

The contract, protocol, runner, analyzer, helper hashes, exact identities,
splits, model settings, quality gates, cost gates, integrity checks, and
terminal decisions are all bound. Focused validation passes 32 tests.

## What authorization would do

Authorization permits one launch on the otherwise-idle 14-core machine. The
launch manifest spends the fresh inspection. Every arm/coordinate runs in a
fresh process. The run publishes no partial rows and permits no rerun,
substitution, gate change, candidate repair, or partial read.

GO requires every quality, fit, prediction, RSS, and integrity gate to pass.
It makes automatic depth the selected v0.12 default outcome, while actual
release publication remains separately owner-gated. NO-GO closes the
candidate for defaults and leaves P3's opt-in unchanged.

Expected cost is 192 fits at up to 600 rounds plus prediction, serialization,
and fresh-worker overhead. Wall time is expected to be many hours and may
approach a day. Any error after launch is terminal.

## What remains unauthorized

Without a later owner record, the harness refuses execution. The current
freeze does not authorize:

- fresh model fitting or the one-shot launch;
- candidate modification, panel change, gate change, or rerun;
- default code changes or v0.12 publication;
- TabArena, CTR23 execution, or any lockbox.

## Exact owner record on approval

If the owner approves the one-shot, Codex will create, commit, and publish a
create-only JSON record with exactly:

```json
{
  "schema_version": 1,
  "authorization_id": "t7b-automatic-depth-fresh-tier-d-v3-owner-run-authorization-v1",
  "contract_id": "t7b-automatic-depth-fresh-tier-d-v3-execution-v1-20260723",
  "execution_contract_sha256": "12ff0db7553b2748eaa75b2e0f0610fa423abc3112df79fb061bb4b59a4dc34d",
  "enumeration_sha256": "c7c76259823d6ee4d3ce6202b127d4bed984493e6153775dfb0f05a105b22851",
  "power_result_sha256": "d6d572e47c672262b007c436cc048b6259a753097e860357523bcec033085ba8",
  "confirmation_run_authorized": true,
  "candidate_modification_authorized": false,
  "panel_change_authorized": false,
  "gate_change_authorized": false,
  "rerun_authorized": false,
  "partial_read_authorized": false,
  "tabarena_authorized": false,
  "ctr23_authorized": false,
  "lockbox_authorized": false,
  "release_publication_authorized": false
}
```

The execution preflight may be generated and published before approval
because it merely projects the already-verified registry and reads no data or
outcomes. No launch occurs until the owner record exists in the clean,
published harness checkout.
