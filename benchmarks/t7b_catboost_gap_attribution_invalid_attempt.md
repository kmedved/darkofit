# Invalid T7b CatBoost-gap attribution attempt

On 2026-07-18, the formal 576-fit T7b invocation failed closed during worker
validation before publishing a raw artifact or any coordinate spool record:

```text
RuntimeError: T7b CatBoost resolved shared policy changed
```

CatBoost 1.2.10 accepted the frozen constructor request
`thread_count=6`, and `model.get_params()["thread_count"]` retained that
value. Its post-fit `model.get_all_params()` output omitted `thread_count`,
however, so the runner observed `None` and incorrectly treated the
introspection omission as a resolved-policy mismatch.

The repair does not change a model parameter, arm, coordinate, seed, split,
metric, or decision rule. It records and validates the constructor-observed
thread count from `get_params()` in a field separate from the resolved
`get_all_params()` snapshot. It does not synthesize a resolved value that
CatBoost did not report.

The raw output path did not exist after the abort, and the T7b spool directory
did not exist. Because three workers were launched concurrently and the
runner has no durable per-fit counter, the number of fits that completed
before the abort is unknown and must not be inferred. This invalid attempt
cannot support a result or performance claim. The complete frozen campaign
must restart from the beginning after the repair receives a new source freeze.
