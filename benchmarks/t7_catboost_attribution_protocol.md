# T7 CatBoost attribution protocol

## Scope

T7 is development-only mechanism attribution. It uses the eight already-spent
development tasks and repeat-0 folds 0–2 in
`native_ordinal_c2_registry.json`. The confirmation tasks in that registry,
the CTR23 lockbox, and every T5 lineage are excluded. No T7 result can change
a default or serve as confirmation evidence.

The immutable DarkoFit control rows in
`native_ordinal_c2_development_raw.json` are a descriptive competitive anchor.
They used the same outer folds and deterministic inner-fit rows. T7 does not
refit or tune DarkoFit.

## CatBoost arms

CatBoost 1.2.10 runs with RMSE, seed 4, six threads, no file writes, and all
other product defaults unless listed:

| Arm | Override |
|---|---|
| `default` | none |
| `ordered` | `boosting_type="Ordered"` |
| `plain` | `boosting_type="Plain"` |
| `border_128` | `border_count=128` |
| `leaf10_no_backtracking` | 10 leaf-estimation iterations, no backtracking |
| `leaf10_any_improvement` | 10 leaf-estimation iterations, AnyImprovement backtracking |
| `ctr_complexity_2` | `max_ctr_complexity=2` |
| `depth_4` | `depth=4` |
| `depth_8` | `depth=8` |

Every coordinate uses the exact C2 inner-fit, validation, and outer-test rows.
Arm order is a deterministic rotation by coordinate, so position is balanced.
Each coordinate is isolated in a subprocess and persisted before aggregation.
Warmup precedes timing.

The fixed `(n,p)` depth policy is assembled without new fits from the depth
arms. Let `density = inner_fit_rows / n_features`: choose depth 4 below 100,
depth 8 at or above 2,500, and the default depth 6 otherwise.

## Attribution contrasts

Report equal-dataset geometric-mean validation and test RMSE ratios, per-task
ratios, wins/losses/ties, worst task, and least-favorable leave-one-task-out:

- Ordered / Plain and Plain / default;
- 128 borders / default;
- 10-step leaf estimation without backtracking / default;
- AnyImprovement / no backtracking at 10 leaf steps;
- CTR complexity 2 / default;
- depths 4 and 8 / default; and
- the fixed `(n,p)` depth policy / default.

Also report every CatBoost arm against the immutable DarkoFit control. These
are attribution measurements, not matched-current-release claims.

## Candidate nomination

The eligible CatBoost-inspired candidates are Ordered, 128 borders, 10-step
leaf estimation with either backtracking setting, CTR complexity 2, and the
fixed depth policy. A candidate survives only if:

- equal-dataset test ratio versus CatBoost default is at most 0.995;
- equal-dataset validation ratio is at most 1.005;
- worst task test ratio is at most 1.02; and
- least-favorable leave-one-task-out test ratio is at most 1.00.

Rank survivors by equal-dataset test ratio and freeze at most three. A
survivor is only a research candidate for a future outcome-unseen protocol.
No post-hoc combinations, dataset-specific settings, retries, or default
changes are authorized.
