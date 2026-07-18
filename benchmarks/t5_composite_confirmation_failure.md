# T5 composite confirmation: fail-closed result

**Decision: `close_t5_composite_candidate`.**

The frozen T5 run stopped in its first wave. Twenty-three current-default
workers completed and were persisted; two tasks failed target validation
before fitting. The composite, ChimeraBoost, and CatBoost waves never started.

| Task | Dataset | Target | Non-finite rows |
|---:|---|---|---:|
| 362367 | Riga-real-estate-dataset | price | 470 / 4,689 |
| 362395 | Nintendo3DS-Games | metacritic | 1,542 / 1,680 |

The protocol forbids dropping or imputing a task after outcomes exist. No task
was changed, no run was resumed, and no default promotion is authorized. All
25 lineages are now spent for confirmation because control outcomes were
scored before the failure.

This is a panel-construction failure, not evidence for or against the T5 model
policy. A future nomination needs a new outcome-unseen panel whose target
validity is checked before authorization.
