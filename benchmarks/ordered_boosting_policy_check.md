# Ordered-boosting auto-policy guardrail

Run on 2026-07-12 with Python 3.13.13, DarkoFit 0.9.0 working-tree code,
CatBoost 1.2.10, LightGBM 4.6.0, four CPU threads, and three shared random
train/validation/test splits per dataset:

```bash
python benchmarks/ordered_boosting_policy_check.py --seeds 3 --threads 4
```

Each DarkoFit variant used 500 maximum rounds and 50-round early stopping.
Values below are mean held-out RMSE; lower is better. `new_auto` was asserted
bitwise-equal to `plain_off` on every split. The optional reference libraries
were run with the same maximum rounds, validation rows, weights, and early
stopping horizon.

| Dataset | Rows / type | Old ordered on | New auto / plain off | Change vs old | CatBoost | LightGBM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Diabetes | 442 numeric | 60.3711 | 59.0936 | **+2.12%** | 57.8006 | 59.9198 |
| Diabetes weighted | 442 numeric | 65.5199 | 65.6309 | -0.17% | 63.2916 | 64.1711 |
| California Housing | 20,640 numeric | 0.477740 | 0.474521 | **+0.67%** | 0.459784 | 0.456687 |
| California Housing weighted | 20,640 numeric | 0.583031 | 0.556579 | **+4.54%** | 0.540417 | 0.546234 |
| Abalone | 4,177 mixed categorical | 2.25664 | 2.21934 | **+1.65%** | 2.20973 | — |
| House Prices | 1,460 mixed categorical | 49,419.9 | 29,694.7 | **+39.91%** | 30,461.5 | — |

The small weighted Diabetes result is effectively neutral and is the only
mean regression. The policy improves the other five case means, but the
House Prices mean deserves a per-seed disclosure because it is driven by a
single catastrophic split rather than a uniform shift: ordered-on scored
26,907 / 23,833 / 97,520 against plain-off 27,326 / 24,625 / 37,133 on seeds
0/1/2. Ordered was mildly better on two splits and 2.6x worse on the third,
where its validation RMSE still looked healthy (24,968 vs 23,815 on the
219-row holdout) — a tail-risk failure mode that a small validation set
cannot catch. Abalone, by contrast, improves on all three seeds
(2.2014→2.1884, 2.1788→2.1498, 2.3898→2.3199). Read the categorical
evidence as "roughly neutral typically, catastrophically unstable
occasionally," which favors plain boosting as the robust default. This
changes the 0.9.0 rule from “ordered on when target
encoding is present” to “ordered leaf update off for scalar regression.”
Categorical fits still use ordered target-statistic preprocessing; only the
separate leave-one-out leaf update is disabled. Explicit
`ordered_boosting=True` remains available for RMSE experiments.

The official three-dataset TabArena smoke remains a separate release gate.
It was not rerun here because AutoGluon and TabArena are not installed in the
available Python environments.
