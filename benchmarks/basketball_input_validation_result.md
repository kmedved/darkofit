# Basketball input-validation and sklearn-compliance result

## Decision

**Ship the input-validation and sklearn-compliance layer.** The frozen
six-block basketball campaign passed every correctness, isolation, stability,
and timing gate. This authorizes the shared public validation boundary,
feature-name enforcement, named categorical resolution, sklearn tags, and the
documented `assume_finite` prediction escape hatch. It does not authorize a
model-default change or a broad quality claim.

The source-bound artifact is
[`basketball_input_validation.json`](basketball_input_validation.json),
produced from clean, pushed `main` at
`f48ec4dd9d85c7c9e69b5cd683998450527bd3f4`. Its SHA-256 is
`8e7548e0f8c1fab115fcde0a902ebc6a05fbd92d81e35d2defacfd79178995b9`.
The frozen protocol SHA-256 is
`e1bb661237e5e9f6b12063c4ed7866e9924d445ebc810105bb3e6b16339586cb`,
and the executed runner SHA-256 is
`28d61b6511946541b0bfb9f66095f206ea18a596888687fe9984ec41d9c30fd7`.
CTR23 and TabArena were not used.

The separately bound scikit-learn 1.7 compliance artifact is
[`basketball_input_validation_sklearn17_compliance.json`](basketball_input_validation_sklearn17_compliance.json)
at SHA-256
`024a3e8bb77d0ec64dead6673df1fdcde8e68566cc3f3a011de8f9f7deb004ee`.
It records the exact 1.7.2 environment, source and test hashes, commands,
outcomes, and preregistered expected failure.

## Correctness and compatibility

All 12 fresh processes used unique, initially empty Numba cache directories.
Every validated and `assume_finite` worker reproduced the immutable prediction
arrays and hashes for:

- creator fold 0;
- the complete held-team view; and
- the corrected 585-row cold-player view.

Every worker also reproduced the 382,557-byte model archive at SHA-256
`50a7e6f0a6f8500a55a6ba088ad25137335ed4354a4b4e908ea17f023c91ec71`,
the feature-importance hash, and one timing-free model-behavior fingerprint.
Each fit retained 1,000 CatBoost-mode trees, resolved learning rate
`0.052312`, and stopped at the iteration limit. No worker emitted a warning or
unexpected output.

Before timing, the full project suite passed with 1,569 tests and 23 skips.
In an isolated Python 3.12.13 environment with scikit-learn 1.7.2, the focused
validation suite then passed 28 tests with one optional-integration skip. The
two `check_estimator` cases passed separately with only the preregistered
sample-weight-equivalence expected failure. Independent review found no
remaining actionable correctness issue.

## Timing result

The campaign retained all observations from six reciprocal,
position-balanced blocks at 18 threads.

| Metric | Validated median | `assume_finite` median | Gate | Result |
| --- | ---: | ---: | ---: | --- |
| First fit | 1.2568 s | 1.2263 s | validated ≤ 1.7674 s | pass |
| First prediction | 2.6245 ms | 2.5600 ms | ratio ≤ 1.10x | pass |
| Validation-only scan | 0.1317 ms | 0.1193 ms | recorded | pass |

The ratio of arm medians for prediction was `1.0252x`. The median reciprocal
paired ratio was `1.0368x`; its IQR/median was `0.0869` against the `0.25`
limit.

| Stability gate | Observed | Limit | Result |
| --- | ---: | ---: | --- |
| Validated fit IQR / median | 0.0698 | 0.25 | pass |
| `assume_finite` fit IQR / median | 0.0221 | 0.25 | pass |
| Validated prediction IQR / median | 0.1468 | 0.50 | pass |
| `assume_finite` prediction IQR / median | 0.0508 | 0.50 | pass |
| Paired prediction-ratio IQR / median | 0.0869 | 0.25 | pass |

## Interpretation

Basketball was the primary and fatal development gate because it is fast and
matches the project's noisy sports-data priority. The ordinary fold,
held-team split, and genuinely cold-player split all passed exactly. The
result closes this compatibility item without spending broader benchmark
evidence. Basketball remains the first screen for subsequent candidates, but
it is not sufficient by itself to promote a universal model policy.
