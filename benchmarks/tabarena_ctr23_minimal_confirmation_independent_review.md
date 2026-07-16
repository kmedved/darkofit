# Independent review: minimal CTR23 regression confirmation

## Verdict

The frozen 90-job campaign completed and its quality calculations are internally
sound, but it did **not** establish the preregistered confirmation. The frozen
analyzer emitted `confirmation_not_established_clean_stop`; after independent
audit, the reviewed publication status is
`confirmation_not_established_protocol_deviation_stop`.

The result must not be used to add folds, tune `A10`, open the lockbox, or
promote a preset/default. Timing and memory-performance evidence is inadmissible
under the registered `quality_only_swap_in` policy.

## Registered result

Negative percentages favor the numerator.

| Estimand | Point estimate | Registered interval statistic | Gate |
| --- | ---: | ---: | --- |
| A10 / ChimeraBoost | 0.942029 (-5.797%) | one-sided 95% upper 1.012091 | **FAIL**: required `< 1.000` |
| A10 / product default | 0.866066 (-13.393%) | simultaneous max-regret 95% upper 1.046121 | **FAIL**: required `<= 1.020` |
| A10 / CatBoost, r0f0 only | 1.055904 (+5.590%) | descriptive 95% interval [0.951335, 1.186517] | descriptive only |

The report-only A10/ChimeraBoost point target of at most 0.995 was met, but the
registered uncertainty bound was not. A10 beat ChimeraBoost on 4 of 9 tasks and
11 of 27 outer splits; the favorable aggregate is heterogeneous and does not
support a superiority claim. `student_performance_por` was the sole
A10/default task point flag above 1.01, at 1.018036.

## Independent statistical reproduction

A standalone in-memory calculation used only the safe JSON/CSV artifacts and
imported no project modules. It reconstructed the 90-job paired grid and made
139 numerical comparisons against the published summary and tables. There were
zero discrepancies; the largest floating-point difference was `2.22e-16`.

It independently reproduced the registered 10,000-draw PCG64 bootstraps, seeds
20260719/20260720/20260721, `method="higher"` quantiles, both failed gates, all
descriptive CatBoost contrasts, and the terminal decision.

Two analyzer invocations over the unchanged source/result state also reproduced
all decision artifacts byte for byte:

| Artifact | SHA-256 |
| --- | --- |
| `paired_children.csv` | `a985ef28fed1f6744826de6682c994f5f4d0ae518badfbcaffb3abd3d076582a` |
| `paired_splits.csv` | `620260c8eb01c40b75c909b20fd3710c97ba41c5d186b8ad06f736dd1248dc30` |
| `per_dataset.csv` | `491e852e7f96eba36627cc0004add42e8c6c7a9b3221f46d269101c39315313e` |
| `analyzer_result.md` | `ce142fb069bd93256f54c19a1abc1f791cdedf0c6dd9db7c9d933c6501699981` |
| `analyzer_summary.json` | `23a0d8b1657fcf13465afe46cfc82d393d86a394fb7dd573ea78a042103d1580` |

## Integrity audit

The audit verified source commit
`5b714927c87b3d0da92558222506477cf9de6772`, the frozen DarkoFit subtree,
protocol/schedule/coordinate hashes, all 90 opaque raw-result hashes and sizes,
and the exact 27 A10, 27 ChimeraBoost, 27 product-default, and 9 CatBoost jobs.
It verified 720 selected children, 648 A10 candidate fits, fitted parameter and
lane contracts, and candidate selection.

Parallel execution was genuine: 45 overlapping waves used two stable worker
PIDs, with maximum start skew 4.21 ms and minimum overlap 4.80 s. The synthetic
preflight executed zero CTR23 fits. The campaign recorded zero failures, worker
restarts, deadline or callback hits, recovery mixing, imputation, and swap-out;
peak combined RSS was 2.40% of physical memory.

### Operational protocol deviation

The protocol says swap-in is allowed but must be measured and retained, and the
frozen protocol declares `swap_in_allowed_and_recorded: true`. The shared
dispatch helper measured swap-in transiently, but the CTR23 runner discarded
that field and persisted only swap-out telemetry; its full-lifecycle samples
also retained only `sout`.

Consequently, zero swap-out and the quality result are well supported, but the
attempt cannot be described as literally fully compliant with the registered
swap-in audit-evidence requirement. This omission does not change RMSE or the
bootstrap calculations, and timing was inadmissible from the outset. Because
metrics have been opened, the registered terminal boundary forbids a rerun.
The deviation is disclosed here instead. The reviewed machine-readable summary
therefore marks `complete_grid_and_safety=false` and
`full_protocol_compliance=false`, while separately retaining that the complete
grid and registered resource limits were observed.

Separately, 210 ChimeraBoost/CatBoost child stop reasons remain semantically
unresolved between early stopping and iteration/no-split termination. Direct
callback instrumentation proves that no time budget fired, so this does not
weaken deadline integrity.

### Literal safe-analysis-boundary deviation

The protocol also says the analyzer may read only the manifest, attestations,
and safe payload and may not open a raw result pickle. For hash and integrity
authentication, the frozen analyzer additionally read the schedule, preflight,
concurrency, and warmup JSON artifacts. It did not import `pickle`, decode, or
deserialize any raw result, but it also read all 90 result files as opaque bytes
to recompute their SHA-256 hashes. That behavior matches the analyzer's own
docstring and the protocol's separate instruction to revalidate hashes, but it
crosses the literal input allowlist and no-open/no-read boundary.

This is disclosed as a second protocol deviation. It does not expose result
semantics to the analyzer or change any quality calculation, but full protocol
compliance remains false.

## Reproduction boundary

The executable verifier authenticates the campaign only from source revision
`5b714927c87b3d0da92558222506477cf9de6772`, whose Git head/tree are bound in
the run manifest. This descendant evidence commit necessarily changes the Git
head/tree, so the checked-in bundle is static hash-bound evidence; it is not a
claim that the analyzer can be rerun unchanged from the descendant revision.

The analyzer never deserialized raw result pickles; as disclosed above, it did
read their opaque bytes for hash verification. Raw model caches, predictions,
targets, and datasets are not part of the committed evidence. “Unseen” means
absent from the audited DarkoFit development histories, not globally
unpublished data.

## Terminal state

Stop here. This protocol authorizes no further folds, task-specific reruns,
post-outcome tuning, lockbox access, or default/preset change. Any future
experiment requires a separately declared evidence boundary and authorization.
