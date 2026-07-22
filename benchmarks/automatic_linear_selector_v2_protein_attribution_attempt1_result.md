# Automatic linear-selector v2 Protein attribution attempt 1

Launched once on 2026-07-22 from clean, published harness
`19bdef2f27496ff4312c1a156d2f6198d358184e`, clean published candidate
`a53d4bf543534678189d87d88dcad87dd2a8bd8f`, and the exact clean TabArena
source `4cd1d2526874962daae048a6f2dcf34aa272f3fa`.

## Terminal result

The create-only launch manifest was written before worker zero, spending
attempt 1 as required. Worker zero then failed while importing TabArena's task
loader because `autogluon.common` was unavailable in the frozen worker
environment:

```text
ModuleNotFoundError: No module named 'autogluon'
```

No model fit began, no worker completed, no raw artifact was created, and no
quality, selector-margin, fit-time, prediction-time, or RSS outcome exists.
The runner wrote the required create-only terminal failure result with
`completed_worker_count=0` and disposition `terminal_execution_failure`.

Artifacts:

- [`automatic_linear_selector_v2_protein_attribution_attempt1_20260722_manifest.json`](automatic_linear_selector_v2_protein_attribution_attempt1_20260722_manifest.json),
  SHA-256
  `4b4471cdba3beab6cc9dc2cce8d1c8835bfa01cebc986321b9541f89e191def4`;
  and
- [`automatic_linear_selector_v2_protein_attribution_attempt1_20260722_result.json`](automatic_linear_selector_v2_protein_attribution_attempt1_20260722_result.json),
  SHA-256
  `e4bb44356c90d18e88c252bc2a9c8d197303e4a4cb750daacee6eda3c104ab0f`.

## Decision

The development contract says a Protein failure is terminal for this exact
candidate identity and prohibits rerunning a failed or inspected attempt.
Accordingly, `automatic_linear_selector_v2` is closed without a scientific
Protein verdict. M6's exact-preservation result remains valid, but it does not
override this terminal execution disposition. No merge, shipping, default,
fresh-confirmation, TabArena, or lockbox authority is created.

The harness defect must be fixed forward by probing the exact TabArena data
loader before writing any future launch manifest. That repair does not reopen
or rerun this candidate.
