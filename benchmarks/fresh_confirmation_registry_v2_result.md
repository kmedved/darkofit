# Fresh confirmation registry v2 result

_Amended 2026-07-17 before any confirmation score, under
[`fresh_confirmation_registry_v2_protocol.md`](fresh_confirmation_registry_v2_protocol.md)._

## Decision

Use registry v2 for confirmation. V1 remains immutable but its
`smooth_numeric` label is superseded by `smooth_process`.

The 14-task primary panel contains five numeric-only complete tasks, seven
categorical complete tasks, and two categorical tasks with missing predictors.
The task IDs, 20 lineages, 60 coordinates, contamination decisions, and
99.9965% conditional power calculation are unchanged.

This correction narrows the future claim: the panel can confirm the selector
on smooth/process-oriented regression sources, not on a numeric-only panel.
No model or target statistic was used to make the correction.

## Evidence

- V2 artifact:
  [`fresh_confirmation_registry_v2.json`](fresh_confirmation_registry_v2.json),
  file SHA-256
  `0d878d690e32f6781a170fa3e5c232eef13d20d51d25b352c96a20ddc87e3970`;
  canonical v2 SHA-256
  `29e14fd855e0190e175e0aa27915d00f41c39978759c6f415f6e43fe245fba5d`.
- V1 parent file SHA-256:
  `37799ed0b788af3c1d69c8f0f7cf37a656fde998ce2d54b5c4a2196c369df4c3`.
- Protocol SHA-256:
  `6bffa0c71c4048e1c6fb95efd7718203246a185afdb672dbb8638742785907f6`.
- Amendment builder SHA-256:
  `8d62e0101a2884d66ee13df4901d7d416d30686e5afb7476f3a8d8ba32d17487`.
