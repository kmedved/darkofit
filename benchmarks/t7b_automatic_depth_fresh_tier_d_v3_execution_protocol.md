# P1-v3 automatic-depth fresh Tier-D execution freeze

_Freeze-review package over the verified as-built registry. This protocol and
its harness are intentionally non-executable until a separate owner
authorization record is committed and published._

Contract identity:
`t7b-automatic-depth-fresh-tier-d-v3-execution-v1-20260723`.

## Bound evidence and panel

The contract binds:

- unchanged candidate `41e948f0c53b1d124e16071a7fa66eba47d084d3`;
- unchanged control `e23d2b164f10374b1c0e02521c33fc96d48980da`;
- fillability enumeration
  `t7b-automatic-depth-fresh-tier-d-v3-enumeration-v2-20260723`;
- the exact 32 eligible identities and their dataset, fingerprint, selected
  view, branch, and three split-coordinate attestations;
- power contract
  `t7b-automatic-depth-fresh-tier-d-v3-power-v1-20260723`;
- qualified primary power `0.998000` and one-sided Wilson lower `0.996657`;
  and
- the unchanged Tier-D quality gates and v1 execution cost/integrity gates.

No execution-time resource discovery, substitution, fallback, target
preflight, or panel recomposition exists. The execution preflight simply
projects the already-verified eligible rows from the hash-bound enumeration
into the exact worker manifest and rechecks every binding.

## Model and workers

Every arm/coordinate runs in a fresh `darko311` worker. Arm order alternates
by lineage index plus coordinate. The measured estimator is scalar-RMSE
CatBoost with `depth=None`, 600 maximum iterations, early stopping patience
30, 0.15 validation, best-model selection, no refit, random state 20260723,
and 14 requested threads. A two-round same-source synthetic warmup occurs
outside timing.

Each worker reloads its OpenML identity and requires exact version, MD5,
selected-view, and split hashes from the enumeration. It records weighted or
ordinary outer RMSE, fit time, three prediction times on the fixed
50,000-row workload, process-tree peak RSS, archive bytes, fitted depth/policy
metadata, ambient Numba thread restoration, and exact safe-NPZ prediction
parity.

## Gates

All five power-design quality gates bind:

- equal-lineage quality geomean at most 0.995;
- 95th-percentile lineage-cluster bootstrap upper ratio at most 1.002;
- leave-one-most-favorable-lineage-out ratio at most 0.998;
- worst lineage ratio at most 1.02; and
- each changed depth branch geomean at most 1.0.

The unchanged Pareto/resource gates also bind:

- equal-lineage fit-time geomean at most 1.0;
- equal-lineage prediction-time geomean at most 1.0;
- candidate peak process-tree RSS below half physical RAM; and
- hybrid RSS failure only when both the candidate/control ratio exceeds 1.10
  and mean absolute delta exceeds 256 MiB.

Archive size is telemetry. All source, environment, row census, pairing,
split, branch, policy, thread-mask, safe-NPZ, and create-only integrity checks
must pass.

## Owner authorization and one shot

The frozen contract has `fresh_access_authorized=false` and
`confirmation_run_authorized=false`. Execution requires a later create-only
owner authorization record that:

- names this exact contract ID and its file SHA-256;
- names the exact enumeration and power-result SHA-256 values;
- states `confirmation_run_authorized=true`;
- preserves candidate, panel, gates, no-rerun, and no-partial-read terms; and
- is committed in the clean, published harness checkout used to launch.

The execution preflight may be created and published before that decision
because it reads no new data or outcomes. The launch manifest is the act that
spends the sole fresh inspection. Any later error is terminal; completed rows
remain unpublished and unread. A complete 192-row raw artifact is published
atomically, analyzed once, and closed by a terminal attestation.

GO promotes the unchanged automatic-depth policy to the v0.12 default, subject
to the separately gated release act. NO-GO closes this candidate for defaults
and leaves P3's explicit opt-in unchanged. There is no rerun, relaxation,
partial read, candidate repair, TabArena, CTR23 execution, or lockbox access.
