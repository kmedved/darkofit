# T7b automatic-depth fresh Tier-D execution

_Prospective execution protocol. Dataset identities were selected from OpenML
task/dataset metadata only; no feature matrix or target value was downloaded
before this protocol, the registry, and the complete harness were committed
and published._

Contract identity:
`t7b-automatic-depth-fresh-tier-d-execution-v1-20260723`.

## Authority and terminal decisions

The owner authorizes exactly one fresh confirmation of unchanged candidate
`41e948f0c53b1d124e16071a7fa66eba47d084d3` against control
`e23d2b164f10374b1c0e02521c33fc96d48980da`. The frozen power design is
`t7b-automatic-depth-shared-tier-d-power-v1-20260723`; its quality gates,
32-lineage 8-by-4 composition, power assumptions, bootstrap, and analysis unit
are incorporated by file hash and may not change.

The owner further decides prospectively:

- **GO:** promote this automatic depth policy to the public default in the
  next release, v0.12. The fresh terminal record is the sole shipping quality
  evidence; M6 and spent-sports numbers remain labeled historical development.
- **NO-GO:** close this candidate for defaults, leave the existing P3 explicit
  opt-in unchanged, and record the failed powered transfer in the registry.

There is no relaxation, second attempt, partial read, alternate subset, or
candidate repair after launch. TabArena, CTR23, and every lockbox remain out
of scope.

## Frozen registry and contamination sequence

The target-blind registry contains 32 primaries plus two ordered reserves per
stratum. Every identity is checked at the exact pre-freeze DarkoFit and
ChimeraBoost revisions for task ID, dataset ID, normalized name, declared
aliases, and repository references. An identity with a known hit is
ineligible. This covers all prior repository campaigns, including this
cycle's spent-sports work.

After this checkpoint is published, target preflight loads primary slots in
frozen order. It may take the first unused identity from the frozen
same-stratum reserve queue only for a value-free reason:
metadata drift, non-finite/non-numeric target, unsupported feature schema,
split/group failure, wrong automatic-depth branch, or an exact/near-lineage
fingerprint alarm against the already-published exposure catalogs. It may not
compute or inspect target moments, correlations, baseline scores, or candidate
quality. The first eligible identity wins its slot. Failure to fill all 32
slots closes before launch; the power result does not transfer.

## Exact splits and coordinates

Each independent lineage supplies folds 0, 1, and 2. Ordinary coordinates are
folds 0 and 2. Fold 1 uses target-independent weights 1.0/1.25 from the low bit
of a frozen SHA-256 assignment; its asymptotic effective-sample fraction is
81/82. Ordinary row splits use a frozen five-fold row-identity hash. Four
declared low-density grouped lineages use a frozen three-fold group-value hash
for both outer test allocation and DarkoFit's inner validation groups. No
group can cross train/test or inner train/validation.

High-density datasets are target-blind capped at 3,250 rows per input feature
by a sorted sample without replacement from NumPy PCG64 seeded by the SHA-256
of the lineage ID and fixed cap seed. With a four-of-five train split and
weighted coordinate, the planned effective density is 2,568.75, above the
candidate's frozen 2,500 boundary while bounding cost. Preflight verifies
every realized coordinate: low roles must be below 100 and high roles at
least 2,500 effective training rows per input feature.

## Fixed model and paired workers

Every arm/coordinate runs in a fresh `darko311` worker with a clean pinned
source checkout and the same 14-thread environment. Arm order alternates by
coordinate. A same-source two-round synthetic warmup occurs outside timing.
The measured estimator is scalar-RMSE CatBoost with `depth=None`, 600 maximum
iterations, early stopping patience 30, 0.15 validation, best-model selection,
no refit, random state 20260723, and 14 requested threads. All other model
behavior is the library's pinned behavior.

The worker records outer weighted/unweighted RMSE, fit time, three prediction
times on the same frozen 50,000-row tiled test workload, process-tree peak RSS,
archive bytes, fitted depth/policy metadata, ambient thread restoration, and
exact safe-NPZ prediction parity. Classification/no-op routes stay covered by
the candidate's invariant evidence and are not fresh quality units.

## Frozen gates

The unchanged power-design quality gates all bind:

- equal-lineage quality geomean at most 0.995;
- 95th-percentile lineage-cluster bootstrap upper ratio at most 1.002;
- leave-one-most-favorable-lineage-out ratio at most 0.998;
- worst lineage ratio at most 1.02; and
- each depth branch geomean at most 1.0.

Costs prevent a quality gain from silently moving the product backward on its
quality-versus-compute frontier:

- equal-lineage fit-time geomean must not exceed 1.0;
- equal-lineage median prediction-time geomean on the standardized workload
  must not exceed 1.0, and every arm must supply three finite timings;
- candidate process-tree peak RSS must stay below half physical RAM; it also
  fails when both equal-lineage RSS ratio exceeds 1.10 and the
  equal-lineage candidate-minus-control delta exceeds 256 MiB. The latter is
  the first binary allocation band inside the owner's stated
  hundreds-of-megabytes harm region; the ratio protects against a material
  proportional regression while the delta prevents tiny denominators from
  binding.

Archive size is telemetry, never a gate. All correctness, provenance,
split/group, branch, safe-NPZ, thread-mask, completeness, and create-only
integrity checks must pass.

## One-shot and publication

Preflight, launch manifest, raw result, analyzed result, and terminal
attestation are create-only. The launch manifest is written only after clean
source, environment, preflight, output-collision, and exclusive-machine
checks. Its creation spends the sole inspection. A later error is terminal;
partial rows remain unread and unpublished. The complete raw artifact is
atomically published only after all 192 arm rows succeed. The analyzer then
runs once and emits GO or NO-GO without reinterpretation. A 12-field
`TESTING_LOG.md` entry and create-only dated result note close the campaign.
