# Benchmark evidence

DarkoFit treats benchmark artifacts as provenance records, not a pool of
numbers to retune after the fact.

The binding decision constitution is
[`SHIPPING_POLICY.md`](SHIPPING_POLICY.md). The active agenda is
[`../PRODUCT_OFFENSE_PLAN.md`](../PRODUCT_OFFENSE_PLAN.md).

## Evidence classes

- **Tier-E:** opt-in APIs, presets, recipes, and behavior-exact engine work.
  Correctness and exactness are binding. Measured quality, speed, and memory
  are reported descriptively with workload and dispersion; they are not
  universal certifications.
- **Tier-D:** defaults and automatic modeling policies. These require a
  source-frozen protocol, design-time power analysis, fresh confirmation data,
  bootstrap uncertainty, leave-one-dataset-out concentration, a bounded harm
  route, declared costs, and a no-rerun decision.

Win counts are never gates. A selector's exact ties and safe declines are
correct behavior, while magnitude, uncertainty, concentration, and worst-case
harm determine whether a default ships.

## Artifact lifecycle

1. Register dataset identity and contamination boundaries.
2. Freeze source, environment, arms, coordinates, metrics, decision rule,
   power calculation, and cost budgets before outcomes.
3. Write raw results create-only and attest source hashes.
4. Analyze the frozen artifact once. A failure is retained as a failure.
5. Keep confirmation panels and lockboxes sealed unless the protocol
   explicitly authorizes access.
6. Use spent evidence for development or prioritization only, never a fresh
   confirmation claim.

Closed candidates are not retroactively promoted under a friendlier Tier-D
rule. Tier-E product surfaces may ship immediately when an earlier rejection
concerned only an inapplicable default-grade or binary engineering bar, but
their documentation must retain the original outcome and scope.

## Generated status

`python benchmarks/make_pareto.py --write` regenerates:

- [`benchmark_status.json`](benchmark_status.json);
- [`benchmark_status.md`](benchmark_status.md); and
- [`../docs/measurements.md`](../docs/measurements.md).

The generator verifies the SHA-256 of every frozen source before rendering.
CI runs the same command with `--check`.
