# Benchmark evidence

DarkoFit treats benchmark artifacts as provenance records, not a pool of
numbers to retune after the fact.

The canonical chronological index is
[`TESTING_LOG.md`](TESTING_LOG.md). It records the question, source/version
boundary, evidence class, result, limitations, and terminal decision for the
major correctness, benchmark, confirmation, and release-verification work.
Individual frozen artifacts remain authoritative.

The current shipping rules are [`../SHIP_RULES.md`](../SHIP_RULES.md), and
the active sequence is [`../R2_PLAN.md`](../R2_PLAN.md).
[`SHIPPING_POLICY.md`](SHIPPING_POLICY.md) remains historical context for
records produced under the retired preregistration/Tier-D regime.

## Evidence classes

Historical entries retain their original Tier-E/Tier-D labels so their
provenance is understandable. They are not current approval gates.

- **Development evidence** is where mechanisms are built and compared.
- **Release validation** is a deliberately rare holdout check, never a tuning
  surface.
- **Release scoreboard** is the compute ladder against the current
  ChimeraBoost release. It answers "are we winning" but is not tuned against.
- **Product verification** covers correctness, compatibility, serialization,
  packaging, and behavior-exact engineering.

Defaults follow the three-part `SHIP_RULES` check: clearly better on
development, not worse on holdout, and revertible. Opt-ins ship on correctness
plus honest characterization.

## Artifact lifecycle

1. Pin source versions and seeds so reruns are meaningful.
2. Label development, release-validation, and release-scoreboard data
   honestly.
3. Keep raw artifacts and hashes for material comparisons.
4. Treat benchmark harnesses as normal software: fix bugs, rerun when needed,
   and note material reruns in `TESTING_LOG.md`.
5. Do not tune against the holdout or the rival release ladder.

Prior frozen records are not rewritten; current conclusions are added as new
dated evidence.

## Generated status

`python benchmarks/make_pareto.py --write` regenerates:

- [`benchmark_status.json`](benchmark_status.json);
- [`benchmark_status.md`](benchmark_status.md); and
- [`../docs/measurements.md`](../docs/measurements.md).

The generator verifies the SHA-256 of every frozen source before rendering.
CI runs the same command with `--check`.
