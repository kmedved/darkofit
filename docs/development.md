# Development

## Setup

```bash
git clone https://github.com/kmedved/darkofit.git
cd darkofit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,tuning,docs]"
```

## Test partitions

The library suite excludes frozen campaign verifiers:

```bash
python -m pytest -q -m "not campaign"
```

Campaign verifiers run separately:

```bash
python -m pytest -q -m campaign
```

Run the full union locally with:

```bash
python -m pytest -q
```

CI runs the library partition on the supported Python matrix and the campaign
partition in a dedicated job.

Regenerate and verify the release benchmark frontier with:

```bash
python benchmarks/make_pareto.py --write
python benchmarks/make_pareto.py --check
```

## Documentation

```bash
mkdocs serve
mkdocs build --strict
```

All internal links must resolve. Archived plans are retained for provenance
but should not be used as current API documentation.

## Release tags

Before creating a release tag, verify both local and remote lineage:

```bash
git fetch --tags origin
git rev-parse --verify --quiet refs/tags/vX.Y.Z
git ls-remote --exit-code --tags origin refs/tags/vX.Y.Z
```

An existing tag must point to the intended DarkoFit release commit and that
commit must be an ancestor of the release branch. Never reuse a
ChimeraBoost-era tag name or force-update a published release tag; preserve a
conflicting historical ref under an explicit archive name first.

## Benchmark discipline

New promotion protocols use alternating fresh-worker blocks and paired timing
ratios. Do not rewrite frozen artifacts, reopen closed candidates without a
materially different mechanism, or inspect sealed confirmation/lockbox
outcomes before their prerequisites pass.
