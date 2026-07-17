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

## Benchmark discipline

New promotion protocols use alternating fresh-worker blocks and paired timing
ratios. Do not rewrite frozen artifacts, reopen closed candidates without a
materially different mechanism, or inspect sealed confirmation/lockbox
outcomes before their prerequisites pass.
