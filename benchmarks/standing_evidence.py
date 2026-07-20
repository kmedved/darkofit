"""Frozen-shape contract for the M5 sentinels and M6 development slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

try:
    from benchmark_adapters import DATASETS, SIZE_SAMPLES
except ImportError:  # pragma: no cover - supports `python -m benchmarks...`
    from benchmarks.benchmark_adapters import DATASETS, SIZE_SAMPLES


CONTRACT_VERSION = "standing-evidence-draft-v1"
M6_CONTRACT_FROZEN = False
M6_BACKTEST_COMPLETE = False


@dataclass(frozen=True)
class SentinelDomain:
    id: str
    task: str
    source: str
    weighted: bool = False


M5_SENTINEL_DOMAINS = (
    SentinelDomain(
        "grouped_entity_regression",
        "regression",
        "generic_group_generator",
    ),
    SentinelDomain(
        "smooth_numeric_regression",
        "regression",
        "synthgen",
    ),
    SentinelDomain(
        "noisy_numeric_regression",
        "regression",
        "synthgen",
    ),
    SentinelDomain(
        "categorical_missing_regression",
        "regression",
        "synthgen",
    ),
    SentinelDomain(
        "high_row_numeric",
        "regression",
        "benchmark_adapters",
    ),
    SentinelDomain(
        "binary_classification",
        "binary",
        "benchmark_adapters",
    ),
    SentinelDomain(
        "multiclass_classification",
        "multiclass",
        "benchmark_adapters",
    ),
    SentinelDomain(
        "weighted_regression",
        "regression",
        "benchmark_adapters",
        weighted=True,
    ),
    SentinelDomain(
        "weighted_classification",
        "binary",
        "benchmark_adapters",
        weighted=True,
    ),
)

M6_DATASETS = (
    "diabetes_resampled",
    "friedman_numeric",
    "wide_numeric_reg",
    "categorical_reg",
    "breast_cancer_resampled",
    "numeric_binary",
    "wine_resampled",
    "numeric_multiclass",
    "categorical_binary",
    "categorical_multiclass",
)
M6_SMOKE_DATASETS = (
    "friedman_numeric",
    "numeric_binary",
    "categorical_binary",
)
M6_MODELS = ("control_default", "candidate_default")
M6_SIZES = ("small",)
M6_SEED_COUNT = 3
M6_WEIGHT_MODES = ("none", "stress")
M6_REPEAT = 1
M6_THREADS = 4


def validate_contract():
    """Fail closed if the standing contract drifts out of adapter coverage."""
    domain_ids = [domain.id for domain in M5_SENTINEL_DOMAINS]
    if len(domain_ids) != len(set(domain_ids)):
        raise RuntimeError("M5 sentinel domain ids must be unique")
    if not any(domain.task == "binary" for domain in M5_SENTINEL_DOMAINS):
        raise RuntimeError("M5 must cover binary classification")
    if not any(domain.task == "multiclass" for domain in M5_SENTINEL_DOMAINS):
        raise RuntimeError("M5 must cover multiclass classification")
    weighted_tasks = {
        domain.task for domain in M5_SENTINEL_DOMAINS if domain.weighted
    }
    if not {"regression", "binary"}.issubset(weighted_tasks):
        raise RuntimeError("M5 must cover weighted regression and classification")

    unknown_datasets = sorted(set(M6_DATASETS) - set(DATASETS))
    if unknown_datasets:
        raise RuntimeError(f"unknown M6 datasets: {unknown_datasets}")
    unknown_sizes = sorted(set(M6_SIZES) - set(SIZE_SAMPLES))
    if unknown_sizes:
        raise RuntimeError(f"unknown M6 sizes: {unknown_sizes}")
    if not set(M6_SMOKE_DATASETS).issubset(M6_DATASETS):
        raise RuntimeError("M6 smoke datasets must be a subset of the full slice")
    tasks = {DATASETS[name].task for name in M6_DATASETS}
    if tasks != {"regression", "binary", "multiclass"}:
        raise RuntimeError(f"M6 task coverage drifted: {sorted(tasks)}")


def m6_expected_grid(*, smoke=False):
    """Return the exact expected M6 row identities for validation and tests."""
    datasets = M6_SMOKE_DATASETS if smoke else M6_DATASETS
    seed_count = 1 if smoke else M6_SEED_COUNT
    return tuple(
        product(
            M6_MODELS,
            datasets,
            M6_SIZES,
            range(seed_count),
            M6_WEIGHT_MODES,
        )
    )


def contract_payload():
    """Return a JSON-ready representation suitable for provenance hashing."""
    return {
        "contract_version": CONTRACT_VERSION,
        "m5_sentinel_domains": [
            asdict(domain) for domain in M5_SENTINEL_DOMAINS
        ],
        "m6": {
            "contract_frozen": M6_CONTRACT_FROZEN,
            "backtest_complete": M6_BACKTEST_COMPLETE,
            "datasets": list(M6_DATASETS),
            "smoke_datasets": list(M6_SMOKE_DATASETS),
            "models": list(M6_MODELS),
            "sizes": list(M6_SIZES),
            "seed_count": M6_SEED_COUNT,
            "weight_modes": list(M6_WEIGHT_MODES),
            "repeat": M6_REPEAT,
            "threads": M6_THREADS,
        },
    }


validate_contract()
