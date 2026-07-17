"""Shared pytest classification for library and frozen campaign verifiers."""

from __future__ import annotations

from pathlib import Path

import pytest


CAMPAIGN_EXACT = frozenset({
    "test_bench_status.py",
    "test_bench_vs_lightgbm.py",
    "test_benchmark_adapters.py",
    "test_default_regret_report.py",
    "test_fused_oblivious_expanded.py",
    "test_fused_subset_oblivious.py",
    "test_fused_variable_hessian_runner.py",
    "test_large_n_engine.py",
    "test_native_ordinal_c2.py",
    "test_native_ordinal_c2_registry.py",
    "test_predict_throughput.py",
    "test_predict_throughput_contiguous_blocks.py",
    "test_predict_throughput_integrated.py",
    "test_profile_darkofit_phases.py",
    "test_remaining9_benchmarks.py",
    "test_smooth_group_linear_selector.py",
    "test_smooth_linear_leaves_development.py",
    "test_synthgen_darkofit_ledger.py",
    "test_synthgen_harvest.py",
    "test_synthgen.py",
    "test_vector_fit_profile.py",
})
CAMPAIGN_PREFIXES = (
    "test_basketball_",
    "test_ctr23_",
    "test_fresh_",
    "test_tabarena_",
)


def is_campaign_module(path: str | Path) -> bool:
    name = Path(path).name
    return name in CAMPAIGN_EXACT or name.startswith(CAMPAIGN_PREFIXES)


def _requested_partition(config) -> str | None:
    markexpr = " ".join(config.getoption("markexpr").split())
    if markexpr == "campaign":
        return "campaign"
    if markexpr == "not campaign":
        return "library"
    return None


def pytest_ignore_collect(collection_path, config):
    path = Path(collection_path)
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    partition = _requested_partition(config)
    if partition == "campaign":
        return True if not is_campaign_module(path) else None
    if partition == "library":
        return True if is_campaign_module(path) else None
    return None


def pytest_collection_modifyitems(items):
    marker = pytest.mark.campaign
    for item in items:
        if is_campaign_module(item.path):
            item.add_marker(marker)
