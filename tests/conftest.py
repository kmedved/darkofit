"""Shared pytest classification for library and frozen campaign verifiers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest


CAMPAIGN_EXACT = frozenset({
    "test_analysis_comparison.py",
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
    "test_panel3_execution.py",
    "test_panel3_registry.py",
    "test_predict_throughput.py",
    "test_predict_throughput_contiguous_blocks.py",
    "test_predict_throughput_integrated.py",
    "test_profile_darkofit_phases.py",
    "test_remaining9_benchmarks.py",
    "test_rssi_linear_leaf_diagnosis.py",
    "test_smooth_cross_features.py",
    "test_smooth_cross_margin_analysis.py",
    "test_smooth_group_linear_selector.py",
    "test_smooth_linear_leaves_development.py",
    "test_synthgen_darkofit_ledger.py",
    "test_synthgen_harvest.py",
    "test_synthgen.py",
    "test_confirmation_target_preflight.py",
    "test_t5_composite_confirmation.py",
    "test_t5_composite_confirmation_failure.py",
    "test_t5_composite_registry.py",
    "test_t7_catboost_attribution.py",
    "test_t7b_catboost_gap_attribution.py",
    "test_t8_distributional_flagship.py",
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


def _assert_analysis_equal(stored, regenerated, path="result"):
    """Compare an analysis exactly except for platform-level FP rounding."""
    assert type(stored) is type(regenerated), (
        f"{path}: {type(stored).__name__} != "
        f"{type(regenerated).__name__}"
    )
    if isinstance(stored, float) and isinstance(regenerated, float):
        assert math.isclose(
            stored,
            regenerated,
            rel_tol=1e-14,
            abs_tol=1e-15,
        ), f"{path}: {stored!r} != {regenerated!r}"
        return
    if isinstance(stored, Mapping) and isinstance(regenerated, Mapping):
        assert stored.keys() == regenerated.keys(), path
        for key in stored:
            _assert_analysis_equal(
                stored[key],
                regenerated[key],
                f"{path}.{key}",
            )
        return
    if (
        isinstance(stored, Sequence)
        and not isinstance(stored, (str, bytes))
        and isinstance(regenerated, Sequence)
        and not isinstance(regenerated, (str, bytes))
    ):
        assert len(stored) == len(regenerated), path
        for index, (stored_item, regenerated_item) in enumerate(
            zip(stored, regenerated)
        ):
            _assert_analysis_equal(
                stored_item,
                regenerated_item,
                f"{path}[{index}]",
            )
        return
    assert stored == regenerated, path


@pytest.fixture
def assert_analysis_equal():
    """Assert artifact reproduction without requiring cross-platform FP ULPs."""
    return _assert_analysis_equal
