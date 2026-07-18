import json
import math
import pickle
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from benchmarks import analyze_synthgen_darkofit_ledger as analysis
from benchmarks import analyze_t9_synthgen_corrected_ledger as corrected
from benchmarks import run_synthgen_darkofit_ledger as runner


def test_split_indices_are_deterministic_disjoint_and_complete():
    train_a, test_a = runner.split_indices(101, 17)
    train_b, test_b = runner.split_indices(101, 17)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)
    assert len(train_a) == 75
    assert len(test_a) == 26
    assert not set(train_a) & set(test_a)
    assert set(train_a) | set(test_a) == set(range(101))
    assert runner.split_hash(train_a, test_a) == runner.split_hash(
        train_b, test_b
    )


def test_manifest_freezes_all_slices_coordinates_and_configs():
    source = {
        "commit": "a" * 40,
        "branch": "main",
        "clean": True,
        "origin_main": "a" * 40,
    }
    manifest = runner.build_manifest(source)
    benchmark = manifest["benchmark"]
    assert len(benchmark["regression_dataset_ids"]) == 48
    assert len(benchmark["categorical_canary_ids"]) == 4
    assert {name: len(ids) for name, ids in benchmark["slices"].items()} == {
        "ordinary_regression": 46,
        "noisy_nonlinear": 8,
        "smooth_linear": 11,
        "categorical_regression": 17,
    }
    assert tuple(benchmark["config_order"]) == runner.CONFIG_ORDER
    assert set(benchmark["configs"]) == set(runner.CONFIG_ORDER)
    assert benchmark["model_random_state_policy"] == "split_seed"
    assert len(manifest["run_fingerprint"]) == 64


def test_pair_summary_uses_seed_then_equal_dataset_geometric_means():
    metrics = {}
    for dataset_id, ratios in ((1, (0.8, 1.0, 1.2)), (2, (0.9, 0.9, 0.9))):
        for seed, ratio in zip(runner.SPLIT_SEEDS, ratios):
            metrics[(dataset_id, seed, runner.CONTROL)] = 2.0
            metrics[(dataset_id, seed, runner.CORE)] = 2.0 * ratio
    summary = analysis.pair_summary(
        metrics, [1, 2], runner.CORE, runner.CONTROL
    )
    first = (0.8 * 1.0 * 1.2) ** (1.0 / 3.0)
    expected = math.sqrt(first * 0.9)
    assert summary["dataset_ratios"]["1"] == pytest.approx(first)
    assert summary["aggregate_ratio"] == pytest.approx(expected)
    assert summary["wins"] == 2
    assert summary["losses"] == 0
    assert summary["maximum_split_regression"] == pytest.approx(0.2)


def test_raw_coordinate_map_rejects_missing_and_duplicates():
    records = [
        {
            "kind": "regression_ledger",
            "dataset_id": 1,
            "seed": 17,
            "arm": runner.CONTROL,
            "rmse": 1.0,
        }
    ]
    mapped = analysis._record_map(
        records,
        kind="regression_ledger",
        metric="rmse",
        expected_ids=[1],
        expected_seeds=(17,),
        expected_arms=(runner.CONTROL,),
    )
    assert mapped == {(1, 17, runner.CONTROL): 1.0}
    with pytest.raises(RuntimeError, match="duplicate"):
        analysis._record_map(
            records * 2,
            kind="regression_ledger",
            metric="rmse",
            expected_ids=[1],
            expected_seeds=(17,),
            expected_arms=(runner.CONTROL,),
        )
    with pytest.raises(RuntimeError, match="missing"):
        analysis._record_map(
            [],
            kind="regression_ledger",
            metric="rmse",
            expected_ids=[1],
            expected_seeds=(17,),
            expected_arms=(runner.CONTROL,),
        )


def test_shard_validation_rejects_duplicate_coordinates():
    task = ("canary_no_variance", 77)
    records = []
    for seed in runner.CANARY_SEEDS:
        for arm in runner.CANARY_CONFIGS:
            records.append(
                {
                    "kind": task[0],
                    "dataset_id": task[1],
                    "dataset_sha256": "d" * 64,
                    "seed": seed,
                    "arm": arm,
                }
            )
    shard = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_fingerprint": "f" * 64,
        "kind": task[0],
        "dataset_id": task[1],
        "dataset_sha256": "d" * 64,
        "splits": [
            {"seed": seed, "indices_sha256": "s" * 64}
            for seed in runner.CANARY_SEEDS
        ],
        "records": records,
    }
    runner._validate_shard(shard, task, "f" * 64)
    shard["records"][-1] = dict(shard["records"][0])
    with pytest.raises(RuntimeError, match="coordinate boundary"):
        runner._validate_shard(shard, task, "f" * 64)


def test_fit_metadata_enforces_public_small_lightgbm_thread_cap():
    base = {
        "iterations_requested": 600,
        "requested_thread_count": 6,
        "resolved_thread_count": 6,
        "selected_tree_mode": "catboost",
        "refit": False,
        "early_stopping_rounds": None,
    }
    runner._validate_fit_metadata(base, canary=False)
    lightgbm = dict(
        base,
        selected_tree_mode="lightgbm",
        resolved_thread_count=2,
    )
    runner._validate_fit_metadata(lightgbm, canary=False)
    with pytest.raises(RuntimeError, match="thread resolution"):
        runner._validate_fit_metadata(
            dict(lightgbm, resolved_thread_count=6), canary=False
        )


def test_all_frozen_regression_configs_fit_and_report_metadata():
    from darkofit import DarkoRegressor

    rng = np.random.default_rng(9)
    X = np.empty((96, 4), dtype=object)
    X[:, :3] = rng.normal(size=(96, 3))
    X[:, 3] = np.where(rng.random(96) > 0.5, "a", "b")
    y = (
        np.asarray(X[:, 0], dtype=float)
        - 0.3 * np.asarray(X[:, 1], dtype=float)
        + rng.normal(scale=0.1, size=96)
    )
    for arm in runner.CONFIG_ORDER:
        params = dict(runner.COMMON_PARAMS)
        params.update(runner.CONFIGS[arm])
        params.update(
            iterations=2,
            depth=2,
            thread_count=1,
            verbose_timing=False,
        )
        with pytest.warns(FutureWarning) if arm == runner.LINEAR_RESIDUAL else _null():
            model = DarkoRegressor(**params).fit(X, y, cat_features=[3])
        prediction = model.predict(X[:5])
        assert np.isfinite(prediction).all()
        assert len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) > 0
        metadata = runner.fitted_metadata(model)
        assert metadata["iterations_requested"] == 2
        assert metadata["requested_thread_count"] == 1
        assert metadata["resolved_thread_count"] == 1
        assert metadata["refit"] is False


def test_frozen_canary_configs_fit_and_report_metadata():
    from darkofit import DarkoClassifier

    rng = np.random.default_rng(10)
    X = np.empty((160, 4), dtype=object)
    X[:, :3] = rng.normal(size=(160, 3))
    X[:, 3] = np.where(rng.random(160) > 0.5, "a", "b")
    y = (
        np.asarray(X[:, 0], dtype=float)
        + 0.5 * (X[:, 3] == "a")
        > 0.0
    ).astype(np.int64)
    for arm in (runner.CONTROL, runner.TS4):
        params = dict(runner.CANARY_PARAMS)
        params.update(runner.CANARY_CONFIGS[arm])
        params.update(
            iterations=2,
            early_stopping_rounds=1,
            thread_count=1,
            verbose_timing=False,
        )
        model = DarkoClassifier(**params).fit(X, y, cat_features=[3])
        probability = model.predict_proba(X[:5])
        assert probability.shape == (5, 2)
        assert np.isfinite(probability).all()
        metadata = runner.fitted_metadata(model)
        assert metadata["iterations_requested"] == 2
        assert metadata["requested_thread_count"] == 1
        assert metadata["resolved_thread_count"] == 1
        assert metadata["early_stopping_rounds"] == 1


def test_t9_corrected_ledger_changes_only_superseded_outcomes():
    result = corrected.analyze()
    assert result["original_agreement_count"] == 6
    assert result["corrected_agreement_count"] == 8
    assert result["adopted_as_probe_tier_direction_finder"] is True
    changed = [
        row["number"]
        for row in result["decisions"]
        if row["original_agreement"] != row["agrees"]
    ]
    assert changed == [3, 5]
    assert result["decisions"][5]["agrees"] is False
    assert result["decisions"][5]["label_superseded"] is False
    assert all(result["adoption_gates"].values())


def test_t9_result_matches_frozen_analyzer():
    stored = json.loads(
        corrected.OUTPUT_JSON.read_text(encoding="utf-8")
    )
    regenerated = corrected.analyze()
    _assert_analysis_equal(stored, regenerated)


def test_analysis_comparison_allows_only_machine_precision_float_drift():
    value = 1.0135136999647203
    adjacent = float(np.nextafter(value, math.inf))
    _assert_analysis_equal(
        {"decision": True, "ratio": value},
        {"decision": True, "ratio": adjacent},
    )
    with pytest.raises(AssertionError, match=r"result\.ratio"):
        _assert_analysis_equal(
            {"decision": True, "ratio": value},
            {"decision": True, "ratio": value + 1e-6},
        )
    with pytest.raises(AssertionError, match=r"result\.decision"):
        _assert_analysis_equal(
            {"decision": True, "ratio": value},
            {"decision": False, "ratio": value},
        )


def _assert_analysis_equal(stored, regenerated, path="result"):
    """Compare an analysis exactly except for platform-level FP rounding."""
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


class _null:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
