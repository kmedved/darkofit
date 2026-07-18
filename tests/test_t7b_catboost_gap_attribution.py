import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import analyze_t7b_catboost_gap_attribution as analyzer
from benchmarks import run_t7b_catboost_gap_attribution as runner


def _score(rmse=1.0, digest="a" * 64):
    return {"rows": 10, "rmse": rmse, "prediction_sha256": digest}


def _resolved(learning_rate=0.05, seed=17, arm="baseline"):
    params = {key: None for key in runner.RESOLVED_KEYS}
    params.update(
        {
            "task_type": "CPU",
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "use_best_model": False,
            "eval_fraction": 0,
            "boosting_type": "Plain",
            "random_strength": 1,
            "bootstrap_type": "MVS",
            "bagging_temperature": None,
            "subsample": 0.8,
            "l2_leaf_reg": 3,
            "one_hot_max_size": 2,
            "leaf_estimation_iterations": 1,
            "leaf_estimation_backtracking": "AnyImprovement",
            "depth": 6,
            "grow_policy": "SymmetricTree",
            "learning_rate": learning_rate,
            "iterations": runner.ITERATIONS,
            "random_seed": seed,
        }
    )
    params.update(runner.ARMS[arm])
    if "bootstrap_type" in runner.ARMS[arm]:
        params["bagging_temperature"] = None
        params["subsample"] = None
    return params


def _integrity_result(task_id=123, seed=17):
    arms = []
    for name in runner.ARM_NAMES:
        arms.append(
            {
                "arm": name,
                "validation": _score(),
                "test": _score(),
                "resolved_params": _resolved(seed=seed, arm=name),
            }
        )
    return {
        "task_id": task_id,
        "seed": seed,
        "categorical_count": 0 if task_id == runner.CARS_TASK_ID else 1,
        "arms": arms,
    }


def _ratio_results(ratio):
    results = []
    execution_index = 0
    for task_id in range(8):
        for fold in runner.FOLDS:
            for seed in runner.SEEDS:
                results.append(
                    {
                        "execution_index": execution_index,
                        "task_id": task_id,
                        "dataset_name": f"task-{task_id}",
                        "fold": fold,
                        "seed": seed,
                        "arms": [
                            {
                                "arm": "baseline",
                                "validation": _score(1.0),
                                "test": _score(1.0),
                            },
                            {
                                "arm": "candidate",
                                "validation": _score(ratio),
                                "test": _score(ratio),
                            },
                        ],
                    }
                )
                execution_index += 1
    return results


def _artifact_source():
    head = "1" * 40
    return {
        "path": str(runner.ROOT),
        "head": head,
        "branch": "main",
        "clean": True,
        "status": [],
        "describe": head[:7],
        "remotes": {"origin": "https://example.invalid/darkofit.git"},
        "tracked_main_refs": {"origin/main": head},
        "remote_branch_head": head,
    }


def _artifact_runtime():
    return {
        "python": "3.12.13",
        "machine": {
            "platform": "test-platform",
            "machine": "test-machine",
            "cpu_brand": None,
            "logical_cpu_count": 18,
            "python": "3.12.13",
            "python_executable": "/test/python",
        },
        "dependencies": {
            "numpy": "test",
            "pandas": "test",
            "scikit-learn": "test",
            "joblib": "test",
            "numba": "test",
            "catboost": "1.2.10",
        },
        "environment": runner.EXPECTED_ENVIRONMENT,
    }


def _artifact_result(expected, coordinate, frozen):
    arms = []
    baseline_validation = _score()
    baseline_test = _score()
    if expected["seed"] == 4:
        baseline_validation["prediction_sha256"] = coordinate[
            "seed4_validation_prediction_sha256"
        ]
        baseline_test["prediction_sha256"] = coordinate[
            "seed4_test_prediction_sha256"
        ]
    for position, arm in enumerate(
        runner._arm_order(expected["execution_index"])
    ):
        validation = dict(baseline_validation) if arm == "baseline" else _score()
        test = dict(baseline_test) if arm == "baseline" else _score()
        if expected["task_id"] == runner.CARS_TASK_ID and arm in {
            "one_hot_max_size_0",
            "one_hot_max_size_255",
        }:
            validation = dict(baseline_validation)
            test = dict(baseline_test)
        arms.append(
            {
                "arm": arm,
                "position": position,
                "overrides": runner.ARMS[arm],
                "fit_seconds": 1.0,
                "validation": validation,
                "test": test,
                "prediction_timing": {
                    "calls": runner.t7.PREDICTION_CALLS,
                    "median_seconds": 0.01,
                    "total_seconds": 0.2,
                },
                "tree_count": runner.ITERATIONS,
                "requested_policy": runner._requested_policy(
                    coordinate, expected["seed"], arm
                ),
                "constructor_params_observed": {
                    "thread_count": runner.THREADS_PER_WORKER
                },
                "resolved_params": _resolved(
                    coordinate["frozen_learning_rate"],
                    expected["seed"],
                    arm,
                ),
            }
        )
    result = {
        **expected,
        "dataset_id": frozen["dataset_id"],
        "dataset_name": frozen["dataset_name"],
        "lineage_cluster": frozen["lineage_cluster"],
        "frozen_learning_rate": coordinate["frozen_learning_rate"],
        "n_features": frozen["n_features"],
        "categorical_count": frozen["categorical_count"],
        "outer_split": frozen["outer_split"],
        "inner_split": frozen["inner_split"],
        "arm_order": list(runner._arm_order(expected["execution_index"])),
        "arms": arms,
        "warmup_seconds": 0.1,
        "peak_rss_bytes": 1_000_000,
    }
    runner._integrity_checks(result, coordinate)
    result["behavior_sha256"] = runner._json_sha256(
        {
            "execution_index": result["execution_index"],
            "task_id": result["task_id"],
            "fold": result["fold"],
            "seed": result["seed"],
            "arms": [
                {
                    "arm": arm["arm"],
                    "validation": arm["validation"],
                    "test": arm["test"],
                    "requested_policy": arm["requested_policy"],
                    "constructor_params_observed": arm[
                        "constructor_params_observed"
                    ],
                    "resolved_params": arm["resolved_params"],
                }
                for arm in arms
            ],
            "integrity": result["integrity"],
        }
    )
    return result


def _raw_artifact():
    freeze = runner._source_freeze()
    declaration, frozen = runner._coordinates()
    source = _artifact_source()
    binding = runner._binding(source, freeze, declaration)
    schedule = runner._schedule(declaration["coordinates"])
    results = [
        _artifact_result(
            expected,
            declaration["coordinates"][expected["coordinate_index"]],
            frozen[expected["coordinate_index"]],
        )
        for expected in schedule
    ]
    payload = {
        "schema_version": 1,
        "name": "darkofit_t7b_catboost_gap_attribution_raw_v1",
        "created_at": "2026-07-18T12:00:00+00:00",
        "development_data_only": True,
        "lockbox_data_used": False,
        "confirmation_outcomes_inspected": False,
        "default_change_authorized": False,
        "source": source,
        "runtime": _artifact_runtime(),
        "protocol": binding,
        "coordinate_count": 24,
        "execution_count": 72,
        "fit_count": 576,
        "resumed_execution_count": 0,
        "results": results,
        "spool_records": [
            {
                "path": runner._spool_path(
                    Path("/spool"),
                    result["task_id"],
                    result["fold"],
                    result["seed"],
                ).name,
                "sha256": analyzer._spool_digest(binding, result),
                "resumed": False,
            }
            for result in results
        ],
    }
    payload["raw_sha256"] = runner._json_sha256(payload)
    return payload


def test_t7b_is_frozen_before_execution_in_a_new_namespace():
    assert runner.PROTOCOL.name.startswith("t7b_")
    assert runner.COORDINATES.name.startswith("t7b_")
    assert runner.FREEZE.name.startswith("t7b_")
    assert runner.DEFAULT_OUTPUT.name.startswith("t7b_")
    assert runner.DEFAULT_SPOOL.name.startswith("t7b-")
    freeze = runner._source_freeze()
    assert freeze["status"] == "frozen_not_executed"
    assert freeze["catboost_version"] == "1.2.10"
    assert freeze["environment"] == runner.EXPECTED_ENVIRONMENT
    assert len(freeze["darkofit_model_head"]) == 40
    assert freeze["runtime"]["task_type"] == "CPU"


def test_t7b_outputs_are_create_only_in_temporary_paths(tmp_path):
    output = tmp_path / "raw.json"
    runner.t7._create_output(output, b"first\n")
    with pytest.raises(RuntimeError, match="refusing existing output"):
        runner.t7._create_output(output, b"second\n")
    assert output.read_bytes() == b"first\n"


def test_t7b_freeze_binds_every_declared_source():
    freeze = runner._source_freeze()
    assert set(freeze["source_sha256"]) == set(runner._source_paths())
    assert "test_classifier" not in freeze["source_sha256"]
    assert all(
        runner._sha256(path) == freeze["source_sha256"][name]
        for name, path in runner._source_paths().items()
    )
    assert all(
        runner._git_blob_sha256(
            path, freeze["darkofit_model_head"]
        )
        == freeze["source_sha256"][name]
        for name, path in runner._source_paths().items()
    )
    canonical = dict(freeze)
    digest = canonical.pop("freeze_sha256")
    assert runner._json_sha256(canonical) == digest


def test_t7b_historical_freeze_does_not_require_live_source(
    tmp_path, monkeypatch
):
    path = tmp_path / "source.py"
    path.write_text("later source\n", encoding="utf-8")
    expected_digest = "a" * 64
    freeze = {
        "schema_version": 1,
        "name": "darkofit_t7b_catboost_gap_attribution_freeze_v1",
        "status": "frozen_not_executed",
        "catboost_version": "1.2.10",
        "darkofit_model_head": "1" * 40,
        "environment": runner.EXPECTED_ENVIRONMENT,
        "runtime": {
            "task_type": "CPU",
            "threads_per_worker": runner.THREADS_PER_WORKER,
            "concurrent_workers": runner.CONCURRENT_WORKERS,
            "iterations": runner.ITERATIONS,
            "seeds": list(runner.SEEDS),
            "arms": list(runner.ARM_NAMES),
        },
        "source_sha256": {"unit": expected_digest},
    }
    freeze["freeze_sha256"] = runner._json_sha256(freeze)
    monkeypatch.setattr(
        runner, "_load_json", lambda _path, _context: dict(freeze)
    )
    monkeypatch.setattr(runner, "_source_paths", lambda: {"unit": path})
    monkeypatch.setattr(
        runner,
        "_git_blob_sha256",
        lambda _path, _revision: expected_digest,
    )
    assert runner._source_freeze(verify_live=False) == freeze
    with pytest.raises(RuntimeError, match="live source"):
        runner._source_freeze()


def test_t7b_environment_contract_is_exact(monkeypatch):
    expected = runner.EXPECTED_ENVIRONMENT
    monkeypatch.setattr(runner, "_runtime_environment", lambda: expected)
    assert runner._validate_runtime_environment(expected) == expected
    changed = {
        "python": dict(expected["python"]),
        "dependencies": dict(expected["dependencies"]),
    }
    changed["dependencies"]["numpy"] = "999"
    monkeypatch.setattr(runner, "_runtime_environment", lambda: changed)
    with pytest.raises(RuntimeError, match="exact frozen"):
        runner._validate_runtime_environment(expected)


def test_t7b_execution_head_allows_only_freeze_commit(monkeypatch):
    source = {"head": "2" * 40}
    freeze = {"darkofit_model_head": "1" * 40}
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1:3] == ["merge-base", "--is-ancestor"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0,
            stdout=(
                next(iter(runner.ALLOWED_POST_MODEL_SOURCE_PATHS)).encode()
                + b"\0"
            ),
            stderr=b"",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner._validate_execution_source(source, freeze) == set(
        runner.ALLOWED_POST_MODEL_SOURCE_PATHS
    )
    assert len(calls) == 2

    def changed_run(command, **_kwargs):
        if command[1:3] == ["merge-base", "--is-ancestor"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0,
            stdout=b"darkofit/booster.py\0",
            stderr=b"",
        )

    monkeypatch.setattr(runner.subprocess, "run", changed_run)
    with pytest.raises(RuntimeError, match="only by"):
        runner._validate_execution_source(source, freeze)


def test_t7b_historical_coordinates_use_model_head_blobs(monkeypatch):
    seen = []

    def fake_blob(path, revision):
        seen.append((path, revision))
        return path.read_bytes()

    monkeypatch.setattr(runner, "_git_blob", fake_blob)
    declaration, frozen = runner._coordinates(source_head="1" * 40)
    assert declaration["coordinate_count"] == 24
    assert len(frozen) == 24
    assert {path for path, _revision in seen} == {
        runner.COORDINATES,
        runner.t7.REGISTRY,
        runner.t7.C2_RAW,
        runner.T7_RAW,
        runner.T7_SUMMARY,
    }
    assert {revision for _path, revision in seen} == {"1" * 40}


def test_t7b_coordinate_freeze_reproduces_t7_defaults_and_c2_schedule():
    declaration, frozen = runner._coordinates()
    assert declaration["seeds"] == [4, 17, 29]
    assert declaration["arms"] == list(runner.ARM_NAMES)
    assert declaration["coordinate_count"] == 24
    assert declaration["coordinates_sha256"] == runner._json_sha256(
        declaration["coordinates"]
    )
    assert len(frozen) == 24
    for coordinate in declaration["coordinates"]:
        result = frozen[coordinate["coordinate_index"]]
        default = next(
            arm for arm in result["arms"] if arm["arm"] == "default"
        )
        assert (
            coordinate["frozen_learning_rate"]
            == default["resolved_params"]["learning_rate"]
        )
        assert (
            coordinate["seed4_validation_prediction_sha256"]
            == default["validation"]["prediction_sha256"]
        )
        assert (
            coordinate["seed4_test_prediction_sha256"]
            == default["test"]["prediction_sha256"]
        )
        assert result["outer_split"]
        assert result["inner_split"]
    cars = [
        row
        for row in declaration["coordinates"]
        if row["task_id"] == runner.CARS_TASK_ID
    ]
    assert len(cars) == 3
    assert all(
        frozen[row["coordinate_index"]]["categorical_count"] == 0
        for row in cars
    )


def test_t7b_schedule_and_arm_rotation_are_exact_and_balanced():
    declaration, _ = runner._coordinates()
    schedule = runner._schedule(declaration["coordinates"])
    assert len(schedule) == 72
    assert [row["execution_index"] for row in schedule] == list(range(72))
    assert {row["seed"] for row in schedule} == set(runner.SEEDS)
    orders = [runner._arm_order(index) for index in range(72)]
    assert all(set(order) == set(runner.ARM_NAMES) for order in orders)
    for position in range(len(runner.ARM_NAMES)):
        counts = {
            arm: sum(order[position] == arm for order in orders)
            for arm in runner.ARM_NAMES
        }
        assert set(counts.values()) == {9}


def test_t7b_arms_and_shared_learning_rate_policy_are_exact():
    assert runner.ARMS == {
        "baseline": {},
        "random_strength_0": {"random_strength": 0},
        "bootstrap_no": {"bootstrap_type": "No"},
        "no_split_noise_or_row_sampling": {
            "random_strength": 0,
            "bootstrap_type": "No",
        },
        "l2_leaf_reg_1": {"l2_leaf_reg": 1},
        "one_hot_max_size_0": {"one_hot_max_size": 0},
        "one_hot_max_size_255": {"one_hot_max_size": 255},
        "leaf10_any_improvement": {
            "leaf_estimation_iterations": 10,
            "leaf_estimation_backtracking": "AnyImprovement",
        },
    }
    coordinate = {"frozen_learning_rate": 0.05}
    params = _resolved(learning_rate=0.05, seed=17)
    runner._validate_resolved(params, coordinate, 17)
    params["learning_rate"] = 0.1
    with pytest.raises(RuntimeError, match="shared policy"):
        runner._validate_resolved(params, coordinate, 17)
    params["learning_rate"] = 0.05
    params["l2_leaf_reg"] = 1
    runner._validate_resolved(params, coordinate, 17, "l2_leaf_reg_1")
    params["l2_leaf_reg"] = 3
    with pytest.raises(RuntimeError, match="arm override"):
        runner._validate_resolved(params, coordinate, 17, "l2_leaf_reg_1")
    numeric_one_hot = _resolved(
        learning_rate=0.05, seed=17, arm="one_hot_max_size_255"
    )
    numeric_one_hot["one_hot_max_size"] = None
    runner._validate_resolved(
        numeric_one_hot,
        coordinate,
        17,
        "one_hot_max_size_255",
        categorical_count=0,
    )
    with pytest.raises(RuntimeError, match="arm override"):
        runner._validate_resolved(
            numeric_one_hot,
            coordinate,
            17,
            "one_hot_max_size_255",
            categorical_count=1,
        )


def test_t7b_thread_count_is_constructor_observed_not_faked_as_resolved():
    class Model:
        def get_all_params(self):
            return _resolved()

        def get_params(self):
            return {"thread_count": runner.THREADS_PER_WORKER}

    resolved = runner._resolved_params(Model())
    observed = runner._constructor_params_observed(Model())
    assert "thread_count" not in resolved
    assert observed == {"thread_count": runner.THREADS_PER_WORKER}
    runner._validate_constructor_params_observed(observed)
    with pytest.raises(RuntimeError, match="constructor-observed"):
        runner._validate_constructor_params_observed({"thread_count": None})


def test_t7b_seed4_baseline_and_cars_negative_controls_fail_closed():
    coordinate = {
        "seed4_validation_prediction_sha256": "a" * 64,
        "seed4_test_prediction_sha256": "a" * 64,
    }
    result = _integrity_result(seed=4)
    runner._integrity_checks(result, coordinate)
    assert result["integrity"]["seed4_t7_baseline_byte_match"] is True
    assert result["integrity"]["arm_isolation_verified"] is True
    result["arms"][0]["test"]["prediction_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="byte-match"):
        runner._integrity_checks(result, coordinate)

    cars = _integrity_result(task_id=runner.CARS_TASK_ID, seed=17)
    runner._integrity_checks(cars, coordinate)
    assert cars["integrity"]["cars_one_hot_negative_control"] == {
        "one_hot_max_size_0": True,
        "one_hot_max_size_255": True,
    }
    by_arm = {arm["arm"]: arm for arm in cars["arms"]}
    by_arm["one_hot_max_size_255"]["test"]["prediction_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="negative control"):
        runner._integrity_checks(cars, coordinate)


def test_t7b_worker_schedule_rejects_wrong_seed_without_loading_data(
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "_source_freeze",
        lambda: {"catboost_version": "1.2.10"},
    )
    declaration = {
        "coordinates": [
            {
                "coordinate_index": 0,
                "task_id": 123,
                "fold": 0,
            }
        ]
    }
    monkeypatch.setattr(
        runner, "_coordinates", lambda: (declaration, {0: {}})
    )
    with pytest.raises(ValueError, match="frozen schedule"):
        runner.run_worker(123, 0, 29, 0)


def test_t7b_subprocess_command_targets_new_runner(tmp_path, monkeypatch):
    expected = {
        "execution_index": 0,
        "coordinate_index": 0,
        "task_id": 123,
        "fold": 0,
        "seed": 4,
    }
    result = {
        **expected,
        "arm_order": list(runner._arm_order(0)),
        "arms": [
            {"arm": arm}
            for arm in runner._arm_order(0)
        ],
        "behavior_sha256": "a" * 64,
    }
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout=runner.WORKER_PREFIX + json.dumps(result) + "\n",
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.t7, "_environment", lambda: {})
    monkeypatch.setattr(
        runner.t7,
        "_create_spool",
        lambda _path, _binding, value, return_publish_state=False: (
            value,
            "b" * 64,
        ),
    )
    stored, _record, resumed = runner._run_one(
        expected, tmp_path, {"binding": True}
    )
    assert stored == result
    assert resumed is False
    assert Path(seen["command"][1]).resolve() == Path(runner.__file__).resolve()
    assert "run_t7_catboost_attribution.py" not in seen["command"][1]


def test_t7b_interpretation_has_opposite_explanatory_and_config_directions(
    monkeypatch,
):
    monkeypatch.setattr(
        analyzer,
        "_hierarchical_bounds",
        lambda records: {
            "draws": analyzer.BOOTSTRAP_DRAWS,
            "seed": 7_017,
            "bonferroni_lower": min(
                row["ratio"] for rows in records.values() for row in rows
            ),
            "bonferroni_upper": max(
                row["ratio"] for rows in records.values() for row in rows
            ),
        },
    )
    contributor = analyzer._contrast(
        _ratio_results(1.02), "candidate", 1.03
    )
    assert contributor["label"] == "contributor"
    assert contributor["historical_gap_fraction_erased_seed4_bridge"] > 0
    promising = analyzer._contrast(
        _ratio_results(0.98), "candidate", 1.03
    )
    assert promising["label"] == "promising_config"
    assert promising["historical_gap_fraction_erased_seed4_bridge"] < 0


def test_t7b_bootstrap_is_deterministic_and_one_sided():
    records = {
        task_id: [
            {
                "fold": fold,
                "seed": seed,
                "ratio": 1.01 + task_id * 0.001,
            }
            for fold in runner.FOLDS
            for seed in runner.SEEDS
        ]
        for task_id in range(8)
    }
    first = analyzer._hierarchical_bounds(records, draws=200, seed=7017)
    second = analyzer._hierarchical_bounds(records, draws=200, seed=7017)
    assert first == second
    assert (
        1
        < first["bonferroni_lower"]
        <= first["bonferroni_upper"]
    )


def _force_bounds(monkeypatch, *, lower, upper):
    monkeypatch.setattr(
        analyzer,
        "_hierarchical_bounds",
        lambda _records: {
            "bonferroni_lower": lower,
            "bonferroni_upper": upper,
        },
    )


def test_t7b_contributor_rejects_one_task_concentration(monkeypatch):
    _force_bounds(monkeypatch, lower=1.01, upper=1.05)
    results = _ratio_results(1.0)
    for result in results:
        if result["task_id"] == 0:
            result["arms"][1]["validation"]["rmse"] = 1.2
            result["arms"][1]["test"]["rmse"] = 1.2
    contrast = analyzer._contrast(results, "candidate", 1.03)
    assert contrast["equal_dataset_test_ratio"] > 1
    assert contrast["least_favorable_contributor_loo_ratio"] == pytest.approx(1)
    assert contrast["contributor_gates"][
        "every_leave_one_task_out_ratio_gt_1"
    ] is False
    assert contrast["label"] == "not_attributed"


def test_t7b_contributor_rejects_seed_block_reversal(monkeypatch):
    _force_bounds(monkeypatch, lower=1.001, upper=1.03)
    results = _ratio_results(1.02)
    for result in results:
        if result["seed"] == 29:
            result["arms"][1]["validation"]["rmse"] = 0.98
            result["arms"][1]["test"]["rmse"] = 0.98
    contrast = analyzer._contrast(results, "candidate", 1.03)
    assert contrast["equal_dataset_test_ratio"] > 1
    assert contrast["seed_test_ratios"]["29"] == pytest.approx(0.98)
    assert contrast["contributor_gates"][
        "every_seed_block_ratio_gt_1"
    ] is False
    assert contrast["label"] == "not_attributed"


def test_t7b_contributor_rejects_validation_reversal(monkeypatch):
    _force_bounds(monkeypatch, lower=1.01, upper=1.03)
    results = _ratio_results(1.02)
    for result in results:
        result["arms"][1]["validation"]["rmse"] = 0.99
    contrast = analyzer._contrast(results, "candidate", 1.03)
    assert contrast["equal_dataset_test_ratio"] > 1
    assert contrast["equal_dataset_validation_ratio"] < 1
    assert contrast["contributor_gates"][
        "equal_dataset_validation_ratio_gt_1"
    ] is False
    assert contrast["label"] == "not_attributed"


def test_t7b_promising_config_rejects_worst_task_harm(monkeypatch):
    _force_bounds(monkeypatch, lower=0.95, upper=0.99)
    results = _ratio_results(0.98)
    for result in results:
        if result["task_id"] == 0:
            result["arms"][1]["test"]["rmse"] = 1.03
    contrast = analyzer._contrast(results, "candidate", 1.03)
    assert contrast["equal_dataset_test_ratio"] <= 0.995
    assert contrast["worst_task_test_ratio"] == pytest.approx(1.03)
    assert contrast["promising_config_gates"][
        "worst_task_test_ratio_lte_1_02"
    ] is False
    assert contrast["label"] == "not_attributed"


def test_t7b_bootstrap_averages_fixed_crossed_seed_blocks():
    ratios = {
        4: math.exp(-1),
        17: 1.0,
        29: math.exp(1),
    }
    records = {
        task_id: [
            {"fold": fold, "seed": seed, "ratio": ratios[seed]}
            for fold in runner.FOLDS
            for seed in runner.SEEDS
        ]
        for task_id in range(8)
    }
    bounds = analyzer._hierarchical_bounds(records, draws=200, seed=7017)
    assert bounds["seed_treatment"] == "fixed_average_within_sampled_fold"
    assert bounds["bonferroni_lower"] == pytest.approx(1.0)
    assert bounds["bonferroni_upper"] == pytest.approx(1.0)


def test_t7b_multiplicity_control_covers_both_directions_per_arm():
    assert analyzer.DIRECTIONAL_HYPOTHESES == 14
    assert analyzer.PER_DIRECTION_ALPHA == pytest.approx(0.05 / 14)
    assert analyzer.LOWER_QUANTILE == analyzer.PER_DIRECTION_ALPHA
    assert analyzer.UPPER_QUANTILE == 1 - analyzer.PER_DIRECTION_ALPHA
    assert analyzer.BOOTSTRAP_DRAWS == 100_000


def _summary_contrast(arm, ratio, label):
    return {
        "arm": arm,
        "label": label,
        "equal_dataset_test_ratio": ratio,
        "equal_dataset_validation_ratio": ratio,
        "seed_test_ratios": {str(seed): ratio for seed in runner.SEEDS},
        "per_task": [
            {"task_id": task_id, "test_ratio": ratio}
            for task_id in range(8)
        ],
    }


def test_t7b_noise_sampling_result_is_descriptive_incremental_not_complementary():
    by_arm = {
        "random_strength_0": _summary_contrast(
            "random_strength_0", 1.01, "not_attributed"
        ),
        "bootstrap_no": _summary_contrast(
            "bootstrap_no", 1.015, "not_attributed"
        ),
        "no_split_noise_or_row_sampling": _summary_contrast(
            "no_split_noise_or_row_sampling", 1.03, "contributor"
        ),
    }
    result = analyzer._incremental_vs_components(by_arm)
    assert result["inferential_claim_authorized"] is False
    assert (
        result["status"]
        == "explanatory_incremental_vs_both_components_descriptive"
    )
    assert "complement" not in result["status"]
    assert result["direct_contrasts"]["bootstrap_no"][
        "combined_over_component_test_ratio"
    ] == pytest.approx(1.03 / 1.015)


def test_t7b_requested_policy_and_arm_isolation_fail_closed():
    coordinate = {"frozen_learning_rate": 0.05}
    policy = runner._requested_policy(coordinate, 17, "baseline")
    runner._validate_requested(policy, coordinate, 17, "baseline")
    changed = {
        "constructor_params": dict(policy["constructor_params"]),
        "fit_policy": policy["fit_policy"],
    }
    changed["constructor_params"]["thread_count"] = 1
    with pytest.raises(RuntimeError, match="requested policy"):
        runner._validate_requested(changed, coordinate, 17, "baseline")

    result = _integrity_result(seed=17)
    by_arm = {arm["arm"]: arm for arm in result["arms"]}
    by_arm["random_strength_0"]["resolved_params"]["depth"] = 8
    with pytest.raises(RuntimeError, match="undeclared parameter"):
        runner._integrity_checks(result, coordinate)


@pytest.mark.parametrize("changed", ["source", "freeze", "declaration"])
def test_t7b_closing_source_recheck_fails_on_any_change(
    monkeypatch, changed
):
    opening_source = {"head": "a" * 40}
    opening_freeze = {"freeze_sha256": "b" * 64}
    opening_declaration = {"coordinates_sha256": "c" * 64}
    closing_source = dict(opening_source)
    closing_freeze = dict(opening_freeze)
    closing_declaration = dict(opening_declaration)
    if changed == "source":
        closing_source["head"] = "d" * 40
    elif changed == "freeze":
        closing_freeze["freeze_sha256"] = "e" * 64
    else:
        closing_declaration["coordinates_sha256"] = "f" * 64
    monkeypatch.setattr(runner, "_git_state", lambda: closing_source)
    monkeypatch.setattr(runner, "_source_freeze", lambda: closing_freeze)
    monkeypatch.setattr(
        runner,
        "_coordinates",
        lambda: (closing_declaration, {}),
    )
    with pytest.raises(RuntimeError, match="changed during execution"):
        runner._verify_closing_state(
            opening_source, opening_freeze, opening_declaration
        )


def test_t7b_spool_digest_binds_binding_and_full_result():
    binding = {"freeze": "a" * 64}
    result = {"task_id": 123, "fold": 2, "seed": 17, "value": 1.0}
    first = analyzer._spool_digest(binding, result)
    changed = dict(result)
    changed["value"] = 2.0
    assert first != analyzer._spool_digest(binding, changed)
    assert first != analyzer._spool_digest({"freeze": "b" * 64}, result)


def test_t7b_raw_validation_binds_file_and_canonical_hashes(
    tmp_path, monkeypatch
):
    raw = _raw_artifact()
    path = tmp_path / "raw.json"
    encoded = (
        json.dumps(raw, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    runner.t7._create_output(path, encoded)
    monkeypatch.setattr(
        runner,
        "_validate_execution_source",
        lambda _source, _freeze: set(
            runner.ALLOWED_POST_MODEL_SOURCE_PATHS
        ),
    )
    loaded, file_sha256 = analyzer.load(
        path, return_file_sha256=True
    )
    assert loaded["raw_sha256"] == raw["raw_sha256"]
    assert file_sha256 != raw["raw_sha256"]
    _force_bounds(monkeypatch, lower=1.0, upper=1.0)
    summary = analyzer.analyze(loaded, raw_file_sha256=file_sha256)
    assert summary["raw_file_sha256"] == file_sha256
    assert summary["raw_canonical_sha256"] == raw["raw_sha256"]

    changed = json.loads(path.read_text())
    changed["source"]["branch"] = "not-main"
    changed.pop("raw_sha256")
    changed["raw_sha256"] = runner._json_sha256(changed)
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid T7b raw artifact"):
        analyzer.load(changed_path)


def test_t7b_analysis_rejects_missing_raw_file_identity():
    with pytest.raises(ValueError, match="exact raw file"):
        analyzer.analyze({}, raw_file_sha256="not-a-sha")


def test_t7b_protocol_predeclares_directional_gates_and_no_promotion():
    text = runner.PROTOCOL.read_text(encoding="utf-8")
    assert "multiplicity-adjusted hierarchical-bootstrap lower bound" in text
    assert "multiplicity-adjusted hierarchical-bootstrap upper bound" in text
    assert "historical CatBoost/DarkoFit" in text
    assert "erased by the CatBoost perturbation" in text
    assert "14 directional claims" in text
    assert "fixed repeat blocks" in text
    assert "promising_config" in text
    assert "No T7b result" in text
    assert "can change a default" in text
    assert "one_hot_max_size=0" in text
    assert "one_hot_max_size=255" in text
    assert "task 361622" in text


def test_t7b_cli_defaults_are_repository_anchored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_args = runner.parse_args([])
    analysis_args = analyzer.parse_args([])
    assert run_args.output == runner.DEFAULT_OUTPUT
    assert run_args.spool == runner.DEFAULT_SPOOL
    assert analysis_args.input == analyzer.DEFAULT_INPUT
    assert analysis_args.output == analyzer.DEFAULT_OUTPUT
    assert analysis_args.markdown == analyzer.DEFAULT_MARKDOWN
