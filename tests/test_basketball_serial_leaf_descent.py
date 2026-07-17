from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import darkofit.tree as tree_module
from benchmarks import basketball_harness as harness
from benchmarks import run_basketball_serial_leaf_descent as experiment


def test_configure_descent_observes_actual_reference_and_candidate_calls(
    monkeypatch
):
    # ``configure_descent`` installs process-wide worker instrumentation.
    # Register every mutated name so the test restores the production module.
    monkeypatch.setattr(
        tree_module, "_update_leaves_with_split", tree_module._update_leaves_with_split
    )
    monkeypatch.setattr(
        tree_module,
        "_update_leaves_with_split_serial",
        tree_module._update_leaves_with_split_serial,
    )
    monkeypatch.setattr(
        tree_module,
        "_update_leaves_with_split_parallel",
        tree_module._update_leaves_with_split_parallel,
    )
    small_X = np.zeros((5, 1), dtype=np.uint8)
    small_leaf = np.zeros(5, dtype=np.int64)

    experiment.configure_descent(experiment.REFERENCE_CONFIG)
    tree_module._update_leaves_with_split(small_X, small_leaf, 0, 0)
    assert experiment.dispatch_counts() == {
        "serial_calls": 0,
        "parallel_calls": 1,
    }

    experiment.configure_descent(experiment.CANDIDATE_CONFIG)
    tree_module._update_leaves_with_split(small_X, small_leaf, 0, 0)
    assert experiment.dispatch_counts() == {
        "serial_calls": 1,
        "parallel_calls": 0,
    }
    np.testing.assert_array_equal(small_leaf, np.zeros(5, dtype=np.int64))


def _fit_metadata():
    return {
        "best_iteration": 1000,
        "fitted_tree_count": 1000,
        "resolved_learning_rate": 0.052312,
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "resolved_thread_count": 18,
        "refit": False,
        "selection_fit": None,
        "final_fit": {
            "iterations_requested": 1000,
            "stop_reason": "iteration_limit",
        },
    }


def _row(fold):
    return {
        "fold": fold,
        "test_indices": [fold],
        "r2": 0.5 + fold / 1000,
        "prediction_sha256": f"prediction-{fold}",
        "archive": {"bytes": 100 + fold, "sha256": f"archive-{fold}"},
        "feature_importance_sha256": f"importance-{fold}",
        "fit_metadata": _fit_metadata(),
    }


def _microbenchmark(ratio=0.02):
    return {
        "outputs_exact": True,
        "serial_over_parallel": ratio,
    }


def _canonical():
    folds = [_row(fold) for fold in range(experiment.creator.N_SPLITS)]
    holdout = {
        "scores": {
            "overlap_exposed_team_holdout": {"r2": 0.53},
            "cold_player_subset": {"r2": 0.50},
        },
        "prediction_sha256": "holdout-prediction",
        "archive": {"bytes": 125, "sha256": "holdout-archive"},
        "feature_importance_sha256": "holdout-importance",
        "fit_metadata": _fit_metadata(),
    }
    base = {
        "mean_r2": experiment.EXPECTED_DEFAULT_MEAN_R2,
        "fold_scores": [row["r2"] for row in folds],
        "folds": folds,
        "holdout": holdout,
        "behavior_fingerprint_sha256": "same-behavior",
        "microbenchmark": _microbenchmark(),
    }
    canonical = {
        experiment.REFERENCE_CONFIG: deepcopy(base),
        experiment.CANDIDATE_CONFIG: deepcopy(base),
    }
    canonical[experiment.REFERENCE_CONFIG]["descent_dispatch"] = {
        "serial_calls": 0,
        "parallel_calls": 66_000,
    }
    canonical[experiment.CANDIDATE_CONFIG]["descent_dispatch"] = {
        "serial_calls": 66_000,
        "parallel_calls": 0,
    }
    return canonical


def test_exactness_analysis_requires_payload_identity_and_dispatch_engagement():
    passing = experiment.analyze_exactness(_canonical())
    assert passing["passes_exactness_gates"] is True

    changed = _canonical()
    changed[experiment.CANDIDATE_CONFIG]["folds"][4]["archive"][
        "sha256"
    ] = "changed"
    failed = experiment.analyze_exactness(changed)
    assert failed["exactness_gates"]["fold_payloads_exact"] is False
    assert failed["passes_exactness_gates"] is False


def test_runtime_analysis_enforces_speed_memory_stability_and_microkernel():
    wall = {
        experiment.REFERENCE_CONFIG: harness.timing_summary([20.0, 20.2, 20.4]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([11.0, 11.1, 11.2]),
    }
    fit = {
        experiment.REFERENCE_CONFIG: harness.timing_summary([19.0, 19.2, 19.4]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([10.3, 10.4, 10.5]),
    }
    predict = {
        experiment.REFERENCE_CONFIG: harness.timing_summary([1.0, 1.01, 1.02]),
        experiment.CANDIDATE_CONFIG: harness.timing_summary([1.0, 1.01, 1.02]),
    }
    rss = {
        experiment.REFERENCE_CONFIG: [100, 101, 102],
        experiment.CANDIDATE_CONFIG: [100, 101, 102],
    }
    microbenchmarks = [_microbenchmark() for _ in range(6)]

    result = experiment.analyze_runtime(
        wall, fit, predict, rss, _canonical(), microbenchmarks
    )
    assert result["passes_runtime_gates"] is True

    slow_kernel = deepcopy(microbenchmarks)
    for item in slow_kernel:
        item["serial_over_parallel"] = 0.2
    result = experiment.analyze_runtime(
        wall, fit, predict, rss, _canonical(), slow_kernel
    )
    assert result["runtime_gates"]["microbenchmark_speedup"] is False
    assert result["passes_runtime_gates"] is False


def test_frozen_execution_binds_protocol_output_threads_and_subtree(monkeypatch):
    args = experiment.parse_args([])
    monkeypatch.setattr(experiment, "EXPECTED_DARKOFIT_SUBTREE", "1" * 40)
    monkeypatch.setattr(experiment, "EXPECTED_TESTS_SUBTREE", "2" * 40)
    monkeypatch.setattr(
        experiment, "EXPECTED_REPOSITORY_MANIFEST_SHA256", "3" * 64
    )
    monkeypatch.setattr(
        experiment, "EXPECTED_SUPPORT_MANIFEST_SHA256", "4" * 64
    )
    objects = {"HEAD:darkofit": "1" * 40, "HEAD:tests": "2" * 40}
    monkeypatch.setattr(experiment, "_git_object", objects.__getitem__)
    monkeypatch.setattr(
        experiment, "_repository_manifest_sha256", lambda: "3" * 64
    )
    monkeypatch.setattr(
        experiment, "_support_manifest_sha256", lambda: "4" * 64
    )

    experiment._validate_frozen_execution(args)

    args.output = args.output.with_name("wrong.json")
    with pytest.raises(RuntimeError, match="output path"):
        experiment._validate_frozen_execution(args)


def test_parse_args_rejects_nonfrozen_resources_and_output(tmp_path):
    args = experiment.parse_args([])
    assert args.threads == experiment.EXPECTED_THREADS
    assert args.output == experiment.DEFAULT_OUTPUT
    with pytest.raises(SystemExit):
        experiment.parse_args(["--threads", "4"])
    with pytest.raises(SystemExit):
        experiment.parse_args(["--output", str(tmp_path / "result.json")])


def test_prerequisite_suite_scrubs_selectors_and_attests_collection(
    monkeypatch,
):
    environments = []
    collection = "\n".join(
        f"{path}::test_attested" for path in experiment.PREREQUISITE_TESTS
    )

    def run(command, **kwargs):
        environments.append(kwargs["env"])
        stdout = collection if "--collect-only" in command else "1 passed\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setenv("PYTEST_ADDOPTS", "--ignore=tests")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_plugin")
    monkeypatch.setattr(experiment.subprocess, "run", run)

    evidence = experiment._run_prerequisite_suite()

    assert evidence["passed"] is True
    assert all(evidence["required_test_files_collected"].values())
    assert len(environments) == 2
    for environment in environments:
        assert "PYTEST_ADDOPTS" not in environment
        assert "PYTEST_PLUGINS" not in environment
        assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
