from copy import deepcopy
from functools import partial

import pytest

import darkofit.booster as booster_module
from benchmarks import basketball_harness as harness
from benchmarks import run_basketball_fused_oblivious as experiment


def test_configure_builder_wraps_only_fused_worker(monkeypatch):
    def builder(*args, **kwargs):
        return args, kwargs

    monkeypatch.setattr(booster_module, "build_oblivious_tree", builder)
    experiment.configure_builder(experiment.DEFAULT_CONFIG)
    reference = booster_module.build_oblivious_tree
    assert isinstance(reference, partial)
    assert reference.func is builder
    assert reference.keywords["fused_oblivious_kernel"] is False

    monkeypatch.setattr(booster_module, "build_oblivious_tree", builder)
    experiment.configure_builder(experiment.FUSED_CONFIG)
    wrapped = booster_module.build_oblivious_tree
    assert isinstance(wrapped, partial)
    assert wrapped.func is builder
    assert wrapped.keywords["fused_oblivious_kernel"] is True
    counter = wrapped.keywords["fused_oblivious_counter"]
    assert counter.shape == (1,)
    assert int(counter[0]) == 0
    with pytest.raises(RuntimeError, match="already wrapped"):
        experiment.configure_builder(experiment.FUSED_CONFIG)


def _fit_metadata():
    return {
        "best_iteration": 1000,
        "fitted_tree_count": 1000,
        "resolved_learning_rate": 0.052312,
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "resolved_thread_count": 4,
        "refit": False,
        "selection_fit": None,
        "final_fit": {
            "iterations_requested": 1000,
            "stop_reason": "iteration_limit",
        },
    }


def test_fitted_metadata_requires_unchanged_default_route():
    metadata = _fit_metadata()
    experiment.validate_fitted_metadata(metadata)
    invalid = deepcopy(metadata)
    invalid["selected_tree_mode"] = "lightgbm"
    with pytest.raises(RuntimeError, match="tree mode"):
        experiment.validate_fitted_metadata(invalid)


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


def _canonical():
    scores = [
        experiment.EXPECTED_DEFAULT_MEAN_R2
        for _ in range(experiment.creator.N_SPLITS)
    ]
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
        "fold_scores": scores,
        "folds": folds,
        "holdout": holdout,
        "behavior_fingerprint_sha256": "same-behavior",
        "fused_kernel_level_invocations": 0,
    }
    canonical = {
        experiment.DEFAULT_CONFIG: deepcopy(base),
        experiment.FUSED_CONFIG: deepcopy(base),
    }
    canonical[experiment.FUSED_CONFIG]["fused_kernel_level_invocations"] = 123
    return canonical


def test_exactness_analysis_fails_closed_on_archive_drift():
    passing = experiment.analyze_exactness(_canonical())
    assert passing["passes_exactness_gates"] is True

    changed = _canonical()
    changed[experiment.FUSED_CONFIG]["folds"][3]["archive"]["sha256"] = "changed"
    failed = experiment.analyze_exactness(changed)
    assert failed["exactness_gates"]["fold_payloads_exact"] is False
    assert failed["passes_exactness_gates"] is False


def test_runtime_analysis_enforces_speed_stability_prediction_and_memory():
    wall = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([10.0, 10.1, 10.2]),
        experiment.FUSED_CONFIG: harness.timing_summary([8.0, 8.1, 8.2]),
    }
    fit = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([9.0, 9.1, 9.2]),
        experiment.FUSED_CONFIG: harness.timing_summary([7.4, 7.5, 7.6]),
    }
    predict = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([1.0, 1.01, 1.02]),
        experiment.FUSED_CONFIG: harness.timing_summary([1.0, 1.01, 1.02]),
    }
    rss = {
        experiment.DEFAULT_CONFIG: [100, 101, 102],
        experiment.FUSED_CONFIG: [102, 103, 104],
    }
    result = experiment.analyze_runtime(wall, fit, predict, rss, _canonical())
    assert result["passes_runtime_gates"] is True

    slow = deepcopy(fit)
    slow[experiment.FUSED_CONFIG] = harness.timing_summary([8.0, 8.1, 8.2])
    result = experiment.analyze_runtime(wall, slow, predict, rss, _canonical())
    assert result["runtime_gates"]["fit_speedup"] is False
    assert result["passes_runtime_gates"] is False


def test_training_only_runtime_policy_records_but_does_not_gate_prediction_noise():
    wall = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([10.0, 10.1, 10.2]),
        experiment.FUSED_CONFIG: harness.timing_summary([8.0, 8.1, 8.2]),
    }
    fit = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([9.0, 9.1, 9.2]),
        experiment.FUSED_CONFIG: harness.timing_summary([7.4, 7.5, 7.6]),
    }
    predict = {
        experiment.DEFAULT_CONFIG: harness.timing_summary([1.0, 1.01, 1.02]),
        experiment.FUSED_CONFIG: harness.timing_summary([1.2, 1.21, 1.22]),
    }
    rss = {
        experiment.DEFAULT_CONFIG: [100, 101, 102],
        experiment.FUSED_CONFIG: [102, 103, 104],
    }

    result = experiment.analyze_runtime(
        wall,
        fit,
        predict,
        rss,
        _canonical(),
        runtime_policy=experiment.RUNTIME_POLICY_TRAINING_ONLY,
    )

    assert result["passes_runtime_gates"] is True
    assert "prediction_no_regression" not in result["runtime_gates"]
    assert result["prediction_timing_disposition"] == (
        "diagnostic_noncausal_for_training_only_candidate"
    )
    original = experiment.analyze_runtime(
        wall,
        fit,
        predict,
        rss,
        _canonical(),
        runtime_policy=experiment.RUNTIME_POLICY_ORIGINAL,
    )
    assert original["runtime_gates"]["prediction_no_regression"] is False
    assert original["passes_runtime_gates"] is False


def test_parse_args_requires_multithreaded_campaign(tmp_path):
    args = experiment.parse_args(
        ["--threads", "3", "--output", str(tmp_path / "result.json")]
    )
    assert args.threads == 3
    assert args.output == tmp_path / "result.json"
    assert args.runtime_policy == experiment.RUNTIME_POLICY_ORIGINAL
    confirmation = experiment.parse_args(
        [
            "--threads",
            "18",
            "--runtime-policy",
            "training-only",
            "--output",
            str(experiment.CONFIRMATION_OUTPUT),
        ]
    )
    assert confirmation.runtime_policy == experiment.RUNTIME_POLICY_TRAINING_ONLY
    with pytest.raises(SystemExit):
        experiment.parse_args(
            [
                "--threads",
                "3",
                "--runtime-policy",
                "training-only",
                "--output",
                str(experiment.CONFIRMATION_OUTPUT),
            ]
        )
    with pytest.raises(SystemExit):
        experiment.parse_args(
            [
                "--threads",
                "18",
                "--runtime-policy",
                "training-only",
            ]
        )
    with pytest.raises(SystemExit):
        experiment.parse_args(["--threads", "2"])


def test_training_only_policy_binds_frozen_candidate_and_protocol(monkeypatch):
    args = experiment.parse_args(
        [
            "--threads",
            "18",
            "--runtime-policy",
            "training-only",
            "--output",
            str(experiment.CONFIRMATION_OUTPUT),
        ]
    )
    monkeypatch.setattr(
        experiment,
        "_current_darkofit_subtree",
        lambda: experiment.CONFIRMATION_DARKOFIT_SUBTREE,
    )
    experiment._validate_runtime_policy(args)

    monkeypatch.setattr(
        experiment,
        "_current_darkofit_subtree",
        lambda: "0" * 40,
    )
    with pytest.raises(RuntimeError, match="candidate subtree changed"):
        experiment._validate_runtime_policy(args)


def test_automatic_policy_binds_promoted_subtree_and_dedicated_output(monkeypatch):
    args = experiment.parse_args(
        [
            "--threads",
            "18",
            "--runtime-policy",
            "automatic-training-only",
            "--output",
            str(experiment.AUTOMATIC_OUTPUT),
        ]
    )
    monkeypatch.setattr(
        experiment,
        "_current_darkofit_subtree",
        lambda: experiment.AUTOMATIC_DARKOFIT_SUBTREE,
    )

    experiment._validate_runtime_policy(args)

    invalid = experiment.parse_args(["--threads", "18"])
    invalid.runtime_policy = experiment.RUNTIME_POLICY_AUTOMATIC
    with pytest.raises(RuntimeError, match="output path is not exact"):
        experiment._validate_runtime_policy(invalid)


def test_atomic_benchmark_publication_never_replaces_existing_output(tmp_path):
    output = tmp_path / "result.json"
    experiment._atomic_write_new_bytes(output, b"first\n")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        experiment._atomic_write_new_bytes(output, b"second\n")

    assert output.read_bytes() == b"first\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_automatic_success_emits_final_promotion_disposition():
    assert experiment._decision_recommendation(
        True, experiment.RUNTIME_POLICY_AUTOMATIC
    ) == "promote_internal_fused_lane"
    assert experiment._decision_recommendation(
        True, experiment.RUNTIME_POLICY_TRAINING_ONLY
    ) == "advance_to_expanded_behavior_tests"
    assert experiment._decision_recommendation(
        False, experiment.RUNTIME_POLICY_AUTOMATIC
    ) == "advance_none"
