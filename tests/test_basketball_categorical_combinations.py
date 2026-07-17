import hashlib
import sys

import numpy as np
import pytest

from benchmarks import run_basketball_categorical_combinations as experiment


def _full_summary():
    timing = {
        "fit": {"iqr_fraction": 0.05},
        "held_prediction": {"iqr_fraction": 0.05},
        "cold_prediction": {"iqr_fraction": 0.05},
        "peak_rss_bytes": {"iqr_fraction": 0.05},
    }
    arm = {
        "timing": timing,
        "held_predictions_exact": True,
        "cold_predictions_exact": True,
        "tree_counts_exact": True,
        "learning_rates_exact": True,
        "combo_pairs_exact": True,
    }
    return {
        "arms": {
            experiment.CONTROL: arm,
            experiment.CANDIDATE: arm,
        },
        "ratios": {
            "fit": 1.2,
            "held_prediction": 1.05,
            "cold_prediction": 1.05,
            "peak_rss_bytes": 1.1,
        },
        "score_deltas": {
            "overlap_exposed_team_holdout": 0.001,
            "cold_player_subset": 0.002,
            "seen_player_subset": 0.0,
        },
        "peak_rss_median_delta_bytes": 10 * experiment.MIB,
    }


def _fold_summary():
    return {
        "mean_delta": 0.003,
        "wins": 7,
        "worst_delta": -0.005,
    }


def _numeric_summary():
    return {
        "train_predictions_exact": True,
        "held_predictions_exact": True,
        "feature_importance_exact": True,
        "tree_count_exact": True,
        "auto_combo_pairs": [],
        "off_combo_pairs": [],
    }


def test_canonical_string_hash_is_length_prefixed_row_major():
    values = np.asarray([["a", 12], ["bc", 3]], dtype=object)
    expected = hashlib.sha256()
    for value in ("a", "12", "bc", "3"):
        encoded = value.encode("utf-8")
        expected.update(len(encoded).to_bytes(8, "little"))
        expected.update(encoded)

    assert experiment._canonical_string_sha256(values) == expected.hexdigest()


def test_model_arms_differ_only_in_categorical_combinations(monkeypatch):
    class FakeChimeraBoostRegressor:
        def __init__(self, **kwargs):
            self._params = dict(kwargs)

        def get_params(self, deep=True):
            return dict(self._params)

    fake_chimeraboost = type(sys)("chimeraboost")
    fake_chimeraboost.ChimeraBoostRegressor = FakeChimeraBoostRegressor
    monkeypatch.setitem(sys.modules, "chimeraboost", fake_chimeraboost)

    control = experiment._build_estimator(experiment.CONTROL)
    candidate = experiment._build_estimator(experiment.CANDIDATE)
    control_params = control.get_params(deep=False)
    candidate_params = candidate.get_params(deep=False)

    assert control_params.pop("cat_combinations") is False
    assert candidate_params.pop("cat_combinations") is True
    assert control_params == candidate_params


def test_parent_binding_rejects_dangling_output_symlink(tmp_path):
    output = tmp_path / "result.json"
    output.symlink_to(tmp_path / "missing.json")

    with pytest.raises(RuntimeError, match="refusing existing output"):
        experiment._validate_parent_binding(output)


def test_worker_environment_uses_frozen_writable_numba_cache():
    environment = experiment._worker_environment()

    assert environment["NUMBA_CACHE_DIR"] == str(experiment.NUMBA_CACHE_ROOT)
    assert experiment.NUMBA_CACHE_ROOT.is_dir()


def test_decision_authorizes_only_when_every_gate_passes():
    routes = {
        "control_zero_pairs": True,
        "candidate_six_expected_pairs": True,
        "all_threads_18": True,
        "all_learning_rates_point_1": True,
        "candidate_six_combination_columns": True,
    }
    passed = experiment._decision(
        _fold_summary(), _full_summary(), _numeric_summary(), routes
    )

    assert passed["passes_all_gates"] is True
    assert passed["darkofit_implementation_authorized"] is True
    assert passed["default_policy_change_authorized"] is False
    assert passed["recommendation"] == (
        "authorize_explicit_default_off_darkofit_port"
    )


def test_quality_failure_closes_without_port():
    folds = _fold_summary()
    folds["mean_delta"] = -0.001
    result = experiment._decision(
        folds,
        _full_summary(),
        _numeric_summary(),
        {
            "control_zero_pairs": True,
            "candidate_six_expected_pairs": True,
            "all_threads_18": True,
            "all_learning_rates_point_1": True,
            "candidate_six_combination_columns": True,
        },
    )

    assert result["quality_passes"] is False
    assert result["darkofit_implementation_authorized"] is False
    assert result["recommendation"] == (
        "close_categorical_combinations_without_port"
    )


def test_resource_or_exactness_failure_cannot_authorize_port():
    full = _full_summary()
    full["ratios"]["held_prediction"] = 1.11
    numeric = _numeric_summary()
    numeric["held_predictions_exact"] = False
    result = experiment._decision(
        _fold_summary(),
        full,
        numeric,
        {
            "control_zero_pairs": True,
            "candidate_six_expected_pairs": True,
            "all_threads_18": True,
            "all_learning_rates_point_1": True,
            "candidate_six_combination_columns": True,
        },
    )

    assert result["quality_passes"] is True
    assert result["behavior_passes"] is False
    assert result["resource_passes"] is False
    assert result["darkofit_implementation_authorized"] is False
    assert result["recommendation"] == (
        "stop_donor_port_on_behavior_or_resource_gate"
    )
