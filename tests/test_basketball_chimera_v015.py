from copy import deepcopy
import sys

import pytest

from benchmarks import basketball_harness as harness
from benchmarks import run_basketball_chimera_v015 as experiment


def test_build_estimator_freezes_product_and_matched_lanes(monkeypatch, tmp_path):
    class FakeChimeraBoostRegressor:
        def __init__(self, **kwargs):
            values = {
                "n_estimators": 2000,
                "learning_rate": None,
                "depth": None,
                "l2_leaf_reg": 1.0,
                "max_bins": 128,
                "early_stopping": True,
                "linear_leaves": None,
                "cross_features": None,
                "cat_combinations": None,
            }
            values.update(kwargs)
            for name, value in values.items():
                setattr(self, name, value)

    fake_chimeraboost = type(sys)("chimeraboost")
    fake_chimeraboost.ChimeraBoostRegressor = FakeChimeraBoostRegressor
    monkeypatch.setitem(sys.modules, "chimeraboost", fake_chimeraboost)
    repo = tmp_path
    darko_product = experiment.build_estimator(
        experiment.PRODUCT_LANE, experiment.DARKOFIT_ARM, repo
    )
    chimera_product = experiment.build_estimator(
        experiment.PRODUCT_LANE, experiment.CHIMERABOOST_ARM, repo
    )
    darko_matched = experiment.build_estimator(
        experiment.MATCHED_LANE, experiment.DARKOFIT_ARM, repo
    )
    chimera_matched = experiment.build_estimator(
        experiment.MATCHED_LANE, experiment.CHIMERABOOST_ARM, repo
    )

    assert darko_product.iterations == 1000
    assert darko_product.learning_rate is None
    assert darko_product.early_stopping is False
    assert chimera_product.n_estimators == 2000
    assert chimera_product.early_stopping is True
    assert chimera_product.linear_leaves is None
    assert chimera_product.cross_features is None

    assert darko_matched.iterations == chimera_matched.n_estimators == 1000
    assert darko_matched.learning_rate == chimera_matched.learning_rate == 0.1
    assert darko_matched.depth == chimera_matched.depth == 6
    assert darko_matched.l2_leaf_reg == chimera_matched.l2_leaf_reg == 1.0
    assert darko_matched.max_bins == chimera_matched.max_bins == 128
    assert darko_matched.early_stopping is chimera_matched.early_stopping is False
    assert darko_matched.linear_leaves is chimera_matched.linear_leaves is False
    assert chimera_matched.cross_features is False
    assert chimera_matched.cat_combinations is False


def _fit_metadata(trees=1000):
    return {
        "best_iteration": trees,
        "fitted_tree_count": trees,
        "resolved_learning_rate": 0.1,
        "resolved_thread_count": 18,
    }


def _row(fold, prediction="same", r2=0.5, trees=1000):
    return {
        "fold": fold,
        "test_indices": [fold],
        "r2": r2,
        "prediction_sha256": f"{prediction}-{fold}",
        "fit_metadata": _fit_metadata(trees),
    }


def _arm(*, mean=0.5, prediction="same", guardrail=0.4, trees=1000):
    folds = [
        _row(fold, prediction=prediction, r2=mean + fold / 1000, trees=trees)
        for fold in range(10)
    ]
    scores = {
        "overlap_exposed_team_holdout": {"r2": guardrail},
        "seen_player_subset": {"r2": guardrail},
        "cold_player_subset": {"r2": guardrail},
    }
    return {
        "mean_r2": mean,
        "fold_scores": [row["r2"] for row in folds],
        "folds": folds,
        "holdout": {
            "scores": scores,
            "prediction_sha256": f"{prediction}-holdout",
            "fit_metadata": _fit_metadata(trees),
        },
    }


def test_product_quality_analysis_is_descriptive_and_guardrail_aware():
    product = {
        experiment.DARKOFIT_ARM: _arm(mean=0.526, guardrail=0.500),
        experiment.CHIMERABOOST_ARM: _arm(mean=0.527, guardrail=0.506),
    }
    result = experiment._quality_analysis(product)
    assert result["broadly_comparable"] is True
    assert result["darkofit_minus_chimeraboost_mean_r2"] == pytest.approx(-0.001)

    product[experiment.CHIMERABOOST_ARM]["holdout"]["scores"][
        "cold_player_subset"
    ]["r2"] = 0.52
    result = experiment._quality_analysis(product)
    assert result["descriptive_quality_gates"][
        "cold_player_within_descriptive_band"
    ] is False
    assert result["broadly_comparable"] is False


def test_matched_exactness_requires_predictions_guardrails_and_tree_counts():
    matched = {
        experiment.DARKOFIT_ARM: _arm(),
        experiment.CHIMERABOOST_ARM: _arm(),
    }
    assert experiment._matched_exactness(matched)["passes_exactness"] is True

    changed = deepcopy(matched)
    changed[experiment.CHIMERABOOST_ARM]["folds"][2][
        "prediction_sha256"
    ] = "changed"
    assert experiment._matched_exactness(changed)["passes_exactness"] is False

    changed = deepcopy(matched)
    changed[experiment.CHIMERABOOST_ARM]["holdout"]["fit_metadata"][
        "fitted_tree_count"
    ] = 999
    assert experiment._matched_exactness(changed)["passes_exactness"] is False


def _values(darko=10.2, chimera=10.0):
    return {
        lane: {
            metric: {
                experiment.DARKOFIT_ARM: [darko, darko * 1.01, darko * 1.02],
                experiment.CHIMERABOOST_ARM: [
                    chimera,
                    chimera * 1.01,
                    chimera * 1.02,
                ],
            }
            for metric in ("wall", "fit", "predict")
        }
        for lane in experiment.LANE_ORDER
    }


def test_timing_analysis_enforces_matched_engine_not_product_policy():
    rss = {
        lane: {
            experiment.DARKOFIT_ARM: [101, 101, 101],
            experiment.CHIMERABOOST_ARM: [100, 100, 100],
        }
        for lane in experiment.LANE_ORDER
    }
    exact = {"passes_exactness": True}
    result = experiment._timing_analysis(_values(), rss, exact)
    assert result["passes_engine_parity"] is True

    slow = _values()
    slow[experiment.MATCHED_LANE]["fit"][experiment.DARKOFIT_ARM] = [12.0] * 3
    result = experiment._timing_analysis(slow, rss, exact)
    assert result["engine_parity_gates"]["fit_engine_parity"] is False
    assert result["passes_engine_parity"] is False


def test_validate_fit_requires_threads_and_matched_tree_count():
    fitted = {"fit_metadata": _fit_metadata()}
    experiment._validate_fit(
        experiment.MATCHED_LANE, experiment.DARKOFIT_ARM, fitted
    )
    fitted["fit_metadata"]["fitted_tree_count"] = 999
    with pytest.raises(RuntimeError, match="expected 1000"):
        experiment._validate_fit(
            experiment.MATCHED_LANE, experiment.DARKOFIT_ARM, fitted
        )

    with pytest.raises(RuntimeError, match="expected fixed 1000"):
        experiment._validate_fit(
            experiment.PRODUCT_LANE, experiment.DARKOFIT_ARM, fitted
        )
    experiment._validate_fit(
        experiment.PRODUCT_LANE, experiment.CHIMERABOOST_ARM, fitted
    )


def test_parse_args_rejects_nonfrozen_resources_and_partial_worker(tmp_path):
    args = experiment.parse_args([])
    assert args.output == experiment.DEFAULT_OUTPUT
    assert args.threads == experiment.EXPECTED_THREADS
    with pytest.raises(SystemExit):
        experiment.parse_args(["--threads", "4"])
    with pytest.raises(SystemExit):
        experiment.parse_args(["--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit):
        experiment.parse_args(["--worker-arm", experiment.DARKOFIT_ARM])


def test_frozen_execution_binds_protocol_output_and_repository(monkeypatch):
    args = experiment.parse_args([])
    monkeypatch.setattr(
        experiment, "EXPECTED_REPOSITORY_MANIFEST_SHA256", "1" * 64
    )
    monkeypatch.setattr(
        experiment, "_repository_manifest_sha256", lambda: "1" * 64
    )
    experiment._validate_frozen_execution(args)

    args.output = args.output.with_name("wrong.json")
    with pytest.raises(RuntimeError, match="output path"):
        experiment._validate_frozen_execution(args)


def test_behavior_payload_omits_times_and_predictions_but_keeps_hashes():
    result = {
        "lane": experiment.MATCHED_LANE,
        "arm": experiment.DARKOFIT_ARM,
        "mean_r2": 0.5,
        "folds": [
            {
                **_row(0),
                "predictions": [1.0],
                "fit_seconds": 1.0,
                "feature_importance_sha256": "importance",
            }
        ],
        "holdout": {
            "scores": {"cold_player_subset": {"r2": 0.4}},
            "prediction_sha256": "holdout",
            "feature_importance_sha256": "importance",
            "fit_metadata": _fit_metadata(),
            "predictions": [1.0],
            "fit_seconds": 1.0,
        },
    }
    payload = experiment._behavior_payload(result)
    assert "predictions" not in payload["folds"][0]
    assert "fit_seconds" not in payload["folds"][0]
    assert payload["folds"][0]["prediction_sha256"] == "same-0"
    assert harness.behavior_fingerprint(payload)
