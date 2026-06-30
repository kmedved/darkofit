"""Tests for the revision-comparison benchmark helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from benchmark_adapters import (  # noqa: E402
    FitConfig,
    RevisionSpec,
    default_revision_specs,
    estimator_kwargs,
    make_sample_weight,
    policy_suite_specs,
    split_case,
)
from bench_compare_revisions import (  # noqa: E402
    _base_row,
    _effective_sampling,
    _selection_timing_fields,
)
from weighted_metrics import metric_bundle  # noqa: E402


class IterationsEstimator:
    def __init__(
        self,
        iterations=1,
        early_stopping_rounds=None,
        depth=6,
        max_bins=128,
        num_leaves=None,
        learning_rate=None,
        thread_count=None,
        random_state=None,
        ordered_boosting=True,
        tree_mode="catboost",
        verbose_timing=False,
        min_child_samples=20,
        min_gain_to_split=0.0,
        sampling="uniform",
        top_rate=0.2,
        other_rate=0.1,
    ):
        pass


class NEstimatorsEstimator:
    def __init__(
        self,
        n_estimators=1,
        early_stopping=True,
        early_stopping_rounds=None,
        depth=6,
        learning_rate=None,
        thread_count=None,
        random_state=None,
        ordered_boosting=False,
        verbose_timing=False,
    ):
        pass


def test_estimator_kwargs_maps_iterations_api():
    cfg = FitConfig(
        iterations=17,
        patience=4,
        depth=3,
        max_bins=64,
        num_leaves=15,
        learning_rate=0.2,
        threads=2,
        ordered_boosting=True,
        min_child_samples=7,
        min_gain_to_split=0.01,
        sampling="goss",
        top_rate=0.3,
        other_rate=0.2,
    )
    variant = RevisionSpec("fork_lightgbm", "/repo", tree_mode="lightgbm")

    kwargs = estimator_kwargs(IterationsEstimator, cfg, variant, seed=11)

    assert kwargs["iterations"] == 17
    assert kwargs["early_stopping_rounds"] == 4
    assert kwargs["depth"] == 3
    assert kwargs["max_bins"] == 64
    assert kwargs["num_leaves"] == 15
    assert kwargs["learning_rate"] == 0.2
    assert kwargs["thread_count"] == 2
    assert kwargs["random_state"] == 11
    assert kwargs["ordered_boosting"] is False
    assert kwargs["tree_mode"] == "lightgbm"
    assert kwargs["verbose_timing"] is True
    assert kwargs["min_child_samples"] == 7
    assert kwargs["min_gain_to_split"] == 0.01
    assert kwargs["sampling"] == "goss"
    assert kwargs["top_rate"] == 0.3
    assert kwargs["other_rate"] == 0.2


def test_estimator_kwargs_omits_newer_params_for_old_estimators():
    cfg = FitConfig(max_bins=64, sampling="goss", top_rate=0.3, other_rate=0.2)
    variant = RevisionSpec("upstream_matched", "/repo")

    kwargs = estimator_kwargs(NEstimatorsEstimator, cfg, variant, seed=0)

    assert "max_bins" not in kwargs
    assert "sampling" not in kwargs
    assert "top_rate" not in kwargs
    assert "other_rate" not in kwargs


def test_effective_sampling_keeps_multiclass_uniform_for_goss_config():
    cfg = FitConfig(sampling="goss", top_rate=0.3, other_rate=0.2)

    assert _effective_sampling("binary", cfg) == "goss"
    assert _effective_sampling("regression", cfg) == "goss"
    assert _effective_sampling("multiclass", cfg) == "uniform"


def test_selection_timing_fields_record_auto_overhead_scope():
    core = type("Core", (), {"tree_mode_": "hybrid"})()
    model = type(
        "Model",
        (),
        {"model_": core, "tree_mode_selection_": {"selected_tree_mode": "hybrid"}},
    )()
    fields = _selection_timing_fields(model, fit_seconds=12.5, boost_seconds=3.0)

    assert fields["selected_tree_mode"] == "hybrid"
    assert fields["timing_scope"] == "selected_model"
    assert fields["selection_overhead_seconds"] == 9.5

    plain = type("Plain", (), {"model_": core})()
    plain_fields = _selection_timing_fields(
        plain, fit_seconds=4.0, boost_seconds=3.5
    )
    assert plain_fields["timing_scope"] == "fit_model"
    assert plain_fields["selection_overhead_seconds"] == ""


def test_estimator_kwargs_num_leaves_only_for_leafwise_modes():
    cfg = FitConfig(num_leaves=15)

    catboost = estimator_kwargs(
        IterationsEstimator,
        cfg,
        RevisionSpec("candidate_catboost", "/repo", tree_mode="catboost"),
        seed=0,
    )
    lightgbm = estimator_kwargs(
        IterationsEstimator,
        cfg,
        RevisionSpec("candidate_lightgbm_leafwise", "/repo", tree_mode="lightgbm"),
        seed=0,
    )
    hybrid = estimator_kwargs(
        IterationsEstimator,
        cfg,
        RevisionSpec("candidate_hybrid_leafwise", "/repo", tree_mode="hybrid"),
        seed=0,
    )
    auto = estimator_kwargs(
        IterationsEstimator,
        cfg,
        RevisionSpec("candidate_tree_auto", "/repo", tree_mode="auto"),
        seed=0,
    )

    assert "num_leaves" not in catboost
    assert lightgbm["num_leaves"] == 15
    assert hybrid["num_leaves"] == 15
    assert auto["num_leaves"] == 15


def test_estimator_kwargs_ordered_boosting_scoped_by_tree_mode():
    cfg = FitConfig(ordered_boosting=True)

    catboost = estimator_kwargs(
        IterationsEstimator,
        cfg,
        RevisionSpec("candidate_catboost", "/repo", tree_mode="catboost"),
        seed=0,
    )
    lightgbm = estimator_kwargs(
        IterationsEstimator,
        cfg,
        RevisionSpec("candidate_lightgbm_leafwise", "/repo", tree_mode="lightgbm"),
        seed=0,
    )
    hybrid = estimator_kwargs(
        IterationsEstimator,
        cfg,
        RevisionSpec("candidate_hybrid_leafwise", "/repo", tree_mode="hybrid"),
        seed=0,
    )
    auto = estimator_kwargs(
        IterationsEstimator,
        cfg,
        RevisionSpec("candidate_tree_auto", "/repo", tree_mode="auto"),
        seed=0,
    )

    assert catboost["ordered_boosting"] is True
    assert lightgbm["ordered_boosting"] is False
    assert hybrid["ordered_boosting"] is False
    assert auto["ordered_boosting"] == "auto"


def test_estimator_kwargs_maps_n_estimators_api_and_rejects_tree_mode():
    cfg = FitConfig(iterations=23, patience=5, threads=1)
    variant = RevisionSpec("upstream_matched", "/repo")

    kwargs = estimator_kwargs(NEstimatorsEstimator, cfg, variant, seed=7)

    assert kwargs["n_estimators"] == 23
    assert kwargs["early_stopping"] is True
    assert kwargs["early_stopping_rounds"] == 5
    assert kwargs["thread_count"] == 1
    assert kwargs["random_state"] == 7

    with pytest.raises(TypeError, match="tree_mode"):
        estimator_kwargs(
            NEstimatorsEstimator,
            cfg,
            RevisionSpec("bad", "/repo", tree_mode="lightgbm"),
            seed=7,
        )


def test_estimator_kwargs_default_variant_keeps_defaults():
    cfg = FitConfig(iterations=999, patience=99, threads=3)
    variant = RevisionSpec("upstream_default", "/repo", use_defaults=True)

    kwargs = estimator_kwargs(NEstimatorsEstimator, cfg, variant, seed=12)

    assert kwargs == {"thread_count": 3, "random_state": 12}


def test_estimator_kwargs_default_tree_mode_variant_keeps_native_defaults():
    cfg = FitConfig(iterations=999, patience=99, threads=3)
    variant = RevisionSpec(
        "candidate_depthwise_default",
        "/repo",
        tree_mode="depthwise",
        use_defaults=True,
    )

    kwargs = estimator_kwargs(IterationsEstimator, cfg, variant, seed=12)

    assert kwargs == {
        "thread_count": 3,
        "random_state": 12,
        "tree_mode": "depthwise",
    }
    assert "iterations" not in kwargs
    assert "depth" not in kwargs


def test_default_revision_specs_expand_expected_labels():
    specs = default_revision_specs("/up", "/fork", "/candidate")

    assert [s.label for s in specs] == [
        "upstream_default",
        "upstream_matched",
        "fork_catboost_matched",
        "fork_lightgbm_leafwise_matched",
        "candidate_catboost",
        "candidate_lightgbm_leafwise",
        "candidate_hybrid_leafwise",
        "candidate_tree_auto",
    ]
    assert specs[3].tree_mode == "lightgbm"
    assert specs[6].tree_mode == "hybrid"
    assert specs[7].tree_mode == "auto"
    assert specs[0].use_defaults is True


def test_policy_suite_specs_expands_default_regret_policies():
    specs = policy_suite_specs("/candidate")

    assert [s.label for s in specs] == [
        "candidate_default",
        "candidate_catboost_explicit",
        "candidate_lightgbm_explicit",
        "candidate_hybrid_explicit",
        "candidate_tree_auto_explicit",
        "candidate_depthwise_default",
    ]
    assert specs[0].use_defaults is True
    assert specs[1].tree_mode == "catboost"
    assert specs[2].tree_mode == "lightgbm"
    assert specs[3].tree_mode == "hybrid"
    assert specs[4].tree_mode == "auto"
    assert specs[5].tree_mode == "depthwise"
    assert specs[5].use_defaults is True


def test_base_row_reports_default_policy_training_rows_without_validation():
    spec = type("Spec", (), {"name": "demo", "task": "regression"})()
    split = {"n_train": 60, "n_val": 20, "n_test": 20, "n_features": 4}
    config = FitConfig(max_bins=128)

    default_row = _base_row(
        RevisionSpec("candidate_default", "/repo", use_defaults=True),
        spec,
        "tiny",
        seed=0,
        weight_mode="none",
        split=split,
        config=config,
    )
    explicit_row = _base_row(
        RevisionSpec("candidate_catboost_explicit", "/repo", tree_mode="catboost"),
        spec,
        "tiny",
        seed=0,
        weight_mode="none",
        split=split,
        config=config,
    )

    assert default_row["n_train"] == 80
    assert default_row["n_val"] == 0
    assert explicit_row["n_train"] == 60
    assert explicit_row["n_val"] == 20


def test_split_case_preserves_weight_modes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = np.array([0, 1] * 50)
    weights = make_sample_weight(y, "binary", "uniform")

    split = split_case(X, y, "binary", seed=3, sample_weight=weights)

    assert split["X_fit"].shape[0] == split["y_fit"].shape[0] == split["w_fit"].shape[0]
    assert split["X_val"].shape[0] == split["y_val"].shape[0] == split["w_val"].shape[0]
    assert split["X_test"].shape[0] == split["y_test"].shape[0] == split["w_test"].shape[0]
    assert split["n_train"] + split["n_val"] + split["n_test"] == 100
    assert np.all(split["w_fit"] == 1.0)


def test_stress_weights_are_mean_normalized():
    y_reg = np.linspace(-2.0, 2.0, 50)
    y_cls = np.array([0] * 80 + [1] * 20)

    w_reg = make_sample_weight(y_reg, "regression", "stress")
    w_cls = make_sample_weight(y_cls, "binary", "stress")

    assert np.isclose(w_reg.mean(), 1.0)
    assert np.isclose(w_cls.mean(), 1.0)
    assert w_reg[-1] > w_reg[0]
    assert w_cls[y_cls == 1].mean() > w_cls[y_cls == 0].mean()


def test_uniform_weight_metrics_equal_unweighted_metrics():
    y = np.array([0, 1, 1, 0])
    pred = np.array([0, 1, 0, 0])
    proba = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.6, 0.4],
        [0.7, 0.3],
    ])

    metrics = metric_bundle(
        "binary",
        y,
        pred,
        proba=proba,
        labels=np.array([0, 1]),
        sample_weight=np.ones_like(y, dtype=float),
    )

    assert metrics["primary_metric"] == "weighted_log_loss"
    assert metrics["weighted_accuracy"] == metrics["accuracy"]
    assert metrics["weighted_f1_macro"] == metrics["f1_macro"]
    assert metrics["weighted_log_loss"] == metrics["log_loss"]
    assert metrics["weighted_brier"] == metrics["brier"]
