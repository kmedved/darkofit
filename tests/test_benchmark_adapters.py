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
    split_case,
)
from weighted_metrics import metric_bundle  # noqa: E402


class IterationsEstimator:
    def __init__(
        self,
        iterations=1,
        early_stopping_rounds=None,
        depth=6,
        num_leaves=None,
        learning_rate=None,
        thread_count=None,
        random_state=None,
        ordered_boosting=True,
        tree_mode="catboost",
        verbose_timing=False,
        min_child_samples=20,
        min_gain_to_split=0.0,
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
        num_leaves=15,
        learning_rate=0.2,
        threads=2,
        ordered_boosting=True,
        min_child_samples=7,
        min_gain_to_split=0.01,
    )
    variant = RevisionSpec("fork_lightgbm", "/repo", tree_mode="lightgbm")

    kwargs = estimator_kwargs(IterationsEstimator, cfg, variant, seed=11)

    assert kwargs["iterations"] == 17
    assert kwargs["early_stopping_rounds"] == 4
    assert kwargs["depth"] == 3
    assert kwargs["num_leaves"] == 15
    assert kwargs["learning_rate"] == 0.2
    assert kwargs["thread_count"] == 2
    assert kwargs["random_state"] == 11
    assert kwargs["ordered_boosting"] is False
    assert kwargs["tree_mode"] == "lightgbm"
    assert kwargs["verbose_timing"] is True
    assert kwargs["min_child_samples"] == 7
    assert kwargs["min_gain_to_split"] == 0.01


def test_estimator_kwargs_num_leaves_only_for_lightgbm():
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

    assert "num_leaves" not in catboost
    assert lightgbm["num_leaves"] == 15


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

    assert catboost["ordered_boosting"] is True
    assert lightgbm["ordered_boosting"] is False


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


def test_default_revision_specs_expand_expected_labels():
    specs = default_revision_specs("/up", "/fork", "/candidate")

    assert [s.label for s in specs] == [
        "upstream_default",
        "upstream_matched",
        "fork_catboost_matched",
        "fork_lightgbm_leafwise_matched",
        "candidate_catboost",
        "candidate_lightgbm_leafwise",
    ]
    assert specs[3].tree_mode == "lightgbm"
    assert specs[0].use_defaults is True


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
