"""Tests for the revision-comparison benchmark helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from benchmark_adapters import (  # noqa: E402
    DATASETS,
    DatasetSpec,
    FitConfig,
    RevisionSpec,
    build_dataset,
    default_revision_specs,
    estimator_kwargs,
    make_groups,
    make_sample_weight,
    split_case,
)
from bench_compare_revisions import _base_row, _peak_rss_mb  # noqa: E402
from weighted_metrics import metric_bundle  # noqa: E402


class IterationsEstimator:
    def __init__(
        self,
        iterations=1,
        early_stopping_rounds=None,
        depth=6,
        learning_rate=None,
        max_bins_ts=None,
        weighted_target_stats=False,
        thread_count=None,
        random_state=None,
        ordered_boosting=True,
        tree_mode="catboost",
        verbose_timing=False,
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
        max_bins_ts=None,
        weighted_target_stats=False,
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
        learning_rate=0.2,
        max_bins_ts=32,
        weighted_target_stats=True,
        threads=2,
        ordered_boosting=True,
    )
    variant = RevisionSpec("fork_lightgbm", "/repo", tree_mode="lightgbm")

    kwargs = estimator_kwargs(IterationsEstimator, cfg, variant, seed=11)

    assert kwargs["iterations"] == 17
    assert kwargs["early_stopping_rounds"] == 4
    assert kwargs["depth"] == 3
    assert kwargs["learning_rate"] == 0.2
    assert kwargs["max_bins_ts"] == 32
    assert kwargs["weighted_target_stats"] is True
    assert kwargs["thread_count"] == 2
    assert kwargs["random_state"] == 11
    assert kwargs["ordered_boosting"] is True
    assert kwargs["tree_mode"] == "lightgbm"
    assert kwargs["verbose_timing"] is True


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


def _write_revision(tmp_path, name, supports_tree_mode):
    root = tmp_path / name
    pkg = root / "chimeraboost"
    pkg.mkdir(parents=True)
    text = "def __init__(self, tree_mode=None): pass\n" if supports_tree_mode else "pass\n"
    (pkg / "sklearn_api.py").write_text(text)
    return str(root)


def test_default_revision_specs_expand_expected_labels(tmp_path):
    upstream = _write_revision(tmp_path, "upstream", supports_tree_mode=False)
    fork = _write_revision(tmp_path, "fork", supports_tree_mode=True)
    candidate = _write_revision(tmp_path, "candidate", supports_tree_mode=True)

    specs = default_revision_specs(upstream, fork, candidate)

    assert [s.label for s in specs] == [
        "upstream_default",
        "upstream_matched",
        "fork_catboost_matched",
        "fork_lightgbm_matched",
        "candidate_catboost",
        "candidate_lightgbm",
    ]
    assert specs[3].tree_mode == "lightgbm"
    assert specs[0].use_defaults is True


def test_default_revision_specs_handles_legacy_fork_without_tree_mode(tmp_path):
    upstream = _write_revision(tmp_path, "upstream", supports_tree_mode=False)
    fork = _write_revision(tmp_path, "fork", supports_tree_mode=False)
    candidate = _write_revision(tmp_path, "candidate", supports_tree_mode=True)

    specs = default_revision_specs(upstream, fork, candidate)

    assert [s.label for s in specs] == [
        "upstream_default",
        "upstream_matched",
        "fork_matched",
        "candidate_catboost",
        "candidate_lightgbm",
    ]
    assert specs[2].tree_mode is None


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


def test_quantile_dataset_specs_carry_loss_and_use_regression_splits():
    spec, X, y, cat_features = build_dataset("quantile_reg_90", "tiny", seed=0)
    weights = make_sample_weight(y, spec.task, "stress")
    split = split_case(X, y, spec.task, seed=0, sample_weight=weights)

    assert DATASETS["quantile_reg_90"].task == "quantile"
    assert spec.loss == "Quantile"
    assert spec.alpha == 0.9
    assert cat_features is None
    assert np.isclose(weights.mean(), 1.0)
    assert split["X_fit"].shape[0] + split["X_val"].shape[0] + split["X_test"].shape[0] == len(y)


def test_default_variant_row_does_not_claim_quantile_loss():
    spec = DatasetSpec(
        "quantile_reg_test",
        "quantile",
        builder=lambda n, rng: None,
        loss="Quantile",
        alpha=0.9,
    )
    split = {"n_train": 10, "n_val": 3, "n_test": 4, "n_features": 2}

    default_row = _base_row(
        RevisionSpec("upstream_default", "/repo", use_defaults=True),
        spec,
        "tiny",
        seed=0,
        split_mode="row",
        weight_mode="none",
        split=split,
    )
    matched_row = _base_row(
        RevisionSpec("upstream_matched", "/repo"),
        spec,
        "tiny",
        seed=0,
        split_mode="row",
        weight_mode="none",
        split=split,
    )

    assert default_row["loss"] == "default"
    assert default_row["alpha"] == ""
    assert matched_row["loss"] == "Quantile"
    assert matched_row["alpha"] == 0.9


def test_peak_rss_helper_reports_positive_memory():
    assert _peak_rss_mb() > 0.0


def test_group_split_keeps_groups_disjoint():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = rng.integers(0, 2, size=300)
    groups = make_groups(len(y), seed=1)
    split = split_case(X, y, "binary", seed=1, groups=groups)

    train_groups = set(split["groups_fit"])
    val_groups = set(split["groups_val"])
    test_groups = set(split["groups_test"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert split["n_groups_train"] == len(train_groups)
    assert split["n_groups_val"] == len(val_groups)
    assert split["n_groups_test"] == len(test_groups)

    weighted = split_case(
        X,
        y,
        "binary",
        seed=1,
        sample_weight=[1.0] * len(y),
        groups=groups,
    )
    assert weighted["w_fit"].shape[0] == weighted["X_fit"].shape[0]

    with pytest.raises(ValueError, match="groups"):
        split_case(X, y, "binary", seed=1, groups=groups[:-1])


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


def test_quantile_metrics_report_pinball_and_coverage():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    pred = np.array([0.5, 0.5, 2.5, 2.5])
    weights = np.array([3.0, 1.0, 3.0, 1.0])

    metrics = metric_bundle(
        "quantile",
        y,
        pred,
        sample_weight=weights,
        alpha=0.9,
    )

    assert metrics["primary_metric"] == "weighted_pinball"
    assert metrics["pinball"] == pytest.approx(0.25)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["weighted_pinball"] != metrics["pinball"]
    assert metrics["weighted_coverage"] == pytest.approx(0.75)
