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
    OPENML_SUITE,
    RevisionSpec,
    _frame_to_dataset,
    build_dataset,
    default_revision_specs,
    estimator_kwargs,
    make_groups,
    make_sample_weight,
    register_external_datasets,
    split_case,
)
from bench_compare_revisions import (  # noqa: E402
    _base_row,
    _path_token,
    _peak_rss_mb,
    _select_variants,
    _validation_eval_set,
)
from bench_compare_revisions import main as compare_revisions_main  # noqa: E402
from check_strict_domination import evaluate_rows  # noqa: E402
from check_strict_domination import _timing_control_limits  # noqa: E402
from bench_levelwise_tuning import main as levelwise_tuning_main  # noqa: E402
from summarize_revision_compare import aggregate  # noqa: E402
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
        n_ensembles=None,
        ensemble_n_jobs=1,
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
        n_ensembles=None,
        ensemble_n_jobs=1,
        verbose_timing=False,
    ):
        pass


class NoEnsembleEstimator:
    def __init__(self, n_estimators=1, early_stopping=True):
        pass


def test_estimator_kwargs_maps_iterations_api():
    cfg = FitConfig(
        iterations=17,
        patience=4,
        depth=3,
        learning_rate=0.2,
        n_ensembles=3,
        ensemble_n_jobs=2,
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
    assert kwargs["n_ensembles"] == 3
    assert kwargs["ensemble_n_jobs"] == 2
    assert kwargs["tree_mode"] == "lightgbm"
    assert kwargs["verbose_timing"] is False


def test_estimator_kwargs_can_enable_verbose_timing():
    cfg = FitConfig(iterations=17, verbose_timing=True)
    variant = RevisionSpec("fork_lightgbm", "/repo", tree_mode="lightgbm")

    kwargs = estimator_kwargs(IterationsEstimator, cfg, variant, seed=11)

    assert kwargs["verbose_timing"] is True


def test_validation_weight_policy_controls_candidate_eval_set():
    data = {
        "X_val": np.array([[1.0], [2.0]]),
        "y_val": np.array([0.0, 1.0]),
        "w_val": np.array([1.0, 3.0]),
    }
    candidate = RevisionSpec("candidate_catboost", "/repo", tree_mode="catboost")
    upstream = RevisionSpec("upstream_matched", "/repo")

    product = _validation_eval_set(
        data,
        candidate,
        FitConfig(validation_weight_policy="product"),
    )
    compatible = _validation_eval_set(
        data,
        candidate,
        FitConfig(validation_weight_policy="upstream-compatible"),
    )
    upstream_product = _validation_eval_set(
        data,
        upstream,
        FitConfig(validation_weight_policy="product"),
    )

    assert len(product) == 3
    assert product[0] is data["X_val"]
    assert product[1] is data["y_val"]
    assert product[2] is data["w_val"]
    assert len(compatible) == 2
    assert compatible[0] is data["X_val"]
    assert compatible[1] is data["y_val"]
    assert len(upstream_product) == 2
    assert upstream_product[0] is data["X_val"]
    assert upstream_product[1] is data["y_val"]


def test_estimator_kwargs_maps_n_estimators_api_and_rejects_tree_mode():
    cfg = FitConfig(
        iterations=23,
        patience=5,
        threads=1,
        n_ensembles=4,
        ensemble_n_jobs=2,
    )
    variant = RevisionSpec("upstream_matched", "/repo")

    kwargs = estimator_kwargs(NEstimatorsEstimator, cfg, variant, seed=7)

    assert kwargs["n_estimators"] == 23
    assert kwargs["early_stopping"] is True
    assert kwargs["early_stopping_rounds"] == 5
    assert kwargs["thread_count"] == 1
    assert kwargs["random_state"] == 7
    assert kwargs["n_ensembles"] == 4
    assert kwargs["ensemble_n_jobs"] == 2

    with pytest.raises(TypeError, match="tree_mode"):
        estimator_kwargs(
            NEstimatorsEstimator,
            cfg,
            RevisionSpec("bad", "/repo", tree_mode="lightgbm"),
            seed=7,
        )


def test_estimator_kwargs_rejects_unsupported_bagging_request():
    cfg = FitConfig(iterations=23, n_ensembles=3, ensemble_n_jobs=1)

    with pytest.raises(TypeError, match="n_ensembles"):
        estimator_kwargs(
            NoEnsembleEstimator,
            cfg,
            RevisionSpec("legacy", "/repo"),
            seed=7,
        )

    single_kwargs = estimator_kwargs(
        NoEnsembleEstimator,
        FitConfig(iterations=23, n_ensembles=1, ensemble_n_jobs=2),
        RevisionSpec("legacy", "/repo"),
        seed=7,
    )
    assert single_kwargs["n_estimators"] == 23
    assert "n_ensembles" not in single_kwargs
    assert "ensemble_n_jobs" not in single_kwargs


def test_estimator_kwargs_default_variant_keeps_defaults():
    cfg = FitConfig(
        iterations=999,
        patience=99,
        threads=3,
        n_ensembles=5,
        ensemble_n_jobs=2,
    )
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


def test_model_selection_preserves_requested_order():
    variants = [
        RevisionSpec("upstream_matched", "/upstream"),
        RevisionSpec("candidate_catboost", "/candidate", tree_mode="catboost"),
        RevisionSpec("candidate_lightgbm", "/candidate", tree_mode="lightgbm"),
    ]

    selected = _select_variants(
        variants,
        ["candidate_catboost", "upstream_matched"],
    )

    assert [variant.label for variant in selected] == [
        "candidate_catboost",
        "upstream_matched",
    ]


def test_model_selection_rejects_unknown_label():
    variants = [RevisionSpec("upstream_matched", "/upstream")]

    with pytest.raises(SystemExit, match="unknown or unavailable variants"):
        _select_variants(variants, ["candidate_catboost"])


def test_revision_summary_separates_split_and_ensemble_dimensions():
    base = {
        "status": "ok",
        "dataset": "d",
        "size": "small",
        "split_mode": "row",
        "weight_mode": "none",
        "ensemble_size": "1",
        "variant": "candidate",
        "primary_metric": "rmse",
        "primary_value": "1.0",
        "fit_seconds": "1.0",
        "predict_seconds": "0.1",
        "best_iteration": "3",
    }
    rows = []
    for split_mode in ("row", "group"):
        for ensemble_size, metric in (("1", "1.0"), ("3", "2.0")):
            rows.append({
                **base,
                "split_mode": split_mode,
                "ensemble_size": ensemble_size,
                "primary_value": metric,
            })

    summary = aggregate(rows)

    assert len(summary) == 4
    assert summary[("d", "small", "row", "none", "1", "candidate")][
        "primary_value"] == 1.0
    assert summary[("d", "small", "row", "none", "3", "candidate")][
        "primary_value"] == 2.0
    assert summary[("d", "small", "group", "none", "1", "candidate")][
        "primary_value"] == 1.0


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


def test_timing_control_can_calibrate_strict_timing_only():
    base = {
        "status": "ok",
        "dataset": "numeric_binary",
        "size": "medium",
        "split_mode": "row",
        "weight_mode": "stress",
        "ensemble_size": "1",
        "seed": "0",
        "primary_metric": "weighted_log_loss",
        "primary_value": "1.0",
        "validation_weight_policy": "upstream-compatible",
    }
    rows = [
        {**base, "variant": "upstream_matched", "fit_seconds": "1.0"},
        {**base, "variant": "candidate_catboost", "fit_seconds": "1.08"},
    ]
    control_rows = [
        {**base, "variant": "upstream_matched", "fit_seconds": "1.0"},
        {**base, "variant": "candidate_matched", "fit_seconds": "1.10"},
    ]

    strict = evaluate_rows(rows, mode="upstream-compatible")
    limits, aggregate_limit = _timing_control_limits(
        control_rows,
        baseline="upstream_matched",
        candidate="candidate_matched",
    )
    calibrated = evaluate_rows(
        rows,
        mode="upstream-compatible",
        timing_control_limits=limits,
        aggregate_timing_control_limit=aggregate_limit,
    )

    assert any(f["kind"] == "timing_regression" for f in strict["failures"])
    assert calibrated["passed"] is True
    assert calibrated["timing_control_rows"] == 1


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


def test_external_dataset_registration_is_lazy():
    register_external_datasets([
        "oml:credit-g",
        "gr:clf_num/credit",
    ])

    assert "credit-g" in OPENML_SUITE
    assert DATASETS["oml:credit-g"].task == "binary"
    assert DATASETS["gr:clf_num/credit"].task == "binary"

    with pytest.raises(KeyError, match="OpenML"):
        register_external_datasets(["oml:not-a-real-dataset"])
    with pytest.raises(KeyError, match="Grinsztajn"):
        register_external_datasets(["gr:clf_num/not-a-real-dataset"])


def test_frame_to_dataset_auto_detects_categoricals_and_missing_values():
    pd = pytest.importorskip("pandas")
    X_df = pd.DataFrame({
        "city": pd.Series(["a", None, "b"], dtype="object"),
        "score": [1.0, 2.5, 3.0],
    })
    y = pd.Series(["yes", "no", "yes"])

    X, y_codes, cat_features = _frame_to_dataset(X_df, y, "auto", "binary")

    assert cat_features == [0]
    assert X.dtype == object
    assert X[1, 0] == "__nan__"
    assert X[0, 1] == 1.0
    assert set(y_codes) == {0, 1}


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
        validation_weight_policy="product",
        ensemble_size=3,
        split=split,
    )
    matched_row = _base_row(
        RevisionSpec("upstream_matched", "/repo"),
        spec,
        "tiny",
        seed=0,
        split_mode="row",
        weight_mode="none",
        validation_weight_policy="product",
        ensemble_size=3,
        split=split,
    )

    assert default_row["loss"] == "default"
    assert default_row["alpha"] == ""
    assert default_row["ensemble_size"] == "default"
    assert matched_row["loss"] == "Quantile"
    assert matched_row["alpha"] == 0.9
    assert matched_row["ensemble_size"] == 3
    assert matched_row["validation_weight_policy"] == "product"


def _strict_row(
    variant,
    *,
    primary=1.0,
    fit=1.0,
    weight_mode="none",
    status="ok",
    policy="upstream-compatible",
):
    return {
        "status": status,
        "error": "" if status == "ok" else "boom",
        "variant": variant,
        "dataset": "d",
        "size": "medium",
        "split_mode": "row",
        "weight_mode": weight_mode,
        "validation_weight_policy": policy,
        "ensemble_size": "1",
        "seed": "0",
        "primary_metric": (
            "weighted_log_loss" if weight_mode != "none" else "log_loss"
        ),
        "primary_value": str(primary),
        "fit_seconds": str(fit),
        "fit_repeat_seconds": str(fit),
    }


def test_strict_domination_checker_passes_ties_within_tolerance():
    rows = [
        _strict_row("upstream_matched", primary=1.0, fit=1.0),
        _strict_row("candidate_catboost", primary=1.0, fit=1.0),
    ]

    report = evaluate_rows(rows)

    assert report["passed"] is True
    assert report["n_compared"] == 1


def test_strict_domination_checker_fails_timing_regression():
    rows = [
        _strict_row("upstream_matched", primary=1.0, fit=1.0),
        _strict_row("candidate_catboost", primary=1.0, fit=1.10),
    ]

    report = evaluate_rows(rows)

    assert report["passed"] is False
    assert {f["kind"] for f in report["failures"]} == {
        "timing_regression",
        "aggregate_timing_regression",
    }


def test_strict_domination_checker_can_use_repeat_median():
    rows = [
        {
            **_strict_row("upstream_matched", primary=1.0, fit=1.0),
            "fit_repeat_seconds": "1.0;2.0;2.0",
        },
        {
            **_strict_row("candidate_catboost", primary=1.0, fit=0.9),
            "fit_repeat_seconds": "0.9;3.0;3.0",
        },
    ]

    min_report = evaluate_rows(rows)
    median_report = evaluate_rows(rows, fit_time_stat="median")

    assert min_report["passed"] is True
    assert median_report["passed"] is False
    assert any(
        f["kind"] == "timing_regression"
        for f in median_report["failures"]
    )


def test_strict_domination_checker_fails_quality_regression():
    rows = [
        _strict_row("upstream_matched", primary=1.0, fit=1.0),
        _strict_row("candidate_catboost", primary=1.01, fit=1.0),
    ]

    report = evaluate_rows(rows)

    assert report["passed"] is False
    assert any(f["kind"] == "quality_regression" for f in report["failures"])


def test_strict_domination_checker_fails_semantic_non_equivalence():
    rows = [
        _strict_row(
            "upstream_matched",
            primary=1.0,
            fit=1.0,
            weight_mode="stress",
        ),
        _strict_row(
            "candidate_catboost",
            primary=1.0,
            fit=1.0,
            weight_mode="stress",
            policy="product",
        ),
    ]

    report = evaluate_rows(rows, mode="upstream-compatible")

    assert report["passed"] is False
    assert {f["kind"] for f in report["failures"]} == {
        "semantic_non_equivalence"
    }


def test_benchmark_clis_reject_zero_ensemble_jobs(tmp_path):
    with pytest.raises(SystemExit, match="ensemble-n-jobs"):
        compare_revisions_main([
            "--candidate",
            ".",
            "--ensemble-n-jobs",
            "0",
            "--csv",
            str(tmp_path / "compare.csv"),
        ])

    with pytest.raises(SystemExit, match="ensemble-n-jobs"):
        levelwise_tuning_main([
            "--ensemble-n-jobs",
            "0",
            "--csv",
            str(tmp_path / "tuning.csv"),
        ])


def test_revision_payload_path_token_handles_external_dataset_names():
    token = _path_token(
        "payload",
        "candidate_catboost",
        "gr:clf_num/credit",
        "tiny",
        0,
        "row",
        "none",
        "ens1",
    )

    assert "/" not in token
    assert ":" not in token
    assert "gr_clf_num_credit" in token


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
