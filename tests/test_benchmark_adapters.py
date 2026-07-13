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
from analyze_tabarena_regression_remaining9 import (  # noqa: E402
    analyze_rows,
    geometric_mean_ratio,
    local_result_row,
    registered_chimera_rows,
    validate_local_rows,
)
from bench_wnba_kalman_replay import (  # noqa: E402
    paired_bootstrap_summaries,
    resolve_metric_r_floor,
    run_replay,
)
from weighted_metrics import metric_bundle  # noqa: E402


SMALL_CONFIRMATION_TASKS = {
    "alpha": (1001, 9),
    "beta": (1002, 9),
}


def _small_confirmation_rows(candidate_ratio=0.99):
    local = []
    chimera = []
    for dataset_index, (dataset, (task_id, split_count)) in enumerate(
        SMALL_CONFIRMATION_TASKS.items()
    ):
        base = 10.0 + dataset_index
        for registered_fold in range(split_count):
            repeat, fold = divmod(registered_fold, 3)
            common = {
                "dataset": dataset,
                "task_id": task_id,
                "repeat": repeat,
                "fold": fold,
                "registered_fold": registered_fold,
            }
            local.append(
                {
                    **common,
                    "config": "default",
                    "rmse": base,
                    "val_rmse": base + 1.0,
                    "train_time_s": 2.0,
                    "infer_time_s": 1.0,
                    "peak_memory_bytes": 100.0,
                }
            )
            local.append(
                {
                    **common,
                    "config": "candidate",
                    "rmse": base * candidate_ratio,
                    "val_rmse": (base + 1.0) * 0.99,
                    "train_time_s": 1.9,
                    "infer_time_s": 1.05,
                    "peak_memory_bytes": 102.0,
                }
            )
            chimera.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "fold": fold,
                    "registered_fold": registered_fold,
                    "rmse": base * 0.98,
                    "val_rmse": (base + 1.0) * 0.98,
                }
            )
    return local, chimera


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


def test_wnba_kalman_replay_warms_up_from_train_rows_without_future_leakage():
    dates = np.array(
        ["2022-01-01", "2022-01-02", "2023-01-01", "2024-01-01"],
        dtype="datetime64[D]",
    )
    game_id = np.arange(4)
    weights = np.ones(4, dtype=np.float64)
    R = np.ones(4, dtype=np.float64)
    train_mask = np.array([True, True, False, False])
    y = np.array([1.0, 1.0, 2.0, 1000.0], dtype=np.float64)
    y_future_changed = y.copy()
    y_future_changed[3] = -1000.0

    replay = run_replay(y, weights, dates, game_id, R, 0.1, train_mask)
    changed = run_replay(
        y_future_changed, weights, dates, game_id, R, 0.1, train_mask
    )

    val_idx = 2
    assert replay["pred"][val_idx] == pytest.approx(1.0)
    assert changed["pred"][val_idx] == pytest.approx(replay["pred"][val_idx])
    assert changed["pred_var"][val_idx] == pytest.approx(
        replay["pred_var"][val_idx]
    )


def test_wnba_kalman_replay_uses_metric_relative_r_floor_by_default():
    args = type("Args", (), {"r_floor": None, "r_floor_fraction": 0.25})()
    heuristic_R = np.array([0.002, 0.004, 2.0], dtype=np.float64)
    weights = np.ones(3, dtype=np.float64)
    train_mask = np.array([True, True, False])

    assert resolve_metric_r_floor(args, heuristic_R, weights, train_mask) == (
        pytest.approx(0.00075)
    )

    args.r_floor = 0.01
    assert resolve_metric_r_floor(args, heuristic_R, weights, train_mask) == (
        pytest.approx(0.01)
    )


def test_wnba_kalman_replay_bootstrap_marks_wins_and_ties():
    pairwise = {
        ("candidate", "2024"): {
            "candidate_nll": [np.zeros(12, dtype=np.float64)],
            "incumbent_nll": [np.ones(12, dtype=np.float64)],
            "candidate_se": [np.ones(12, dtype=np.float64)],
            "incumbent_se": [np.ones(12, dtype=np.float64)],
            "candidate_z2": [np.ones(12, dtype=np.float64)],
            "incumbent_z2": [np.full(12, 1.5, dtype=np.float64)],
        }
    }

    summary = paired_bootstrap_summaries(pairwise, n_boot=64, seed=0)[
        ("candidate", "2024")
    ]

    assert summary["nll"]["result"] == "win"
    assert summary["rmse"]["result"] == "tie"
    assert summary["nis_closeness"]["result"] == "win"


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


def test_remaining9_geometric_mean_ratio_is_paired_and_strict():
    assert geometric_mean_ratio([2.0, 8.0], [1.0, 4.0]) == pytest.approx(2.0)
    with pytest.raises(RuntimeError, match="different lengths"):
        geometric_mean_ratio([1.0], [1.0, 2.0])
    with pytest.raises(RuntimeError, match="positive and finite"):
        geometric_mean_ratio([0.0], [1.0])


def test_remaining9_local_result_row_identifies_only_frozen_configs():
    def payload(hyperparameters):
        config_number = 1 if not hyperparameters else 2
        suffix = f"_c{config_number}_remaining9_confirm"
        children = {
            f"S1F{seed + 1}": {
                "name": f"S1F{seed + 1}",
                "model_type": "DarkoFitModel",
                "is_valid": True,
                "can_infer": True,
                "hyperparameters_user": dict(hyperparameters),
                "hyperparameters_fit": {},
                "hyperparameters": {
                    "iterations": 1000,
                    "early_stopping": True,
                    "tree_mode": "catboost",
                    "diagnostic_warnings": "never",
                    **hyperparameters,
                    "random_state": seed,
                },
            }
            for seed in range(8)
        }
        return {
            "problem_type": "regression",
            "metric": "rmse",
            "metric_error": 2.0,
            "metric_error_val": 2.1,
            "time_train_s": 3.0,
            "time_infer_s": 0.2,
            "memory_usage": {"peak_mem_cpu": 1000},
            "framework": f"DarkoFit_c{config_number}_remaining9_confirm_BAG_L1",
            "task_metadata": {
                "name": "alpha",
                "tid": 1001,
                "repeat": 0,
                "fold": 1,
                "split_idx": 1,
            },
            "method_metadata": {
                "model_hyperparameters": {
                    **hyperparameters,
                    "ag_args": {"name_suffix": suffix},
                    "ag_args_ensemble": {
                        "model_random_seed": 0,
                        "vary_seed_across_folds": True,
                        "fold_fitting_strategy": "sequential_local",
                        "ag.max_time_limit": 3600,
                    },
                },
                "info": {
                    "model_type": "StackerEnsembleModel",
                    "is_valid": True,
                    "can_infer": True,
                    "bagged_info": {
                        "child_model_type": "DarkoFitModel",
                        "num_child_models": 8,
                        "child_model_names": [
                            f"S1F{fold}" for fold in range(1, 9)
                        ],
                        "_n_repeats": 1,
                        "_k_per_n_repeat": [8],
                        "child_hyperparameters_user": dict(hyperparameters),
                        "child_hyperparameters_fit": {},
                        "child_hyperparameters": {
                            "iterations": 1000,
                            "early_stopping": True,
                            "tree_mode": "catboost",
                            "diagnostic_warnings": "never",
                            **hyperparameters,
                            "random_state": 0,
                        },
                    },
                    "children_info": children,
                },
            },
            "experiment_metadata": {
                "experiment_cls": "OOFExperimentRunner",
                "method_cls": "AGSingleBagWrapper",
            },
        }

    default = local_result_row(
        payload({}), source="default", task_split_counts=SMALL_CONFIRMATION_TASKS
    )
    candidate = local_result_row(
        payload(
            {
                "l2_leaf_reg": 1.0,
                "max_bins": 128,
                "learning_rate": 0.1,
                "ts_permutations": 1,
            }
        ),
        source="candidate",
        task_split_counts=SMALL_CONFIRMATION_TASKS,
    )

    assert default["config"] == "default"
    assert candidate["config"] == "candidate"
    with pytest.raises(RuntimeError, match="unexpected non-AutoGluon"):
        local_result_row(
            payload({"learning_rate": 0.2}),
            source="unknown",
            task_split_counts=SMALL_CONFIRMATION_TASKS,
        )

    incomplete = payload({})
    incomplete["method_metadata"]["info"]["children_info"].pop("S1F8")
    with pytest.raises(RuntimeError, match="expected 8 fitted child models"):
        local_result_row(
            incomplete,
            source="incomplete",
            task_split_counts=SMALL_CONFIRMATION_TASKS,
        )

    renamed = payload({})
    renamed["method_metadata"]["info"]["children_info"]["renamed"] = renamed[
        "method_metadata"
    ]["info"]["children_info"].pop("S1F8")
    with pytest.raises(RuntimeError, match="unexpected child model names"):
        local_result_row(
            renamed,
            source="renamed",
            task_split_counts=SMALL_CONFIRMATION_TASKS,
        )

    bad_seed = payload({})
    bad_seed["method_metadata"]["info"]["children_info"]["S1F8"][
        "hyperparameters"
    ]["random_state"] = 6
    with pytest.raises(RuntimeError, match="child S1F8 has seed 6; expected 7"):
        local_result_row(
            bad_seed,
            source="bad-seed",
            task_split_counts=SMALL_CONFIRMATION_TASKS,
        )

    bad_ensemble = payload({})
    bad_ensemble["method_metadata"]["model_hyperparameters"][
        "ag_args_ensemble"
    ]["fold_fitting_strategy"] = "parallel_local"
    with pytest.raises(RuntimeError, match="unexpected ag_args_ensemble"):
        local_result_row(
            bad_ensemble,
            source="bad-ensemble",
            task_split_counts=SMALL_CONFIRMATION_TASKS,
        )

    bad_child_type = payload({})
    bad_child_type["method_metadata"]["info"]["children_info"]["S1F1"][
        "model_type"
    ] = "OtherModel"
    with pytest.raises(RuntimeError, match="is not a DarkoFitModel"):
        local_result_row(
            bad_child_type,
            source="bad-child-type",
            task_split_counts=SMALL_CONFIRMATION_TASKS,
        )

    bad_child_params = payload({})
    bad_child_params["method_metadata"]["info"]["children_info"]["S1F1"][
        "hyperparameters_user"
    ] = {"learning_rate": 0.2}
    with pytest.raises(RuntimeError, match="user hyperparameters do not match"):
        local_result_row(
            bad_child_params,
            source="bad-child-params",
            task_split_counts=SMALL_CONFIRMATION_TASKS,
        )

    bad_child_fit = payload({})
    bad_child_fit["method_metadata"]["info"]["children_info"]["S1F2"][
        "hyperparameters_fit"
    ] = {"iterations": 12}
    with pytest.raises(RuntimeError, match="fitted hyperparameter overrides"):
        local_result_row(
            bad_child_fit,
            source="bad-child-fit",
            task_split_counts=SMALL_CONFIRMATION_TASKS,
        )


def test_remaining9_validation_rejects_incomplete_and_imputed_panels():
    local, _ = _small_confirmation_rows()
    validate_local_rows(local, task_split_counts=SMALL_CONFIRMATION_TASKS)
    with pytest.raises(RuntimeError, match="expected 18 unique successful candidate"):
        validate_local_rows(local[:-1], task_split_counts=SMALL_CONFIRMATION_TASKS)

    registered = []
    for dataset in SMALL_CONFIRMATION_TASKS:
        for fold in range(9):
            registered.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "method": "CHIMERA (default)",
                    "metric_error": 1.0,
                    "metric_error_val": 1.0,
                    "metric": "rmse",
                    "problem_type": "regression",
                    "imputed": fold == 0 and dataset == "alpha",
                }
            )
    with pytest.raises(RuntimeError, match="imputed"):
        registered_chimera_rows(
            registered, task_split_counts=SMALL_CONFIRMATION_TASKS
        )


def test_remaining9_analysis_emits_ratios_and_passes_predeclared_gates():
    local, chimera = _small_confirmation_rows(candidate_ratio=0.99)

    tidy, summary = analyze_rows(
        local, chimera, task_split_counts=SMALL_CONFIRMATION_TASKS
    )

    assert len(tidy) == 18
    assert tidy[0]["candidate_default_rmse_log_ratio"] == pytest.approx(
        np.log(0.99)
    )
    assert summary["equal_dataset"]["candidate_default_rmse"][
        "ratio"
    ] == pytest.approx(0.99)
    assert len(summary["common_repeat_aggregates"]) == 3
    assert all(item["repeat_wins"] == 3 for item in summary["datasets"])
    assert summary["matched_chimera"]["equal_dataset"][
        "candidate_chimera_rmse"
    ]["ratio"] == pytest.approx(0.99 / 0.98)
    assert summary["counts"]["expected_child_fits"] == 16 * 18
    assert summary["gates"]["advance"] is True


def test_remaining9_analysis_fails_exact_worst_split_gate():
    local, chimera = _small_confirmation_rows(candidate_ratio=0.98)
    bad = next(
        row
        for row in local
        if row["config"] == "candidate"
        and row["dataset"] == "alpha"
        and row["repeat"] == 0
        and row["fold"] == 0
    )
    bad["rmse"] = 10.0 * 1.021

    _, summary = analyze_rows(
        local, chimera, task_split_counts=SMALL_CONFIRMATION_TASKS
    )

    assert summary["gates"]["equal_dataset_rmse_improves_at_least_0_5pct"]
    assert not summary["gates"]["no_split_regresses_more_than_2pct"]
    assert not summary["gates"]["advance"]


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
