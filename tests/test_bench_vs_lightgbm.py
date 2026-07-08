"""Tests for the one-off LightGBM comparison harness."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from bench_vs_lightgbm import (  # noqa: E402
    Result,
    _chimera_model_kwargs,
    _encode_lightgbm_fit,
    _encode_lightgbm_test,
    _lightgbm_model_kwargs,
    _normalize_threads,
    _result_from_prediction,
    _resolve_benchmark_capacity,
    _resolve_default_depth,
    _resolve_profile,
    parse_args,
)


def _resolved_args(argv):
    return _resolve_benchmark_capacity(
        _resolve_default_depth(
            _normalize_threads(_resolve_profile(parse_args(argv)))
        )
    )


def test_default_depth_matches_tree_mode():
    lightgbm_args = _resolve_default_depth(parse_args(["--tree-mode", "lightgbm"]))
    hybrid_args = _resolve_default_depth(parse_args(["--tree-mode", "hybrid"]))
    auto_args = _resolve_default_depth(parse_args(["--tree-mode", "auto"]))
    catboost_args = _resolve_default_depth(parse_args(["--tree-mode", "catboost"]))
    depthwise_args = _resolve_default_depth(parse_args(["--tree-mode", "depthwise"]))

    assert lightgbm_args.depth == -1
    assert hybrid_args.depth == -1
    assert auto_args.depth is None
    assert catboost_args.depth == 6
    assert depthwise_args.depth is None


def test_explicit_depth_is_preserved_for_lightgbm_mode():
    args = _resolve_default_depth(
        parse_args(["--tree-mode", "lightgbm", "--depth", "6"])
    )

    assert args.depth == 6


def test_chimera_max_bins_arg_is_preserved():
    args = _resolved_args(["--chimera-max-bins", "64"])

    assert args.chimera_max_bins == 64


def test_chimera_l2_leaf_reg_arg_is_preserved():
    args = _resolved_args(["--chimera-l2-leaf-reg", "0.5"])

    assert args.chimera_l2_leaf_reg == 0.5


def test_chimera_row_and_column_sampling_args_are_preserved():
    args = _resolved_args(["--chimera-subsample", "0.8", "--chimera-colsample", "0.7"])

    assert args.chimera_subsample == 0.8
    assert args.chimera_colsample == 0.7


def test_chimera_multiclass_tree_strategy_arg_is_preserved():
    args = _resolved_args(["--chimera-multiclass-tree-strategy", "shared_vector"])

    assert args.chimera_multiclass_tree_strategy == "shared_vector"


def test_lightgbm_train_and_test_encoding_are_separate():
    X_fit = np.array([["a", 1.0], ["b", 2.0]], dtype=object)
    X_val = np.array([["b", 3.0]], dtype=object)
    X_test = np.array([["missing", 4.0]], dtype=object)

    X_fit_lgb, X_val_lgb, cat_features, encoder = _encode_lightgbm_fit(
        X_fit, X_val, [0]
    )
    X_test_lgb = _encode_lightgbm_test(X_test, cat_features, encoder)

    assert X_fit_lgb.shape == X_fit.shape
    assert X_val_lgb.shape == X_val.shape
    assert X_test_lgb.shape == X_test.shape
    assert X_test_lgb[0, 0] == -1.0


def test_lightgbm_mode_matches_leaf_capacity_by_default():
    args = _resolved_args(["--tree-mode", "lightgbm"])

    assert args.lightgbm_num_leaves == 64
    assert args.chimera_num_leaves == 64
    assert args.chimera_effective_num_leaves == 64


def test_hybrid_mode_matches_leaf_capacity_by_default():
    args = _resolved_args(["--tree-mode", "hybrid"])

    assert args.lightgbm_num_leaves == 64
    assert args.chimera_num_leaves == 64
    assert args.chimera_effective_num_leaves == 64


def test_auto_mode_passes_leaf_capacity_for_leafwise_candidates():
    args = _resolved_args(["--tree-mode", "auto"])

    assert args.lightgbm_num_leaves == 64
    assert args.chimera_num_leaves == 64
    assert args.chimera_effective_num_leaves is None


def test_result_records_fitted_tree_mode_and_resolved_auto_capacity():
    spec = SimpleNamespace(name="synthetic", task="regression")
    args = _resolved_args(["--tree-mode", "auto", "--threads", "0"])
    core = SimpleNamespace(
        tree_mode_="hybrid",
        auto_params_={"tree": {"max_leaves": 31}},
        sampling_="uniform",
        top_rate=None,
        other_rate=None,
    )
    model = SimpleNamespace(model_=core, best_iteration_=7)

    result = _result_from_prediction(
        spec=spec,
        size_name="small",
        seed=0,
        model_name="ChimeraBoost",
        model=model,
        y_test=np.array([1.0, 2.0]),
        pred=np.array([1.0, 2.0]),
        proba=None,
        fit_seconds=0.1,
        predict_seconds=0.01,
        n_train=10,
        n_test=2,
        n_features=3,
        chimera_effective_num_leaves=None,
        lightgbm_num_leaves=64,
        args=args,
    )

    assert result.chimera_fitted_tree_mode == "hybrid"
    assert result.chimera_resolved_num_leaves == 31
    assert result.lightgbm_num_leaves == 64
    assert result.profile == "matched"
    assert result.chimera_requested_tree_mode == "auto"
    assert result.benchmark_threads is None
    assert result.chimera_max_bins == 254
    assert result.lightgbm_max_bin == 255
    assert result.chimera_learning_rate == 0.1
    assert result.lightgbm_learning_rate == 0.1


def test_csv_schema_includes_resolved_profile_columns():
    fields = list(Result.__dataclass_fields__)

    for field in [
        "profile",
        "chimera_requested_tree_mode",
        "benchmark_threads",
        "chimera_max_bins",
        "lightgbm_max_bin",
        "chimera_learning_rate",
        "lightgbm_learning_rate",
        "chimera_l2_leaf_reg",
        "lightgbm_lambda_l2",
        "chimera_min_child_samples",
        "lightgbm_min_child_samples",
        "chimera_min_child_weight",
        "lightgbm_min_sum_hessian_in_leaf",
        "match_lightgbm_leaves",
    ]:
        assert field in fields


def test_explicit_chimera_num_leaves_is_preserved():
    args = _resolved_args(
        [
            "--tree-mode",
            "lightgbm",
            "--lightgbm-num-leaves",
            "64",
            "--chimera-num-leaves",
            "127",
        ]
    )

    assert args.chimera_num_leaves == 127
    assert args.chimera_effective_num_leaves == 127


def test_leaf_matching_can_be_disabled_for_native_default_probe():
    args = _resolved_args(["--tree-mode", "lightgbm", "--no-match-lightgbm-leaves"])

    assert args.chimera_num_leaves is None
    assert args.chimera_effective_num_leaves == 31


def test_matched_profile_resolves_auditable_estimator_kwargs():
    spec = SimpleNamespace(task="regression")
    args = _resolved_args(["--tree-mode", "lightgbm"])

    chimera_kwargs = _chimera_model_kwargs(spec, args, seed=7)
    lightgbm_kwargs = _lightgbm_model_kwargs(args, objective=None, seed=7)

    assert args.profile == "matched"
    assert chimera_kwargs["learning_rate"] == 0.1
    assert lightgbm_kwargs["learning_rate"] == 0.1
    assert chimera_kwargs["max_bins"] == 254
    assert lightgbm_kwargs["max_bin"] == 255
    assert chimera_kwargs["num_leaves"] == 64
    assert lightgbm_kwargs["num_leaves"] == 64
    assert chimera_kwargs["l2_leaf_reg"] == 3.0
    assert lightgbm_kwargs["reg_lambda"] == 3.0
    assert chimera_kwargs["min_child_weight"] == 1.0
    assert lightgbm_kwargs["min_sum_hessian_in_leaf"] == 1.0


def test_native_profile_uses_native_defaults_and_no_matched_leaf_mutation():
    spec = SimpleNamespace(task="regression")
    args = _resolved_args(["--profile", "native", "--tree-mode", "lightgbm"])

    chimera_kwargs = _chimera_model_kwargs(spec, args, seed=7)
    lightgbm_kwargs = _lightgbm_model_kwargs(args, objective=None, seed=7)

    assert args.profile == "native"
    assert args.match_lightgbm_leaves is False
    assert args.lightgbm_num_leaves == 31
    assert args.chimera_num_leaves is None
    assert args.chimera_effective_num_leaves == 31
    assert chimera_kwargs["learning_rate"] is None
    assert lightgbm_kwargs["learning_rate"] == 0.1
    assert chimera_kwargs["max_bins"] == 254
    assert lightgbm_kwargs["max_bin"] == 255
    assert lightgbm_kwargs["num_leaves"] == 31
    assert chimera_kwargs["l2_leaf_reg"] == 3.0
    assert lightgbm_kwargs["reg_lambda"] == 0.0
    assert lightgbm_kwargs["min_sum_hessian_in_leaf"] == 1e-3


def test_explicit_cli_knobs_override_profile_defaults():
    args = _resolved_args(
        [
            "--profile",
            "native",
            "--tree-mode",
            "lightgbm",
            "--match-lightgbm-leaves",
            "--learning-rate",
            "0.2",
            "--chimera-max-bins",
            "64",
            "--lightgbm-max-bin",
            "64",
            "--lightgbm-num-leaves",
            "47",
            "--lightgbm-lambda-l2",
            "2.5",
        ]
    )

    assert args.match_lightgbm_leaves is True
    assert args.chimera_num_leaves == 47
    assert args.learning_rate == 0.2
    assert args.chimera_max_bins == 64
    assert args.lightgbm_max_bin == 64
    assert args.lightgbm_num_leaves == 47
    assert args.lightgbm_lambda_l2 == 2.5


def test_threads_zero_normalizes_to_all_libraries_default():
    args = _resolved_args(["--threads", "0"])

    assert args.threads is None
    assert _lightgbm_model_kwargs(args, objective=None, seed=0)["n_jobs"] == -1
