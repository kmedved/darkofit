"""Tests for the one-off LightGBM comparison harness."""

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from bench_vs_lightgbm import (  # noqa: E402
    _resolve_benchmark_capacity,
    _resolve_default_depth,
    parse_args,
)


def test_default_depth_matches_tree_mode():
    lightgbm_args = _resolve_default_depth(parse_args(["--tree-mode", "lightgbm"]))
    catboost_args = _resolve_default_depth(parse_args(["--tree-mode", "catboost"]))
    depthwise_args = _resolve_default_depth(parse_args(["--tree-mode", "depthwise"]))

    assert lightgbm_args.depth == -1
    assert catboost_args.depth == 6
    assert depthwise_args.depth is None


def test_explicit_depth_is_preserved_for_lightgbm_mode():
    args = _resolve_default_depth(
        parse_args(["--tree-mode", "lightgbm", "--depth", "6"])
    )

    assert args.depth == 6


def test_chimera_max_bins_arg_is_preserved():
    args = _resolve_default_depth(parse_args(["--chimera-max-bins", "64"]))

    assert args.chimera_max_bins == 64


def test_chimera_l2_leaf_reg_arg_is_preserved():
    args = _resolve_default_depth(parse_args(["--chimera-l2-leaf-reg", "0.5"]))

    assert args.chimera_l2_leaf_reg == 0.5


def test_chimera_row_and_column_sampling_args_are_preserved():
    args = _resolve_default_depth(
        parse_args(["--chimera-subsample", "0.8", "--chimera-colsample", "0.7"])
    )

    assert args.chimera_subsample == 0.8
    assert args.chimera_colsample == 0.7


def test_chimera_multiclass_tree_strategy_arg_is_preserved():
    args = _resolve_benchmark_capacity(
        _resolve_default_depth(
            parse_args(["--chimera-multiclass-tree-strategy", "shared_vector"])
        )
    )

    assert args.chimera_multiclass_tree_strategy == "shared_vector"


def test_lightgbm_mode_matches_leaf_capacity_by_default():
    args = _resolve_benchmark_capacity(
        _resolve_default_depth(parse_args(["--tree-mode", "lightgbm"]))
    )

    assert args.lightgbm_num_leaves == 64
    assert args.chimera_num_leaves == 64
    assert args.chimera_effective_num_leaves == 64


def test_explicit_chimera_num_leaves_is_preserved():
    args = _resolve_benchmark_capacity(
        _resolve_default_depth(
            parse_args(
                [
                    "--tree-mode",
                    "lightgbm",
                    "--lightgbm-num-leaves",
                    "64",
                    "--chimera-num-leaves",
                    "127",
                ]
            )
        )
    )

    assert args.chimera_num_leaves == 127
    assert args.chimera_effective_num_leaves == 127


def test_leaf_matching_can_be_disabled_for_native_default_probe():
    args = _resolve_benchmark_capacity(
        _resolve_default_depth(
            parse_args(["--tree-mode", "lightgbm", "--no-match-lightgbm-leaves"])
        )
    )

    assert args.chimera_num_leaves is None
    assert args.chimera_effective_num_leaves == 31
