"""Tests for the one-off LightGBM comparison harness."""

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from bench_vs_lightgbm import _resolve_default_depth, parse_args  # noqa: E402


def test_default_depth_matches_tree_mode():
    lightgbm_args = _resolve_default_depth(parse_args(["--tree-mode", "lightgbm"]))
    catboost_args = _resolve_default_depth(parse_args(["--tree-mode", "catboost"]))

    assert lightgbm_args.depth == -1
    assert catboost_args.depth == 6


def test_explicit_depth_is_preserved_for_lightgbm_mode():
    args = _resolve_default_depth(
        parse_args(["--tree-mode", "lightgbm", "--depth", "6"])
    )

    assert args.depth == 6


def test_chimera_max_bins_arg_is_preserved():
    args = _resolve_default_depth(parse_args(["--chimera-max-bins", "64"]))

    assert args.chimera_max_bins == 64
