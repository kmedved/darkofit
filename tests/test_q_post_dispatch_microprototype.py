from __future__ import annotations

import copy

import numpy as np
import numba
import pytest

from benchmarks import run_q_post_dispatch_microprototype as bench
from darkofit import DarkoRegressor
from darkofit import tree


def test_quantized_pack_is_exact_for_representable_values():
    grad = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    packed = np.empty(grad.shape, dtype=np.int64)
    bench.quantize_pack_unit_hessian(
        grad, 2.0, np.int64(8), np.uint64(41), packed
    )
    unpacked_grad = packed >> np.int64(32)
    unpacked_count = packed & np.int64(0xFFFFFFFF)
    np.testing.assert_array_equal(unpacked_grad, [-8, -4, 0, 4, 8])
    np.testing.assert_array_equal(unpacked_count, 1)


def test_packed_accumulation_does_not_bleed_between_halves():
    qgrad = np.array([-17, 4, 8, 5], dtype=np.int64)
    packed = (qgrad << np.int64(32)) + np.int64(1)
    total = packed.sum(dtype=np.int64)
    assert total >> np.int64(32) == qgrad.sum()
    assert total & np.int64(0xFFFFFFFF) == len(qgrad)
    for rows in bench.ROWS:
        qmax = bench.qmax_for_rows(rows)
        assert rows * qmax <= 2**31 - 1
        assert rows < 2**32


def _buffers(n_features, max_leaves, max_bins):
    shape = (n_features, max_leaves)
    return (
        np.empty(shape),
        np.empty(shape),
        np.empty(shape),
        np.empty(shape),
        np.empty(shape),
        np.empty(shape, dtype=np.int64),
    )


def test_packed_kernel_matches_float_split_on_representable_gradients():
    rng = np.random.default_rng(7)
    X = rng.integers(0, 8, size=(256, 4), dtype=np.uint8)
    grad = rng.integers(-8, 9, size=256).astype(np.float64)
    leaf = rng.integers(0, 4, size=256, dtype=np.int64)
    n_bins = np.full(4, 8, dtype=np.int64)
    mask = np.ones(4, dtype=np.int64)
    hg = np.zeros((4, 8, 8))
    hh = np.zeros_like(hg)
    float_scratch = _buffers(4, 8, 8)
    expected = tree._build_histograms_unit_hess_and_best_split(
        X,
        grad,
        leaf,
        4,
        hg,
        hh,
        n_bins,
        1.0,
        mask,
        1.0,
        *float_scratch,
    )
    packed = (grad.astype(np.int64) << np.int64(32)) + np.int64(1)
    packed_hist = np.zeros((4, 8, 8), dtype=np.int64)
    packed_scratch = _buffers(4, 8, 8)
    actual = bench.packed_unit_hessian_best_split(
        X,
        packed,
        1.0,
        leaf,
        4,
        packed_hist,
        n_bins,
        1.0,
        mask,
        1.0,
        *packed_scratch,
    )
    assert actual[:2] == expected[:2]
    assert actual[2] == pytest.approx(expected[2])


def test_benchmark_patch_engages_inside_real_fit_and_restores_threads():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(512, 5))
    y = X[:, 0] - 0.4 * X[:, 1] + rng.normal(0.0, 0.1, X.shape[0])
    ambient = numba.get_num_threads()
    with bench.packed_kernel_patch() as state:
        fitted = DarkoRegressor(
            iterations=4,
            depth=3,
            max_bins=16,
            learning_rate=0.1,
            l2_leaf_reg=1.0,
            min_child_samples=1,
            tree_mode="catboost",
            linear_leaves=False,
            ordered_boosting=False,
            early_stopping=False,
            use_best_model=False,
            thread_count=min(3, numba.config.NUMBA_NUM_THREADS),
            oblivious_kernel="fused",
            diagnostic_warnings="never",
            random_state=4,
        ).fit(X, y)
    assert state["quantized_trees"] == len(fitted.model_.trees_)
    assert state["packed_levels"] == sum(
        fitted_tree.depth for fitted_tree in fitted.model_.trees_
    )
    assert numba.get_num_threads() == ambient


def _synthetic_rows(candidate_ratio):
    rows = []
    for train_rows in bench.ROWS:
        for block in range(bench.REPEATS):
            for arm in bench.ARMS:
                candidate = arm == "packed_candidate"
                fit = candidate_ratio if candidate else 1.0
                rows.append({
                    "train_rows": train_rows,
                    "block": block,
                    "arm": arm,
                    "fit_seconds": fit,
                    "predict_seconds": 1.0,
                    "peak_rss_bytes": 100,
                    "holdout_rmse": 1.0,
                    "prediction_sha256": f"{train_rows}-candidate",
                    "fitted_sha256": f"{train_rows}-candidate",
                    "tree_count": bench.ITERATIONS,
                    "retained_levels": bench.ITERATIONS * bench.DEPTH,
                    "dispatch": {
                        "requested": "fused" if candidate else "auto",
                        "resolved": (
                            "fused"
                            if candidate or train_rows == 500_000
                            else "unfused"
                        ),
                    },
                    "packed": ({
                        "qmax": bench.qmax_for_rows(train_rows),
                        "packed_levels": bench.ITERATIONS * bench.DEPTH,
                        "quantized_trees": bench.ITERATIONS,
                    } if candidate else None),
                    "numba_threads_before": 14,
                    "numba_threads_after": 14,
                    "worker_stderr": "",
                })
    return rows


def test_analysis_funds_only_a_stable_material_candidate():
    funded = bench.analyze(_synthetic_rows(0.85))
    closed = bench.analyze(_synthetic_rows(0.95))
    assert funded["q1_funded"] is True
    assert funded["disposition"] == "fund_private_q1_design"
    assert closed["q1_funded"] is False
    assert closed["disposition"] == "close_q_at_microprototype"


def test_analysis_rejects_duplicate_or_missing_rows():
    rows = _synthetic_rows(0.85)
    with pytest.raises(RuntimeError, match="exact grid"):
        bench.analyze(rows[:-1])
    duplicate = copy.deepcopy(rows)
    duplicate.append(copy.deepcopy(rows[0]))
    with pytest.raises(RuntimeError, match="duplicate"):
        bench.analyze(duplicate)


def test_outputs_are_create_only(tmp_path, monkeypatch):
    raw = tmp_path / "raw.json"
    result = tmp_path / "result.md"
    raw.write_text("existing")
    args = bench.parse_args([
        "--expected-source-sha",
        "a" * 40,
        "--raw-output",
        str(raw),
        "--result-output",
        str(result),
    ])
    monkeypatch.setattr(
        bench,
        "source_state",
        lambda expected: pytest.fail("source must not be inspected"),
    )
    with pytest.raises(FileExistsError, match="create-only"):
        bench.run(args)
