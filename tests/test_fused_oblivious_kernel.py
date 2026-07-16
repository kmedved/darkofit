import numpy as np
import pytest

import darkofit.tree as tree_module
from darkofit.tree import (
    _best_split,
    _build_histograms_unit_hess_and_best_split,
    _build_histograms_unit_hess_into,
    build_oblivious_tree,
)


def _set_threads(count):
    import numba

    previous = numba.get_num_threads()
    numba.set_num_threads(min(count, numba.config.NUMBA_NUM_THREADS))
    return previous


def _split_buffers(n_features, n_leaves):
    return (
        *(np.empty((n_features, n_leaves)) for _ in range(5)),
        np.empty((n_features, n_leaves), dtype=np.int64),
    )


def _hist_buffers(n_features, n_leaves, max_bins, fill=0.0):
    gradient = np.full(
        (n_features, n_leaves, max_bins), fill, dtype=np.float64
    )
    return gradient, gradient.copy()


def _varied_bin_case(seed=11):
    rng = np.random.default_rng(seed)
    n_samples = 257
    n_bins = np.array([3, 7, 5, 9, 4], dtype=np.int64)
    X = np.column_stack(
        [rng.integers(0, count, size=n_samples) for count in n_bins]
    ).astype(np.uint8)
    X = np.asfortranarray(X)
    grad = rng.normal(size=n_samples)
    leaf = rng.choice(
        np.array([0, 1, 3, 6], dtype=np.int64), size=n_samples
    )
    return X, grad, leaf, n_bins


@pytest.mark.parametrize("l2", [0.0, 3.0])
@pytest.mark.parametrize("min_child_weight", [0.0, 1.0, 8.0])
@pytest.mark.parametrize(
    "feature_mask",
    [
        np.ones(5, dtype=np.int64),
        np.array([1, 0, 1, 0, 1], dtype=np.int64),
    ],
)
def test_fused_unit_hessian_kernel_is_exact(
    l2, min_child_weight, feature_mask
):
    previous = _set_threads(4)
    try:
        X, grad, leaf, n_bins = _varied_bin_case()
        n_features = X.shape[1]
        n_leaves = 8
        max_bins = int(n_bins.max())
        reference_g, reference_h = _hist_buffers(
            n_features, n_leaves, max_bins, fill=17.0
        )
        fused_g, fused_h = _hist_buffers(
            n_features, n_leaves, max_bins, fill=-23.0
        )
        reference_scratch = _split_buffers(n_features, n_leaves)
        fused_scratch = _split_buffers(n_features, n_leaves)

        _build_histograms_unit_hess_into(
            X, grad, leaf, n_leaves, reference_g, reference_h
        )
        expected = _best_split(
            reference_g,
            reference_h,
            n_bins,
            l2,
            feature_mask,
            min_child_weight,
            n_leaves,
            *reference_scratch,
        )
        actual = _build_histograms_unit_hess_and_best_split(
            X,
            grad,
            leaf,
            n_leaves,
            fused_g,
            fused_h,
            n_bins,
            l2,
            feature_mask,
            min_child_weight,
            *fused_scratch,
        )

        assert actual == expected
        np.testing.assert_array_equal(fused_g, reference_g)
        np.testing.assert_array_equal(fused_h, reference_h)
    finally:
        _set_threads(previous)


@pytest.mark.parametrize(
    "feature_mask",
    [
        np.ones(5, dtype=np.int64),
        np.array([1, 0, 1, 1, 0], dtype=np.int64),
    ],
)
def test_fused_builder_preserves_complete_tree_and_training_state(feature_mask):
    previous = _set_threads(4)
    try:
        X, grad, _leaf, n_bins = _varied_bin_case(seed=19)
        hess = np.ones(len(X), dtype=np.float64)
        n_features = X.shape[1]
        max_depth = 5
        max_leaves = 1 << max_depth
        max_bins = int(n_bins.max())
        common = {
            "n_bins_per_feature": n_bins,
            "max_depth": max_depth,
            "l2": 3.0,
            "lr": 0.1,
            "feature_mask": feature_mask,
            "min_child_weight": 1.0,
            "return_training_state": True,
            "X_hist_binned": np.asfortranarray(X),
            "X_route_binned": X,
            "constant_hessian": True,
            "level_histogram_subtraction": False,
        }

        reference = build_oblivious_tree(
            X,
            grad,
            hess,
            hist_buffers=_hist_buffers(
                n_features, max_leaves, max_bins
            ),
            split_buffers=_split_buffers(n_features, max_leaves),
            fused_oblivious_kernel=False,
            **common,
        )
        fused_counter = np.zeros(1, dtype=np.int64)
        fused = build_oblivious_tree(
            X,
            grad,
            hess,
            hist_buffers=_hist_buffers(
                n_features, max_leaves, max_bins
            ),
            split_buffers=_split_buffers(n_features, max_leaves),
            fused_oblivious_kernel=True,
            fused_oblivious_counter=fused_counter,
            **common,
        )

        assert int(fused_counter[0]) > 0

        for actual, expected in zip(fused[1:], reference[1:]):
            np.testing.assert_array_equal(actual, expected)
        for name in ("splits_feat", "splits_thr", "gains", "values"):
            np.testing.assert_array_equal(
                getattr(fused[0], name), getattr(reference[0], name)
            )
        np.testing.assert_array_equal(fused[0].predict(X), reference[0].predict(X))
    finally:
        _set_threads(previous)


@pytest.mark.parametrize(
    "ineligible",
    [
        {"constant_hessian": False},
        {"row_indices": np.arange(0, 257, 2, dtype=np.int64)},
        {"feature_indices": np.array([0, 2, 4], dtype=np.int64)},
        {"level_histogram_subtraction": True},
        {"random_strength": 0.1},
    ],
)
def test_ineligible_builder_lanes_do_not_call_fused_kernel(
    monkeypatch, ineligible
):
    previous = _set_threads(4)
    try:
        X, grad, _leaf, n_bins = _varied_bin_case(seed=31)
        hess = np.ones(len(X), dtype=np.float64)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("ineligible lane called the fused kernel")

        monkeypatch.setattr(
            tree_module,
            "_build_histograms_unit_hess_and_best_split",
            fail_if_called,
        )
        kwargs = {
            "constant_hessian": True,
            "level_histogram_subtraction": False,
            **ineligible,
        }
        fused_counter = np.zeros(1, dtype=np.int64)
        if "feature_indices" in kwargs:
            feature_mask = np.zeros(X.shape[1], dtype=np.int64)
            feature_mask[kwargs["feature_indices"]] = 1
            kwargs["feature_mask"] = feature_mask
        build_oblivious_tree(
            X,
            grad,
            hess,
            n_bins,
            max_depth=3,
            l2=3.0,
            lr=0.1,
            fused_oblivious_kernel=True,
            fused_oblivious_counter=fused_counter,
            **kwargs,
        )
        assert int(fused_counter[0]) == 0
    finally:
        _set_threads(previous)
