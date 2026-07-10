"""Lane/layout equivalence tests for tree-builder contracts."""

from contextlib import contextmanager

import numpy as np
import pytest


@contextmanager
def _numba_threads(count):
    numba = pytest.importorskip("numba")
    original = numba.get_num_threads()
    try:
        numba.set_num_threads(count)
    except ValueError:
        pytest.skip(f"numba runtime does not allow {count} threads")
    try:
        yield
    finally:
        numba.set_num_threads(original)


def _data(seed=17, n=900, p=8, max_bins=32):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, max_bins - 1, size=(n, p), dtype=np.uint16)
    grad = (
        np.sin(X[:, 0].astype(np.float64))
        - 0.4 * X[:, 1].astype(np.float64)
        + rng.normal(0.0, 0.2, size=n)
    )
    hess = rng.uniform(0.5, 2.0, size=n)
    n_bins = np.full(p, max_bins, dtype=np.int64)
    return X, grad, hess, n_bins


def _assert_scalar_tree_equal(a, b):
    for attr in (
        "features",
        "thresholds",
        "left_child",
        "right_child",
        "leaf_index",
        "splits_feat",
        "splits_thr",
    ):
        if hasattr(a, attr):
            assert np.array_equal(getattr(a, attr), getattr(b, attr)), attr
    if hasattr(a, "node_features"):
        assert np.array_equal(a.node_features, b.node_features)
        assert np.array_equal(a.node_thresholds, b.node_thresholds)
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.gains, b.gains)
    assert a.depth == b.depth


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        ("build_oblivious_tree", {}),
        ("build_levelwise_tree", {}),
        ("build_leafwise_tree", {"max_leaves": 7, "leafwise_row_layout": "prefix"}),
        ("build_leafwise_tree", {"max_leaves": 7, "leafwise_row_layout": "segmented"}),
        ("build_hybrid_tree", {"max_leaves": 7, "leafwise_row_layout": "prefix"}),
    ],
)
@pytest.mark.parametrize("thread_count", [1, 2, 4])
def test_route_binned_c_and_f_order_are_bit_identical(
    builder_name, kwargs, thread_count
):
    tree_mod = pytest.importorskip("darkofit.tree")
    builder = getattr(tree_mod, builder_name)
    X, grad, hess, n_bins = _data()
    X_route_f = np.asfortranarray(X)

    with _numba_threads(thread_count):
        base = builder(
            X, grad, hess, n_bins, 3, 2.0, 0.1,
            return_training_state=True,
            X_route_binned=X,
            **kwargs,
        )
        routed = builder(
            X, grad, hess, n_bins, 3, 2.0, 0.1,
            return_training_state=True,
            X_route_binned=X_route_f,
            **kwargs,
        )

    _assert_scalar_tree_equal(base[0], routed[0])
    assert np.array_equal(base[1], routed[1])
    assert np.array_equal(base[2], routed[2])
    assert np.array_equal(base[3], routed[3])
    assert np.array_equal(base[0].predict(X), routed[0].predict(X))


def test_hist_binned_does_not_drive_leaf_routing_without_route_kwarg():
    import numba

    from darkofit.tree import build_oblivious_tree

    X, grad, hess, n_bins = _data(seed=19)
    bad = np.zeros_like(X)
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        base_tree, base_leaf, *_ = build_oblivious_tree(
            X, grad, hess, n_bins, 3, 2.0, 0.1,
            return_training_state=True,
        )
        hist_tree, hist_leaf, *_ = build_oblivious_tree(
            X, grad, hess, n_bins, 3, 2.0, 0.1,
            return_training_state=True,
            X_hist_binned=bad,
        )
        route_tree, route_leaf, *_ = build_oblivious_tree(
            X, grad, hess, n_bins, 3, 2.0, 0.1,
            return_training_state=True,
            X_route_binned=bad,
        )
    finally:
        numba.set_num_threads(original_threads)

    assert np.array_equal(hist_leaf, base_leaf)
    assert np.array_equal(hist_tree.predict(X), base_tree.predict(X))
    assert not np.array_equal(route_leaf, base_leaf)
    assert not np.array_equal(route_tree.predict(X), base_tree.predict(X))


@pytest.mark.parametrize("thread_count", [1, 2, 4])
def test_multiclass_route_binned_c_and_f_order_are_bit_identical(thread_count):
    from darkofit.tree import build_leafwise_multiclass_tree

    X, grad1, hess1, n_bins = _data(seed=23, n=700, p=7)
    grad = np.vstack([grad1, -0.5 * grad1, 0.25 * grad1])
    hess = np.vstack([hess1, hess1 + 0.1, hess1 + 0.2])
    X_route_f = np.asfortranarray(X)

    with _numba_threads(thread_count):
        base = build_leafwise_multiclass_tree(
            X, grad, hess, n_bins, 3, 2.0, 0.1,
            max_leaves=7,
            return_training_state=True,
            X_route_binned=X,
        )
        routed = build_leafwise_multiclass_tree(
            X, grad, hess, n_bins, 3, 2.0, 0.1,
            max_leaves=7,
            return_training_state=True,
            X_route_binned=X_route_f,
        )

    _assert_scalar_tree_equal(base[0], routed[0])
    assert np.array_equal(base[1], routed[1])
    assert np.array_equal(base[2], routed[2])
    assert np.array_equal(base[3], routed[3])


def test_leafwise_row_layout_resolver_keeps_auto_on_safe_prefix_paths():
    from darkofit.tree import _resolve_leafwise_row_layout

    feature_mask = np.ones(6, dtype=np.int64)

    def resolve(requested="auto", **overrides):
        params = dict(
            n_samples=200_000,
            n_features=6,
            row_indices=None,
            feature_indices=None,
            feature_mask=feature_mask,
            reuse_leaf_histograms=True,
            max_leaves=31,
            fast_lane_eligible=False,
        )
        params.update(overrides)
        return _resolve_leafwise_row_layout(requested, **params)

    assert resolve("auto") == "prefix"
    assert resolve("auto", fast_lane_eligible=True) == "prefix"
    assert resolve("auto", row_indices=np.arange(32, dtype=np.int64)) == "prefix"
    assert resolve("auto", reuse_leaf_histograms=False) == "prefix"

    selected = np.array([0, 2, 4], dtype=np.int64)
    selected_mask = np.zeros_like(feature_mask)
    selected_mask[selected] = 1
    assert resolve(
        "auto",
        feature_indices=selected,
        feature_mask=selected_mask,
    ) == "prefix"

    assert resolve("segmented") == "segmented"
    assert resolve("segmented", fast_lane_eligible=True) == "segmented"
    with pytest.raises(ValueError, match="leafwise_row_layout='segmented'"):
        resolve("segmented", reuse_leaf_histograms=False)
