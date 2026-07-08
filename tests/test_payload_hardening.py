"""Regression tests for review blockers: corrupt-archive tree semantics,
class-minor buffer contracts, exact-solver boundaries, and prep round-trip."""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor
from chimeraboost.booster import (
    _array_content_signature,
    _exact_mvs_probabilities,
    _exact_weighted_goss_probabilities,
)
from chimeraboost.tree import build_leafwise_multiclass_tree


def _make_regression(n=400, f=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = X[:, 0] * 2.0 + np.sin(X[:, 1]) + rng.normal(scale=0.2, size=n)
    return X, y


def _mutated_archive(src, dst, **updates):
    with np.load(src, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    for key, mutate in updates.items():
        arrays[key] = mutate(np.array(arrays[key]))
    np.savez_compressed(dst, **arrays)
    return dst


def _fit_and_save(tmp_path, tree_mode, name):
    X, y = _make_regression()
    model = ChimeraBoostRegressor(
        iterations=12, tree_mode=tree_mode, learning_rate=0.3,
        random_state=0, early_stopping=False,
    )
    model.fit(X, y)
    path = str(tmp_path / name)
    model.save_model(path)
    return X, model, path


def _assert_load_rejected(path, match):
    with pytest.raises(ValueError, match=match):
        ChimeraBoostRegressor.load_model(path)


# ---------------------------------------------------------------------------
# Blocker: nonoblivious tree payloads must be structurally valid.
# ---------------------------------------------------------------------------

def _first_internal_and_leaf(arrays):
    left = arrays["trees__left_child_flat"]
    internal = int(np.flatnonzero(left >= 0)[0])
    leaf = int(np.flatnonzero(left < 0)[0])
    return internal, leaf


def test_load_rejects_nonoblivious_child_self_loop(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")

    def corrupt(arr):
        arr[0] = 0  # root's left child points back at the root
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "self_loop.npz"),
        trees__left_child_flat=corrupt,
    )
    _assert_load_rejected(bad, "invalid ChimeraBoost model")


def test_load_rejects_nonoblivious_internal_node_with_one_child(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")
    with np.load(path, allow_pickle=False) as data:
        internal, _ = _first_internal_and_leaf(
            {k: data[k] for k in data.files}
        )

    def corrupt(arr):
        arr[internal] = -1  # internal node loses its right child
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "one_child.npz"),
        trees__right_child_flat=corrupt,
    )
    _assert_load_rejected(bad, "invalid ChimeraBoost model")


def test_load_rejects_nonoblivious_terminal_without_leaf_index(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")
    with np.load(path, allow_pickle=False) as data:
        _, leaf = _first_internal_and_leaf({k: data[k] for k in data.files})

    def corrupt(arr):
        arr[leaf] = -1  # terminal node would predict values[-1]
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "leafless.npz"),
        trees__leaf_index_flat=corrupt,
    )
    _assert_load_rejected(bad, "invalid ChimeraBoost model")


def test_load_rejects_nonoblivious_out_of_range_feature(tmp_path):
    X, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")
    with np.load(path, allow_pickle=False) as data:
        internal, _ = _first_internal_and_leaf(
            {k: data[k] for k in data.files}
        )

    def corrupt(arr):
        arr[internal] = X.shape[1] + 7  # beyond the binned feature space
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "oob_feature.npz"),
        trees__features_flat=corrupt,
    )
    _assert_load_rejected(bad, "out of range")


def test_load_rejects_nonoblivious_out_of_range_threshold(tmp_path):
    _, _, path = _fit_and_save(tmp_path, "lightgbm", "m.npz")
    with np.load(path, allow_pickle=False) as data:
        internal, _ = _first_internal_and_leaf(
            {k: data[k] for k in data.files}
        )

    def corrupt(arr):
        arr[internal] = 60000  # far beyond any per-feature bin count
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "oob_threshold.npz"),
        trees__thresholds_flat=corrupt,
    )
    _assert_load_rejected(bad, "out of range")


# ---------------------------------------------------------------------------
# Blocker: oblivious and levelwise split payloads must stay in bounds.
# ---------------------------------------------------------------------------

def test_load_rejects_oblivious_out_of_range_payloads(tmp_path):
    X, _, path = _fit_and_save(tmp_path, "catboost", "m.npz")

    def bad_feature(arr):
        arr[0] = X.shape[1] + 5
        return arr

    def bad_threshold(arr):
        arr[0] = 60000
        return arr

    _assert_load_rejected(
        _mutated_archive(
            path, str(tmp_path / "obl_feat.npz"),
            trees__feats_flat=bad_feature,
        ),
        "out of range",
    )
    _assert_load_rejected(
        _mutated_archive(
            path, str(tmp_path / "obl_thr.npz"),
            trees__thrs_flat=bad_threshold,
        ),
        "out of range",
    )


def test_load_rejects_levelwise_out_of_range_node_feature(tmp_path):
    X, _, path = _fit_and_save(tmp_path, "depthwise", "m.npz")

    def corrupt(arr):
        arr[:] = X.shape[1] + 3  # every slot, so active ones are included
        return arr

    bad = _mutated_archive(
        path, str(tmp_path / "lvl_feat.npz"),
        trees__node_features_flat=corrupt,
    )
    _assert_load_rejected(bad, "out of range")


# ---------------------------------------------------------------------------
# Blocker: encoder statistics must match the categorical payload.
# ---------------------------------------------------------------------------

def test_load_rejects_truncated_encoder_statistics(tmp_path):
    rng = np.random.default_rng(3)
    n = 300
    X = np.empty((n, 2), dtype=object)
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = np.array(["a", "b", "c", "d"], dtype=object)[
        rng.integers(0, 4, size=n)
    ]
    y = rng.normal(size=n)
    model = ChimeraBoostRegressor(
        iterations=8, random_state=0, early_stopping=False
    )
    model.fit(X, y, cat_features=[1])
    path = str(tmp_path / "cats.npz")
    model.save_model(path)

    # Drop the last category's statistics while keeping the payload
    # self-consistent (offsets still match the flat array lengths).
    bad = _mutated_archive(
        path, str(tmp_path / "cats_truncated.npz"),
        enc0__sums_flat=lambda arr: arr[:-1],
        enc0__counts_flat=lambda arr: arr[:-1],
        enc0__offsets=lambda arr: np.concatenate((arr[:-1], [arr[-1] - 1])),
    )
    _assert_load_rejected(bad, "do not match the categorical payload")


# ---------------------------------------------------------------------------
# Blocker: oversized class-minor histogram buffers must be rejected.
# ---------------------------------------------------------------------------

def test_multiclass_builder_rejects_oversized_class_dimension():
    rng = np.random.default_rng(5)
    n, f, K = 200, 4, 3
    X_binned = rng.integers(0, 8, size=(n, f)).astype(np.uint8)
    grad = rng.normal(size=(K, n))
    hess = np.abs(rng.normal(size=(K, n))) + 0.1
    n_bins = np.full(f, 8, dtype=np.int64)
    max_leaves = 7
    # Stale nonzero lanes in the extra class slot must never reach scoring.
    hg = np.full((f, max_leaves, 8, K + 1), 123.0)
    hh = np.full((f, max_leaves, 8, K + 1), 123.0)
    hc = np.zeros((f, max_leaves, 8))
    with pytest.raises(ValueError, match="class dimension"):
        build_leafwise_multiclass_tree(
            X_binned, grad, hess, n_bins, 4, 3.0, 0.1,
            hist_buffers=(hg, hh, hc), max_leaves=max_leaves,
        )


# ---------------------------------------------------------------------------
# Blocker: exact samplers must hit their targets at piecewise boundaries.
# ---------------------------------------------------------------------------

def test_exact_weighted_goss_hits_target_mass_at_boundaries():
    mass = np.array([3.0, 1.0, 1.0, 1.0, 0.5])
    breakpoints = np.cumsum(np.sort(mass)[::-1])
    targets = []
    for b in breakpoints[:-1]:
        for shift in (-3, -1, 0, 1, 3):
            t = b
            for _ in range(abs(shift)):
                t = np.nextafter(t, np.inf if shift > 0 else -np.inf)
            targets.append(float(t))
    targets += list(np.linspace(0.05, float(np.sum(mass)) - 1e-9, 23))
    for target in targets:
        probs = _exact_weighted_goss_probabilities(mass, target)
        achieved = float(np.sum(probs * mass))
        assert achieved == pytest.approx(target, rel=1e-9), (
            f"target_mass={target!r} achieved={achieved!r}"
        )


def test_exact_mvs_hits_target_count_at_boundaries():
    importance = np.array([4.0, 2.0, 2.0, 1.0, 0.25, 0.25])
    targets = list(np.linspace(0.5, importance.size - 1e-9, 29))
    # Exact saturation boundaries: theta == importance[k] transitions.
    sorted_desc = np.sort(importance)[::-1]
    prefix = np.concatenate(([0.0], np.cumsum(sorted_desc)))
    for k in range(1, importance.size):
        theta = sorted_desc[k - 1]
        boundary = k - 1 + (prefix[-1] - prefix[k - 1]) / theta
        if 0.0 < boundary < importance.size:
            for shift in (-2, 0, 2):
                t = boundary
                for _ in range(abs(shift)):
                    t = np.nextafter(t, np.inf if shift > 0 else -np.inf)
                targets.append(float(t))
    for target in targets:
        probs = _exact_mvs_probabilities(importance, target)
        assert float(np.sum(probs)) == pytest.approx(target, rel=1e-9), (
            f"target={target!r} sum={float(np.sum(probs))!r}"
        )
        assert np.all(probs <= 1.0 + 1e-15)


# ---------------------------------------------------------------------------
# Concern: bin_sample_count must survive save/load.
# ---------------------------------------------------------------------------


def test_object_cache_signature_separates_repr_hash_collisions():
    class ReprHashCollision:
        def __init__(self, group):
            self.group = group

        def __repr__(self):
            return "same"

        def __hash__(self):
            return 7

        def __eq__(self, other):
            return (
                isinstance(other, ReprHashCollision)
                and self.group == other.group
            )

    a = np.array([[ReprHashCollision("left")]], dtype=object)
    b = np.array([[ReprHashCollision("right")]], dtype=object)

    assert _array_content_signature(a) != _array_content_signature(b)
    assert _array_content_signature(a) == _array_content_signature(a)


@pytest.mark.parametrize("bin_sample_count", [1234, None])
def test_bin_sample_count_round_trips(tmp_path, bin_sample_count):
    X, y = _make_regression(n=250)
    model = ChimeraBoostRegressor(
        iterations=6, random_state=0, early_stopping=False,
        bin_sample_count=bin_sample_count,
    )
    model.fit(X, y)
    path = str(tmp_path / "bsc.npz")
    model.save_model(path)
    loaded = ChimeraBoostRegressor.load_model(path)
    assert loaded.model_.prep_.bin_sample_count == bin_sample_count
    assert np.array_equal(model.predict(X), loaded.predict(X))


# ---------------------------------------------------------------------------
# Sanity: valid models across all serializable modes still load and predict
# identically after the new validation (guards against over-rejection).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_mode", ["catboost", "lightgbm", "hybrid",
                                       "depthwise"])
def test_hardened_load_accepts_valid_models(tmp_path, tree_mode):
    X, _, path = _fit_and_save(tmp_path, tree_mode, f"{tree_mode}.npz")
    loaded = ChimeraBoostRegressor.load_model(path)
    reference = ChimeraBoostRegressor.load_model(path)
    assert np.array_equal(loaded.predict(X), reference.predict(X))


def test_hardened_load_accepts_valid_multiclass_models(tmp_path):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(300, 5))
    y = rng.integers(0, 3, size=300)
    for strategy in ("shared_vector", "per_class"):
        model = ChimeraBoostClassifier(
            iterations=8, tree_mode="lightgbm", random_state=0,
            early_stopping=False, multiclass_tree_strategy=strategy,
        )
        model.fit(X, y)
        path = str(tmp_path / f"mc_{strategy}.npz")
        model.save_model(path)
        loaded = ChimeraBoostClassifier.load_model(path)
        assert np.array_equal(model.predict_proba(X), loaded.predict_proba(X))
