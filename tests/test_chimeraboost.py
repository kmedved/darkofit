"""Test suite for ChimeraBoost. Run with: pytest -q"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier


def test_binner_uses_smallest_safe_unsigned_dtype():
    from chimeraboost.binning import Binner

    X = np.arange(300.0)[:, None]
    X[::17, 0] = np.nan

    for max_bins, expected_dtype in [
        (128, np.uint8),
        (254, np.uint8),
        (255, np.uint8),
        (256, np.uint16),
    ]:
        binner = Binner(max_bins=max_bins).fit(X)
        X_binned = binner.transform(X)

        assert X_binned.dtype == np.dtype(expected_dtype)
        assert int(X_binned.max()) < int(binner.n_bins_.max())
        assert np.all(X_binned[~np.isfinite(X[:, 0]), 0] == binner.n_bins_[0] - 1)

    low_cardinality = np.array([[0.0], [1.0], [2.0], [np.nan]])
    X_binned = Binner(max_bins=512).fit_transform(low_cardinality)
    assert X_binned.dtype == np.dtype(np.uint8)


def test_regressor_beats_mean_baseline():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = ChimeraBoostRegressor(iterations=300, random_state=0).fit(Xtr, ytr)
    rmse = np.sqrt(mean_squared_error(yte, m.predict(Xte)))
    baseline = np.sqrt(mean_squared_error(yte, np.full_like(yte, ytr.mean())))
    # diabetes is tiny and noisy; this is a single split, so the bound is loose
    # on purpose -- it checks the model meaningfully beats the mean, not a precise
    # ratio. (With early stopping or min_child_weight tuning it does better, but
    # this test exercises the bare default path.)
    assert rmse < 0.93 * baseline


def test_classifier_high_auc():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    m = ChimeraBoostClassifier(iterations=300, random_state=0).fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    assert auc > 0.97
    proba = m.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_ordered_ts_resists_leakage():
    """Ordered target stats should generalize on a high-cardinality categorical
    far better than the train/test gap a leaky encoder would show."""
    rng = np.random.default_rng(0)
    n, n_levels = 5000, 2500
    cat = rng.integers(0, n_levels, n)
    num = rng.normal(size=(n, 3))
    logit = 1.2 * num[:, 0] - num[:, 1] + rng.normal(0, 1, n)
    y = (logit > np.median(logit)).astype(int)
    X = np.empty((n, 4), dtype=object)
    X[:, 0] = np.array([f"id_{c}" for c in cat], dtype=object)
    X[:, 1:] = num
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )
    m = ChimeraBoostClassifier(iterations=200, random_state=1)
    m.fit(Xtr, ytr, cat_features=[0])
    tr = roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])
    te = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    assert te > 0.85          # generalizes
    assert tr - te < 0.10     # small gap, i.e. not memorizing the noise column


def test_early_stopping_trims_trees():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    m = ChimeraBoostClassifier(
        iterations=1000, early_stopping_rounds=20, random_state=0
    )
    m.fit(Xtr, ytr, eval_set=(Xte, yte))
    assert m.best_iteration_ < 1000


def test_handles_nan_and_unseen_categories():
    rng = np.random.default_rng(0)
    n = 1500
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = np.array([f"c{c}" for c in rng.integers(0, 8, n)], dtype=object)
    num = rng.normal(size=(n, 2))
    num[rng.random(n) < 0.1, 0] = np.nan
    X[:, 1:] = num
    y = ((num[:, 1] > 0) | (rng.random(n) < 0.3)).astype(int)
    m = ChimeraBoostClassifier(iterations=80, random_state=0)
    m.fit(X, y, cat_features=[0])
    Xnew = np.array([["c_UNSEEN", np.nan, 0.5], ["c3", 1.0, -0.5]], dtype=object)
    p = m.predict_proba(Xnew)
    assert p.shape == (2, 2)
    assert np.all((p >= 0) & (p <= 1))


def test_categorical_transform_preserves_missing_and_unseen_codes():
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.target_encoding import factorize

    raw = np.array(["b", "a", "b", None, np.nan, "__nan__"], dtype=object)
    codes, categories = factorize(raw)
    cat_to_code = {v: i for i, v in enumerate(categories)}
    assert codes[3] == cat_to_code["__nan__"]
    assert codes[4] == cat_to_code["__nan__"]
    assert codes[5] == cat_to_code["__nan__"]

    X = np.array([
        ["red", 1.0],
        ["blue", 2.0],
        ["red", 3.0],
        ["__nan__", 4.0],
    ], dtype=object)
    prep = FeaturePreprocessor(16, 1.0, 0)
    prep.fit_transform(X, [np.array([0.0, 1.0, 0.0, 1.0])], cat_features=[0])

    Xt = np.array([
        ["red", 10.0],
        ["green", 11.0],
        [None, 12.0],
        [np.nan, 13.0],
        ["__nan__", 14.0],
    ], dtype=object)
    transformed = prep._codes_for_transform(Xt)[:, 0]
    expected = np.array([
        prep.cat_maps_[0]["red"],
        -1,
        prep.cat_maps_[0]["__nan__"],
        prep.cat_maps_[0]["__nan__"],
        prep.cat_maps_[0]["__nan__"],
    ])
    assert np.array_equal(transformed, expected)

    X_num = np.array([[1.0], [2.0], [1.0], [3.0]], dtype=object)
    prep_num = FeaturePreprocessor(16, 1.0, 0)
    prep_num.fit_transform(X_num, [np.array([0.0, 1.0, 0.0, 1.0])], [0])
    Xt_num = np.array([[2.0], [4.0], [np.nan], [None]], dtype=object)
    expected_num = np.array([prep_num.cat_maps_[0][2.0], -1, -1, -1])
    assert np.array_equal(prep_num._codes_for_transform(Xt_num)[:, 0], expected_num)


def test_loaded_pandas_factorize_fast_path_preserves_missing_codes():
    pytest.importorskip("pandas")
    from chimeraboost.target_encoding import factorize

    raw = np.array(["b", "a", "b", None, np.nan, "__nan__"], dtype=object)
    codes, categories = factorize(raw)
    cat_to_code = {v: i for i, v in enumerate(categories)}

    assert codes[0] == codes[2]
    assert codes[1] != codes[0]
    assert codes[3] == cat_to_code["__nan__"]
    assert codes[4] == cat_to_code["__nan__"]
    assert codes[5] == cat_to_code["__nan__"]


def test_explicit_lr_overrides_auto():
    X, y = load_diabetes(return_X_y=True)
    m = ChimeraBoostRegressor(iterations=50, learning_rate=0.123).fit(X, y)
    assert m.model_.lr_ == 0.123


def test_verbose_timing_records_regression_fit_phases():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=0)
    m = ChimeraBoostRegressor(
        iterations=5, depth=2, early_stopping_rounds=3,
        verbose_timing=True, random_state=0
    ).fit(Xtr, ytr, eval_set=(Xv, yv))

    expected = {
        "preprocess", "grad_hess", "tree_build", "train_update",
        "validation_predict", "loss_eval",
    }
    assert set(m.timing_) == expected
    assert all(v >= 0.0 for v in m.timing_.values())
    assert m.timing_["preprocess"] > 0.0
    assert m.timing_["tree_build"] > 0.0
    assert m.timing_["validation_predict"] > 0.0


def test_verbose_timing_records_multiclass_fit_phases():
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    m = ChimeraBoostClassifier(
        iterations=3, depth=2, verbose_timing=True, random_state=0
    ).fit(X, y)

    assert m.timing_["preprocess"] > 0.0
    assert m.timing_["grad_hess"] > 0.0
    assert m.timing_["tree_build"] > 0.0
    assert m.timing_["validation_predict"] == 0.0


def test_multiclass_accuracy():
    from sklearn.datasets import load_wine, load_iris
    for load in (load_wine, load_iris):
        X, y = load(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y
        )
        m = ChimeraBoostClassifier(iterations=200, random_state=0).fit(Xtr, ytr)
        assert m.n_classes_ == 3
        proba = m.predict_proba(Xte)
        assert proba.shape == (len(yte), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (m.predict(Xte) == yte).mean() > 0.9


def test_multiclass_preserves_string_labels_and_categoricals():
    rng = np.random.default_rng(0)
    n = 2000
    region = rng.choice(["N", "S", "E"], n)
    x = rng.normal(size=(n, 2))
    score = np.select([region == "N", region == "S"], [1.5, -1.0], 0.0) + 0.4 * x[:, 0]
    y = np.array(["low", "mid", "high"])[np.digitize(score, [-0.3, 1.0])]
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = region
    X[:, 1:] = x
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
    m = ChimeraBoostClassifier(iterations=150, random_state=1)
    m.fit(Xtr, ytr, cat_features=[0])
    assert set(m.classes_) == {"low", "mid", "high"}
    assert set(np.unique(m.predict(Xte))).issubset({"low", "mid", "high"})


def test_multiclass_tree_builder_receives_class_column_views(monkeypatch):
    """Multiclass gradients should be laid out so per-class slices avoid copies."""
    import chimeraboost.booster as booster
    from sklearn.datasets import load_wine

    seen = []
    original = booster.build_oblivious_tree

    def wrapped_build_tree(X_binned, grad, hess, *args, **kwargs):
        seen.append((grad.flags.c_contiguous, hess.flags.c_contiguous,
                     grad.flags.owndata, hess.flags.owndata))
        return original(X_binned, grad, hess, *args, **kwargs)

    monkeypatch.setattr(booster, "build_oblivious_tree", wrapped_build_tree)
    X, y = load_wine(return_X_y=True)
    ChimeraBoostClassifier(iterations=2, random_state=0).fit(X, y)

    assert seen
    assert all(g_contig and h_contig for g_contig, h_contig, _, _ in seen)
    assert not any(g_own or h_own for _, _, g_own, h_own in seen)


def test_multiclass_preprocessor_receives_class_major_target_views(monkeypatch):
    """Per-class target-stat targets should be row views of one class-major Y."""
    import chimeraboost.booster as booster
    from sklearn.datasets import load_wine

    seen = []
    original = booster.FeaturePreprocessor.fit_transform

    def wrapped_fit_transform(self, X, encode_targets, cat_features):
        seen.append([
            (target.flags.c_contiguous, target.flags.owndata)
            for target in encode_targets
        ])
        return original(self, X, encode_targets, cat_features)

    monkeypatch.setattr(
        booster.FeaturePreprocessor, "fit_transform", wrapped_fit_transform
    )
    X, y = load_wine(return_X_y=True)
    ChimeraBoostClassifier(iterations=1, random_state=0).fit(X, y)

    assert seen
    assert all(contiguous and not owns_data for contiguous, owns_data in seen[0])


def test_multiclass_class_major_loss_matches_row_major():
    from chimeraboost.losses import MultiSoftmax

    rng = np.random.default_rng(11)
    Y = np.eye(4)[rng.integers(0, 4, size=200)]
    F = rng.normal(size=(200, 4))
    w = rng.uniform(0.5, 2.0, size=200)
    loss = MultiSoftmax(4)

    grad, hess = loss.grad_hess(Y, F)
    Y_class = np.ascontiguousarray(Y.T)
    F_class = np.ascontiguousarray(F.T)
    grad_c, hess_c = loss.grad_hess_class_major(
        Y_class, F_class
    )

    assert np.array_equal(loss.init(Y), loss.init_class_major(Y_class))
    assert np.allclose(loss.init(Y, w), loss.init_class_major(Y_class, w))
    assert np.array_equal(grad, grad_c.T)
    assert np.array_equal(hess, hess_c.T)
    assert loss.eval(Y, F) == loss.eval_class_major(Y_class, F_class)
    assert loss.eval(Y, F, w) == loss.eval_class_major(Y_class, F_class, w)


def test_feature_importances():
    rng = np.random.default_rng(0)
    n = 3000
    strong = rng.normal(size=n)
    noise = rng.normal(size=(n, 4))
    y = (strong + 0.1 * rng.normal(size=n) > 0).astype(int)
    X = np.column_stack([strong, noise])
    m = ChimeraBoostClassifier(iterations=100, random_state=0).fit(X, y)
    imp = m.feature_importances_
    assert imp.shape == (5,)
    assert abs(imp.sum() - 1.0) < 1e-6
    assert imp.argmax() == 0          # the informative feature dominates


def test_mae_loss_beats_rmse_on_mae_metric():
    from sklearn.metrics import mean_absolute_error
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    mae = ChimeraBoostRegressor(iterations=300, loss="MAE", random_state=0).fit(Xtr, ytr)
    rmse = ChimeraBoostRegressor(iterations=300, loss="RMSE", random_state=0).fit(Xtr, ytr)
    assert (mean_absolute_error(yte, mae.predict(Xte))
            <= mean_absolute_error(yte, rmse.predict(Xte)) + 1.0)


@pytest.mark.parametrize("loss_name", ["MAE", "Quantile"])
@pytest.mark.parametrize("weights", [None, np.array([
    1.0, 2.0, 0.5, 1.5, 3.0, 0.75, 1.25, 2.5, 0.8, 1.2, 1.7, 0.9
])])
@pytest.mark.parametrize("n_leaves", [8, 32])
def test_leaf_correction_matches_mask_semantics(loss_name, weights, n_leaves):
    from chimeraboost.booster import GradientBoosting
    from chimeraboost.losses import MAE, Quantile
    from chimeraboost.tree import ObliviousTree

    residuals = np.array([1.2, -0.4, 2.5, 0.0, -1.5, 3.2,
                          -0.7, 1.1, 0.8, -2.0, 4.1, -3.3])
    leaf = np.array([3, 0, 3, 1, 7, 1, 3, 0, 6, 7, 1, 3])
    lr = 0.37
    loss_obj = MAE() if loss_name == "MAE" else Quantile(alpha=0.3)

    expected = np.zeros(n_leaves)
    for l in range(n_leaves):
        mask = leaf == l
        w = weights[mask] if weights is not None else None
        expected[l] = lr * loss_obj.leaf_value(residuals[mask], w)

    tree = ObliviousTree(
        np.array([0, 1, 2]),
        np.array([0, 0, 0]),
        np.full(n_leaves, -999.0),
    )
    booster = GradientBoosting(loss=loss_name, learning_rate=lr)
    booster.lr_ = lr
    booster.loss_ = loss_obj

    booster._correct_leaves(
        tree, np.empty((residuals.shape[0], 0), dtype=np.uint16),
        residuals, weights, leaf=leaf
    )

    assert np.allclose(tree.values, expected)


def test_quantile_calibration_on_large_data():
    rng = np.random.default_rng(0)
    n = 10000
    X = rng.normal(size=(n, 5))
    y = 2 * X[:, 0] + rng.normal(0, 1, n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    qlo = ChimeraBoostRegressor(iterations=300, depth=4, loss="Quantile",
                                alpha=0.1, random_state=0).fit(Xtr, ytr)
    qhi = ChimeraBoostRegressor(iterations=300, depth=4, loss="Quantile",
                                alpha=0.9, random_state=0).fit(Xtr, ytr)
    cov = np.mean((yte >= qlo.predict(Xte)) & (yte <= qhi.predict(Xte)))
    assert 0.7 < cov < 0.88           # ~0.80 target band



def test_staged_predict_matches_final():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    r = ChimeraBoostRegressor(iterations=50, random_state=0).fit(Xtr, ytr)
    stages = list(r.staged_predict(Xte))
    assert len(stages) == r.best_iteration_
    assert np.allclose(stages[-1], r.predict(Xte))


def test_colsample_runs_and_keeps_accuracy():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    m = ChimeraBoostClassifier(iterations=150, colsample=0.5,
                               random_state=0).fit(Xtr, ytr)
    assert roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) > 0.97


def test_thread_count_records_effective_threads():
    import numba
    X, y = load_breast_cancer(return_X_y=True)
    m = ChimeraBoostClassifier(iterations=30, thread_count=1, random_state=0).fit(X, y)
    assert m.model_.n_threads_ == 1
    # None -> all detected cores
    m2 = ChimeraBoostClassifier(iterations=30, thread_count=None, random_state=0).fit(X, y)
    assert m2.model_.n_threads_ == numba.config.NUMBA_NUM_THREADS
    # over-request is clamped, never exceeds detected cores
    m3 = ChimeraBoostClassifier(iterations=30, thread_count=9999, random_state=0).fit(X, y)
    assert m3.model_.n_threads_ <= numba.config.NUMBA_NUM_THREADS


def test_thread_count_does_not_change_predictions():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    a = ChimeraBoostRegressor(iterations=80, thread_count=1, random_state=0).fit(Xtr, ytr)
    b = ChimeraBoostRegressor(iterations=80, thread_count=None, random_state=0).fit(Xtr, ytr)
    # histogram sums are deterministic regardless of thread count
    assert np.allclose(a.predict(Xte), b.predict(Xte))


def test_single_thread_fit_skips_threaded_split_buffers(monkeypatch):
    """Serial split search should not allocate threaded scratch buffers."""
    import chimeraboost.booster as booster

    def fail_if_called(self, n_features):
        raise AssertionError("threaded split buffers should not be allocated")

    monkeypatch.setattr(booster._BaseBooster, "_alloc_split_buffers", fail_if_called)
    X, y = load_diabetes(return_X_y=True)
    ChimeraBoostRegressor(
        iterations=3, depth=2, thread_count=1, random_state=0
    ).fit(X[:120], y[:120])


def test_min_child_weight_controls_depth_overfitting():
    """With min_child_weight active, increasing depth should NOT degrade test
    accuracy (the constraint stops growth before sparse leaves overfit). This is
    the property that fixes the oblivious-tree depth anomaly."""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=4000, n_features=30, n_informative=20,
                           noise=20, random_state=1000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=0)

    def rmse_at(depth, mcw):
        m = ChimeraBoostRegressor(iterations=1500, depth=depth,
                                  min_child_weight=mcw, early_stopping_rounds=50,
                                  random_state=0).fit(Xf, yf, eval_set=(Xv, yv))
        return np.sqrt(np.mean((yte - m.predict(Xte)) ** 2))

    # Unconstrained (mcw=1): deeper overfits -> depth 8 clearly worse than depth 4.
    assert rmse_at(8, 1) > rmse_at(4, 1)
    # Constrained (mcw=20): depth 8 should be no worse than a small tolerance
    # above depth 6 -- growth is capped, so extra depth is harmless.
    assert rmse_at(8, 20) <= rmse_at(6, 20) + 0.5


def test_min_child_weight_param_plumbing():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    m = ChimeraBoostClassifier(iterations=50, min_child_weight=30,
                               random_state=0).fit(X, y)
    assert m.model_.min_child_weight == 30.0


def test_shared_histogram_buffers_match_standalone():
    """A tree built with pre-allocated shared buffers must be identical to one
    built with its own freshly-allocated buffers (same math, no realloc)."""
    import numpy as np
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 12))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 800)).astype(float)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    nb = prep.n_bins_
    grad = (y - y.mean()); hess = np.ones(len(y))

    depth = 6
    standalone = build_oblivious_tree(Xb, grad, hess, nb, depth, 3.0, 0.1)
    nfeat = Xb.shape[1]; maxbins = int(nb.max()); maxleaves = 1 << depth
    bufs = (np.zeros((nfeat, maxleaves, maxbins)),
            np.zeros((nfeat, maxleaves, maxbins)))
    shared = build_oblivious_tree(Xb, grad, hess, nb, depth, 3.0, 0.1,
                                  hist_buffers=bufs)
    assert np.array_equal(standalone.splits_feat, shared.splits_feat)
    assert np.array_equal(standalone.splits_thr, shared.splits_thr)
    assert np.allclose(standalone.values, shared.values)

    # Reusing the SAME buffers for a second, different tree must not leak state.
    y2 = (X[:, 3] - X[:, 4] + rng.normal(0, 0.5, 800)).astype(float)
    g2 = (y2 - y2.mean())
    again = build_oblivious_tree(Xb, g2, hess, nb, depth, 3.0, 0.1,
                                 hist_buffers=bufs)
    fresh = build_oblivious_tree(Xb, g2, hess, nb, depth, 3.0, 0.1)
    assert np.array_equal(again.splits_feat, fresh.splits_feat)
    assert np.allclose(again.values, fresh.values)


def test_shared_split_buffers_match_standalone_threaded():
    """Threaded split-search scratch buffers must be reusable across trees."""
    import numba
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(13)
    X = rng.normal(size=(1000, 14))
    y = 1.2 * X[:, 0] - 0.9 * X[:, 3] + rng.normal(0, 0.4, 1000)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y - y.mean()
    hess = np.ones(len(y))
    depth = 5
    split_buffers = tuple(np.empty((Xb.shape[1], 1 << depth)) for _ in range(5))

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        standalone = build_oblivious_tree(
            Xb, grad, hess, prep.n_bins_, depth, 3.0, 0.1,
            return_training_state=True,
        )
        shared = build_oblivious_tree(
            Xb, grad, hess, prep.n_bins_, depth, 3.0, 0.1,
            split_buffers=split_buffers, return_training_state=True,
        )

        y2 = X[:, 5] + 0.8 * X[:, 6] + rng.normal(0, 0.4, 1000)
        g2 = y2 - y2.mean()
        again = build_oblivious_tree(
            Xb, g2, hess, prep.n_bins_, depth, 3.0, 0.1,
            split_buffers=split_buffers, return_training_state=True,
        )
        fresh = build_oblivious_tree(
            Xb, g2, hess, prep.n_bins_, depth, 3.0, 0.1,
            return_training_state=True,
        )
    finally:
        numba.set_num_threads(old_threads)

    for a, b in [(standalone, shared), (fresh, again)]:
        tree_a, leaf_a, G_a, H_a = a
        tree_b, leaf_b, G_b, H_b = b
        assert np.array_equal(tree_a.splits_feat, tree_b.splits_feat)
        assert np.array_equal(tree_a.splits_thr, tree_b.splits_thr)
        assert np.allclose(tree_a.values, tree_b.values)
        assert np.array_equal(leaf_a, leaf_b)
        assert np.allclose(G_a, G_b)
        assert np.allclose(H_a, H_b)


def test_returned_training_state_matches_tree_apply_and_bincount():
    """The optional training state returned by the tree builder must be exactly
    the same leaf routing and sums that callers would recompute externally."""
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree
    rng = np.random.default_rng(1)
    X = rng.normal(size=(900, 10))
    y = (1.5 * X[:, 0] - 0.7 * X[:, 2] + rng.normal(0, 0.4, 900))
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))

    tree, leaf, leaf_G, leaf_H = build_oblivious_tree(
        Xb, grad, hess, prep.n_bins_, 5, 3.0, 0.1,
        return_training_state=True,
    )

    expected_leaf = tree.apply(Xb)
    assert np.array_equal(leaf, expected_leaf)
    assert np.array_equal(leaf_G, np.bincount(leaf, weights=grad,
                                             minlength=len(leaf_G)))
    assert np.array_equal(leaf_H, np.bincount(leaf, weights=hess,
                                             minlength=len(leaf_H)))


def test_row_indices_training_state_uses_only_selected_rows():
    """Direct row-index tree builds should not include unselected row sums."""
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    rng = np.random.default_rng(11)
    X = rng.normal(size=(700, 9))
    y = X[:, 0] - 0.5 * X[:, 3] + rng.normal(0, 0.3, 700)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    row_indices = np.flatnonzero(rng.random(len(y)) < 0.4).astype(np.int64)

    tree, leaf, leaf_G, leaf_H = build_oblivious_tree(
        Xb, grad, hess, prep.n_bins_, 5, 3.0, 0.1,
        row_indices=row_indices, return_training_state=True,
    )

    expected_G = np.bincount(
        leaf[row_indices], weights=grad[row_indices], minlength=len(leaf_G)
    )
    expected_H = np.bincount(
        leaf[row_indices], weights=hess[row_indices], minlength=len(leaf_H)
    )
    assert np.array_equal(leaf, tree.apply(Xb))
    assert np.array_equal(leaf_G, expected_G)
    assert np.array_equal(leaf_H, expected_H)


def test_add_predict_matches_predict():
    """In-place tree prediction is an allocation-saving equivalent of predict."""
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree
    rng = np.random.default_rng(2)
    X = rng.normal(size=(700, 8))
    y = (X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.3, 700))
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    tree = build_oblivious_tree(Xb, grad, hess, prep.n_bins_, 5, 3.0, 0.1)

    out = np.zeros(Xb.shape[0])
    tree.add_predict(Xb, out)
    assert np.array_equal(out, tree.predict(Xb))


def test_levelwise_tree_add_predict_matches_predict():
    """The LightGBM-like tree representation must route predict paths alike."""
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_levelwise_tree

    rng = np.random.default_rng(2026)
    X = rng.normal(size=(900, 10))
    y = (
        1.5 * X[:, 0]
        + np.where(X[:, 1] > 0, X[:, 2], -X[:, 3])
        + rng.normal(0, 0.3, 900)
    )
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))

    tree, leaf, leaf_G, leaf_H = build_levelwise_tree(
        Xb, grad, hess, prep.n_bins_, 4, 3.0, 0.1,
        return_training_state=True,
    )

    out = np.zeros(Xb.shape[0])
    tree.add_predict(Xb, out)
    assert tree.depth > 0
    assert np.array_equal(out, tree.predict(Xb))
    assert np.array_equal(leaf, tree.apply(Xb))
    assert np.array_equal(leaf_G, np.bincount(leaf, weights=grad,
                                             minlength=len(leaf_G)))
    assert np.array_equal(leaf_H, np.bincount(leaf, weights=hess,
                                             minlength=len(leaf_H)))


def test_tree_mode_aliases_and_lightgbm_plumbing():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    catboost = ChimeraBoostClassifier(
        iterations=12, depth=3, tree_mode="catboost", random_state=0
    ).fit(Xtr, ytr)
    oblivious = ChimeraBoostClassifier(
        iterations=12, depth=3, tree_mode="oblivious", random_state=0
    ).fit(Xtr, ytr)
    lightgbm = ChimeraBoostClassifier(
        iterations=12, depth=3, tree_mode="lightgbm", random_state=0
    ).fit(Xtr, ytr)

    assert catboost.model_.tree_mode_ == "catboost"
    assert oblivious.model_.tree_mode_ == "catboost"
    assert lightgbm.model_.tree_mode_ == "lightgbm"
    assert np.array_equal(catboost.predict_proba(Xte), oblivious.predict_proba(Xte))
    assert lightgbm.predict_proba(Xte).shape == (len(Xte), 2)
    assert abs(lightgbm.feature_importances_.sum() - 1.0) < 1e-6


def test_l2_zero_illegal_splits_do_not_divide_by_zero():
    """Illegal empty-side split candidates must be discarded before gain math."""
    import numba
    from chimeraboost.tree import _best_split, _best_split_serial

    hg = np.zeros((1, 1, 3))
    hh = np.zeros((1, 1, 3))
    hg[0, 0, 0] = 5.0
    hh[0, 0, 0] = 10.0
    n_bins = np.array([3], dtype=np.int64)
    feat_mask = np.array([1], dtype=np.int64)
    scratch = tuple(np.empty((1, 1)) for _ in range(5))

    assert _best_split_serial(hg, hh, n_bins, 0.0, feat_mask, 1.0, 1)[1] == -1

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        assert _best_split(
            hg, hh, n_bins, 0.0, feat_mask, 1.0, 1,
            scratch[0], scratch[1], scratch[2], scratch[3], scratch[4],
        )[1] == -1
    finally:
        numba.set_num_threads(old_threads)


def test_best_split_serial_matches_parallel_histogram_search():
    """Parent-gain optimizations must not diverge between split-search paths."""
    import numba
    from chimeraboost.tree import _best_split, _best_split_serial

    rng = np.random.default_rng(12)
    hg = rng.normal(size=(7, 4, 8))
    hh = rng.uniform(0.05, 2.0, size=(7, 4, 8))
    n_bins = np.array([8, 7, 6, 8, 5, 4, 7], dtype=np.int64)
    feat_mask = np.array([1, 1, 0, 1, 1, 0, 1], dtype=np.int64)
    scratch = tuple(np.empty((hg.shape[0], 4)) for _ in range(5))

    serial = _best_split_serial(hg, hh, n_bins, 2.0, feat_mask, 0.1, 4)
    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        parallel = _best_split(
            hg, hh, n_bins, 2.0, feat_mask, 0.1, 4,
            scratch[0], scratch[1], scratch[2], scratch[3], scratch[4],
        )
    finally:
        numba.set_num_threads(old_threads)

    assert serial[:2] == parallel[:2]
    assert np.isclose(serial[2], parallel[2])


def test_ordered_leaf_update_l2_zero_singleton_is_finite():
    """Leave-one-out ordered updates should remain finite when l2=0."""
    from chimeraboost.booster import _ordered_leaf_update

    leaf = np.array([0, 1, 1])
    grad = np.array([1.0, 2.0, 3.0])
    hess = np.ones(3)
    leaf_G = np.array([1.0, 5.0])
    leaf_H = np.array([1.0, 2.0])

    update = _ordered_leaf_update(0.1, leaf, leaf_G, leaf_H, grad, hess, 0.0)

    assert np.all(np.isfinite(update))
    assert update[0] == 0.0


def test_feature_contiguous_hist_layout_matches_c_order_tree_build():
    """The optional F-order histogram matrix must not change tree structure."""
    import numba
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(3)
    X = rng.normal(size=(1000, 14))
    y = (X[:, 0] - 0.5 * X[:, 4] + rng.normal(0, 0.5, 1000))
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    Xb_hist = np.asfortranarray(Xb)
    grad = y.mean() - y
    hess = np.ones(len(y))

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        base = build_oblivious_tree(
            Xb, grad, hess, prep.n_bins_, 5, 3.0, 0.1,
            return_training_state=True,
        )
        layout = build_oblivious_tree(
            Xb, grad, hess, prep.n_bins_, 5, 3.0, 0.1,
            return_training_state=True, X_hist_binned=Xb_hist,
        )
    finally:
        numba.set_num_threads(old_threads)

    base_tree, base_leaf, base_G, base_H = base
    layout_tree, layout_leaf, layout_G, layout_H = layout
    assert np.array_equal(base_tree.splits_feat, layout_tree.splits_feat)
    assert np.array_equal(base_tree.splits_thr, layout_tree.splits_thr)
    assert np.array_equal(base_tree.values, layout_tree.values)
    assert np.array_equal(base_leaf, layout_leaf)
    assert np.array_equal(base_G, layout_G)
    assert np.array_equal(base_H, layout_H)


def test_selected_feature_histograms_match_masked_full_histograms():
    """Column-subsampled histogram building must match the old mask-only path."""
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    rng = np.random.default_rng(4)
    X = rng.normal(size=(1200, 16))
    y = (X[:, 1] - 0.8 * X[:, 7] + 0.5 * X[:, 12] + rng.normal(0, 0.4, 1200))
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    selected = np.array([1, 3, 7, 12], dtype=np.int64)
    mask = np.zeros(Xb.shape[1], dtype=np.int64)
    mask[selected] = 1

    full = build_oblivious_tree(
        Xb, grad, hess, prep.n_bins_, 5, 3.0, 0.1,
        feature_mask=mask, return_training_state=True,
    )
    selected_only = build_oblivious_tree(
        Xb, grad, hess, prep.n_bins_, 5, 3.0, 0.1,
        feature_mask=mask, feature_indices=selected,
        return_training_state=True,
    )

    full_tree, full_leaf, full_G, full_H = full
    sel_tree, sel_leaf, sel_G, sel_H = selected_only
    assert np.array_equal(full_tree.splits_feat, sel_tree.splits_feat)
    assert np.array_equal(full_tree.splits_thr, sel_tree.splits_thr)
    assert np.array_equal(full_tree.values, sel_tree.values)
    assert np.array_equal(full_leaf, sel_leaf)
    assert np.array_equal(full_G, sel_G)
    assert np.array_equal(full_H, sel_H)


def test_feature_indices_without_mask_self_mask_reused_histograms():
    """Selected histograms must not let stale unselected columns enter splits."""
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    rng = np.random.default_rng(5)
    X = rng.normal(size=(1000, 8))
    y = 4.0 * X[:, 0] + 0.1 * X[:, 3] + rng.normal(0, 0.2, 1000)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    selected = np.array([2, 3, 5], dtype=np.int64)
    mask = np.zeros(Xb.shape[1], dtype=np.int64)
    mask[selected] = 1
    hist_buffers = (
        np.zeros((Xb.shape[1], 1 << 4, int(prep.n_bins_.max()))),
        np.zeros((Xb.shape[1], 1 << 4, int(prep.n_bins_.max()))),
    )

    build_oblivious_tree(
        Xb, grad, hess, prep.n_bins_, 4, 3.0, 0.1,
        hist_buffers=hist_buffers, return_training_state=True,
    )
    inferred = build_oblivious_tree(
        Xb, grad, hess, prep.n_bins_, 4, 3.0, 0.1,
        feature_indices=selected, hist_buffers=hist_buffers,
        return_training_state=True,
    )
    explicit = build_oblivious_tree(
        Xb, grad, hess, prep.n_bins_, 4, 3.0, 0.1,
        feature_mask=mask, feature_indices=selected,
        return_training_state=True,
    )

    inferred_tree, inferred_leaf, inferred_G, inferred_H = inferred
    explicit_tree, explicit_leaf, explicit_G, explicit_H = explicit
    assert np.all(np.isin(inferred_tree.splits_feat, selected))
    assert np.array_equal(inferred_tree.splits_feat, explicit_tree.splits_feat)
    assert np.array_equal(inferred_tree.splits_thr, explicit_tree.splits_thr)
    assert np.array_equal(inferred_tree.values, explicit_tree.values)
    assert np.array_equal(inferred_leaf, explicit_leaf)
    assert np.array_equal(inferred_G, explicit_G)
    assert np.array_equal(inferred_H, explicit_H)


def test_feature_indices_must_match_feature_mask():
    """A mismatched selected-column mask would leave stale histograms eligible."""
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    rng = np.random.default_rng(6)
    X = rng.normal(size=(256, 5))
    y = X[:, 1] + rng.normal(0, 0.1, 256)
    prep = FeaturePreprocessor(32, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    selected = np.array([1, 3], dtype=np.int64)
    mismatched_mask = np.ones(Xb.shape[1], dtype=np.int64)

    with pytest.raises(ValueError, match="feature_indices must match feature_mask"):
        build_oblivious_tree(
            Xb, grad, hess, prep.n_bins_, 3, 3.0, 0.1,
            feature_mask=mismatched_mask, feature_indices=selected,
        )


def test_hist_buffers_must_be_large_enough():
    """Reusable histogram buffers should fail before Numba can write past them."""
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    rng = np.random.default_rng(17)
    X = rng.normal(size=(128, 4))
    y = X[:, 0] + rng.normal(0, 0.1, 128)
    prep = FeaturePreprocessor(32, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    bad_buffers = (
        np.zeros((Xb.shape[1] - 1, 1 << 3, int(prep.n_bins_.max()))),
        np.zeros((Xb.shape[1] - 1, 1 << 3, int(prep.n_bins_.max()))),
    )

    with pytest.raises(ValueError, match="hist_buffers are too small"):
        build_oblivious_tree(
            Xb, grad, hess, prep.n_bins_, 3, 3.0, 0.1,
            hist_buffers=bad_buffers,
        )


def test_histogram_layout_copy_must_match_training_shape():
    """X_hist_binned is indexed with the same rows/leaves as X_binned."""
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    rng = np.random.default_rng(18)
    X = rng.normal(size=(128, 4))
    y = X[:, 0] + rng.normal(0, 0.1, 128)
    prep = FeaturePreprocessor(32, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))

    with pytest.raises(ValueError, match="X_hist_binned must have the same shape"):
        build_oblivious_tree(
            Xb, grad, hess, prep.n_bins_, 3, 3.0, 0.1,
            X_hist_binned=Xb[:-1],
        )


def test_row_index_histograms_match_zeroed_subsample():
    """Selected-row histograms must match scanning all rows with zeroed grads."""
    import numba
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    rng = np.random.default_rng(7)
    X = rng.normal(size=(900, 12))
    y = 1.5 * X[:, 0] - 0.7 * X[:, 5] + rng.normal(0, 0.4, 900)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    row_mask = rng.random(len(y)) < 0.45
    row_indices = np.flatnonzero(row_mask).astype(np.int64)
    g = np.where(row_mask, grad, 0.0)
    h = np.where(row_mask, hess, 0.0)
    selected = np.array([0, 2, 5, 8], dtype=np.int64)
    feature_mask = np.zeros(Xb.shape[1], dtype=np.int64)
    feature_mask[selected] = 1

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        zeroed = build_oblivious_tree(
            Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
            return_training_state=True,
        )
        indexed = build_oblivious_tree(
            Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
            row_indices=row_indices, return_training_state=True,
        )
        zeroed_selected = build_oblivious_tree(
            Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
            feature_mask=feature_mask, feature_indices=selected,
            return_training_state=True,
        )
        indexed_selected = build_oblivious_tree(
            Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
            feature_mask=feature_mask, feature_indices=selected,
            row_indices=row_indices, return_training_state=True,
        )
    finally:
        numba.set_num_threads(old_threads)

    zeroed_tree, zeroed_leaf, zeroed_G, zeroed_H = zeroed
    indexed_tree, indexed_leaf, indexed_G, indexed_H = indexed
    assert np.array_equal(zeroed_tree.splits_feat, indexed_tree.splits_feat)
    assert np.array_equal(zeroed_tree.splits_thr, indexed_tree.splits_thr)
    assert np.array_equal(zeroed_tree.values, indexed_tree.values)
    assert np.array_equal(zeroed_leaf, indexed_leaf)
    assert np.array_equal(zeroed_G, indexed_G)
    assert np.array_equal(zeroed_H, indexed_H)

    ordered_update = -0.1 * (indexed_G[indexed_leaf] - g) / (
        np.maximum(indexed_H[indexed_leaf] - h, 0.0) + 3.0
    )
    assert np.allclose(ordered_update[~row_mask],
                       indexed_tree.predict(Xb)[~row_mask])

    zeroed_tree, zeroed_leaf, zeroed_G, zeroed_H = zeroed_selected
    indexed_tree, indexed_leaf, indexed_G, indexed_H = indexed_selected
    assert np.array_equal(zeroed_tree.splits_feat, indexed_tree.splits_feat)
    assert np.array_equal(zeroed_tree.splits_thr, indexed_tree.splits_thr)
    assert np.array_equal(zeroed_tree.values, indexed_tree.values)
    assert np.array_equal(zeroed_leaf, indexed_leaf)
    assert np.array_equal(zeroed_G, indexed_G)
    assert np.array_equal(zeroed_H, indexed_H)


def test_row_and_feature_index_histograms_match_zeroed_subsample_serial():
    """Selected rows compose with selected features in the single-thread path."""
    import numba
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    rng = np.random.default_rng(8)
    X = rng.normal(size=(850, 14))
    y = X[:, 2] - 0.9 * X[:, 9] + 0.3 * X[:, 11] + rng.normal(0, 0.3, 850)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    row_mask = rng.random(len(y)) < 0.5
    row_indices = np.flatnonzero(row_mask).astype(np.int64)
    g = np.where(row_mask, grad, 0.0)
    h = np.where(row_mask, hess, 0.0)
    selected = np.array([2, 4, 9, 11], dtype=np.int64)
    feature_mask = np.zeros(Xb.shape[1], dtype=np.int64)
    feature_mask[selected] = 1

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        zeroed = build_oblivious_tree(
            Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
            feature_mask=feature_mask, feature_indices=selected,
            return_training_state=True,
        )
        indexed = build_oblivious_tree(
            Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
            feature_mask=feature_mask, feature_indices=selected,
            row_indices=row_indices, return_training_state=True,
        )
    finally:
        numba.set_num_threads(old_threads)

    zeroed_tree, zeroed_leaf, zeroed_G, zeroed_H = zeroed
    indexed_tree, indexed_leaf, indexed_G, indexed_H = indexed
    assert np.array_equal(zeroed_tree.splits_feat, indexed_tree.splits_feat)
    assert np.array_equal(zeroed_tree.splits_thr, indexed_tree.splits_thr)
    assert np.array_equal(zeroed_tree.values, indexed_tree.values)
    assert np.array_equal(zeroed_leaf, indexed_leaf)
    assert np.array_equal(zeroed_G, indexed_G)
    assert np.array_equal(zeroed_H, indexed_H)


def test_constant_hessian_histograms_match_generic_threaded():
    """Unit-Hessian histogram kernels must match generic hess=ones kernels."""
    import numba
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(9)
    X = rng.normal(size=(900, 15))
    y = 1.2 * X[:, 0] - 0.9 * X[:, 6] + 0.4 * X[:, 12] + rng.normal(0, 0.3, 900)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    Xb_hist = np.asfortranarray(Xb)
    grad = y.mean() - y
    hess = np.ones(len(y))
    row_mask = rng.random(len(y)) < 0.55
    row_indices = np.flatnonzero(row_mask).astype(np.int64)
    g_sub = np.where(row_mask, grad, 0.0)
    h_sub = np.where(row_mask, hess, 0.0)
    selected = np.array([0, 3, 6, 12], dtype=np.int64)
    feature_mask = np.zeros(Xb.shape[1], dtype=np.int64)
    feature_mask[selected] = 1

    cases = [
        (grad, hess, {}),
        (grad, hess, {"feature_mask": feature_mask, "feature_indices": selected}),
        (g_sub, h_sub, {"row_indices": row_indices}),
        (
            g_sub,
            h_sub,
            {
                "feature_mask": feature_mask,
                "feature_indices": selected,
                "row_indices": row_indices,
            },
        ),
    ]

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        for g, h, extra in cases:
            generic = build_oblivious_tree(
                Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
                X_hist_binned=Xb_hist, return_training_state=True, **extra
            )
            fast = build_oblivious_tree(
                Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
                X_hist_binned=Xb_hist, return_training_state=True,
                constant_hessian=True, **extra
            )
            generic_tree, generic_leaf, generic_G, generic_H = generic
            fast_tree, fast_leaf, fast_G, fast_H = fast
            assert np.array_equal(generic_tree.splits_feat, fast_tree.splits_feat)
            assert np.array_equal(generic_tree.splits_thr, fast_tree.splits_thr)
            assert np.array_equal(generic_tree.values, fast_tree.values)
            assert np.array_equal(generic_leaf, fast_leaf)
            assert np.array_equal(generic_G, fast_G)
            assert np.array_equal(generic_H, fast_H)
    finally:
        numba.set_num_threads(old_threads)


def test_constant_hessian_histograms_match_generic_serial():
    """The single-thread unit-Hessian kernels must preserve tree state exactly."""
    import numba
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_oblivious_tree

    rng = np.random.default_rng(10)
    X = rng.normal(size=(850, 13))
    y = X[:, 2] + 0.8 * X[:, 8] - 0.4 * X[:, 10] + rng.normal(0, 0.3, 850)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    row_mask = rng.random(len(y)) < 0.5
    row_indices = np.flatnonzero(row_mask).astype(np.int64)
    g_sub = np.where(row_mask, grad, 0.0)
    h_sub = np.where(row_mask, hess, 0.0)
    selected = np.array([2, 5, 8, 10], dtype=np.int64)
    feature_mask = np.zeros(Xb.shape[1], dtype=np.int64)
    feature_mask[selected] = 1

    cases = [
        (grad, hess, {}),
        (grad, hess, {"feature_mask": feature_mask, "feature_indices": selected}),
        (g_sub, h_sub, {"row_indices": row_indices}),
        (
            g_sub,
            h_sub,
            {
                "feature_mask": feature_mask,
                "feature_indices": selected,
                "row_indices": row_indices,
            },
        ),
    ]

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        for g, h, extra in cases:
            generic = build_oblivious_tree(
                Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
                return_training_state=True, **extra
            )
            fast = build_oblivious_tree(
                Xb, g, h, prep.n_bins_, 5, 3.0, 0.1,
                return_training_state=True, constant_hessian=True, **extra
            )
            generic_tree, generic_leaf, generic_G, generic_H = generic
            fast_tree, fast_leaf, fast_G, fast_H = fast
            assert np.array_equal(generic_tree.splits_feat, fast_tree.splits_feat)
            assert np.array_equal(generic_tree.splits_thr, fast_tree.splits_thr)
            assert np.array_equal(generic_tree.values, fast_tree.values)
            assert np.array_equal(generic_leaf, fast_leaf)
            assert np.array_equal(generic_G, fast_G)
            assert np.array_equal(generic_H, fast_H)
    finally:
        numba.set_num_threads(old_threads)


# ---------------------------------------------------------------------------
# sample_weight tests
# ---------------------------------------------------------------------------

def test_sample_weight_uniform_equals_no_weight_rmse():
    """sample_weight=ones must give bitwise-identical predictions to no weight
    for RMSE: normalized ones leave grad/hess unchanged, np.average(y,w=None)==mean."""
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    w = np.ones(len(ytr))
    m_none = ChimeraBoostRegressor(iterations=80, random_state=0).fit(Xtr, ytr)
    m_ones = ChimeraBoostRegressor(iterations=80, random_state=0).fit(
        Xtr, ytr, sample_weight=w
    )
    assert np.array_equal(m_none.predict(Xte), m_ones.predict(Xte))


def test_sample_weight_uniform_equals_no_weight_logloss():
    """Same exact-equality check for binary classification (Logloss)."""
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    w = np.ones(len(ytr))
    m_none = ChimeraBoostClassifier(iterations=80, random_state=0).fit(Xtr, ytr)
    m_ones = ChimeraBoostClassifier(iterations=80, random_state=0).fit(
        Xtr, ytr, sample_weight=w
    )
    assert np.array_equal(m_none.predict_proba(Xte), m_ones.predict_proba(Xte))


def test_sample_weight_uniform_equals_no_weight_multiclass():
    """Same exact-equality check for multiclass (softmax)."""
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    w = np.ones(len(ytr))
    m_none = ChimeraBoostClassifier(iterations=80, random_state=0).fit(Xtr, ytr)
    m_ones = ChimeraBoostClassifier(iterations=80, random_state=0).fit(
        Xtr, ytr, sample_weight=w
    )
    assert np.array_equal(m_none.predict_proba(Xte), m_ones.predict_proba(Xte))


def test_sample_weight_shifts_predictions():
    """Up-weighting the high-y half of the training set should push the mean
    prediction higher on held-out data relative to the unweighted model."""
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(size=(n, 5))
    y = 3.0 * X[:, 0] + rng.normal(0, 0.5, n)   # strong signal in col 0
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0)

    # Build weights: samples with above-median y get weight 5, others get 1.
    w_high = np.where(ytr >= np.median(ytr), 5.0, 1.0)
    w_low  = np.where(ytr <  np.median(ytr), 5.0, 1.0)

    m_base = ChimeraBoostRegressor(iterations=150, random_state=0).fit(Xtr, ytr)
    m_high = ChimeraBoostRegressor(iterations=150, random_state=0).fit(
        Xtr, ytr, sample_weight=w_high
    )
    m_low  = ChimeraBoostRegressor(iterations=150, random_state=0).fit(
        Xtr, ytr, sample_weight=w_low
    )
    mean_base = m_base.predict(Xte).mean()
    mean_high = m_high.predict(Xte).mean()
    mean_low  = m_low.predict(Xte).mean()

    # Up-weighting high-y samples → higher mean predictions, and vice-versa.
    assert mean_high > mean_base > mean_low


def test_sample_weight_early_stopping_slices_correctly():
    """When early_stopping=True, the weight array must be sliced to match the
    training split; the fit should complete without error and stop early."""
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(7)
    w = rng.uniform(0.5, 2.0, len(y))
    m = ChimeraBoostClassifier(
        iterations=500, early_stopping=True, validation_fraction=0.15,
        early_stopping_rounds=20, random_state=0
    ).fit(X, y, sample_weight=w)
    assert m.best_iteration_ < 500


def test_empty_tree_stops_boosting_early():
    """When splits are exhausted, the booster should stop rather than bank
    useless depth-0 trees until the iteration ceiling."""
    import numpy as np
    # One informative feature, aggressive min_child_weight -> splits run out fast.
    X = np.array([[0.0]] * 60 + [[1.0]] * 60)
    y = np.array([0.0] * 60 + [1.0] * 60)
    m = ChimeraBoostRegressor(iterations=1000, min_child_weight=30,
                              random_state=0).fit(X, y)
    assert len(m.model_.trees_) < 1000
