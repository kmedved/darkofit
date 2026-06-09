"""Test suite for ChimeraBoost. Run with: pytest -q"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

from chimeraboost import ChimeraBoostRegressor, ChimeraBoostClassifier


def test_loss_grad_hess_into_matches_allocating_paths():
    from chimeraboost.losses import Logloss, MAE, MultiSoftmax, Quantile, RMSE

    rng = np.random.default_rng(30)
    y_reg = rng.normal(size=40)
    raw_reg = rng.normal(size=40)
    weights = rng.uniform(0.2, 2.0, size=40)
    y_bin = rng.integers(0, 2, size=40).astype(float)

    for loss, y in [
        (RMSE(), y_reg),
        (MAE(), y_reg),
        (Quantile(0.3), y_reg),
        (Logloss(), y_bin),
    ]:
        for w in (None, weights):
            grad, hess = loss.grad_hess(y, raw_reg)
            if w is not None:
                grad = grad * w
                hess = hess * w
            grad_out = np.empty_like(raw_reg)
            hess_out = np.empty_like(raw_reg)
            loss.grad_hess_into(y, raw_reg, w, grad_out, hess_out)
            assert np.array_equal(grad_out, grad)
            assert np.array_equal(hess_out, hess)

    K = 4
    labels = rng.integers(0, K, size=40)
    Y = np.zeros((K, len(labels)))
    Y[labels, np.arange(len(labels))] = 1.0
    F = rng.normal(size=Y.shape)
    loss = MultiSoftmax(K)
    for w in (None, weights):
        grad, hess = loss.grad_hess_class_major(Y, F)
        if w is not None:
            grad = grad * w[None, :]
            hess = hess * w[None, :]
        grad_out = np.empty_like(F)
        hess_out = np.empty_like(F)
        loss.grad_hess_class_major_into(Y, F, w, grad_out, hess_out)
        assert np.array_equal(grad_out, grad)
        assert np.array_equal(hess_out, hess)


def test_classification_grad_hess_into_extreme_values_match_allocating_paths():
    from chimeraboost.losses import Logloss, MultiSoftmax

    y = np.array([0.0, 1.0, 0.0, 1.0])
    raw = np.array([-1000.0, -60.0, 60.0, 1000.0])
    weights = np.array([0.5, 2.0, 0.25, 3.0])
    loss = Logloss()
    for w in (None, weights):
        grad, hess = loss.grad_hess(y, raw)
        if w is not None:
            grad = grad * w
            hess = hess * w
        grad_out = np.empty_like(raw)
        hess_out = np.empty_like(raw)
        loss.grad_hess_into(y, raw, w, grad_out, hess_out)
        assert np.array_equal(grad_out, grad)
        assert np.array_equal(hess_out, hess)

    Y = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    F = np.array([
        [800.0, -800.0, 0.0],
        [799.0, -799.0, 1.0],
        [798.0, -798.0, -1.0],
    ])
    loss = MultiSoftmax(3)
    for w in (None, weights[:3]):
        grad, hess = loss.grad_hess_class_major(Y, F)
        if w is not None:
            grad = grad * w[None, :]
            hess = hess * w[None, :]
        grad_out = np.empty_like(F)
        hess_out = np.empty_like(F)
        loss.grad_hess_class_major_into(Y, F, w, grad_out, hess_out)
        assert np.array_equal(grad_out, grad)
        assert np.array_equal(hess_out, hess)


def test_logloss_eval_matches_clipped_probability_formula():
    from chimeraboost.losses import Logloss

    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    raw = np.array([-1000.0, -60.0, -1.5, 4.0, 1000.0])
    weights = np.array([0.5, 2.0, 0.25, 3.0, 1.5])
    loss = Logloss()

    p = 1.0 / (1.0 + np.exp(-np.clip(raw, -700.0, 700.0)))
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    ce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    assert np.isclose(loss.eval(y, raw), np.average(ce))
    assert np.isclose(loss.eval(y, raw, weights), np.average(ce, weights=weights))


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


def test_preprocessor_can_include_raw_category_code_features():
    from chimeraboost.preprocessing import FeaturePreprocessor

    X = np.array([
        ["red", "north", 1.0],
        ["blue", "south", 2.0],
        ["red", "south", 3.0],
        ["green", "north", 4.0],
    ], dtype=object)
    y = np.array([0.0, 1.0, 0.5, 1.5])

    default = FeaturePreprocessor(16, 1.0, 0)
    Xb_default = default.fit_transform(X, [y], cat_features=[0, 1])
    with_codes = FeaturePreprocessor(16, 1.0, 0, include_cat_codes=True)
    Xb_codes = with_codes.fit_transform(X, [y], cat_features=[0, 1])

    assert Xb_default.shape[1] == 3
    assert Xb_codes.shape[1] == 5
    assert np.array_equal(with_codes.feature_map_, np.array([2, 0, 1, 0, 1]))

    Xt = np.array([
        ["red", "south", 10.0],
        ["purple", "north", 11.0],
    ], dtype=object)
    assert with_codes.transform(Xt).shape[1] == 5


def test_kfold_target_encoding_uses_out_of_fold_totals():
    from chimeraboost.target_encoding import OrderedTargetEncoder

    codes = np.array([[0], [1], [2], [2]], dtype=np.int64)
    y = np.array([10.0, -10.0, 1.0, 3.0])
    prior = y.mean()

    enc = OrderedTargetEncoder(1.0, 0, mode="kfold", n_folds=2)
    train_encoded = enc.fit_transform(codes, y)

    assert train_encoded[0, 0] == prior
    assert train_encoded[1, 0] == prior
    assert not np.allclose(enc.transform(codes[:2])[:, 0], prior)


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

    def wrapped_fit_transform(self, X, encode_targets, cat_features,
                              sample_weight=None):
        seen.append([
            (target.flags.c_contiguous, target.flags.owndata)
            for target in encode_targets
        ])
        return original(
            self, X, encode_targets, cat_features,
            sample_weight=sample_weight
        )

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


def test_lightgbm_small_fit_caps_thread_count_as_maximum():
    X, y = load_breast_cancer(return_X_y=True)
    m = ChimeraBoostClassifier(
        iterations=3, tree_mode="lightgbm", num_leaves=7,
        thread_count=8, random_state=0
    ).fit(X, y)
    assert m.model_.n_threads_ <= 2


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
    """The experimental depth-wise tree representation must route predict paths alike."""
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


def _reference_leafwise_splits(Xb, grad, hess, n_bins, max_leaves, max_depth,
                               l2, min_child_weight, min_child_samples,
                               min_gain):
    leaf = np.zeros(Xb.shape[0], dtype=np.int64)
    leaf_depth = [0]
    splits = []
    for _ in range(max_leaves - 1):
        best = None
        for l in range(len(leaf_depth)):
            if max_depth >= 0 and leaf_depth[l] >= max_depth:
                continue
            rows = np.flatnonzero(leaf == l)
            if not len(rows):
                continue
            Gt = grad[rows].sum()
            Ht = hess[rows].sum()
            Ct = np.count_nonzero(hess[rows] > 0)
            if Ht <= 0 or Ct <= 0:
                continue
            parent = Gt * Gt / (Ht + l2)
            for f in range(Xb.shape[1]):
                for t in range(n_bins[f] - 1):
                    left_mask = Xb[rows, f] <= t
                    right_mask = ~left_mask
                    left_rows = rows[left_mask]
                    right_rows = rows[right_mask]
                    HL = hess[left_rows].sum()
                    HR = hess[right_rows].sum()
                    CL = np.count_nonzero(hess[left_rows] > 0)
                    CR = np.count_nonzero(hess[right_rows] > 0)
                    if (
                        HL < min_child_weight
                        or HR < min_child_weight
                        or CL < min_child_samples
                        or CR < min_child_samples
                    ):
                        continue
                    GL = grad[left_rows].sum()
                    GR = grad[right_rows].sum()
                    gain = (
                        GL * GL / (HL + l2)
                        + GR * GR / (HR + l2)
                        - parent
                    )
                    if best is None or gain > best[0]:
                        best = (gain, l, f, t)
        if best is None or best[0] <= min_gain:
            break
        gain, l, f, t = best
        new_leaf = len(leaf_depth)
        leaf[(leaf == l) & (Xb[:, f] > t)] = new_leaf
        leaf_depth[l] += 1
        leaf_depth.append(leaf_depth[l])
        splits.append((f, t, gain))
    return splits, leaf


def test_leafwise_tree_matches_bruteforce_reference():
    """LightGBM mode must grow best-first leaf-wise, not depth-wise."""
    from chimeraboost.tree import add_leaf_values_inplace, build_leafwise_tree

    Xb = np.array([
        [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
        [2, 0], [2, 1], [2, 2], [2, 2],
    ], dtype=np.uint8)
    y = np.array([-1.5, -1.2, -1.0, -0.1, 0.1, 0.2, 1.0, 1.2, 2.5, 2.7])
    grad = y.mean() - y
    hess = np.ones_like(grad)
    n_bins = np.array([3, 3], dtype=np.int64)

    tree, leaf, leaf_G, leaf_H = build_leafwise_tree(
        Xb, grad, hess, n_bins, 3, 1.0, 0.1,
        max_leaves=4, min_child_samples=2, min_child_weight=1.0,
        min_gain_to_split=0.0, return_training_state=True,
    )
    ref_splits, ref_leaf = _reference_leafwise_splits(
        Xb, grad, hess, n_bins, 4, 3, 1.0, 1.0, 2, 0.0
    )

    assert tree.n_leaves == 1 + len(ref_splits)
    assert tree.depth <= 3
    assert np.array_equal(tree.splits_feat, [s[0] for s in ref_splits])
    assert np.array_equal(tree.splits_thr, [s[1] for s in ref_splits])
    assert np.allclose(tree.gains, [s[2] for s in ref_splits])
    assert np.array_equal(leaf, ref_leaf)
    assert np.array_equal(tree.apply(Xb), ref_leaf)
    assert np.array_equal(leaf_G, np.bincount(leaf, weights=grad,
                                             minlength=tree.n_leaves))
    assert np.array_equal(leaf_H, np.bincount(leaf, weights=hess,
                                             minlength=tree.n_leaves))
    out = np.zeros(Xb.shape[0])
    tree.add_predict(Xb, out)
    assert np.array_equal(out, tree.predict(Xb))
    direct = np.zeros(Xb.shape[0])
    add_leaf_values_inplace(leaf, tree.values, direct)
    assert np.array_equal(direct, out)


def test_leafwise_multiclass_leaf_update_matches_predict():
    """Training leaf ids should be reusable for shared multiclass updates."""
    from chimeraboost.tree import (
        add_multiclass_leaf_values_inplace,
        build_leafwise_multiclass_tree,
    )

    Xb = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [2, 0], [2, 1], [2, 2], [0, 2],
    ], dtype=np.uint8)
    grad = np.array([
        [0.3, -0.2, 0.1, -0.4, 0.2, 0.0, -0.3, 0.1],
        [-0.1, 0.4, -0.2, 0.2, -0.3, 0.1, 0.2, -0.4],
        [-0.2, -0.2, 0.1, 0.2, 0.1, -0.1, 0.1, 0.3],
    ], dtype=np.float64)
    hess = np.full_like(grad, 0.5)
    n_bins = np.array([3, 3], dtype=np.int64)

    tree, leaf, _, _ = build_leafwise_multiclass_tree(
        Xb, grad, hess, n_bins, 3, 1.0, 0.1,
        max_leaves=4, min_child_samples=1, min_child_weight=0.0,
        min_gain_to_split=0.0, return_training_state=True,
    )

    predicted = np.zeros_like(grad)
    tree.add_predict_class_major(Xb, predicted)
    direct = np.zeros_like(grad)
    add_multiclass_leaf_values_inplace(leaf, tree.values, direct)
    assert np.array_equal(direct, predicted)


def test_multiclass_refill_subtract_matches_two_step():
    from chimeraboost.tree import (
        _refill_multiclass_leaf_segment_histograms_counts_into,
        _refill_multiclass_left_subtract_right_counts_into,
        _refill_multiclass_right_subtract_left_counts_into,
    )

    rng = np.random.default_rng(59)
    Xb = rng.integers(0, 23, size=(128, 9), dtype=np.uint8)
    grad = rng.normal(size=(4, Xb.shape[0]))
    hess = rng.uniform(0.05, 1.5, size=grad.shape)
    row_order = np.arange(Xb.shape[0], dtype=np.int64)
    leaf_start = np.array([0, 37, 88, 128], dtype=np.int64)
    left_leaf = 1
    right_leaf = 2

    base_hg = rng.normal(size=(4, Xb.shape[1], 3, 23))
    base_hh = rng.uniform(0.0, 5.0, size=base_hg.shape)
    base_hc = rng.uniform(0.0, 5.0, size=(Xb.shape[1], 3, 23))

    fused_hg = base_hg.copy()
    fused_hh = base_hh.copy()
    fused_hc = base_hc.copy()
    ref_hg = base_hg.copy()
    ref_hh = base_hh.copy()
    ref_hc = base_hc.copy()
    _refill_multiclass_right_subtract_left_counts_into(
        Xb, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
        fused_hg, fused_hh, fused_hc
    )
    _refill_multiclass_leaf_segment_histograms_counts_into(
        Xb, grad, hess, row_order, leaf_start,
        np.array([right_leaf], dtype=np.int64), 1, ref_hg, ref_hh, ref_hc
    )
    ref_hg[:, :, left_leaf] -= ref_hg[:, :, right_leaf]
    ref_hh[:, :, left_leaf] -= ref_hh[:, :, right_leaf]
    ref_hc[:, left_leaf] -= ref_hc[:, right_leaf]
    assert np.array_equal(fused_hg, ref_hg)
    assert np.array_equal(fused_hh, ref_hh)
    assert np.array_equal(fused_hc, ref_hc)

    fused_hg = base_hg.copy()
    fused_hh = base_hh.copy()
    fused_hc = base_hc.copy()
    ref_hg = base_hg.copy()
    ref_hh = base_hh.copy()
    ref_hc = base_hc.copy()
    parent_hg = ref_hg[:, :, left_leaf].copy()
    parent_hh = ref_hh[:, :, left_leaf].copy()
    parent_hc = ref_hc[:, left_leaf].copy()
    _refill_multiclass_left_subtract_right_counts_into(
        Xb, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
        fused_hg, fused_hh, fused_hc
    )
    _refill_multiclass_leaf_segment_histograms_counts_into(
        Xb, grad, hess, row_order, leaf_start,
        np.array([left_leaf], dtype=np.int64), 1, ref_hg, ref_hh, ref_hc
    )
    ref_hg[:, :, right_leaf] = parent_hg - ref_hg[:, :, left_leaf]
    ref_hh[:, :, right_leaf] = parent_hh - ref_hh[:, :, left_leaf]
    ref_hc[:, right_leaf] = parent_hc - ref_hc[:, left_leaf]
    assert np.array_equal(fused_hg, ref_hg)
    assert np.array_equal(fused_hh, ref_hh)
    assert np.array_equal(fused_hc, ref_hc)


def test_leafwise_multiclass_histogram_subtraction_matches_full_refill():
    from chimeraboost.tree import build_leafwise_multiclass_tree

    rng = np.random.default_rng(60)
    Xb = rng.integers(0, 48, size=(900, 14), dtype=np.uint8)
    grad = rng.normal(size=(3, Xb.shape[0]))
    hess = rng.uniform(0.05, 1.5, size=grad.shape)
    n_bins = np.full(Xb.shape[1], 48, dtype=np.int64)

    reused = build_leafwise_multiclass_tree(
        Xb, grad, hess, n_bins, 6, 1.2, 0.1,
        max_leaves=13, min_child_samples=5, min_child_weight=0.1,
        min_gain_to_split=0.0, return_training_state=True,
        reuse_leaf_histograms=True,
    )
    full = build_leafwise_multiclass_tree(
        Xb, grad, hess, n_bins, 6, 1.2, 0.1,
        max_leaves=13, min_child_samples=5, min_child_weight=0.1,
        min_gain_to_split=0.0, return_training_state=True,
        reuse_leaf_histograms=False,
    )

    reused_tree, reused_leaf, reused_G, reused_H = reused
    full_tree, full_leaf, full_G, full_H = full
    assert np.array_equal(reused_tree.features, full_tree.features)
    assert np.array_equal(reused_tree.thresholds, full_tree.thresholds)
    assert np.array_equal(reused_tree.left_child, full_tree.left_child)
    assert np.array_equal(reused_tree.right_child, full_tree.right_child)
    assert np.array_equal(reused_tree.leaf_index, full_tree.leaf_index)
    assert np.array_equal(reused_tree.splits_feat, full_tree.splits_feat)
    assert np.array_equal(reused_tree.splits_thr, full_tree.splits_thr)
    assert np.allclose(reused_tree.gains, full_tree.gains)
    assert np.allclose(reused_tree.values, full_tree.values)
    assert np.array_equal(reused_leaf, full_leaf)
    assert np.allclose(reused_G, full_G)
    assert np.allclose(reused_H, full_H)


def test_leafwise_no_split_tree_predicts_root_value():
    from chimeraboost.tree import build_leafwise_tree

    Xb = np.array([[0], [1], [2]], dtype=np.uint8)
    grad = np.array([1.0, 2.0, 3.0])
    hess = np.ones_like(grad)
    tree, leaf, leaf_G, leaf_H = build_leafwise_tree(
        Xb, grad, hess, np.array([3], dtype=np.int64), 3, 1.0, 0.1,
        max_leaves=1, return_training_state=True,
    )

    expected = np.full(Xb.shape[0], tree.values[0])
    out = np.zeros(Xb.shape[0])
    tree.add_predict(Xb, out)
    assert tree.n_splits == 0
    assert np.array_equal(tree.apply(Xb), leaf)
    assert np.array_equal(tree.predict(Xb), expected)
    assert np.array_equal(out, expected)
    assert leaf_G[0] == grad.sum()
    assert leaf_H[0] == hess.sum()


def test_leafwise_constant_hessian_reuses_hessian_counts(monkeypatch):
    import numba
    import chimeraboost.tree as tree_mod
    from chimeraboost.preprocessing import FeaturePreprocessor

    rng = np.random.default_rng(21)
    X = rng.normal(size=(700, 9))
    y = 1.3 * X[:, 0] - 0.8 * X[:, 4] + rng.normal(0, 0.3, 700)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))
    row_indices = np.flatnonzero(rng.random(len(y)) < 0.6).astype(np.int64)

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        generic = tree_mod.build_leafwise_tree(
            Xb, grad, hess, prep.n_bins_, 4, 3.0, 0.1,
            max_leaves=8, min_child_samples=5, return_training_state=True,
        )
        generic_rows = tree_mod.build_leafwise_tree(
            Xb, grad, hess, prep.n_bins_, 4, 3.0, 0.1,
            max_leaves=8, min_child_samples=5, row_indices=row_indices,
            return_training_state=True,
        )

        def fail_count_build(*args, **kwargs):
            raise AssertionError("constant-Hessian leafwise path should reuse hh as counts")

        monkeypatch.setattr(tree_mod, "_build_counts_into_serial", fail_count_build)
        monkeypatch.setattr(tree_mod, "_build_counts_rows_into_serial", fail_count_build)

        fast = tree_mod.build_leafwise_tree(
            Xb, grad, hess, prep.n_bins_, 4, 3.0, 0.1,
            max_leaves=8, min_child_samples=5, constant_hessian=True,
            return_training_state=True,
        )
        fast_rows = tree_mod.build_leafwise_tree(
            Xb, grad, hess, prep.n_bins_, 4, 3.0, 0.1,
            max_leaves=8, min_child_samples=5, row_indices=row_indices,
            constant_hessian=True, return_training_state=True,
        )
    finally:
        numba.set_num_threads(old_threads)

    for expected, actual in ((generic, fast), (generic_rows, fast_rows)):
        tree_a, leaf_a, G_a, H_a = expected
        tree_b, leaf_b, G_b, H_b = actual
        assert np.array_equal(tree_a.splits_feat, tree_b.splits_feat)
        assert np.array_equal(tree_a.splits_thr, tree_b.splits_thr)
        assert np.array_equal(tree_a.gains, tree_b.gains)
        assert np.array_equal(tree_a.values, tree_b.values)
        assert np.array_equal(leaf_a, leaf_b)
        assert np.array_equal(G_a, G_b)
        assert np.array_equal(H_a, H_b)


def test_leafwise_selected_rows_features_match_zeroed_nonconstant_histograms():
    import numba
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_leafwise_tree

    rng = np.random.default_rng(22)
    X = rng.normal(size=(700, 11))
    y = 1.2 * X[:, 2] - 0.7 * X[:, 6] + rng.normal(0, 0.3, 700)
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = rng.uniform(0.1, 2.0, size=len(y))
    row_mask = rng.random(len(y)) < 0.55
    row_indices = np.flatnonzero(row_mask).astype(np.int64)
    g = np.where(row_mask, grad, 0.0)
    h = np.where(row_mask, hess, 0.0)
    selected = np.array([2, 4, 6, 9], dtype=np.int64)
    feature_mask = np.zeros(Xb.shape[1], dtype=np.int64)
    feature_mask[selected] = 1

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        zeroed = build_leafwise_tree(
            Xb, g, h, prep.n_bins_, 4, 3.0, 0.1,
            max_leaves=8, min_child_samples=5, feature_mask=feature_mask,
            feature_indices=selected, return_training_state=True,
        )
        indexed = build_leafwise_tree(
            Xb, g, h, prep.n_bins_, 4, 3.0, 0.1,
            max_leaves=8, min_child_samples=5, feature_mask=feature_mask,
            feature_indices=selected, row_indices=row_indices,
            return_training_state=True,
        )
    finally:
        numba.set_num_threads(old_threads)

    zeroed_tree, zeroed_leaf, zeroed_G, zeroed_H = zeroed
    indexed_tree, indexed_leaf, indexed_G, indexed_H = indexed
    assert np.array_equal(zeroed_tree.splits_feat, indexed_tree.splits_feat)
    assert np.array_equal(zeroed_tree.splits_thr, indexed_tree.splits_thr)
    assert np.allclose(zeroed_tree.gains, indexed_tree.gains)
    assert np.array_equal(zeroed_tree.values, indexed_tree.values)
    assert np.array_equal(zeroed_leaf, indexed_leaf)
    assert np.array_equal(zeroed_G, indexed_G)
    assert np.array_equal(zeroed_H, indexed_H)


def test_leafwise_cached_splits_match_full_rescore():
    import numba
    from chimeraboost.preprocessing import FeaturePreprocessor
    from chimeraboost.tree import build_leafwise_tree

    rng = np.random.default_rng(23)
    X = rng.normal(size=(850, 12))
    y = (
        1.7 * X[:, 1]
        - 1.1 * X[:, 5]
        + 0.9 * (X[:, 7] > 0.0)
        + rng.normal(0, 0.35, X.shape[0])
    )
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess_nonconstant = rng.uniform(0.15, 2.5, size=len(y))
    hess_constant = np.ones(len(y))
    row_indices = np.flatnonzero(rng.random(len(y)) < 0.65).astype(np.int64)
    selected = np.array([1, 3, 5, 7, 10], dtype=np.int64)
    feature_mask = np.zeros(Xb.shape[1], dtype=np.int64)
    feature_mask[selected] = 1

    cases = [
        dict(hess=hess_nonconstant, extra={}),
        dict(hess=hess_constant, extra={"constant_hessian": True}),
        dict(
            hess=hess_nonconstant,
            extra={"feature_indices": selected, "feature_mask": feature_mask},
        ),
        dict(
            hess=hess_constant,
            extra={
                "constant_hessian": True,
                "feature_indices": selected,
                "feature_mask": feature_mask,
            },
        ),
        dict(hess=hess_nonconstant, extra={"row_indices": row_indices}),
        dict(
            hess=hess_constant,
            extra={"constant_hessian": True, "row_indices": row_indices},
        ),
        dict(
            hess=hess_nonconstant,
            extra={
                "row_indices": row_indices,
                "feature_indices": selected,
                "feature_mask": feature_mask,
            },
        ),
    ]

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        results = []
        for case in cases:
            common = dict(
                max_leaves=10,
                min_child_samples=6,
                min_child_weight=1.0,
                min_gain_to_split=0.0,
                return_training_state=True,
                **case["extra"],
            )
            cached = build_leafwise_tree(
                Xb, grad, case["hess"], prep.n_bins_, 5, 3.0, 0.1,
                **common,
            )
            full = build_leafwise_tree(
                Xb, grad, case["hess"], prep.n_bins_, 5, 3.0, 0.1,
                recompute_all_leaf_splits=True,
                reuse_leaf_histograms=False,
                **common,
            )
            results.append((cached, full))
    finally:
        numba.set_num_threads(old_threads)

    for cached, full in results:
        cached_tree, cached_leaf, cached_G, cached_H = cached
        full_tree, full_leaf, full_G, full_H = full
        assert np.array_equal(cached_tree.features, full_tree.features)
        assert np.array_equal(cached_tree.thresholds, full_tree.thresholds)
        assert np.array_equal(cached_tree.left_child, full_tree.left_child)
        assert np.array_equal(cached_tree.right_child, full_tree.right_child)
        assert np.array_equal(cached_tree.leaf_index, full_tree.leaf_index)
        assert np.array_equal(cached_tree.splits_feat, full_tree.splits_feat)
        assert np.array_equal(cached_tree.splits_thr, full_tree.splits_thr)
        assert np.allclose(cached_tree.gains, full_tree.gains)
        assert np.array_equal(cached_tree.values, full_tree.values)
        assert np.array_equal(cached_leaf, full_leaf)
        assert np.array_equal(cached_G, full_G)
        assert np.array_equal(cached_H, full_H)
        assert np.array_equal(cached_tree.predict(Xb), full_tree.predict(Xb))


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
    non_oblivious = ChimeraBoostClassifier(
        iterations=12, depth=3, tree_mode="non_oblivious", random_state=0
    ).fit(Xtr, ytr)

    assert catboost.model_.tree_mode_ == "catboost"
    assert oblivious.model_.tree_mode_ == "catboost"
    assert lightgbm.model_.tree_mode_ == "lightgbm"
    assert non_oblivious.model_.tree_mode_ == "depthwise"
    assert catboost.model_.ordered_boosting_ is True
    assert non_oblivious.model_.ordered_boosting_ is True
    assert lightgbm.model_.ordered_boosting_ is False
    assert np.array_equal(catboost.predict_proba(Xte), oblivious.predict_proba(Xte))
    assert lightgbm.predict_proba(Xte).shape == (len(Xte), 2)
    assert abs(lightgbm.feature_importances_.sum() - 1.0) < 1e-6


def test_lightgbm_mode_rejects_ordered_boosting_true():
    X, y = load_breast_cancer(return_X_y=True)
    with pytest.raises(ValueError, match="ordered_boosting=True"):
        ChimeraBoostClassifier(
            iterations=2, tree_mode="lightgbm", ordered_boosting=True
        ).fit(X[:80], y[:80])


def test_tree_mode_default_depth_resolution():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.75, random_state=4, stratify=y
    )

    catboost = ChimeraBoostClassifier(
        iterations=2, tree_mode="catboost", random_state=0
    ).fit(Xtr, ytr)
    depthwise = ChimeraBoostClassifier(
        iterations=2, tree_mode="depthwise", random_state=0
    ).fit(Xtr, ytr)
    lightgbm = ChimeraBoostClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=64, random_state=0
    ).fit(Xtr, ytr)
    explicit = ChimeraBoostClassifier(
        iterations=2, tree_mode="lightgbm", depth=3, num_leaves=64,
        random_state=0
    ).fit(Xtr, ytr)

    assert catboost.model_.depth == 6
    assert depthwise.model_.depth == 6
    assert lightgbm.model_.depth == -1
    assert explicit.model_.depth == 3


def test_lightgbm_mode_adds_category_code_features_for_scalar_tasks():
    X = np.array([
        ["red", "north", 1.0],
        ["blue", "south", 2.0],
        ["red", "south", 3.0],
        ["green", "north", 4.0],
        ["blue", "north", 5.0],
        ["green", "south", 6.0],
    ], dtype=object)
    y_reg = np.array([0.0, 1.0, 0.5, 1.5, 1.2, 1.8])
    y_bin = np.array([0, 1, 0, 1, 1, 0])

    catboost = ChimeraBoostRegressor(
        iterations=1, tree_mode="catboost", random_state=0
    ).fit(X, y_reg, cat_features=[0, 1])
    lightgbm_reg = ChimeraBoostRegressor(
        iterations=1, tree_mode="lightgbm", num_leaves=3, random_state=0
    ).fit(X, y_reg, cat_features=[0, 1])
    lightgbm_binary = ChimeraBoostClassifier(
        iterations=1, tree_mode="lightgbm", num_leaves=3, random_state=0
    ).fit(X, y_bin, cat_features=[0, 1])

    assert catboost.model_.prep_.n_bins_.shape[0] == 3
    assert lightgbm_reg.model_.prep_.n_bins_.shape[0] == 5
    assert lightgbm_binary.model_.prep_.n_bins_.shape[0] == 5
    assert catboost.model_.prep_.target_encoding_mode == "ordered"
    assert lightgbm_reg.model_.prep_.target_encoding_mode == "kfold"
    assert lightgbm_binary.model_.prep_.target_encoding_mode == "kfold"
    assert lightgbm_reg.model_.prep_.cat_smoothing == 3.0
    assert lightgbm_binary.model_.prep_.cat_smoothing == 1.0
    assert np.array_equal(
        lightgbm_reg.model_.prep_.feature_map_, np.array([2, 0, 1, 0, 1])
    )
    assert np.array_equal(
        lightgbm_binary.model_.prep_.feature_map_, np.array([2, 0, 1, 0, 1])
    )


def test_explicit_cat_smoothing_preserved_for_lightgbm_regression():
    X = np.array([
        ["red", 1.0],
        ["blue", 2.0],
        ["red", 3.0],
        ["green", 4.0],
        ["blue", 5.0],
        ["green", 6.0],
    ], dtype=object)
    y = np.array([0.0, 1.0, 0.5, 1.5, 1.2, 1.8])

    model = ChimeraBoostRegressor(
        iterations=1,
        tree_mode="lightgbm",
        num_leaves=3,
        cat_smoothing=2.0,
        random_state=0,
    ).fit(X, y, cat_features=[0])

    assert model.model_.prep_.cat_smoothing == 2.0


def test_public_api_rejects_unsupported_lightgbm_options():
    X, y = load_breast_cancer(return_X_y=True)
    with pytest.raises(ValueError, match="num_leaves"):
        ChimeraBoostClassifier(
            iterations=2, tree_mode="catboost", num_leaves=7
        ).fit(X[:80], y[:80])
    with pytest.raises(ValueError, match="depth"):
        ChimeraBoostClassifier(
            iterations=2, tree_mode="lightgbm", depth=0
        ).fit(X[:80], y[:80])


def test_cat_smoothing_must_be_positive():
    X = np.array([
        ["a", 0.0],
        ["b", 1.0],
        ["a", 2.0],
        ["b", 3.0],
    ], dtype=object)
    y = np.array([0.0, 1.0, 0.5, 1.5])

    with pytest.raises(ValueError, match="cat_smoothing must be positive"):
        ChimeraBoostRegressor(iterations=1, cat_smoothing=0.0).fit(
            X, y, cat_features=[0]
        )


def test_sparse_inputs_raise_clear_error():
    sparse = pytest.importorskip("scipy.sparse")
    X, y = load_breast_cancer(return_X_y=True)
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        ChimeraBoostClassifier(iterations=2).fit(sparse.csr_matrix(X), y)

    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        ChimeraBoostClassifier(iterations=2).fit(
            Xtr, ytr, eval_set=(sparse.csr_matrix(Xv), yv)
        )

    clf = ChimeraBoostClassifier(iterations=2, random_state=0).fit(Xtr, ytr)
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        clf.predict(sparse.csr_matrix(Xv))
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        list(clf.staged_predict_proba(sparse.csr_matrix(Xv)))

    Xr, yr = load_diabetes(return_X_y=True)
    Xr_tr, Xr_v, yr_tr, _ = train_test_split(
        Xr, yr, test_size=0.2, random_state=0
    )
    reg = ChimeraBoostRegressor(iterations=2, random_state=0).fit(Xr_tr, yr_tr)
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        reg.predict(sparse.csr_matrix(Xr_v))
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        list(reg.staged_predict(sparse.csr_matrix(Xr_v)))


def test_lightgbm_mode_enforces_leaf_constraints():
    X, y = load_diabetes(return_X_y=True)
    model = ChimeraBoostRegressor(
        iterations=3, tree_mode="lightgbm", num_leaves=3, depth=2,
        min_child_samples=30, random_state=0
    ).fit(X, y)
    assert model.model_.tree_mode_ == "lightgbm"
    for tree in model.model_.trees_:
        assert tree.n_leaves <= 3
        assert tree.depth <= 2


def test_lightgbm_num_leaves_capped_by_positive_depth():
    X, y = load_diabetes(return_X_y=True)
    model = ChimeraBoostRegressor(
        iterations=2, tree_mode="lightgbm", num_leaves=1000, depth=2,
        min_child_samples=5, random_state=0
    ).fit(X, y)
    assert model.model_._max_tree_leaves() == 4
    for tree in model.model_.trees_:
        assert tree.n_leaves <= 4


def test_lightgbm_scalar_no_split_first_tree_keeps_initial_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    Xv = np.array([[10.0], [11.0]])
    yv = np.array([10.0, 12.0])
    model = ChimeraBoostRegressor(
        iterations=5, tree_mode="lightgbm", num_leaves=1, random_state=0
    ).fit(X, y, eval_set=(Xv, yv))

    assert model.best_iteration_ == 0
    assert model.best_score_ == model.model_.best_score_
    expected_val = model.model_.loss_.eval(
        yv, np.full(len(yv), model.model_.init_)
    )
    assert model.best_score_ == expected_val
    assert np.all(np.isfinite(model.predict(X)))


def test_lightgbm_multiclass_no_split_first_round_keeps_initial_model():
    X = np.array([
        [0.0], [1.0], [2.0], [3.0], [4.0], [5.0],
        [6.0], [7.0], [8.0],
    ])
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    Xv = np.array([[9.0], [10.0], [11.0]])
    yv = np.array([0, 1, 2])
    model = ChimeraBoostClassifier(
        iterations=5, tree_mode="lightgbm", num_leaves=1, random_state=0
    ).fit(X, y, eval_set=(Xv, yv))

    assert model.best_iteration_ == 0
    assert model.best_score_ == model.model_.best_score_
    Yv = np.zeros((3, len(yv)))
    Yv[yv, np.arange(len(yv))] = 1.0
    Fv = np.tile(model.model_.init_[:, None], (1, len(yv)))
    expected_val = model.model_.loss_.eval_class_major(Yv, Fv)
    assert model.best_score_ == expected_val
    proba = model.predict_proba(X)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lightgbm_shared_multiclass_tree_routes_only_categorical():
    X_num = np.array([
        [0.0, 0.1], [0.2, 0.0], [1.0, 0.8], [1.2, 1.1],
        [2.0, 2.2], [2.2, 1.9], [0.1, 0.2], [1.1, 1.0],
        [2.1, 2.0],
    ])
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    numeric = ChimeraBoostClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=3,
        min_child_samples=1, min_child_weight=0.0, random_state=0
    ).fit(X_num, y)
    assert isinstance(numeric.model_.trees_[0], list)

    X_cat = np.empty((len(y), 2), dtype=object)
    X_cat[:, 0] = np.array(
        ["a", "a", "b", "b", "c", "c", "a", "b", "c"], dtype=object
    )
    X_cat[:, 1] = X_num[:, 1]
    categorical = ChimeraBoostClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=3,
        min_child_samples=1, min_child_weight=0.0, random_state=0
    ).fit(X_cat, y, cat_features=[0])
    assert hasattr(categorical.model_.trees_[0], "add_predict_class_major")


def test_lightgbm_zero_weight_rows_do_not_affect_tree_structure():
    from chimeraboost.tree import build_leafwise_tree

    X_active = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]],
        dtype=np.uint8,
    )
    grad_active = np.array([2.0, 1.5, 0.5, -0.5, -1.5, -2.0])
    hess_active = np.ones_like(grad_active)
    X_zero = np.array([[2, 2], [2, 2], [0, 2], [0, 2]], dtype=np.uint8)
    grad_zero = np.zeros(4)
    hess_zero = np.zeros(4)
    n_bins = np.array([3, 3], dtype=np.int64)

    active_tree = build_leafwise_tree(
        X_active, grad_active, hess_active, n_bins, 3, 1.0, 0.1,
        max_leaves=3, min_child_samples=2, min_child_weight=1.0,
    )
    full_tree = build_leafwise_tree(
        np.vstack([X_active, X_zero]),
        np.concatenate([grad_active, grad_zero]),
        np.concatenate([hess_active, hess_zero]),
        n_bins, 3, 1.0, 0.1,
        max_leaves=3, min_child_samples=2, min_child_weight=1.0,
    )

    assert np.array_equal(full_tree.splits_feat, active_tree.splits_feat)
    assert np.array_equal(full_tree.splits_thr, active_tree.splits_thr)
    assert np.array_equal(full_tree.gains, active_tree.gains)
    assert np.array_equal(full_tree.predict(X_active), active_tree.predict(X_active))


def test_partition_last_leaf_keeps_stable_segments():
    from chimeraboost.tree import _partition_leaf_rows

    Xb = np.array([[0], [3], [1], [4], [2], [5]], dtype=np.uint8)
    row_order = np.array([4, 5, 0, 1, 2, 3], dtype=np.int64)
    row_scratch = np.empty_like(row_order)
    leaf = np.array([1, 1, 1, 1, 0, 0], dtype=np.int64)
    leaf_start = np.array([0, 2, 6, 0], dtype=np.int64)

    _partition_leaf_rows(
        Xb, row_order, row_scratch, leaf, leaf_start,
        2, 1, 2, 0, 2
    )

    assert np.array_equal(row_order, np.array([4, 5, 0, 2, 1, 3]))
    assert np.array_equal(leaf, np.array([1, 2, 1, 2, 0, 0]))
    assert np.array_equal(leaf_start[:4], np.array([0, 2, 4, 6]))


def test_partition_middle_leaf_keeps_stable_segments():
    from chimeraboost.tree import _partition_leaf_rows

    Xb = np.array([[0], [0], [0], [5], [1], [0], [0], [0], [0], [0]],
                  dtype=np.uint8)
    row_order = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    row_scratch = np.empty_like(row_order)
    leaf = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3], dtype=np.int64)
    leaf_start = np.array([0, 2, 5, 8, 10, 0], dtype=np.int64)

    _partition_leaf_rows(
        Xb, row_order, row_scratch, leaf, leaf_start,
        4, 1, 4, 0, 2
    )

    assert np.array_equal(row_order, np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 3]))
    assert np.array_equal(leaf, np.array([0, 0, 1, 4, 1, 2, 2, 2, 3, 3]))
    assert np.array_equal(leaf_start[:6], np.array([0, 2, 4, 7, 9, 10]))


def test_unit_hess_histogram_subtraction_matches_generic_hg_hh():
    from chimeraboost.tree import (
        _subtract_right_child_histograms_into_left_serial,
        _subtract_right_child_unit_hess_histograms_into_left_serial,
    )

    rng = np.random.default_rng(46)
    hg = rng.normal(size=(6, 5, 11))
    hh = rng.uniform(0.0, 5.0, size=(6, 5, 11))
    hc = rng.uniform(0.0, 5.0, size=(6, 5, 11))
    unit_hg = hg.copy()
    unit_hh = hh.copy()
    generic_hg = hg.copy()
    generic_hh = hh.copy()
    generic_hc = hc.copy()

    _subtract_right_child_unit_hess_histograms_into_left_serial(
        2, 4, unit_hg, unit_hh
    )
    _subtract_right_child_histograms_into_left_serial(
        2, 4, generic_hg, generic_hh, generic_hc
    )

    assert np.array_equal(unit_hg, generic_hg)
    assert np.array_equal(unit_hh, generic_hh)
    assert np.array_equal(generic_hc[:, 2], hc[:, 2] - hc[:, 4])


def test_fused_unit_hess_refill_subtract_matches_two_step():
    from chimeraboost.tree import (
        _refill_left_subtract_right_unit_hess_into,
        _refill_left_subtract_right_unit_hess_selected_into,
        _refill_leaf_segment_histograms_unit_hess_into,
        _refill_leaf_segment_histograms_unit_hess_selected_into,
        _refill_right_subtract_left_unit_hess_into,
        _refill_right_subtract_left_unit_hess_selected_into,
        _subtract_right_child_unit_hess_histograms_into_left,
        _subtract_right_child_unit_hess_histograms_selected_into_left,
    )

    rng = np.random.default_rng(47)
    Xb = rng.integers(0, 17, size=(90, 8), dtype=np.uint8)
    grad = rng.normal(size=Xb.shape[0])
    row_order = np.arange(Xb.shape[0], dtype=np.int64)
    leaf_start = np.array([0, 20, 55, 90], dtype=np.int64)
    left_leaf = 1
    right_leaf = 2
    leaf_ids = np.array([right_leaf], dtype=np.int64)
    selected = np.array([0, 3, 5, 7], dtype=np.int64)

    for use_selected in (False, True):
        base_hg = rng.normal(size=(Xb.shape[1], 3, 17))
        base_hh = rng.uniform(0.0, 5.0, size=(Xb.shape[1], 3, 17))
        fused_hg = base_hg.copy()
        fused_hh = base_hh.copy()
        ref_hg = base_hg.copy()
        ref_hh = base_hh.copy()

        if use_selected:
            _refill_right_subtract_left_unit_hess_selected_into(
                Xb, grad, row_order, leaf_start, left_leaf, right_leaf,
                selected, fused_hg, fused_hh
            )
            _refill_leaf_segment_histograms_unit_hess_selected_into(
                Xb, grad, row_order, leaf_start, leaf_ids, 1,
                ref_hg, ref_hh, selected
            )
            _subtract_right_child_unit_hess_histograms_selected_into_left(
                left_leaf, right_leaf, selected, ref_hg, ref_hh
            )
            untouched = np.setdiff1d(np.arange(Xb.shape[1]), selected)
            assert np.array_equal(fused_hg[untouched], base_hg[untouched])
            assert np.array_equal(fused_hh[untouched], base_hh[untouched])
        else:
            _refill_right_subtract_left_unit_hess_into(
                Xb, grad, row_order, leaf_start, left_leaf, right_leaf,
                fused_hg, fused_hh
            )
            _refill_leaf_segment_histograms_unit_hess_into(
                Xb, grad, row_order, leaf_start, leaf_ids, 1, ref_hg, ref_hh
            )
            _subtract_right_child_unit_hess_histograms_into_left(
                left_leaf, right_leaf, ref_hg, ref_hh
            )

        assert np.array_equal(fused_hg, ref_hg)
        assert np.array_equal(fused_hh, ref_hh)

        fused_hg = base_hg.copy()
        fused_hh = base_hh.copy()
        ref_hg = base_hg.copy()
        ref_hh = base_hh.copy()
        if use_selected:
            _refill_left_subtract_right_unit_hess_selected_into(
                Xb, grad, row_order, leaf_start, left_leaf, right_leaf,
                selected, fused_hg, fused_hh
            )
            parent_hg = ref_hg[:, left_leaf].copy()
            parent_hh = ref_hh[:, left_leaf].copy()
            _refill_leaf_segment_histograms_unit_hess_selected_into(
                Xb, grad, row_order, leaf_start,
                np.array([left_leaf], dtype=np.int64), 1,
                ref_hg, ref_hh, selected
            )
            ref_hg[selected, right_leaf] = (
                parent_hg[selected] - ref_hg[selected, left_leaf]
            )
            ref_hh[selected, right_leaf] = (
                parent_hh[selected] - ref_hh[selected, left_leaf]
            )
            untouched = np.setdiff1d(np.arange(Xb.shape[1]), selected)
            assert np.array_equal(fused_hg[untouched], base_hg[untouched])
            assert np.array_equal(fused_hh[untouched], base_hh[untouched])
        else:
            _refill_left_subtract_right_unit_hess_into(
                Xb, grad, row_order, leaf_start, left_leaf, right_leaf,
                fused_hg, fused_hh
            )
            parent_hg = ref_hg[:, left_leaf].copy()
            parent_hh = ref_hh[:, left_leaf].copy()
            _refill_leaf_segment_histograms_unit_hess_into(
                Xb, grad, row_order, leaf_start,
                np.array([left_leaf], dtype=np.int64), 1, ref_hg, ref_hh
            )
            ref_hg[:, right_leaf] = parent_hg - ref_hg[:, left_leaf]
            ref_hh[:, right_leaf] = parent_hh - ref_hh[:, left_leaf]

        assert np.array_equal(fused_hg, ref_hg)
        assert np.array_equal(fused_hh, ref_hh)


def test_fused_counts_refill_subtract_matches_two_step():
    from chimeraboost.tree import (
        _build_histograms_counts_into,
        _build_histograms_counts_positive_into,
        _refill_left_subtract_right_counts_into,
        _refill_left_subtract_right_counts_positive_into,
        _refill_left_subtract_right_counts_selected_into,
        _refill_leaf_segment_histograms_counts_into,
        _refill_leaf_segment_histograms_counts_positive_into,
        _refill_leaf_segment_histograms_counts_selected_into,
        _refill_right_subtract_left_counts_into,
        _refill_right_subtract_left_counts_positive_into,
        _refill_right_subtract_left_counts_selected_into,
        _subtract_right_child_histograms_into_left,
        _subtract_right_child_histograms_selected_into_left,
    )

    rng = np.random.default_rng(48)
    Xb = rng.integers(0, 19, size=(96, 9), dtype=np.uint8)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.0, 1.5, size=Xb.shape[0])
    hess[::7] = 0.0
    row_order = np.arange(Xb.shape[0], dtype=np.int64)
    leaf_start = np.array([0, 18, 57, 96], dtype=np.int64)
    left_leaf = 1
    right_leaf = 2
    leaf_ids = np.array([right_leaf], dtype=np.int64)
    selected = np.array([1, 2, 4, 8], dtype=np.int64)

    for use_selected in (False, True):
        base_hg = rng.normal(size=(Xb.shape[1], 3, 19))
        base_hh = rng.uniform(0.0, 5.0, size=(Xb.shape[1], 3, 19))
        base_hc = rng.uniform(0.0, 5.0, size=(Xb.shape[1], 3, 19))
        fused_hg = base_hg.copy()
        fused_hh = base_hh.copy()
        fused_hc = base_hc.copy()
        ref_hg = base_hg.copy()
        ref_hh = base_hh.copy()
        ref_hc = base_hc.copy()

        if use_selected:
            _refill_right_subtract_left_counts_selected_into(
                Xb, grad, hess, row_order, leaf_start, left_leaf,
                right_leaf, selected, fused_hg, fused_hh, fused_hc
            )
            _refill_leaf_segment_histograms_counts_selected_into(
                Xb, grad, hess, row_order, leaf_start, leaf_ids, 1,
                ref_hg, ref_hh, ref_hc, selected
            )
            _subtract_right_child_histograms_selected_into_left(
                left_leaf, right_leaf, selected, ref_hg, ref_hh, ref_hc
            )
            untouched = np.setdiff1d(np.arange(Xb.shape[1]), selected)
            assert np.array_equal(fused_hg[untouched], base_hg[untouched])
            assert np.array_equal(fused_hh[untouched], base_hh[untouched])
            assert np.array_equal(fused_hc[untouched], base_hc[untouched])
        else:
            _refill_right_subtract_left_counts_into(
                Xb, grad, hess, row_order, leaf_start, left_leaf,
                right_leaf, fused_hg, fused_hh, fused_hc
            )
            _refill_leaf_segment_histograms_counts_into(
                Xb, grad, hess, row_order, leaf_start, leaf_ids, 1,
                ref_hg, ref_hh, ref_hc
            )
            _subtract_right_child_histograms_into_left(
                left_leaf, right_leaf, ref_hg, ref_hh, ref_hc
            )

        assert np.array_equal(fused_hg, ref_hg)
        assert np.array_equal(fused_hh, ref_hh)
        assert np.array_equal(fused_hc, ref_hc)

        fused_hg = base_hg.copy()
        fused_hh = base_hh.copy()
        fused_hc = base_hc.copy()
        ref_hg = base_hg.copy()
        ref_hh = base_hh.copy()
        ref_hc = base_hc.copy()
        if use_selected:
            _refill_left_subtract_right_counts_selected_into(
                Xb, grad, hess, row_order, leaf_start, left_leaf,
                right_leaf, selected, fused_hg, fused_hh, fused_hc
            )
            parent_hg = ref_hg[:, left_leaf].copy()
            parent_hh = ref_hh[:, left_leaf].copy()
            parent_hc = ref_hc[:, left_leaf].copy()
            _refill_leaf_segment_histograms_counts_selected_into(
                Xb, grad, hess, row_order, leaf_start,
                np.array([left_leaf], dtype=np.int64), 1,
                ref_hg, ref_hh, ref_hc, selected
            )
            ref_hg[selected, right_leaf] = (
                parent_hg[selected] - ref_hg[selected, left_leaf]
            )
            ref_hh[selected, right_leaf] = (
                parent_hh[selected] - ref_hh[selected, left_leaf]
            )
            ref_hc[selected, right_leaf] = (
                parent_hc[selected] - ref_hc[selected, left_leaf]
            )
            untouched = np.setdiff1d(np.arange(Xb.shape[1]), selected)
            assert np.array_equal(fused_hg[untouched], base_hg[untouched])
            assert np.array_equal(fused_hh[untouched], base_hh[untouched])
            assert np.array_equal(fused_hc[untouched], base_hc[untouched])
        else:
            _refill_left_subtract_right_counts_into(
                Xb, grad, hess, row_order, leaf_start, left_leaf,
                right_leaf, fused_hg, fused_hh, fused_hc
            )
            parent_hg = ref_hg[:, left_leaf].copy()
            parent_hh = ref_hh[:, left_leaf].copy()
            parent_hc = ref_hc[:, left_leaf].copy()
            _refill_leaf_segment_histograms_counts_into(
                Xb, grad, hess, row_order, leaf_start,
                np.array([left_leaf], dtype=np.int64), 1,
                ref_hg, ref_hh, ref_hc
            )
            ref_hg[:, right_leaf] = parent_hg - ref_hg[:, left_leaf]
            ref_hh[:, right_leaf] = parent_hh - ref_hh[:, left_leaf]
            ref_hc[:, right_leaf] = parent_hc - ref_hc[:, left_leaf]

        assert np.array_equal(fused_hg, ref_hg)
        assert np.array_equal(fused_hh, ref_hh)
        assert np.array_equal(fused_hc, ref_hc)

        if not use_selected:
            fused_hg = base_hg.copy()
            fused_hh = base_hh.copy()
            fused_hc = base_hc.copy()
            ref_hg = base_hg.copy()
            ref_hh = base_hh.copy()
            ref_hc = base_hc.copy()
            _refill_left_subtract_right_counts_positive_into(
                Xb, grad, hess, row_order, leaf_start, left_leaf,
                right_leaf, fused_hg, fused_hh, fused_hc
            )
            parent_hg = ref_hg[:, left_leaf].copy()
            parent_hh = ref_hh[:, left_leaf].copy()
            parent_hc = ref_hc[:, left_leaf].copy()
            _refill_leaf_segment_histograms_counts_positive_into(
                Xb, grad, hess, row_order, leaf_start,
                np.array([left_leaf], dtype=np.int64), 1,
                ref_hg, ref_hh, ref_hc
            )
            ref_hg[:, right_leaf] = parent_hg - ref_hg[:, left_leaf]
            ref_hh[:, right_leaf] = parent_hh - ref_hh[:, left_leaf]
            ref_hc[:, right_leaf] = parent_hc - ref_hc[:, left_leaf]
            assert np.array_equal(fused_hg, ref_hg)
            assert np.array_equal(fused_hh, ref_hh)
            assert np.array_equal(fused_hc, ref_hc)


def test_positive_hessian_count_histograms_match_generic():
    from chimeraboost.tree import (
        _build_histograms_counts_into,
        _build_histograms_counts_positive_into,
        _refill_leaf_segment_histograms_counts_into,
        _refill_leaf_segment_histograms_counts_positive_into,
        _refill_right_subtract_left_counts_into,
        _refill_right_subtract_left_counts_positive_into,
    )

    rng = np.random.default_rng(53)
    Xb = rng.integers(0, 21, size=(140, 11), dtype=np.uint8)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.05, 1.6, size=Xb.shape[0])
    leaf = rng.integers(0, 4, size=Xb.shape[0], dtype=np.int64)
    row_order = np.arange(Xb.shape[0], dtype=np.int64)
    leaf_start = np.array([0, 31, 66, 101, 140], dtype=np.int64)
    leaf_ids = np.array([1, 3], dtype=np.int64)

    generic = tuple(np.empty((Xb.shape[1], 4, 21)) for _ in range(3))
    positive = tuple(np.empty((Xb.shape[1], 4, 21)) for _ in range(3))
    _build_histograms_counts_into(Xb, grad, hess, leaf, 4, *generic)
    _build_histograms_counts_positive_into(Xb, grad, hess, leaf, 4, *positive)
    for a, b in zip(generic, positive):
        assert np.array_equal(a, b)

    generic = tuple(rng.normal(size=(Xb.shape[1], 4, 21)) for _ in range(3))
    positive = tuple(arr.copy() for arr in generic)
    _refill_leaf_segment_histograms_counts_into(
        Xb, grad, hess, row_order, leaf_start, leaf_ids, 2, *generic
    )
    _refill_leaf_segment_histograms_counts_positive_into(
        Xb, grad, hess, row_order, leaf_start, leaf_ids, 2, *positive
    )
    for a, b in zip(generic, positive):
        assert np.array_equal(a, b)

    generic = tuple(rng.normal(size=(Xb.shape[1], 4, 21)) for _ in range(3))
    positive = tuple(arr.copy() for arr in generic)
    _refill_right_subtract_left_counts_into(
        Xb, grad, hess, row_order, leaf_start, 1, 3, *generic
    )
    _refill_right_subtract_left_counts_positive_into(
        Xb, grad, hess, row_order, leaf_start, 1, 3, *positive
    )
    for a, b in zip(generic, positive):
        assert np.array_equal(a, b)


def test_leafwise_histogram_subtraction_matches_full_refill():
    from chimeraboost.tree import build_leafwise_tree

    rng = np.random.default_rng(41)
    Xb = rng.integers(0, 16, size=(512, 9), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 16, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.2, 1.8, size=Xb.shape[0])

    for constant_hessian, h in [(False, hess), (True, np.ones_like(hess))]:
        reused = build_leafwise_tree(
            Xb, grad, h, n_bins, 5, 1.0, 0.1,
            max_leaves=12, min_child_samples=4, min_child_weight=0.0,
            min_gain_to_split=0.0, return_training_state=True,
            constant_hessian=constant_hessian,
            reuse_leaf_histograms=True,
        )
        full = build_leafwise_tree(
            Xb, grad, h, n_bins, 5, 1.0, 0.1,
            max_leaves=12, min_child_samples=4, min_child_weight=0.0,
            min_gain_to_split=0.0, return_training_state=True,
            constant_hessian=constant_hessian,
            reuse_leaf_histograms=False,
        )
        reused_tree, reused_leaf, reused_G, reused_H = reused
        full_tree, full_leaf, full_G, full_H = full
        assert np.array_equal(reused_tree.splits_feat, full_tree.splits_feat)
        assert np.array_equal(reused_tree.splits_thr, full_tree.splits_thr)
        assert np.allclose(reused_tree.gains, full_tree.gains)
        assert np.array_equal(reused_leaf, full_leaf)
        assert np.allclose(reused_tree.values, full_tree.values)
        assert np.allclose(reused_G, full_G)
        assert np.allclose(reused_H, full_H)


def test_leafwise_positive_hessian_route_matches_generic_tree():
    from chimeraboost.tree import build_leafwise_tree

    rng = np.random.default_rng(54)
    Xb = rng.integers(0, 64, size=(900, 16), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 64, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.05, 1.5, size=Xb.shape[0])

    generic = build_leafwise_tree(
        Xb, grad, hess, n_bins, 6, 1.2, 0.1,
        max_leaves=14, min_child_samples=5, min_child_weight=0.1,
        min_gain_to_split=0.0, return_training_state=True,
        hessian_always_positive=False,
    )
    positive = build_leafwise_tree(
        Xb, grad, hess, n_bins, 6, 1.2, 0.1,
        max_leaves=14, min_child_samples=5, min_child_weight=0.1,
        min_gain_to_split=0.0, return_training_state=True,
        hessian_always_positive=True,
    )

    generic_tree, generic_leaf, generic_G, generic_H = generic
    positive_tree, positive_leaf, positive_G, positive_H = positive
    assert np.array_equal(positive_tree.features, generic_tree.features)
    assert np.array_equal(positive_tree.thresholds, generic_tree.thresholds)
    assert np.array_equal(positive_tree.left_child, generic_tree.left_child)
    assert np.array_equal(positive_tree.right_child, generic_tree.right_child)
    assert np.array_equal(positive_tree.leaf_index, generic_tree.leaf_index)
    assert np.array_equal(positive_tree.splits_feat, generic_tree.splits_feat)
    assert np.array_equal(positive_tree.splits_thr, generic_tree.splits_thr)
    assert np.allclose(positive_tree.gains, generic_tree.gains)
    assert np.allclose(positive_tree.values, generic_tree.values)
    assert np.array_equal(positive_leaf, generic_leaf)
    assert np.allclose(positive_G, generic_G)
    assert np.allclose(positive_H, generic_H)
    assert np.array_equal(positive_tree.predict(Xb), generic_tree.predict(Xb))


def test_leafwise_selected_feature_histogram_reuse_threaded():
    import numba
    from chimeraboost.tree import build_leafwise_tree

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(43)
    Xb = rng.integers(0, 32, size=(900, 14), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 32, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.2, 1.8, size=Xb.shape[0])
    selected = np.array([1, 2, 5, 8, 13], dtype=np.int64)
    feature_mask = np.zeros(Xb.shape[1], dtype=np.int64)
    feature_mask[selected] = 1

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        reused = build_leafwise_tree(
            Xb, grad, hess, n_bins, 5, 1.0, 0.1,
            feature_mask=feature_mask, feature_indices=selected,
            max_leaves=10, min_child_samples=4, min_child_weight=0.0,
            min_gain_to_split=0.0, return_training_state=True,
            reuse_leaf_histograms=True,
        )
        full = build_leafwise_tree(
            Xb, grad, hess, n_bins, 5, 1.0, 0.1,
            feature_mask=feature_mask, feature_indices=selected,
            max_leaves=10, min_child_samples=4, min_child_weight=0.0,
            min_gain_to_split=0.0, return_training_state=True,
            reuse_leaf_histograms=False,
        )
    finally:
        numba.set_num_threads(old_threads)

    reused_tree, reused_leaf, reused_G, reused_H = reused
    full_tree, full_leaf, full_G, full_H = full
    assert np.array_equal(reused_tree.features, full_tree.features)
    assert np.array_equal(reused_tree.thresholds, full_tree.thresholds)
    assert np.array_equal(reused_tree.left_child, full_tree.left_child)
    assert np.array_equal(reused_tree.right_child, full_tree.right_child)
    assert np.array_equal(reused_tree.leaf_index, full_tree.leaf_index)
    assert np.array_equal(reused_tree.splits_feat, full_tree.splits_feat)
    assert np.array_equal(reused_tree.splits_thr, full_tree.splits_thr)
    assert np.allclose(reused_tree.gains, full_tree.gains)
    assert np.allclose(reused_tree.values, full_tree.values)
    assert np.array_equal(reused_leaf, full_leaf)
    assert np.allclose(reused_G, full_G)
    assert np.allclose(reused_H, full_H)


def test_leafwise_changed_leaf_feature_parallel_split_matches_reference():
    import numba
    from chimeraboost.tree import (
        _best_splits_for_leaf_ids_counts,
        _best_splits_for_leaf_ids_counts_feature_parallel,
    )

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(44)
    n_features = 17
    n_leaves = 7
    max_bins = 13
    n_bins = rng.integers(5, max_bins + 1, size=n_features, dtype=np.int64)
    hg = rng.normal(size=(n_features, n_leaves, max_bins))
    hh = rng.uniform(0.05, 2.0, size=(n_features, n_leaves, max_bins))
    hc = rng.integers(0, 4, size=(n_features, n_leaves, max_bins)).astype(float)
    feature_mask = np.ones(n_features, dtype=np.int64)
    feature_mask[[3, 9]] = 0
    leaf_ids = np.array([1, 4], dtype=np.int64)

    ref_feat = np.full(n_leaves, -99, dtype=np.int64)
    ref_thr = np.full(n_leaves, -99, dtype=np.int64)
    ref_gain = np.full(n_leaves, np.nan)
    new_feat = ref_feat.copy()
    new_thr = ref_thr.copy()
    new_gain = ref_gain.copy()
    feature_gain = np.empty((n_features, n_leaves), dtype=np.float64)
    feature_thr = np.empty((n_features, n_leaves), dtype=np.float64)

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        _best_splits_for_leaf_ids_counts(
            hg, hh, hc, n_bins, 1.7, feature_mask, 0.2, 3,
            leaf_ids, leaf_ids.shape[0], ref_feat, ref_thr, ref_gain
        )
        _best_splits_for_leaf_ids_counts_feature_parallel(
            hg, hh, hc, n_bins, 1.7, feature_mask, 0.2, 3,
            leaf_ids, leaf_ids.shape[0], feature_gain, feature_thr,
            new_feat, new_thr, new_gain
        )
    finally:
        numba.set_num_threads(old_threads)

    assert np.array_equal(new_feat[leaf_ids], ref_feat[leaf_ids])
    assert np.array_equal(new_thr[leaf_ids], ref_thr[leaf_ids])
    assert np.allclose(new_gain[leaf_ids], ref_gain[leaf_ids])


def test_leafwise_full_feature_positive_split_matches_reference():
    from chimeraboost.tree import (
        _best_splits_by_leaf_counts,
        _best_splits_by_leaf_counts_full_features,
        _best_splits_for_leaf_ids_counts,
        _best_splits_for_leaf_ids_counts_full_features,
        _build_histograms_counts_positive_into,
    )

    rng = np.random.default_rng(47)
    n_samples = 900
    n_features = 14
    n_leaves = 7
    max_bins = 13
    n_bins = rng.integers(5, max_bins + 1, size=n_features, dtype=np.int64)
    Xb = np.empty((n_samples, n_features), dtype=np.uint8)
    for f in range(n_features):
        Xb[:, f] = rng.integers(0, n_bins[f], size=n_samples)
    grad = rng.normal(size=n_samples)
    hess = rng.uniform(0.05, 1.0, size=n_samples)
    leaf = rng.integers(0, n_leaves, size=n_samples, dtype=np.int64)
    feature_mask = np.ones(n_features, dtype=np.int64)
    hg = np.zeros((n_features, n_leaves, max_bins), dtype=np.float64)
    hh = np.zeros_like(hg)
    hc = np.zeros_like(hg)
    _build_histograms_counts_positive_into(
        Xb, grad, hess, leaf, n_leaves, hg, hh, hc
    )
    counts = np.bincount(leaf, minlength=n_leaves).astype(np.int64)
    leaf_start = np.zeros(n_leaves + 1, dtype=np.int64)
    leaf_start[1:] = np.cumsum(counts)

    ref_feat = np.full(n_leaves, -99, dtype=np.int64)
    ref_thr = np.full(n_leaves, -99, dtype=np.int64)
    ref_gain = np.full(n_leaves, np.nan)
    new_feat = ref_feat.copy()
    new_thr = ref_thr.copy()
    new_gain = ref_gain.copy()
    _best_splits_by_leaf_counts(
        hg, hh, hc, n_bins, 1.7, feature_mask, 0.2, 3, n_leaves,
        ref_feat, ref_thr, ref_gain
    )
    _best_splits_by_leaf_counts_full_features(
        hg, hh, hc, n_bins, 1.7, 0.2, 3, n_leaves, leaf_start,
        new_feat, new_thr, new_gain
    )

    assert np.array_equal(new_feat, ref_feat)
    assert np.array_equal(new_thr, ref_thr)
    assert np.allclose(new_gain, ref_gain)

    leaf_ids = np.array([1, 4], dtype=np.int64)
    ref_feat[:] = -99
    ref_thr[:] = -99
    ref_gain[:] = np.nan
    new_feat[:] = -99
    new_thr[:] = -99
    new_gain[:] = np.nan
    _best_splits_for_leaf_ids_counts(
        hg, hh, hc, n_bins, 1.7, feature_mask, 0.2, 3,
        leaf_ids, leaf_ids.shape[0], ref_feat, ref_thr, ref_gain
    )
    _best_splits_for_leaf_ids_counts_full_features(
        hg, hh, hc, n_bins, 1.7, 0.2, 3,
        leaf_ids, leaf_ids.shape[0], leaf_start, new_feat, new_thr, new_gain
    )

    assert np.array_equal(new_feat[leaf_ids], ref_feat[leaf_ids])
    assert np.array_equal(new_thr[leaf_ids], ref_thr[leaf_ids])
    assert np.allclose(new_gain[leaf_ids], ref_gain[leaf_ids])


def test_leafwise_positive_split_fast_path_matches_generic_tree():
    import numba
    from chimeraboost.tree import build_leafwise_tree

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(48)
    Xb = rng.integers(0, 32, size=(1000, 18), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 32, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.05, 1.0, size=Xb.shape[0])

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        fast = build_leafwise_tree(
            Xb, grad, hess, n_bins, -1, 1.0, 0.1,
            max_leaves=12, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            hessian_always_positive=True,
        )
        generic = build_leafwise_tree(
            Xb, grad, hess, n_bins, -1, 1.0, 0.1,
            max_leaves=12, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            hessian_always_positive=False,
        )
    finally:
        numba.set_num_threads(old_threads)

    fast_tree, fast_leaf, fast_G, fast_H = fast
    generic_tree, generic_leaf, generic_G, generic_H = generic
    assert np.array_equal(fast_tree.features, generic_tree.features)
    assert np.array_equal(fast_tree.thresholds, generic_tree.thresholds)
    assert np.array_equal(fast_tree.left_child, generic_tree.left_child)
    assert np.array_equal(fast_tree.right_child, generic_tree.right_child)
    assert np.array_equal(fast_tree.leaf_index, generic_tree.leaf_index)
    assert np.array_equal(fast_tree.splits_feat, generic_tree.splits_feat)
    assert np.array_equal(fast_tree.splits_thr, generic_tree.splits_thr)
    assert np.allclose(fast_tree.gains, generic_tree.gains)
    assert np.allclose(fast_tree.values, generic_tree.values)
    assert np.array_equal(fast_leaf, generic_leaf)
    assert np.allclose(fast_G, generic_G)
    assert np.allclose(fast_H, generic_H)


def test_leafwise_threaded_changed_leaf_split_matches_full_rescore():
    import numba
    from chimeraboost.tree import build_leafwise_tree

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(45)
    Xb = rng.integers(0, 48, size=(1100, 19), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 48, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.15, 1.6, size=Xb.shape[0])

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        changed_only = build_leafwise_tree(
            Xb, grad, hess, n_bins, 6, 1.5, 0.1,
            max_leaves=14, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            reuse_leaf_histograms=True,
        )
        full_rescore = build_leafwise_tree(
            Xb, grad, hess, n_bins, 6, 1.5, 0.1,
            max_leaves=14, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            recompute_all_leaf_splits=True,
            reuse_leaf_histograms=False,
        )
    finally:
        numba.set_num_threads(old_threads)

    changed_tree, changed_leaf, changed_G, changed_H = changed_only
    full_tree, full_leaf, full_G, full_H = full_rescore
    assert np.array_equal(changed_tree.features, full_tree.features)
    assert np.array_equal(changed_tree.thresholds, full_tree.thresholds)
    assert np.array_equal(changed_tree.left_child, full_tree.left_child)
    assert np.array_equal(changed_tree.right_child, full_tree.right_child)
    assert np.array_equal(changed_tree.leaf_index, full_tree.leaf_index)
    assert np.array_equal(changed_tree.splits_feat, full_tree.splits_feat)
    assert np.array_equal(changed_tree.splits_thr, full_tree.splits_thr)
    assert np.allclose(changed_tree.gains, full_tree.gains)
    assert np.allclose(changed_tree.values, full_tree.values)
    assert np.array_equal(changed_leaf, full_leaf)
    assert np.allclose(changed_G, full_G)
    assert np.allclose(changed_H, full_H)
    assert np.array_equal(changed_tree.predict(Xb), full_tree.predict(Xb))


def test_lightgbm_thread_determinism():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=2, stratify=y
    )
    one = ChimeraBoostClassifier(
        iterations=8, tree_mode="lightgbm", num_leaves=7, depth=3,
        thread_count=1, random_state=0
    ).fit(Xtr, ytr)
    two = ChimeraBoostClassifier(
        iterations=8, tree_mode="lightgbm", num_leaves=7, depth=3,
        thread_count=2, random_state=0
    ).fit(Xtr, ytr)
    assert np.allclose(one.predict_proba(Xte), two.predict_proba(Xte))


def test_non_oblivious_parallel_add_predict_matches_serial():
    import numba
    from chimeraboost.tree import (
        _predict_non_oblivious_multiclass_tree_add,
        _predict_non_oblivious_multiclass_tree_add_parallel,
        _predict_non_oblivious_tree_add,
        _predict_non_oblivious_tree_add_parallel,
    )

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(51)
    Xb = rng.integers(0, 64, size=(1500, 6), dtype=np.uint8)
    features = np.array([0, 2, -1, -1, 5, -1, -1], dtype=np.int64)
    thresholds = np.array([31, 20, -1, -1, 44, -1, -1], dtype=np.int64)
    left_child = np.array([1, 2, -1, -1, 5, -1, -1], dtype=np.int64)
    right_child = np.array([4, 3, -1, -1, 6, -1, -1], dtype=np.int64)
    leaf_index = np.array([-1, -1, 0, 1, -1, 2, 3], dtype=np.int64)
    values = rng.normal(size=4)
    multi_values = rng.normal(size=(4, 3))
    serial = rng.normal(size=Xb.shape[0])
    parallel = serial.copy()
    serial_multi = rng.normal(size=(3, Xb.shape[0]))
    parallel_multi = serial_multi.copy()

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        _predict_non_oblivious_tree_add(
            Xb, features, thresholds, left_child, right_child, leaf_index,
            values, serial
        )
        _predict_non_oblivious_tree_add_parallel(
            Xb, features, thresholds, left_child, right_child, leaf_index,
            values, parallel
        )
        _predict_non_oblivious_multiclass_tree_add(
            Xb, features, thresholds, left_child, right_child, leaf_index,
            multi_values, serial_multi
        )
        _predict_non_oblivious_multiclass_tree_add_parallel(
            Xb, features, thresholds, left_child, right_child, leaf_index,
            multi_values, parallel_multi
        )
    finally:
        numba.set_num_threads(old_threads)

    assert np.array_equal(parallel, serial)
    assert np.array_equal(parallel_multi, serial_multi)


def test_classifier_staged_predictions_match_final():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    model = ChimeraBoostClassifier(iterations=12, random_state=0).fit(Xtr, ytr)
    stages = list(model.staged_predict_proba(Xte))
    assert len(stages) == model.best_iteration_
    assert np.allclose(stages[-1], model.predict_proba(Xte))


def test_multiclass_staged_predictions_match_final():
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    model = ChimeraBoostClassifier(iterations=8, random_state=0).fit(Xtr, ytr)
    stages = list(model.staged_predict_proba(Xte))
    assert len(stages) == model.best_iteration_
    assert np.allclose(stages[-1], model.predict_proba(Xte))
    assert np.allclose(stages[-1].sum(axis=1), 1.0)


def test_multiclass_subsampling_shared_per_round(monkeypatch):
    import chimeraboost.booster as booster
    from sklearn.datasets import load_wine

    calls = []
    original = booster.build_oblivious_tree

    def wrapped_build_tree(*args, **kwargs):
        row_indices = kwargs.get("row_indices")
        feature_indices = kwargs.get("feature_indices")
        calls.append((
            None if row_indices is None else row_indices.copy(),
            None if feature_indices is None else feature_indices.copy(),
        ))
        return original(*args, **kwargs)

    monkeypatch.setattr(booster, "build_oblivious_tree", wrapped_build_tree)
    X, y = load_wine(return_X_y=True)
    ChimeraBoostClassifier(
        iterations=1, subsample=0.6, colsample=0.5, random_state=0
    ).fit(X, y)

    assert len(calls) == len(np.unique(y))
    first_rows, first_features = calls[0]
    assert first_rows is not None
    assert first_features is not None
    for row_indices, feature_indices in calls[1:]:
        assert np.array_equal(row_indices, first_rows)
        assert np.array_equal(feature_indices, first_features)


def test_lightgbm_numeric_multiclass_training_update_uses_leaf_ids(monkeypatch):
    from chimeraboost import ChimeraBoostClassifier
    from chimeraboost.tree import NonObliviousTree

    rng = np.random.default_rng(58)
    X = rng.normal(size=(120, 6))
    y = np.repeat(np.arange(3), 40)
    order = rng.permutation(len(y))

    def fail_add_predict(self, X_binned, out):
        raise AssertionError("training update should reuse returned leaf ids")

    monkeypatch.setattr(NonObliviousTree, "add_predict", fail_add_predict)
    ChimeraBoostClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=5, depth=3,
        random_state=0
    ).fit(X[order], y[order])


def test_goss_subsample_keeps_large_gradients_and_scales_sampled_rows():
    from chimeraboost.booster import GradientBoosting

    grad = np.array([0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 8.0, -9.0])
    hess = np.ones_like(grad)
    booster = GradientBoosting(
        iterations=1, sampling="goss", top_rate=0.25, other_rate=0.25,
        random_state=0
    )
    g, h, row_indices = booster._maybe_subsample(
        grad, hess, np.random.default_rng(0)
    )

    assert row_indices is not None
    assert set([6, 7]).issubset(set(row_indices.tolist()))
    assert np.count_nonzero(h) == 4
    assert g[6] == grad[6]
    assert g[7] == grad[7]
    assert h[6] == 1.0
    assert h[7] == 1.0
    sampled_small = [i for i in row_indices if i not in {6, 7}]
    assert len(sampled_small) == 2
    assert np.all(h[sampled_small] == 3.0)
    assert np.all(g[sampled_small] == grad[sampled_small] * 3.0)
    unsampled = [i for i in range(grad.shape[0]) if i not in row_indices]
    assert np.all(g[unsampled] == 0.0)
    assert np.all(h[unsampled] == 0.0)


def test_goss_lightgbm_scalar_fit_uses_sampled_nonconstant_hessians(monkeypatch):
    import chimeraboost.booster as booster

    calls = []
    original = booster.build_leafwise_tree

    def wrapped_build_tree(*args, **kwargs):
        calls.append((
            kwargs.get("row_indices"),
            kwargs.get("constant_hessian"),
        ))
        return original(*args, **kwargs)

    monkeypatch.setattr(booster, "build_leafwise_tree", wrapped_build_tree)
    X, y = load_diabetes(return_X_y=True)
    model = ChimeraBoostRegressor(
        iterations=1, tree_mode="lightgbm", num_leaves=7, depth=3,
        sampling="goss", top_rate=0.2, other_rate=0.2, random_state=0
    ).fit(X[:120], y[:120])

    assert np.all(np.isfinite(model.predict(X[:5])))
    assert calls
    row_indices, constant_hessian = calls[0]
    assert row_indices is not None
    assert constant_hessian is False


def test_goss_rejects_uniform_subsample_and_multiclass():
    X, y = load_diabetes(return_X_y=True)
    with pytest.raises(ValueError, match="use subsample=1.0"):
        ChimeraBoostRegressor(
            iterations=1, tree_mode="lightgbm", sampling="goss",
            subsample=0.8, random_state=0
        ).fit(X[:80], y[:80])

    Xc = np.vstack([X[:30], X[30:60], X[60:90]])
    yc = np.repeat([0, 1, 2], 30)
    with pytest.raises(ValueError, match="binary classification and regression"):
        ChimeraBoostClassifier(
            iterations=1, tree_mode="lightgbm", sampling="goss",
            random_state=0
        ).fit(Xc, yc)


def test_multiclass_no_split_class_tree_is_boosting_noop(monkeypatch):
    import chimeraboost.booster as booster

    class FakeTree:
        def __init__(self, depth, value):
            self.depth = depth
            self.values = np.array([value], dtype=np.float64)
            self.splits_feat = np.array([0], dtype=np.int64) if depth else np.array([], dtype=np.int64)
            self.gains = np.array([1.0], dtype=np.float64) if depth else np.array([], dtype=np.float64)

        def add_predict(self, X_binned, out):
            out += self.values[0]

    calls = {"n": 0}

    def fake_build_tree(X_binned, grad, hess, *args, **kwargs):
        k = calls["n"] % 3
        calls["n"] += 1
        tree = FakeTree(0, 5.0) if k == 0 else FakeTree(1, 1.0)
        leaf = np.zeros(X_binned.shape[0], dtype=np.int64)
        return tree, leaf, np.array([0.0]), np.array([1.0])

    monkeypatch.setattr(booster, "build_oblivious_tree", fake_build_tree)
    X = np.arange(18, dtype=np.float64).reshape(9, 2)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    model = ChimeraBoostClassifier(iterations=1, random_state=0).fit(X, y)

    raw = model.model_.predict_raw(X)
    assert calls["n"] == 3
    assert model.model_.trees_[0][0].depth == 0
    assert model.model_.trees_[0][0].values[0] == 0.0
    assert np.allclose(raw[:, 0], model.model_.init_[0])
    assert np.allclose(raw[:, 1], model.model_.init_[1] + 1.0)
    assert np.allclose(raw[:, 2], model.model_.init_[2] + 1.0)


def test_early_stopped_prediction_matches_best_prefix():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=3, stratify=y
    )
    model = ChimeraBoostClassifier(
        iterations=120, early_stopping=True, early_stopping_rounds=5,
        validation_fraction=0.2, tree_mode="lightgbm", num_leaves=7,
        depth=3, random_state=0
    ).fit(Xtr, ytr)
    assert model.best_iteration_ < 120
    stages = list(model.staged_predict_proba(Xte))
    assert len(stages) == model.best_iteration_
    assert np.allclose(stages[-1], model.predict_proba(Xte))


def test_eval_labels_must_be_training_classes():
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    bad_yv = yv.copy()
    bad_yv[0] = 99
    with pytest.raises(ValueError, match="eval_set contains labels"):
        ChimeraBoostClassifier(iterations=2).fit(Xtr, ytr, eval_set=(Xv, bad_yv))


def test_invalid_sample_weights_raise():
    X, y = load_diabetes(return_X_y=True)
    for bad in [
        np.ones((len(y), 1)),
        np.full(len(y), np.nan),
        -np.ones(len(y)),
        np.zeros(len(y)),
    ]:
        with pytest.raises(ValueError):
            ChimeraBoostRegressor(iterations=2).fit(X, y, sample_weight=bad)


def test_invalid_eval_sample_weights_raise():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=0)
    with pytest.raises(ValueError, match="eval_sample_weight"):
        ChimeraBoostRegressor(iterations=2).fit(
            Xtr, ytr, eval_set=(Xv, yv), eval_sample_weight=np.ones(len(yv) + 1)
        )


def test_weighted_validation_changes_early_stopping_path():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = 3.0 * X[:, 0] + rng.normal(0.0, 0.2, 400)
    Xtr, ytr = X[:300], y[:300]
    Xv = np.array([
        [4.0, 0.0, 0.0, 0.0],
        [-4.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
        [0.0, -4.0, 0.0, 0.0],
    ])
    yv = np.array([12.0, -12.0, 50.0, -50.0])

    easy_weight = np.array([50.0, 1.0, 1.0, 1.0])
    hard_weight = np.array([1.0, 1.0, 50.0, 50.0])
    easy = ChimeraBoostRegressor(
        iterations=80, early_stopping=True, early_stopping_rounds=5,
        learning_rate=0.2, depth=2, random_state=0
    ).fit(Xtr, ytr, eval_set=(Xv, yv), eval_sample_weight=easy_weight)
    hard = ChimeraBoostRegressor(
        iterations=80, early_stopping=True, early_stopping_rounds=5,
        learning_rate=0.2, depth=2, random_state=0
    ).fit(Xtr, ytr, eval_set=(Xv, yv), eval_sample_weight=hard_weight)

    assert easy.model_.valid_history_[0] != hard.model_.valid_history_[0]
    assert easy.best_iteration_ != hard.best_iteration_


def test_weighted_categorical_target_encoding_changes_stats():
    from chimeraboost.preprocessing import FeaturePreprocessor

    X = np.array([["a"], ["a"], ["b"], ["b"]], dtype=object)
    y = np.array([0.0, 1.0, 0.0, 10.0])
    prep_unweighted = FeaturePreprocessor(max_bins=8, cat_smoothing=1.0,
                                          random_state=0)
    prep_weighted = FeaturePreprocessor(max_bins=8, cat_smoothing=1.0,
                                        random_state=0)
    prep_unweighted.fit_transform(X, [y], cat_features=[0])
    prep_weighted.fit_transform(
        X, [y], cat_features=[0],
        sample_weight=np.array([1.0, 1.0, 1.0, 10.0])
    )

    # Category b's prediction-time target statistic should move toward 10 when
    # its high-target row is up-weighted.
    unweighted_b = prep_unweighted.encoders_[0].transform(
        np.array([[1]], dtype=np.int64)
    )[0, 0]
    weighted_b = prep_weighted.encoders_[0].transform(
        np.array([[1]], dtype=np.int64)
    )[0, 0]
    assert weighted_b > unweighted_b


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


def test_lightgbm_uniform_weights_equal_no_weights():
    from sklearn.datasets import load_wine

    cases = [
        (ChimeraBoostRegressor, load_diabetes(return_X_y=True), "predict"),
        (ChimeraBoostClassifier, load_breast_cancer(return_X_y=True), "predict_proba"),
        (ChimeraBoostClassifier, load_wine(return_X_y=True), "predict_proba"),
    ]
    for estimator_cls, (X, y), predict_name in cases:
        stratify = None if estimator_cls is ChimeraBoostRegressor else y
        Xtr, Xte, ytr, _ = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=stratify
        )
        kwargs = dict(
            iterations=20, tree_mode="lightgbm", num_leaves=7,
            depth=3, random_state=0
        )
        m_none = estimator_cls(**kwargs).fit(Xtr, ytr)
        m_ones = estimator_cls(**kwargs).fit(
            Xtr, ytr, sample_weight=np.ones(len(ytr))
        )
        pred_none = getattr(m_none, predict_name)(Xte)
        pred_ones = getattr(m_ones, predict_name)(Xte)
        assert np.array_equal(pred_none, pred_ones)


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
