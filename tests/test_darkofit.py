"""Test suite for DarkoFit. Run with: pytest -q"""

import inspect
import warnings

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

from darkofit import DarkoRegressor, DarkoClassifier


def test_descend_leaves_matches_numpy_reference():
    from darkofit.tree import _descend_leaves

    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 6000))
        d = int(rng.integers(0, 6))
        leaf = rng.integers(0, 1 << d, size=n).astype(np.int64)
        Xf = rng.integers(0, 260, size=n).astype(np.uint16)
        threshold = int(rng.integers(-1, 260))

        expected = (leaf << 1) + (Xf > threshold).astype(np.int64)
        actual = leaf.copy()
        _descend_leaves(actual, Xf, threshold)

        assert np.array_equal(actual, expected)


def test_binning_transform_matches_searchsorted_reference():
    from darkofit.binning import Binner, _bin_dtype_for_n_bins

    def reference(binner, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape, dtype=_bin_dtype_for_n_bins(binner.n_bins_))
        for f in range(X.shape[1]):
            col = X[:, f]
            borders = binner.borders_[f]
            binned = np.searchsorted(borders, col, side="right").astype(out.dtype)
            binned[~np.isfinite(col)] = len(borders) + 1
            out[:, f] = binned
        return out

    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 10))
    X[:, 2] = rng.integers(0, 4, size=4000)
    X[:, 5] = 1.0
    X[rng.integers(0, 4000, 300), rng.integers(0, 10, 300)] = np.nan
    X[rng.integers(0, 4000, 100), rng.integers(0, 10, 100)] = np.inf
    X[rng.integers(0, 4000, 100), rng.integers(0, 10, 100)] = -np.inf

    binner = Binner(max_bins=64).fit(X)
    assert np.array_equal(binner.transform(X), reference(binner, X))
    assert np.array_equal(binner.transform(X[:1]), reference(binner, X[:1]))
    assert binner.transform(X[:0]).shape == (0, 10)

    blocks = [X[:, :4], X[:, 4:]]
    block_binner = Binner(max_bins=64).fit_blocks(blocks)
    assert np.array_equal(block_binner.transform_blocks(blocks), reference(block_binner, X))


def test_feature_borders_match_full_unique_reference():
    from darkofit.binning import _feature_borders, _unique_if_at_most

    def reference(col, max_bins):
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            return np.array([], dtype=np.float64)
        uniq = np.unique(finite)
        if uniq.size <= max_bins:
            return ((uniq[:-1] + uniq[1:]) / 2.0).astype(np.float64)
        qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
        return np.unique(np.quantile(finite, qs)).astype(np.float64)

    low_card = np.array([3.0, 1.0, 3.0, np.nan, 2.0, np.inf, 1.0])
    high_card = np.r_[np.linspace(-2.0, 2.0, 2000), [np.nan, np.inf, -np.inf]]

    assert np.array_equal(_feature_borders(low_card, 8), reference(low_card, 8))
    assert np.array_equal(_feature_borders(high_card, 16), reference(high_card, 16))

    def old_unique_reference(values, max_unique):
        seen = set()
        for value in values:
            seen.add(float(value))
            if len(seen) > max_unique:
                return None
        return np.array(sorted(seen), dtype=np.float64)

    cases = [
        (np.array([], dtype=np.float64), 4),
        (np.ones(12, dtype=np.float64), 4),
        (np.array([3.0, 1.0, 3.0, 2.0], dtype=np.float64), 4),
        (np.arange(20, dtype=np.float64), 8),
        (np.arange(5000, dtype=np.float64), 64),
        (np.r_[np.tile([0.0, 1.0], 2048), np.arange(2.0, 80.0)], 8),
        (np.tile([1.0, 2.0, 3.0], 2500), 4),
        (np.array([-0.0, 0.0, 1.0, -0.0], dtype=np.float64), 4),
    ]
    for values, max_unique in cases:
        got = _unique_if_at_most(values, max_unique)
        expected = old_unique_reference(values, max_unique)
        if expected is None:
            assert got is None
        else:
            assert got.dtype == np.float64
            assert np.array_equal(got, expected)

    signed_zero = np.array([-0.0, 0.0, 1.0, 1.0, np.nan, np.inf])
    assert np.array_equal(
        _feature_borders(signed_zero, 8),
        reference(signed_zero, 8),
    )


def test_loss_grad_hess_into_matches_allocating_paths():
    from darkofit.losses import Logloss, MAE, MultiSoftmax, Quantile, RMSE

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
            np.testing.assert_allclose(grad_out, grad, rtol=0.0, atol=1e-15)
            np.testing.assert_allclose(hess_out, hess, rtol=0.0, atol=1e-15)

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
        # NumPy's vectorized softmax and Numba's scalar-loop kernel can differ
        # by a final rounding bit across libm/platform combinations. Require
        # numerical equivalence at machine precision, not cross-backend
        # bitwise identity.
        np.testing.assert_allclose(grad_out, grad, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(hess_out, hess, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5, np.nan, np.inf])
def test_quantile_alpha_must_be_finite_probability(alpha):
    from darkofit.losses import Quantile

    with pytest.raises(ValueError, match="alpha"):
        Quantile(alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        DarkoRegressor(
            iterations=1, loss="Quantile", alpha=alpha
        ).fit(
            np.arange(12, dtype=float).reshape(6, 2),
            np.arange(6, dtype=float),
            sample_weight=np.ones(6),
        )


def test_classification_grad_hess_into_extreme_values_match_allocating_paths():
    from darkofit.losses import Logloss, MultiSoftmax

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
    from darkofit.losses import Logloss

    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    raw = np.array([-1000.0, -60.0, -1.5, 4.0, 1000.0])
    weights = np.array([0.5, 2.0, 0.25, 3.0, 1.5])
    loss = Logloss()

    p = 1.0 / (1.0 + np.exp(-np.clip(raw, -700.0, 700.0)))
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    ce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    assert np.isclose(loss.eval(y, raw), np.average(ce))
    assert np.isclose(loss.eval(y, raw, weights), np.average(ce, weights=weights))


def test_multisoftmax_eval_class_major_matches_clipped_probability_formula():
    from darkofit.losses import MultiSoftmax

    rng = np.random.default_rng(79)
    K = 4
    n = 17
    labels = rng.integers(0, K, size=n)
    Y = np.zeros((K, n))
    Y[labels, np.arange(n)] = 1.0
    F = rng.normal(size=(K, n)) * 3.0
    F[:, 0] = np.array([800.0, 799.0, 798.0, 797.0])
    F[:, 1] = np.array([-800.0, -799.0, -798.0, -797.0])
    weights = rng.uniform(0.2, 2.0, size=n)

    z = F - F.max(axis=0, keepdims=True)
    P = np.exp(z)
    P /= P.sum(axis=0, keepdims=True)
    P = np.clip(P, 1e-12, 1.0)
    ce = -np.sum(Y * np.log(P), axis=0)

    loss = MultiSoftmax(K)
    assert np.isclose(loss.eval_class_major(Y, F), np.average(ce))
    assert np.isclose(loss.eval_class_major(Y, F, weights), np.average(ce, weights=weights))
    assert np.isclose(loss.eval_class_major_labels(labels, F), np.average(ce))
    assert np.isclose(
        loss.eval_class_major_labels(labels, F, weights),
        np.average(ce, weights=weights),
    )


def test_binner_uses_smallest_safe_unsigned_dtype():
    from darkofit.binning import Binner

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
    m = DarkoRegressor(iterations=300, random_state=0).fit(Xtr, ytr)
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
    m = DarkoClassifier(iterations=300, random_state=0).fit(Xtr, ytr)
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
    m = DarkoClassifier(iterations=200, random_state=1)
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
    m = DarkoClassifier(
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
    m = DarkoClassifier(iterations=80, random_state=0)
    m.fit(X, y, cat_features=[0])
    Xnew = np.array([["c_UNSEEN", np.nan, 0.5], ["c3", 1.0, -0.5]], dtype=object)
    p = m.predict_proba(Xnew)
    assert p.shape == (2, 2)
    assert np.all((p >= 0) & (p <= 1))


def test_categorical_transform_preserves_missing_and_unseen_codes():
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.target_encoding import _MISSING_CATEGORY, factorize

    raw = np.array(["b", "a", "b", None, np.nan, "__nan__"], dtype=object)
    codes, categories = factorize(raw)
    cat_to_code = {v: i for i, v in enumerate(categories)}
    assert codes[3] == cat_to_code[_MISSING_CATEGORY]
    assert codes[4] == cat_to_code[_MISSING_CATEGORY]
    assert codes[5] == cat_to_code["__nan__"]
    assert codes[5] != codes[3]

    X = np.array([
        ["red", 1.0],
        ["blue", 2.0],
        ["red", 3.0],
        ["__nan__", 4.0],
        [None, 5.0],
    ], dtype=object)
    prep = FeaturePreprocessor(16, 1.0, 0)
    prep.fit_transform(
        X, [np.array([0.0, 1.0, 0.0, 1.0, 0.5])], cat_features=[0]
    )

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
        prep.cat_maps_[0][_MISSING_CATEGORY],
        prep.cat_maps_[0][_MISSING_CATEGORY],
        prep.cat_maps_[0]["__nan__"],
    ])
    assert np.array_equal(transformed, expected)

    X_num = np.array([[1.0], [2.0], [1.0], [3.0]], dtype=object)
    prep_num = FeaturePreprocessor(16, 1.0, 0)
    prep_num.fit_transform(X_num, [np.array([0.0, 1.0, 0.0, 1.0])], [0])
    Xt_num = np.array([[2.0], [4.0], [np.nan], [None]], dtype=object)
    expected_num = np.array([prep_num.cat_maps_[0][2.0], -1, -1, -1])
    assert np.array_equal(prep_num._codes_for_transform(Xt_num)[:, 0], expected_num)


def test_factorize_treats_numpy_float_nan_as_missing_without_pandas_fast_path(monkeypatch):
    import darkofit.target_encoding as target_encoding

    monkeypatch.setitem(target_encoding.sys.modules, "pandas", None)
    raw = np.array([np.float32("nan"), np.float64("nan"), 1.0], dtype=object)
    codes, categories = target_encoding.factorize(raw)
    cat_to_code = {v: i for i, v in enumerate(categories)}

    assert target_encoding._MISSING_CATEGORY in cat_to_code
    assert codes[0] == cat_to_code[target_encoding._MISSING_CATEGORY]
    assert codes[1] == cat_to_code[target_encoding._MISSING_CATEGORY]
    assert codes[2] != cat_to_code[target_encoding._MISSING_CATEGORY]


def test_preprocessor_can_include_raw_category_code_features():
    from darkofit.preprocessing import FeaturePreprocessor

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
    Xt_binned = with_codes.transform(Xt)
    assert Xt_binned.shape[1] == 5
    assert Xt_binned[1, 1] == with_codes.n_bins_[1] - 1
    assert Xt_binned[1, 2] != with_codes.n_bins_[2] - 1


def test_preprocessor_transform_rejects_truncated_categorical_input():
    from darkofit.preprocessing import FeaturePreprocessor

    X = np.array([
        ["red", 1.0, 2.0, 3.0],
        ["blue", 2.0, 3.0, 4.0],
        ["red", 3.0, 4.0, 5.0],
        ["green", 4.0, 5.0, 6.0],
    ], dtype=object)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    prep = FeaturePreprocessor(16, 1.0, 0)
    prep.fit_transform(X, [y], cat_features=[0])

    with pytest.raises(ValueError, match="fitted preprocessor expects 4"):
        prep.transform(X[:, 1:])


def test_kfold_target_encoding_uses_out_of_fold_totals():
    from darkofit.target_encoding import OrderedTargetEncoder

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
    from darkofit.target_encoding import _MISSING_CATEGORY, factorize

    raw = np.array(["b", "a", "b", None, np.nan, "__nan__"], dtype=object)
    codes, categories = factorize(raw)
    cat_to_code = {v: i for i, v in enumerate(categories)}

    assert codes[0] == codes[2]
    assert codes[1] != codes[0]
    assert codes[3] == cat_to_code[_MISSING_CATEGORY]
    assert codes[4] == cat_to_code[_MISSING_CATEGORY]
    assert codes[5] == cat_to_code["__nan__"]
    assert codes[5] != codes[3]


def test_explicit_lr_overrides_auto():
    X, y = load_diabetes(return_X_y=True)
    m = DarkoRegressor(iterations=50, learning_rate=0.123).fit(X, y)
    assert m.model_.lr_ == 0.123


def test_verbose_timing_records_regression_fit_phases():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=0)
    m = DarkoRegressor(
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
    m = DarkoClassifier(
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
        m = DarkoClassifier(iterations=200, random_state=0).fit(Xtr, ytr)
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
    m = DarkoClassifier(iterations=150, random_state=1)
    m.fit(Xtr, ytr, cat_features=[0])
    assert set(m.classes_) == {"low", "mid", "high"}
    assert set(np.unique(m.predict(Xte))).issubset({"low", "mid", "high"})


def test_multiclass_tree_builder_receives_class_column_views(monkeypatch):
    """Multiclass gradients should be laid out so per-class slices avoid copies."""
    import darkofit.booster as booster
    from sklearn.datasets import load_wine

    seen = []
    original = booster.build_oblivious_tree

    def wrapped_build_tree(X_binned, grad, hess, *args, **kwargs):
        seen.append((grad.flags.c_contiguous, hess.flags.c_contiguous,
                     grad.flags.owndata, hess.flags.owndata))
        return original(X_binned, grad, hess, *args, **kwargs)

    monkeypatch.setattr(booster, "build_oblivious_tree", wrapped_build_tree)
    X, y = load_wine(return_X_y=True)
    DarkoClassifier(iterations=2, random_state=0).fit(X, y)

    assert seen
    assert all(g_contig and h_contig for g_contig, h_contig, _, _ in seen)
    assert not any(g_own or h_own for _, _, g_own, h_own in seen)


def test_multiclass_preprocessor_receives_class_major_target_views(monkeypatch):
    """Per-class target-stat targets should be row views of one class-major Y."""
    import darkofit.booster as booster
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
    DarkoClassifier(iterations=1, random_state=0).fit(X, y)

    assert seen
    assert all(contiguous and not owns_data for contiguous, owns_data in seen[0])


def test_preprocessing_cache_reduces_auto_probe_fit_transform_count(monkeypatch):
    import darkofit.booster as booster_mod

    calls = []
    original = booster_mod.FeaturePreprocessor.fit_transform

    def wrapped_fit_transform(self, X, encode_targets, cat_features,
                              sample_weight=None):
        calls.append((
            self.target_encoding_mode,
            bool(self.include_cat_codes),
            tuple([] if cat_features is None else cat_features),
            len(encode_targets),
        ))
        return original(
            self, X, encode_targets, cat_features, sample_weight=sample_weight
        )

    monkeypatch.setattr(
        booster_mod.FeaturePreprocessor, "fit_transform", wrapped_fit_transform
    )
    rng = np.random.default_rng(20260709)
    X = np.column_stack([
        rng.choice(np.array(["red", "blue", "green"], dtype=object), size=90),
        rng.normal(size=90),
    ]).astype(object)
    y = np.asarray(X[:, 1], dtype=np.float64) + rng.normal(0.0, 0.1, size=90)

    model = DarkoRegressor(
        iterations=6,
        tree_mode="auto",
        early_stopping=True,
        validation_fraction=0.25,
        auto_learning_rate_probe=True,
        auto_learning_rate_probe_iterations=2,
        random_state=0,
    ).fit(X, y, cat_features=[0])

    assert np.all(np.isfinite(model.predict(X[:5])))
    assert sorted(calls) == [
        ("kfold", True, (0,), 1),
        ("ordered", False, (0,), 1),
    ]


def test_preprocessing_cache_key_separates_data_targets_and_weights(monkeypatch):
    import darkofit.booster as booster_mod

    calls = 0
    original = booster_mod.FeaturePreprocessor.fit_transform

    def wrapped_fit_transform(self, X, encode_targets, cat_features,
                              sample_weight=None):
        nonlocal calls
        calls += 1
        return original(
            self, X, encode_targets, cat_features, sample_weight=sample_weight
        )

    monkeypatch.setattr(
        booster_mod.FeaturePreprocessor, "fit_transform", wrapped_fit_transform
    )

    cache = {}
    X = np.array([
        ["a", 0.0], ["b", 1.0], ["a", 2.0], ["c", 3.0],
        ["b", 4.0], ["c", 5.0], ["a", 6.0], ["b", 7.0],
    ], dtype=object)
    y = np.linspace(0.0, 1.0, X.shape[0])
    eval_X = X[:3].copy()
    eval_y = y[:3].copy()

    def fit_cached(X_fit, y_fit, *, weights=None, eval_pair=(eval_X, eval_y)):
        model = booster_mod.GradientBoosting(
            iterations=0, tree_mode="lightgbm", random_state=0
        )
        model._preprocessing_cache = cache
        model.fit(
            X_fit, y_fit, cat_features=[0],
            sample_weight=weights,
            eval_set=eval_pair,
        )
        return model

    first = fit_cached(X, y)
    second = fit_cached(X.copy(), y.copy())
    assert calls == 1
    assert np.array_equal(first.predict_raw(X), second.predict_raw(X))
    unseen = np.array([["new-category", 1.5]], dtype=object)
    assert np.all(np.isfinite(second.predict_raw(unseen)))

    X_changed = X.copy()
    X_changed[0, 0] = "z"
    fit_cached(X_changed, y)
    assert calls == 2

    fit_cached(X, y + 0.5)
    assert calls == 3

    fit_cached(X, y, weights=np.linspace(0.5, 1.5, X.shape[0]))
    assert calls == 4

    # The cached artifacts never depend on the eval set, so a different eval
    # set (or none at all, as in a same-data refit) reuses the entry.
    eval_changed = eval_X.copy()
    eval_changed[0, 0] = "unseen"
    reused = fit_cached(X, y, eval_pair=(eval_changed, eval_y))
    assert calls == 4
    assert np.array_equal(first.predict_raw(X), reused.predict_raw(X))
    fit_cached(X, y, eval_pair=None)
    assert calls == 4


def test_refit_preprocessing_cache_gating(monkeypatch):
    """Refit enables the preprocessing cache only when reuse is possible
    (explicit eval set, same training rows), and empties it after fit."""
    import darkofit.booster as booster_mod

    calls = 0
    original = booster_mod.FeaturePreprocessor.fit_transform

    def wrapped_fit_transform(self, X, encode_targets, cat_features,
                              sample_weight=None):
        nonlocal calls
        calls += 1
        return original(
            self, X, encode_targets, cat_features, sample_weight=sample_weight
        )

    monkeypatch.setattr(
        booster_mod.FeaturePreprocessor, "fit_transform", wrapped_fit_transform
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = X[:, 0] + rng.normal(size=400)
    Xv = rng.normal(size=(80, 5))
    yv = Xv[:, 0] + rng.normal(size=80)

    explicit = DarkoRegressor(
        iterations=20, early_stopping=True, refit=True, random_state=0
    ).fit(X, y, eval_set=(Xv, yv))
    assert calls == 1  # refit reused the selection fit's preprocessing
    assert explicit.model_._preprocessing_cache == {}  # emptied after fit

    calls = 0
    auto_split = DarkoRegressor(
        iterations=20, early_stopping=True, refit=True, random_state=0
    ).fit(X, y)
    # Selection trains on a subset and refit on all rows: reuse is
    # impossible, so no cache is attached and both fits preprocess.
    assert calls == 2
    assert getattr(auto_split.model_, "_preprocessing_cache", None) is None

    y_class = (y > np.median(y)).astype(np.int64)
    yv_class = (yv > np.median(y)).astype(np.int64)

    calls = 0
    explicit_classifier = DarkoClassifier(
        iterations=20, early_stopping=True, refit=True, random_state=0
    ).fit(X, y_class, eval_set=(Xv, yv_class))
    assert calls == 1
    assert explicit_classifier.model_._preprocessing_cache == {}

    calls = 0
    auto_classifier = DarkoClassifier(
        iterations=20, early_stopping=True, refit=True, random_state=0
    ).fit(X, y_class)
    assert calls == 2
    assert getattr(auto_classifier.model_, "_preprocessing_cache", None) is None


def test_preprocessing_cache_does_not_share_scalar_and_multiclass(monkeypatch):
    import darkofit.booster as booster_mod

    calls = 0
    original = booster_mod.FeaturePreprocessor.fit_transform

    def wrapped_fit_transform(self, X, encode_targets, cat_features,
                              sample_weight=None):
        nonlocal calls
        calls += 1
        return original(
            self, X, encode_targets, cat_features, sample_weight=sample_weight
        )

    monkeypatch.setattr(
        booster_mod.FeaturePreprocessor, "fit_transform", wrapped_fit_transform
    )
    cache = {}
    X = np.array([
        ["a", 0.0], ["b", 1.0], ["c", 2.0], ["a", 3.0],
        ["b", 4.0], ["c", 5.0], ["a", 6.0], ["b", 7.0],
        ["c", 8.0],
    ], dtype=object)
    y_binary = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
    y_multi = np.array(["x", "y", "z"] * 3, dtype=object)

    scalar = booster_mod.GradientBoosting(
        iterations=0, loss="Logloss", tree_mode="lightgbm", random_state=0
    )
    scalar._preprocessing_cache = cache
    scalar.fit(X, y_binary, cat_features=[0])
    multiclass = booster_mod.MulticlassBoosting(
        iterations=0, tree_mode="lightgbm", random_state=0
    )
    multiclass._preprocessing_cache = cache
    multiclass.fit(X, y_multi, cat_features=[0])

    assert calls == 2
    assert (
        scalar.prep_.feature_map_.shape[0]
        != multiclass.prep_.feature_map_.shape[0]
    )


def test_wrapper_new_options_preserve_existing_positional_tail():
    for estimator in (DarkoRegressor, DarkoClassifier):
        params = list(inspect.signature(estimator.__init__).parameters)
        auto_idx = params.index("auto_learning_rate_probe")
        assert auto_idx < params.index("histogram_dtype")
        assert auto_idx < params.index("leaf_dtype")
        assert auto_idx < params.index("ts_permutations")
        assert auto_idx < params.index("target_ordered_cat_codes")


def test_histogram_dtype_validation_and_multiclass_scope():
    import darkofit.booster as booster_mod

    assert booster_mod.GradientBoosting(
        histogram_dtype=np.float32
    ).histogram_dtype == "float32"
    with pytest.raises(ValueError, match="histogram_dtype"):
        booster_mod.GradientBoosting(histogram_dtype="float16")

    X = np.arange(18, dtype=np.float64).reshape(9, 2)
    y = np.array([0, 1, 2] * 3)
    with pytest.raises(ValueError, match="scalar GradientBoosting"):
        booster_mod.MulticlassBoosting(
            iterations=1, histogram_dtype="float32", random_state=0
        ).fit(X, y)


def test_float32_histogram_streams_preserve_sampler_inputs(monkeypatch):
    import darkofit.booster as booster_mod

    rng = np.random.default_rng(20260710)
    X = rng.normal(size=(220, 4))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0.0, 0.2, size=220)
    original = booster_mod.GradientBoosting._maybe_subsample
    captured = {}

    def wrapped_maybe_subsample(self, grad, hess, rng_):
        assert grad.dtype == np.float64
        assert hess.dtype == np.float64
        g, h, row_indices = original(self, grad, hess, rng_)
        captured.setdefault(self.histogram_dtype_, []).append(
            None if row_indices is None else row_indices.copy()
        )
        return g, h, row_indices

    monkeypatch.setattr(
        booster_mod.GradientBoosting, "_maybe_subsample", wrapped_maybe_subsample
    )

    for dtype in ("float64", "float32"):
        model = booster_mod.GradientBoosting(
            iterations=1,
            tree_mode="lightgbm",
            sampling="mvs",
            subsample=0.45,
            histogram_dtype=dtype,
            random_state=17,
        )
        model.fit(X, y)

    assert len(captured["float64"]) == len(captured["float32"]) == 1
    assert np.array_equal(captured["float64"][0], captured["float32"][0])


def test_float32_histogram_streams_keep_well_separated_splits_structural():
    import darkofit.booster as booster_mod

    rng = np.random.default_rng(20260711)
    X = rng.normal(size=(260, 5))
    X[:, 0] = np.r_[np.full(130, -3.0), np.full(130, 3.0)]
    y = np.r_[np.full(130, -8.0), np.full(130, 8.0)]

    def fit(dtype, mode):
        model = booster_mod.GradientBoosting(
            iterations=1,
            depth=2,
            tree_mode=mode,
            min_child_samples=1,
            min_child_weight=0.0,
            histogram_dtype=dtype,
            random_state=0,
        )
        model.fit(X, y)
        return model.trees_[0], model.predict_raw(X)

    for mode in ("catboost", "lightgbm", "hybrid"):
        tree64, pred64 = fit("float64", mode)
        tree32, pred32 = fit("float32", mode)
        assert tree64.depth == tree32.depth
        for attr in (
            "splits_feat", "splits_thr", "gains", "features", "thresholds",
            "left_child", "right_child",
        ):
            if hasattr(tree64, attr):
                assert np.array_equal(getattr(tree64, attr), getattr(tree32, attr))
        assert np.allclose(pred64, pred32, rtol=0.0, atol=1e-6)


def test_float32_histogram_streams_are_deterministic_for_same_seed():
    rng = np.random.default_rng(20260712)
    X = rng.normal(size=(180, 6))
    y = X[:, 0] * 2.0 - X[:, 2] + rng.normal(0.0, 0.15, size=180)

    def fit_once():
        model = DarkoRegressor(
            iterations=12,
            learning_rate=0.08,
            depth=3,
            tree_mode="lightgbm",
            histogram_dtype="float32",
            thread_count=2,
            random_state=7,
        ).fit(X, y)
        return model

    first = fit_once()
    second = fit_once()

    assert np.array_equal(first.predict(X), second.predict(X))
    assert np.array_equal(first.model_.train_history_, second.model_.train_history_)


def test_float32_histogram_streams_real_data_metric_parity():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=20260712
    )

    def fit(dtype):
        model = DarkoRegressor(
            iterations=35,
            learning_rate=0.07,
            depth=3,
            tree_mode="lightgbm",
            histogram_dtype=dtype,
            random_state=3,
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    pred64 = fit("float64")
    pred32 = fit("float32")
    rmse64 = np.sqrt(mean_squared_error(y_test, pred64))
    rmse32 = np.sqrt(mean_squared_error(y_test, pred32))

    assert abs(rmse32 - rmse64) <= 0.02
    assert np.max(np.abs(pred32 - pred64)) <= 0.25


def test_float32_histogram_dtype_persists_through_save_load(tmp_path):
    X, y = load_diabetes(return_X_y=True)
    model = DarkoRegressor(
        iterations=5,
        tree_mode="lightgbm",
        histogram_dtype="float32",
        random_state=0,
    ).fit(X[:80], y[:80])

    path = tmp_path / "float32_histogram_streams.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)

    assert loaded.model_.histogram_dtype == "float32"
    assert loaded.model_.histogram_dtype_ == "float32"
    assert np.array_equal(model.predict(X[:10]), loaded.predict(X[:10]))


def _assert_tree_structure_equal(a, b):
    for attr in (
        "splits_feat", "splits_thr", "gains", "features", "thresholds",
        "left_child", "right_child", "leaf_index", "node_features",
        "node_thresholds", "splits_feat", "splits_thr", "values",
    ):
        if hasattr(a, attr):
            assert np.array_equal(getattr(a, attr), getattr(b, attr)), attr
    assert a.depth == b.depth


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        ("build_oblivious_tree", {}),
        ("build_levelwise_tree", {}),
        ("build_leafwise_tree", {"max_leaves": 7}),
        ("build_hybrid_tree", {"max_leaves": 7}),
    ],
)
def test_uint32_leaf_dtype_matches_int64_scalar_builders(builder_name, kwargs):
    import darkofit.tree as tree_mod

    rng = np.random.default_rng(20260717)
    X_binned = rng.integers(0, 31, size=(360, 6), dtype=np.uint8)
    grad = rng.normal(size=X_binned.shape[0])
    hess = rng.uniform(0.1, 2.0, size=X_binned.shape[0])
    n_bins = np.full(X_binned.shape[1], 31, dtype=np.int64)
    builder = getattr(tree_mod, builder_name)
    builder_kwargs = dict(kwargs)
    if builder_name in {"build_leafwise_tree", "build_hybrid_tree"}:
        builder_kwargs["min_child_samples"] = 3

    int_tree, int_leaf, int_G, int_H = builder(
        X_binned, grad, hess, n_bins, 3, 1.5, 0.08,
        return_training_state=True,
        min_child_weight=0.05,
        leaf_dtype="int64",
        **builder_kwargs,
    )
    uint_tree, uint_leaf, uint_G, uint_H = builder(
        X_binned, grad, hess, n_bins, 3, 1.5, 0.08,
        return_training_state=True,
        min_child_weight=0.05,
        leaf_dtype=np.uint32,
        **builder_kwargs,
    )

    assert uint_leaf.dtype == np.uint32
    _assert_tree_structure_equal(int_tree, uint_tree)
    assert np.array_equal(int_leaf, uint_leaf)
    assert np.array_equal(int_G, uint_G)
    assert np.array_equal(int_H, uint_H)
    assert np.array_equal(int_tree.predict(X_binned), uint_tree.predict(X_binned))
    assert np.array_equal(
        uint_G, np.bincount(uint_leaf, weights=grad, minlength=len(uint_tree.values))
    )


def test_uint32_leaf_dtype_matches_int64_multiclass_builder():
    from darkofit.tree import build_leafwise_multiclass_tree

    rng = np.random.default_rng(20260718)
    X_binned = rng.integers(0, 23, size=(280, 5), dtype=np.uint8)
    grad = rng.normal(size=(4, X_binned.shape[0]))
    hess = rng.uniform(0.1, 1.6, size=grad.shape)
    n_bins = np.full(X_binned.shape[1], 23, dtype=np.int64)

    int_tree, int_leaf, int_G, int_H = build_leafwise_multiclass_tree(
        X_binned, grad, hess, n_bins, 4, 1.2, 0.1,
        max_leaves=8, min_child_samples=3, min_child_weight=0.1,
        return_training_state=True, leaf_dtype="int64",
    )
    uint_tree, uint_leaf, uint_G, uint_H = build_leafwise_multiclass_tree(
        X_binned, grad, hess, n_bins, 4, 1.2, 0.1,
        max_leaves=8, min_child_samples=3, min_child_weight=0.1,
        return_training_state=True, leaf_dtype="uint32",
    )

    assert uint_leaf.dtype == np.uint32
    _assert_tree_structure_equal(int_tree, uint_tree)
    assert np.array_equal(int_leaf, uint_leaf)
    assert np.array_equal(int_G, uint_G)
    assert np.array_equal(int_H, uint_H)
    int_pred = np.zeros_like(grad)
    uint_pred = np.zeros_like(grad)
    int_tree.add_predict_class_major(X_binned, int_pred)
    uint_tree.add_predict_class_major(X_binned, uint_pred)
    assert np.array_equal(int_pred, uint_pred)


def test_uint32_leaf_dtype_public_fit_and_save_load(tmp_path):
    import darkofit.booster as booster_mod

    with pytest.raises(ValueError, match="leaf_dtype"):
        booster_mod.GradientBoosting(leaf_dtype="uint16")

    X, y = load_diabetes(return_X_y=True)
    model = DarkoRegressor(
        iterations=8,
        tree_mode="lightgbm",
        histogram_dtype="float32",
        leaf_dtype="uint32",
        random_state=0,
    ).fit(X[:120], y[:120])

    path = tmp_path / "uint32_leaf_dtype.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    assert loaded.model_.leaf_dtype == "uint32"
    assert loaded.model_.leaf_dtype_ == "uint32"
    assert np.array_equal(model.predict(X[:12]), loaded.predict(X[:12]))

    Xc, yc = load_breast_cancer(return_X_y=True)
    ordered = DarkoClassifier(
        iterations=3,
        tree_mode="catboost",
        ordered_boosting=True,
        leaf_dtype="uint32",
        random_state=1,
    ).fit(Xc[:80], yc[:80])
    assert np.all(np.isfinite(ordered.predict_proba(Xc[:10])))


def test_multiclass_eval_set_uses_label_indices_without_one_hot(monkeypatch):
    """Validation loss should not allocate a dense eval one-hot matrix."""
    import darkofit.booster as booster

    calls = []
    original = booster._one_hot_class_major

    def wrapped_one_hot(y_idx, n_classes):
        calls.append((len(y_idx), int(n_classes)))
        return original(y_idx, n_classes)

    monkeypatch.setattr(booster, "_one_hot_class_major", wrapped_one_hot)
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.1],
        [2.0, 0.2],
        [3.0, 1.0],
        [4.0, 1.1],
        [5.0, 1.2],
        [6.0, 2.0],
        [7.0, 2.1],
        [8.0, 2.2],
    ])
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    DarkoClassifier(iterations=1, random_state=0).fit(
        X[:6], y[:6], eval_set=(X[6:], y[6:])
    )

    assert calls == [(6, 3)]


def test_multiclass_class_major_loss_matches_row_major():
    from darkofit.losses import MultiSoftmax

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
    assert np.isclose(loss.eval(Y, F), loss.eval_class_major(Y_class, F_class))
    assert np.isclose(loss.eval(Y, F, w), loss.eval_class_major(Y_class, F_class, w))


def test_feature_importances():
    rng = np.random.default_rng(0)
    n = 3000
    strong = rng.normal(size=n)
    noise = rng.normal(size=(n, 4))
    y = (strong + 0.1 * rng.normal(size=n) > 0).astype(int)
    X = np.column_stack([strong, noise])
    m = DarkoClassifier(iterations=100, random_state=0).fit(X, y)
    imp = m.feature_importances_
    assert imp.shape == (5,)
    assert abs(imp.sum() - 1.0) < 1e-6
    assert imp.argmax() == 0          # the informative feature dominates


def test_best_model_truncation_rebuilds_feature_importance():
    from types import SimpleNamespace

    from darkofit.booster import GradientBoosting, MulticlassBoosting

    prep = SimpleNamespace(
        feature_map_=np.array([0, 1, 0], dtype=np.int64),
        n_input_features_=2,
    )
    first = SimpleNamespace(
        splits_feat=np.array([0, 2], dtype=np.int64),
        gains=np.array([1.0, 3.0]),
        depth=2,
    )
    second = SimpleNamespace(
        splits_feat=np.array([1], dtype=np.int64),
        gains=np.array([100.0]),
        depth=1,
    )
    other_class = SimpleNamespace(
        splits_feat=np.array([1], dtype=np.int64),
        gains=np.array([2.0]),
        depth=1,
    )

    scalar = GradientBoosting(iterations=2)
    scalar.prep_ = prep
    scalar.trees_ = [first, second]
    scalar._importance = np.array([4.0, 100.0])
    scalar._flat_cache_ = object()
    scalar.use_best_model_ = True

    scalar._truncate_to_best_model(0, [1.0, 2.0])

    assert len(scalar.trees_) == 1
    assert scalar._flat_cache_ is None
    assert np.array_equal(scalar._importance, np.array([4.0, 0.0]))

    multi = MulticlassBoosting(iterations=2)
    multi.prep_ = prep
    multi.trees_ = [[first, other_class], [second]]
    multi._importance = np.array([4.0, 102.0])
    multi._flat_cache_ = object()
    multi.use_best_model_ = True

    multi._truncate_to_best_model(0, [1.0, 2.0])

    assert len(multi.trees_) == 1
    assert multi._flat_cache_ is None
    assert np.array_equal(multi._importance, np.array([4.0, 2.0]))


def test_mae_loss_is_robust_to_training_outliers_on_mae_metric():
    from sklearn.metrics import mean_absolute_error

    rng = np.random.default_rng(42)
    X = rng.uniform(-2.0, 2.0, size=(800, 3))
    y = (
        3.0 * X[:, 0]
        - 2.0 * X[:, 1]
        + 0.5 * X[:, 2]
        + rng.normal(0.0, 0.2, size=800)
    )
    y_train = y[:600].copy()
    outliers = rng.choice(600, 90, replace=False)
    y_train[outliers] += (
        rng.choice(np.array([-1.0, 1.0]), size=outliers.size)
        * rng.uniform(20.0, 50.0, size=outliers.size)
    )
    common = dict(
        iterations=100,
        learning_rate=0.05,
        depth=4,
        ordered_boosting=False,
        random_state=0,
    )
    mae = DarkoRegressor(loss="MAE", **common).fit(X[:600], y_train)
    rmse = DarkoRegressor(loss="RMSE", **common).fit(X[:600], y_train)

    mae_error = mean_absolute_error(y[600:], mae.predict(X[600:]))
    rmse_error = mean_absolute_error(y[600:], rmse.predict(X[600:]))
    assert mae_error < 0.5 * rmse_error


@pytest.mark.parametrize("loss_name", ["MAE", "Quantile"])
@pytest.mark.parametrize("weights", [None, np.array([
    1.0, 2.0, 0.5, 1.5, 3.0, 0.75, 1.25, 2.5, 0.8, 1.2, 1.7, 0.9
])])
@pytest.mark.parametrize("n_leaves", [8, 32])
def test_leaf_correction_matches_mask_semantics(loss_name, weights, n_leaves):
    from darkofit.booster import GradientBoosting
    from darkofit.losses import MAE, Quantile
    from darkofit.tree import ObliviousTree

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
    qlo = DarkoRegressor(iterations=300, depth=4, loss="Quantile",
                                alpha=0.1, random_state=0).fit(Xtr, ytr)
    qhi = DarkoRegressor(iterations=300, depth=4, loss="Quantile",
                                alpha=0.9, random_state=0).fit(Xtr, ytr)
    cov = np.mean((yte >= qlo.predict(Xte)) & (yte <= qhi.predict(Xte)))
    assert 0.7 < cov < 0.88           # ~0.80 target band



def test_staged_predict_matches_final():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    r = DarkoRegressor(iterations=50, random_state=0).fit(Xtr, ytr)
    stages = list(r.staged_predict(Xte))
    assert len(stages) == r.best_iteration_
    assert np.allclose(stages[-1], r.predict(Xte))


def test_colsample_runs_and_keeps_accuracy():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    m = DarkoClassifier(iterations=150, colsample=0.5,
                               random_state=0).fit(Xtr, ytr)
    assert roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) > 0.97


def test_thread_count_records_effective_threads():
    import numba
    X, y = load_breast_cancer(return_X_y=True)
    m = DarkoClassifier(iterations=30, thread_count=1, random_state=0).fit(X, y)
    assert m.model_.n_threads_ == 1
    # None -> all detected cores
    m2 = DarkoClassifier(iterations=30, thread_count=None, random_state=0).fit(X, y)
    assert m2.model_.n_threads_ == numba.config.NUMBA_NUM_THREADS
    # over-request is clamped, never exceeds detected cores
    m3 = DarkoClassifier(iterations=30, thread_count=9999, random_state=0).fit(X, y)
    assert m3.model_.n_threads_ <= numba.config.NUMBA_NUM_THREADS


def test_lightgbm_small_fit_caps_thread_count_as_maximum():
    X, y = load_breast_cancer(return_X_y=True)
    m = DarkoClassifier(
        iterations=3, tree_mode="lightgbm", num_leaves=7,
        thread_count=8, random_state=0
    ).fit(X, y)
    assert m.model_.n_threads_ <= 2


def test_prediction_restores_each_child_fitted_thread_count(monkeypatch):
    """Mixed-mode bags must not inherit the last fitted child's thread mask."""
    import darkofit.booster as booster

    X, y = load_diabetes(return_X_y=True)
    cat = DarkoRegressor(
        iterations=2, tree_mode="catboost", thread_count=1, random_state=0
    ).fit(X[:120], y[:120])
    leafwise = DarkoRegressor(
        iterations=2, tree_mode="lightgbm", num_leaves=7,
        thread_count=1, random_state=0,
    ).fit(X[:120], y[:120])
    cat.model_.n_threads_ = 4
    leafwise.model_.n_threads_ = 2

    restored = []

    def record_thread_count(thread_count):
        restored.append(int(thread_count))
        return int(thread_count)

    monkeypatch.setattr(booster, "_apply_thread_count", record_thread_count)

    cat.predict(X[:8])
    leafwise.predict(X[:8])
    cat.predict(X[:8])
    cat_stages = cat.staged_predict(X[:8])
    leafwise_stages = leafwise.staged_predict(X[:8])
    next(cat_stages)
    next(leafwise_stages)
    next(cat_stages)
    next(leafwise_stages)

    assert restored == [4, 2, 4, 4, 2, 4, 2]


def test_thread_count_does_not_change_predictions():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    a = DarkoRegressor(iterations=80, thread_count=1, random_state=0).fit(Xtr, ytr)
    b = DarkoRegressor(iterations=80, thread_count=None, random_state=0).fit(Xtr, ytr)
    # histogram sums are deterministic regardless of thread count
    assert np.allclose(a.predict(Xte), b.predict(Xte))


def test_single_thread_fit_skips_threaded_split_buffers(monkeypatch):
    """Serial split search should not allocate threaded scratch buffers."""
    import darkofit.booster as booster

    def fail_if_called(self, n_features):
        raise AssertionError("threaded split buffers should not be allocated")

    monkeypatch.setattr(booster._BaseBooster, "_alloc_split_buffers", fail_if_called)
    X, y = load_diabetes(return_X_y=True)
    DarkoRegressor(
        iterations=3, depth=2, thread_count=1, random_state=0
    ).fit(X[:120], y[:120])


def test_unweighted_rmse_empty_child_fix_matches_zero_weight_proxy():
    """For unit Hessians, mcw=1 and mcw=0 differ only on empty children."""
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=320,
        n_features=8,
        n_informative=7,
        noise=4.0,
        random_state=19,
    )
    common = dict(
        iterations=80,
        learning_rate=0.1,
        depth=6,
        ordered_boosting=False,
        random_state=7,
        thread_count=1,
    )
    constrained = DarkoRegressor(min_child_weight=1.0, **common).fit(X, y)
    proxy = DarkoRegressor(min_child_weight=0.0, **common).fit(X, y)

    assert np.array_equal(constrained.predict(X), proxy.predict(X))
    assert len(constrained.model_.trees_) == len(proxy.model_.trees_)
    for actual, expected in zip(constrained.model_.trees_, proxy.model_.trees_):
        assert np.array_equal(actual.splits_feat, expected.splits_feat)
        assert np.array_equal(actual.splits_thr, expected.splits_thr)
        assert np.array_equal(actual.values, expected.values)


def test_min_child_weight_param_plumbing():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    m = DarkoClassifier(iterations=50, min_child_weight=30,
                               random_state=0).fit(X, y)
    assert m.model_.min_child_weight == 30.0


def test_shared_histogram_buffers_match_standalone():
    """A tree built with pre-allocated shared buffers must be identical to one
    built with its own freshly-allocated buffers (same math, no realloc)."""
    import numpy as np
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree
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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    # Preserve the documented historical five-buffer direct-caller contract;
    # the optimized estimator-owned path supplies a reusable sixth array.
    split_buffers = tuple(
        np.empty((Xb.shape[1], 1 << depth)) for _ in range(5)
    )

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree
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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree
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


def test_hybrid_tree_uses_shared_prefix_then_nonoblivious_representation():
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import NonObliviousTree, build_hybrid_tree

    rng = np.random.default_rng(37)
    X = rng.normal(size=(700, 8))
    y = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + np.where(X[:, 2] > 0, X[:, 3], -X[:, 4])
        + rng.normal(0, 0.1, 700)
    )
    prep = FeaturePreprocessor(64, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    grad = y.mean() - y
    hess = np.ones(len(y))

    tree, leaf, leaf_G, leaf_H = build_hybrid_tree(
        Xb, grad, hess, prep.n_bins_, 4, 1.0, 0.1,
        max_leaves=8, min_child_samples=5, return_training_state=True,
    )

    assert isinstance(tree, NonObliviousTree)
    assert tree.n_leaves >= 4
    assert tree.left_child[0] >= 0
    assert tree.right_child[0] >= 0
    left = tree.left_child[0]
    right = tree.right_child[0]
    assert tree.features[left] == tree.features[right]
    assert tree.thresholds[left] == tree.thresholds[right]
    assert leaf.shape == (Xb.shape[0],)
    assert leaf_G.shape == leaf_H.shape == (tree.n_leaves,)
    out = np.zeros(Xb.shape[0])
    tree.add_predict(Xb, out)
    assert np.array_equal(out, tree.predict(Xb))


def test_hybrid_shared_prefix_uses_random_strength_picker(monkeypatch):
    import darkofit.tree as tree_mod

    calls = []
    original = tree_mod._best_shared_split_counts_with_noise_py

    def capture(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(tree_mod, "_best_shared_split_counts_with_noise_py", capture)

    rng = np.random.default_rng(38)
    Xb = rng.integers(0, 16, size=(160, 5), dtype=np.uint8)
    grad = rng.normal(size=Xb.shape[0])
    hess = np.ones(Xb.shape[0], dtype=np.float64)
    n_bins = np.full(Xb.shape[1], 16, dtype=np.int64)

    tree_mod.build_hybrid_tree(
        Xb, grad, hess, n_bins, 4, 1.0, 0.1,
        max_leaves=4,
        min_child_samples=2,
        min_child_weight=0.0,
        random_strength=0.5,
        split_seed=11,
        tree_iteration=3,
    )

    assert calls
    assert calls[0][9] == 0.5
    assert calls[0][11] == 3


def test_levelwise_tree_add_predict_matches_predict():
    """The experimental depth-wise tree representation must route predict paths alike."""
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_levelwise_tree

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
    from darkofit.tree import add_leaf_values_inplace, build_leafwise_tree

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
    from darkofit.tree import (
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
    from darkofit.tree import (
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


def test_multiclass_class_minor_refill_subtract_matches_two_step():
    from darkofit.tree import (
        _refill_multiclass_leaf_segment_histograms_counts_class_minor_into,
        _refill_multiclass_left_subtract_right_counts_class_minor_into,
        _refill_multiclass_right_subtract_left_counts_class_minor_into,
    )

    rng = np.random.default_rng(20260715)
    X_binned = rng.integers(0, 23, size=(128, 9), dtype=np.uint8)
    grad = rng.normal(size=(X_binned.shape[0], 4))
    hess = rng.uniform(0.05, 1.5, size=grad.shape)
    row_order = np.arange(X_binned.shape[0], dtype=np.int64)
    leaf_start = np.array([0, 37, 88, 128], dtype=np.int64)
    left_leaf = 1
    right_leaf = 2

    base_hg = rng.normal(size=(X_binned.shape[1], 3, 23, 4))
    base_hh = rng.uniform(0.0, 5.0, size=base_hg.shape)
    base_hc = rng.uniform(0.0, 5.0, size=(X_binned.shape[1], 3, 23))

    fused_hg = base_hg.copy()
    fused_hh = base_hh.copy()
    fused_hc = base_hc.copy()
    ref_hg = base_hg.copy()
    ref_hh = base_hh.copy()
    ref_hc = base_hc.copy()
    _refill_multiclass_right_subtract_left_counts_class_minor_into(
        X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
        fused_hg, fused_hh, fused_hc
    )
    _refill_multiclass_leaf_segment_histograms_counts_class_minor_into(
        X_binned, grad, hess, row_order, leaf_start,
        np.array([right_leaf], dtype=np.int64), 1, ref_hg, ref_hh, ref_hc
    )
    ref_hg[:, left_leaf] -= ref_hg[:, right_leaf]
    ref_hh[:, left_leaf] -= ref_hh[:, right_leaf]
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
    parent_hg = ref_hg[:, left_leaf].copy()
    parent_hh = ref_hh[:, left_leaf].copy()
    parent_hc = ref_hc[:, left_leaf].copy()
    _refill_multiclass_left_subtract_right_counts_class_minor_into(
        X_binned, grad, hess, row_order, leaf_start, left_leaf, right_leaf,
        fused_hg, fused_hh, fused_hc
    )
    _refill_multiclass_leaf_segment_histograms_counts_class_minor_into(
        X_binned, grad, hess, row_order, leaf_start,
        np.array([left_leaf], dtype=np.int64), 1, ref_hg, ref_hh, ref_hc
    )
    ref_hg[:, right_leaf] = parent_hg - ref_hg[:, left_leaf]
    ref_hh[:, right_leaf] = parent_hh - ref_hh[:, left_leaf]
    ref_hc[:, right_leaf] = parent_hc - ref_hc[:, left_leaf]
    assert np.array_equal(fused_hg, ref_hg)
    assert np.array_equal(fused_hh, ref_hh)
    assert np.array_equal(fused_hc, ref_hc)


def test_leafwise_multiclass_histogram_subtraction_matches_full_refill():
    from darkofit.tree import build_leafwise_multiclass_tree

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
    from darkofit.tree import build_leafwise_tree

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
    import darkofit.tree as tree_mod
    from darkofit.preprocessing import FeaturePreprocessor

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_leafwise_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_leafwise_tree

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

    catboost = DarkoClassifier(
        iterations=12, depth=3, tree_mode="catboost", random_state=0
    ).fit(Xtr, ytr)
    oblivious = DarkoClassifier(
        iterations=12, depth=3, tree_mode="oblivious", random_state=0
    ).fit(Xtr, ytr)
    lightgbm = DarkoClassifier(
        iterations=12, depth=3, tree_mode="lightgbm", random_state=0
    ).fit(Xtr, ytr)
    hybrid = DarkoClassifier(
        iterations=12, depth=3, tree_mode="hybrid", random_state=0
    ).fit(Xtr, ytr)
    non_oblivious = DarkoClassifier(
        iterations=12, depth=3, tree_mode="non_oblivious", random_state=0
    ).fit(Xtr, ytr)

    assert catboost.model_.tree_mode_ == "catboost"
    assert oblivious.model_.tree_mode_ == "catboost"
    assert lightgbm.model_.tree_mode_ == "lightgbm"
    assert hybrid.model_.tree_mode_ == "hybrid"
    assert non_oblivious.model_.tree_mode_ == "depthwise"
    assert catboost.model_.ordered_boosting_ is True
    assert non_oblivious.model_.ordered_boosting_ is True
    assert lightgbm.model_.ordered_boosting_ is False
    assert hybrid.model_.ordered_boosting_ is False
    assert np.array_equal(catboost.predict_proba(Xte), oblivious.predict_proba(Xte))
    assert lightgbm.predict_proba(Xte).shape == (len(Xte), 2)
    assert hybrid.predict_proba(Xte).shape == (len(Xte), 2)
    assert abs(lightgbm.feature_importances_.sum() - 1.0) < 1e-6
    assert abs(hybrid.feature_importances_.sum() - 1.0) < 1e-6


def test_ordered_boosting_auto_is_task_aware():
    """auto keeps ordered boosting for classification and disables the
    separate ordered leaf update for scalar regression."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] + rng.normal(scale=0.1, size=120)
    Xc = np.column_stack([rng.integers(0, 5, size=120), X])

    numeric = DarkoRegressor(iterations=4, random_state=0).fit(X, y)
    assert numeric.model_.ordered_boosting_ is False
    tree_meta = numeric.model_.auto_params_["tree"]
    assert tree_meta["ordered_boosting_rule"] == "auto_off_scalar_regression"
    assert tree_meta["ordered_boosting_input"] == "auto"

    # Leaf-adjusted losses never apply the ordered update, so auto resolves
    # off even with categorical features, and explicit True is rejected
    # instead of being silently ignored.
    for loss in ("MAE", "Quantile"):
        m = DarkoRegressor(iterations=4, loss=loss, random_state=0).fit(X, y)
        assert m.model_.ordered_boosting_ is False
        mc = DarkoRegressor(iterations=4, loss=loss, random_state=0).fit(
            Xc, y, cat_features=[0]
        )
        assert mc.model_.ordered_boosting_ is False
        assert (
            mc.model_.auto_params_["tree"]["ordered_boosting_rule"]
            == "auto_off_adjusted_leaf_loss"
        )
        with pytest.raises(ValueError, match="leaf-adjusted"):
            DarkoRegressor(
                iterations=4, loss=loss, ordered_boosting=True, random_state=0
            ).fit(X, y)

    categorical = DarkoRegressor(iterations=4, random_state=0).fit(
        Xc, y, cat_features=[0]
    )
    assert categorical.model_.ordered_boosting_ is False
    assert (
        categorical.model_.auto_params_["tree"]["ordered_boosting_rule"]
        == "auto_off_scalar_regression"
    )

    clf = DarkoClassifier(iterations=4, random_state=0).fit(
        X, (y > 0).astype(int)
    )
    assert clf.model_.ordered_boosting_ is True

    explicit = DarkoRegressor(
        iterations=4, ordered_boosting=True, random_state=0
    ).fit(X, y)
    assert explicit.model_.ordered_boosting_ is True
    assert (
        explicit.model_.auto_params_["tree"]["ordered_boosting_rule"]
        == "explicit"
    )


def test_leafwise_modes_reject_ordered_boosting_true():
    X, y = load_breast_cancer(return_X_y=True)
    for tree_mode in ("lightgbm", "hybrid"):
        with pytest.raises(ValueError, match="ordered_boosting=True"):
            DarkoClassifier(
                iterations=2, tree_mode=tree_mode, ordered_boosting=True
            ).fit(X[:80], y[:80])


def test_tree_mode_default_depth_resolution():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.75, random_state=4, stratify=y
    )
    Xr, yr = load_diabetes(return_X_y=True)

    catboost = DarkoClassifier(
        iterations=2, tree_mode="catboost", random_state=0
    ).fit(Xtr, ytr)
    depthwise = DarkoClassifier(
        iterations=2, tree_mode="depthwise", random_state=0
    ).fit(Xtr, ytr)
    lightgbm = DarkoClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=64, random_state=0
    ).fit(Xtr, ytr)
    hybrid = DarkoClassifier(
        iterations=2, tree_mode="hybrid", num_leaves=64, random_state=0
    ).fit(Xtr, ytr)
    explicit = DarkoClassifier(
        iterations=2, tree_mode="lightgbm", depth=3, num_leaves=64,
        random_state=0
    ).fit(Xtr, ytr)
    depthwise_reg = DarkoRegressor(
        iterations=2, tree_mode="depthwise", random_state=0
    ).fit(Xr, yr)

    assert catboost.model_.depth == 6
    assert depthwise.model_.depth == 6
    assert depthwise_reg.model_.depth == 2
    depth_meta = depthwise_reg.model_.auto_params_["auto_structure"]
    assert depth_meta["resolved"]["depth"]["source"] == "default"
    assert (
        depth_meta["candidates"]["depth"]["rule"]
        == "depthwise_rmse_shallow_default"
    )
    assert lightgbm.model_.depth == -1
    assert hybrid.model_.depth == -1
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

    catboost = DarkoRegressor(
        iterations=1, tree_mode="catboost", random_state=0
    ).fit(X, y_reg, cat_features=[0, 1])
    lightgbm_reg = DarkoRegressor(
        iterations=1, tree_mode="lightgbm", num_leaves=3, random_state=0
    ).fit(X, y_reg, cat_features=[0, 1])
    lightgbm_binary = DarkoClassifier(
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
    assert lightgbm_reg.model_.auto_params_["binning"]["cat_smoothing_input"] == 1.0
    assert lightgbm_reg.model_.auto_params_["binning"]["cat_smoothing_resolved"] == 3.0
    assert lightgbm_binary.model_.auto_params_["binning"]["cat_smoothing_resolved"] == 1.0
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

    model = DarkoRegressor(
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
        DarkoClassifier(
            iterations=2, tree_mode="catboost", num_leaves=7
        ).fit(X[:80], y[:80])
    with pytest.raises(ValueError, match="depth"):
        DarkoClassifier(
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
        DarkoRegressor(iterations=1, cat_smoothing=0.0).fit(
            X, y, cat_features=[0]
        )


def test_sparse_inputs_raise_clear_error():
    sparse = pytest.importorskip("scipy.sparse")
    X, y = load_breast_cancer(return_X_y=True)
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        DarkoClassifier(iterations=2).fit(sparse.csr_matrix(X), y)

    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        DarkoClassifier(iterations=2).fit(
            Xtr, ytr, eval_set=(sparse.csr_matrix(Xv), yv)
        )

    clf = DarkoClassifier(iterations=2, random_state=0).fit(Xtr, ytr)
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        clf.predict(sparse.csr_matrix(Xv))
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        list(clf.staged_predict_proba(sparse.csr_matrix(Xv)))

    Xr, yr = load_diabetes(return_X_y=True)
    Xr_tr, Xr_v, yr_tr, _ = train_test_split(
        Xr, yr, test_size=0.2, random_state=0
    )
    reg = DarkoRegressor(iterations=2, random_state=0).fit(Xr_tr, yr_tr)
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        reg.predict(sparse.csr_matrix(Xr_v))
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        list(reg.staged_predict(sparse.csr_matrix(Xr_v)))


def test_cat_features_accepts_numpy_arrays_across_public_layers():
    from darkofit.booster import GradientBoosting, MulticlassBoosting

    X = np.array([
        ["red", 0.0],
        ["blue", 1.0],
        ["red", 2.0],
        ["green", 3.0],
        ["blue", 4.0],
        ["green", 5.0],
    ], dtype=object)
    y_reg = np.array([0.0, 1.0, 0.2, 1.4, 1.1, 1.7])
    y_bin = np.array([0, 1, 0, 1, 1, 0])
    y_multi = np.array(["a", "b", "c", "a", "b", "c"])
    cat_features = np.array([0], dtype=np.int64)

    reg = DarkoRegressor(iterations=3, random_state=0).fit(
        X, y_reg, cat_features=cat_features
    )
    clf = DarkoClassifier(iterations=3, random_state=0).fit(
        X, y_bin, cat_features=cat_features
    )
    core = GradientBoosting(iterations=3, random_state=0).fit(
        X, y_reg, cat_features=cat_features
    )
    multiclass = MulticlassBoosting(iterations=3, random_state=0).fit(
        X, y_multi, cat_features=cat_features
    )

    assert np.isfinite(reg.predict(X)).all()
    assert np.isfinite(clf.predict_proba(X)).all()
    assert np.isfinite(core.predict_raw(X)).all()
    assert np.isfinite(multiclass.predict_raw(X)).all()


def test_cat_features_validation_has_clear_errors():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)

    with pytest.raises(ValueError, match="out of bounds"):
        DarkoRegressor(iterations=1).fit(
            X, y, cat_features=np.array([2], dtype=np.int64)
        )
    with pytest.raises(ValueError, match="integer column indices"):
        DarkoRegressor(iterations=1).fit(
            X, y, cat_features=np.array([0.0])
        )
    with pytest.raises(ValueError, match="integer column indices"):
        DarkoRegressor(iterations=1).fit(
            X, y, cat_features=np.array([True])
        )


def test_training_targets_must_be_1d_nonempty_and_match_rows():
    from darkofit.booster import GradientBoosting, MulticlassBoosting

    X = np.arange(12, dtype=float).reshape(6, 2)
    y_reg = np.arange(6, dtype=float)
    y_cls = np.array([0, 1, 0, 1, 0, 1])

    estimators = [
        DarkoRegressor(iterations=1),
        GradientBoosting(iterations=1),
    ]
    for estimator in estimators:
        with pytest.raises(ValueError, match=r"y must have shape \(6,\)"):
            estimator.fit(X, np.array([1.0]))
        with pytest.warns(DataConversionWarning, match="column-vector y"):
            estimator.fit(X, y_reg.reshape(-1, 1))
        with pytest.raises(ValueError, match="at least one sample"):
            estimator.fit(X[:0], y_reg[:0])

    classifiers = [
        DarkoClassifier(iterations=1),
        MulticlassBoosting(iterations=1),
    ]
    for estimator in classifiers:
        with pytest.raises(ValueError, match=r"y must have shape \(6,\)"):
            estimator.fit(X, np.array([0, 1]))
        with pytest.warns(DataConversionWarning, match="column-vector y"):
            estimator.fit(X, y_cls.reshape(-1, 1))
        with pytest.raises(ValueError, match="at least one sample"):
            estimator.fit(X[:0], y_cls[:0])


def test_eval_targets_must_be_1d_nonempty_and_match_eval_rows():
    from darkofit.booster import GradientBoosting, MulticlassBoosting

    X = np.arange(20, dtype=float).reshape(10, 2)
    y_reg = np.arange(10, dtype=float)
    y_cls = np.tile([0, 1], 5)
    Xv = X[:4]

    for estimator in [
        DarkoRegressor(iterations=1),
        GradientBoosting(iterations=1),
    ]:
        with pytest.raises(ValueError, match=r"eval_set\[1\] must have shape"):
            estimator.fit(X, y_reg, eval_set=(Xv, np.arange(5, dtype=float)))
        with pytest.warns(DataConversionWarning, match=r"column-vector eval_set\[1\]"):
            estimator.fit(X, y_reg, eval_set=(Xv, np.arange(4).reshape(-1, 1)))

    for estimator in [
        DarkoClassifier(iterations=1),
        MulticlassBoosting(iterations=1),
    ]:
        with pytest.raises(ValueError, match=r"eval_set\[1\] must have shape"):
            estimator.fit(X, y_cls, eval_set=(Xv, np.array([0, 1, 0, 1, 0])))
        with pytest.warns(DataConversionWarning, match=r"column-vector eval_set\[1\]"):
            estimator.fit(X, y_cls, eval_set=(Xv, np.array([[0], [1], [0], [1]])))


def test_wrappers_accept_column_vector_targets_with_warning():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y_reg = np.linspace(0.0, 1.0, 20)
    y_cls = np.tile([0, 1], 10)

    with pytest.warns(DataConversionWarning, match="column-vector y"):
        reg = DarkoRegressor(iterations=2, random_state=0).fit(
            X, y_reg.reshape(-1, 1)
        )
    assert reg.predict(X[:2]).shape == (2,)

    with pytest.warns(DataConversionWarning, match="column-vector y"):
        clf = DarkoClassifier(iterations=2, random_state=0).fit(
            X, y_cls.reshape(-1, 1)
        )
    assert clf.predict_proba(X[:2]).shape == (2, 2)


def test_wrapper_boundaries_reject_bad_learning_rate_and_continuous_labels():
    X = np.arange(40, dtype=float).reshape(20, 2)

    for learning_rate in (0.0, -0.1, np.inf, np.nan):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            DarkoRegressor(
                iterations=2, learning_rate=learning_rate
            ).fit(X, np.arange(20, dtype=float))

    with pytest.raises(ValueError, match="Unknown label type: continuous"):
        DarkoClassifier(iterations=2).fit(
            X, np.linspace(0.0, 1.0, 20)
        )


def test_wrappers_accept_random_state_objects_and_eval_set_lists():
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = X[:, 0] - X[:, 1]
    reg = DarkoRegressor(
        iterations=2, random_state=np.random.RandomState(0)
    ).fit(X[:30], y[:30], eval_set=[(X[30:], y[30:])])
    assert reg.predict(X[:3]).shape == (3,)

    labels = np.tile([0, 1], 20)
    clf = DarkoClassifier(
        iterations=2, random_state=np.random.default_rng(1)
    ).fit(X[:30], labels[:30], eval_set=[(X[30:], labels[30:])])
    assert clf.predict_proba(X[:3]).shape == (3, 2)


def test_wrapper_save_persists_normalized_random_state_seed(tmp_path):
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = X[:, 0] - X[:, 1]
    model = DarkoRegressor(
        iterations=3, random_state=np.random.default_rng(0)
    ).fit(X, y)
    path = tmp_path / "wrapper-rng.npz"

    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)

    assert isinstance(loaded.random_state, int)
    assert loaded.random_state == model.model_._fit_random_state_seed_
    assert np.array_equal(model.predict(X[:5]), loaded.predict(X[:5]))


def test_pandas_nullable_numeric_eval_set_and_predict_are_coerced():
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame({
        "a": pd.Series([1, 2, pd.NA, 4, 5, 6, 7, pd.NA], dtype="Int64"),
        "b": pd.Series([0.5, pd.NA, 1.5, 2.0, 2.5, 3.0, pd.NA, 4.0],
                       dtype="Float64"),
    })
    y = np.array([1.0, 1.5, 1.2, 2.0, 2.4, 2.8, 3.0, 3.5])
    X_train, X_eval = X.iloc[:6], X.iloc[6:]
    y_train, y_eval = y[:6], y[6:]

    reg = DarkoRegressor(iterations=2, random_state=0).fit(
        X_train, y_train, eval_set=(X_eval, y_eval)
    )
    pred = reg.predict(X_eval)

    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))


def test_sklearn_wrappers_raise_not_fitted_for_prediction_and_save(tmp_path):
    from sklearn.exceptions import NotFittedError

    X = np.arange(12, dtype=float).reshape(6, 2)
    unfitted_reg = DarkoRegressor()
    unfitted_clf = DarkoClassifier()

    with pytest.raises(NotFittedError):
        unfitted_reg.predict(X)
    with pytest.raises(NotFittedError):
        list(unfitted_reg.staged_predict(X))
    with pytest.raises(NotFittedError):
        unfitted_reg.save_model(tmp_path / "reg.npz")

    with pytest.raises(NotFittedError):
        unfitted_clf.predict_proba(X)
    with pytest.raises(NotFittedError):
        unfitted_clf.predict(X)
    with pytest.raises(NotFittedError):
        list(unfitted_clf.staged_predict_raw(X))
    with pytest.raises(NotFittedError):
        unfitted_clf.save_model(tmp_path / "clf.npz")


def test_wrappers_record_and_enforce_feature_count():
    X, y = load_diabetes(return_X_y=True)
    reg = DarkoRegressor(iterations=2, random_state=0).fit(X[:80], y[:80])
    assert reg.n_features_in_ == X.shape[1]

    with pytest.raises(ValueError, match="expecting"):
        reg.predict(X[:5, :-1])
    with pytest.raises(ValueError, match="expecting"):
        list(reg.staged_predict(X[:5, :-1]))
    with pytest.raises(ValueError, match="eval_set\\[0\\] has"):
        DarkoRegressor(iterations=2, random_state=0).fit(
            X[:80], y[:80], eval_set=(X[:10, :-1], y[:10])
        )

    Xc, yc = load_breast_cancer(return_X_y=True)
    clf = DarkoClassifier(iterations=2, random_state=0).fit(
        Xc[:100], yc[:100]
    )
    assert clf.n_features_in_ == Xc.shape[1]

    with pytest.raises(ValueError, match="expecting"):
        clf.predict_proba(Xc[:5, :-1])
    with pytest.raises(ValueError, match="expecting"):
        list(clf.staged_predict_proba(Xc[:5, :-1]))


def test_zero_feature_inputs_raise_clear_value_error():
    X = np.empty((8, 0), dtype=np.float64)
    y = np.arange(8, dtype=np.float64)

    with pytest.raises(ValueError, match="at least one feature"):
        DarkoRegressor(iterations=1).fit(X, y)


def test_wrappers_enforce_named_feature_order_when_input_has_names():
    class NamedArray:
        def __init__(self, data, columns):
            self._data = np.asarray(data, dtype=np.float64)
            self.columns = columns
            self.shape = self._data.shape

        def __array__(self, dtype=None):
            return np.asarray(self._data, dtype=dtype)

    X_data = np.column_stack([
        np.linspace(0.0, 1.0, 30),
        np.linspace(1.0, 0.0, 30),
    ])
    y = 4.0 * X_data[:, 0] - X_data[:, 1]
    X_ab = NamedArray(X_data, ["a", "b"])
    X_ba = NamedArray(X_data[:, [1, 0]], ["b", "a"])

    model = DarkoRegressor(iterations=2, random_state=0).fit(X_ab, y)
    assert model.feature_names_in_.tolist() == ["a", "b"]

    with pytest.raises(ValueError, match="feature names"):
        model.predict(X_ba)
    with pytest.raises(ValueError, match="feature names"):
        list(model.staged_predict(X_ba))
    with pytest.raises(ValueError, match="feature names"):
        DarkoRegressor(iterations=1, random_state=0).fit(
            X_ab, y, eval_set=(X_ba, y)
        )


def test_core_boosters_validate_prediction_feature_count():
    from darkofit.booster import GradientBoosting, MulticlassBoosting

    X = np.arange(24, dtype=float).reshape(12, 2)
    y = X[:, 0] - X[:, 1]
    core = GradientBoosting(iterations=2, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="expecting 2 features"):
        core.predict_raw(np.column_stack([X, np.ones(X.shape[0])]))
    with pytest.raises(ValueError, match="expecting 2 features"):
        list(core.staged_predict_raw(X[:, :1]))

    y_multi = np.tile([0, 1, 2], 4)
    multi = MulticlassBoosting(iterations=2, random_state=0).fit(X, y_multi)
    with pytest.raises(ValueError, match="expecting 2 features"):
        multi.predict_raw(np.column_stack([X, np.ones(X.shape[0])]))
    with pytest.raises(ValueError, match="expecting 2 features"):
        list(multi.staged_predict_raw(X[:, :1]))


def test_failed_refit_does_not_publish_partial_wrapper_state():
    X, y = load_diabetes(return_X_y=True)
    reg = DarkoRegressor(iterations=2, random_state=0).fit(X[:80], y[:80])
    old_reg_pred = reg.predict(X[:5])
    old_reg_n_features = reg.n_features_in_

    X_wide = np.column_stack([X[:80], np.ones(80)])
    with pytest.raises(ValueError, match="sample_weight"):
        reg.fit(X_wide, y[:80], sample_weight=np.ones(79))

    assert reg.n_features_in_ == old_reg_n_features
    assert np.array_equal(reg.predict(X[:5]), old_reg_pred)

    Xc, yc = load_breast_cancer(return_X_y=True)
    clf = DarkoClassifier(iterations=2, random_state=0).fit(
        Xc[:120], yc[:120]
    )
    old_clf_proba = clf.predict_proba(Xc[:5])
    old_classes = clf.classes_.copy()
    old_clf_n_features = clf.n_features_in_

    with pytest.raises(ValueError, match="Need at least 2 classes"):
        clf.fit(np.column_stack([Xc[:20], np.ones(20)]), np.zeros(20))

    assert clf.n_features_in_ == old_clf_n_features
    assert np.array_equal(clf.classes_, old_classes)
    assert np.array_equal(clf.predict_proba(Xc[:5]), old_clf_proba)


def test_lightgbm_mode_enforces_leaf_constraints():
    X, y = load_diabetes(return_X_y=True)
    model = DarkoRegressor(
        iterations=3, tree_mode="lightgbm", num_leaves=3, depth=2,
        min_child_samples=30, random_state=0
    ).fit(X, y)
    assert model.model_.tree_mode_ == "lightgbm"
    for tree in model.model_.trees_:
        assert tree.n_leaves <= 3
        assert tree.depth <= 2


def test_lightgbm_num_leaves_capped_by_positive_depth():
    X, y = load_diabetes(return_X_y=True)
    model = DarkoRegressor(
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
    model = DarkoRegressor(
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
    model = DarkoClassifier(
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


def test_lightgbm_shared_multiclass_tree_routes_when_auto_compatible():
    X_num = np.array([
        [0.0, 0.1], [0.2, 0.0], [1.0, 0.8], [1.2, 1.1],
        [2.0, 2.2], [2.2, 1.9], [0.1, 0.2], [1.1, 1.0],
        [2.1, 2.0],
    ])
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    numeric = DarkoClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=3,
        min_child_samples=1, min_child_weight=0.0, random_state=0
    ).fit(X_num, y)
    assert numeric.model_.multiclass_tree_strategy_ == "shared_vector"
    assert hasattr(numeric.model_.trees_[0], "add_predict_class_major")

    per_class = DarkoClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=3,
        min_child_samples=1, min_child_weight=0.0,
        multiclass_tree_strategy="per_class", random_state=0
    ).fit(X_num, y)
    assert per_class.model_.multiclass_tree_strategy_ == "per_class"
    assert isinstance(per_class.model_.trees_[0], list)

    X_cat = np.empty((len(y), 2), dtype=object)
    X_cat[:, 0] = np.array(
        ["a", "a", "b", "b", "c", "c", "a", "b", "c"], dtype=object
    )
    X_cat[:, 1] = X_num[:, 1]
    categorical = DarkoClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=3,
        min_child_samples=1, min_child_weight=0.0, random_state=0
    ).fit(X_cat, y, cat_features=[0])
    assert hasattr(categorical.model_.trees_[0], "add_predict_class_major")


def test_multiclass_count_histogram_counts_any_positive_class_hessian():
    from darkofit.tree import _build_multiclass_histograms_counts_into

    X_binned = np.array([[0], [1], [0]], dtype=np.uint8)
    grad = np.zeros((3, 3), dtype=np.float64)
    hess = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    leaf = np.zeros(3, dtype=np.int64)
    hg = np.zeros((3, 1, 1, 2), dtype=np.float64)
    hh = np.zeros_like(hg)
    hc = np.zeros((1, 1, 2), dtype=np.float64)

    _build_multiclass_histograms_counts_into(
        X_binned, grad, hess, leaf, 1, hg, hh, hc
    )

    assert np.array_equal(hc[0, 0], np.array([2.0, 1.0]))


def test_multiclass_class_minor_histograms_match_class_major_reference():
    from darkofit.tree import (
        _build_multiclass_histograms_counts_class_minor_into,
        _build_multiclass_histograms_counts_into,
        _class_major_views_from_class_minor_histograms,
    )

    rng = np.random.default_rng(20260713)
    n_samples, n_features, n_classes, n_leaves, max_bins = 96, 6, 4, 3, 11
    X_binned = rng.integers(
        0, max_bins, size=(n_samples, n_features), dtype=np.uint8
    )
    leaf = rng.integers(0, n_leaves, size=n_samples, dtype=np.int64)
    grad = rng.normal(size=(n_classes, n_samples))
    hess = rng.uniform(0.0, 1.5, size=grad.shape)

    ref_hg = np.zeros((n_classes, n_features, n_leaves, max_bins))
    ref_hh = np.zeros_like(ref_hg)
    ref_hc = np.zeros((n_features, n_leaves, max_bins))
    _build_multiclass_histograms_counts_into(
        X_binned, grad, hess, leaf, n_leaves, ref_hg, ref_hh, ref_hc
    )

    grad_row = np.ascontiguousarray(grad.T)
    hess_row = np.ascontiguousarray(hess.T)
    got_hg = np.zeros((n_features, n_leaves, max_bins, n_classes))
    got_hh = np.zeros_like(got_hg)
    got_hc = np.zeros_like(ref_hc)
    _build_multiclass_histograms_counts_class_minor_into(
        X_binned, grad_row, hess_row, leaf, n_leaves, got_hg, got_hh, got_hc
    )
    got_hg_view, got_hh_view = _class_major_views_from_class_minor_histograms(
        got_hg, got_hh
    )

    assert got_hg.shape == (n_features, n_leaves, max_bins, n_classes)
    assert np.array_equal(got_hg_view, ref_hg)
    assert np.array_equal(got_hh_view, ref_hh)
    assert np.array_equal(got_hc, ref_hc)


def test_leafwise_multiclass_accepts_class_minor_hist_buffers():
    from darkofit.tree import build_leafwise_multiclass_tree

    rng = np.random.default_rng(20260714)
    X_binned = rng.integers(0, 19, size=(400, 7), dtype=np.uint8)
    grad = rng.normal(size=(4, X_binned.shape[0]))
    hess = rng.uniform(0.05, 1.25, size=grad.shape)
    n_bins = np.full(X_binned.shape[1], 19, dtype=np.int64)
    max_leaves = 9

    base = build_leafwise_multiclass_tree(
        X_binned, grad, hess, n_bins, 5, 1.3, 0.08,
        max_leaves=max_leaves, min_child_samples=4, min_child_weight=0.1,
        min_gain_to_split=0.0, return_training_state=True,
    )
    buffers = (
        np.zeros((X_binned.shape[1], max_leaves, int(n_bins.max()), grad.shape[0])),
        np.zeros((X_binned.shape[1], max_leaves, int(n_bins.max()), grad.shape[0])),
        np.zeros((X_binned.shape[1], max_leaves, int(n_bins.max()))),
    )
    supplied = build_leafwise_multiclass_tree(
        X_binned, grad, hess, n_bins, 5, 1.3, 0.08,
        max_leaves=max_leaves, min_child_samples=4, min_child_weight=0.1,
        min_gain_to_split=0.0, return_training_state=True,
        hist_buffers=buffers,
        grad_row_major=np.ascontiguousarray(grad.T),
        hess_row_major=np.ascontiguousarray(hess.T),
    )

    base_tree, base_leaf, base_G, base_H = base
    supplied_tree, supplied_leaf, supplied_G, supplied_H = supplied
    for attr in (
        "features", "thresholds", "left_child", "right_child",
        "leaf_index", "splits_feat", "splits_thr", "gains", "values",
    ):
        assert np.array_equal(
            getattr(base_tree, attr), getattr(supplied_tree, attr)
        ), attr
    assert base_tree.depth == supplied_tree.depth
    assert np.array_equal(base_leaf, supplied_leaf)
    assert np.array_equal(base_G, supplied_G)
    assert np.array_equal(base_H, supplied_H)


def test_multiclass_shared_split_allows_empty_class_hessian():
    from darkofit.tree import (
        _best_multiclass_splits_counts_for_leaf_ids_with_noise_class_minor_py,
        _best_multiclass_splits_counts_for_leaf_ids_with_noise_py,
        _best_multiclass_splits_for_leaf_ids_counts_class_minor,
        _best_multiclass_splits_for_leaf_ids_counts,
    )

    hg = np.zeros((3, 1, 1, 2), dtype=np.float64)
    hh = np.zeros_like(hg)
    hc = np.array([[[5.0, 5.0]]], dtype=np.float64)
    # Class 0 is absent in this leaf. Classes 1/2 carry enough total Hessian.
    hg[1, 0, 0, 0] = 2.0
    hg[1, 0, 0, 1] = -2.0
    hh[1, 0, 0, :] = 5.0
    hh[2, 0, 0, :] = np.array([1.0, 4.0])
    n_bins = np.array([2], dtype=np.int64)
    feat_mask = np.array([1], dtype=np.int64)
    leaf_ids = np.array([0], dtype=np.int64)
    l2_by_class = np.ones(3, dtype=np.float64)

    out_feat = np.full(1, -1, dtype=np.int64)
    out_thr = np.full(1, -1, dtype=np.int64)
    out_gain = np.full(1, -np.inf, dtype=np.float64)
    _best_multiclass_splits_for_leaf_ids_counts(
        hg, hh, hc, n_bins, l2_by_class, feat_mask, 1.0, 1.0,
        leaf_ids, 1, out_feat, out_thr, out_gain
    )

    assert out_feat[0] == 0
    assert out_thr[0] == 0
    assert out_gain[0] > 0.0
    expected_gain = out_gain[0]

    out_feat[:] = -1
    out_thr[:] = -1
    out_gain[:] = -np.inf
    _best_multiclass_splits_counts_for_leaf_ids_with_noise_py(
        hg, hh, hc, n_bins, l2_by_class, feat_mask, 1.0, 1.0,
        leaf_ids, 1, out_feat, out_thr, out_gain,
        0.0, 0, 0, 0, 0.0,
    )

    assert out_feat[0] == 0
    assert out_thr[0] == 0
    assert out_gain[0] > 0.0
    assert out_gain[0] == expected_gain

    hg_cm = np.moveaxis(hg, 0, 3).copy()
    hh_cm = np.moveaxis(hh, 0, 3).copy()
    out_feat[:] = -1
    out_thr[:] = -1
    out_gain[:] = -np.inf
    _best_multiclass_splits_for_leaf_ids_counts_class_minor(
        hg_cm, hh_cm, hc, n_bins, l2_by_class, feat_mask, 1.0, 1.0,
        leaf_ids, 1, out_feat, out_thr, out_gain
    )

    assert out_feat[0] == 0
    assert out_thr[0] == 0
    assert out_gain[0] == expected_gain

    out_feat[:] = -1
    out_thr[:] = -1
    out_gain[:] = -np.inf
    _best_multiclass_splits_counts_for_leaf_ids_with_noise_class_minor_py(
        hg_cm, hh_cm, hc, n_bins, l2_by_class, feat_mask, 1.0, 1.0,
        leaf_ids, 1, out_feat, out_thr, out_gain,
        0.0, 0, 0, 0, 0.0,
    )

    assert out_feat[0] == 0
    assert out_thr[0] == 0
    assert out_gain[0] == expected_gain


def test_inactive_stochastic_settings_preserve_shared_vector_multiclass_lightgbm():
    X_num = np.array([
        [0.0, 0.1], [0.2, 0.0], [1.0, 0.8], [1.2, 1.1],
        [2.0, 2.2], [2.2, 1.9], [0.1, 0.2], [1.1, 1.0],
        [2.1, 2.0],
    ])
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    X_cat = np.empty((len(y), 2), dtype=object)
    X_cat[:, 0] = np.array(
        ["a", "a", "b", "b", "c", "c", "a", "b", "c"], dtype=object
    )
    X_cat[:, 1] = X_num[:, 1]
    kw = dict(
        iterations=2,
        tree_mode="lightgbm",
        num_leaves=3,
        min_child_samples=1,
        min_child_weight=0.0,
        random_state=0,
    )

    base = DarkoClassifier(**kw).fit(X_cat, y, cat_features=[0])
    zero_bootstrap = DarkoClassifier(
        **kw, bootstrap_type="bayesian", bagging_temperature=0.0
    ).fit(X_cat, y, cat_features=[0])
    full_mvs = DarkoClassifier(
        **kw, sampling="mvs", subsample=1.0
    ).fit(X_cat, y, cat_features=[0])
    explicit = DarkoClassifier(
        **kw,
        multiclass_tree_strategy="shared_vector",
        bootstrap_type="bayesian",
        bagging_temperature=0.0,
        sampling="mvs",
        subsample=1.0,
    ).fit(X_cat, y, cat_features=[0])

    for model in (base, zero_bootstrap, full_mvs, explicit):
        assert model.model_.multiclass_tree_strategy_ == "shared_vector"
        assert hasattr(model.model_.trees_[0], "add_predict_class_major")
        assert np.array_equal(base.predict_proba(X_cat), model.predict_proba(X_cat))


def test_lightgbm_numeric_multiclass_can_force_shared_vector_tree():
    X = np.array([
        [0.0, 0.1], [0.2, 0.0], [1.0, 0.8], [1.2, 1.1],
        [2.0, 2.2], [2.2, 1.9], [0.1, 0.2], [1.1, 1.0],
        [2.1, 2.0], [0.3, 0.1], [1.3, 1.2], [2.3, 2.1],
    ])
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])

    model = DarkoClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=3,
        min_child_samples=1, min_child_weight=0.0,
        multiclass_tree_strategy="shared_vector", random_state=0
    ).fit(X, y)

    assert model.model_.multiclass_tree_strategy_ == "shared_vector"
    assert hasattr(model.model_.trees_[0], "add_predict_class_major")
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_lightgbm_shared_vector_strategy_requires_compatible_training_mode():
    X = np.array([
        [0.0, 0.1], [0.2, 0.0], [1.0, 0.8], [1.2, 1.1],
        [2.0, 2.2], [2.2, 1.9], [0.1, 0.2], [1.1, 1.0],
        [2.1, 2.0],
    ])
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])

    with pytest.raises(ValueError, match="multiclass_tree_strategy"):
        DarkoClassifier(
            iterations=1, tree_mode="catboost",
            multiclass_tree_strategy="shared_vector", random_state=0
        ).fit(X, y)

    with pytest.raises(ValueError, match="multiclass_tree_strategy"):
        DarkoClassifier(
            iterations=1, tree_mode="lightgbm",
            multiclass_tree_strategy="bogus", random_state=0
        ).fit(X, y)


def test_lightgbm_zero_weight_rows_do_not_affect_tree_structure():
    from darkofit.tree import build_leafwise_tree

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
    from darkofit.tree import _partition_leaf_rows

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
    from darkofit.tree import _partition_leaf_rows

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
    from darkofit.tree import (
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
    from darkofit.tree import (
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
    from darkofit.tree import (
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
    from darkofit.tree import (
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
    from darkofit.tree import build_leafwise_tree

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


@pytest.mark.parametrize("thread_count", [1, 2, 4])
def test_leafwise_segmented_row_layout_matches_prefix_layout(thread_count):
    numba = pytest.importorskip("numba")
    from darkofit.tree import build_leafwise_tree

    rng = np.random.default_rng(63)
    Xb = rng.integers(0, 48, size=(900, 13), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 48, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.1, 1.7, size=Xb.shape[0])

    cases = [
        (True, np.ones_like(hess), False),
        (False, hess, False),
        (False, hess, True),
    ]
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(thread_count)
        for constant_hessian, h, hessian_always_positive in cases:
            prefix = build_leafwise_tree(
                Xb, grad, h, n_bins, 6, 1.0, 0.1,
                max_leaves=16, min_child_samples=5, min_child_weight=0.1,
                min_gain_to_split=0.0, return_training_state=True,
                constant_hessian=constant_hessian,
                hessian_always_positive=hessian_always_positive,
                leafwise_row_layout="prefix",
            )
            segmented = build_leafwise_tree(
                Xb, grad, h, n_bins, 6, 1.0, 0.1,
                max_leaves=16, min_child_samples=5, min_child_weight=0.1,
                min_gain_to_split=0.0, return_training_state=True,
                constant_hessian=constant_hessian,
                hessian_always_positive=hessian_always_positive,
                leafwise_row_layout="segmented",
            )

            prefix_tree, prefix_leaf, prefix_G, prefix_H = prefix
            segmented_tree, segmented_leaf, segmented_G, segmented_H = segmented
            assert np.array_equal(segmented_tree.features, prefix_tree.features)
            assert np.array_equal(segmented_tree.thresholds, prefix_tree.thresholds)
            assert np.array_equal(segmented_tree.left_child, prefix_tree.left_child)
            assert np.array_equal(segmented_tree.right_child, prefix_tree.right_child)
            assert np.array_equal(segmented_tree.leaf_index, prefix_tree.leaf_index)
            assert np.array_equal(segmented_tree.splits_feat, prefix_tree.splits_feat)
            assert np.array_equal(segmented_tree.splits_thr, prefix_tree.splits_thr)
            assert np.array_equal(segmented_tree.gains, prefix_tree.gains)
            assert np.array_equal(segmented_leaf, prefix_leaf)
            assert np.array_equal(segmented_tree.values, prefix_tree.values)
            assert np.array_equal(segmented_G, prefix_G)
            assert np.array_equal(segmented_H, prefix_H)
            assert np.array_equal(segmented_tree.predict(Xb), prefix_tree.predict(Xb))
    finally:
        numba.set_num_threads(original_threads)


def test_leafwise_segmented_row_layout_guard_and_auto_fallback():
    from darkofit.tree import build_leafwise_tree

    rng = np.random.default_rng(64)
    Xb = rng.integers(0, 16, size=(256, 7), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 16, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = np.ones(Xb.shape[0])
    selected = np.array([0, 2, 4], dtype=np.int64)
    feature_mask = np.zeros(Xb.shape[1], dtype=np.int64)
    feature_mask[selected] = 1

    with pytest.raises(ValueError, match="leafwise_row_layout='segmented'"):
        build_leafwise_tree(
            Xb, grad, hess, n_bins, 5, 1.0, 0.1,
            max_leaves=8, feature_mask=feature_mask,
            feature_indices=selected, constant_hessian=True,
            leafwise_row_layout="segmented",
        )

    prefix = build_leafwise_tree(
        Xb, grad, hess, n_bins, 5, 1.0, 0.1,
        max_leaves=8, feature_mask=feature_mask,
        feature_indices=selected, constant_hessian=True,
        leafwise_row_layout="prefix", return_training_state=True,
    )
    auto = build_leafwise_tree(
        Xb, grad, hess, n_bins, 5, 1.0, 0.1,
        max_leaves=8, feature_mask=feature_mask,
        feature_indices=selected, constant_hessian=True,
        leafwise_row_layout="auto", return_training_state=True,
    )
    assert np.array_equal(auto[0].splits_feat, prefix[0].splits_feat)
    assert np.array_equal(auto[0].splits_thr, prefix[0].splits_thr)
    assert np.array_equal(auto[1], prefix[1])

    full_prefix = build_leafwise_tree(
        Xb, grad, hess, n_bins, 5, 1.0, 0.1,
        max_leaves=8, constant_hessian=True,
        leafwise_row_layout="prefix", return_training_state=True,
    )
    full_auto = build_leafwise_tree(
        Xb, grad, hess, n_bins, 5, 1.0, 0.1,
        max_leaves=8, constant_hessian=True,
        leafwise_row_layout="auto", return_training_state=True,
    )
    assert np.array_equal(full_auto[0].splits_feat, full_prefix[0].splits_feat)
    assert np.array_equal(full_auto[0].splits_thr, full_prefix[0].splits_thr)
    assert np.array_equal(full_auto[1], full_prefix[1])


def test_leafwise_positive_hessian_route_matches_generic_tree():
    from darkofit.tree import build_leafwise_tree

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


def test_leafwise_positive_hessian_no_reuse_matches_generic_tree():
    import numba
    from darkofit.tree import build_leafwise_tree

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(62)
    Xb = rng.integers(0, 32, size=(800, 14), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 32, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.05, 1.2, size=Xb.shape[0])

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        generic = build_leafwise_tree(
            Xb, grad, hess, n_bins, 6, 1.0, 0.1,
            max_leaves=12, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            hessian_always_positive=False,
            reuse_leaf_histograms=False,
        )
        positive = build_leafwise_tree(
            Xb, grad, hess, n_bins, 6, 1.0, 0.1,
            max_leaves=12, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            hessian_always_positive=True,
            reuse_leaf_histograms=False,
        )
    finally:
        numba.set_num_threads(old_threads)

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
    from darkofit.tree import build_leafwise_tree

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
    from darkofit.tree import (
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
    from darkofit.tree import (
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
    from darkofit.tree import build_leafwise_tree

    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires at least two numba threads")

    rng = np.random.default_rng(48)
    Xb = rng.integers(0, 32, size=(1000, 18), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 32, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.05, 1.0, size=Xb.shape[0])

    old_threads = numba.get_num_threads()
    try:
        cases = [2]
        if numba.config.NUMBA_NUM_THREADS >= 4:
            cases.append(4)
        for n_threads in cases:
            numba.set_num_threads(n_threads)
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
    finally:
        numba.set_num_threads(old_threads)


def test_leafwise_fused_changed_leaf_scoring_matches_default_path():
    import numba
    from darkofit.tree import build_leafwise_tree

    if numba.config.NUMBA_NUM_THREADS < 4:
        pytest.skip("requires at least four numba threads")

    rng = np.random.default_rng(65)
    Xb = rng.integers(0, 64, size=(1200, 18), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 64, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.05, 1.5, size=Xb.shape[0])

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        default = build_leafwise_tree(
            Xb, grad, hess, n_bins, 6, 1.0, 0.1,
            max_leaves=14, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            hessian_always_positive=True,
            fused_changed_leaf_scoring=False,
        )
        fused = build_leafwise_tree(
            Xb, grad, hess, n_bins, 6, 1.0, 0.1,
            max_leaves=14, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            hessian_always_positive=True,
            fused_changed_leaf_scoring=True,
        )
        feature_mask = np.ones(Xb.shape[1], dtype=np.int64)
        feature_mask[::3] = 0
        masked_default = build_leafwise_tree(
            Xb, grad, hess, n_bins, 6, 1.0, 0.1,
            max_leaves=14, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            hessian_always_positive=True,
            feature_mask=feature_mask,
            fused_changed_leaf_scoring=False,
        )
        masked_fused_requested = build_leafwise_tree(
            Xb, grad, hess, n_bins, 6, 1.0, 0.1,
            max_leaves=14, min_child_samples=5, min_child_weight=0.1,
            min_gain_to_split=0.0, return_training_state=True,
            hessian_always_positive=True,
            feature_mask=feature_mask,
            fused_changed_leaf_scoring=True,
        )
    finally:
        numba.set_num_threads(old_threads)

    default_tree, default_leaf, default_G, default_H = default
    fused_tree, fused_leaf, fused_G, fused_H = fused
    assert np.array_equal(fused_tree.features, default_tree.features)
    assert np.array_equal(fused_tree.thresholds, default_tree.thresholds)
    assert np.array_equal(fused_tree.left_child, default_tree.left_child)
    assert np.array_equal(fused_tree.right_child, default_tree.right_child)
    assert np.array_equal(fused_tree.leaf_index, default_tree.leaf_index)
    assert np.array_equal(fused_tree.splits_feat, default_tree.splits_feat)
    assert np.array_equal(fused_tree.splits_thr, default_tree.splits_thr)
    assert np.allclose(fused_tree.gains, default_tree.gains)
    assert np.allclose(fused_tree.values, default_tree.values)
    assert np.array_equal(fused_leaf, default_leaf)
    assert np.allclose(fused_G, default_G)
    assert np.allclose(fused_H, default_H)
    assert np.array_equal(fused_tree.predict(Xb), default_tree.predict(Xb))
    masked_default_tree, masked_default_leaf, _, _ = masked_default
    masked_fused_tree, masked_fused_leaf, _, _ = masked_fused_requested
    assert np.array_equal(masked_fused_tree.features, masked_default_tree.features)
    assert np.array_equal(masked_fused_tree.thresholds, masked_default_tree.thresholds)
    assert np.array_equal(masked_fused_leaf, masked_default_leaf)


def test_lightgbm_scalar_routes_fused_changed_leaf_scoring_only_when_eligible():
    from darkofit.booster import GradientBoosting

    booster = GradientBoosting(
        tree_mode="lightgbm",
        num_leaves=31,
        random_strength=0.0,
    )
    booster.tree_mode_ = "lightgbm"
    booster.n_threads_ = 4

    def fused_enabled(**overrides):
        params = dict(
            fmask=None,
            findices=None,
            row_indices=None,
            hist_buffers=None,
            split_buffers=None,
            X_hist_binned=None,
            X_route_binned=None,
            use_constant_hessian=False,
            hessian_always_positive=True,
            rowpar_buffers=None,
            tree_iteration=0,
        )
        params.update(overrides)
        return booster._builder_kwargs(**params)["fused_changed_leaf_scoring"]

    assert fused_enabled() is True
    assert fused_enabled(use_constant_hessian=True) is False
    assert fused_enabled(hessian_always_positive=False) is False
    assert fused_enabled(fmask=np.ones(2, dtype=np.uint8)) is False
    assert fused_enabled(findices=np.array([0], dtype=np.int64)) is False
    assert fused_enabled(row_indices=np.array([0], dtype=np.int64)) is False
    assert fused_enabled(rowpar_buffers=(np.empty((1, 1, 1, 1)),)) is False

    booster.random_strength = 0.1
    assert fused_enabled() is False
    booster.random_strength = 0.0
    booster.n_threads_ = 2
    assert fused_enabled() is False


def test_leafwise_threaded_changed_leaf_split_matches_full_rescore():
    import numba
    from darkofit.tree import build_leafwise_tree

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
    one = DarkoClassifier(
        iterations=8, tree_mode="lightgbm", num_leaves=7, depth=3,
        thread_count=1, random_state=0
    ).fit(Xtr, ytr)
    two = DarkoClassifier(
        iterations=8, tree_mode="lightgbm", num_leaves=7, depth=3,
        thread_count=2, random_state=0
    ).fit(Xtr, ytr)
    assert np.allclose(one.predict_proba(Xte), two.predict_proba(Xte))


def test_non_oblivious_parallel_add_predict_matches_serial():
    import numba
    from darkofit.tree import (
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
    model = DarkoClassifier(iterations=12, random_state=0).fit(Xtr, ytr)
    stages = list(model.staged_predict_proba(Xte))
    assert len(stages) == model.best_iteration_
    assert np.allclose(stages[-1], model.predict_proba(Xte))


def test_multiclass_staged_predictions_match_final():
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    model = DarkoClassifier(iterations=8, random_state=0).fit(Xtr, ytr)
    stages = list(model.staged_predict_proba(Xte))
    assert len(stages) == model.best_iteration_
    assert np.allclose(stages[-1], model.predict_proba(Xte))
    assert np.allclose(stages[-1].sum(axis=1), 1.0)


def test_multiclass_predict_labels_use_raw_margin_argmax(monkeypatch):
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(
        X, y, test_size=0.25, random_state=2, stratify=y
    )
    model = DarkoClassifier(iterations=8, random_state=0).fit(Xtr, ytr)
    expected = model.classes_[np.argmax(model.model_.predict_raw(Xte), axis=1)]
    expected_stages = [
        model.classes_[np.argmax(raw, axis=1)]
        for raw in model.model_.staged_predict_raw(Xte)
    ]

    def fail_transform(raw):
        raise AssertionError("predict labels should not compute softmax")

    monkeypatch.setattr(model.model_.loss_, "transform", fail_transform)

    assert np.array_equal(model.predict(Xte), expected)
    got_stages = list(model.staged_predict(Xte))
    assert len(got_stages) == len(expected_stages)
    for got, want in zip(got_stages, expected_stages):
        assert np.array_equal(got, want)


def test_multiclass_subsampling_shared_per_round(monkeypatch):
    import darkofit.booster as booster
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
    DarkoClassifier(
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
    from darkofit import DarkoClassifier
    from darkofit.tree import NonObliviousTree

    rng = np.random.default_rng(58)
    X = rng.normal(size=(120, 6))
    y = np.repeat(np.arange(3), 40)
    order = rng.permutation(len(y))

    def fail_add_predict(self, X_binned, out):
        raise AssertionError("training update should reuse returned leaf ids")

    monkeypatch.setattr(NonObliviousTree, "add_predict", fail_add_predict)
    DarkoClassifier(
        iterations=2, tree_mode="lightgbm", num_leaves=5, depth=3,
        random_state=0
    ).fit(X[order], y[order])


def test_lightgbm_numeric_multiclass_marks_unweighted_hessians_positive(monkeypatch):
    import darkofit.booster as booster
    from darkofit import DarkoClassifier

    rng = np.random.default_rng(59)
    X = rng.normal(size=(120, 6))
    y = np.repeat(np.arange(3), 40)
    order = rng.permutation(len(y))
    calls = []
    original = booster.build_leafwise_tree

    def wrapped_build_tree(*args, **kwargs):
        calls.append(bool(kwargs.get("hessian_always_positive", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(booster, "build_leafwise_tree", wrapped_build_tree)
    DarkoClassifier(
        iterations=1, tree_mode="lightgbm", num_leaves=5, depth=3,
        multiclass_tree_strategy="per_class",
        random_state=0
    ).fit(X[order], y[order])

    assert calls
    assert all(calls)


def test_lightgbm_numeric_multiclass_weighted_hessians_use_generic_path(monkeypatch):
    import darkofit.booster as booster
    from darkofit import DarkoClassifier

    rng = np.random.default_rng(60)
    X = rng.normal(size=(120, 6))
    y = np.repeat(np.arange(3), 40)
    order = rng.permutation(len(y))
    weights = np.ones_like(y, dtype=np.float64)
    weights[order[:10]] = 0.0
    calls = []
    original = booster.build_leafwise_tree

    def wrapped_build_tree(*args, **kwargs):
        calls.append(bool(kwargs.get("hessian_always_positive", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(booster, "build_leafwise_tree", wrapped_build_tree)
    DarkoClassifier(
        iterations=1, tree_mode="lightgbm", num_leaves=5, depth=3,
        multiclass_tree_strategy="per_class",
        random_state=0
    ).fit(X[order], y[order], sample_weight=weights[order])

    assert calls
    assert not any(calls)


def test_lightgbm_multiclass_uses_task_specific_default_l2():
    from darkofit import DarkoClassifier

    rng = np.random.default_rng(61)
    X = rng.normal(size=(90, 5))
    y = np.repeat(np.arange(3), 30)
    order = rng.permutation(len(y))

    lightgbm_default = DarkoClassifier(
        iterations=1, tree_mode="lightgbm", num_leaves=5, depth=3,
        random_state=0
    ).fit(X[order], y[order])
    catboost_default = DarkoClassifier(
        iterations=1, tree_mode="catboost", depth=3, random_state=0
    ).fit(X[order], y[order])
    lightgbm_explicit = DarkoClassifier(
        iterations=1, tree_mode="lightgbm", num_leaves=5, depth=3,
        l2_leaf_reg=2.0, random_state=0
    ).fit(X[order], y[order])

    assert lightgbm_default.l2_leaf_reg == "auto"
    assert lightgbm_default.model_.l2_leaf_reg == 1.0
    assert lightgbm_default.model_.l2_leaf_reg_ == 1.0
    assert catboost_default.l2_leaf_reg == "auto"
    assert catboost_default.model_.l2_leaf_reg == 3.0
    assert catboost_default.model_.l2_leaf_reg_ == 3.0
    assert lightgbm_explicit.model_.l2_leaf_reg == 2.0
    assert lightgbm_explicit.model_.l2_leaf_reg_ == 2.0


def test_goss_subsample_keeps_large_gradients_and_scales_sampled_rows():
    from darkofit.booster import GradientBoosting

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
    import darkofit.booster as booster

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
    model = DarkoRegressor(
        iterations=1, tree_mode="lightgbm", num_leaves=7, depth=3,
        sampling="goss", top_rate=0.2, other_rate=0.2, random_state=0
    ).fit(X[:120], y[:120])

    assert np.all(np.isfinite(model.predict(X[:5])))
    assert calls
    row_indices, constant_hessian = calls[0]
    assert row_indices is not None
    assert constant_hessian is False


def test_goss_rejects_uniform_subsample():
    X, y = load_diabetes(return_X_y=True)
    for subsample in (0.8, 1.2):
        with pytest.raises(ValueError, match="subsample must be 1.0"):
            DarkoRegressor(
                iterations=1, tree_mode="lightgbm", sampling="goss",
                subsample=subsample, random_state=0
            ).fit(X[:80], y[:80])

    for subsample in (0.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="subsample must"):
            DarkoRegressor(
                iterations=1, tree_mode="lightgbm", sampling="uniform",
                subsample=subsample, random_state=0
            ).fit(X[:80], y[:80])

    # Multiclass GOSS is supported now; it fits without error.
    Xc = np.vstack([X[:30], X[30:60], X[60:90]])
    yc = np.repeat([0, 1, 2], 30)
    m = DarkoClassifier(
        iterations=3, tree_mode="lightgbm", sampling="goss",
        random_state=0
    ).fit(Xc, yc)
    assert m.predict_proba(Xc).shape == (90, 3)


def test_multiclass_no_split_class_tree_is_boosting_noop(monkeypatch):
    import darkofit.booster as booster

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
    model = DarkoClassifier(iterations=1, random_state=0).fit(X, y)

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
    model = DarkoClassifier(
        iterations=120, early_stopping=True, early_stopping_rounds=5,
        validation_fraction=0.2, tree_mode="lightgbm", num_leaves=7,
        depth=3, learning_rate=0.2, random_state=0
    ).fit(Xtr, ytr)
    assert model.best_iteration_ < 120
    stages = list(model.staged_predict_proba(Xte))
    assert len(stages) == model.best_iteration_
    assert np.allclose(stages[-1], model.predict_proba(Xte))


def test_eval_set_keeps_best_prefix_without_patience():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    kw = dict(iterations=80, learning_rate=0.1, depth=2, random_state=0)

    keep_all = DarkoRegressor(
        **kw, use_best_model=False
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    best_n = int(np.argmin(keep_all.model_.valid_history_)) + 1
    assert best_n < len(keep_all.model_.trees_)

    best = DarkoRegressor(**kw).fit(Xtr, ytr, eval_set=(Xv, yv))
    assert best.best_iteration_ == best_n
    assert len(best.model_.trees_) == best_n
    assert best.best_score_ == min(keep_all.model_.valid_history_)
    assert best.model_.auto_params_["early_stopping"]["use_best_model"] is True
    assert (
        keep_all.model_.auto_params_["early_stopping"]["use_best_model"]
        is False
    )

    keep_all_best_pred = list(keep_all.staged_predict(Xv))[best_n - 1]
    assert np.allclose(best.predict(Xv), keep_all_best_pred)


def test_patience_stop_respects_use_best_model_false():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    model = DarkoRegressor(
        iterations=120,
        learning_rate=0.3,
        depth=2,
        early_stopping_rounds=5,
        use_best_model=False,
        random_state=0,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))

    best_n = int(np.argmin(model.model_.valid_history_)) + 1
    assert model.n_estimators_ > best_n
    assert model.model_.auto_params_["early_stopping"]["use_best_model"] is False
    assert (
        model.model_.auto_params_["early_stopping"]["best_prefix_policy"]
        == "disabled"
    )


def test_refit_uses_best_prefix_selection_without_patience():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    model = DarkoRegressor(
        iterations=80,
        learning_rate=0.1,
        depth=2,
        random_state=0,
        refit=True,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))

    assert model.refit_ is True
    assert model.selection_model_.best_iteration_ < 80
    assert model.refit_n_estimators_ == model.selection_model_.best_iteration_
    assert model.n_estimators_ == model.selection_model_.best_iteration_


def test_auto_early_stopping_patience_resolves_from_learning_rate():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    slow = DarkoRegressor(
        iterations=3,
        learning_rate=0.05,
        early_stopping=True,
        random_state=0,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    slow_meta = slow.model_.auto_params_["early_stopping"]
    assert slow_meta["rounds_input"] == "auto"
    assert slow_meta["rounds"] == 100
    assert slow_meta["rounds_rule"] == "ceil(5/lr)_clipped_20_200"

    clipped_slow = DarkoRegressor(
        iterations=3,
        learning_rate=0.001,
        early_stopping_rounds="auto",
        random_state=0,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    assert clipped_slow.model_.auto_params_["early_stopping"]["rounds"] == 200

    hot = DarkoRegressor(
        iterations=3,
        learning_rate=0.5,
        early_stopping_rounds="auto",
        random_state=0,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    hot_meta = hot.model_.auto_params_["early_stopping"]
    assert hot_meta["rounds"] == 20

    explicit = DarkoRegressor(
        iterations=3,
        learning_rate=0.05,
        early_stopping=True,
        early_stopping_rounds=7,
        random_state=0,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    explicit_meta = explicit.model_.auto_params_["early_stopping"]
    assert explicit_meta["rounds"] == 7
    assert explicit_meta["rounds_rule"] == "explicit"


def test_multiclass_eval_set_keeps_best_prefix_without_patience():
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    kw = dict(iterations=50, learning_rate=0.5, depth=2, random_state=0)

    keep_all = DarkoClassifier(
        **kw, use_best_model=False
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    best_n = int(np.argmin(keep_all.model_.valid_history_)) + 1
    assert best_n < len(keep_all.model_.trees_)

    best = DarkoClassifier(**kw).fit(Xtr, ytr, eval_set=(Xv, yv))
    assert best.best_iteration_ == best_n
    assert len(best.model_.trees_) == best_n
    assert best.best_score_ == min(keep_all.model_.valid_history_)

    keep_all_best_proba = list(keep_all.staged_predict_proba(Xv))[best_n - 1]
    assert np.allclose(best.predict_proba(Xv), keep_all_best_proba)


def test_auto_learning_rate_catboost_transplant_corridor():
    from darkofit.auto_params import auto_learning_rate

    lr = auto_learning_rate(
        "RMSE", n_eff=20_000, iterations=1000,
        use_best_model=False, tree_mode="catboost", max_leaves=64
    )
    assert 0.065 < lr < 0.067

    low_eff_lr = auto_learning_rate(
        "RMSE", n_eff=500, iterations=1000,
        use_best_model=False, tree_mode="catboost", max_leaves=64
    )
    assert low_eff_lr < lr

    lightgbm_lr = auto_learning_rate(
        "RMSE", n_eff=20_000, iterations=1000,
        use_best_model=False, tree_mode="lightgbm", max_leaves=127
    )
    assert lightgbm_lr < lr

    weighted_lightgbm_lr = auto_learning_rate(
        "RMSE", n_eff=16_000, iterations=1000,
        use_best_model=False, tree_mode="lightgbm", max_leaves=31,
        n_eff_fraction=0.8,
    )
    unweighted_lightgbm_lr = auto_learning_rate(
        "RMSE", n_eff=20_000, iterations=1000,
        use_best_model=False, tree_mode="lightgbm", max_leaves=31,
        n_eff_fraction=1.0,
    )
    assert unweighted_lightgbm_lr < weighted_lightgbm_lr

    unweighted_catboost_lr = auto_learning_rate(
        "RMSE", n_eff=20_000, iterations=1000,
        use_best_model=False, tree_mode="catboost", max_leaves=64,
        n_eff_fraction=1.0,
    )
    weighted_catboost_lr = auto_learning_rate(
        "RMSE", n_eff=17_000, iterations=1000,
        use_best_model=False, tree_mode="catboost", max_leaves=64,
        n_eff_fraction=0.85,
    )
    assert weighted_catboost_lr > unweighted_catboost_lr

    weighted_mae_lr = auto_learning_rate(
        "MAE", n_eff=17_000, iterations=1000,
        use_best_model=False, tree_mode="catboost", max_leaves=64,
        n_eff_fraction=0.85,
    )
    unweighted_mae_lr = auto_learning_rate(
        "MAE", n_eff=17_000, iterations=1000,
        use_best_model=False, tree_mode="catboost", max_leaves=64,
        n_eff_fraction=1.0,
    )
    assert weighted_mae_lr == unweighted_mae_lr


def test_auto_learning_rate_uses_bounded_feature_shrinkage():
    from darkofit.auto_params import (
        AUTO_LR_FEATURE_MULTIPLIER_MIN,
        auto_learning_rate_details,
    )

    base = auto_learning_rate_details(
        "RMSE", n_eff=1000, iterations=1000,
        use_best_model=False, tree_mode="catboost", max_leaves=64,
        p_model=10,
    )
    wide = auto_learning_rate_details(
        "RMSE", n_eff=1000, iterations=1000,
        use_best_model=False, tree_mode="catboost", max_leaves=64,
        p_model=10_000,
    )

    assert base["feature_multiplier"] == 1.0
    assert base["feature_shrinkage_active"] is False
    assert wide["feature_multiplier"] == AUTO_LR_FEATURE_MULTIPLIER_MIN
    assert wide["feature_shrinkage_active"] is True
    assert wide["raw_auto"] < base["raw_auto"]
    assert wide["p_model"] == 10_000


def test_auto_learning_rate_details_record_clipping_bounds():
    from darkofit.auto_params import (
        AUTO_LR_MAX,
        AUTO_LR_MIN,
        auto_learning_rate_details,
    )

    hot = auto_learning_rate_details(
        "RMSE", n_eff=100, iterations=1,
        use_best_model=False, tree_mode="catboost", max_leaves=64
    )
    assert hot["resolved"] == AUTO_LR_MAX
    assert hot["clipped"] is True
    assert hot["clip_bound"] == "max"
    assert hot["raw_auto"] > AUTO_LR_MAX

    slow = auto_learning_rate_details(
        "Logloss", n_eff=2, iterations=1_000_000,
        use_best_model=False, tree_mode="lightgbm", max_leaves=31
    )
    assert slow["resolved"] == AUTO_LR_MIN
    assert slow["clipped"] is True
    assert slow["clip_bound"] == "min"
    assert slow["raw_auto"] < AUTO_LR_MIN


def test_auto_learning_rate_uniform_weights_match_none():
    rng = np.random.default_rng(86)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] + rng.normal(0.0, 0.1, size=80)

    no_weight = DarkoRegressor(
        iterations=3, random_state=0, eval_train_loss=False
    ).fit(X, y)
    ones = DarkoRegressor(
        iterations=3, random_state=0, eval_train_loss=False
    ).fit(X, y, sample_weight=np.ones(len(y)))

    assert ones.model_.auto_params_["learning_rate"]["rule"] == (
        "catboost-transplant-v2"
    )
    assert ones.model_.auto_params_["binning"]["max_bins"] == 254
    assert ones.learning_rate_ == no_weight.learning_rate_


def test_auto_params_records_warnings_and_diagnostics():
    import darkofit.booster as booster_mod

    rng = np.random.default_rng(91)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] - 0.25 * X[:, 1] + rng.normal(0.0, 0.1, size=80)
    w = np.ones(80)
    w[0] = 1000.0

    booster_mod.reset_diagnostic_warning_registry()
    with warnings.catch_warnings(record=True) as first_caught:
        warnings.simplefilter("always")
        model = DarkoRegressor(
            iterations=1,
            random_state=0,
            eval_train_loss=False,
        ).fit(X, y, sample_weight=w)
    with warnings.catch_warnings(record=True) as second_caught:
        warnings.simplefilter("always")
        DarkoRegressor(
            iterations=1,
            random_state=1,
            eval_train_loss=False,
        ).fit(X, y, sample_weight=w)
    with warnings.catch_warnings(record=True) as never_caught:
        warnings.simplefilter("always")
        never = DarkoRegressor(
            iterations=1,
            random_state=2,
            eval_train_loss=False,
            diagnostic_warnings="never",
        ).fit(X, y, sample_weight=w)

    messages = [str(warning.message) for warning in first_caught]
    assert any("automatic learning rate clipped" in msg for msg in messages)
    assert any("effective sample size is low" in msg for msg in messages)
    assert [str(warning.message) for warning in second_caught] == []
    assert [str(warning.message) for warning in never_caught] == []

    meta = model.model_.auto_params_
    warning_codes = {
        warning["code"] for warning in meta["diagnostics"]["warnings"]
    }
    assert "learning_rate_clipped_max" in warning_codes
    assert "low_effective_sample_size_fraction" in warning_codes
    assert meta["learning_rate"]["clipped"] is True
    assert meta["learning_rate"]["clip_bound"] == "max"
    assert meta["learning_rate"]["raw_auto"] > meta["learning_rate"]["clip_max"]
    assert meta["diagnostics"]["learning_rate_clipped"] is True
    assert meta["diagnostics"]["learning_rate_clip_bound"] == "max"
    assert meta["diagnostics"]["low_effective_sample_size_fraction_threshold"] == 0.3
    assert meta["diagnostics"]["effective_sample_size_fraction"] < 0.3
    assert meta["binning"]["numeric_binning_weighted"] is True
    assert meta["diagnostics"]["weighted_binning_active"] is True
    assert (
        meta["diagnostics"]["observed_max_bins"]
        == meta["binning"]["observed_max_bins"]
    )
    assert (
        meta["diagnostics"]["observed_total_bins"]
        == meta["binning"]["observed_total_bins"]
    )
    assert meta["features"]["feature_expansion_factor"] == 1.0
    assert meta["diagnostics"]["feature_expansion_factor"] == 1.0
    assert meta["diagnostics"]["runtime_warning_policy"] == "once"
    assert set(meta["diagnostics"]["runtime_warnings_emitted"]) == warning_codes
    assert meta["diagnostics"]["best_prefix_policy"] == "disabled"
    never_meta = never.model_.auto_params_
    assert {
        warning["code"] for warning in never_meta["diagnostics"]["warnings"]
    } == warning_codes
    assert never_meta["diagnostics"]["runtime_warning_policy"] == "never"
    assert never_meta["diagnostics"]["runtime_warnings_emitted"] == []

    booster_mod.reset_diagnostic_warning_registry()
    with warnings.catch_warnings(record=True) as always_caught:
        warnings.simplefilter("always")
        DarkoRegressor(
            iterations=1,
            random_state=3,
            eval_train_loss=False,
            diagnostic_warnings="always",
        ).fit(X, y, sample_weight=w)
    with warnings.catch_warnings(record=True) as after_always_caught:
        warnings.simplefilter("always")
        DarkoRegressor(
            iterations=1,
            random_state=4,
            eval_train_loss=False,
        ).fit(X, y, sample_weight=w)
    assert len(always_caught) >= 2
    assert len(after_always_caught) >= 2


def test_get_refit_params_freezes_learning_rate_and_exact_rounds():
    X, y = load_breast_cancer(return_X_y=True)
    model = DarkoClassifier(
        iterations=120, early_stopping=True, early_stopping_rounds=5,
        validation_fraction=0.2, random_state=0
    ).fit(X, y)

    params = model.get_refit_params()
    assert params["iterations"] == model.best_n_estimators_
    assert params["iterations"] == model.n_estimators_
    assert params["learning_rate"] == model.learning_rate_
    assert params["learning_rate"] == model.model_.lr_
    assert params["early_stopping"] is False
    assert params["early_stopping_rounds"] is None


def test_get_refit_params_scaled_strategies_use_empirical_split():
    X, y = load_breast_cancer(return_X_y=True)
    model = DarkoClassifier(
        iterations=120, early_stopping=True, early_stopping_rounds=5,
        validation_fraction=0.2, random_state=0
    ).fit(X, y)

    scale = model._selection_n_total_ / model._selection_n_train_
    sqrt_params = model.get_refit_params(strategy="sqrt")
    linear_params = model.get_refit_params(strategy="linear")
    scaled_params = model.get_refit_params(strategy="scaled")

    assert sqrt_params["iterations"] == int(
        np.ceil(model.best_n_estimators_ * np.sqrt(scale))
    )
    assert linear_params["iterations"] == int(
        np.ceil(model.best_n_estimators_ * scale)
    )
    assert scaled_params["iterations"] == linear_params["iterations"]


def test_get_refit_params_scaled_requires_auto_split():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    model = DarkoClassifier(
        iterations=80, early_stopping_rounds=5, random_state=0
    ).fit(Xtr, ytr, eval_set=(Xv, yv))

    assert model.get_refit_params(strategy="exact")["iterations"] == (
        model.best_n_estimators_
    )
    with pytest.raises(ValueError, match="automatic validation split"):
        model.get_refit_params(strategy="sqrt")


def test_get_refit_params_regressor_preserves_loss_params():
    X, y = load_diabetes(return_X_y=True)
    model = DarkoRegressor(
        iterations=60, loss="Quantile", alpha=0.8,
        early_stopping=True, early_stopping_rounds=5, random_state=0
    ).fit(X, y)

    params = model.get_refit_params(strategy="best")
    assert params["iterations"] == model.best_n_estimators_
    assert params["learning_rate"] == model.learning_rate_
    assert params["loss"] == "Quantile"
    assert params["alpha"] == 0.8
    assert params["early_stopping"] is False


def test_early_stopping_refit_trains_final_model_with_exact_rounds():
    X, y = load_breast_cancer(return_X_y=True)
    model = DarkoClassifier(
        iterations=120, early_stopping=True, early_stopping_rounds=5,
        validation_fraction=0.2, refit=True, random_state=0
    ).fit(X, y)

    assert model.refit_ is True
    assert model.refit_strategy_ == "exact"
    assert model.selection_model_ is not model.model_
    assert model.best_n_estimators_ == len(model.selection_model_.trees_)
    assert model.n_estimators_ == model.best_n_estimators_
    assert model.refit_n_estimators_ == model.n_estimators_
    assert model.learning_rate_ == model.selection_model_.lr_
    assert model.best_score_ == model.selection_model_.best_score_
    assert model.get_refit_params()["refit"] is False


def test_early_stopping_refit_scaled_rounds_use_empirical_split():
    X, y = load_breast_cancer(return_X_y=True)
    model = DarkoClassifier(
        iterations=120, early_stopping=True, early_stopping_rounds=5,
        validation_fraction=0.2, refit=True, refit_strategy="sqrt",
        random_state=0
    ).fit(X, y)

    scale = model._selection_n_total_ / model._selection_n_train_
    expected = int(np.ceil(model.best_n_estimators_ * np.sqrt(scale)))
    assert model.refit_ is True
    assert model.refit_strategy_ == "sqrt"
    assert model.n_estimators_ == expected
    assert model.refit_n_estimators_ == expected


def test_refit_freezes_resolved_auto_structure_across_size_boundary():
    rng = np.random.default_rng(109)
    X = rng.normal(size=(5400, 2))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(0.0, 0.2, size=5400)

    model = DarkoRegressor(
        iterations=3,
        depth="auto",
        early_stopping=True,
        early_stopping_rounds=2,
        validation_fraction=0.1,
        refit=True,
        random_state=0,
        eval_train_loss=False,
    ).fit(X, y)

    assert model.selection_model_.depth == 5
    assert model.model_.depth == 5
    assert model.get_refit_params()["depth"] == 5


def test_refit_without_early_stopping_does_not_double_fit():
    X, y = load_diabetes(return_X_y=True)
    model = DarkoRegressor(
        iterations=8, refit=True, random_state=0
    ).fit(X, y)

    assert model.refit_ is False
    assert not hasattr(model, "selection_model_")
    assert model.n_estimators_ == 8
    assert model.best_n_estimators_ == 8


def test_refit_metadata_round_trips_through_save_load(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    model = DarkoClassifier(
        iterations=120, early_stopping=True, early_stopping_rounds=5,
        validation_fraction=0.2, refit=True, refit_strategy="sqrt",
        random_state=0
    ).fit(X, y)
    path = tmp_path / "refit_clf.npz"

    model.save_model(path)
    loaded = DarkoClassifier.load_model(path)

    assert loaded.refit_ is True
    assert loaded.refit_strategy_ == "sqrt"
    assert loaded.refit_n_estimators_ == model.refit_n_estimators_
    assert loaded.best_n_estimators_ == model.best_n_estimators_
    assert loaded.best_score_ == model.best_score_
    assert loaded.learning_rate_ == model.learning_rate_
    assert loaded.n_estimators_ == model.n_estimators_
    assert loaded.selection_model_ is None
    assert loaded.selection_model_persisted_ is False
    assert loaded.get_refit_params(strategy="sqrt")["iterations"] == (
        model.get_refit_params(strategy="sqrt")["iterations"]
    )


def test_loaded_refit_selection_persistence_flag_is_not_stale(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    model = DarkoClassifier(
        iterations=12, early_stopping=True, early_stopping_rounds=2,
        validation_fraction=0.2, refit=True, random_state=0
    ).fit(X[:120], y[:120])
    path = tmp_path / "refit-flag.npz"
    model.save_model(path)
    loaded = DarkoClassifier.load_model(path)
    assert loaded.selection_model_persisted_ is False

    loaded.fit(X[:120], y[:120])
    assert loaded.selection_model_persisted_ is True

    loaded.set_params(refit=False).fit(X[:120], y[:120])
    assert not hasattr(loaded, "selection_model_persisted_")


def test_refit_strategy_errors_before_partial_fit():
    X, y = load_breast_cancer(return_X_y=True)
    model = DarkoClassifier(
        iterations=20, early_stopping=True, refit=True,
        refit_strategy="bogus", random_state=0
    )
    with pytest.raises(ValueError, match="unknown refit strategy"):
        model.fit(X, y)
    assert not hasattr(model, "model_")
    assert not hasattr(model, "_selection_n_total_")

    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    model = DarkoClassifier(
        iterations=20, early_stopping_rounds=5, refit=True,
        refit_strategy="linear", random_state=0
    )
    with pytest.raises(ValueError, match="automatic validation split"):
        model.fit(Xtr, ytr, eval_set=(Xv, yv))
    assert not hasattr(model, "model_")


def test_eval_labels_must_be_training_classes():
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    bad_yv = yv.copy()
    bad_yv[0] = 99
    with pytest.raises(ValueError, match="eval_set contains labels"):
        DarkoClassifier(iterations=2).fit(Xtr, ytr, eval_set=(Xv, bad_yv))


def test_invalid_sample_weights_raise():
    X, y = load_diabetes(return_X_y=True)
    for bad in [
        np.ones((len(y), 1)),
        np.full(len(y), np.nan),
        -np.ones(len(y)),
        np.zeros(len(y)),
    ]:
        with pytest.raises(ValueError):
            DarkoRegressor(iterations=2).fit(X, y, sample_weight=bad)


def test_invalid_numeric_targets_raise():
    X, y = load_diabetes(return_X_y=True)
    for bad in [np.full(len(y), np.nan), np.full(len(y), np.inf)]:
        with pytest.raises(ValueError, match="y must contain only finite values"):
            DarkoRegressor(iterations=2).fit(X, bad)

    y_binary = np.tile([0.0, 1.0], 20)
    y_binary[3] = np.nan
    with pytest.raises(ValueError, match="y must contain only finite values"):
        DarkoClassifier(iterations=2).fit(X[:40], y_binary)


def test_invalid_eval_sample_weights_raise():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=0)
    with pytest.raises(ValueError, match="eval_sample_weight"):
        DarkoRegressor(iterations=2).fit(
            Xtr, ytr, eval_set=(Xv, yv), eval_sample_weight=np.ones(len(yv) + 1)
        )


def test_eval_sample_weight_requires_eval_set_for_all_fit_entries():
    from darkofit.booster import GradientBoosting, MulticlassBoosting

    X = np.arange(80, dtype=np.float64).reshape(40, 2)
    y_reg = np.linspace(-1.0, 1.0, 40)
    y_cls = np.tile([0, 1], 20)
    weights = np.ones(40)
    message = "eval_sample_weight requires an explicit eval_set"

    with pytest.raises(ValueError, match=message):
        DarkoRegressor(
            iterations=2, tree_mode="catboost", early_stopping=False
        ).fit(X, y_reg, eval_sample_weight=weights)

    with pytest.raises(ValueError, match=message):
        DarkoClassifier(
            iterations=2, tree_mode="catboost", early_stopping=False
        ).fit(X, y_cls, eval_sample_weight=weights)

    with pytest.raises(ValueError, match=message):
        GradientBoosting(iterations=2).fit(
            X, y_reg, eval_sample_weight=weights
        )

    with pytest.raises(ValueError, match=message):
        MulticlassBoosting(iterations=2).fit(
            X, y_cls, eval_sample_weight=weights
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
    easy = DarkoRegressor(
        iterations=80, early_stopping=True, early_stopping_rounds=5,
        learning_rate=0.2, depth=2, random_state=0
    ).fit(Xtr, ytr, eval_set=(Xv, yv), eval_sample_weight=easy_weight)
    hard = DarkoRegressor(
        iterations=80, early_stopping=True, early_stopping_rounds=5,
        learning_rate=0.2, depth=2, random_state=0
    ).fit(Xtr, ytr, eval_set=(Xv, yv), eval_sample_weight=hard_weight)

    assert easy.model_.valid_history_[0] != hard.model_.valid_history_[0]
    assert easy.best_iteration_ != hard.best_iteration_


def test_weighted_categorical_target_encoding_changes_stats():
    from darkofit.preprocessing import FeaturePreprocessor

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


def test_shared_split_empty_children_are_zero_gain_with_l2_zero():
    """A pure leaf must not veto a useful shared split or divide by zero."""
    import numba
    from darkofit.tree import _best_split, _best_split_serial

    # Leaf 0 gains 1.0 from the threshold.  Leaf 1 is already pure for that
    # threshold, so its empty right child contributes zero rather than vetoing
    # the shared split.  l2=0 proves gain math never divides by that empty side.
    hg = np.array([[[1.0, -1.0], [0.0, 0.0]]])
    hh = np.array([[[2.0, 2.0], [4.0, 0.0]]])
    n_bins = np.array([2], dtype=np.int64)
    feat_mask = np.array([1], dtype=np.int64)
    scratch = (
        *(np.empty((1, 2)) for _ in range(5)),
        np.empty((1, 2), dtype=np.int64),
    )

    serial = _best_split_serial(hg, hh, n_bins, 0.0, feat_mask, 1.0, 2)

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        parallel = _best_split(
            hg, hh, n_bins, 0.0, feat_mask, 1.0, 2,
            scratch[0], scratch[1], scratch[2], scratch[3], scratch[4],
            scratch[5],
        )
    finally:
        numba.set_num_threads(old_threads)

    for result in (serial, parallel):
        assert result[:2] == (0, 0)
        assert result[2] == pytest.approx(1.0)
        assert np.isfinite(result[2])


def test_all_shared_split_paths_exempt_only_empty_children():
    """Keep five shared-search implementations aligned on child legality."""
    import numba
    from darkofit.tree import (
        _best_shared_split_counts,
        _best_shared_split_counts_with_noise_py,
        _best_split,
        _best_split_serial,
        _best_split_with_noise_py,
        _best_splits_by_leaf,
        _best_splits_by_leaf_counts,
    )

    n_bins = np.array([2], dtype=np.int64)
    feat_mask = np.array([1], dtype=np.int64)

    def run_all(hg, hh, hc, min_child_weight=1.0, min_child_samples=2):
        scratch = (
            *(np.empty((1, 2)) for _ in range(5)),
            np.empty((1, 2), dtype=np.int64),
        )
        old_threads = numba.get_num_threads()
        try:
            numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
            parallel = _best_split(
                hg, hh, n_bins, 1.0, feat_mask, min_child_weight, 2,
                scratch[0], scratch[1], scratch[2], scratch[3], scratch[4],
                scratch[5],
            )
        finally:
            numba.set_num_threads(old_threads)
        return (
            _best_split_serial(
                hg, hh, n_bins, 1.0, feat_mask, min_child_weight, 2
            ),
            parallel,
            _best_split_with_noise_py(
                hg, hh, n_bins, 1.0, feat_mask, min_child_weight, 2,
                0.0, 0, 0, 0, 0.0,
            ),
            _best_shared_split_counts(
                hg, hh, hc, n_bins, 1.0, feat_mask, min_child_weight,
                min_child_samples, 2,
            ),
            _best_shared_split_counts_with_noise_py(
                hg, hh, hc, n_bins, 1.0, feat_mask, min_child_weight,
                min_child_samples, 2, 0.0, 0, 0, 0, 0.0,
            ),
        )

    # A useful leaf plus a pure leaf is legal whether the empty child is on
    # the right or the left.  Its expected total gain is 2 / 3.
    base_hg = np.array([[[1.0, -1.0], [0.0, 0.0]]])
    base_hh = np.array([[[2.0, 2.0], [4.0, 0.0]]])
    base_hc = np.array([[[2.0, 2.0], [4.0, 0.0]]])
    for hg, hh, hc in (
        (base_hg, base_hh, base_hc),
        (base_hg[:, :, ::-1], base_hh[:, :, ::-1], base_hc[:, :, ::-1]),
    ):
        for result in run_all(hg, hh, hc):
            assert result[:2] == (0, 0)
            assert result[2] == pytest.approx(2.0 / 3.0)

    # A non-empty child below min_child_weight remains illegal everywhere.
    sparse_hh = np.array([[[2.0, 2.0], [3.5, 0.5]]])
    sparse_hc = np.array([[[2.0, 2.0], [3.0, 1.0]]])
    for result in run_all(base_hg, sparse_hh, sparse_hc):
        assert result[1] == -1
        assert result[2] == -np.inf

    # A float32 total can round away a small positive right suffix.  Structural
    # emptiness must not be inferred from that cancelled subtraction result.
    rounded_hg = base_hg.astype(np.float32)
    rounded_hh = np.array(
        [[[2.0, 2.0], [float(1 << 24), 1.0]]], dtype=np.float32
    )
    rounded_hc = np.array([[[2.0, 2.0], [2.0, 1.0]]])
    for result in run_all(
        rounded_hg,
        rounded_hh,
        rounded_hc,
        min_child_weight=1.5,
        min_child_samples=1,
    ):
        assert result[1] == -1
        assert result[2] == -np.inf

    # Counts add a separate constraint in the hybrid shared-trunk paths.
    count_only_hh = np.array([[[2.0, 2.0], [3.0, 1.0]]])
    results = run_all(base_hg, count_only_hh, sparse_hc)
    for result in results[:3]:
        assert result[:2] == (0, 0)
    for result in results[3:]:
        assert result[1] == -1
        assert result[2] == -np.inf

    # Per-leaf builders intentionally keep the strict rule: an empty child
    # means that particular leaf was not split, rather than a harmless pure
    # leaf in a split shared with productive siblings.
    out_feat = np.empty(2, dtype=np.int64)
    out_thr = np.empty(2, dtype=np.int64)
    out_gain = np.empty(2, dtype=np.float64)
    _best_splits_by_leaf(
        base_hg, base_hh, n_bins, 1.0, feat_mask, 1.0, 2,
        out_feat, out_thr, out_gain,
    )
    assert np.array_equal(out_thr, np.array([0, -1]))
    _best_splits_by_leaf_counts(
        base_hg, base_hh, base_hc, n_bins, 1.0, feat_mask, 1.0, 2, 2,
        out_feat, out_thr, out_gain,
    )
    assert np.array_equal(out_thr, np.array([0, -1]))


def test_best_split_serial_matches_parallel_histogram_search():
    """Parent-gain optimizations must not diverge between split-search paths."""
    import numba
    from darkofit.tree import _best_split, _best_split_serial

    rng = np.random.default_rng(12)
    hg = rng.normal(size=(7, 4, 8))
    hh = rng.uniform(0.05, 2.0, size=(7, 4, 8))
    n_bins = np.array([8, 7, 6, 8, 5, 4, 7], dtype=np.int64)
    feat_mask = np.array([1, 1, 0, 1, 1, 0, 1], dtype=np.int64)
    scratch = (
        *(np.empty((hg.shape[0], 4)) for _ in range(5)),
        np.empty((hg.shape[0], 4), dtype=np.int64),
    )

    serial = _best_split_serial(hg, hh, n_bins, 2.0, feat_mask, 0.1, 4)
    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        parallel = _best_split(
            hg, hh, n_bins, 2.0, feat_mask, 0.1, 4,
            scratch[0], scratch[1], scratch[2], scratch[3], scratch[4],
            scratch[5],
        )
    finally:
        numba.set_num_threads(old_threads)

    assert serial[:2] == parallel[:2]
    assert np.isclose(serial[2], parallel[2])


def test_ordered_leaf_update_l2_zero_singleton_is_finite():
    """Leave-one-out ordered updates should remain finite when l2=0."""
    from darkofit.tree import ordered_leaf_update_inplace

    leaf = np.array([0, 1, 1], dtype=np.int64)
    grad = np.array([1.0, 2.0, 3.0])
    hess = np.ones(3)
    leaf_G = np.array([1.0, 5.0])
    leaf_H = np.array([1.0, 2.0])
    F = np.zeros(3)

    ordered_leaf_update_inplace(leaf, leaf_G, leaf_H, grad, hess, 0.1, 0.0, F)

    assert np.all(np.isfinite(F))
    # The singleton leaf's leave-one-out denominator is zero, so its row gets
    # no update.
    assert F[0] == 0.0


def test_feature_contiguous_hist_layout_matches_c_order_tree_build():
    """The optional F-order histogram matrix must not change tree structure."""
    import numba
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import build_oblivious_tree

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
    m_none = DarkoRegressor(iterations=80, random_state=0).fit(Xtr, ytr)
    m_ones = DarkoRegressor(iterations=80, random_state=0).fit(
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
    m_none = DarkoClassifier(iterations=80, random_state=0).fit(Xtr, ytr)
    m_ones = DarkoClassifier(iterations=80, random_state=0).fit(
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
    m_none = DarkoClassifier(iterations=80, random_state=0).fit(Xtr, ytr)
    m_ones = DarkoClassifier(iterations=80, random_state=0).fit(
        Xtr, ytr, sample_weight=w
    )
    assert np.array_equal(m_none.predict_proba(Xte), m_ones.predict_proba(Xte))


def test_lightgbm_uniform_weights_equal_no_weights():
    from sklearn.datasets import load_wine

    cases = [
        (DarkoRegressor, load_diabetes(return_X_y=True), "predict"),
        (DarkoClassifier, load_breast_cancer(return_X_y=True), "predict_proba"),
        (DarkoClassifier, load_wine(return_X_y=True), "predict_proba"),
    ]
    for estimator_cls, (X, y), predict_name in cases:
        stratify = None if estimator_cls is DarkoRegressor else y
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

    m_base = DarkoRegressor(iterations=150, random_state=0).fit(Xtr, ytr)
    m_high = DarkoRegressor(iterations=150, random_state=0).fit(
        Xtr, ytr, sample_weight=w_high
    )
    m_low  = DarkoRegressor(iterations=150, random_state=0).fit(
        Xtr, ytr, sample_weight=w_low
    )
    mean_base = m_base.predict(Xte).mean()
    mean_high = m_high.predict(Xte).mean()
    mean_low  = m_low.predict(Xte).mean()

    # Up-weighting high-y samples → higher mean predictions, and vice-versa.
    assert mean_high > mean_base > mean_low


def test_sample_weight_early_stopping_slices_correctly():
    """When early_stopping=True, the weight array must be sliced to match the
    training split; the fit should complete and record split-consistent sums."""
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(7)
    w = rng.uniform(0.5, 2.0, len(y))
    m = DarkoClassifier(
        iterations=500, early_stopping=True, validation_fraction=0.15,
        early_stopping_rounds=20, random_state=0
    ).fit(X, y, sample_weight=w)
    meta = m.model_.auto_params_
    assert m.best_iteration_ <= 500
    assert meta["sample_weight"]["provided"] is True
    assert np.isclose(
        meta["sample_weight"]["normalized_sum"], m._selection_n_train_
    )
    assert meta["early_stopping"]["eval_sample_weight_provided"] is True
    assert meta["early_stopping"]["eval_n_samples"] == (
        m._selection_n_total_ - m._selection_n_train_
    )


def test_empty_tree_stops_boosting_early():
    """When splits are exhausted, the booster should stop rather than bank
    useless depth-0 trees until the iteration ceiling."""
    import numpy as np
    # One informative feature, aggressive min_child_weight -> splits run out fast.
    X = np.array([[0.0]] * 60 + [[1.0]] * 60)
    y = np.array([0.0] * 60 + [1.0] * 60)
    m = DarkoRegressor(iterations=1000, min_child_weight=30,
                              random_state=0).fit(X, y)
    assert len(m.model_.trees_) < 1000


def test_sampled_depth_zero_tree_retries_next_row_sample():
    from darkofit.booster import GradientBoosting

    class SingletonFirstSampleBoosting(GradientBoosting):
        def _maybe_subsample(self, grad, hess, rng):
            self.sample_calls = getattr(self, "sample_calls", 0) + 1
            if self.sample_calls == 1:
                mask = np.zeros(grad.shape[0], dtype=bool)
                mask[0] = True
                row_indices = np.array([0], dtype=np.int64)
                self._record_sampling_diagnostic(row_indices, grad.shape[0])
                return (
                    np.where(mask, grad, 0.0),
                    np.where(mask, hess, 0.0),
                    row_indices,
                )
            self._record_sampling_diagnostic(None, grad.shape[0])
            return grad, hess, None

    X = np.r_[np.zeros((30, 1)), np.ones((30, 1))]
    y = np.r_[np.zeros(30), np.ones(30)]
    model = SingletonFirstSampleBoosting(
        iterations=3,
        depth=1,
        min_child_samples=1,
        min_child_weight=0.0,
        random_state=0,
    ).fit(X, y)

    assert model.sample_calls == len(model.trees_) + 1
    assert len(model.trees_) > 0
    assert model.trees_[0].depth > 0


def test_sampled_depth_zero_retries_are_capped_on_no_signal_data():
    from darkofit.booster import _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES

    X = np.zeros((80, 2), dtype=np.float64)
    y = np.zeros(80, dtype=np.float64)
    model = DarkoRegressor(
        iterations=1000,
        subsample=0.5,
        random_state=0,
    ).fit(X, y)

    meta = model.model_.auto_params_["stochastic_regularization"]
    assert model.best_iteration_ == 0
    assert meta["sampling_rounds"] == _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES + 1


def test_multiclass_sampled_depth_zero_retries_are_capped_on_no_signal_data():
    from darkofit.booster import _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES

    X = np.zeros((90, 2), dtype=np.float64)
    y = np.tile([0, 1, 2], 30)
    model = DarkoClassifier(
        iterations=1000,
        subsample=0.5,
        random_state=0,
    ).fit(X, y)

    meta = model.model_.auto_params_["stochastic_regularization"]
    assert model.best_iteration_ == 0
    assert meta["sampling_rounds"] == _MAX_CONSECUTIVE_SAMPLED_DEPTH0_RETRIES + 1


def test_sampled_retry_best_model_uses_successful_prefix_index():
    from darkofit.booster import GradientBoosting

    class CaptureBestIterBoosting(GradientBoosting):
        def _maybe_subsample(self, grad, hess, rng):
            self.sample_calls = getattr(self, "sample_calls", 0) + 1
            if self.sample_calls == 1:
                mask = np.zeros(grad.shape[0], dtype=bool)
                mask[0] = True
                row_indices = np.array([0], dtype=np.int64)
                self._record_sampling_diagnostic(row_indices, grad.shape[0])
                return (
                    np.where(mask, grad, 0.0),
                    np.where(mask, hess, 0.0),
                    row_indices,
                )
            self._record_sampling_diagnostic(None, grad.shape[0])
            return grad, hess, None

        def _truncate_to_best_model(self, best_iter, valid_history):
            self.captured_best_iter = int(best_iter)
            super()._truncate_to_best_model(best_iter, valid_history)

    X = np.r_[np.zeros((30, 1)), np.ones((30, 1))]
    y = np.r_[np.zeros(30), np.ones(30)]
    model = CaptureBestIterBoosting(
        iterations=2,
        depth=1,
        min_child_samples=1,
        min_child_weight=0.0,
        use_best_model=True,
        random_state=0,
    ).fit(X, y, eval_set=(X, y))

    assert model.sample_calls == 2
    assert model.captured_best_iter == 0
    assert len(model.valid_history_) == len(model.trees_) == 1


def test_eval_train_loss_false_skips_train_history():
    """Train-loss evaluation (opt-in since 0.9.0) must not change the fitted
    model, only populate train_history_ (train loss never influences tree
    growth)."""
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=0)

    on = DarkoRegressor(
        iterations=40, random_state=0, eval_train_loss=True
    ).fit(Xtr, ytr)
    off = DarkoRegressor(iterations=40, random_state=0).fit(Xtr, ytr)
    assert len(on.model_.train_history_) == 40
    assert off.model_.train_history_ == []
    assert np.array_equal(on.predict(Xv), off.predict(Xv))
    # Without an eval set, best_score_ falls back to a final train evaluation.
    assert np.isclose(off.best_score_, on.model_.train_history_[-1])

    # With early stopping, the eval-set path is untouched.
    es_on = DarkoRegressor(
        iterations=300, early_stopping_rounds=20, random_state=0,
        eval_train_loss=True
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    es_off = DarkoRegressor(
        iterations=300, early_stopping_rounds=20, random_state=0,
        eval_train_loss=False
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    assert es_off.best_iteration_ == es_on.best_iteration_
    assert es_off.model_.valid_history_ == es_on.model_.valid_history_
    assert np.array_equal(es_on.predict(Xv), es_off.predict(Xv))


def test_eval_train_loss_false_multiclass_and_verbose_override(capsys):
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    on = DarkoClassifier(
        iterations=8, random_state=0, eval_train_loss=True
    ).fit(X, y)
    off = DarkoClassifier(iterations=8, random_state=0).fit(X, y)
    assert len(on.model_.train_history_) == 8
    assert off.model_.train_history_ == []
    assert np.array_equal(on.predict_proba(X), off.predict_proba(X))

    # verbose needs the train loss for its progress log, so it forces the
    # evaluation back on even when eval_train_loss=False.
    verbose = DarkoClassifier(
        iterations=8, random_state=0, eval_train_loss=False, verbose=True
    ).fit(X, y)
    capsys.readouterr()
    assert len(verbose.model_.train_history_) == 8


def test_auto_params_records_resolved_regression_context(tmp_path):
    rng = np.random.default_rng(83)
    X = rng.normal(size=(80, 4))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(0.0, 0.1, size=80)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.25, random_state=0)
    w = np.linspace(0.5, 2.0, len(ytr))
    wv = np.linspace(1.0, 3.0, len(yv))

    model = DarkoRegressor(
        iterations=6,
        learning_rate=0.07,
        depth=3,
        max_bins=16,
        early_stopping_rounds=2,
        random_state=0,
    ).fit(Xtr, ytr, eval_set=(Xv, yv), sample_weight=w, eval_sample_weight=wv)
    meta = model.model_.auto_params_

    w_norm = w * (len(w) / w.sum())
    wv_norm = wv * (len(wv) / wv.sum())
    n_eff = (w_norm.sum() ** 2) / np.dot(w_norm, w_norm)
    eval_n_eff = (wv_norm.sum() ** 2) / np.dot(wv_norm, wv_norm)

    assert meta["loss"] == "RMSE"
    assert meta["iterations"] == 6
    assert meta["learning_rate"]["resolved"] == 0.07
    assert meta["learning_rate"]["source"] == "explicit"
    assert meta["learning_rate"]["input"] == 0.07
    assert meta["learning_rate"]["rule"] == "explicit"
    assert meta["learning_rate"]["p_model"] == 4
    assert meta["learning_rate"]["feature_multiplier"] == 1.0
    assert meta["learning_rate"]["feature_shrinkage_active"] is False
    assert meta["learning_rate"]["clipped"] is False
    assert meta["sample_weight"]["provided"] is True
    assert np.isclose(meta["sample_weight"]["effective_sample_size"], n_eff)
    assert meta["features"]["raw_feature_count"] == 4
    assert meta["features"]["model_feature_count"] == 4
    assert meta["features"]["feature_expansion_factor"] == 1.0
    assert meta["tree"]["tree_mode"] == "catboost"
    assert meta["tree"]["depth"] == 3
    assert meta["tree"]["max_leaves"] == 8
    assert np.isclose(meta["tree"]["l2_leaf_reg"], model.model_.l2_leaf_reg)
    assert meta["tree"]["l2_leaf_reg"] > 3.0
    assert meta["tree"]["min_child_samples"] == 20
    assert meta["tree"]["min_child_weight"] == 1.0
    assert meta["binning"]["max_bins"] == 16
    assert meta["binning"]["numeric_binning_weighted"] is True
    assert meta["binning"]["weighted_sampling"] is False
    assert meta["early_stopping"]["enabled"] is True
    assert meta["early_stopping"]["rounds"] == 2
    assert meta["early_stopping"]["best_prefix_policy"] == "validation_best_prefix"
    assert meta["early_stopping"]["eval_n_samples"] == len(yv)
    assert np.isclose(meta["early_stopping"]["eval_effective_sample_size"], eval_n_eff)
    assert meta["diagnostics"]["warnings"] == []
    assert meta["diagnostics"]["weighted_binning_active"] is True
    assert meta["diagnostics"]["best_prefix_policy"] == "validation_best_prefix"

    path = tmp_path / "reg.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    assert loaded.model_.auto_params_["tree"]["max_leaves"] == 8
    assert loaded.model_.auto_params_["learning_rate"]["resolved"] == 0.07


def test_auto_params_records_classifier_context():
    X, y = load_breast_cancer(return_X_y=True)
    model = DarkoClassifier(
        iterations=3,
        tree_mode="lightgbm",
        random_state=0,
        eval_train_loss=False,
    ).fit(X, y)
    meta = model.model_.auto_params_

    assert meta["loss"] == "Logloss"
    assert meta["learning_rate"]["source"] == "auto"
    assert meta["learning_rate"]["p_model"] == X.shape[1]
    assert meta["learning_rate"]["feature_multiplier"] <= 1.0
    assert meta["auto_policy"]["lightgbm_unweighted_lr_multiplier"] == 0.421916
    assert meta["features"]["raw_feature_count"] == X.shape[1]
    assert meta["tree"]["tree_mode"] == "lightgbm"
    assert meta["tree"]["max_leaves"] == 31
    assert meta["tree"]["l2_leaf_reg"] == 1.0
    assert meta["early_stopping"]["enabled"] is False


def test_sklearn_default_l2_leaf_reg_is_auto():
    reg = DarkoRegressor()
    clf = DarkoClassifier()

    assert reg.l2_leaf_reg == "auto"
    assert clf.l2_leaf_reg == "auto"
    assert reg.get_params()["l2_leaf_reg"] == "auto"
    assert clf.get_params()["l2_leaf_reg"] == "auto"


def test_explicit_classifier_l2_leaf_reg_is_preserved_in_lightgbm_mode():
    X, y = load_breast_cancer(return_X_y=True)
    model = DarkoClassifier(
        iterations=2,
        tree_mode="lightgbm",
        l2_leaf_reg=3.0,
        random_state=0,
    ).fit(X[:80], y[:80])

    assert model.model_.l2_leaf_reg == 3.0
    assert model.model_.auto_params_["tree"]["l2_leaf_reg"] == 3.0


def test_early_stopping_rejects_string_values():
    X, y = load_diabetes(return_X_y=True)

    with pytest.raises(ValueError, match="early_stopping must be a bool"):
        DarkoRegressor(
            iterations=2, early_stopping="auto", random_state=0
        ).fit(X[:40], y[:40])

    with pytest.raises(ValueError, match="early_stopping must be a bool"):
        DarkoClassifier(
            iterations=2, early_stopping="false", random_state=0
        ).fit(X[:40], (y[:40] > np.median(y[:40])).astype(int))


def test_early_stopping_min_delta_records_legacy_explicit_and_auto():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] - 0.25 * X[:, 1] + rng.normal(0.0, 0.05, size=120)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.25, random_state=0)

    legacy = DarkoRegressor(
        iterations=4,
        early_stopping_rounds=2,
        random_state=0,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    explicit = DarkoRegressor(
        iterations=4,
        early_stopping_rounds=2,
        early_stopping_min_delta=0.123,
        random_state=0,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    auto = DarkoRegressor(
        iterations=4,
        early_stopping_rounds=2,
        early_stopping_min_delta="auto",
        random_state=0,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))

    assert legacy.model_.auto_params_["early_stopping"]["min_delta"] == 1e-9
    assert legacy.model_.auto_params_["early_stopping"]["min_delta_rule"] == "legacy_1e-9"
    assert explicit.model_.auto_params_["early_stopping"]["min_delta"] == 0.123
    assert explicit.model_.auto_params_["early_stopping"]["min_delta_rule"] == "explicit"
    assert auto.model_.auto_params_["early_stopping"]["min_delta"] > 0.0
    assert auto.model_.auto_params_["early_stopping"]["min_delta_rule"] == "auto"


def test_early_stopping_min_delta_does_not_gate_best_prefix_argmin():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    kw = dict(
        iterations=80,
        learning_rate=0.03,
        depth=2,
        early_stopping_min_delta=10.0,
        random_state=0,
    )

    keep_all = DarkoRegressor(
        **kw, use_best_model=False
    ).fit(Xtr, ytr, eval_set=(Xv, yv))
    best_n = int(np.argmin(keep_all.model_.valid_history_)) + 1
    assert best_n > 1

    best = DarkoRegressor(**kw).fit(Xtr, ytr, eval_set=(Xv, yv))
    assert best.best_score_ == min(keep_all.model_.valid_history_)
    assert best.n_estimators_ == best_n


def test_validation_fraction_auto_weighted_stratified_and_refit_metadata():
    rng = np.random.default_rng(102)
    X = rng.normal(size=(96, 5))
    y = np.r_[rng.normal(-2.0, 0.2, 48), rng.normal(2.0, 0.2, 48)]
    w = np.ones(96)
    w[:6] = 25.0

    model = DarkoRegressor(
        iterations=6,
        early_stopping=True,
        validation_fraction="auto",
        validation_strategy="weighted_stratified",
        refit=True,
        random_state=0,
    ).fit(X, y, sample_weight=w)

    split = model.model_.auto_params_["validation_split"]
    assert split["source"] == "refit_full_data"
    assert split["selection_source"] == "automatic"
    assert split["validation_strategy"] == "weighted_stratified"
    assert split["realized_validation_strategy"] == "refit_full_data"
    assert split["validation_fraction_resolved"] is None
    assert split["eval_n_samples"] is None
    assert split["refit"] is True
    selection_split = model.model_.auto_params_["selection_validation_split"]
    assert selection_split["source"] == "automatic"
    assert selection_split["validation_fraction_input"] == "auto"
    assert selection_split["validation_strategy"] == "weighted_stratified"
    assert selection_split["realized_validation_strategy"] == "weighted_target_stratified"
    assert 0.10 <= selection_split["validation_fraction_resolved"] <= 0.25
    assert model.refit_ is True


def test_tree_mode_auto_selects_records_refits_and_round_trips(tmp_path):
    rng = np.random.default_rng(108)
    X = rng.normal(size=(120, 5))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(0.0, 0.1, size=120)

    model = DarkoRegressor(
        iterations=5,
        tree_mode="auto",
        validation_fraction=0.2,
        refit=True,
        random_state=0,
        eval_train_loss=False,
    ).fit(X, y)

    selection = model.model_.auto_params_["tree_mode_selection"]
    assert selection["enabled"] is True
    assert {c["tree_mode"] for c in selection["candidates"]} == {
        "catboost", "lightgbm", "hybrid",
    }
    assert selection["selected_tree_mode"] in {
        "catboost", "lightgbm", "hybrid",
    }
    assert model.model_.tree_mode_ == selection["selected_tree_mode"]
    assert model.selection_model_.tree_mode_ == selection["selected_tree_mode"]
    assert model.refit_ is True
    assert model.get_refit_params()["tree_mode"] == selection["selected_tree_mode"]

    split = model.model_.auto_params_["validation_split"]
    assert split["source"] == "refit_full_data"
    assert split["selection_source"] == "automatic_tree_mode_selection"
    selection_split = model.model_.auto_params_["selection_validation_split"]
    assert selection_split["source"] == "automatic_tree_mode_selection"

    path = tmp_path / "auto-reg.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    assert loaded.tree_mode_selection_ == model.tree_mode_selection_
    assert loaded.model_.auto_params_["tree_mode_selection"] == selection
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_tree_mode_auto_classifier_uses_explicit_eval_set():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(
        X[:180], y[:180], test_size=0.25, random_state=0, stratify=y[:180]
    )

    model = DarkoClassifier(
        iterations=4,
        tree_mode="auto",
        random_state=0,
        eval_train_loss=False,
    ).fit(Xtr, ytr, eval_set=(Xv, yv))

    selection = model.model_.auto_params_["tree_mode_selection"]
    assert selection["selected_tree_mode"] == model.model_.tree_mode_
    assert len(selection["candidates"]) == 3
    split = model.model_.auto_params_["validation_split"]
    assert split["source"] == "explicit_eval_set"
    assert split["refit"] is False
    assert model.predict_proba(Xv).shape == (len(Xv), 2)


def test_tree_mode_auto_rejects_ordered_boosting_true():
    X, y = load_diabetes(return_X_y=True)

    for ordered_boosting in (True, 1):
        with pytest.raises(ValueError, match="tree_mode='auto'"):
            DarkoRegressor(
                iterations=2,
                tree_mode="auto",
                ordered_boosting=ordered_boosting,
                random_state=0,
            ).fit(X[:60], y[:60])


def test_tree_mode_selection_score_uses_retained_model_score():
    from types import SimpleNamespace

    helper = DarkoRegressor()
    retained = SimpleNamespace(
        valid_history_=[3.0, 1.0, 2.5],
        best_score_=1.0,
        use_best_model_=False,
    )
    truncated = SimpleNamespace(
        valid_history_=[3.0, 1.0, 2.5],
        best_score_=1.0,
        use_best_model_=True,
    )

    assert helper._tree_mode_selection_score(retained) == 2.5
    assert helper._tree_mode_selection_score(truncated) == 1.0


def test_tree_mode_auto_preserves_callers_numba_thread_mask():
    import numba

    if numba.config.NUMBA_NUM_THREADS <= 1:
        pytest.skip("requires two available Numba threads")

    class FakeAutoModel:
        def __init__(self, kwargs):
            self.tree_mode_ = kwargs["tree_mode"]
            self.n_threads_ = 1 if self.tree_mode_ == "catboost" else 2
            self.best_score_ = 0.0 if self.tree_mode_ == "catboost" else 10.0
            self.valid_history_ = [self.best_score_]
            self.use_best_model_ = False
            self.best_iteration_ = 1
            self.trees_ = [object()]
            self.lr_ = 0.1

        def fit(self, *args, **kwargs):
            return self

    previous_threads = numba.get_num_threads()
    ambient_threads = min(4, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(ambient_threads)
    try:
        helper = DarkoRegressor(tree_mode="auto")
        X = np.zeros((6, 2), dtype=np.float64)
        y = np.zeros(6, dtype=np.float64)

        model, _, metadata = helper._fit_tree_mode_auto(
            lambda kwargs: FakeAutoModel(kwargs),
            {"iterations": 1},
            X,
            y,
            cat_features=None,
            eval_set=(X[:2], y[:2]),
            sample_weight=None,
            eval_sample_weight=None,
        )

        assert model.tree_mode_ == "catboost"
        assert metadata["selected_tree_mode"] == "catboost"
        assert numba.get_num_threads() == ambient_threads
    finally:
        numba.set_num_threads(previous_threads)


def test_learning_rate_probe_scores_retained_model_prefix():
    class FakeProbeModel:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.auto_params_ = {
                "features": {"model_feature_count": 3},
                "learning_rate": {"p_model": 3},
            }
            self.tree_mode_ = kwargs.get("tree_mode", "catboost")
            self.loss_name = "RMSE"
            self.use_best_model_ = False
            self.valid_history_ = [5.0, 1.0, 4.0]
            self.best_score_ = 1.0
            self.best_iteration_ = 2

        def fit(self, *args, **kwargs):
            return self

        def _max_tree_leaves(self):
            return 8

    helper = DarkoRegressor(
        iterations=10,
        learning_rate=None,
        auto_learning_rate_probe=True,
        auto_learning_rate_probe_iterations=3,
    )
    X = np.zeros((6, 3), dtype=np.float64)
    y = np.zeros(6, dtype=np.float64)
    Xv = np.zeros((2, 3), dtype=np.float64)
    yv = np.zeros(2, dtype=np.float64)

    best_lr, meta = helper._run_learning_rate_probe(
        lambda kwargs: FakeProbeModel(kwargs),
        X,
        y,
        cat_features=None,
        eval_set=(Xv, yv),
        sample_weight=None,
        eval_sample_weight=None,
        fit_kwargs={"iterations": 10, "tree_mode": "catboost"},
    )

    assert best_lr == meta["selected_learning_rate"]
    assert meta["enabled"] is True
    assert meta["selected_score"] == 4.0
    assert {candidate["score"] for candidate in meta["candidates"]} == {4.0}


def test_auto_validation_rejects_eval_sample_weight_without_eval_set():
    X = np.arange(120, dtype=np.float64).reshape(60, 2)
    y_reg = np.linspace(-1.0, 1.0, 60)
    y_cls = np.tile([0, 1], 30)
    message = "eval_sample_weight requires an explicit eval_set"

    with pytest.raises(ValueError, match=message):
        DarkoRegressor(
            iterations=2,
            tree_mode="auto",
            validation_fraction=0.2,
            random_state=0,
        ).fit(X, y_reg, eval_sample_weight=np.ones(12))

    with pytest.raises(ValueError, match=message):
        DarkoClassifier(
            iterations=2,
            tree_mode="auto",
            validation_fraction=0.2,
            random_state=0,
        ).fit(X, y_cls, eval_sample_weight=np.ones(12))


def test_auto_validation_split_requires_positive_weight_mass_on_both_sides():
    from darkofit.sklearn_api import _make_eval_split

    X = np.arange(80, dtype=np.float64).reshape(40, 2)
    y = np.linspace(-1.0, 1.0, 40)
    weights = np.zeros(40, dtype=np.float64)
    weights[0] = 1.0

    with pytest.raises(ValueError, match="positive sample_weight mass"):
        _make_eval_split(
            X, y, 0.2, random_state=0, sample_weight=weights
        )


def test_weighted_stratified_rejects_silent_noop_split_modes():
    X = np.arange(80, dtype=np.float64).reshape(40, 2)
    y_reg = np.linspace(-1.0, 1.0, 40)
    groups = np.repeat(np.arange(20), 2)

    with pytest.raises(ValueError, match="ungrouped regression"):
        DarkoRegressor(
            iterations=2,
            early_stopping=True,
            validation_strategy="weighted_stratified",
        ).fit(X, y_reg, groups=groups)

    y_cls = np.r_[np.zeros(20), np.ones(20)]
    with pytest.raises(ValueError, match="regression automatic validation"):
        DarkoClassifier(
            iterations=2,
            early_stopping=True,
            validation_strategy="weighted_stratified",
        ).fit(X, y_cls)


def test_group_validation_strategy_is_explicit_and_matches_legacy_group_split():
    from darkofit.sklearn_api import _make_eval_split

    X = np.arange(240, dtype=np.float64).reshape(80, 3)
    y = np.linspace(-1.0, 1.0, 80)
    groups = np.repeat(np.arange(20), 4)

    explicit_train, explicit_val, explicit_policy = _make_eval_split(
        X,
        y,
        0.2,
        random_state=17,
        groups=groups,
        validation_strategy="group",
    )
    legacy_train, legacy_val, legacy_policy = _make_eval_split(
        X,
        y,
        0.2,
        random_state=17,
        groups=groups,
        validation_strategy="random",
    )

    assert np.array_equal(explicit_train, legacy_train)
    assert np.array_equal(explicit_val, legacy_val)
    assert explicit_policy == legacy_policy == "group_shuffle"
    assert set(groups[explicit_train]).isdisjoint(groups[explicit_val])


def test_group_validation_strategy_requires_groups_and_records_intent():
    X = np.arange(240, dtype=np.float64).reshape(80, 3)
    y = np.linspace(-1.0, 1.0, 80)
    groups = np.repeat(np.arange(20), 4)

    with pytest.raises(
        ValueError, match="validation_strategy='group' requires groups"
    ):
        DarkoRegressor(
            iterations=2,
            early_stopping=True,
            validation_strategy="group",
        ).fit(X, y)

    model = DarkoRegressor(
        iterations=3,
        early_stopping=True,
        validation_strategy="group",
        random_state=17,
    ).fit(X, y, groups=groups)

    split = model.model_.auto_params_["validation_split"]
    assert split["validation_strategy"] == "group"
    assert split["realized_validation_strategy"] == "group_shuffle"
    assert split["groups_provided"] is True


def test_group_validation_strategy_stratifies_classification_without_leakage():
    from darkofit.sklearn_api import _make_eval_split

    X = np.arange(240, dtype=np.float64).reshape(80, 3)
    groups = np.repeat(np.arange(20), 4)
    y = np.tile(np.array([0, 0, 1, 1]), 20)

    train, val, policy = _make_eval_split(
        X,
        y,
        0.2,
        random_state=19,
        groups=groups,
        stratify=y,
        validation_strategy="group",
    )

    assert policy == "class_stratified_group"
    assert set(groups[train]).isdisjoint(groups[val])
    assert set(y[train]) == {0, 1}
    assert set(y[val]) == {0, 1}


def test_weighted_stratified_default_fraction_caps_small_regression_strata():
    rng = np.random.default_rng(107)
    X = rng.normal(size=(40, 4))
    y = np.linspace(-2.0, 2.0, 40) + rng.normal(0.0, 0.01, 40)
    w = np.linspace(0.5, 3.0, 40)

    model = DarkoRegressor(
        iterations=4,
        early_stopping=True,
        validation_fraction=0.1,
        validation_strategy="weighted_stratified",
        random_state=0,
    ).fit(X, y, sample_weight=w)

    split = model.model_.auto_params_["validation_split"]
    assert split["validation_strategy"] == "weighted_stratified"
    assert split["realized_validation_strategy"] == "weighted_target_stratified"
    assert split["eval_n_samples"] == 4


def test_wrapper_validates_sample_weight_before_auto_validation_split():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    y = np.linspace(-1.0, 1.0, 20)

    with pytest.raises(ValueError, match=r"sample_weight must have shape"):
        DarkoRegressor(
            iterations=2,
            early_stopping=True,
            validation_fraction="auto",
            validation_strategy="weighted_stratified",
        ).fit(X, y, sample_weight=np.ones(19))

    bad = np.ones(20)
    bad[0] = -1.0
    with pytest.raises(ValueError, match="sample_weight must be nonnegative"):
        DarkoRegressor(
            iterations=2,
            early_stopping=True,
            validation_fraction="auto",
        ).fit(X, y, sample_weight=bad)


def test_auto_structure_and_cat_smoothing_are_opt_in_and_recorded():
    rng = np.random.default_rng(103)
    numeric = rng.normal(size=(140, 3)).astype(object)
    cats = np.array([f"c{j % 9}" for j in range(140)], dtype=object)[:, None]
    X = np.concatenate([numeric, cats], axis=1)
    y = rng.normal(size=140)
    w = np.ones(140)
    w[:8] = 12.0

    model = DarkoRegressor(
        iterations=3,
        depth="auto",
        l2_leaf_reg="auto",
        min_child_samples="auto",
        min_child_weight="auto",
        cat_smoothing="auto",
        random_state=0,
    ).fit(X, y, cat_features=[3], sample_weight=w)

    meta = model.model_.auto_params_["auto_structure"]["resolved"]
    assert meta["depth"]["source"] == "auto"
    assert isinstance(model.model_.depth, int)
    assert meta["l2_leaf_reg"]["source"] == "auto"
    assert meta["min_child_samples"]["source"] == "auto"
    assert meta["min_child_weight"]["source"] == "auto"
    assert meta["cat_smoothing"]["source"] == "auto"
    assert model.model_.prep_.cat_smoothing == meta["cat_smoothing"]["resolved"]


def test_learning_rate_probe_is_opt_in_and_records_candidates():
    rng = np.random.default_rng(104)
    X = rng.normal(size=(110, 4))
    y = X[:, 0] + rng.normal(0.0, 0.1, size=110)

    disabled = DarkoRegressor(
        iterations=1000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=0,
    ).fit(X, y)
    probed = DarkoRegressor(
        iterations=1000,
        early_stopping=True,
        validation_fraction=0.2,
        auto_learning_rate_probe=True,
        auto_learning_rate_probe_iterations=3,
        random_state=0,
    ).fit(X, y)

    disabled_meta = disabled.model_.auto_params_["learning_rate_probe"]
    probe_meta = probed.model_.auto_params_["learning_rate_probe"]
    assert disabled_meta == {"enabled": False, "reason": "disabled"}
    assert probe_meta["enabled"] is True
    assert probe_meta["probe_iterations"] == 3
    assert probe_meta["final_iterations"] == 1000
    assert probe_meta["base_learning_rate"] == disabled.model_.lr_
    assert probe_meta["base_learning_rate_full_iterations"] == disabled.model_.lr_
    assert probe_meta["base_learning_rate_short_iterations"] != disabled.model_.lr_
    assert len(probe_meta["candidates"]) >= 5
    assert sum(c["source"] == "auto_base" for c in probe_meta["candidates"]) == 1
    assert probed.model_.lr_ == probe_meta["selected_learning_rate"]


def test_ordered_leaf_update_inplace_matches_numpy_formula():
    from darkofit.tree import ordered_leaf_update_inplace

    rng = np.random.default_rng(11)
    n, n_leaves = 500, 8
    leaf = rng.integers(0, n_leaves, size=n).astype(np.int64)
    grad = rng.normal(size=n)
    hess = rng.uniform(0.1, 2.0, size=n)
    # Force one singleton leaf so the zero-denominator fallback is exercised
    # when l2 = 0.
    leaf[0] = n_leaves - 1
    leaf[leaf == n_leaves - 1] = 0
    leaf[0] = n_leaves - 1
    leaf_G = np.zeros(n_leaves)
    leaf_H = np.zeros(n_leaves)
    np.add.at(leaf_G, leaf, grad)
    np.add.at(leaf_H, leaf, hess)

    for l2 in (0.0, 3.0):
        F_kernel = rng.normal(size=n)
        F_numpy = F_kernel.copy()
        lr = 0.1

        numerator = leaf_G[leaf] - grad
        denominator = np.maximum(leaf_H[leaf] - hess, 0.0) + l2
        update = np.zeros_like(numerator)
        np.divide(-lr * numerator, denominator, out=update,
                  where=denominator > 0.0)
        F_numpy += update

        ordered_leaf_update_inplace(
            leaf, leaf_G, leaf_H, grad, hess, lr, l2, F_kernel
        )
        assert np.array_equal(F_kernel, F_numpy)


def test_bin_transform_kernel_matches_searchsorted_reference():
    """The numba binning kernel must reproduce the numpy searchsorted path
    bit-for-bit, including NaN/inf routing and low-cardinality columns."""
    from darkofit.binning import Binner

    rng = np.random.default_rng(5)
    n = 3000
    X = np.column_stack([
        rng.normal(size=n),                       # smooth numeric
        rng.integers(0, 3, n).astype(float),      # low cardinality
        np.full(n, 7.0),                          # constant (no borders)
        rng.normal(size=n),                       # gets NaN/inf injected
        np.full(n, np.nan),                       # all-missing column
    ])
    X[::7, 3] = np.nan
    X[3::11, 3] = np.inf
    X[5::13, 3] = -np.inf

    binner = Binner(max_bins=16).fit(X)
    got = binner.transform(X)

    for f in range(X.shape[1]):
        borders = binner.borders_[f]
        col = X[:, f]
        nan_bin = len(borders) + 1
        ref = np.searchsorted(borders, col, side="right")
        ref[~np.isfinite(col)] = nan_bin
        assert np.array_equal(got[:, f].astype(np.int64), ref), f"feature {f}"


def test_binner_sampling_is_deterministic_and_off_for_small_data():
    from darkofit.binning import Binner

    rng = np.random.default_rng(8)
    X = rng.normal(size=(5000, 3))

    # n <= sample_count: borders must be identical to the unsampled path.
    full = Binner(max_bins=32, sample_count=None).fit(X)
    capped = Binner(max_bins=32, sample_count=5000, random_state=0).fit(X)
    for a, b in zip(full.borders_, capped.borders_):
        assert np.array_equal(a, b)

    # n > sample_count: deterministic under a fixed seed, borders stay sane.
    s1 = Binner(max_bins=32, sample_count=1000, random_state=0).fit(X)
    s2 = Binner(max_bins=32, sample_count=1000, random_state=0).fit(X)
    for a, b in zip(s1.borders_, s2.borders_):
        assert np.array_equal(a, b)
    t = s1.transform(X)
    assert t.max() < s1.n_bins_.max()
    # Sampled borders approximate the full-data quantiles.
    assert all(
        len(b) > 0 and np.all(np.diff(b) > 0) for b in s1.borders_
    )


def test_binner_all_ones_weights_match_unweighted_borders():
    from darkofit.binning import Binner

    rng = np.random.default_rng(84)
    X = rng.normal(size=(200, 4))
    X[::9, 2] = np.nan
    unweighted = Binner(max_bins=16, sample_count=None).fit(X)
    weighted = Binner(max_bins=16, sample_count=None).fit(
        X, sample_weight=np.ones(len(X))
    )

    assert weighted.weighted_ is False
    for a, b in zip(unweighted.borders_, weighted.borders_):
        assert np.array_equal(a, b)
    assert np.array_equal(unweighted.transform(X), weighted.transform(X))


def test_binner_weighted_borders_follow_weighted_mass():
    from darkofit.binning import Binner

    X = np.arange(100, dtype=float).reshape(-1, 1)
    weights = np.ones(100)
    weights[-10:] = 100.0

    unweighted = Binner(max_bins=4, sample_count=None).fit(X)
    weighted = Binner(max_bins=4, sample_count=None).fit(
        X, sample_weight=weights
    )

    assert weighted.weighted_ is True
    assert weighted.borders_[0][0] > unweighted.borders_[0][0]
    assert weighted.borders_[0][-1] > unweighted.borders_[0][-1]


def test_binner_sampled_weighted_borders_use_weights_once():
    from darkofit.binning import Binner

    rng = np.random.default_rng(85)
    X = rng.normal(size=(500, 3))
    weights = np.linspace(0.25, 4.0, len(X))
    sample_count = 80
    seed = 17

    sampled = Binner(
        max_bins=16, sample_count=sample_count, random_state=seed
    ).fit(X, sample_weight=weights)
    sample_idx = np.sort(
        np.random.default_rng(seed).choice(len(X), sample_count, replace=False)
    )
    expected = Binner(max_bins=16, sample_count=None).fit(
        X[sample_idx], sample_weight=weights[sample_idx]
    )

    assert sampled.weighted_ is True
    assert sampled.weighted_sampling_ is True
    assert sampled.weighted_sample_count_ == sample_count
    for got, want in zip(sampled.borders_, expected.borders_):
        assert np.array_equal(got, want)


def test_binner_weighted_sampling_uses_positive_weight_support():
    from darkofit.binning import Binner

    X = np.arange(500, dtype=np.float64).reshape(-1, 1)
    positive_idx = np.arange(490, 500)
    weights = np.zeros(500, dtype=np.float64)
    weights[positive_idx] = np.arange(1, positive_idx.size + 1, dtype=float)

    sampled = Binner(
        max_bins=4, sample_count=50, random_state=0
    ).fit(X, sample_weight=weights)
    expected = Binner(max_bins=4, sample_count=None).fit(
        X[positive_idx], sample_weight=weights[positive_idx]
    )

    assert sampled.weighted_ is True
    assert sampled.weighted_sampling_ is True
    assert sampled.weighted_sample_count_ == positive_idx.size
    assert np.array_equal(sampled.borders_[0], expected.borders_[0])


def test_preprocessor_blocks_match_stacked_reference():
    """Binning blocks separately must equal binning the hstacked matrix."""
    from darkofit.binning import Binner
    from darkofit.preprocessing import FeaturePreprocessor

    rng = np.random.default_rng(2)
    n = 1500
    region = rng.choice(["a", "b", "c", "d"], n)
    X = np.empty((n, 4), dtype=object)
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = region
    X[:, 2] = rng.normal(size=n) * 10
    X[:, 3] = rng.choice(["x", "y"], n)
    y = rng.normal(size=n)

    prep = FeaturePreprocessor(64, 1.0, 0, include_cat_codes=True,
                               target_encoding_mode="kfold")
    Xb = prep.fit_transform(X, [y], cat_features=[1, 3])

    num = np.asarray(X[:, [0, 2]], dtype=np.float64)
    codes = prep._codes_for_transform(X)  # same X -> same codes as fit time

    # Fit-time check: borders must equal those learned from the hstacked
    # matrix the old implementation materialized. The encoder is seeded, so
    # replaying it reproduces the fit-time (out-of-fold) encoded block.
    from darkofit.target_encoding import OrderedTargetEncoder
    enc_replay = OrderedTargetEncoder(1.0, 0, mode="kfold", n_folds=20)
    encoded_fit = enc_replay.fit_transform(codes, y)
    stacked_fit = np.hstack([num, codes.astype(np.float64), encoded_fit])
    ref = Binner(max_bins=64).fit(stacked_fit)
    assert len(ref.borders_) == len(prep.binner_.borders_)
    for a, b in zip(ref.borders_, prep.binner_.borders_):
        assert np.array_equal(a, b)

    # Transform-time check: blockwise binning must equal searchsorted over
    # the stacked transform-time matrix (full-total encodings).
    raw_codes = codes.astype(np.float64)
    raw_codes[raw_codes < 0] = np.nan
    encoded = prep.encoders_[0].transform(codes)
    stacked_transform = np.hstack([num, raw_codes, encoded])
    Xb_t = prep.transform(X)
    for f in range(stacked_transform.shape[1]):
        got = Xb_t[:, f].astype(np.int64)
        borders = prep.binner_.borders_[f]
        want = np.searchsorted(borders, stacked_transform[:, f], side="right")
        want[~np.isfinite(stacked_transform[:, f])] = len(borders) + 1
        assert np.array_equal(got, want), f"column {f}"
    assert Xb.shape == Xb_t.shape


def test_ts_permutations_one_matches_default_ordered_statistics():
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.target_encoding import OrderedTargetEncoder

    rng = np.random.default_rng(411)
    codes = rng.integers(0, 5, size=(180, 2), dtype=np.int64)
    y = rng.normal(size=180)
    weights = rng.uniform(0.1, 2.0, size=180)

    default = OrderedTargetEncoder(1.7, 19, mode="ordered")
    explicit = OrderedTargetEncoder(1.7, 19, mode="ordered", ts_permutations=1)
    default_out = default.fit_transform(codes, y, sample_weight=weights)
    explicit_out = explicit.fit_transform(codes, y, sample_weight=weights)

    assert np.array_equal(default_out, explicit_out)
    for got, want in zip(explicit.sums_, default.sums_):
        assert np.array_equal(got, want)
    for got, want in zip(explicit.counts_, default.counts_):
        assert np.array_equal(got, want)

    X = np.empty((180, 3), dtype=object)
    X[:, 0] = rng.normal(size=180)
    X[:, 1] = rng.choice(["a", "b", "c", "d"], size=180)
    X[:, 2] = rng.choice(["x", "y"], size=180)
    prep_default = FeaturePreprocessor(32, 1.0, 7)
    prep_p1 = FeaturePreprocessor(32, 1.0, 7, ts_permutations=1)
    Xb_default = prep_default.fit_transform(X, [y], cat_features=[1, 2])
    Xb_p1 = prep_p1.fit_transform(X, [y], cat_features=[1, 2])

    assert np.array_equal(Xb_default, Xb_p1)
    assert np.array_equal(
        prep_default.encoders_[0].transform(prep_default._codes_for_transform(X)),
        prep_p1.encoders_[0].transform(prep_p1._codes_for_transform(X)),
    )


def test_multi_permutation_ordered_statistics_exclude_own_label():
    from darkofit.target_encoding import OrderedTargetEncoder

    y = np.linspace(-3.0, 4.0, 40)
    codes = np.arange(y.shape[0], dtype=np.int64).reshape(-1, 1)
    weights = np.linspace(0.5, 2.0, y.shape[0])

    for p in (2, 5):
        enc = OrderedTargetEncoder(0.75, 3, ts_permutations=p)
        got = enc.fit_transform(codes, y)
        assert np.array_equal(got[:, 0], np.full(y.shape[0], np.mean(y)))

        weighted = OrderedTargetEncoder(0.75, 3, ts_permutations=p)
        got_weighted = weighted.fit_transform(codes, y, sample_weight=weights)
        prior = np.average(y, weights=weights)
        assert np.array_equal(got_weighted[:, 0], np.full(y.shape[0], prior))


def test_multi_permutation_ordered_statistics_shrink_early_variance():
    from darkofit.target_encoding import OrderedTargetEncoder

    n = 240
    seed = 123
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    codes = np.zeros((n, 1), dtype=np.int64)

    one = OrderedTargetEncoder(1.0, seed, ts_permutations=1)
    many = OrderedTargetEncoder(1.0, seed, ts_permutations=8)
    enc_one = one.fit_transform(codes, y)[:, 0]
    enc_many = many.fit_transform(codes, y)[:, 0]
    early = np.random.default_rng(seed).permutation(n)[:40]

    assert np.var(enc_many[early]) < 0.25 * np.var(enc_one[early])


def test_ts_permutations_round_trips_and_refit_params(tmp_path):
    X, y = _cat_dataset(n=180, seed=31)
    model = DarkoRegressor(
        iterations=8,
        random_state=0,
        ts_permutations=3,
        auto_learning_rate_probe=True,
        auto_learning_rate_probe_values=[0.03, 0.1],
        auto_learning_rate_probe_iterations=3,
    )
    model.fit(X, y, cat_features=[1])

    assert model.model_.prep_.ts_permutations == 3
    assert model.model_.prep_.encoders_[0].ts_permutations == 3
    assert model.get_refit_params()["ts_permutations"] == 3

    path = tmp_path / "ts_permutations.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)

    assert loaded.get_params()["ts_permutations"] == 3
    assert loaded.model_.prep_.ts_permutations == 3
    assert loaded.model_.prep_.encoders_[0].ts_permutations == 3
    assert np.array_equal(model.predict(X[:30]), loaded.predict(X[:30]))


def test_target_ordered_raw_codes_are_monotone_and_do_not_remap_encoders():
    from darkofit.preprocessing import FeaturePreprocessor

    cats = np.array([
        "low", "low", "low", "mid", "mid", "mid",
        "high", "high", "high", "high",
    ], dtype=object)
    y = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 1.0, 1.1, 1.2, 1.3])
    X = cats.reshape(-1, 1)

    prep = FeaturePreprocessor(
        8,
        1.0,
        0,
        include_cat_codes=True,
        target_ordered_cat_codes="leaky_full",
        bin_sample_count=None,
    )
    prep.fit_transform(X, [y], cat_features=[0])

    cat_map = prep.cat_maps_[0]
    remap = prep.cat_code_remaps_[0]
    assert remap[cat_map["low"]] < remap[cat_map["mid"]] < remap[cat_map["high"]]

    codes = prep._codes_for_transform(X)
    raw = prep._raw_code_block(codes)[:, 0]
    assert raw[cats == "low"].max() < raw[cats == "mid"].min()
    assert raw[cats == "mid"].max() < raw[cats == "high"].min()

    # Encoder totals remain indexed by original factorization codes. The raw
    # code remap must not mutate the shared codes consumed by target statistics.
    enc = prep.encoders_[0]
    assert enc.counts_[0][cat_map["low"]] == 3.0
    assert enc.counts_[0][cat_map["mid"]] == 3.0
    assert enc.counts_[0][cat_map["high"]] == 4.0


def test_target_ordered_raw_code_ties_use_category_code_order():
    from darkofit.preprocessing import FeaturePreprocessor

    X = np.array(["b", "a", "c", "b", "a", "c"], dtype=object).reshape(-1, 1)
    y = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    prep = FeaturePreprocessor(
        8,
        1.0,
        0,
        include_cat_codes=True,
        target_ordered_cat_codes="leaky_full",
    )
    prep.fit_transform(X, [y], cat_features=[0])
    cat_map = prep.cat_maps_[0]
    remap = prep.cat_code_remaps_[0]

    assert cat_map["b"] == 0
    assert cat_map["a"] == 1
    assert remap[cat_map["b"]] < remap[cat_map["a"]] < remap[cat_map["c"]]


def test_target_ordered_raw_codes_are_explicit_leaky_opt_in():
    from darkofit.preprocessing import FeaturePreprocessor

    with pytest.raises(ValueError, match="leaky_full"):
        FeaturePreprocessor(target_ordered_cat_codes=True)

    X = np.array(["solo", "pair", "pair"], dtype=object).reshape(-1, 1)
    y = np.array([10.0, 0.0, 1.0])
    prep = FeaturePreprocessor(
        8,
        10.0,
        0,
        include_cat_codes=True,
        target_ordered_cat_codes="leaky_full",
    )
    prep.fit_transform(X, [y], cat_features=[0])
    assert prep.target_ordered_cat_codes == "leaky_full"
    assert len(prep.cat_code_remaps_) == 1


def test_target_ordered_cat_codes_leave_catboost_mode_unchanged():
    X, y = _cat_dataset(n=220, seed=41)
    off = DarkoRegressor(
        iterations=8,
        random_state=0,
        tree_mode="catboost",
        target_ordered_cat_codes="off",
    ).fit(X, y, cat_features=[1])
    leaky = DarkoRegressor(
        iterations=8,
        random_state=0,
        tree_mode="catboost",
        target_ordered_cat_codes="leaky_full",
    ).fit(X, y, cat_features=[1])

    assert not off.model_.prep_.include_cat_codes
    assert not leaky.model_.prep_.include_cat_codes
    assert np.array_equal(off.predict(X[:60]), leaky.predict(X[:60]))


def test_target_ordered_cat_codes_are_scalar_only_for_raw_code_block():
    X = np.empty((90, 2), dtype=object)
    X[:, 0] = np.tile(["a", "b", "c"], 30)
    X[:, 1] = np.linspace(-1.0, 1.0, 90)
    y = np.tile([0, 1, 2], 30)

    model = DarkoClassifier(
        iterations=2,
        random_state=0,
        tree_mode="lightgbm",
        target_ordered_cat_codes="leaky_full",
    )
    with pytest.raises(ValueError, match="scalar-only"):
        model.fit(X, y, cat_features=[0])


def test_target_ordered_cat_codes_round_trip_v3_with_unseen(tmp_path):
    import json

    X, y = _cat_dataset(n=260, seed=43)
    model = DarkoRegressor(
        iterations=10,
        random_state=0,
        tree_mode="lightgbm",
        target_ordered_cat_codes="leaky_full",
        max_bins=16,
    ).fit(X, y, cat_features=[1])

    path = tmp_path / "target-ordered-codes.npz"
    model.save_model(path)
    arrays = _read_archive_arrays(path)
    header = json.loads(str(arrays["header"]))

    assert header["format_version"] == 3
    assert header["prep"]["target_ordered_cat_codes"] == "leaky_full"
    assert header["prep"]["target_ordered_cat_code_policy"] == (
        "full_target_smoothed_leaky_opt_in"
    )
    assert "cat0__code_remap" in arrays

    loaded = DarkoRegressor.load_model(path)
    assert loaded.get_params()["target_ordered_cat_codes"] == "leaky_full"
    assert np.array_equal(
        loaded.model_.prep_.cat_code_remaps_[0],
        model.model_.prep_.cat_code_remaps_[0],
    )

    X_new = X[:50].copy()
    X_new[0, 1] = "brand-new"
    X_new[1, 1] = None
    assert np.array_equal(model.predict(X_new), loaded.predict(X_new))


def test_target_ordered_cat_code_archive_corruption_rejected(tmp_path):
    import json

    X, y = _cat_dataset(n=180, seed=44)
    model = DarkoRegressor(
        iterations=4,
        random_state=0,
        tree_mode="lightgbm",
        target_ordered_cat_codes="leaky_full",
    ).fit(X, y, cat_features=[1])
    path = tmp_path / "target-ordered-corrupt.npz"
    model.save_model(path)
    arrays = _read_archive_arrays(path)

    missing = dict(arrays)
    missing.pop("cat0__code_remap")
    missing_path = _write_archive(tmp_path / "missing-remap.npz", missing)
    with pytest.raises(ValueError, match="missing categorical code remap"):
        DarkoRegressor.load_model(missing_path)

    downgraded = dict(arrays)
    header = json.loads(str(downgraded["header"]))
    header["format_version"] = 2
    downgraded["header"] = np.array(json.dumps(header))
    downgraded_path = _write_archive(tmp_path / "downgraded-remap.npz", downgraded)
    with pytest.raises(ValueError, match="format version 3"):
        DarkoRegressor.load_model(downgraded_path)


def test_target_ordered_raw_codes_match_pandas_and_dict_lookup_paths():
    from darkofit.preprocessing import FeaturePreprocessor

    pytest.importorskip("pandas")

    X = np.empty((80, 2), dtype=object)
    X[:, 0] = np.tile(["a", "b", "c", None], 20)
    X[:, 1] = np.linspace(-2.0, 2.0, 80)
    y = np.where(X[:, 0] == "c", 1.0, 0.0)
    prep = FeaturePreprocessor(
        8,
        1.0,
        0,
        include_cat_codes=True,
        target_ordered_cat_codes="leaky_full",
    )
    prep.fit_transform(X, [y], cat_features=[0])
    X_new = X[:20].copy()
    X_new[0, 0] = "unseen"
    X_new[1, 0] = np.nan

    fast = prep._raw_code_block(prep._codes_for_transform(X_new))
    prep._pandas_codes_for_column = lambda pd, col, j: None
    slow = prep._raw_code_block(prep._codes_for_transform(X_new))

    assert np.array_equal(fast, slow, equal_nan=True)
    assert np.isnan(fast[0, 0])
    assert np.isfinite(fast[1, 0])


def test_codes_for_transform_pandas_path_matches_dict_path():
    from darkofit.preprocessing import FeaturePreprocessor

    pytest.importorskip("pandas")
    rng = np.random.default_rng(4)
    n = 400
    col = rng.choice(["red", "green", "blue"], n).astype(object)
    col[::17] = None
    col[3::23] = np.nan
    X = np.empty((n, 2), dtype=object)
    X[:, 0] = col
    X[:, 1] = rng.normal(size=n)
    y = rng.normal(size=n)

    prep = FeaturePreprocessor(32, 1.0, 0)
    prep.fit_transform(X, [y], cat_features=[0])

    X_new = X.copy()
    X_new[0, 0] = "purple"   # unseen -> -1
    X_new[1, 0] = None       # missing -> private sentinel code
    X_new[2, 0] = np.nan

    fast = prep._codes_for_transform(X_new)

    # Force the dict fallback by making the pandas path report failure.
    prep_fallback = FeaturePreprocessor(32, 1.0, 0)
    prep_fallback.fit_transform(X, [y], cat_features=[0])
    prep_fallback._pandas_codes_for_column = lambda pd, col, j: None
    slow = prep_fallback._codes_for_transform(X_new)

    assert np.array_equal(fast, slow)
    assert fast[0, 0] == -1


def _rowpar_test_data(n=40_000, n_feat=8, max_bins=16, seed=9):
    """Binned data plus power-of-two grad/hess so float sums are exact and
    row-parallel chunked summation must match feature-parallel bitwise."""
    rng = np.random.default_rng(seed)
    X_binned = rng.integers(0, max_bins - 1, size=(n, n_feat)).astype(np.uint8)
    n_bins = np.full(n_feat, max_bins, dtype=np.int64)
    grad = rng.integers(-8, 9, size=n).astype(np.float64)
    hess = np.choose(rng.integers(0, 3, size=n), [0.5, 1.0, 2.0])
    return X_binned, n_bins, grad, hess


def _alloc_test_rowpar(n_feat, leaf_slots, max_bins, n_arrays):
    import numba
    T = numba.get_num_threads()
    return tuple(
        np.zeros((T, n_feat, leaf_slots, max_bins)) for _ in range(n_arrays)
    )


@pytest.mark.parametrize("constant_hessian", [False, True])
def test_oblivious_rowpar_matches_feature_parallel(constant_hessian):
    import numba
    from darkofit.tree import build_oblivious_tree

    if numba.get_num_threads() < 2:
        pytest.skip("requires multithreaded numba")
    X_binned, n_bins, grad, hess = _rowpar_test_data()
    if constant_hessian:
        hess = np.ones_like(hess)
    depth = 4
    Xf = np.asfortranarray(X_binned)

    base, leaf_b, G_b, H_b = build_oblivious_tree(
        X_binned, grad, hess, n_bins, depth, 3.0, 0.1,
        X_hist_binned=Xf, constant_hessian=constant_hessian,
        return_training_state=True,
    )
    rowpar = _alloc_test_rowpar(X_binned.shape[1], 1 << depth, 16, 2)
    fast, leaf_f, G_f, H_f = build_oblivious_tree(
        X_binned, grad, hess, n_bins, depth, 3.0, 0.1,
        X_hist_binned=Xf, constant_hessian=constant_hessian,
        return_training_state=True, rowpar_buffers=rowpar,
    )
    assert np.array_equal(base.splits_feat, fast.splits_feat)
    assert np.array_equal(base.splits_thr, fast.splits_thr)
    assert np.array_equal(base.values, fast.values)
    assert np.array_equal(base.gains, fast.gains)
    assert np.array_equal(leaf_b, leaf_f)
    assert np.array_equal(G_b, G_f) and np.array_equal(H_b, H_f)


@pytest.mark.parametrize("case", ["nonconstant", "constant", "positive"])
def test_leafwise_rowpar_matches_feature_parallel(case):
    import numba
    from darkofit.tree import build_leafwise_tree

    if numba.get_num_threads() < 2:
        pytest.skip("requires multithreaded numba")
    X_binned, n_bins, grad, hess = _rowpar_test_data(n=60_000)
    kw = dict(max_leaves=31, min_child_samples=5)
    if case == "constant":
        hess = np.ones_like(hess)
        kw["constant_hessian"] = True
    elif case == "positive":
        kw["hessian_always_positive"] = True
    Xf = np.asfortranarray(X_binned)

    base, leaf_b, G_b, H_b = build_leafwise_tree(
        X_binned, grad, hess, n_bins, -1, 3.0, 0.1,
        X_hist_binned=Xf, return_training_state=True, **kw,
    )
    rowpar = _alloc_test_rowpar(X_binned.shape[1], 1, 16, 3)
    fast, leaf_f, G_f, H_f = build_leafwise_tree(
        X_binned, grad, hess, n_bins, -1, 3.0, 0.1,
        X_hist_binned=Xf, return_training_state=True,
        rowpar_buffers=rowpar, **kw,
    )
    assert np.array_equal(base.splits_feat, fast.splits_feat)
    assert np.array_equal(base.splits_thr, fast.splits_thr)
    assert np.array_equal(base.values, fast.values)
    assert np.array_equal(base.gains, fast.gains)
    assert np.array_equal(leaf_b, leaf_f)
    assert np.array_equal(G_b, G_f) and np.array_equal(H_b, H_f)
    assert fast.n_leaves > 8  # the tree actually grew through refills


def test_leafwise_rowpar_with_row_indices_matches():
    """Subsampled rows flow through row_order segments; the rowpar refill
    must agree with the feature-parallel selected-row path exactly."""
    import numba
    from darkofit.tree import build_leafwise_tree

    if numba.get_num_threads() < 2:
        pytest.skip("requires multithreaded numba")
    X_binned, n_bins, grad, hess = _rowpar_test_data(n=60_000)
    rng = np.random.default_rng(3)
    rows = np.sort(rng.choice(60_000, size=45_000, replace=False)).astype(np.int64)
    mask = np.zeros(60_000, dtype=bool)
    mask[rows] = True
    g = np.where(mask, grad, 0.0)
    h = np.where(mask, hess, 0.0)
    Xf = np.asfortranarray(X_binned)
    kw = dict(max_leaves=31, min_child_samples=5, row_indices=rows)

    base, leaf_b, G_b, H_b = build_leafwise_tree(
        X_binned, g, h, n_bins, -1, 3.0, 0.1,
        X_hist_binned=Xf, return_training_state=True, **kw,
    )
    rowpar = _alloc_test_rowpar(X_binned.shape[1], 1, 16, 3)
    fast, leaf_f, G_f, H_f = build_leafwise_tree(
        X_binned, g, h, n_bins, -1, 3.0, 0.1,
        X_hist_binned=Xf, return_training_state=True,
        rowpar_buffers=rowpar, **kw,
    )
    assert np.array_equal(base.splits_feat, fast.splits_feat)
    assert np.array_equal(base.splits_thr, fast.splits_thr)
    assert np.array_equal(base.values, fast.values)
    assert np.array_equal(leaf_b, leaf_f)
    assert np.array_equal(G_b, G_f) and np.array_equal(H_b, H_f)


def test_booster_allocates_rowpar_buffers_by_mode():
    from darkofit.booster import GradientBoosting

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30_000, 6))
    y = X[:, 0] + rng.normal(0, 0.1, 30_000)
    n_bins = np.full(6, 128, dtype=np.int64)

    # Default ('auto') keeps the feature-parallel kernels: no buffers.
    default = GradientBoosting(iterations=2, depth=4, random_state=0)
    default.fit(X, y)
    assert default._alloc_rowpar_buffers(6, n_bins, 30_000) is None

    m = GradientBoosting(iterations=2, depth=4, random_state=0,
                         histogram_parallelism="row")
    m.fit(X, y)
    buffers = m._alloc_rowpar_buffers(6, n_bins, 30_000)
    if m.n_threads_ > 1:
        assert buffers is not None and len(buffers) == 2
        assert buffers[0].shape[2] == 1 << 4
    # lightgbm mode: per-segment locals, three arrays, one leaf slot
    m2 = GradientBoosting(iterations=2, tree_mode="lightgbm", random_state=0,
                          histogram_parallelism="row")
    m2.fit(X, y)
    buffers2 = m2._alloc_rowpar_buffers(6, n_bins, 30_000)
    if m2.n_threads_ > 1:
        assert buffers2 is not None and len(buffers2) == 3
        assert buffers2[0].shape[2] == 1
    # tiny fits and huge locals fall back
    assert m._alloc_rowpar_buffers(6, n_bins, 500) is None
    assert m._alloc_rowpar_buffers(2_000_000, n_bins, 10**6) is None
    with pytest.raises(ValueError):
        GradientBoosting(histogram_parallelism="diagonal")


def test_histogram_parallelism_row_keeps_quality():
    """Opt-in row-parallel fits must train a model of the same quality."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=30_000, n_features=10, noise=10,
                           random_state=3)
    base = DarkoRegressor(iterations=60, random_state=0).fit(X, y)
    row = DarkoRegressor(iterations=60, random_state=0,
                                histogram_parallelism="row").fit(X, y)
    rmse_base = np.sqrt(np.mean((y - base.predict(X)) ** 2))
    rmse_row = np.sqrt(np.mean((y - row.predict(X)) ** 2))
    assert abs(rmse_base - rmse_row) < 0.05 * rmse_base

    lgb_base = DarkoRegressor(
        iterations=60, tree_mode="lightgbm", random_state=0
    ).fit(X, y)
    lgb_row = DarkoRegressor(
        iterations=60, tree_mode="lightgbm", random_state=0,
        histogram_parallelism="row"
    ).fit(X, y)
    rmse_lgb_base = np.sqrt(np.mean((y - lgb_base.predict(X)) ** 2))
    rmse_lgb_row = np.sqrt(np.mean((y - lgb_row.predict(X)) ** 2))
    assert abs(rmse_lgb_base - rmse_lgb_row) < 0.05 * rmse_lgb_base


def test_levelwise_rowpar_matches_feature_parallel():
    import numba
    from darkofit.tree import build_levelwise_tree

    if numba.get_num_threads() < 2:
        pytest.skip("requires multithreaded numba")
    X_binned, n_bins, grad, hess = _rowpar_test_data()
    depth = 4
    Xf = np.asfortranarray(X_binned)

    base, leaf_b, G_b, H_b = build_levelwise_tree(
        X_binned, grad, hess, n_bins, depth, 3.0, 0.1,
        X_hist_binned=Xf, return_training_state=True,
    )
    rowpar = _alloc_test_rowpar(X_binned.shape[1], 1 << depth, 16, 2)
    fast, leaf_f, G_f, H_f = build_levelwise_tree(
        X_binned, grad, hess, n_bins, depth, 3.0, 0.1,
        X_hist_binned=Xf, return_training_state=True, rowpar_buffers=rowpar,
    )
    assert np.array_equal(base.node_features, fast.node_features)
    assert np.array_equal(base.node_thresholds, fast.node_thresholds)
    assert np.array_equal(base.values, fast.values)
    assert np.array_equal(leaf_b, leaf_f)
    assert np.array_equal(G_b, G_f) and np.array_equal(H_b, H_f)


def test_interleaved_hist_buffers_bitwise_match_separate(monkeypatch):
    """Low-thread fits use lane views of one interleaved base array; results
    must be bitwise identical to separate buffers (same summation order)."""
    import darkofit.booster as booster_mod

    X, y = load_breast_cancer(return_X_y=True)

    interleaved = DarkoClassifier(
        iterations=40, thread_count=2, random_state=0
    ).fit(X, y)
    bufs = interleaved.model_._alloc_hist_buffers(
        5, np.full(5, 32, dtype=np.int64)
    )
    # Lane views of one interleaved base: 16-byte last-axis stride (two
    # adjacent float64 lanes), one shared base array. np.shares_memory would
    # be False here because interleaved lanes never overlap byte-for-byte.
    assert bufs[0].base is not None and bufs[0].base is bufs[1].base
    assert bufs[0].strides[-1] == 16

    monkeypatch.setattr(
        booster_mod._BaseBooster, "_HIST_INTERLEAVE_MAX_THREADS", -1
    )
    separate = DarkoClassifier(
        iterations=40, thread_count=2, random_state=0
    ).fit(X, y)
    sep_bufs = separate.model_._alloc_hist_buffers(
        5, np.full(5, 32, dtype=np.int64)
    )
    assert sep_bufs[0].base is None or sep_bufs[0].base is not sep_bufs[1].base
    assert np.array_equal(
        interleaved.predict_proba(X), separate.predict_proba(X)
    )


def test_interleaved_hist_buffers_lightgbm_mode_bitwise(monkeypatch):
    import darkofit.booster as booster_mod

    X, y = load_diabetes(return_X_y=True)
    a = DarkoRegressor(
        iterations=40, tree_mode="lightgbm", thread_count=2, random_state=0
    ).fit(X, y)
    monkeypatch.setattr(
        booster_mod._BaseBooster, "_HIST_INTERLEAVE_MAX_THREADS", -1
    )
    b = DarkoRegressor(
        iterations=40, tree_mode="lightgbm", thread_count=2, random_state=0
    ).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))


def _loop_predict_raw(model, X):
    """Reference per-tree prediction loop (what predict_raw used to do)."""
    X = (np.asarray(X, dtype=object) if model.prep_.cat_features_
         else np.asarray(X, dtype=np.float64))
    Xb = model.prep_.transform(X)
    F = np.full(Xb.shape[0], model.init_, dtype=np.float64)
    for tree in model.trees_:
        tree.add_predict(Xb, F)
    return F


def _loop_predict_raw_multiclass(model, X):
    X = (np.asarray(X, dtype=object) if model.prep_.cat_features_
         else np.asarray(X, dtype=np.float64))
    Xb = model.prep_.transform(X)
    F = np.tile(model.init_[:, None], (1, Xb.shape[0]))
    for round_trees in model.trees_:
        if hasattr(round_trees, "add_predict_class_major"):
            round_trees.add_predict_class_major(Xb, F)
        else:
            for k in range(model.n_classes_):
                round_trees[k].add_predict(Xb, F[k])
    return F.T


def test_flat_prediction_matches_tree_loop_bitwise():
    from sklearn.datasets import make_regression, make_classification

    Xr, yr = make_regression(n_samples=3000, n_features=12, noise=5,
                             random_state=0)
    Xc, yc = make_classification(n_samples=3000, n_features=12,
                                 random_state=0)

    cat = DarkoRegressor(iterations=60, random_state=0).fit(Xr, yr)
    assert np.array_equal(cat.predict(Xr), _loop_predict_raw(cat.model_, Xr))

    lgb = DarkoRegressor(iterations=60, tree_mode="lightgbm",
                                random_state=0).fit(Xr, yr)
    assert np.array_equal(lgb.predict(Xr), _loop_predict_raw(lgb.model_, Xr))

    hybrid = DarkoRegressor(iterations=60, tree_mode="hybrid",
                                   random_state=0).fit(Xr, yr)
    assert np.array_equal(
        hybrid.predict(Xr), _loop_predict_raw(hybrid.model_, Xr)
    )

    binary = DarkoClassifier(iterations=60, tree_mode="lightgbm",
                                    random_state=0).fit(Xc, yc)
    raw = binary.model_.predict_raw(Xc)
    assert np.array_equal(raw, _loop_predict_raw(binary.model_, Xc))

    depthwise = DarkoRegressor(
        iterations=60, tree_mode="depthwise", random_state=0
    ).fit(Xr, yr)
    assert np.array_equal(
        depthwise.predict(Xr), _loop_predict_raw(depthwise.model_, Xr)
    )


def test_flat_multiclass_prediction_matches_loop_bitwise():
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)

    for kw in (
        {},                                            # catboost per-class
        {"tree_mode": "lightgbm"},                     # lightgbm per-class
        {"tree_mode": "hybrid"},                       # hybrid per-class
        {"tree_mode": "lightgbm",
         "multiclass_tree_strategy": "shared_vector"},  # vector leaves
        {"tree_mode": "depthwise"},                    # depthwise per-class
    ):
        m = DarkoClassifier(iterations=20, random_state=0, **kw)
        m.fit(X, y)
        got = m.model_.predict_raw(X)
        want = _loop_predict_raw_multiclass(m.model_, X)
        assert np.array_equal(got, want), kw


def test_flat_prediction_fallback_and_refit_invalidation():
    from sklearn.datasets import make_regression
    from darkofit.booster import GradientBoosting

    X, y = make_regression(n_samples=2000, n_features=8, noise=5,
                           random_state=1)

    # Experimental depthwise trees flatten into a level-wise batch predictor.
    depthwise = GradientBoosting(iterations=10, tree_mode="depthwise",
                                 depth=4, random_state=0)
    depthwise.fit(X, y)
    assert depthwise._flat_ensemble() is not None
    assert np.array_equal(depthwise.predict_raw(X), _loop_predict_raw(depthwise, X))

    # Refitting the same booster object must invalidate the flat cache.
    m = GradientBoosting(iterations=15, random_state=0)
    m.fit(X, y)
    first = m.predict_raw(X)
    assert m._flat_cache_[0] is m.trees_
    X2, y2 = make_regression(n_samples=2000, n_features=8, noise=5,
                             random_state=2)
    m.fit(X2, y2)
    second = m.predict_raw(X2)
    fresh = GradientBoosting(iterations=15, random_state=0).fit(X2, y2)
    assert np.array_equal(second, fresh.predict_raw(X2))
    assert not np.array_equal(first[:10], second[:10])


def test_flat_kernels_direct_parity_all_families():
    """Exercise every flat-ensemble kernel directly against the per-tree
    loop, including families the predict router currently sends to the loop
    (explicit-node trees), since they back model serialization."""
    import numba
    from sklearn.datasets import make_regression, load_wine
    from darkofit.flat_model import (
        FlatLevelwiseEnsemble, FlatNonObliviousEnsemble,
        FlatObliviousEnsemble, flat_predict_preferred
    )

    Xr, yr = make_regression(n_samples=2500, n_features=10, noise=5,
                             random_state=0)
    lgb = DarkoRegressor(iterations=40, tree_mode="lightgbm",
                                random_state=0).fit(Xr, yr)
    flat = lgb.model_._flat_ensemble()
    assert isinstance(flat, FlatNonObliviousEnsemble)
    assert not flat_predict_preferred(flat)  # router keeps the loop
    Xb = lgb.model_.prep_.transform(np.asarray(Xr, dtype=np.float64))
    got = np.zeros(Xb.shape[0])
    flat.add_predict(Xb, got)
    want = np.zeros(Xb.shape[0])
    for tree in lgb.model_.trees_:
        tree.add_predict(Xb, want)
    assert np.array_equal(got, want)

    depthwise = DarkoRegressor(
        iterations=40, tree_mode="depthwise", random_state=0
    ).fit(Xr, yr)
    flat = depthwise.model_._flat_ensemble()
    assert isinstance(flat, FlatLevelwiseEnsemble)
    assert flat_predict_preferred(flat) == (numba.get_num_threads() > 1)
    Xb = depthwise.model_.prep_.transform(np.asarray(Xr, dtype=np.float64))
    got = np.zeros(Xb.shape[0])
    flat.add_predict(Xb, got)
    want = np.zeros(Xb.shape[0])
    for tree in depthwise.model_.trees_:
        tree.add_predict(Xb, want)
    assert np.array_equal(got, want)

    Xw, yw = load_wine(return_X_y=True)
    for kw, expect_pref in (
        ({}, numba.get_num_threads() > 1),               # oblivious class
        ({"tree_mode": "lightgbm"}, False),              # nonobl class
        ({"tree_mode": "lightgbm",
          "multiclass_tree_strategy": "shared_vector"}, False),
        ({"tree_mode": "depthwise"}, numba.get_num_threads() > 1),
    ):
        m = DarkoClassifier(iterations=15, random_state=0, **kw)
        m.fit(Xw, yw)
        flat = m.model_._flat_ensemble()
        assert flat is not None
        assert flat_predict_preferred(flat) == expect_pref, kw
        Xb = m.model_.prep_.transform(np.asarray(Xw, dtype=np.float64))
        K = m.model_.n_classes_
        got = np.zeros((K, Xb.shape[0]))
        flat.add_predict_class_major(Xb, got)
        want = np.zeros((K, Xb.shape[0]))
        for rt in m.model_.trees_:
            if hasattr(rt, "add_predict_class_major"):
                rt.add_predict_class_major(Xb, want)
            else:
                for k in range(K):
                    rt[k].add_predict(Xb, want[k])
        assert np.array_equal(got, want), kw


def _cat_dataset(n=1500, seed=0, with_nan=True):
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "east"], n).astype(object)
    if with_nan:
        region[::13] = None
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = region
    X[:, 2] = rng.normal(size=n) * 4
    y = (np.where(region == "north", 1.5, -0.5).astype(float)
         + 0.7 * X[:, 0].astype(float) + rng.normal(0, 0.3, n))
    return X, y


def _read_archive_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _write_archive(path, arrays):
    np.savez_compressed(path, **arrays)
    return path


def test_save_load_regressor_round_trip(tmp_path):
    X, y = _cat_dataset()
    path = str(tmp_path / "reg.npz")

    for kw in ({}, {"tree_mode": "lightgbm"}, {"tree_mode": "hybrid"},
               {"tree_mode": "depthwise"},
               {"loss": "Quantile", "alpha": 0.8}):
        m = DarkoRegressor(iterations=40, random_state=0, **kw)
        m.fit(X, y, cat_features=[1])
        m.save_model(path)
        loaded = DarkoRegressor.load_model(path)

        # Unseen and missing categories at predict time round-trip too.
        X_new = X[:50].copy()
        X_new[0, 1] = "west"
        X_new[1, 1] = None
        assert np.array_equal(m.predict(X_new), loaded.predict(X_new)), kw
        assert np.array_equal(m.feature_importances_,
                              loaded.feature_importances_)
        assert loaded.best_iteration_ == m.best_iteration_
        assert loaded.get_params()["tree_mode"] == m.get_params()["tree_mode"]
        assert loaded.n_features_in_ == X.shape[1]

    no_ext = tmp_path / "reg_no_ext"
    m = DarkoRegressor(iterations=5, random_state=0).fit(
        X[:80], y[:80], cat_features=[1]
    )
    m.save_model(no_ext)
    loaded = DarkoRegressor.load_model(no_ext)
    assert np.array_equal(m.predict(X[:10]), loaded.predict(X[:10]))
    assert loaded.model_.timing_ == m.model_.timing_
    assert loaded.model_.train_history_ == m.model_.train_history_


def test_load_legacy_v1_missing_category_archive(tmp_path):
    import json

    from darkofit.serialization import (
        FORMAT_VERSION,
        _KIND_MISSING,
        _KIND_STR,
    )
    from darkofit.target_encoding import _MISSING_CATEGORY

    X, y = _cat_dataset(n=180, seed=8)
    model = DarkoRegressor(iterations=5, random_state=0).fit(
        X, y, cat_features=[1]
    )
    path = tmp_path / "current-missing.npz"
    model.save_model(path)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    header = json.loads(str(arrays["header"]))
    assert FORMAT_VERSION == 5
    assert header["format_version"] == 2

    kinds = arrays["cat0__kinds"].copy()
    missing_pos = np.flatnonzero(kinds == _KIND_MISSING)
    assert missing_pos.size
    max_len = max(7, *(len(str(v)) for v in arrays["cat0__values"]))
    values = arrays["cat0__values"].astype(f"<U{max_len}")
    values[missing_pos] = "__nan__"
    kinds[missing_pos] = _KIND_STR
    header["format_version"] = 1
    arrays["header"] = np.array(json.dumps(header))
    arrays["cat0__values"] = values
    arrays["cat0__kinds"] = kinds

    legacy_path = tmp_path / "legacy-v1-missing.npz"
    np.savez_compressed(legacy_path, **arrays)
    loaded = DarkoRegressor.load_model(legacy_path)

    cat_map = loaded.model_.prep_.cat_maps_[0]
    assert _MISSING_CATEGORY in cat_map
    assert cat_map["__nan__"] == cat_map[_MISSING_CATEGORY]
    X_new = X[:30].copy()
    X_new[0, 1] = None
    X_new[1, 1] = np.nan
    assert np.array_equal(model.predict(X_new), loaded.predict(X_new))

    missing_row = X[:1].copy()
    legacy_string_row = X[:1].copy()
    missing_row[0, 1] = None
    legacy_string_row[0, 1] = "__nan__"
    assert np.array_equal(
        loaded.predict(missing_row),
        loaded.predict(legacy_string_row),
    )

    upgraded_path = tmp_path / "upgraded-v2-missing.npz"
    loaded.save_model(upgraded_path)
    reloaded = DarkoRegressor.load_model(upgraded_path)
    reloaded_map = reloaded.model_.prep_.cat_maps_[0]
    assert reloaded_map["__nan__"] == reloaded_map[_MISSING_CATEGORY]
    assert np.array_equal(
        reloaded.predict(missing_row),
        reloaded.predict(legacy_string_row),
    )


def test_load_rejects_corrupt_category_and_class_payloads(tmp_path):
    from darkofit.booster import MulticlassBoosting
    from darkofit.serialization import _KIND_BOOL
    from sklearn.datasets import load_wine

    X, y = _cat_dataset(n=160, seed=11)
    cat_path = tmp_path / "cat.npz"
    DarkoRegressor(iterations=3, random_state=0).fit(
        X, y, cat_features=[1]
    ).save_model(cat_path)

    arrays = _read_archive_arrays(cat_path)
    arrays["cat0__kinds"] = arrays["cat0__kinds"][:-1]
    with pytest.raises(ValueError, match="values and kinds length mismatch"):
        DarkoRegressor.load_model(
            _write_archive(tmp_path / "cat-kind-short.npz", arrays)
        )

    arrays = _read_archive_arrays(cat_path)
    arrays["cat0__values"] = arrays["cat0__values"].reshape(1, -1)
    with pytest.raises(ValueError, match="values must be 1-dimensional"):
        DarkoRegressor.load_model(
            _write_archive(tmp_path / "cat-values-2d.npz", arrays)
        )

    X_bool = X.copy()
    X_bool[:, 1] = np.arange(X_bool.shape[0]) % 2 == 0
    bool_path = tmp_path / "bool-cat.npz"
    DarkoRegressor(iterations=3, random_state=0).fit(
        X_bool, y, cat_features=[1]
    ).save_model(bool_path)
    arrays = _read_archive_arrays(bool_path)
    bool_pos = np.flatnonzero(arrays["cat0__kinds"] == _KIND_BOOL)
    assert bool_pos.size
    values = arrays["cat0__values"].astype("<U16")
    values[bool_pos[0]] = "not-bool"
    arrays["cat0__values"] = values
    with pytest.raises(ValueError, match="bool payload"):
        DarkoRegressor.load_model(
            _write_archive(tmp_path / "cat-bad-bool.npz", arrays)
        )

    Xw, yw_num = load_wine(return_X_y=True)
    yw = np.array(["low", "mid", "high"], dtype=object)[yw_num]
    core_path = tmp_path / "core-multiclass.npz"
    MulticlassBoosting(iterations=3, random_state=0).fit(Xw, yw).save_model(
        core_path
    )
    arrays = _read_archive_arrays(core_path)
    arrays["classes_kinds"] = arrays["classes_kinds"][:-1]
    with pytest.raises(ValueError, match="values and kinds length mismatch"):
        MulticlassBoosting.load_model(
            _write_archive(tmp_path / "core-class-kind-short.npz", arrays)
        )

    arrays = _read_archive_arrays(core_path)
    values = arrays["classes"].astype("<U16")
    values[0] = "not-bool"
    arrays["classes"] = values
    arrays["classes_kinds"] = arrays["classes_kinds"].copy()
    arrays["classes_kinds"][0] = _KIND_BOOL
    with pytest.raises(ValueError, match="bool payload"):
        MulticlassBoosting.load_model(
            _write_archive(tmp_path / "core-class-bad-bool.npz", arrays)
        )

    wrapper_path = tmp_path / "wrapper-binary.npz"
    yb = np.where(y[:120] > np.median(y[:120]), "yes", "no").astype(object)
    DarkoClassifier(iterations=3, random_state=0).fit(
        np.asarray(X[:120, [0, 2]], dtype=np.float64), yb
    ).save_model(wrapper_path)

    arrays = _read_archive_arrays(wrapper_path)
    arrays["wrapper__classes_kinds"] = arrays["wrapper__classes_kinds"][:-1]
    with pytest.raises(ValueError, match="values and kinds length mismatch"):
        DarkoClassifier.load_model(
            _write_archive(tmp_path / "wrapper-class-kind-short.npz", arrays)
        )

    arrays = _read_archive_arrays(wrapper_path)
    values = arrays["wrapper__classes"].astype("<U16")
    values[0] = "not-bool"
    arrays["wrapper__classes"] = values
    arrays["wrapper__classes_kinds"] = arrays["wrapper__classes_kinds"].copy()
    arrays["wrapper__classes_kinds"][0] = _KIND_BOOL
    with pytest.raises(ValueError, match="bool payload"):
        DarkoClassifier.load_model(
            _write_archive(tmp_path / "wrapper-class-bad-bool.npz", arrays)
        )


def test_save_load_classifier_round_trip(tmp_path):
    from sklearn.datasets import load_breast_cancer, load_wine

    path = str(tmp_path / "clf.npz")

    # Binary with numeric labels.
    Xb, yb = load_breast_cancer(return_X_y=True)
    for kw in ({}, {"tree_mode": "depthwise"}):
        binary = DarkoClassifier(
            iterations=30, random_state=0, **kw
        ).fit(Xb, yb)
        binary.save_model(path)
        loaded = DarkoClassifier.load_model(path)
        assert np.array_equal(
            binary.predict_proba(Xb), loaded.predict_proba(Xb)
        ), kw
        assert np.array_equal(binary.classes_, loaded.classes_)
        assert loaded.n_features_in_ == Xb.shape[1]

    # Multiclass with string labels, both tree strategies.
    Xw, yw_num = load_wine(return_X_y=True)
    yw = np.array(["lo", "mid", "hi"])[yw_num]
    for kw in ({}, {"tree_mode": "lightgbm"}, {"tree_mode": "hybrid"},
               {"tree_mode": "depthwise"},
               {"tree_mode": "lightgbm",
                "multiclass_tree_strategy": "shared_vector"}):
        mc = DarkoClassifier(iterations=15, random_state=0, **kw)
        mc.fit(Xw, yw)
        mc.save_model(path)
        loaded = DarkoClassifier.load_model(path)
        assert np.array_equal(mc.predict_proba(Xw), loaded.predict_proba(Xw)), kw
        assert list(loaded.classes_) == list(mc.classes_)
        assert np.array_equal(mc.predict(Xw), loaded.predict(Xw))
        assert loaded.n_features_in_ == Xw.shape[1]


def test_save_load_booster_level_and_errors(tmp_path):
    from darkofit.booster import GradientBoosting, MulticlassBoosting
    from darkofit.serialization import save_booster
    from sklearn.datasets import load_wine

    path = str(tmp_path / "booster.npz")
    X, y = load_diabetes(return_X_y=True)
    m = GradientBoosting(iterations=20, random_state=0).fit(X, y)
    m.save_model(path)
    loaded = GradientBoosting.load_model(path)
    assert np.array_equal(m.predict_raw(X), loaded.predict_raw(X))
    assert loaded.timing_ == m.timing_
    assert loaded.train_history_ == m.train_history_
    assert loaded.valid_history_ == m.valid_history_
    assert loaded.n_threads_ >= 1

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["trees__value_offsets"] = arrays["trees__value_offsets"][:1]
    bad_offsets = tmp_path / "bad-offsets.npz"
    np.savez_compressed(bad_offsets, **arrays)
    with pytest.raises(ValueError, match="offsets length"):
        GradientBoosting.load_model(bad_offsets)

    with pytest.raises(ValueError, match="object-dtype"):
        save_booster(
            m,
            tmp_path / "bad-object.npz",
            wrapper_arrays={"bad": np.array([object()], dtype=object)},
        )

    auto = GradientBoosting(
        iterations=3,
        depth="auto",
        l2_leaf_reg="auto",
        min_child_samples="auto",
        min_child_weight="auto",
        cat_smoothing="auto",
        random_state=0,
    ).fit(X[:120], y[:120])
    auto.save_model(path)
    auto_loaded = GradientBoosting.load_model(path)
    assert auto_loaded.depth == "auto"
    assert auto_loaded.l2_leaf_reg == "auto"
    assert auto_loaded.min_child_samples == "auto"
    assert auto_loaded.min_child_weight == "auto"
    assert auto_loaded.cat_smoothing == "auto"
    assert np.array_equal(auto.predict_raw(X[:20]), auto_loaded.predict_raw(X[:20]))

    with pytest.raises(TypeError):
        MulticlassBoosting.load_model(path)

    with pytest.raises(ValueError):
        GradientBoosting(iterations=2).save_model(path)  # unfitted

    empty = GradientBoosting(
        iterations=5, tree_mode="depthwise", random_state=0
    ).fit(np.zeros((40, 2), dtype=np.float64), np.zeros(40, dtype=np.float64))
    assert empty.best_iteration_ == 0
    assert empty.trees_ == []
    empty.save_model(path)
    empty_loaded = GradientBoosting.load_model(path)
    assert empty_loaded.trees_ == []
    assert empty_loaded.best_iteration_ == 0
    assert np.array_equal(
        empty.predict_raw(np.zeros((3, 2), dtype=np.float64)),
        empty_loaded.predict_raw(np.zeros((3, 2), dtype=np.float64)),
    )

    depthwise = GradientBoosting(
        iterations=3, tree_mode="depthwise", depth=3, random_state=0
    ).fit(X, y)
    depthwise_loop = _loop_predict_raw(depthwise, X)
    depthwise.save_model(path)
    depthwise_loaded = GradientBoosting.load_model(path)
    assert np.array_equal(depthwise.predict_raw(X), depthwise_loaded.predict_raw(X))
    assert np.array_equal(depthwise_loop, depthwise_loaded.predict_raw(X))
    for before, after in zip(depthwise.trees_, depthwise_loaded.trees_):
        assert np.array_equal(before.node_features, after.node_features)
        assert np.array_equal(before.node_thresholds, after.node_thresholds)
        assert np.array_equal(before.values, after.values)
        assert np.array_equal(before.splits_feat, after.splits_feat)
        assert np.array_equal(before.splits_thr, after.splits_thr)
        assert np.array_equal(before.gains, after.gains)
    flat = depthwise_loaded._flat_ensemble()
    assert flat is not None
    Xb_depthwise = depthwise_loaded.prep_.transform(np.asarray(X, dtype=np.float64))
    flat_pred = np.full(Xb_depthwise.shape[0], depthwise_loaded.init_)
    flat.add_predict(Xb_depthwise, flat_pred)
    assert np.array_equal(flat_pred, depthwise_loop)

    arrays = _read_archive_arrays(path)
    arrays["trees__node_thresholds_flat"] = (
        arrays["trees__node_thresholds_flat"][:-1]
    )
    bad_level_nodes = tmp_path / "bad-level-node-sibling.npz"
    np.savez_compressed(bad_level_nodes, **arrays)
    with pytest.raises(ValueError, match="offsets do not match array length"):
        GradientBoosting.load_model(bad_level_nodes)

    arrays = _read_archive_arrays(path)
    arrays["trees__node_widths"] = arrays["trees__node_widths"].copy()
    arrays["trees__node_widths"][0] += 1
    bad_level_shape = tmp_path / "bad-level-node-shape.npz"
    np.savez_compressed(bad_level_shape, **arrays)
    with pytest.raises(ValueError, match="node payload"):
        GradientBoosting.load_model(bad_level_shape)

    arrays = _read_archive_arrays(path)
    arrays["trees__values_flat"] = arrays["trees__values_flat"][:-1]
    bad_level_values = tmp_path / "bad-level-values.npz"
    np.savez_compressed(bad_level_values, **arrays)
    with pytest.raises(ValueError, match="offsets do not match array length"):
        GradientBoosting.load_model(bad_level_values)

    arrays = _read_archive_arrays(path)
    arrays["trees__splits_thr_flat"] = arrays["trees__splits_thr_flat"][:-1]
    bad_level_splits = tmp_path / "bad-level-splits.npz"
    np.savez_compressed(bad_level_splits, **arrays)
    with pytest.raises(ValueError, match="offsets do not match array length"):
        GradientBoosting.load_model(bad_level_splits)

    Xw, yw = load_wine(return_X_y=True)
    mc = MulticlassBoosting(iterations=10, random_state=0).fit(Xw, yw)
    mc.save_model(path)
    mc_loaded = MulticlassBoosting.load_model(path)
    assert np.array_equal(mc.predict_raw(Xw), mc_loaded.predict_raw(Xw))
    assert np.array_equal(mc.classes_, mc_loaded.classes_)

    mc_depthwise = MulticlassBoosting(
        iterations=5, tree_mode="depthwise", depth=3, random_state=0
    ).fit(Xw, yw)
    mc_depthwise.save_model(path)
    mc_depthwise_loaded = MulticlassBoosting.load_model(path)
    assert np.array_equal(
        mc_depthwise.predict_raw(Xw),
        mc_depthwise_loaded.predict_raw(Xw),
    )
    assert np.array_equal(mc_depthwise.classes_, mc_depthwise_loaded.classes_)


def test_core_booster_save_accepts_random_state_objects(tmp_path):
    from darkofit.booster import GradientBoosting

    X, y = load_diabetes(return_X_y=True)
    model = GradientBoosting(
        iterations=3, random_state=np.random.default_rng(0)
    ).fit(X[:80], y[:80])
    path = tmp_path / "rng-core.npz"

    model.save_model(path)
    loaded = GradientBoosting.load_model(path)

    assert isinstance(loaded.random_state, int)
    assert np.array_equal(model.predict_raw(X[:10]), loaded.predict_raw(X[:10]))


def test_load_preserves_saved_resolved_threads_without_setting_numba(tmp_path):
    import numba
    from darkofit.booster import GradientBoosting

    X, y = load_diabetes(return_X_y=True)
    model = GradientBoosting(
        iterations=2, tree_mode="lightgbm", num_leaves=3,
        thread_count=8, random_state=0
    ).fit(X[:120], y[:120])
    path = tmp_path / "threads.npz"
    model.save_model(path)

    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        loaded = GradientBoosting.load_model(path)
        assert loaded.n_threads_ == model.n_threads_
        assert numba.get_num_threads() == 1
    finally:
        numba.set_num_threads(original_threads)


def test_load_rejects_mismatched_tree_and_encoder_flat_arrays(tmp_path):
    import json

    from darkofit.booster import GradientBoosting
    from sklearn.datasets import load_wine

    X, y = _cat_dataset(n=180, seed=5)
    path = tmp_path / "cat-lightgbm.npz"
    model = GradientBoosting(
        iterations=3, tree_mode="lightgbm", num_leaves=3, random_state=0
    ).fit(X, y, cat_features=[1])
    model.save_model(path)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["trees__thresholds_flat"] = arrays["trees__thresholds_flat"][:-1]
    bad_tree = tmp_path / "bad-tree-sibling.npz"
    np.savez_compressed(bad_tree, **arrays)
    with pytest.raises(ValueError, match="offsets do not match array length"):
        GradientBoosting.load_model(bad_tree)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["trees__node_offsets"] = arrays["trees__node_offsets"].astype(np.float64)
    bad_offset_dtype = tmp_path / "bad-tree-offset-dtype.npz"
    np.savez_compressed(bad_offset_dtype, **arrays)
    with pytest.raises(ValueError, match="offsets must contain integer values"):
        GradientBoosting.load_model(bad_offset_dtype)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["trees__features_flat"] = arrays["trees__features_flat"].astype(np.float64)
    bad_index_dtype = tmp_path / "bad-tree-index-dtype.npz"
    np.savez_compressed(bad_index_dtype, **arrays)
    with pytest.raises(ValueError, match="features must contain integer values"):
        GradientBoosting.load_model(bad_index_dtype)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["enc0__counts_flat"] = arrays["enc0__counts_flat"][:-1]
    bad_encoder = tmp_path / "bad-encoder-sibling.npz"
    np.savez_compressed(bad_encoder, **arrays)
    with pytest.raises(ValueError, match="offsets do not match array length"):
        GradientBoosting.load_model(bad_encoder)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["bin__border_offsets"] = arrays["bin__border_offsets"][:-1]
    bad_binner_offsets = tmp_path / "bad-binner-offsets.npz"
    np.savez_compressed(bad_binner_offsets, **arrays)
    with pytest.raises(ValueError, match="offsets length"):
        GradientBoosting.load_model(bad_binner_offsets)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["bin__n_bins"] = arrays["bin__n_bins"].copy()
    arrays["bin__n_bins"][0] += 1
    bad_n_bins = tmp_path / "bad-binner-n-bins.npz"
    np.savez_compressed(bad_n_bins, **arrays)
    with pytest.raises(ValueError, match="n_bins do not match"):
        GradientBoosting.load_model(bad_n_bins)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    header = json.loads(str(arrays["header"]))
    arrays["prep__feature_map"] = arrays["prep__feature_map"].copy()
    arrays["prep__feature_map"][0] = header["n_input_features"]
    bad_feature_map = tmp_path / "bad-feature-map.npz"
    np.savez_compressed(bad_feature_map, **arrays)
    with pytest.raises(ValueError, match="feature_map contains out-of-range"):
        GradientBoosting.load_model(bad_feature_map)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["bin__block_widths"] = arrays["bin__block_widths"].copy()
    arrays["bin__block_widths"][0] += 1
    bad_block_widths = tmp_path / "bad-binner-block-widths.npz"
    np.savez_compressed(bad_block_widths, **arrays)
    with pytest.raises(ValueError, match="block_widths"):
        GradientBoosting.load_model(bad_block_widths)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["cat0__kinds"] = arrays["cat0__kinds"].copy()
    arrays["cat0__kinds"][0] = 99
    bad_kind = tmp_path / "bad-category-kind.npz"
    np.savez_compressed(bad_kind, **arrays)
    with pytest.raises(ValueError, match="unknown category kind"):
        GradientBoosting.load_model(bad_kind)

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    header = json.loads(str(arrays["header"]))
    header["tree_kind"] = "mystery"
    arrays["header"] = np.array(json.dumps(header))
    bad_tree_kind = tmp_path / "bad-tree-kind.npz"
    np.savez_compressed(bad_tree_kind, **arrays)
    with pytest.raises(ValueError, match="unknown tree kind"):
        GradientBoosting.load_model(bad_tree_kind)

    scalar_path = tmp_path / "scalar-oblivious.npz"
    scalar = GradientBoosting(iterations=4, random_state=0).fit(
        np.asarray(X[:, [0, 2]], dtype=np.float64), y
    )
    scalar.save_model(scalar_path)
    with np.load(scalar_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    keep = len(arrays["trees__depths"]) - 1
    assert keep > 0
    split_end = arrays["trees__split_offsets"][keep]
    value_end = arrays["trees__value_offsets"][keep]
    arrays["trees__depths"] = arrays["trees__depths"][:keep]
    arrays["trees__feats_flat"] = arrays["trees__feats_flat"][:split_end]
    arrays["trees__thrs_flat"] = arrays["trees__thrs_flat"][:split_end]
    arrays["trees__gains_flat"] = arrays["trees__gains_flat"][:split_end]
    arrays["trees__values_flat"] = arrays["trees__values_flat"][:value_end]
    arrays["trees__split_offsets"] = arrays["trees__split_offsets"][:keep + 1]
    arrays["trees__value_offsets"] = arrays["trees__value_offsets"][:keep + 1]
    bad_scalar_count = tmp_path / "bad-scalar-tree-count.npz"
    np.savez_compressed(bad_scalar_count, **arrays)
    with pytest.raises(ValueError, match="tree count"):
        GradientBoosting.load_model(bad_scalar_count)

    with np.load(scalar_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    header = json.loads(str(arrays["header"]))
    assert header["best_iteration"] > 0
    header["tree_kind"] = "empty"
    arrays["header"] = np.array(json.dumps(header))
    bad_empty = tmp_path / "bad-empty-tree-kind.npz"
    np.savez_compressed(bad_empty, **arrays)
    with pytest.raises(ValueError, match="empty tree kind"):
        GradientBoosting.load_model(bad_empty)

    Xw, yw = load_wine(return_X_y=True)
    per_class = DarkoClassifier(iterations=3, random_state=0).fit(Xw, yw)
    per_class_path = tmp_path / "per-class.npz"
    per_class.save_model(per_class_path)
    with np.load(per_class_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    header = json.loads(str(arrays["header"]))
    header["n_rounds"] += 1
    arrays["header"] = np.array(json.dumps(header))
    bad_round_count = tmp_path / "bad-per-class-round-count.npz"
    np.savez_compressed(bad_round_count, **arrays)
    with pytest.raises(ValueError, match="tree count"):
        DarkoClassifier.load_model(bad_round_count)

    shared = DarkoClassifier(
        iterations=3,
        tree_mode="lightgbm",
        multiclass_tree_strategy="shared_vector",
        random_state=0,
    ).fit(Xw, yw)
    shared_path = tmp_path / "shared-vector.npz"
    shared.save_model(shared_path)

    with np.load(shared_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["trees__values_flat"] = arrays["trees__values_flat"][:, 0]
    bad_value_shape = tmp_path / "bad-multiclass-value-shape.npz"
    np.savez_compressed(bad_value_shape, **arrays)
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        DarkoClassifier.load_model(bad_value_shape)

    with np.load(shared_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["trees__values_flat"] = arrays["trees__values_flat"][:, :2]
    bad_value_width = tmp_path / "bad-multiclass-value-width.npz"
    np.savez_compressed(bad_value_width, **arrays)
    with pytest.raises(ValueError, match="value width"):
        DarkoClassifier.load_model(bad_value_width)

    with np.load(shared_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    header = json.loads(str(arrays["header"]))
    header["best_iteration"] += 1
    arrays["header"] = np.array(json.dumps(header))
    bad_shared_count = tmp_path / "bad-shared-tree-count.npz"
    np.savez_compressed(bad_shared_count, **arrays)
    with pytest.raises(ValueError, match="tree count"):
        DarkoClassifier.load_model(bad_shared_count)


def test_save_load_weighted_fit_round_trip(tmp_path):
    X, y = _cat_dataset(seed=4)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, len(y))
    m = DarkoRegressor(iterations=30, random_state=0)
    m.fit(X, y, cat_features=[1], sample_weight=w)
    path = str(tmp_path / "weighted.npz")
    m.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    assert np.array_equal(m.predict(X), loaded.predict(X))


def test_pickle_round_trip():
    import pickle

    X, y = _cat_dataset(seed=7)
    m = DarkoRegressor(iterations=25, random_state=0)
    m.fit(X, y, cat_features=[1])
    clone = pickle.loads(pickle.dumps(m))
    assert np.array_equal(m.predict(X), clone.predict(X))

    from sklearn.datasets import load_wine
    Xw, yw = load_wine(return_X_y=True)
    mc = DarkoClassifier(iterations=10, random_state=0).fit(Xw, yw)
    clone = pickle.loads(pickle.dumps(mc))
    assert np.array_equal(mc.predict_proba(Xw), clone.predict_proba(Xw))


def test_load_model_cross_class_errors(tmp_path):
    from sklearn.datasets import load_wine

    Xw, yw = load_wine(return_X_y=True)
    X, y = load_diabetes(return_X_y=True)

    clf_path = str(tmp_path / "clf.npz")
    DarkoClassifier(iterations=5, random_state=0).fit(
        Xw, yw
    ).save_model(clf_path)
    with pytest.raises(TypeError):
        DarkoRegressor.load_model(clf_path)

    reg_path = str(tmp_path / "reg.npz")
    DarkoRegressor(iterations=5, random_state=0).fit(
        X, y
    ).save_model(reg_path)
    with pytest.raises(TypeError):
        DarkoClassifier.load_model(reg_path)

    # A booster-level multiclass save still loads through the classifier
    # wrapper (class labels live on the booster there).
    from darkofit.booster import MulticlassBoosting
    booster_path = str(tmp_path / "mc.npz")
    MulticlassBoosting(iterations=5, random_state=0).fit(
        Xw, yw
    ).save_model(booster_path)
    loaded = DarkoClassifier.load_model(booster_path)
    assert loaded.n_classes_ == 3
    assert loaded.n_features_in_ == Xw.shape[1]
    assert loaded.predict_proba(Xw).shape == (len(yw), 3)


@pytest.mark.parametrize("threads", [1, 0])  # 0 -> all available
@pytest.mark.parametrize("constant_hessian", [False, True])
def test_oblivious_level_subtraction_matches_full_rebuild(threads,
                                                          constant_hessian):
    import numba
    from darkofit.tree import build_oblivious_tree

    if threads == 0:
        threads = numba.config.NUMBA_NUM_THREADS
    if threads > 1 and numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("requires multithreaded numba")
    rng = np.random.default_rng(17)
    n, n_feat, max_bins = 40_000, 8, 16
    # Skewed bins so deeper levels produce empty and near-empty children,
    # exercising the exact empty-side handling.
    X_binned = np.minimum(
        rng.geometric(0.35, size=(n, n_feat)) - 1, max_bins - 2
    ).astype(np.uint8)
    n_bins = np.full(n_feat, max_bins, dtype=np.int64)
    grad = rng.integers(-8, 9, size=n).astype(np.float64)
    hess = (np.ones(n) if constant_hessian
            else np.choose(rng.integers(0, 3, size=n), [0.5, 1.0, 2.0]))
    Xf = np.asfortranarray(X_binned)

    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        kw = dict(X_hist_binned=Xf, constant_hessian=constant_hessian,
                  return_training_state=True)
        base, leaf_b, G_b, H_b = build_oblivious_tree(
            X_binned, grad, hess, n_bins, 5, 3.0, 0.1,
            level_histogram_subtraction=False, **kw
        )
        fast, leaf_f, G_f, H_f = build_oblivious_tree(
            X_binned, grad, hess, n_bins, 5, 3.0, 0.1,
            level_histogram_subtraction=True, **kw
        )
    finally:
        numba.set_num_threads(old_threads)

    assert np.array_equal(base.splits_feat, fast.splits_feat)
    assert np.array_equal(base.splits_thr, fast.splits_thr)
    assert np.array_equal(base.values, fast.values)
    assert np.array_equal(base.gains, fast.gains)
    assert np.array_equal(leaf_b, leaf_f)
    assert np.array_equal(G_b, G_f) and np.array_equal(H_b, H_f)
    assert base.depth >= 4  # deep enough to exercise several levels


def test_levelwise_level_subtraction_matches_full_rebuild():
    import numba
    from darkofit.tree import build_levelwise_tree

    rng = np.random.default_rng(23)
    n, n_feat, max_bins = 30_000, 6, 16
    X_binned = np.minimum(
        rng.geometric(0.4, size=(n, n_feat)) - 1, max_bins - 2
    ).astype(np.uint8)
    n_bins = np.full(n_feat, max_bins, dtype=np.int64)
    grad = rng.integers(-8, 9, size=n).astype(np.float64)
    hess = np.choose(rng.integers(0, 3, size=n), [0.5, 1.0, 2.0])
    Xf = np.asfortranarray(X_binned)
    kw = dict(X_hist_binned=Xf, return_training_state=True,
              min_child_weight=8.0)

    base, leaf_b, G_b, H_b = build_levelwise_tree(
        X_binned, grad, hess, n_bins, 5, 3.0, 0.1,
        level_histogram_subtraction=False, **kw
    )
    fast, leaf_f, G_f, H_f = build_levelwise_tree(
        X_binned, grad, hess, n_bins, 5, 3.0, 0.1,
        level_histogram_subtraction=True, **kw
    )
    assert np.array_equal(base.node_features, fast.node_features)
    assert np.array_equal(base.node_thresholds, fast.node_thresholds)
    assert np.array_equal(base.values, fast.values)
    assert np.array_equal(leaf_b, leaf_f)
    assert np.array_equal(G_b, G_f) and np.array_equal(H_b, H_f)


def test_level_subtraction_float_quality_parity(monkeypatch):
    """With real-valued gradients the subtraction differs only by float64
    rounding; fitted-model quality must be unchanged. Both lanes are forced
    explicitly because 'auto' is thread-count dependent."""
    from sklearn.datasets import make_regression
    import darkofit.tree as tree_mod
    import darkofit.booster as booster_mod

    X, y = make_regression(n_samples=20_000, n_features=15, noise=15,
                           random_state=5)
    original = tree_mod.build_oblivious_tree

    # This test deliberately forces a builder path that the fitted dispatch
    # selector never chooses at this thread count. Keep it outside dispatch
    # engagement claims while comparing the two internal arithmetic paths.
    monkeypatch.setattr(
        booster_mod._BaseBooster,
        "_oblivious_functional_ineligibility",
        lambda *_args, **_kwargs: "row_parallel_histograms_active",
    )

    def forced(setting):
        def build(*args, **kwargs):
            kwargs["level_histogram_subtraction"] = setting
            return original(*args, **kwargs)
        return build

    monkeypatch.setattr(booster_mod, "build_oblivious_tree", forced(True))
    m = DarkoRegressor(iterations=80, random_state=0).fit(X, y)
    rmse_subtract = np.sqrt(np.mean((y - m.predict(X)) ** 2))

    monkeypatch.setattr(booster_mod, "build_oblivious_tree", forced(False))
    m2 = DarkoRegressor(iterations=80, random_state=0).fit(X, y)
    rmse_full = np.sqrt(np.mean((y - m2.predict(X)) ** 2))
    assert abs(rmse_subtract - rmse_full) < 0.02 * rmse_full


def test_level_subtraction_auto_resolution():
    from darkofit.tree import (
        _LEVEL_SUBTRACTION_MAX_THREADS, _resolve_level_subtraction
    )
    import numba

    old = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        assert _resolve_level_subtraction("auto") is True
        assert _resolve_level_subtraction(False) is False
        if numba.config.NUMBA_NUM_THREADS > _LEVEL_SUBTRACTION_MAX_THREADS:
            numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
            assert _resolve_level_subtraction("auto") is False
            assert _resolve_level_subtraction(True) is True
    finally:
        numba.set_num_threads(old)


@pytest.mark.parametrize("tree_mode", ["catboost", "lightgbm"])
def test_multiclass_fused_root_histograms_bitwise(tree_mode, monkeypatch):
    """The fused class-major root pass accumulates rows in the same order as
    each per-class root scan, so fits must be bitwise identical with the
    fused pass stripped."""
    import darkofit.booster as booster_mod
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    kw = dict(iterations=12, random_state=0, tree_mode=tree_mode)
    fused = DarkoClassifier(**kw).fit(X, y)

    name = ("build_leafwise_tree" if tree_mode == "lightgbm"
            else "build_oblivious_tree")
    original = getattr(booster_mod, name)

    def strip_root(*args, **kwargs):
        kwargs.pop("root_histograms", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(booster_mod, name, strip_root)
    plain = DarkoClassifier(**kw).fit(X, y)
    assert np.array_equal(fused.predict_proba(X), plain.predict_proba(X))


def test_leafwise_root_histograms_kwarg_matches_self_scan():
    from darkofit.preprocessing import FeaturePreprocessor
    from darkofit.tree import (
        _build_multiclass_histograms_counts_into, build_leafwise_tree
    )

    rng = np.random.default_rng(6)
    n, F = 20_000, 8
    X = rng.normal(size=(n, F))
    y = X[:, 0] + rng.normal(0, 0.5, n)
    prep = FeaturePreprocessor(32, 1.0, 0)
    Xb = prep.fit_transform(X, [y], None)
    nb = prep.n_bins_
    B = int(nb.max())
    K = 3
    grad = rng.normal(size=(K, n))
    hess = np.maximum(rng.uniform(0.01, 0.25, size=(K, n)), 1e-6)
    Xf = np.asfortranarray(Xb)

    root_g = np.zeros((K, F, 1, B))
    root_h = np.zeros((K, F, 1, B))
    root_c = np.zeros((F, 1, B))
    _build_multiclass_histograms_counts_into(
        Xf, grad, hess, np.zeros(n, dtype=np.int64), 1, root_g, root_h, root_c
    )

    for k in range(K):
        base = build_leafwise_tree(
            Xb, grad[k], hess[k], nb, -1, 1.0, 0.1, max_leaves=15,
            X_hist_binned=Xf,
        )
        fast = build_leafwise_tree(
            Xb, grad[k], hess[k], nb, -1, 1.0, 0.1, max_leaves=15,
            X_hist_binned=Xf,
            root_histograms=(root_g[k, :, 0, :], root_h[k, :, 0, :],
                             root_c[:, 0, :]),
        )
        assert np.array_equal(base.splits_feat, fast.splits_feat)
        assert np.array_equal(base.splits_thr, fast.splits_thr)
        assert np.array_equal(base.values, fast.values)


def test_multiclass_goss_runs_and_validates():
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split

    X, y = load_wine(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=0, stratify=y)
    for tree_mode in ("catboost", "lightgbm"):
        m = DarkoClassifier(
            iterations=120, random_state=0, sampling="goss",
            top_rate=0.3, other_rate=0.2, tree_mode=tree_mode,
        ).fit(Xtr, ytr)
        acc = (m.predict(Xte) == yte).mean()
        assert acc > 0.85, (tree_mode, acc)

    with pytest.raises(ValueError):
        DarkoClassifier(
            iterations=5, sampling="goss", tree_mode="lightgbm",
            multiclass_tree_strategy="shared_vector",
        ).fit(X, y)
    with pytest.raises(ValueError):
        DarkoClassifier(
            iterations=5, sampling="goss", subsample=0.8,
        ).fit(X, y)


def test_stochastic_regularization_defaults_match_disabled_explicit():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    base = DarkoRegressor(iterations=20, random_state=0).fit(X, y)
    explicit = DarkoRegressor(
        iterations=20,
        random_state=0,
        bootstrap_type="none",
        bagging_temperature=1.0,
        sampling="uniform",
        mvs_reg=3.0,
        random_strength=0.0,
    ).fit(X, y)

    assert np.array_equal(base.predict(X), explicit.predict(X))
    meta = explicit.model_.auto_params_["stochastic_regularization"]
    assert meta["bayesian_bootstrap_active"] is False
    assert meta["random_strength_active"] is False


def test_bayesian_bootstrap_is_seeded_and_all_ones_weight_equivalent():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    kw = dict(
        iterations=25,
        random_state=11,
        bootstrap_type="bayesian",
        bagging_temperature=1.0,
    )
    a = DarkoRegressor(**kw).fit(X, y)
    b = DarkoRegressor(**kw).fit(X, y)
    ones = DarkoRegressor(**kw).fit(X, y, sample_weight=np.ones(len(y)))
    different_seed = DarkoRegressor(
        **{**kw, "random_state": 12}
    ).fit(X, y)

    assert np.array_equal(a.predict(X), b.predict(X))
    assert np.array_equal(a.predict(X), ones.predict(X))
    assert not np.array_equal(a.predict(X), different_seed.predict(X))
    meta = a.model_.auto_params_["stochastic_regularization"]
    assert meta["bayesian_bootstrap_active"] is True
    assert meta["bayesian_bootstrap_rounds"] == len(a.model_.trees_)


def test_bayesian_bootstrap_temperature_zero_matches_no_bootstrap():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    base = DarkoRegressor(iterations=20, random_state=0).fit(X, y)
    zero = DarkoRegressor(
        iterations=20,
        random_state=0,
        bootstrap_type="bayesian",
        bagging_temperature=0.0,
    ).fit(X, y)

    assert np.array_equal(base.predict(X), zero.predict(X))


def test_mvs_sampling_diagnostics_and_full_fraction_parity():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    full_uniform = DarkoRegressor(
        iterations=20, random_state=0, sampling="uniform", subsample=1.0
    ).fit(X, y)
    full_mvs = DarkoRegressor(
        iterations=20, random_state=0, sampling="mvs", subsample=1.0
    ).fit(X, y)
    sampled = DarkoRegressor(
        iterations=10,
        random_state=0,
        sampling="mvs",
        subsample=0.5,
        mvs_reg=2.0,
    ).fit(X, y)

    assert np.array_equal(full_uniform.predict(X), full_mvs.predict(X))
    meta = sampled.model_.auto_params_["stochastic_regularization"]
    assert meta["mvs_active"] is True
    assert meta["sampling_rounds"] == len(sampled.model_.trees_)
    assert 0.2 <= meta["average_sampled_row_fraction"] <= 0.8
    assert sampled.model_.auto_params_["sampling"]["mvs_reg"] == 2.0


def _old_mvs_probabilities_reference(importance, target):
    importance = np.asarray(importance, dtype=np.float64)
    n_samples = importance.shape[0]
    if target <= 0.0:
        return np.zeros(n_samples, dtype=np.float64)
    if (
        not np.all(np.isfinite(importance))
        or float(np.sum(importance)) <= 0.0
        or np.all(importance <= 0.0)
    ):
        return np.full(n_samples, target / n_samples, dtype=np.float64)
    lo = 0.0
    hi = float(np.max(importance)) * n_samples / max(target, 1.0)
    if hi <= 0.0:
        return np.full(n_samples, target / n_samples, dtype=np.float64)
    for _ in range(48):
        mid = (lo + hi) * 0.5
        if mid <= 0.0:
            lo = mid
            continue
        expected = np.minimum(1.0, importance / mid).sum()
        if expected > target:
            lo = mid
        else:
            hi = mid
    probs = np.minimum(1.0, importance / max(hi, 1e-300))
    probs *= target / max(float(probs.sum()), 1e-300)
    return np.minimum(1.0, probs)


def _old_weighted_goss_probabilities_reference(mass, target_mass):
    mass = np.asarray(mass, dtype=np.float64)
    if (
        mass.size == 0
        or target_mass <= 0.0
        or not np.all(np.isfinite(mass))
        or float(np.sum(mass)) <= 0.0
    ):
        return None
    target_mass = min(float(target_mass), float(np.sum(mass)))
    lo = 0.0
    hi = 1.0 / max(float(np.max(mass)), 1e-300)
    while np.sum(np.minimum(1.0, hi * mass) * mass) < target_mass:
        hi *= 2.0
    for _ in range(48):
        mid = (lo + hi) * 0.5
        expected_mass = np.sum(np.minimum(1.0, mid * mass) * mass)
        if expected_mass < target_mass:
            lo = mid
        else:
            hi = mid
    return np.minimum(1.0, hi * mass)


def test_exact_mvs_probabilities_match_bisection_reference():
    from darkofit.booster import GradientBoosting

    booster = GradientBoosting(iterations=1, sampling="mvs", subsample=0.4)
    cases = [
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 3.0, 9.0]),
        np.array([1e-12, 1e-9, 1.0, 5.0, 20.0]),
        np.array([0.2, 4.0]),
        np.linspace(0.1, 3.0, 17),
    ]
    for importance in cases:
        target = booster.subsample * importance.shape[0]
        got = booster._mvs_probabilities(importance)
        expected = _old_mvs_probabilities_reference(importance, target)
        assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)
        assert np.all((0.0 <= got) & (got <= 1.0))

    assert booster._mvs_probabilities(np.arange(4.0)) is not None
    full = GradientBoosting(iterations=1, sampling="mvs", subsample=1.0)
    assert full._mvs_probabilities(np.arange(4.0)) is None

    invalid = np.array([np.nan, 1.0, 2.0])
    assert np.allclose(
        booster._mvs_probabilities(invalid),
        _old_mvs_probabilities_reference(invalid, booster.subsample * 3),
        rtol=0.0,
        atol=0.0,
    )
    zeros = np.zeros(5)
    assert np.allclose(
        booster._mvs_probabilities(zeros),
        _old_mvs_probabilities_reference(zeros, booster.subsample * 5),
        rtol=0.0,
        atol=0.0,
    )


def test_exact_weighted_goss_probabilities_match_bisection_reference():
    from darkofit.booster import GradientBoosting

    booster = GradientBoosting(
        iterations=1, sampling="weighted_goss", top_rate=0.25, other_rate=0.35
    )
    cases = [
        (np.array([1.0, 1.0, 1.0, 1.0]), 1.4),
        (np.array([0.0, 0.5, 2.0, 8.0]), 2.5),
        (np.array([1e-12, 1e-6, 0.25, 4.0, 9.0]), 3.0),
        (np.array([0.2]), 1.0),
        (np.linspace(0.2, 2.0, 19), 5.5),
    ]
    for mass, target_mass in cases:
        got = booster._weighted_goss_probabilities(mass, target_mass)
        expected = _old_weighted_goss_probabilities_reference(mass, target_mass)
        assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)
        assert np.all((0.0 <= got) & (got <= 1.0))

    assert booster._weighted_goss_probabilities(np.array([]), 1.0) is None
    assert booster._weighted_goss_probabilities(np.array([np.nan, 1.0]), 1.0) is None
    assert booster._weighted_goss_probabilities(np.zeros(3), 1.0) is None


def test_mvs_realized_sample_count_matches_probability_mass():
    from darkofit.booster import GradientBoosting

    rng = np.random.default_rng(20260707)
    importance = rng.lognormal(size=200)
    booster = GradientBoosting(iterations=1, sampling="mvs", subsample=0.37)
    probs = booster._mvs_probabilities(importance)
    expected = float(np.sum(probs))
    draws = []
    for seed in range(600):
        mask = np.random.default_rng(seed).random(probs.shape[0]) < probs
        draws.append(float(np.sum(mask)))
    realized = float(np.mean(draws))
    std_err = float(np.sqrt(np.sum(probs * (1.0 - probs)) / len(draws)))
    assert abs(realized - expected) <= 4.0 * std_err


def test_weighted_goss_realized_sample_mass_matches_probability_mass():
    from darkofit.booster import GradientBoosting

    rng = np.random.default_rng(20260708)
    mass = rng.lognormal(mean=-0.1, sigma=0.7, size=180)
    target_mass = 0.35 * float(np.sum(mass))
    booster = GradientBoosting(
        iterations=1, sampling="weighted_goss",
        top_rate=0.2, other_rate=0.35,
    )
    probs = booster._weighted_goss_probabilities(mass, target_mass)
    expected = float(np.sum(probs * mass))
    draws = []
    for seed in range(600):
        mask = np.random.default_rng(seed).random(probs.shape[0]) < probs
        draws.append(float(np.sum(mass[mask])))
    realized = float(np.mean(draws))
    variance = float(np.sum(probs * (1.0 - probs) * mass * mass))
    std_err = float(np.sqrt(variance / len(draws)))
    assert abs(realized - expected) <= 4.0 * std_err


def test_multiclass_mvs_sampling_diagnostics_once_per_round():
    from sklearn.datasets import load_wine

    X, y = load_wine(return_X_y=True)
    model = DarkoClassifier(
        iterations=6,
        sampling="mvs",
        subsample=0.5,
        random_state=0,
    ).fit(X, y)

    meta = model.model_.auto_params_["stochastic_regularization"]
    assert meta["mvs_active"] is True
    assert meta["sampling_rounds"] == len(model.model_.trees_)


def test_mvs_rejects_invalid_subsample():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    for subsample in (0.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="subsample must"):
            DarkoRegressor(
                iterations=1,
                sampling="mvs",
                subsample=subsample,
            ).fit(X, y)


def test_weighted_goss_is_opt_in_and_records_diagnostics():
    rng = np.random.default_rng(107)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] - X[:, 1] + rng.normal(0.0, 0.1, size=120)
    w = np.ones(120)
    w[:10] = 15.0

    model = DarkoRegressor(
        iterations=8,
        random_state=0,
        tree_mode="lightgbm",
        num_leaves=7,
        min_child_samples=2,
        min_child_weight=0.0,
        sampling="weighted_goss",
        top_rate=0.25,
        other_rate=0.25,
    ).fit(X, y, sample_weight=w)

    meta = model.model_.auto_params_["stochastic_regularization"]
    assert model.model_.sampling_ == "weighted_goss"
    assert meta["weighted_goss_active"] is True
    assert meta["row_sampling_active"] is True
    assert meta["sampling_rounds"] == len(model.model_.trees_)
    assert 0.0 < meta["average_sampled_row_fraction"] < 1.0


def test_weighted_goss_uniform_mass_fast_path_avoids_full_sort(monkeypatch):
    import darkofit.booster as booster_mod

    def fail_argsort(*args, **kwargs):
        raise AssertionError("uniform-mass weighted GOSS should not full-sort")

    monkeypatch.setattr(booster_mod.np, "argsort", fail_argsort)

    grad = np.linspace(-1.0, 1.0, 20)
    hess = np.ones_like(grad)
    score = np.abs(grad)
    mass = np.ones_like(grad)
    booster = booster_mod.GradientBoosting(
        iterations=1, sampling="weighted_goss",
        top_rate=0.25, other_rate=0.25,
    )
    g, h, row_indices = booster._weighted_goss_subsample_from_score(
        grad, hess, score, mass, np.random.default_rng(0)
    )

    assert row_indices is not None
    top_count = int(np.ceil(booster.top_rate * grad.shape[0]))
    top_idx = np.argpartition(score, grad.shape[0] - top_count)[-top_count:]
    assert set(top_idx).issubset(set(row_indices.tolist()))
    assert np.array_equal(g[top_idx], grad[top_idx])
    assert np.array_equal(h[top_idx], hess[top_idx])
    unsampled = np.ones(grad.shape[0], dtype=bool)
    unsampled[row_indices] = False
    assert np.all(g[unsampled] == 0.0)


def test_weighted_goss_nonuniform_top_mass_avoids_full_score_sort(monkeypatch):
    import darkofit.booster as booster_mod

    original_argsort = booster_mod.np.argsort
    n_samples = 10_000

    def fail_full_score_argsort(a, *args, **kwargs):
        if np.asarray(a).shape == (n_samples,):
            raise AssertionError("weighted GOSS should not full-sort scores")
        return original_argsort(a, *args, **kwargs)

    monkeypatch.setattr(booster_mod.np, "argsort", fail_full_score_argsort)

    grad = np.linspace(-1.0, 1.0, n_samples)
    hess = np.ones_like(grad)
    score = np.linspace(0.0, 1.0, n_samples)
    mass = np.linspace(0.5, 1.5, n_samples)
    booster = booster_mod.GradientBoosting(
        iterations=1, sampling="weighted_goss",
        top_rate=0.2, other_rate=0.2,
    )
    g, h, row_indices = booster._weighted_goss_subsample_from_score(
        grad, hess, score, mass, np.random.default_rng(0)
    )

    full_order = original_argsort(score)[::-1]
    top_count = int(
        np.searchsorted(
            np.cumsum(mass[full_order]),
            booster.top_rate * float(np.sum(mass)),
            side="left",
        )
        + 1
    )
    top_idx = full_order[:top_count]

    assert row_indices is not None
    assert set(top_idx).issubset(set(row_indices.tolist()))
    assert np.array_equal(g[top_idx], grad[top_idx])
    assert np.array_equal(h[top_idx], hess[top_idx])


def test_weighted_goss_nonuniform_final_scaling_and_multiclass_shared_rows():
    from darkofit.booster import GradientBoosting

    class FixedRng:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float64)

        def random(self, size):
            assert size <= self.values.shape[0]
            return self.values[:size]

    grad = np.array([-2.0, -0.5, 0.25, 1.5, 0.75, -1.25])
    hess = np.linspace(0.4, 1.4, grad.shape[0])
    score = np.abs(grad)
    mass = np.array([0.5, 1.5, 0.25, 3.0, 0.75, 2.0])
    booster = GradientBoosting(
        iterations=1, sampling="weighted_goss",
        top_rate=0.3, other_rate=0.4,
    )

    top_idx = booster._weighted_goss_top_indices(
        score, mass, booster.top_rate * float(np.sum(mass))
    )
    remaining_mask = np.ones(grad.shape[0], dtype=bool)
    remaining_mask[top_idx] = False
    remaining_idx = np.flatnonzero(remaining_mask)
    probs = booster._weighted_goss_probabilities(
        mass[remaining_idx], booster.other_rate * float(np.sum(mass))
    )
    draws = np.array([0.0, 0.95, 0.15, 0.85, 0.35])[:remaining_idx.shape[0]]
    other_idx = remaining_idx[draws < probs]

    g, h, row_indices = booster._weighted_goss_subsample_from_score(
        grad, hess, score, mass, FixedRng(draws)
    )
    assert np.array_equal(row_indices, np.sort(np.concatenate((top_idx, other_idx))))
    assert np.array_equal(g[top_idx], grad[top_idx])
    assert np.array_equal(h[top_idx], hess[top_idx])
    for row in other_idx:
        pos = int(np.flatnonzero(remaining_idx == row)[0])
        assert g[row] == grad[row] / probs[pos]
        assert h[row] == hess[row] / probs[pos]
    unsampled = np.ones(grad.shape[0], dtype=bool)
    unsampled[row_indices] = False
    assert np.all(g[unsampled] == 0.0)
    assert np.all(h[unsampled] == 0.0)

    grad_mc = np.vstack([grad, -0.25 * grad, 0.5 * grad])
    hess_mc = np.vstack([hess, hess + 0.1, hess + 0.2])
    g_mc, h_mc, row_indices_mc = booster._weighted_goss_subsample_from_score(
        grad_mc, hess_mc, score, mass, FixedRng(draws)
    )
    assert np.array_equal(row_indices_mc, row_indices)
    assert np.all(g_mc[:, unsampled] == 0.0)
    assert np.all(h_mc[:, unsampled] == 0.0)
    assert np.array_equal(g_mc[:, top_idx], grad_mc[:, top_idx])
    assert np.array_equal(h_mc[:, top_idx], hess_mc[:, top_idx])


def test_weighted_goss_empty_other_draw_does_not_force_biased_row():
    from darkofit.booster import GradientBoosting

    class EmptyDrawRng:
        def random(self, size):
            return np.ones(size, dtype=np.float64)

    grad = np.linspace(-1.0, 1.0, 12)
    hess = np.ones_like(grad)
    score = np.abs(grad)
    uniform_mass = np.ones_like(grad)
    nonuniform_mass = np.linspace(0.5, 1.5, grad.shape[0])
    booster = GradientBoosting(
        iterations=1, sampling="weighted_goss",
        top_rate=0.25, other_rate=0.1,
    )

    g, h, row_indices = booster._weighted_goss_subsample_from_score(
        grad, hess, score, uniform_mass, EmptyDrawRng()
    )
    top_count = int(np.ceil(booster.top_rate * grad.shape[0]))
    top_idx = np.argpartition(score, grad.shape[0] - top_count)[-top_count:]
    assert np.array_equal(row_indices, np.sort(top_idx))
    assert np.array_equal(g[top_idx], grad[top_idx])
    not_top = np.ones(grad.shape[0], dtype=bool)
    not_top[top_idx] = False
    assert np.all(g[not_top] == 0.0)
    assert np.all(h[not_top] == 0.0)

    g2, h2, row_indices2 = booster._weighted_goss_subsample_from_score(
        grad, hess, score, nonuniform_mass, EmptyDrawRng()
    )
    top_idx2 = booster._weighted_goss_top_indices(
        score, nonuniform_mass, booster.top_rate * float(np.sum(nonuniform_mass))
    )
    assert np.array_equal(row_indices2, np.sort(top_idx2))
    not_top2 = np.ones(grad.shape[0], dtype=bool)
    not_top2[top_idx2] = False
    assert np.all(g2[not_top2] == 0.0)
    assert np.all(h2[not_top2] == 0.0)


def test_multiclass_mvs_uses_shared_row_sample_per_round(monkeypatch):
    from sklearn.datasets import load_wine
    import darkofit.booster as booster_mod

    X, y = load_wine(return_X_y=True)
    seen = []
    original = booster_mod.build_oblivious_tree

    def capture(*args, **kwargs):
        rows = kwargs.get("row_indices")
        seen.append(None if rows is None else tuple(rows.tolist()))
        return original(*args, **kwargs)

    monkeypatch.setattr(booster_mod, "build_oblivious_tree", capture)
    model = DarkoClassifier(
        iterations=3,
        random_state=0,
        sampling="mvs",
        subsample=0.6,
        multiclass_tree_strategy="per_class",
    ).fit(X, y)

    K = model.n_classes_
    assert len(seen) == len(model.model_.trees_) * K
    for start in range(0, len(seen), K):
        assert len(set(seen[start:start + K])) == 1


def test_random_strength_is_seeded_and_stores_true_gain():
    from darkofit.tree import build_oblivious_tree

    Xb = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
        [3, 0],
        [3, 1],
    ], dtype=np.uint8)
    grad = np.array([-0.7, -0.6, -0.1, 0.0, 0.1, 0.2, 0.6, 0.7])
    hess = np.ones_like(grad)
    n_bins = np.array([4, 2], dtype=np.int64)

    a = build_oblivious_tree(
        Xb, grad, hess, n_bins, 2, 1.0, 0.1,
        random_strength=5.0, split_seed=4, tree_iteration=2,
    )
    b = build_oblivious_tree(
        Xb, grad, hess, n_bins, 2, 1.0, 0.1,
        random_strength=5.0, split_seed=4, tree_iteration=2,
    )
    c = build_oblivious_tree(
        Xb, grad, hess, n_bins, 2, 1.0, 0.1,
        random_strength=5.0, split_seed=5, tree_iteration=2,
    )

    assert np.array_equal(a.splits_feat, b.splits_feat)
    assert np.array_equal(a.splits_thr, b.splits_thr)
    assert np.array_equal(a.gains, b.gains)
    assert np.all(np.isfinite(a.gains))
    assert np.all(a.gains < 10.0)
    assert (
        not np.array_equal(a.splits_feat, c.splits_feat)
        or not np.array_equal(a.splits_thr, c.splits_thr)
    )


def test_random_strength_filters_min_gain_before_noisy_argmax():
    from darkofit.tree import _best_split_with_noise_py

    hg = np.zeros((2, 1, 2), dtype=np.float64)
    hh = np.ones((2, 1, 2), dtype=np.float64)
    hg[0, 0, :] = [1.0, -1.0]      # true gain = 2.0
    hg[1, 0, :] = [0.5, -0.5]      # true gain = 0.5, below threshold
    n_bins = np.array([2, 2], dtype=np.int64)
    feat_mask = np.ones(2, dtype=np.uint8)

    f, t, gain = _best_split_with_noise_py(
        hg,
        hh,
        n_bins,
        0.0,
        feat_mask,
        0.0,
        1,
        10.0,
        0,
        0,
        0,
        1.0,
    )

    assert (f, t) == (0, 0)
    assert gain == 2.0


def test_leafwise_random_strength_rescores_all_leaves(monkeypatch):
    import darkofit.tree as tree_mod

    calls = []
    original = tree_mod._best_splits_counts_for_leaf_ids_with_noise_py

    def capture(*args, **kwargs):
        leaf_ids = args[8]
        n_leaf_ids = args[9]
        calls.append((leaf_ids[:n_leaf_ids].copy(), int(n_leaf_ids)))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        tree_mod, "_best_splits_counts_for_leaf_ids_with_noise_py", capture
    )
    rng = np.random.default_rng(47)
    Xb = rng.integers(0, 24, size=(180, 8), dtype=np.uint8)
    n_bins = np.full(Xb.shape[1], 24, dtype=np.int64)
    grad = rng.normal(size=Xb.shape[0])
    hess = rng.uniform(0.2, 1.4, size=Xb.shape[0])

    tree_mod.build_leafwise_tree(
        Xb,
        grad,
        hess,
        n_bins,
        5,
        1.0,
        0.1,
        max_leaves=6,
        min_child_samples=2,
        min_child_weight=0.1,
        min_gain_to_split=0.0,
        random_strength=0.5,
        split_seed=0,
        tree_iteration=0,
    )

    assert any(
        n_leaf_ids > 2 and np.array_equal(leaf_ids, np.arange(n_leaf_ids))
        for leaf_ids, n_leaf_ids in calls
    )


def test_stochastic_regularization_persists_through_save_load(tmp_path):
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = DarkoRegressor(
        iterations=8,
        random_state=0,
        sampling="mvs",
        subsample=0.7,
        mvs_reg=2.5,
        bootstrap_type="bayesian",
        bagging_temperature=0.4,
        random_strength=0.3,
    ).fit(X, y)
    path = tmp_path / "stochastic.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)

    assert loaded.model_.sampling == "mvs"
    assert loaded.model_.bootstrap_type == "bayesian"
    assert loaded.model_.bagging_temperature == 0.4
    assert loaded.model_.mvs_reg == 2.5
    assert loaded.model_.random_strength == 0.3
    assert loaded.model_.auto_params_["stochastic_regularization"]["mvs_active"] is True
    assert np.array_equal(model.predict(X), loaded.predict(X))
