"""Public input-boundary and scikit-learn compatibility tests."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.estimator_checks import check_estimator

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.booster import (
    DistributionalBoosting,
    GradientBoosting,
    MulticlassBoosting,
)


def _regression_data(n=48):
    rng = np.random.default_rng(14)
    X = rng.normal(size=(n, 3))
    y = 1.3 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, y


def _regressor(**kwargs):
    return DarkoRegressor(
        iterations=3,
        thread_count=1,
        diagnostic_warnings="never",
        **kwargs,
    )


@pytest.mark.parametrize(
    "factory",
    [
        _regressor,
        lambda: GradientBoosting(
            iterations=3, thread_count=1, diagnostic_warnings="never"
        ),
    ],
)
@pytest.mark.parametrize("kind", ["masked", "complex", "infinite"])
def test_invalid_numeric_fit_input_is_rejected_before_lossy_conversion(
    factory, kind
):
    X, y = _regression_data()
    if kind == "masked":
        bad = np.ma.array(X, mask=np.zeros_like(X, dtype=bool))
        error, match = TypeError, "filled"
    elif kind == "complex":
        bad = X.astype(np.complex128) + 1j
        error, match = ValueError, "Complex"
    else:
        bad = X.copy()
        bad[0, 0] = np.inf
        error, match = ValueError, "infinity"
    with pytest.raises(error, match=match):
        factory().fit(bad, y)


@pytest.mark.parametrize(
    "factory",
    [
        _regressor,
        lambda: GradientBoosting(
            iterations=3, thread_count=1, diagnostic_warnings="never"
        ),
    ],
)
@pytest.mark.parametrize("kind", ["masked", "complex", "infinite"])
def test_invalid_eval_and_predict_input_is_rejected(factory, kind):
    X, y = _regression_data()
    if kind == "masked":
        bad = np.ma.array(X[:8], mask=np.zeros_like(X[:8], dtype=bool))
        error, match = TypeError, "filled"
    elif kind == "complex":
        bad = X[:8].astype(np.complex128) + 1j
        error, match = ValueError, "Complex"
    else:
        bad = X[:8].copy()
        bad[0, 0] = -np.inf
        error, match = ValueError, "infinity"
    with pytest.raises(error, match=match):
        factory().fit(X[8:], y[8:], eval_set=(bad, y[:8]))
    model = factory().fit(X, y)
    with pytest.raises(error, match=match):
        model.predict_raw(bad) if isinstance(
            model, GradientBoosting
        ) else model.predict(bad)
    staged = (
        model.staged_predict_raw(bad)
        if isinstance(model, GradientBoosting)
        else model.staged_predict(bad)
    )
    with pytest.raises(error, match=match):
        next(staged)


@pytest.mark.parametrize(
    "factory",
    [
        _regressor,
        lambda: GradientBoosting(
            iterations=3, thread_count=1, diagnostic_warnings="never"
        ),
    ],
)
def test_assume_finite_skips_all_feature_infinity_checks(factory):
    X, y = _regression_data()
    X_inf = X.copy()
    X_inf[0, 0] = np.inf
    X_inf[8, 1] = -np.inf
    X_nan = X_inf.copy()
    X_nan[~np.isfinite(X_nan)] = np.nan

    with pytest.raises(ValueError, match="infinity"):
        factory().fit(X_inf, y)
    model = factory().fit(X, y)
    with pytest.raises(ValueError, match="infinity"):
        (
            model.predict_raw(X_inf[:2])
            if isinstance(model, GradientBoosting)
            else model.predict(X_inf[:2])
        )

    with config_context(assume_finite=True):
        inf_model = factory().fit(
            X_inf[8:],
            y[8:],
            eval_set=(X_inf[:8], y[:8]),
        )
        nan_model = factory().fit(
            X_nan[8:],
            y[8:],
            eval_set=(X_nan[:8], y[:8]),
        )
        inf_prediction = (
            inf_model.predict_raw(X_inf)
            if isinstance(inf_model, GradientBoosting)
            else inf_model.predict(X_inf)
        )
        nan_prediction = (
            nan_model.predict_raw(X_nan)
            if isinstance(nan_model, GradientBoosting)
            else nan_model.predict(X_nan)
        )
        y_inf = y.copy()
        y_inf[0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            factory().fit(X, y_inf)
    np.testing.assert_array_equal(inf_prediction, nan_prediction)


@pytest.mark.parametrize("categorical", [False, True])
def test_wrapper_prediction_coerces_and_scans_only_once(
    monkeypatch, categorical
):
    import darkofit.booster as booster_module
    import darkofit.sklearn_api as sklearn_module

    X, y = _regression_data()
    if categorical:
        X = pd.DataFrame(
            {
                "value": X[:, 0],
                "team": np.where(X[:, 1] > 0, "A", "B"),
                "z": X[:, 2],
            }
        )
        model = _regressor().fit(X, y, cat_features=["team"])
        X_predict = X.iloc[:8]
    else:
        model = _regressor().fit(X, y)
        X_predict = X[:8]

    calls = []
    original = sklearn_module.coerce_feature_matrix

    def counted(*args, **kwargs):
        calls.append(kwargs.get("name", "X"))
        return original(*args, **kwargs)

    monkeypatch.setattr(sklearn_module, "coerce_feature_matrix", counted)
    monkeypatch.setattr(booster_module, "coerce_feature_matrix", counted)
    assert model.predict(X_predict).shape == (8,)
    assert calls == ["X"]


def test_categorical_predict_success_path_avoids_elementwise_complex_scan(
    monkeypatch,
):
    import darkofit._validation as validation_module

    X, y = _regression_data()
    frame = pd.DataFrame(
        {
            "value": X[:, 0],
            "team": np.where(X[:, 1] > 0, "A", "B"),
            "z": X[:, 2],
        }
    )
    model = _regressor().fit(frame, y, cat_features=["team"])

    def forbidden_scan(_values):
        raise AssertionError(
            "valid categorical prediction must not scan numeric object cells"
        )

    monkeypatch.setattr(
        validation_module, "_contains_complex", forbidden_scan
    )
    assert model.predict(frame.iloc[:8]).shape == (8,)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1 + 0j, id="python-complex"),
        pytest.param(np.complex64(1 + 2j), id="numpy-complex64"),
        pytest.param(np.complex128(1 + 0j), id="numpy-complex128"),
    ],
)
def test_categorical_predict_rejects_object_boxed_complex_values(value):
    X, y = _regression_data()
    mixed = np.asarray(X, dtype=object)
    mixed[:, 1] = np.where(X[:, 1] > 0, "A", "B")
    model = _regressor().fit(mixed, y, cat_features=[1])
    bad = mixed[:4].copy()
    bad[0, 0] = value

    with pytest.raises(ValueError, match=r"^Complex data not supported\.$"):
        model.predict(bad)


def test_categorical_complex_rejection_precedes_nonnumeric_conversion_error():
    X, y = _regression_data()
    mixed = np.asarray(X, dtype=object)
    mixed[:, 1] = np.where(X[:, 1] > 0, "A", "B")
    model = _regressor().fit(mixed, y, cat_features=[1])
    bad = mixed[:4].copy()
    bad[0, 0] = "not numeric"
    bad[1, 0] = 1 + 2j

    with pytest.raises(ValueError, match=r"^Complex data not supported\.$"):
        model.predict(bad)


def test_nan_remains_supported_in_numeric_and_categorical_paths():
    X, y = _regression_data()
    X[0, 0] = np.nan
    numeric = _regressor().fit(X, y)
    assert np.all(np.isfinite(numeric.predict(X[:4])))

    frame = pd.DataFrame(
        {"value": X[:, 0], "team": np.where(X[:, 1] > 0, "A", "B")}
    )
    frame.loc[0, "team"] = None
    categorical = _regressor().fit(frame, y, cat_features=["team"])
    assert np.all(np.isfinite(categorical.predict(frame.iloc[:4])))


def test_named_categorical_features_resolve_and_duplicates_fail():
    X, y = _regression_data()
    frame = pd.DataFrame(
        {"value": X[:, 0], "team": np.where(X[:, 1] > 0, "A", "B"), "z": X[:, 2]}
    )
    wrapper = _regressor().fit(frame, y, cat_features=["team"])
    core = GradientBoosting(
        iterations=3, thread_count=1, diagnostic_warnings="never"
    ).fit(frame, y, cat_features=["team"])
    assert wrapper.model_.prep_.cat_features_ == [1]
    assert core.prep_.cat_features_ == [1]
    assert wrapper.feature_names_in_.tolist() == ["value", "team", "z"]

    with pytest.raises(ValueError, match="unknown column"):
        _regressor().fit(frame, y, cat_features=["missing"])
    with pytest.raises(ValueError, match="duplicate"):
        _regressor().fit(frame, y, cat_features=["team", 1])
    with pytest.raises(ValueError, match="named columns"):
        _regressor().fit(frame.to_numpy(object), y, cat_features=["team"])


def test_unmarked_nonnumeric_error_names_the_column():
    X, y = _regression_data()
    frame = pd.DataFrame(
        {"value": X[:, 0], "team_name": np.where(X[:, 1] > 0, "A", "B")}
    )
    with pytest.raises(ValueError, match="team_name.*cat_features"):
        _regressor().fit(frame, y)


def test_feature_names_are_ordered_and_named_unnamed_transitions_warn():
    X, y = _regression_data()
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    named = _regressor().fit(frame, y)
    with pytest.raises(ValueError, match="feature names"):
        named.predict(frame[["c", "b", "a"]])
    with pytest.warns(UserWarning, match="without feature names"):
        named.predict(X[:3])

    unnamed = _regressor().fit(X, y)
    with pytest.warns(UserWarning, match="has feature names"):
        unnamed.predict(frame.iloc[:3])
    with pytest.raises(ValueError, match="feature names"):
        _regressor().fit(
            frame[8:],
            y[8:],
            eval_set=(frame[:8].rename(columns={"a": "renamed"}), y[:8]),
        )


def test_nullable_pandas_numeric_values_are_missing_not_conversion_errors():
    X, y = _regression_data()
    nullable = pd.array(np.round(10 * X[:, 0]), dtype="Int64")
    nullable[::7] = pd.NA
    frame = pd.DataFrame({"a": nullable, "b": X[:, 1]})
    model = _regressor().fit(frame, y)
    prediction = model.predict(frame.iloc[:8])
    assert np.all(np.isfinite(prediction))


class _PolarsLike:
    def __init__(self, values, columns):
        self._values = np.asarray(values)
        self.columns = list(columns)
        self.shape = self._values.shape

    def to_numpy(self):
        return self._values


class _PyArrowLike:
    def __init__(self, values, columns):
        self._values = np.asarray(values)
        self.column_names = list(columns)
        self.shape = self._values.shape

    def to_pandas(self):
        raise AssertionError("PyArrow-like conversion must not require pandas")

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._values, dtype=dtype)


@pytest.mark.parametrize("frame_type", [_PolarsLike, _PyArrowLike])
def test_frame_like_names_and_conversion_do_not_require_optional_imports(frame_type):
    X, y = _regression_data()
    frame = frame_type(X, ["a", "b", "c"])
    model = _regressor().fit(frame, y)
    assert model.feature_names_in_.tolist() == ["a", "b", "c"]
    assert model.predict(frame).shape == (len(y),)


def test_real_pyarrow_and_polars_inputs_when_installed():
    X, y = _regression_data()
    checked = 0
    try:
        import pyarrow as pa
    except ImportError:
        pass
    else:
        table = pa.table({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2]})
        assert _regressor().fit(table, y).predict(table).shape == (len(y),)
        checked += 1
    try:
        import polars as pl
    except ImportError:
        pass
    else:
        frame = pl.DataFrame({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2]})
        assert _regressor().fit(frame, y).predict(frame).shape == (len(y),)
        checked += 1
    if checked == 0:
        pytest.skip("neither pyarrow nor polars is installed")


def test_distributional_and_multiclass_core_prediction_boundaries():
    X, y = _regression_data(64)
    dist = DistributionalBoosting(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=3,
        thread_count=1,
        diagnostic_warnings="never",
    ).fit(X, y)
    multiclass = MulticlassBoosting(
        iterations=3, thread_count=1, diagnostic_warnings="never"
    ).fit(X, np.arange(len(X)) % 3)
    bad = X[:3].copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        dist.predict_dist(bad)
    with pytest.raises(ValueError, match="infinity"):
        multiclass.predict_raw(bad)
    assert dist.predict_raw(X[:0]).shape == (0, 2)
    assert all(
        parameter.shape == (0,)
        for parameter in dist.predict_dist(X[:0])
    )
    assert multiclass.predict_raw(X[:0]).shape == (0, 3)


def test_zero_row_prediction_returns_empty_outputs():
    X, y = _regression_data(64)
    empty = X[:0]

    regressor = _regressor().fit(X, y)
    assert regressor.predict(empty).shape == (0,)
    assert all(
        prediction.shape == (0,)
        for prediction in regressor.staged_predict(empty)
    )
    with pytest.raises(ValueError, match="expecting 3 features"):
        regressor.predict(np.empty((0, 2)))
    with pytest.raises(ValueError, match="0 sample"):
        _regressor().fit(X, y, eval_set=(empty, y[:0]))

    core = GradientBoosting(
        iterations=3,
        thread_count=1,
        diagnostic_warnings="never",
    ).fit(X, y)
    assert core.predict_raw(empty).shape == (0,)
    assert all(
        prediction.shape == (0,)
        for prediction in core.staged_predict_raw(empty)
    )

    binary = DarkoClassifier(
        iterations=3,
        thread_count=1,
        diagnostic_warnings="never",
    ).fit(X, np.arange(len(X)) % 2)
    assert binary.predict(empty).shape == (0,)
    assert binary.predict_proba(empty).shape == (0, 2)
    assert all(
        probability.shape == (0, 2)
        for probability in binary.staged_predict_proba(empty)
    )

    multiclass = DarkoClassifier(
        iterations=3,
        thread_count=1,
        diagnostic_warnings="never",
    ).fit(X, np.arange(len(X)) % 3)
    assert multiclass.predict(empty).shape == (0,)
    assert multiclass.predict_proba(empty).shape == (0, 3)

    distributional = DarkoRegressor(
        loss="Gaussian",
        tree_mode="lightgbm",
        iterations=3,
        thread_count=1,
        diagnostic_warnings="never",
    ).fit(X, y)
    assert distributional.predict(empty).shape == (0,)
    assert distributional.predict_variance(empty).shape == (0,)
    assert all(
        parameter.shape == (0,)
        for parameter in distributional.predict_dist(empty)
    )
    assert all(
        bound.shape == (0,)
        for bound in distributional.predict_interval(empty)
    )
    assert distributional.sample(
        empty, n_samples=4, random_state=0
    ).shape == (0, 4)


def test_zero_row_categorical_prediction_preserves_feature_schema():
    X, y = _regression_data()
    frame = pd.DataFrame(
        {
            "value": X[:, 0],
            "team": np.where(X[:, 1] > 0, "A", "B"),
            "z": X[:, 2],
        }
    )
    model = _regressor().fit(frame, y, cat_features=["team"])

    prediction = model.predict(frame.iloc[:0])
    assert prediction.shape == (0,)


def test_archive_round_trip_retains_input_metadata(tmp_path):
    X, y = _regression_data()
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    model = _regressor().fit(frame, y)
    path = tmp_path / "named.npz"
    model.save_model(path)
    loaded = DarkoRegressor.load_model(path)
    assert loaded.n_features_in_ == 3
    assert loaded.feature_names_in_.tolist() == ["a", "b", "c"]
    assert np.array_equal(model.predict(frame), loaded.predict(frame))

    core = GradientBoosting(
        iterations=3, thread_count=1, diagnostic_warnings="never"
    ).fit(frame, y)
    core_path = tmp_path / "named-core.npz"
    core.save_model(core_path)
    loaded_core = GradientBoosting.load_model(core_path)
    assert loaded_core.n_features_in_ == 3
    assert loaded_core.feature_names_in_.tolist() == ["a", "b", "c"]
    assert np.array_equal(
        core.predict_raw(frame), loaded_core.predict_raw(frame)
    )
    with pytest.raises(ValueError, match="feature names"):
        loaded_core.predict_raw(frame[["c", "b", "a"]])


def test_failed_core_refit_preserves_published_input_metadata():
    X, y = _regression_data()
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    model = GradientBoosting(
        iterations=3, thread_count=1, diagnostic_warnings="never"
    ).fit(frame, y)
    expected = model.predict_raw(frame.iloc[:4])

    wider = frame.assign(extra=1.0)
    with pytest.raises(ValueError, match="requires y"):
        model.fit(wider, None)
    assert model.n_features_in_ == 3
    assert model.feature_names_in_.tolist() == ["a", "b", "c"]
    assert np.array_equal(model.predict_raw(frame.iloc[:4]), expected)

    with pytest.raises(ValueError, match="feature names"):
        model.fit(
            frame,
            y,
            eval_set=(frame.rename(columns={"a": "renamed"}), y),
        )
    assert model.n_features_in_ == 3
    assert model.feature_names_in_.tolist() == ["a", "b", "c"]
    assert np.array_equal(model.predict_raw(frame.iloc[:4]), expected)


def test_sklearn_messages_and_tags():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="requires y"):
        _regressor().fit(X, None)
    with pytest.raises(ValueError, match="Reshape your data"):
        _regressor().fit(X[:, 0], y)
    with pytest.raises(ValueError, match="0 sample"):
        _regressor().fit(np.empty((0, 3)), np.empty(0))
    with pytest.raises(ValueError, match="zero"):
        _regressor().fit(X, y, sample_weight=np.zeros(len(y)))
    with pytest.warns(DataConversionWarning, match="1d array"):
        _regressor().fit(X, y[:, None])
    with pytest.raises(ValueError, match="1 class"):
        DarkoClassifier(
            iterations=3, thread_count=1, diagnostic_warnings="never"
        ).fit(X, np.zeros(len(y)))

    for estimator in (_regressor(), DarkoClassifier(iterations=3)):
        tags = estimator.__sklearn_tags__()
        if isinstance(tags, dict):
            assert tags["allow_nan"]
            assert "sparse" not in tags.get("X_types", ())
        else:
            assert tags.input_tags.allow_nan
            assert not tags.input_tags.sparse


@pytest.mark.parametrize("estimator", [DarkoRegressor, DarkoClassifier])
def test_full_sklearn_estimator_compliance(estimator):
    if "expected_failed_checks" not in inspect.signature(check_estimator).parameters:
        pytest.skip("expected-failure registration requires scikit-learn 1.6+")
    expected = {
        "check_sample_weight_equivalence_on_dense_data": (
            "weights reweight the loss but are not bit-exactly equivalent to "
            "integer row repetition"
        ),
        "check_sample_weight_equivalence_on_sparse_data": (
            "sparse matrices are intentionally unsupported"
        ),
    }
    check_estimator(
        estimator(thread_count=1, diagnostic_warnings="never"),
        expected_failed_checks=expected,
        on_skip=None,
    )
