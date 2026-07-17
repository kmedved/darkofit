from __future__ import annotations

import json
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.tuning import DarkoStepwiseSearchCV


PARAMS = {
    "iterations": 12,
    "learning_rate": 0.1,
    "depth": 3,
    "l2_leaf_reg": 1.0,
    "max_bins": 16,
    "ordered_boosting": False,
    "early_stopping": False,
    "random_state": 4,
    "thread_count": 1,
    "diagnostic_warnings": "never",
}


def _frame(n: int = 180):
    rng = np.random.default_rng(42)
    levels = np.asarray(["low", "medium", "high"], dtype=object)
    nominal = np.asarray(["east", "west", "north"], dtype=object)
    level = levels[rng.integers(0, len(levels), size=n)]
    region = nominal[rng.integers(0, len(nominal), size=n)]
    x = rng.normal(size=n)
    rank = pd.Series(level).map({"low": 0.0, "medium": 1.0, "high": 2.0})
    y = x + rank.to_numpy() + 0.2 * (region == "north") + rng.normal(scale=0.05, size=n)
    return pd.DataFrame({"x": x, "level": level, "region": region}), y


def _encoded(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["level"] = result["level"].map({"low": 0.0, "medium": 1.0, "high": 2.0})
    return result


class _NamedArray:
    def __init__(self, values, columns):
        self._values = np.asarray(values, dtype=object)
        self.columns = list(columns)
        self.shape = self._values.shape

    def to_numpy(self, dtype=None, na_value=None):
        del na_value
        return np.asarray(self._values, dtype=dtype)


def test_explicit_ordinal_matches_external_numeric_encoding_bitwise():
    X, y = _frame()
    ordinal = DarkoRegressor(**PARAMS).fit(
        X,
        y,
        cat_features=["level", "region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    numeric = DarkoRegressor(**PARAMS).fit(
        _encoded(X),
        y,
        cat_features=["region"],
    )
    assert np.array_equal(ordinal.predict(X), numeric.predict(_encoded(X)))
    assert ordinal.model_.prep_.cat_features_ == [2]
    assert ordinal.model_.prep_.num_features_ == [0, 1]
    assert np.array_equal(
        ordinal.model_.prep_.feature_map_,
        numeric.model_.prep_.feature_map_,
    )
    assert len(ordinal.model_.prep_.n_bins_) == X.shape[1]
    metadata = ordinal.model_.auto_params_["ordinal_features"]
    assert metadata == {
        "mode": "explicit",
        "active": True,
        "feature_count": 1,
        "feature_indices": [1],
        "feature_names": ["level"],
        "sources": ["explicit"],
        "nominal_categorical_count": 1,
        "added_columns": 0,
        "target_stat_blocks_added": 0,
        "target_used": False,
        "unknown_policy": "fail_closed",
        "missing_policy": "numeric_missing_bin",
    }


def test_none_is_an_exact_noop_and_adds_no_metadata():
    X, y = _frame()
    control = DarkoRegressor(**PARAMS).fit(X, y, cat_features=["level", "region"])
    explicit_none = DarkoRegressor(**PARAMS).fit(
        X,
        y,
        cat_features=["level", "region"],
        ordinal_features=None,
    )
    assert np.array_equal(control.predict(X), explicit_none.predict(X))
    assert control.model_.auto_params_ == explicit_none.model_.auto_params_
    assert "ordinal_features" not in control.model_.auto_params_
    assert explicit_none.ordinal_features_mode_ == "off"
    assert explicit_none.ordinal_features_ == []


def test_unknown_categories_fail_closed_and_missing_uses_numeric_missing_bin():
    X, y = _frame()
    X.loc[0, "level"] = None
    model = DarkoRegressor(**PARAMS).fit(
        X,
        y,
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    missing = X.iloc[:2].copy()
    missing.loc[missing.index[1], "level"] = np.nan
    assert np.isfinite(model.predict(missing)).all()
    unknown = X.iloc[:1].copy()
    unknown.loc[unknown.index[0], "level"] = "very high"
    with pytest.raises(ValueError, match="unknown ordinal category 'very high'"):
        model.predict(unknown)


def test_unhashable_observed_category_uses_the_unknown_category_error():
    X = pd.DataFrame(
        {
            "x": np.arange(4.0),
            "level": ["low", "high", ["low"], "low"],
        }
    )
    with pytest.raises(ValueError, match="unknown ordinal category"):
        DarkoRegressor(**PARAMS).fit(
            X,
            np.arange(4.0),
            ordinal_features={"level": ["low", "high"]},
        )


def test_auto_detects_integer_codes_and_ordered_pandas_categories_only():
    rng = np.random.default_rng(3)
    integers = pd.DataFrame(
        {
            "x": rng.normal(size=100),
            "level": rng.integers(1, 5, size=100),
        }
    )
    y = integers["x"] + integers["level"]
    integer_model = DarkoRegressor(**PARAMS).fit(
        integers,
        y,
        cat_features=["level"],
        ordinal_features="auto",
    )
    assert integer_model.ordinal_features_[0]["categories"] == [1, 2, 3, 4]
    assert integer_model.ordinal_features_[0]["source"] == "auto_integer_codes"
    assert integer_model.model_.prep_.cat_features_ == []

    ordered = pd.DataFrame(
        {
            "x": rng.normal(size=100),
            "level": pd.Categorical(
                np.resize(["small", "medium", "large"], 100),
                categories=["small", "medium", "large"],
                ordered=True,
            ),
        }
    )
    ordered_model = DarkoRegressor(**PARAMS).fit(
        ordered,
        rng.normal(size=100),
        ordinal_features="auto",
    )
    assert ordered_model.ordinal_features_[0]["categories"] == [
        "small",
        "medium",
        "large",
    ]
    assert ordered_model.ordinal_features_[0]["source"] == "auto_ordered_categorical"

    plain = ordered.copy()
    plain["level"] = plain["level"].astype(object)
    plain_model = DarkoRegressor(**PARAMS).fit(
        plain,
        rng.normal(size=100),
        cat_features=["level"],
        ordinal_features="auto",
    )
    assert plain_model.ordinal_features_ == []
    assert plain_model.model_.prep_.cat_features_ == [1]
    assert plain_model.model_.auto_params_["ordinal_features"]["active"] is False


@pytest.mark.parametrize(
    ("values", "categories"),
    [
        (["only"] * 8, ["only"]),
        ([None] * 8, []),
    ],
)
def test_auto_ordered_pandas_handles_degenerate_category_vocabularies(
    values, categories
):
    X = pd.DataFrame(
        {
            "x": np.arange(8.0),
            "level": pd.Categorical(
                values,
                categories=categories,
                ordered=True,
            ),
        }
    )
    model = DarkoRegressor(**PARAMS).fit(
        X,
        np.arange(8.0),
        ordinal_features="auto",
    )
    assert model.ordinal_features_[0]["categories"] == categories
    assert np.isfinite(model.predict(X)).all()


def test_byte_categories_match_their_json_safe_normalized_declaration():
    X = np.asarray(
        [
            [0.0, b"low"],
            [1.0, b"high"],
            [2.0, b"low"],
            [3.0, b"high"],
        ],
        dtype=object,
    )
    model = DarkoRegressor(**PARAMS).fit(
        X,
        np.arange(4.0),
        ordinal_features={1: [b"low", b"high"]},
    )
    assert np.isfinite(model.predict(X)).all()
    assert model.ordinal_features_[0]["categories"] == ["low", "high"]


def test_invalid_utf8_ordinal_bytes_fail_closed():
    X = np.asarray(
        [[0.0, b"low"], [1.0, b"high"], [2.0, b"low"]],
        dtype=object,
    )
    with pytest.raises(ValueError, match="byte categories must be valid UTF-8"):
        DarkoRegressor(**PARAMS).fit(
            X,
            np.arange(3.0),
            ordinal_features={1: [b"\xff", b"high"]},
        )

    model = DarkoRegressor(**PARAMS).fit(
        X,
        np.arange(3.0),
        ordinal_features={1: [b"low", b"high"]},
    )
    invalid = np.asarray([[3.0, b"\xff"]], dtype=object)
    with pytest.raises(ValueError, match="unknown ordinal category"):
        model.predict(invalid)


def test_numpy_extended_float_categories_are_normalized():
    X = np.asarray(
        [[0.0, np.longdouble(1.0)], [1.0, np.longdouble(2.0)]] * 4,
        dtype=object,
    )
    model = DarkoRegressor(**PARAMS).fit(
        X,
        np.arange(8.0),
        ordinal_features={
            1: [np.longdouble(1.0), np.longdouble(2.0)]
        },
    )
    assert model.ordinal_features_[0]["categories"] == [1.0, 2.0]
    assert np.isfinite(model.predict(X)).all()


def test_ordinal_eval_set_and_classifier_paths():
    X, y = _frame()
    params = {
        **PARAMS,
        "early_stopping": True,
        "early_stopping_rounds": 3,
    }
    regressor = DarkoRegressor(**params).fit(
        X.iloc[:140],
        y[:140],
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
        eval_set=(X.iloc[140:], y[140:]),
    )
    assert np.isfinite(regressor.predict(X.iloc[140:])).all()

    labels = (y > np.median(y)).astype(int)
    classifier = DarkoClassifier(**PARAMS).fit(
        X,
        labels,
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    probabilities = classifier.predict_proba(X.iloc[:5])
    assert probabilities.shape == (5, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert classifier.model_.auto_params_["ordinal_features"]["active"] is True


def test_eval_feature_names_are_checked_before_ordinal_transformation():
    X, y = _frame()
    reordered = X[["region", "x", "level"]]
    with pytest.raises(ValueError, match="feature names.*do not match"):
        DarkoRegressor(**PARAMS).fit(
            X,
            y,
            cat_features=["region"],
            ordinal_features={"level": ["low", "medium", "high"]},
            eval_set=(reordered, y),
        )


def test_named_non_pandas_input_keeps_name_validation_before_conversion():
    X, y = _frame()
    named = _NamedArray(X.to_numpy(), X.columns)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = DarkoRegressor(**PARAMS).fit(
            named,
            y,
            cat_features=["region"],
            ordinal_features={"level": ["low", "medium", "high"]},
            eval_set=(named, y),
        )
        prediction = model.predict(named)
    assert np.isfinite(prediction).all()
    assert np.array_equal(model.feature_names_in_, np.asarray(X.columns))
    assert not any("feature names" in str(item.message) for item in caught)


@pytest.mark.parametrize("estimator_cls", [DarkoRegressor, DarkoClassifier])
def test_ordinal_mapping_survives_safe_round_trip(tmp_path, estimator_cls):
    X, y = _frame()
    target = y if estimator_cls is DarkoRegressor else (y > np.median(y)).astype(int)
    model = estimator_cls(**PARAMS).fit(
        X,
        target,
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    before = (
        model.predict(X) if estimator_cls is DarkoRegressor else model.predict_proba(X)
    )
    path = tmp_path / "ordinal.npz"
    model.save_model(path)
    loaded = estimator_cls.load_model(path)
    after = (
        loaded.predict(X)
        if estimator_cls is DarkoRegressor
        else loaded.predict_proba(X)
    )
    assert np.array_equal(before, after)
    assert loaded.ordinal_features_ == model.ordinal_features_


def test_corrupt_ordinal_wrapper_state_is_rejected(tmp_path):
    X, y = _frame()
    model = DarkoRegressor(**PARAMS).fit(
        X,
        y,
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    valid = tmp_path / "valid.npz"
    corrupt = tmp_path / "corrupt.npz"
    model.save_model(valid)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    header = json.loads(arrays["header"].item())
    header["wrapper"]["state"]["ordinal_features"][0]["index"] = 999
    arrays["header"] = np.asarray(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)
    with pytest.raises(ValueError, match="ordinal feature index is invalid"):
        DarkoRegressor.load_model(corrupt)


def test_corrupt_ordinal_feature_name_state_is_rejected(tmp_path):
    X, y = _frame()
    model = DarkoRegressor(**PARAMS).fit(
        X,
        y,
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    valid = tmp_path / "valid.npz"
    corrupt = tmp_path / "corrupt.npz"
    model.save_model(valid)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    header = json.loads(arrays["header"].item())
    header["wrapper"]["state"]["feature_names_in"] = []
    arrays["header"] = np.asarray(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)
    with pytest.raises(ValueError, match="feature name state does not match"):
        DarkoRegressor.load_model(corrupt)


@pytest.mark.parametrize(
    ("state_path", "error"),
    [
        (("ordinal_features_mode",), "ordinal feature mode is invalid"),
        (
            ("ordinal_features", 0, "source"),
            "ordinal feature source is invalid",
        ),
    ],
)
def test_unhashable_ordinal_state_discriminators_are_rejected(
    tmp_path, state_path, error
):
    X, y = _frame()
    model = DarkoRegressor(**PARAMS).fit(
        X,
        y,
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    valid = tmp_path / "valid.npz"
    corrupt = tmp_path / "corrupt.npz"
    model.save_model(valid)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    header = json.loads(arrays["header"].item())
    target = header["wrapper"]["state"]
    for key in state_path[:-1]:
        target = target[key]
    target[state_path[-1]] = []
    arrays["header"] = np.asarray(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)
    with pytest.raises(ValueError, match=error):
        DarkoRegressor.load_model(corrupt)


def test_ordinal_state_cannot_relabel_a_fitted_nominal_feature(tmp_path):
    X, y = _frame()
    model = DarkoRegressor(**PARAMS).fit(
        X,
        y,
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    valid = tmp_path / "valid.npz"
    corrupt = tmp_path / "corrupt.npz"
    model.save_model(valid)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    header = json.loads(arrays["header"].item())
    record = header["wrapper"]["state"]["ordinal_features"][0]
    record["index"] = 2
    record["name"] = "region"
    arrays["header"] = np.asarray(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)
    with pytest.raises(ValueError, match="does not match fitted preprocessing"):
        DarkoRegressor.load_model(corrupt)


def test_failed_refit_does_not_corrupt_an_existing_ordinal_model():
    X, y = _frame()
    model = DarkoRegressor(**PARAMS).fit(
        X,
        y,
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    before = model.predict(X)
    with pytest.raises(ValueError, match="unknown ordinal category"):
        model.fit(
            X,
            y,
            cat_features=["region"],
            ordinal_features={"level": ["small", "large"]},
        )
    assert np.array_equal(model.predict(X), before)
    assert model.ordinal_features_[0]["categories"] == [
        "low",
        "medium",
        "high",
    ]


def test_refit_selection_model_retains_ordinal_metadata():
    X, y = _frame()
    model = DarkoRegressor(
        **{
            **PARAMS,
            "early_stopping": True,
            "early_stopping_rounds": 3,
            "refit": True,
        }
    ).fit(
        X,
        y,
        cat_features=["region"],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    assert model.model_.auto_params_["ordinal_features"]["active"] is True
    assert (
        model.selection_model_.auto_params_["ordinal_features"]["active"] is True
    )


@pytest.mark.parametrize(
    ("ordinal_features", "error"),
    [
        ({"level": ["low"]}, "at least two"),
        ({"level": ["low", "low"]}, "duplicate category"),
        ({"level": {"low", "high"}}, "ordered category sequence"),
        ({"level": frozenset({"low", "high"})}, "ordered category sequence"),
        ({"level": {"low": 0, "high": 1}}, "ordered category sequence"),
        ({"unknown": ["low", "high"]}, "unknown column name"),
        ("unsafe", "mapping, 'auto', or None"),
        (["level"], "mapping, 'auto', or None"),
    ],
)
def test_invalid_ordinal_declarations_fail_before_fit(ordinal_features, error):
    X, y = _frame()
    with pytest.raises((TypeError, ValueError), match=error):
        DarkoRegressor(**PARAMS).fit(
            X,
            y,
            cat_features=["region"],
            ordinal_features=ordinal_features,
        )


def test_sparse_ordinal_input_keeps_the_public_dense_input_error():
    X = sparse.csr_matrix(np.arange(24.0).reshape(12, 2))
    with pytest.raises(ValueError, match="sparse matrices are not supported"):
        DarkoRegressor(**PARAMS).fit(
            X,
            np.arange(12.0),
            ordinal_features={1: [0.0, 1.0]},
        )


def test_ordinal_archive_uses_no_pickle_payload(tmp_path):
    X, y = _frame()
    model = DarkoRegressor(**PARAMS).fit(
        X,
        y,
        ordinal_features={"level": ["low", "medium", "high"]},
        cat_features=["region"],
    )
    path = tmp_path / "ordinal.npz"
    model.save_model(path)
    with zipfile.ZipFile(path) as archive:
        assert all(name.endswith(".npy") for name in archive.namelist())
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)


def test_stepwise_search_propagates_ordinal_declaration_to_refit():
    X, y = _frame(120)
    search = DarkoStepwiseSearchCV(
        DarkoRegressor(**PARAMS),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=2,
        refit=True,
        random_state=4,
        resume=False,
    )
    search.fit(
        X,
        y,
        cat_features=[1, 2],
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    assert search.best_estimator_.ordinal_features_[0]["name"] == "level"
    assert (
        search.best_estimator_.model_.auto_params_["ordinal_features"]["active"] is True
    )
    assert np.isfinite(search.predict(X.iloc[:4])).all()


def test_stepwise_search_does_not_tune_nominal_only_params_for_ordinal_only_data():
    X, y = _frame(120)
    X = X[["x", "level"]]
    search = DarkoStepwiseSearchCV(
        DarkoRegressor(**PARAMS),
        phases=("binning_categorical",),
        tree_modes=("catboost",),
        n_trials={"binning_categorical": 1},
        cv=2,
        refit=False,
        random_state=4,
        resume=False,
    )
    search.fit(
        X,
        y,
        ordinal_features={"level": ["low", "medium", "high"]},
    )
    suggested = search.best_trial_.user_attrs["params_suggested"]
    assert "max_bins" in suggested
    assert "cat_smoothing" not in suggested


def test_stepwise_search_freezes_one_shot_explicit_category_iterables():
    X, y = _frame(120)
    search = DarkoStepwiseSearchCV(
        DarkoRegressor(**PARAMS),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=2,
        refit=True,
        random_state=4,
        resume=False,
    )
    search.fit(
        X,
        y,
        cat_features=[2],
        ordinal_features={
            "level": (value for value in ["low", "medium", "high"])
        },
    )
    assert search.ordinal_features_ == {
        1: ["low", "medium", "high"]
    }
    assert search.best_estimator_.ordinal_features_[0]["categories"] == [
        "low",
        "medium",
        "high",
    ]


def test_stepwise_search_freezes_auto_integer_vocabulary_across_folds():
    X = pd.DataFrame(
        {
            "x": np.arange(12.0),
            "level": [0] * 5 + [1] * 5 + [2] * 2,
        }
    )
    y = X["x"].to_numpy() + X["level"].to_numpy()
    splits = [
        (np.arange(10), np.arange(10, 12)),
        (np.r_[0:5, 10:12], np.arange(5, 10)),
    ]
    search = DarkoStepwiseSearchCV(
        DarkoRegressor(**PARAMS),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=splits,
        refit=True,
        random_state=4,
        resume=False,
        error_score="raise",
    )
    search.fit(
        X,
        y,
        cat_features=[1],
        ordinal_features="auto",
    )
    assert search.ordinal_features_ == "auto"
    assert search.best_estimator_.ordinal_features_[0]["categories"] == [
        0,
        1,
        2,
    ]
    assert (
        search.best_estimator_.ordinal_features_[0]["source"]
        == "auto_integer_codes"
    )


def test_stepwise_search_freezes_inactive_auto_resolution_across_folds():
    X = pd.DataFrame(
        {
            "x": np.arange(8.0),
            "level": [0, 1, 0, 1, 0, 1, 0, "nominal"],
        }
    )
    splits = [
        (np.arange(7), np.asarray([7])),
        (np.asarray([0, 1, 2, 3, 7]), np.asarray([4, 5, 6])),
    ]
    search = DarkoStepwiseSearchCV(
        DarkoRegressor(**PARAMS),
        phases=("probe",),
        tree_modes=("catboost",),
        n_trials={"probe": 1},
        cv=splits,
        refit=True,
        random_state=4,
        resume=False,
        error_score="raise",
    )
    search.fit(
        X,
        np.arange(8.0),
        cat_features=[1],
        ordinal_features="auto",
    )
    assert search.ordinal_features_ == "auto"
    assert search.best_estimator_.ordinal_features_mode_ == "auto"
    assert search.best_estimator_.ordinal_features_ == []
    assert search.best_estimator_.model_.prep_.cat_features_ == [1]
    fold_metadata = search.best_trial_.user_attrs["fold_auto_params"]
    assert all("ordinal_features" not in metadata for metadata in fold_metadata)
