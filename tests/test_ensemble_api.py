import io
import json

import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.sklearn_api import (
    _FrozenAutoOrdinalFeatures,
    _ensemble_bootstrap_plan,
)


def _regression_data(seed=41, n=220):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 1.4 * X[:, 0] - 0.6 * X[:, 1] + rng.normal(scale=0.2, size=n)
    return X, y


def _params(**extra):
    params = {
        "iterations": 24,
        "learning_rate": 0.1,
        "depth": 4,
        "early_stopping_rounds": 5,
        "random_state": 17,
    }
    params.update(extra)
    return params


def test_single_member_default_preserves_single_estimator_behavior():
    X, y = _regression_data()
    default = DarkoRegressor(**_params()).fit(X, y)
    explicit = DarkoRegressor(**_params(n_ensembles=1)).fit(X, y)

    assert not hasattr(default, "estimators_")
    assert not hasattr(explicit, "estimators_")
    np.testing.assert_array_equal(default.predict(X), explicit.predict(X))


def test_regression_ensemble_is_deterministic_mean_with_shared_preprocessing():
    X, y = _regression_data()
    left = DarkoRegressor(**_params(n_ensembles=3)).fit(X, y)
    right = DarkoRegressor(**_params(n_ensembles=3)).fit(X, y)

    expected = np.mean(
        np.stack([member.predict(X) for member in left.estimators_]),
        axis=0,
    )
    np.testing.assert_array_equal(left.predict(X), expected)
    np.testing.assert_array_equal(left.predict(X), right.predict(X))
    assert left.ensemble_metadata_ == right.ensemble_metadata_
    assert left.ensemble_metadata_["claim_tier"] == "E"
    assert left.ensemble_metadata_["default_changed"] is False
    assert (
        left.ensemble_metadata_["shared_preprocessing"]
        == "numeric_target_free"
    )
    borders = [
        member.model_.prep_.binner_._borders_flat_
        for member in left.estimators_
    ]
    for candidate in borders[1:]:
        np.testing.assert_array_equal(candidate, borders[0])
    for member in left.ensemble_metadata_["members"]:
        assert member["oob_rows"] > 0
        assert member["validation_source"] == "explicit_eval_set"


def test_regression_ensemble_shap_averages_and_remains_additive():
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=3)).fit(X, y)
    values = model.shap_values(X[:9])

    expected = np.mean(
        np.stack(
            [member.shap_values(X[:9]) for member in model.estimators_],
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        values.sum(axis=1) + model.expected_value_,
        model.predict(X[:9]),
        rtol=0.0,
        atol=2e-13,
    )


def test_group_bootstrap_is_group_disjoint_and_public_fit_records_it():
    X, y = _regression_data(n=240)
    groups = np.repeat(np.arange(40), 6)
    plan = _ensemble_bootstrap_plan(
        len(X),
        19,
        bootstrap="groups",
        groups=groups,
    )
    assert set(groups[plan["sampled"]]).isdisjoint(groups[plan["oob"]])

    model = DarkoRegressor(
        **_params(
            n_ensembles=3,
            ensemble_bootstrap="groups",
        )
    ).fit(X, y, groups=groups)
    assert model.ensemble_metadata_["bootstrap"] == "groups"
    assert all(
        member["group_disjoint"] is True
        and member["oob_groups"] > 0
        for member in model.ensemble_metadata_["members"]
    )


def test_group_bootstrap_forwards_groups_to_explicit_oob_member_fits():
    X, y = _regression_data(n=240)
    groups = np.repeat(np.arange(40), 6)
    model = DarkoRegressor(
        **_params(
            n_ensembles=2,
            ensemble_bootstrap="groups",
            validation_strategy="group",
        )
    ).fit(X, y, groups=groups)

    for member in model.estimators_:
        validation = member.model_.auto_params_["validation_split"]
        assert validation["source"] == "explicit_eval_set"
        assert validation["groups_provided"] is True


def test_row_bootstrap_rejects_groups_instead_of_silently_splitting_entities():
    X, y = _regression_data(n=120)
    groups = np.repeat(np.arange(20), 6)

    with pytest.raises(ValueError, match="ensemble_bootstrap='groups'"):
        DarkoRegressor(**_params(n_ensembles=2)).fit(
            X, y, groups=groups
        )


def test_categorical_ensemble_uses_member_local_target_statistics():
    pd = pytest.importorskip("pandas")
    X_num, y = _regression_data(n=180)
    X = pd.DataFrame({
        "value": X_num[:, 0],
        "kind": np.where(X_num[:, 1] > 0, "up", "down"),
    })
    model = DarkoRegressor(**_params(n_ensembles=3)).fit(
        X, y, cat_features=["kind"]
    )

    assert model.ensemble_metadata_["shared_preprocessing"] == "member_local"
    assert (
        model.ensemble_metadata_["shared_preprocessing_fallback_reason"]
        == "categorical_or_ordinal_features"
    )
    expected = np.mean(
        np.stack([member.predict(X) for member in model.estimators_]),
        axis=0,
    )
    np.testing.assert_array_equal(model.predict(X), expected)


@pytest.mark.parametrize(
    ("estimator", "target_kind"),
    [
        (DarkoRegressor, "regression"),
        (DarkoClassifier, "classification"),
    ],
)
def test_ensemble_accepts_explicit_string_ordinals(estimator, target_kind):
    pd = pytest.importorskip("pandas")
    X_num, y = _regression_data(n=180)
    grades = np.resize(np.asarray(["low", "mid", "high"]), len(y))
    X = pd.DataFrame({"grade": grades, "value": X_num[:, 0]})
    if target_kind == "classification":
        y = (grades == "high").astype(np.int64)

    model = estimator(**_params(n_ensembles=3)).fit(
        X,
        y,
        ordinal_features={"grade": ["low", "mid", "high"]},
    )

    assert model.ensemble_metadata_["shared_preprocessing"] == "member_local"
    assert all(
        member.ordinal_feature_indices_.tolist() == [0]
        for member in model.estimators_
    )
    assert model.predict(X[:4]).shape == (4,)


def test_ensemble_snapshots_one_shot_explicit_ordinal_categories():
    pd = pytest.importorskip("pandas")
    X_num, y = _regression_data(n=180)
    grades = np.resize(np.asarray(["low", "mid", "high"]), len(y))
    X = pd.DataFrame({"grade": grades, "value": X_num[:, 0]})

    model = DarkoRegressor(**_params(n_ensembles=2)).fit(
        X,
        y,
        ordinal_features={
            "grade": (value for value in ("low", "mid", "high"))
        },
    )

    assert all(
        member.ordinal_features_[0]["categories"] == ["low", "mid", "high"]
        for member in model.estimators_
    )


def test_ensemble_preserves_frozen_auto_ordinal_category_rules():
    pd = pytest.importorskip("pandas")
    X_num, y = _regression_data(n=180)
    X = pd.DataFrame({
        "grade": pd.Categorical(
            ["only"] * len(y),
            categories=["only"],
            ordered=True,
        ),
        "value": X_num[:, 0],
    })

    model = DarkoRegressor(**_params(n_ensembles=2)).fit(
        X,
        y,
        ordinal_features=_FrozenAutoOrdinalFeatures({0: ["only"]}),
    )

    assert all(
        member.ordinal_features_[0]["categories"] == ["only"]
        for member in model.estimators_
    )


def test_ensemble_treats_ordinal_false_as_off_for_shared_preprocessing():
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(
        X, y, ordinal_features=False
    )

    assert (
        model.ensemble_metadata_["shared_preprocessing"]
        == "numeric_target_free"
    )


def test_ensemble_invalid_ordinal_string_uses_public_validation_error():
    X, y = _regression_data()
    with pytest.raises(
        ValueError, match="ordinal_features must be a mapping, 'auto', or None"
    ):
        DarkoRegressor(**_params(n_ensembles=2)).fit(
            X, y, ordinal_features="invalid"
        )


@pytest.mark.parametrize("multiclass", [False, True])
def test_classifier_ensemble_soft_votes(multiclass):
    X, y_cont = _regression_data(n=260)
    if multiclass:
        y = np.digitize(y_cont, np.quantile(y_cont, [1 / 3, 2 / 3]))
    else:
        y = (y_cont > np.median(y_cont)).astype(np.int64)
    model = DarkoClassifier(**_params(n_ensembles=3)).fit(X, y)

    expected = np.mean(
        np.stack(
            [member.predict_proba(X) for member in model.estimators_],
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_array_equal(model.predict_proba(X), expected)
    np.testing.assert_array_equal(
        model.predict(X), model.classes_[np.argmax(expected, axis=1)]
    )


@pytest.mark.parametrize(
    ("estimator", "target"),
    [
        (DarkoRegressor, "regression"),
        (DarkoClassifier, "classification"),
    ],
)
def test_ensemble_safe_roundtrip(tmp_path, estimator, target):
    X, y = _regression_data()
    if target == "classification":
        y = (y > np.median(y)).astype(np.int64)
    model = estimator(**_params(n_ensembles=3)).fit(X, y)
    path = tmp_path / "ensemble.npz"
    model.save_model(path)

    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        assert header["archive_kind"] == "darkofit_ensemble"
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    restored = estimator.load_model(path)
    if target == "classification":
        np.testing.assert_array_equal(
            model.predict_proba(X), restored.predict_proba(X)
        )
    else:
        np.testing.assert_array_equal(model.predict(X), restored.predict(X))
    assert restored.ensemble_metadata_ == model.ensemble_metadata_
    assert len(restored.estimators_) == 3


def test_public_ensemble_archives_remain_format_one(tmp_path):
    X, y = _regression_data(n=100)
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(X, y)
    path = tmp_path / "ensemble.npz"
    model.save_model(path)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        assert header["ensemble_format_version"] == 1

    restored = DarkoRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_ensemble_preserves_arrow_feature_schema_and_roundtrips(tmp_path):
    pa = pytest.importorskip("pyarrow")
    X, y = _regression_data(n=120)
    table = pa.table({
        "a": X[:, 0],
        "b": X[:, 1],
        "c": X[:, 2],
        "d": X[:, 3],
        "e": X[:, 4],
    })
    model = DarkoRegressor(
        **_params(iterations=4, n_ensembles=2)
    ).fit(table, y)

    assert model.feature_names_in_.tolist() == ["a", "b", "c", "d", "e"]
    with pytest.raises(ValueError, match="feature names"):
        model.predict(
            pa.table({
                "b": X[:, 1],
                "a": X[:, 0],
                "c": X[:, 2],
                "d": X[:, 3],
                "e": X[:, 4],
            })
        )

    path = tmp_path / "arrow-schema-ensemble.npz"
    model.save_model(path)
    restored = DarkoRegressor.load_model(path)
    assert restored.feature_names_in_.tolist() == [
        "a", "b", "c", "d", "e"
    ]
    np.testing.assert_array_equal(restored.predict(table), model.predict(table))


def test_single_load_rejects_reordered_ordinal_category_provenance(tmp_path):
    pd = pytest.importorskip("pandas")
    X_num, y = _regression_data(n=180)
    grades = np.resize(np.asarray(["low", "mid", "high"]), len(y))
    X = pd.DataFrame({"grade": grades, "value": X_num[:, 0]})
    model = DarkoRegressor(**_params(iterations=4)).fit(
        X,
        y,
        ordinal_features={"grade": ["low", "mid", "high"]},
    )
    source = tmp_path / "ordinal-source.npz"
    corrupt = tmp_path / "ordinal-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["state"]["ordinal_features"][0]["categories"] = [
        "high",
        "mid",
        "low",
    ]
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="ordinal wrapper state"):
        DarkoRegressor.load_model(corrupt)


def test_single_load_rejects_injected_preset_provenance(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(iterations=2)).fit(X, y)
    source = tmp_path / "preset-source.npz"
    corrupt = tmp_path / "preset-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["params"]["preset"] = "accuracy"
    header["wrapper"]["state"]["preset"] = "accuracy"
    header["wrapper"]["state"]["preset_params"] = {
        "iterations": 10_000,
        "tree_mode": "auto",
        "l2_leaf_reg": 3.0,
        "max_bins": 128,
        "learning_rate": 0.1,
        "ts_permutations": 1,
        "linear_residual": False,
        "early_stopping": True,
        "use_best_model": True,
    }
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="preset wrapper and booster"):
        DarkoRegressor.load_model(corrupt)


def test_single_load_rejects_tree_selection_wrapper_forgery(tmp_path):
    X, y = _regression_data(n=120)
    model = DarkoRegressor(
        **_params(iterations=1, tree_mode="auto")
    ).fit(X, y)
    source = tmp_path / "selection-source.npz"
    corrupt = tmp_path / "selection-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["state"]["tree_mode_selection"][
        "selected_score"
    ] += 1.0
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="selection wrapper and booster"):
        DarkoRegressor.load_model(corrupt)


def test_single_load_rejects_injected_refit_provenance(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(iterations=2)).fit(X, y)
    source = tmp_path / "refit-source.npz"
    corrupt = tmp_path / "refit-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    state = header["wrapper"]["state"]
    state["refit"] = True
    state["refit_n_estimators"] = len(model.model_.trees_)
    state["refit_strategy"] = "exact"
    state["selection_model_persisted"] = False
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="refit wrapper and booster"):
        DarkoRegressor.load_model(corrupt)


def test_ensemble_load_rejects_contradictory_provenance(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(X, y)
    source = tmp_path / "source.npz"
    corrupt = tmp_path / "corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["metadata"]["aggregation"] = "soft_vote"
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="aggregation"):
        DarkoRegressor.load_model(corrupt)


@pytest.mark.parametrize(
    ("estimator", "target_kind", "corrupt_loss"),
    [
        (DarkoClassifier, "classification", "RMSE"),
        (DarkoRegressor, "regression", "Logloss"),
    ],
)
def test_ensemble_load_rejects_incompatible_member_model_families(
    tmp_path, estimator, target_kind, corrupt_loss
):
    X, y = _regression_data()
    if target_kind == "classification":
        y = (y > np.median(y)).astype(np.int64)
    model = estimator(**_params(n_ensembles=2)).fit(X, y)
    source = tmp_path / f"{target_kind}-family-source.npz"
    corrupt = tmp_path / f"{target_kind}-family-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    outer_header = json.loads(str(arrays["header"]))
    if target_kind == "regression":
        outer_header["params"]["loss"] = corrupt_loss
        arrays["header"] = np.array(json.dumps(outer_header))
    for name in ("member_0000", "member_0001"):
        with np.load(
            io.BytesIO(arrays[name].tobytes()), allow_pickle=False
        ) as member_archive:
            member_arrays = {
                key: member_archive[key].copy()
                for key in member_archive.files
            }
        member_header = json.loads(str(member_arrays["header"]))
        member_header["loss_name"] = corrupt_loss
        member_arrays["header"] = np.array(json.dumps(member_header))
        payload = io.BytesIO()
        np.savez_compressed(payload, **member_arrays)
        arrays[name] = np.frombuffer(
            payload.getvalue(), dtype=np.uint8
        ).copy()
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="family"):
        estimator.load_model(corrupt)


@pytest.mark.parametrize(
    ("estimator", "target_kind", "corrupt_loss"),
    [
        (DarkoClassifier, "classification", "RMSE"),
        (DarkoRegressor, "regression", "Logloss"),
    ],
)
def test_single_load_rejects_incompatible_booster_family(
    tmp_path, estimator, target_kind, corrupt_loss
):
    X, y = _regression_data()
    if target_kind == "classification":
        y = (y > np.median(y)).astype(np.int64)
    model = estimator(**_params(iterations=4)).fit(X, y)
    source = tmp_path / f"single-{target_kind}-source.npz"
    corrupt = tmp_path / f"single-{target_kind}-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["loss_name"] = corrupt_loss
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="booster family"):
        estimator.load_model(corrupt)


def test_ensemble_load_rejects_member_selection_param_mismatch(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(X, y)
    source = tmp_path / "member-selection-source.npz"
    corrupt = tmp_path / "member-selection-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    with np.load(
        io.BytesIO(arrays["member_0000"].tobytes()), allow_pickle=False
    ) as member_archive:
        member_arrays = {
            key: member_archive[key].copy()
            for key in member_archive.files
        }
    member_header = json.loads(str(member_arrays["header"]))
    member_header["wrapper"]["params"]["refit"] = True
    member_arrays["header"] = np.array(json.dumps(member_header))
    payload = io.BytesIO()
    np.savez_compressed(payload, **member_arrays)
    arrays["member_0000"] = np.frombuffer(
        payload.getvalue(), dtype=np.uint8
    ).copy()
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="selection params"):
        DarkoRegressor.load_model(corrupt)


def test_ensemble_load_rejects_nested_member_before_recursive_decode(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(X, y)
    source = tmp_path / "ensemble-source.npz"
    corrupt = tmp_path / "nested-member-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["member_0000"] = np.frombuffer(
        source.read_bytes(), dtype=np.uint8
    ).copy()
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="nested ensemble member"):
        DarkoRegressor.load_model(corrupt)


def test_single_quantile_save_freezes_fitted_loss_alpha_and_random_state(
    tmp_path,
):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(iterations=4, loss="Quantile", alpha=0.8)
    ).fit(X, y)
    expected = model.predict(X)
    model.set_params(loss="MAE", alpha=0.2, random_state=None)
    path = tmp_path / "single-quantile-frozen.npz"
    model.save_model(path)

    restored = DarkoRegressor.load_model(path)

    assert restored.loss == "Quantile"
    assert restored.alpha == 0.8
    assert restored.random_state == 17
    np.testing.assert_array_equal(restored.predict(X), expected)


def test_single_load_rejects_self_consistent_negative_random_state(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(iterations=4)).fit(X, y)
    source = tmp_path / "single-negative-seed-source.npz"
    corrupt = tmp_path / "single-negative-seed-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["params"]["random_state"] = -1
    header["wrapper"]["params"]["random_state"] = -1
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="random state"):
        DarkoRegressor.load_model(corrupt)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("loss", "MAE", "wrapper loss"),
        ("alpha", 0.2, "quantile alpha"),
        ("random_state", True, "random state"),
        ("random_state", 999, "random state"),
        ("linear_residual", True, "linear residual parameter"),
        ("linear_residual_alpha", 2.0, "linear_residual_alpha"),
        (
            "linear_residual_fit_intercept",
            False,
            "linear_residual_fit_intercept",
        ),
    ],
)
def test_single_load_rejects_wrapper_fitted_param_mismatches(
    tmp_path, key, value, message
):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(iterations=4, loss="Quantile", alpha=0.8)
    ).fit(X, y)
    source = tmp_path / f"single-{key}-source.npz"
    corrupt = tmp_path / f"single-{key}-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["params"][key] = value
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match=message):
        DarkoRegressor.load_model(corrupt)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("best_n_estimators", 999, "fitted estimator count"),
        ("best_n_estimators", None, "fitted estimator count"),
        ("learning_rate", 999.0, "fitted learning rate"),
        ("learning_rate", 10**1000, "fitted learning rate"),
        ("best_score", 999.0, "fitted score"),
        ("n_features_in", 999, "input feature count"),
        ("feature_names_in", ["only_one"], "feature name"),
    ],
)
def test_single_load_rejects_wrapper_fitted_state_mismatches(
    tmp_path, key, value, message
):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(iterations=4)).fit(X, y)
    source = tmp_path / f"single-state-{key}-source.npz"
    corrupt = tmp_path / f"single-state-{key}-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["state"][key] = value
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match=message):
        DarkoRegressor.load_model(corrupt)


def test_single_load_rejects_refit_selection_count_forgery(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(
            iterations=12,
            early_stopping=True,
            validation_fraction=0.2,
            refit=True,
            refit_strategy="sqrt",
        )
    ).fit(X, y)
    source = tmp_path / "refit-selection-count-source.npz"
    corrupt = tmp_path / "refit-selection-count-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["state"]["best_n_estimators"] = 0
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="selection and refit"):
        DarkoRegressor.load_model(corrupt)


def test_single_load_normalizes_missing_scaled_refit_counts(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(
            iterations=4,
            early_stopping=True,
            refit=True,
            refit_strategy="exact",
        )
    ).fit(X[:180], y[:180], eval_set=(X[180:], y[180:]))
    source = tmp_path / "explicit-refit-source.npz"
    corrupt = tmp_path / "explicit-refit-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["state"]["refit_strategy"] = "sqrt"
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="selection sample counts"):
        DarkoRegressor.load_model(corrupt)


@pytest.mark.parametrize("refit_strategy", ["exact", "sqrt", "linear"])
def test_single_load_normalizes_oversized_refit_counts(
    tmp_path, refit_strategy
):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(
            iterations=4,
            early_stopping=True,
            refit=True,
            refit_strategy=refit_strategy,
        )
    ).fit(X, y)
    source = tmp_path / f"{refit_strategy}-refit-source.npz"
    corrupt = tmp_path / f"{refit_strategy}-refit-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    oversized_count = 10**1000
    header["wrapper"]["state"]["selection_n_total"] = oversized_count
    auto_params = header["auto_params"]
    auto_params["selection_validation_split"][
        "original_n_samples"
    ] = oversized_count
    auto_params["diagnostics"]["selection_validation_split"][
        "original_n_samples"
    ] = oversized_count
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="selection and refit"):
        DarkoRegressor.load_model(corrupt)


def test_single_linear_residual_save_freezes_fitted_params(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(
            iterations=4,
            linear_residual=True,
            linear_residual_alpha=0.3,
        )
    )
    with pytest.warns(FutureWarning, match="linear_residual"):
        model.fit(X, y)
    expected = model.predict(X)
    model.set_params(
        linear_residual=False,
        linear_residual_alpha=9.0,
        linear_residual_fit_intercept=False,
        linear_residual_standardize=False,
    )
    path = tmp_path / "single-linear-residual-frozen.npz"
    model.save_model(path)

    restored = DarkoRegressor.load_model(path)

    assert restored.linear_residual is True
    assert restored.linear_residual_alpha == 0.3
    assert restored.linear_residual_fit_intercept is True
    assert restored.linear_residual_standardize is True
    assert restored.linear_residual_enabled_ is True
    np.testing.assert_array_equal(restored.predict(X), expected)


def test_single_linear_residual_selector_survives_repeated_roundtrip(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(
            iterations=4,
            linear_residual=True,
            linear_residual_features=[0, 1],
        )
    )
    with pytest.warns(FutureWarning, match="linear_residual"):
        model.fit(X, y)
    expected = model.predict(X)
    first_path = tmp_path / "single-linear-selector-first.npz"
    second_path = tmp_path / "single-linear-selector-second.npz"
    model.save_model(first_path)

    first = DarkoRegressor.load_model(first_path)
    first.save_model(second_path)
    second = DarkoRegressor.load_model(second_path)

    assert first.linear_residual_features == [0, 1]
    assert second.linear_residual_features == [0, 1]
    np.testing.assert_array_equal(second.predict(X), expected)


@pytest.mark.parametrize(
    ("field", "value", "removed_param"),
    [
        ("linear_residual_enabled", 0, None),
        (
            "linear_residual_fit_intercept",
            1,
            "linear_residual_fit_intercept",
        ),
        (
            "linear_residual_standardize",
            1,
            "linear_residual_standardize",
        ),
        ("linear_residual_alpha", "1.0", "linear_residual_alpha"),
        ("linear_residual_version", "1", None),
    ],
)
def test_single_load_rejects_untyped_linear_residual_state(
    tmp_path, field, value, removed_param
):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(iterations=4)).fit(X, y)
    source = tmp_path / f"single-{field}-source.npz"
    corrupt = tmp_path / f"single-{field}-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    if removed_param is not None:
        header["wrapper"]["params"].pop(removed_param)
    header["wrapper"]["state"][field] = value
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="linear residual|linear_residual"):
        DarkoRegressor.load_model(corrupt)


def test_single_load_rejects_self_consistent_linear_residual_forgery(
    tmp_path,
):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(
            iterations=4,
            linear_residual=True,
            linear_residual_alpha=0.3,
        )
    )
    with pytest.warns(FutureWarning, match="linear_residual"):
        model.fit(X, y)
    source = tmp_path / "single-linear-provenance-source.npz"
    corrupt = tmp_path / "single-linear-provenance-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["params"]["linear_residual_alpha"] = 0.4
    header["wrapper"]["state"]["linear_residual_alpha"] = 0.4
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="wrapper and booster provenance"):
        DarkoRegressor.load_model(corrupt)


def test_single_load_rejects_stripped_linear_residual_state(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(iterations=4, linear_residual=True)
    )
    with pytest.warns(FutureWarning, match="linear_residual"):
        model.fit(X, y)
    source = tmp_path / "single-linear-strip-source.npz"
    corrupt = tmp_path / "single-linear-strip-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["params"]["linear_residual"] = False
    state = header["wrapper"]["state"]
    for name in list(state):
        if name.startswith("linear_residual"):
            state.pop(name)
    for name in list(arrays):
        if name.startswith("wrapper__linear_residual"):
            arrays.pop(name)
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="booster linear residual provenance"):
        DarkoRegressor.load_model(corrupt)


def test_classifier_load_rejects_wrapper_labels_that_disagree_with_booster(
    tmp_path,
):
    X, y_cont = _regression_data(n=260)
    y = np.digitize(y_cont, np.quantile(y_cont, [1 / 3, 2 / 3]))
    model = DarkoClassifier(**_params(iterations=4)).fit(X, y)
    source = tmp_path / "classifier-labels-source.npz"
    corrupt = tmp_path / "classifier-labels-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["wrapper__classes"] = arrays["wrapper__classes"][::-1].copy()
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="class labels"):
        DarkoClassifier.load_model(corrupt)


@pytest.mark.parametrize(
    "corrupt_classes",
    [
        pytest.param([np.nan, np.nan], id="duplicate-nan"),
        pytest.param([0.0, np.inf], id="infinite"),
        pytest.param([1, 1], id="duplicate"),
    ],
)
def test_classifier_load_rejects_invalid_binary_class_labels(
    tmp_path, corrupt_classes
):
    X, y_cont = _regression_data()
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    model = DarkoClassifier(**_params(iterations=4)).fit(X, y)
    source = tmp_path / "classifier-binary-labels-source.npz"
    corrupt = tmp_path / "classifier-binary-labels-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["wrapper__classes"] = np.asarray(corrupt_classes)
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="class labels"):
        DarkoClassifier.load_model(corrupt)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("member", "best_iteration", 999, "fitted metadata"),
        ("member", "learning_rate", 999.0, "fitted metadata"),
        ("member", "stop_reason", "fabricated", "fitted metadata"),
        ("member", "bootstrap_rows", 999, "sampling metadata"),
        ("member", "bootstrap_unique_rows", 1, "sampling metadata"),
        ("member", "bootstrap_attempts", 129, "attempt count"),
        ("member", "member", 0.0, "member provenance"),
        ("member", "seed", "as_float", "member provenance"),
        ("metadata", "version", 1.0, "ensemble provenance"),
        ("metadata", "member_count", 2.0, "member count"),
        ("metadata", "input_feature_count", 999, "input feature count"),
        ("params", "ensemble_bootstrap", "groups", "params"),
        ("params", "random_state", 17.0, "random state"),
        ("params", "random_state", 999, "random state"),
        ("params", "loss", "MAE", "ensemble loss"),
        ("params", "oblivious_kernel", "unfused", "oblivious_kernel"),
    ],
)
def test_ensemble_load_rejects_payload_metadata_mismatches(
    tmp_path, section, key, value, message
):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(X, y)
    source = tmp_path / f"{section}-{key}-source.npz"
    corrupt = tmp_path / f"{section}-{key}-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    if section == "member":
        if key == "seed" and value == "as_float":
            value = float(header["metadata"]["members"][0]["seed"])
        header["metadata"]["members"][0][key] = value
    else:
        header[section][key] = value
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match=message):
        DarkoRegressor.load_model(corrupt)


def test_ensemble_save_freezes_fitted_ensemble_params(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=3)).fit(X, y)
    expected = model.predict(X)
    model.set_params(
        n_ensembles=1,
        ensemble_bootstrap="groups",
        ensemble_shared_preprocessing=False,
        random_state=None,
        loss="MAE",
        refit=True,
        refit_strategy="linear",
        early_stopping=False,
        use_best_model=False,
        preset="accuracy",
        selection_rounds=2,
        tree_mode="auto",
        interval_calibration="conformal",
        dist_calibration="affine",
        sigma_calibration="scalar",
        oblivious_kernel="unfused",
        linear_residual=True,
        linear_residual_alpha=2.0,
        linear_residual_fit_intercept=False,
        linear_residual_standardize=False,
    )

    path = tmp_path / "frozen-ensemble-params.npz"
    model.save_model(path)
    restored = DarkoRegressor.load_model(path)

    assert restored.n_ensembles == 3
    assert restored.ensemble_bootstrap == "rows"
    assert restored.ensemble_shared_preprocessing is True
    assert restored.random_state == 17
    assert restored.loss == "RMSE"
    assert restored.refit is False
    assert restored.refit_strategy == "exact"
    assert restored.early_stopping is True
    assert restored.use_best_model is True
    assert restored.preset is None
    assert restored.selection_rounds is None
    assert restored.tree_mode == "catboost"
    assert restored.interval_calibration is None
    assert restored.dist_calibration is None
    assert restored.sigma_calibration is None
    assert restored.oblivious_kernel == "auto"
    assert all(
        member.oblivious_kernel == member.model_.oblivious_kernel == "auto"
        for member in restored.estimators_
    )
    assert restored.linear_residual is False
    assert restored.linear_residual_alpha == 1.0
    assert restored.linear_residual_fit_intercept is True
    assert restored.linear_residual_standardize is True
    np.testing.assert_array_equal(restored.predict(X), expected)


def test_quantile_ensemble_load_rejects_untyped_outer_alpha(tmp_path):
    X, y = _regression_data()
    model = DarkoRegressor(
        **_params(
            iterations=4,
            n_ensembles=2,
            loss="Quantile",
            alpha=0.8,
        )
    ).fit(X, y)
    source = tmp_path / "quantile-outer-alpha-source.npz"
    corrupt = tmp_path / "quantile-outer-alpha-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["params"]["alpha"] = "0.8"
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="ensemble quantile"):
        DarkoRegressor.load_model(corrupt)


def test_ensemble_load_rejects_shared_preprocessing_schema_contradiction(
    tmp_path,
):
    pd = pytest.importorskip("pandas")
    X_num, y = _regression_data(n=180)
    X = pd.DataFrame({
        "value": X_num[:, 0],
        "kind": np.where(X_num[:, 1] > 0, "up", "down"),
    })
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(
        X, y, cat_features=["kind"]
    )
    source = tmp_path / "categorical-member-local-source.npz"
    corrupt = tmp_path / "categorical-forged-shared.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["metadata"]["shared_preprocessing"] = "numeric_target_free"
    header["metadata"]["shared_preprocessing_fallback_reason"] = None
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="fitted feature schema"):
        DarkoRegressor.load_model(corrupt)


def test_ensemble_refit_clears_stale_single_model_state():
    pd = pytest.importorskip("pandas")
    X, y = _regression_data()
    frame = pd.DataFrame(X, columns=[f"x{index}" for index in range(X.shape[1])])
    model = DarkoRegressor(
        **_params(linear_residual=True)
    )
    with pytest.warns(FutureWarning, match="linear_residual"):
        model.fit(frame, y)
    assert model.linear_residual_active_ is True

    model.set_params(n_ensembles=2, linear_residual=False)
    model.fit(X, y)

    assert not hasattr(model, "feature_names_in_")
    assert not hasattr(model, "linear_residual_active_")
    assert model.shap_values(X[:3]).shape == (3, X.shape[1])


def test_failed_refit_preserves_the_existing_ensemble_transactionally():
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(X, y)
    expected = model.predict(X)
    expected_estimators = model.estimators_
    model.set_params(n_ensembles=1, linear_leaves="invalid")

    with pytest.raises(TypeError, match="linear_leaves must be a bool"):
        model.fit(X, y)

    assert model.estimators_ is expected_estimators
    np.testing.assert_array_equal(model.predict(X), expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_ensembles": 0}, "at least 1"),
        ({"n_ensembles": 257}, "at most 256"),
        ({"n_ensembles": True}, "positive integer"),
        (
            {"n_ensembles": 2, "ensemble_bootstrap": "groups"},
            "requires groups",
        ),
        (
            {"n_ensembles": 2, "ensemble_bootstrap": "unknown"},
            "must be 'rows' or 'groups'",
        ),
    ],
)
def test_ensemble_parameter_validation(kwargs, message):
    X, y = _regression_data()
    with pytest.raises((TypeError, ValueError), match=message):
        DarkoRegressor(**_params(**kwargs)).fit(X, y)


def test_ensemble_rejects_ambiguous_eval_and_distributional_aggregation():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="out-of-bag"):
        DarkoRegressor(**_params(n_ensembles=2)).fit(
            X, y, eval_set=(X[:20], y[:20])
        )
    with pytest.raises(ValueError, match="scalar regression"):
        DarkoRegressor(
            **_params(
                n_ensembles=2,
                loss="Gaussian",
                tree_mode="lightgbm",
            )
        ).fit(X, y)
    with pytest.raises(ValueError, match="refit=True"):
        DarkoRegressor(**_params(n_ensembles=2, refit=True)).fit(X, y)


def test_empty_callback_collection_remains_a_noop_for_ensemble():
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(
        X, y, callbacks=()
    )
    assert len(model.estimators_) == 2


@pytest.mark.parametrize(
    "callbacks",
    [
        pytest.param((callback for callback in ()), id="generator"),
        pytest.param(np.empty(0, dtype=object), id="numpy-array"),
    ],
)
def test_empty_callback_iterables_remain_noops_for_ensemble(callbacks):
    X, y = _regression_data()
    model = DarkoRegressor(**_params(n_ensembles=2)).fit(
        X, y, callbacks=callbacks
    )
    assert len(model.estimators_) == 2
