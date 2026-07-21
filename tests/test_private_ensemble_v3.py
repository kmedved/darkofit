import json

import numba
import numpy as np
import pytest

from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.sklearn_api import (
    _ensemble_bootstrap_plan,
    _ensemble_without_replacement_plan,
    _fit_private_ensemble_v3,
    _resolve_private_ensemble_v3_policy,
)


def _regression_data(seed=20260720, n=160):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 1.5 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.2, size=n)
    return X, y


def _params(**extra):
    params = {
        "iterations": 8,
        "depth": 3,
        "early_stopping_rounds": 3,
        "random_state": 17,
        "n_ensembles": 2,
        "diagnostic_warnings": "never",
    }
    params.update(extra)
    return params


def _fit_private(estimator, X, y, **kwargs):
    params = {
        "sampling": "without_replacement",
        "sampling_unit": "rows",
        "sample_fraction": 0.8,
        "member_policy": "donor_balanced_v1",
    }
    params.update(kwargs)
    return _fit_private_ensemble_v3(estimator, X, y, **params)


def test_private_row_plan_is_deterministic_unique_and_exact_complement():
    left = _ensemble_without_replacement_plan(
        20,
        91,
        sampling_unit="rows",
        sample_fraction=0.8,
    )
    right = _ensemble_without_replacement_plan(
        20,
        91,
        sampling_unit="rows",
        sample_fraction=0.8,
    )

    np.testing.assert_array_equal(left["sampled"], right["sampled"])
    np.testing.assert_array_equal(left["oob"], right["oob"])
    assert len(left["sampled"]) == 16
    assert len(np.unique(left["sampled"])) == 16
    assert set(left["sampled"]).isdisjoint(left["oob"])
    assert sorted(np.r_[left["sampled"], left["oob"]]) == list(range(20))


def test_private_group_plan_is_without_replacement_and_group_disjoint():
    groups = np.repeat(np.arange(10), np.arange(2, 12))
    plan = _ensemble_without_replacement_plan(
        len(groups),
        73,
        sampling_unit="groups",
        sample_fraction=0.8,
        groups=groups,
    )

    sampled_groups = groups[plan["sampled"]]
    oob_groups = groups[plan["oob"]]
    assert len(np.unique(sampled_groups)) == 8
    assert plan["sampled_group_draws"] == 8
    assert plan["sampled_unique_groups"] == 8
    assert plan["oob_groups"] == 2
    assert set(sampled_groups).isdisjoint(oob_groups)
    assert len(np.unique(plan["sampled"])) == len(plan["sampled"])


def test_private_plan_retries_for_class_safety_and_fails_when_impossible():
    y = np.resize(np.array([0, 1]), 40)
    plan = _ensemble_without_replacement_plan(
        len(y),
        12,
        sampling_unit="rows",
        sample_fraction=0.8,
        y=y,
        required_class_count=2,
    )
    assert np.unique(y[plan["sampled"]]).tolist() == [0, 1]
    assert np.unique(y[plan["oob"]]).tolist() == [0, 1]

    impossible = np.zeros(40, dtype=np.int64)
    impossible[0] = 1
    with pytest.raises(RuntimeError, match="class-safe"):
        _ensemble_without_replacement_plan(
            len(impossible),
            12,
            sampling_unit="rows",
            sample_fraction=0.8,
            y=impossible,
            required_class_count=2,
            max_attempts=8,
        )


@pytest.mark.parametrize("bootstrap", ["rows", "groups"])
def test_bootstrap_plan_requires_every_class_in_training_and_oob(bootstrap):
    impossible = np.zeros(40, dtype=np.int64)
    impossible[0] = 1
    groups = None
    if bootstrap == "groups":
        groups = np.repeat(np.arange(8), 5)
        impossible[:5] = 1

    with pytest.raises(RuntimeError, match="class-safe"):
        _ensemble_bootstrap_plan(
            len(impossible),
            12,
            bootstrap=bootstrap,
            groups=groups,
            y=impossible,
            required_class_count=2,
            max_attempts=8,
        )


@pytest.mark.parametrize("sampling", ["bootstrap", "without_replacement"])
def test_ensemble_plans_require_positive_weight_for_each_class_on_both_sides(
    sampling,
):
    y = np.resize(np.array([0, 1]), 40)
    weights = np.zeros(len(y))
    weights[:2] = 1.0
    kwargs = {
        "n_rows": len(y),
        "seed": 12,
        "y": y,
        "required_class_count": 2,
        "sample_weight": weights,
        "max_attempts": 8,
    }
    if sampling == "bootstrap":
        call = _ensemble_bootstrap_plan
        kwargs["bootstrap"] = "rows"
    else:
        call = _ensemble_without_replacement_plan
        kwargs.update(sampling_unit="rows", sample_fraction=0.8)

    with pytest.raises(RuntimeError, match="class-safe"):
        call(**kwargs)


def test_private_plan_rejects_nonpositive_weight_partition():
    weights = np.zeros(20)
    weights[0] = 1.0
    with pytest.raises(RuntimeError, match="usable"):
        _ensemble_without_replacement_plan(
            len(weights),
            5,
            sampling_unit="rows",
            sample_fraction=0.8,
            sample_weight=weights,
            max_attempts=8,
        )


def test_private_member_policy_changes_only_declared_fields_and_honors_explicit():
    estimator = DarkoRegressor(**_params(learning_rate=None, colsample=1.0))
    policy, explicit, resolutions, member_params = (
        _resolve_private_ensemble_v3_policy(
            estimator,
            "donor_balanced_v1",
            (),
        )
    )
    assert policy == "donor_balanced_v1"
    assert explicit == ()
    assert member_params == {"learning_rate": 0.15, "colsample": 0.85}
    assert set(resolutions) == {"learning_rate", "colsample"}
    assert {record["source"] for record in resolutions.values()} == {
        "member_policy"
    }

    _, explicit, resolutions, member_params = (
        _resolve_private_ensemble_v3_policy(
            estimator,
            "donor_balanced_v1",
            ("colsample", "learning_rate"),
        )
    )
    assert explicit == ("learning_rate", "colsample")
    assert member_params == {"learning_rate": None, "colsample": 1.0}
    assert {record["source"] for record in resolutions.values()} == {
        "explicit_user"
    }


def test_private_combined_regression_is_deterministic_mean_and_records_contract():
    X, y = _regression_data()
    left = _fit_private(DarkoRegressor(**_params()), X, y)
    right = _fit_private(DarkoRegressor(**_params()), X, y)

    expected = np.mean(
        np.stack([member.predict(X) for member in left.estimators_]),
        axis=0,
    )
    np.testing.assert_array_equal(left.predict(X), expected)
    np.testing.assert_array_equal(left.predict(X), right.predict(X))
    expected_shap = np.mean(
        np.stack([member.shap_values(X[:8]) for member in left.estimators_]),
        axis=0,
    )
    np.testing.assert_array_equal(left.shap_values(X[:8]), expected_shap)
    assert left.ensemble_metadata_ == right.ensemble_metadata_
    metadata = left.ensemble_metadata_
    assert metadata["version"] == 2
    assert metadata["sampling"] == "without_replacement"
    assert metadata["sample_fraction"] == 0.8
    assert metadata["member_policy"] == "donor_balanced_v1"
    assert metadata["sequential"] is True
    assert metadata["public_fit_surface"] is False
    for record, member in zip(metadata["members"], left.estimators_):
        assert record["sampled_rows"] == record["sampled_unique_rows"] == 128
        assert record["oob_rows"] == 32
        assert record["fitted_thread_count"] == member.model_.n_threads_
        assert record["constructor_learning_rate"] == 0.15
        assert record["constructor_colsample"] == 0.85


def test_private_policy_only_uses_existing_bootstrap_sampling():
    X, y = _regression_data(n=120)
    model = _fit_private_ensemble_v3(
        DarkoRegressor(**_params()),
        X,
        y,
        sampling="bootstrap",
        sampling_unit="rows",
        member_policy="donor_balanced_v1",
    )

    assert model.ensemble_metadata_["sampling"] == "bootstrap"
    assert model.ensemble_metadata_["sample_fraction"] is None
    assert all(
        record["sampled_rows"] == len(X)
        and record["sampled_unique_rows"] < len(X)
        for record in model.ensemble_metadata_["members"]
    )


def test_private_group_bootstrap_with_uneven_groups_survives_safe_roundtrip(
    tmp_path,
):
    group_sizes = np.arange(2, 12)
    groups = np.repeat(np.arange(len(group_sizes)), group_sizes)
    X, y = _regression_data(n=len(groups))
    model = _fit_private_ensemble_v3(
        DarkoRegressor(**_params(n_ensembles=4, iterations=4)),
        X,
        y,
        sampling="bootstrap",
        sampling_unit="groups",
        member_policy="none",
        groups=groups,
    )
    assert any(
        record["sampled_rows"] != len(X)
        for record in model.ensemble_metadata_["members"]
    )

    path = tmp_path / "private-group-bootstrap.npz"
    model.save_model(path)
    restored = DarkoRegressor.load_model(path)

    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.ensemble_metadata_ == model.ensemble_metadata_


def test_private_explicit_none_and_normal_default_survive_safe_roundtrip(tmp_path):
    X, y = _regression_data(n=100)
    model = _fit_private(
        DarkoRegressor(
            **_params(iterations=4, learning_rate=None, colsample=1.0)
        ),
        X,
        y,
        explicit_user_params=("learning_rate", "colsample"),
    )
    resolutions = model.ensemble_metadata_["policy_resolutions"]
    assert resolutions["learning_rate"] == {
        "base": None,
        "resolved": None,
        "source": "explicit_user",
    }
    assert resolutions["colsample"] == {
        "base": 1.0,
        "resolved": 1.0,
        "source": "explicit_user",
    }
    assert all(member.learning_rate is None for member in model.estimators_)
    assert all(member.colsample == 1.0 for member in model.estimators_)

    path = tmp_path / "private-explicit-defaults.npz"
    model.save_model(path)
    restored = DarkoRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.ensemble_metadata_ == model.ensemble_metadata_
    assert all(member.learning_rate is None for member in restored.estimators_)


def test_private_group_sampling_aligns_frame_weights_and_groups():
    pd = pytest.importorskip("pandas")
    X_num, y = _regression_data(n=180)
    X = pd.DataFrame({
        "x0": X_num[:, 0],
        "kind": np.where(X_num[:, 1] > 0, "up", "down"),
    })
    groups = np.repeat(np.arange(30), 6)
    weights = np.linspace(0.2, 2.0, len(y))
    model = _fit_private(
        DarkoRegressor(**_params()),
        X,
        y,
        sampling_unit="groups",
        groups=groups,
        sample_weight=weights,
        cat_features=["kind"],
    )

    assert model.ensemble_metadata_["sampling_unit"] == "groups"
    assert model.ensemble_metadata_["shared_preprocessing"] == "member_local"
    assert all(
        record["group_disjoint"] is True
        and record["sampled_unique_groups"] == 24
        and record["oob_groups"] == 6
        for record in model.ensemble_metadata_["members"]
    )
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize("backend", ["pyarrow", "polars"])
def test_private_row_sampling_preserves_optional_frame_schema(backend):
    X_num, y = _regression_data(n=100)
    columns = {f"x{index}": X_num[:, index] for index in range(X_num.shape[1])}
    if backend == "pyarrow":
        module = pytest.importorskip("pyarrow")
        X = module.table(columns)
    else:
        module = pytest.importorskip("polars")
        X = module.DataFrame(columns)

    model = _fit_private(
        DarkoRegressor(**_params(iterations=3)),
        X,
        y,
        sample_weight=np.linspace(0.5, 1.5, len(y)),
    )
    assert model.feature_names_in_.tolist() == list(columns)
    assert np.isfinite(model.predict(X)).all()


def test_private_classifier_soft_votes_and_safe_roundtrips(tmp_path):
    X, continuous = _regression_data(n=180)
    y = np.digitize(continuous, np.quantile(continuous, [1 / 3, 2 / 3]))
    model = _fit_private(DarkoClassifier(**_params()), X, y)
    expected = np.mean(
        np.stack(
            [member.predict_proba(X) for member in model.estimators_],
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_array_equal(model.predict_proba(X), expected)

    path = tmp_path / "private-classifier.npz"
    model.save_model(path)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        assert header["ensemble_format_version"] == 2
        for index in range(len(model.estimators_)):
            assert archive[f"member_{index:04d}_sampled_indices"].dtype == np.dtype(
                "<i8"
            )
            assert archive[f"member_{index:04d}_oob_indices"].dtype == np.dtype(
                "<i8"
            )
    restored = DarkoClassifier.load_model(path)
    np.testing.assert_array_equal(restored.predict_proba(X), expected)
    assert restored.ensemble_metadata_ == model.ensemble_metadata_
    second = tmp_path / "private-classifier-resaved.npz"
    restored.save_model(second)
    assert second.read_bytes() == path.read_bytes()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda metadata: metadata.__setitem__("sample_fraction", 0.7),
            "sampling",
        ),
        (
            lambda metadata: metadata["members"][0].__setitem__(
                "fitted_thread_count", 999
            ),
            "fitted metadata",
        ),
        (
            lambda metadata: metadata["policy_resolutions"]["colsample"].__setitem__(
                "resolved", 1.0
            ),
            "policy resolution",
        ),
        (
            lambda metadata: metadata["members"][0].__setitem__(
                "sampled_rows", metadata["members"][0]["sampled_rows"] - 1
            ),
            "sample counts",
        ),
        (
            lambda metadata: metadata["members"][0].__setitem__(
                "sampled_indices_sha256", "0" * 64
            ),
            "index digest",
        ),
        (
            lambda metadata: metadata["members"][0].__setitem__(
                "oob_indices_sha256", "f" * 64
            ),
            "index digest",
        ),
        (
            lambda metadata: metadata.__setitem__(
                "explicit_user_params", [["colsample"]]
            ),
            "policy",
        ),
    ],
)
def test_private_safe_load_rejects_forged_metadata(
    tmp_path, mutate, message
):
    X, y = _regression_data(n=100)
    model = _fit_private(DarkoRegressor(**_params(iterations=4)), X, y)
    source = tmp_path / "private-source.npz"
    corrupt = tmp_path / "private-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    mutate(header["metadata"])
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match=message):
        DarkoRegressor.load_model(corrupt)


def test_private_safe_load_rejects_forged_index_payload(tmp_path):
    X, y = _regression_data(n=100)
    model = _fit_private(DarkoRegressor(**_params(iterations=4)), X, y)
    source = tmp_path / "private-source.npz"
    corrupt = tmp_path / "private-corrupt.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    sampled = arrays["member_0000_sampled_indices"]
    sampled[0] = (sampled[0] + 1) % len(X)
    np.savez_compressed(corrupt, **arrays)

    with pytest.raises(ValueError, match="index (digest|provenance)"):
        DarkoRegressor.load_model(corrupt)


def test_private_safe_load_rejects_legacy_archive_without_index_payloads(
    tmp_path,
):
    X, y = _regression_data(n=100)
    model = _fit_private(DarkoRegressor(**_params(iterations=4)), X, y)
    source = tmp_path / "private-source.npz"
    legacy = tmp_path / "private-legacy.npz"
    model.save_model(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {
            name: archive[name].copy()
            for name in archive.files
            if not name.endswith(("_sampled_indices", "_oob_indices"))
        }
    header = json.loads(str(arrays["header"]))
    header["ensemble_format_version"] = 1
    arrays["header"] = np.array(json.dumps(header))
    np.savez_compressed(legacy, **arrays)

    with pytest.raises(ValueError, match="index provenance payload is missing"):
        DarkoRegressor.load_model(legacy)


def test_private_fit_failure_restores_fresh_and_previously_fitted_state(
    monkeypatch,
):
    X, y = _regression_data(n=100)
    fresh = DarkoRegressor(**_params(iterations=3))
    original_fit = DarkoRegressor.fit
    calls = 0

    def fail_second_member(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("member failure")
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(DarkoRegressor, "fit", fail_second_member)
    with pytest.raises(RuntimeError, match="member failure"):
        _fit_private(fresh, X, y)
    assert not hasattr(fresh, "model_")
    assert not hasattr(fresh, "estimators_")

    monkeypatch.setattr(DarkoRegressor, "fit", original_fit)
    fitted = DarkoRegressor(**_params(n_ensembles=1, iterations=3)).fit(X, y)
    expected = fitted.predict(X)
    fitted.set_params(n_ensembles=2)
    with pytest.raises(ValueError, match="sample_fraction"):
        _fit_private_ensemble_v3(
            fitted,
            X,
            y,
            sampling="without_replacement",
            sampling_unit="rows",
            sample_fraction=0.7,
        )
    np.testing.assert_array_equal(fitted.predict(X), expected)
    assert not hasattr(fitted, "estimators_")


def test_private_fit_restores_ambient_numba_thread_mask():
    X, y = _regression_data(n=90)
    ambient = numba.get_num_threads()
    try:
        model = _fit_private(
            DarkoRegressor(
                **_params(
                    iterations=3,
                    tree_mode="lightgbm",
                    thread_count=2,
                )
            ),
            X,
            y,
        )
        assert all(
            record["fitted_thread_count"] == 2
            for record in model.ensemble_metadata_["members"]
        )
        assert numba.get_num_threads() == ambient
        model.predict(X)
        assert numba.get_num_threads() == ambient
    finally:
        numba.set_num_threads(ambient)


def test_public_ensemble_fit_remains_version_one_bootstrap():
    X, y = _regression_data(n=100)
    model = DarkoRegressor(**_params(iterations=3)).fit(X, y)
    assert model.ensemble_metadata_["version"] == 1
    assert model.ensemble_metadata_["bootstrap"] == "rows"
    assert "private_prototype" not in model.ensemble_metadata_


def test_private_control_is_prediction_exact_to_public_bootstrap():
    X, y = _regression_data(n=120)
    public = DarkoRegressor(**_params(iterations=5)).fit(X, y)
    private = _fit_private_ensemble_v3(
        DarkoRegressor(**_params(iterations=5)),
        X,
        y,
        sampling="bootstrap",
        sampling_unit="rows",
        member_policy="none",
    )

    np.testing.assert_array_equal(private.predict(X), public.predict(X))
    assert [
        record["bootstrap_indices_sha256"]
        for record in public.ensemble_metadata_["members"]
    ] == [
        record["sampled_indices_sha256"]
        for record in private.ensemble_metadata_["members"]
    ]
    assert [
        record["oob_indices_sha256"]
        for record in public.ensemble_metadata_["members"]
    ] == [
        record["oob_indices_sha256"]
        for record in private.ensemble_metadata_["members"]
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling": "unknown"},
        {"member_policy": "unknown"},
        {"explicit_user_params": ["depth"]},
        {"explicit_user_params": ["colsample", "colsample"]},
    ],
)
def test_private_contract_controls_fail_closed(kwargs):
    X, y = _regression_data(n=80)
    params = {
        "sampling": "without_replacement",
        "sampling_unit": "rows",
        "sample_fraction": 0.8,
        "member_policy": "none",
    }
    params.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        _fit_private_ensemble_v3(
            DarkoRegressor(**_params(iterations=2)),
            X,
            y,
            **params,
        )
