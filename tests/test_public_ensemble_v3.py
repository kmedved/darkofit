import io
import json

import numba
import numpy as np
import pytest
from sklearn.base import clone

from darkofit import DarkoClassifier, DarkoRegressor


def _data(seed=20260722, n=120):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    signal = 1.3 * X[:, 0] - 0.6 * X[:, 1] + 0.25 * X[:, 2]
    y = signal + rng.normal(scale=0.25, size=n)
    return X, y


def _params(**extra):
    params = {
        "iterations": 4,
        "depth": 3,
        "early_stopping_rounds": 2,
        "random_state": 43,
        "n_ensembles": 8,
        "ensemble_mode": "v3",
        "diagnostic_warnings": "never",
    }
    params.update(extra)
    return params


def _rewrite_archive(path, output, mutate):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    mutate(arrays)
    np.savez_compressed(output, **arrays)


def _rewrite_header(path, output, mutate):
    def apply(arrays):
        header = json.loads(str(arrays["header"]))
        mutate(header)
        arrays["header"] = np.array(json.dumps(header))

    _rewrite_archive(path, output, apply)


def _strip_parallelism_from_member(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["params"].pop("ensemble_parallelism")
    arrays["header"] = np.array(json.dumps(header))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()


def test_public_constructor_surface_is_clone_safe_and_bootstrap_is_strict():
    estimator = DarkoRegressor(
        ensemble_mode="v3",
        ensemble_member_learning_rate=None,
        ensemble_member_colsample=1.0,
    )
    assert clone(estimator).get_params(deep=False) == estimator.get_params(
        deep=False
    )
    defaults = DarkoRegressor().get_params(deep=False)
    assert defaults["ensemble_mode"] == "bootstrap"
    assert defaults["ensemble_member_learning_rate"] == "policy"
    assert defaults["ensemble_member_colsample"] == "policy"
    assert defaults["ensemble_parallelism"] == "auto"

    X, y = _data(n=60)
    with pytest.raises(ValueError, match="ensemble_mode"):
        DarkoRegressor(ensemble_mode="unknown").fit(X, y)
    with pytest.raises(ValueError, match="must remain 'policy'"):
        DarkoRegressor(ensemble_member_learning_rate=0.1).fit(X, y)
    with pytest.raises(ValueError, match="requires n_ensembles=8"):
        DarkoRegressor(**_params(n_ensembles=7)).fit(X, y)
    with pytest.raises(ValueError, match="ensemble_parallelism"):
        DarkoRegressor(ensemble_parallelism="sequential").fit(X, y)
    with pytest.raises(ValueError, match="ensemble_parallelism"):
        DarkoRegressor(**_params(ensemble_parallelism="unknown")).fit(X, y)


def test_explicit_bootstrap_mode_preserves_legacy_ensemble_behavior(tmp_path):
    X, y = _data(n=90)
    params = {
        "iterations": 3,
        "depth": 2,
        "early_stopping_rounds": 2,
        "random_state": 11,
        "n_ensembles": 2,
        "diagnostic_warnings": "never",
    }
    implicit = DarkoRegressor(**params).fit(X, y)
    explicit = DarkoRegressor(**params, ensemble_mode="bootstrap").fit(X, y)
    np.testing.assert_array_equal(explicit.predict(X), implicit.predict(X))
    assert explicit.ensemble_metadata_ == implicit.ensemble_metadata_
    path = tmp_path / "legacy-bootstrap.npz"
    explicit.save_model(path)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
    assert header["ensemble_format_version"] == 1


def test_public_v3_explicit_overrides_metadata_and_v4_roundtrip(tmp_path):
    X, y = _data(n=100)
    model = DarkoRegressor(**_params(
        learning_rate=0.07,
        colsample=0.6,
        ensemble_member_learning_rate=None,
        ensemble_member_colsample=1.0,
    )).fit(X, y)

    metadata = model.ensemble_metadata_
    assert metadata["version"] == 2
    assert metadata["ensemble_mode"] == "v3"
    assert metadata["recipe_contract"] == "ensemble-v3-public-contract-v1"
    assert metadata["recipe_version"] == 1
    assert metadata["public_fit_surface"] is True
    assert "private_prototype" not in metadata
    assert "future_constructor_params" not in metadata
    assert metadata["sampling"] == "without_replacement"
    assert metadata["sample_fraction"] == 0.8
    assert metadata["member_policy"] == "donor_balanced_v1"
    assert metadata["explicit_user_params"] == [
        "learning_rate",
        "colsample",
    ]
    assert metadata["base_constructor_params"]["learning_rate"] == 0.07
    assert metadata["base_constructor_params"]["colsample"] == 0.6
    assert metadata["base_constructor_params"]["ensemble_mode"] == "v3"
    assert metadata["base_constructor_params"]["ensemble_parallelism"] == (
        "auto"
    )
    assert metadata["parallel_dispatch"]["requested"] == "auto"
    assert metadata["parallel_dispatch"]["route"] == "sequential_fallback"
    assert model.ensemble_parallel_dispatch_ == metadata["parallel_dispatch"]
    assert all(member.n_ensembles == 1 for member in model.estimators_)
    assert all(member.ensemble_mode == "bootstrap" for member in model.estimators_)
    assert all(
        member.ensemble_member_learning_rate == "policy"
        and member.ensemble_member_colsample == "policy"
        and member.ensemble_parallelism == "auto"
        for member in model.estimators_
    )
    assert all(member.learning_rate is None for member in model.estimators_)
    assert all(member.colsample == 1.0 for member in model.estimators_)
    np.testing.assert_array_equal(
        model.predict(X),
        np.mean([member.predict(X) for member in model.estimators_], axis=0),
    )
    np.testing.assert_array_equal(
        model.shap_values(X[:6]),
        np.mean(
            [member.shap_values(X[:6]) for member in model.estimators_],
            axis=0,
        ),
    )

    path = tmp_path / "public-v3.npz"
    resaved = tmp_path / "public-v3-resaved.npz"
    model.save_model(path)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        assert header["ensemble_format_version"] == 4
        assert header["params"] == metadata["base_constructor_params"]
        assert "private_group_codes" not in archive.files
        assert "group_codes" not in archive.files
        assert len([
            name for name in archive.files if name.endswith("_sampled_indices")
        ]) == 8
    restored = DarkoRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.ensemble_metadata_ == metadata
    assert restored.get_params(deep=False) == model.get_params(deep=False)
    assert clone(restored).get_params(deep=False) == model.get_params(deep=False)
    assert (
        restored.ensemble_parallel_dispatch_
        == model.ensemble_parallel_dispatch_
    )
    restored.save_model(resaved)
    assert resaved.read_bytes() == path.read_bytes()


def test_public_v3_loads_pre_parallelism_v1_archive(tmp_path):
    X, y = _data(n=80)
    model = DarkoRegressor(
        **_params(ensemble_parallelism="sequential")
    ).fit(X, y)
    source = tmp_path / "current.npz"
    legacy = tmp_path / "legacy-v1.npz"
    model.save_model(source)

    def make_legacy(arrays):
        header = json.loads(str(arrays["header"]))
        header["params"].pop("ensemble_parallelism")
        metadata = header["metadata"]
        metadata["version"] = 1
        metadata.pop("parallel_dispatch")
        metadata["base_constructor_params"].pop("ensemble_parallelism")
        for record in metadata["members"]:
            record["member_constructor_params"].pop("ensemble_parallelism")
        arrays["header"] = np.array(json.dumps(header))
        for index in range(8):
            name = f"member_{index:04d}"
            arrays[name] = _strip_parallelism_from_member(
                np.asarray(arrays[name], dtype=np.uint8).tobytes()
            )

    _rewrite_archive(source, legacy, make_legacy)
    restored = DarkoRegressor.load_model(legacy)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.ensemble_parallelism == "auto"
    assert not hasattr(restored, "ensemble_parallel_dispatch_")


def test_public_v3_group_classifier_is_disjoint_and_restores_threads(tmp_path):
    group_sizes = np.arange(2, 14)
    groups = np.repeat(np.arange(len(group_sizes)), group_sizes)
    X, continuous = _data(n=len(groups))
    y = np.digitize(continuous, np.quantile(continuous, [1 / 3, 2 / 3]))
    ambient = numba.get_num_threads()
    try:
        model = DarkoClassifier(**_params(
            ensemble_bootstrap="groups",
            thread_count=2,
        )).fit(
            X,
            y,
            groups=groups,
            sample_weight=np.linspace(0.5, 1.5, len(y)),
        )
        assert numba.get_num_threads() == ambient
        expected = np.mean(
            [member.predict_proba(X) for member in model.estimators_], axis=0
        )
        np.testing.assert_array_equal(model.predict_proba(X), expected)
        assert numba.get_num_threads() == ambient
        list(model.staged_predict_proba(X[:10]))
        assert numba.get_num_threads() == ambient
        assert all(
            record["group_disjoint"] is True
            and record["sampled_unique_groups"] == 10
            and record["oob_groups"] == 2
            for record in model.ensemble_metadata_["members"]
        )
        path = tmp_path / "public-v3-groups.npz"
        model.save_model(path)
        assert numba.get_num_threads() == ambient
        with np.load(path, allow_pickle=False) as archive:
            assert "group_codes" in archive.files
            assert "private_group_codes" not in archive.files
        restored = DarkoClassifier.load_model(path)
        assert numba.get_num_threads() == ambient
        np.testing.assert_array_equal(restored.predict_proba(X), expected)
        np.testing.assert_array_equal(
            restored._ensemble_group_codes_, model._ensemble_group_codes_
        )
    finally:
        numba.set_num_threads(ambient)


def test_public_v3_binary_classifier_uses_soft_vote():
    X, continuous = _data(n=100)
    y = (continuous > np.median(continuous)).astype(np.int64)
    model = DarkoClassifier(**_params()).fit(X, y)
    expected = np.mean(
        [member.predict_proba(X) for member in model.estimators_], axis=0
    )
    np.testing.assert_array_equal(model.predict_proba(X), expected)


def test_public_v3_forced_parallel_is_exact_and_round_trips(tmp_path):
    X, y = _data(n=100)
    sequential = DarkoRegressor(
        **_params(
            ensemble_parallelism="sequential",
            thread_count=14,
        )
    ).fit(X, y)
    parallel = DarkoRegressor(
        **_params(
            ensemble_parallelism="parallel",
            thread_count=14,
        )
    ).fit(X, y)

    np.testing.assert_array_equal(parallel.predict(X), sequential.predict(X))
    assert parallel.ensemble_parallel_dispatch_["route"] == "process_parallel"
    assert parallel.ensemble_parallel_dispatch_["reason"] == (
        "user_forced_parallel"
    )
    assert sequential.ensemble_parallel_dispatch_["route"] == (
        "sequential_fallback"
    )
    assert sequential.ensemble_parallel_dispatch_["reason"] == (
        "user_forced_sequential"
    )
    path = tmp_path / "parallel-v3.npz"
    parallel.save_model(path)
    restored = DarkoRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(X), parallel.predict(X))
    assert restored.ensemble_parallel_dispatch_ == (
        parallel.ensemble_parallel_dispatch_
    )


def test_public_v3_invalid_refit_restores_existing_single_model():
    X, y = _data(n=80)
    estimator = DarkoRegressor(
        iterations=2,
        depth=2,
        random_state=9,
        diagnostic_warnings="never",
    ).fit(X, y)
    expected = estimator.predict(X)
    estimator.set_params(ensemble_mode="v3", n_ensembles=7)
    with pytest.raises(ValueError, match="requires n_ensembles=8"):
        estimator.fit(X, y)
    np.testing.assert_array_equal(estimator.predict(X), expected)
    assert not hasattr(estimator, "estimators_")


@pytest.mark.parametrize("loss", ["RMSE", "MAE", "Quantile"])
def test_public_v3_supports_all_scalar_regression_losses(loss):
    X, y = _data(n=80)
    model = DarkoRegressor(**_params(loss=loss, alpha=0.8)).fit(X, y)
    assert model.predict(X[:5]).shape == (5,)


def test_public_v3_supports_categorical_and_explicit_ordinal_inputs():
    pd = pytest.importorskip("pandas")
    X_num, y = _data(n=90)
    levels = np.array(["low", "mid", "high"])
    X = pd.DataFrame({
        "value": X_num[:, 0],
        "kind": np.where(X_num[:, 1] >= 0.0, "up", "down"),
        "tier": levels[np.digitize(X_num[:, 2], [-0.5, 0.5])],
    })
    model = DarkoRegressor(**_params()).fit(
        X,
        y,
        cat_features=["kind"],
        ordinal_features={"tier": ("low", "mid", "high")},
    )
    assert model.predict(X[:4]).shape == (4,)
    assert model.ensemble_metadata_["shared_preprocessing"] == "member_local"


@pytest.mark.parametrize(
    ("estimator_kwargs", "fit_kwargs", "message"),
    [
        ({"preset": "accuracy"}, {}, "preset is not supported"),
        ({"tree_mode": "auto"}, {}, "tree_mode='auto'"),
        ({"auto_learning_rate_probe": True}, {}, "auto_learning_rate_probe=True"),
        ({"refit": True}, {}, "refit=True"),
        ({"loss": "Gaussian"}, {}, "scalar regression losses only"),
        ({}, {"callbacks": [lambda _: False]}, "callbacks are not supported"),
        ({}, {"eval_set": (np.zeros((4, 5)), np.zeros(4))}, "eval_set"),
        ({}, {"ordinal_features": "auto"}, "ordinal_features='auto'"),
        ({}, {"groups": np.arange(80)}, "groups cannot be used"),
    ],
)
def test_public_v3_unsupported_surfaces_fail_transactionally(
    estimator_kwargs,
    fit_kwargs,
    message,
):
    X, y = _data(n=80)
    estimator = DarkoRegressor(**_params(**estimator_kwargs))
    with pytest.raises(ValueError, match=message):
        estimator.fit(X, y, **fit_kwargs)
    assert not hasattr(estimator, "model_")
    assert not hasattr(estimator, "estimators_")


@pytest.mark.parametrize(
    "corruption",
    ["version", "extra", "metadata", "params", "parallel_dispatch"],
)
def test_public_v4_load_rejects_schema_and_contract_corruption(
    tmp_path,
    corruption,
):
    X, y = _data(n=80)
    model = DarkoRegressor(**_params(iterations=2)).fit(X, y)
    source = tmp_path / "source.npz"
    corrupt = tmp_path / f"corrupt-{corruption}.npz"
    model.save_model(source)
    if corruption == "extra":
        _rewrite_archive(
            source,
            corrupt,
            lambda arrays: arrays.__setitem__("unexpected", np.array([1])),
        )
    else:
        def mutate(header):
            if corruption == "version":
                header["ensemble_format_version"] = 3
            elif corruption == "metadata":
                header["metadata"]["recipe_version"] = 2
            elif corruption == "parallel_dispatch":
                header["metadata"]["parallel_dispatch"]["member_work"] += 1
            else:
                header["params"]["ensemble_member_colsample"] = 0.7

        _rewrite_header(source, corrupt, mutate)
    with pytest.raises(ValueError, match="invalid DarkoFit model"):
        DarkoRegressor.load_model(corrupt)


def test_public_v3_rejects_nested_member_payload(tmp_path):
    X, y = _data(n=80)
    model = DarkoRegressor(**_params(iterations=2)).fit(X, y)
    source = tmp_path / "source.npz"
    corrupt = tmp_path / "nested.npz"
    model.save_model(source)

    def mutate(arrays):
        arrays["member_0000"] = np.frombuffer(
            source.read_bytes(), dtype=np.uint8
        ).copy()

    _rewrite_archive(source, corrupt, mutate)
    with pytest.raises(ValueError, match="nested ensemble member"):
        DarkoRegressor.load_model(corrupt)
