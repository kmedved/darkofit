import json
from copy import deepcopy

import numba
import numpy as np
import pytest
from sklearn.base import clone

import darkofit.booster as booster_module
from darkofit import DarkoClassifier, DarkoRegressor
from darkofit.booster import (
    DistributionalBoosting,
    GradientBoosting,
    MulticlassBoosting,
    _resolve_oblivious_kernel_dispatch,
)


def _numeric_data(*, seed=20260721, n_rows=240, n_features=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    y = 1.2 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.2, size=n_rows)
    return X, y


def _core_params(**overrides):
    params = {
        "iterations": 4,
        "learning_rate": 0.1,
        "depth": 4,
        "max_bins": 64,
        "min_child_weight": 0.0,
        "min_child_samples": 2,
        "thread_count": 4,
        "random_state": 17,
        "tree_mode": "catboost",
        "diagnostic_warnings": "never",
    }
    params.update(overrides)
    return params


def _projected_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name].copy()
            for name in archive.files
            if name != "header"
        }
        header = json.loads(str(archive["header"]))
    header["params"].pop("oblivious_kernel")
    header["auto_params"].pop("oblivious_kernel_dispatch")
    wrapper = header.get("wrapper")
    if isinstance(wrapper, dict) and isinstance(wrapper.get("params"), dict):
        wrapper["params"].pop("oblivious_kernel", None)
    return header, arrays


def _assert_projected_archives_exact(left, right):
    left_header, left_arrays = _projected_archive(left)
    right_header, right_arrays = _projected_archive(right)
    assert left_header == right_header
    assert set(left_arrays) == set(right_arrays)
    for name in left_arrays:
        np.testing.assert_array_equal(left_arrays[name], right_arrays[name])


def _rewrite_dispatch_metadata(source, destination, mutate):
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    metadata = deepcopy(header["auto_params"]["oblivious_kernel_dispatch"])
    mutate(metadata)
    header["auto_params"]["oblivious_kernel_dispatch"] = metadata
    arrays["header"] = np.array(json.dumps(header, sort_keys=True))
    np.savez_compressed(destination, **arrays)


def _rewrite_wrapper_oblivious_kernel(source, destination, value):
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    header = json.loads(str(arrays["header"]))
    header["wrapper"]["params"]["oblivious_kernel"] = value
    arrays["header"] = np.array(json.dumps(header, sort_keys=True))
    np.savez_compressed(destination, **arrays)


def test_oblivious_dispatch_rule_is_static_and_threshold_ties_choose_unfused():
    inputs = {
        "functional_ineligibility": None,
        "n_rows": 8_192,
        "n_active_features": 15,
        "n_threads": 4,
        "depth": 6,
        "max_realized_bins": 129,
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "logical_cpu_count": 14,
    }
    scan_work = 8_192 * 4

    tied, tied_instrument = _resolve_oblivious_kernel_dispatch(
        "auto", threshold=scan_work, **inputs
    )
    below, below_instrument = _resolve_oblivious_kernel_dispatch(
        "auto", threshold=scan_work + 1, **inputs
    )

    assert tied == _resolve_oblivious_kernel_dispatch(
        "auto", threshold=scan_work, **inputs
    )[0]
    assert tied_instrument is below_instrument is True
    assert tied["resolved"] == "unfused"
    assert tied["reason"] == "at_or_above_threshold"
    assert tied["scan_work"] == scan_work
    assert below["resolved"] == "fused"
    assert below["reason"] == "below_threshold"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("platform_system", "Linux", "unsupported_platform"),
        ("n_rows", 8_191, "rows_outside_envelope"),
        ("n_active_features", 7, "features_outside_envelope"),
        ("n_threads", 15, "threads_outside_envelope"),
        ("depth", 9, "depth_outside_envelope"),
        ("max_realized_bins", 64, "bins_outside_envelope"),
        ("max_realized_bins", 256, "bins_outside_envelope"),
    ],
)
def test_oblivious_dispatch_auto_fallback_reasons_are_explicit(
    field, value, reason
):
    inputs = {
        "functional_ineligibility": None,
        "n_rows": 8_192,
        "n_active_features": 15,
        "n_threads": 4,
        "depth": 6,
        "max_realized_bins": 129,
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "logical_cpu_count": 14,
        "threshold": 0,
    }
    inputs[field] = value

    metadata, instrument = _resolve_oblivious_kernel_dispatch("auto", **inputs)

    assert metadata["resolved"] == "fused"
    assert metadata["reason"] == reason
    assert metadata["automatic_eligible"] is False
    assert instrument is False


def test_oblivious_dispatch_explicit_mode_rejects_ineligible_configuration():
    with pytest.raises(ValueError, match="row sampling active"):
        _resolve_oblivious_kernel_dispatch(
            "unfused",
            functional_ineligibility="row_sampling_active",
            n_rows=10_000,
            n_active_features=15,
            n_threads=4,
            depth=6,
            max_realized_bins=129,
            platform_system="Darwin",
            platform_machine="arm64",
            logical_cpu_count=14,
            threshold=None,
        )


@pytest.mark.parametrize("case", ["rmse", "weighted_rmse", "binary_logloss"])
def test_public_forced_lanes_are_prediction_and_projected_archive_exact(
    tmp_path, case
):
    X, y = _numeric_data(seed=31)
    fit_kwargs = {}
    params = _core_params()
    if case == "weighted_rmse":
        fit_kwargs["sample_weight"] = np.linspace(0.5, 1.5, len(y))
    elif case == "binary_logloss":
        y = (y > np.median(y)).astype(np.float64)
        params["loss"] = "Logloss"

    fused = GradientBoosting(
        **params, oblivious_kernel="fused"
    ).fit(X, y, **fit_kwargs)
    unfused = GradientBoosting(
        **params, oblivious_kernel="unfused"
    ).fit(X, y, **fit_kwargs)

    np.testing.assert_array_equal(unfused.predict_raw(X), fused.predict_raw(X))
    np.testing.assert_array_equal(
        unfused.feature_importances_, fused.feature_importances_
    )
    assert fused.oblivious_kernel_dispatch_["fused_level_count"] > 0
    assert fused.oblivious_kernel_dispatch_["unfused_level_count"] == 0
    assert unfused.oblivious_kernel_dispatch_["unfused_level_count"] > 0
    assert unfused.oblivious_kernel_dispatch_["fused_level_count"] == 0
    assert (
        fused.auto_params_["oblivious_kernel_dispatch"]
        == fused.oblivious_kernel_dispatch_
    )

    fused_path = tmp_path / f"{case}-fused.npz"
    unfused_path = tmp_path / f"{case}-unfused.npz"
    fused.save_model(fused_path)
    unfused.save_model(unfused_path)
    _assert_projected_archives_exact(fused_path, unfused_path)


def test_public_forced_categorical_lanes_are_exact(tmp_path):
    X_num, y = _numeric_data(seed=37)
    categories = np.asarray(["guard", "wing", "big"], dtype=object)
    X = np.empty((len(y), 3), dtype=object)
    X[:, 0] = X_num[:, 0]
    X[:, 1] = categories[np.arange(len(y)) % len(categories)]
    X[:, 2] = X_num[:, 1]

    fused = GradientBoosting(
        **_core_params(), oblivious_kernel="fused"
    ).fit(X, y, cat_features=[1])
    unfused = GradientBoosting(
        **_core_params(), oblivious_kernel="unfused"
    ).fit(X, y, cat_features=[1])

    np.testing.assert_array_equal(unfused.predict_raw(X), fused.predict_raw(X))
    fused_path = tmp_path / "categorical-fused.npz"
    unfused_path = tmp_path / "categorical-unfused.npz"
    fused.save_model(fused_path)
    unfused.save_model(unfused_path)
    _assert_projected_archives_exact(fused_path, unfused_path)


def test_auto_threshold_injection_engages_unfused_without_runtime_racing(
    monkeypatch,
):
    X, y = _numeric_data(seed=41, n_rows=8_192, n_features=8)
    monkeypatch.setattr(booster_module, "_OBLIVIOUS_KERNEL_AUTO_THRESHOLD", 0)
    monkeypatch.setattr(booster_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(booster_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(booster_module.os, "cpu_count", lambda: 14)

    model = GradientBoosting(
        **_core_params(iterations=1), oblivious_kernel="auto"
    ).fit(X, y)
    metadata = model.oblivious_kernel_dispatch_

    assert metadata["automatic_eligible"] is True
    assert metadata["resolved"] == "unfused"
    assert metadata["reason"] == "at_or_above_threshold"
    assert metadata["threshold"] == 0
    assert metadata["unfused_level_count"] > 0
    assert metadata["fused_level_count"] == 0


def test_auto_fallback_records_actual_fused_level_engagement(monkeypatch):
    monkeypatch.setattr(booster_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(booster_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(booster_module.os, "cpu_count", lambda: 14)
    X, y = _numeric_data(seed=42)
    model = GradientBoosting(**_core_params(), oblivious_kernel="auto").fit(
        X, y
    )
    metadata = model.oblivious_kernel_dispatch_

    assert metadata["reason"] == "rows_outside_envelope"
    assert metadata["engaged"] is True
    assert metadata["fused_level_count"] > 0
    assert metadata["unfused_level_count"] == 0


def test_fit_fails_if_builder_counters_disagree_with_selected_lane(monkeypatch):
    X, y = _numeric_data(seed=44)
    original = booster_module.build_oblivious_tree

    def contradictory_builder(*args, **kwargs):
        fused_counter = kwargs.pop("fused_oblivious_counter", None)
        unfused_counter = kwargs.pop("unfused_oblivious_counter", None)
        assert fused_counter is not None
        assert unfused_counter is not None
        result = original(*args, **kwargs)
        unfused_counter[0] += 1
        return result

    monkeypatch.setattr(
        booster_module, "build_oblivious_tree", contradictory_builder
    )

    with pytest.raises(RuntimeError, match="actual builder engagement"):
        GradientBoosting(
            **_core_params(iterations=1), oblivious_kernel="fused"
        ).fit(X, y)


def test_zero_iteration_fit_records_no_builder_engagement():
    X, y = _numeric_data(seed=45)
    model = GradientBoosting(
        **_core_params(iterations=0), oblivious_kernel="fused"
    ).fit(X, y)

    metadata = model.oblivious_kernel_dispatch_
    assert metadata["engaged"] is False
    assert metadata["fused_level_count"] == 0
    assert metadata["unfused_level_count"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"tree_mode": "lightgbm"},
        {"subsample": 0.8},
        {"colsample": 0.8},
        {"thread_count": 2},
        {"random_strength": 0.1},
    ],
)
def test_explicit_unfused_fit_rejects_unsupported_lanes(overrides):
    X, y = _numeric_data(seed=43)
    with pytest.raises(ValueError, match="failed eligibility"):
        GradientBoosting(
            **_core_params(**overrides), oblivious_kernel="unfused"
        ).fit(X, y)


def test_auto_multiclass_fallback_is_recorded_and_explicit_mode_is_rejected():
    X, y_raw = _numeric_data(seed=47)
    y = np.mod(np.arange(len(y_raw)), 3)
    auto = MulticlassBoosting(
        **_core_params(iterations=1), oblivious_kernel="auto"
    ).fit(X, y)

    assert auto.oblivious_kernel_dispatch_["reason"] == "non_scalar_booster"
    assert auto.oblivious_kernel_dispatch_["engaged"] is False
    with pytest.raises(ValueError, match="non scalar booster"):
        MulticlassBoosting(
            **_core_params(iterations=1), oblivious_kernel="unfused"
        ).fit(X, y)


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"tree_mode": "lightgbm"}, "tree_mode_not_catboost"),
        ({"tree_mode": "hybrid"}, "tree_mode_not_catboost"),
        ({"subsample": 0.8}, "row_sampling_active"),
        ({"colsample": 0.8}, "feature_sampling_active"),
        ({"random_strength": 0.1}, "split_randomness_active"),
        ({"thread_count": 1}, "insufficient_threads"),
        ({"thread_count": 2}, "insufficient_threads"),
    ],
)
def test_auto_unsupported_lanes_record_loud_fallback(overrides, expected_reason):
    X, y = _numeric_data(seed=49)
    model = GradientBoosting(
        **_core_params(iterations=1, **overrides), oblivious_kernel="auto"
    ).fit(X, y)

    assert model.oblivious_kernel_dispatch_["reason"] == expected_reason
    assert model.oblivious_kernel_dispatch_["engaged"] is False


def test_auto_row_parallel_and_distributional_fallbacks_are_recorded():
    X, y = _numeric_data(seed=51, n_rows=1_200)
    row_parallel = GradientBoosting(
        **_core_params(iterations=1, histogram_parallelism="row"),
        oblivious_kernel="auto",
    ).fit(X, y)
    assert (
        row_parallel.oblivious_kernel_dispatch_["reason"]
        == "row_parallel_histograms_active"
    )
    assert row_parallel.oblivious_kernel_dispatch_["engaged"] is False

    distributional = DistributionalBoosting(
        **_core_params(iterations=1, tree_mode="lightgbm"),
        loss="Gaussian",
        oblivious_kernel="auto",
    ).fit(X, y)
    assert (
        distributional.oblivious_kernel_dispatch_["reason"]
        == "non_scalar_booster"
    )
    assert distributional.oblivious_kernel_dispatch_["engaged"] is False


def test_public_binary_classifier_forced_lanes_are_exact(tmp_path):
    X, y_raw = _numeric_data(seed=52)
    y = (y_raw > np.median(y_raw)).astype(np.int64)
    params = _core_params()
    fused = DarkoClassifier(
        **params, oblivious_kernel="fused"
    ).fit(X, y)
    unfused = DarkoClassifier(
        **params, oblivious_kernel="unfused"
    ).fit(X, y)

    np.testing.assert_array_equal(
        unfused.predict_proba(X), fused.predict_proba(X)
    )
    fused_path = tmp_path / "binary-fused.npz"
    unfused_path = tmp_path / "binary-unfused.npz"
    fused.save_model(fused_path)
    unfused.save_model(unfused_path)
    _assert_projected_archives_exact(fused_path, unfused_path)


def test_public_forced_callback_stop_is_exact(tmp_path):
    X, y = _numeric_data(seed=54)

    class StopAfterTwo:
        stop_reason = "dispatch_test_limit"

        def __call__(self, progress):
            return progress.rounds_completed >= 2

    fitted = []
    for lane in ("fused", "unfused"):
        fitted.append(
            GradientBoosting(
                **_core_params(iterations=8), oblivious_kernel=lane
            ).fit(X, y, callbacks=StopAfterTwo())
        )
    fused, unfused = fitted
    assert fused.stop_reason_ == unfused.stop_reason_ == "dispatch_test_limit"
    assert fused.best_iteration_ == unfused.best_iteration_ == 2
    np.testing.assert_array_equal(
        fused.predict_raw(X), unfused.predict_raw(X)
    )
    fused_path = tmp_path / "callback-fused.npz"
    unfused_path = tmp_path / "callback-unfused.npz"
    fused.save_model(fused_path)
    unfused.save_model(unfused_path)
    _assert_projected_archives_exact(fused_path, unfused_path)


def test_public_forced_early_stopping_refit_is_exact(tmp_path):
    X, y = _numeric_data(seed=56, n_rows=320)
    params = _core_params(
        iterations=24,
        early_stopping=True,
        early_stopping_rounds=3,
        validation_fraction=0.2,
        refit=True,
    )
    fused = DarkoRegressor(
        **params, oblivious_kernel="fused"
    ).fit(X, y)
    unfused = DarkoRegressor(
        **params, oblivious_kernel="unfused"
    ).fit(X, y)

    assert fused.refit_ is unfused.refit_ is True
    assert fused.n_estimators_ == unfused.n_estimators_
    assert fused.best_n_estimators_ == unfused.best_n_estimators_
    np.testing.assert_array_equal(unfused.predict(X), fused.predict(X))
    fused_path = tmp_path / "refit-fused.npz"
    unfused_path = tmp_path / "refit-unfused.npz"
    fused.save_model(fused_path)
    unfused.save_model(unfused_path)
    _assert_projected_archives_exact(fused_path, unfused_path)


def test_malformed_and_auto_tree_mode_explicit_overrides_fail_loudly():
    X, y = _numeric_data(seed=55)
    with pytest.raises(ValueError, match="oblivious_kernel must be one of"):
        GradientBoosting(oblivious_kernel="bogus")
    with pytest.raises(ValueError, match="requires an explicit.*tree_mode"):
        DarkoRegressor(oblivious_kernel="unfused", tree_mode="auto").fit(X, y)


def test_wrapper_clone_and_safe_load_preserve_forced_dispatch_metadata(tmp_path):
    X, y = _numeric_data(seed=53)
    estimator = DarkoRegressor(
        **_core_params(), oblivious_kernel="unfused"
    )
    assert clone(estimator).get_params()["oblivious_kernel"] == "unfused"

    fitted = estimator.fit(X, y)
    path = tmp_path / "wrapper-unfused.npz"
    fitted.save_model(path)
    loaded = DarkoRegressor.load_model(path)

    assert loaded.get_params()["oblivious_kernel"] == "unfused"
    assert (
        loaded.model_.oblivious_kernel_dispatch_
        == fitted.model_.oblivious_kernel_dispatch_
    )
    np.testing.assert_array_equal(loaded.predict(X), fitted.predict(X))
    resaved = tmp_path / "wrapper-unfused-resaved.npz"
    loaded.save_model(resaved)
    _assert_projected_archives_exact(path, resaved)


def test_wrapper_save_uses_fitted_oblivious_kernel_after_param_mutation(tmp_path):
    X, y = _numeric_data(seed=54)
    fitted = DarkoRegressor(
        **_core_params(), oblivious_kernel="fused"
    ).fit(X, y)
    fitted.set_params(oblivious_kernel="unfused")

    path = tmp_path / "wrapper-fitted-kernel.npz"
    fitted.save_model(path)
    loaded = DarkoRegressor.load_model(path)

    assert loaded.oblivious_kernel == "fused"
    assert loaded.model_.oblivious_kernel == "fused"


def test_wrapper_load_rejects_oblivious_kernel_that_differs_from_booster(
    tmp_path,
):
    X, y = _numeric_data(seed=56)
    fitted = DarkoRegressor(
        **_core_params(), oblivious_kernel="fused"
    ).fit(X, y)
    source = tmp_path / "wrapper-kernel-source.npz"
    tampered = tmp_path / "wrapper-kernel-tampered.npz"
    fitted.save_model(source)
    _rewrite_wrapper_oblivious_kernel(source, tampered, "unfused")

    with pytest.raises(ValueError, match="oblivious kernel.*loaded booster"):
        DarkoRegressor.load_model(tampered)


def test_safe_load_rejects_inconsistent_dispatch_metadata(tmp_path):
    X, y = _numeric_data(seed=59)
    model = GradientBoosting(
        **_core_params(), oblivious_kernel="unfused"
    ).fit(X, y)
    source = tmp_path / "source.npz"
    tampered = tmp_path / "tampered.npz"
    model.save_model(source)
    _rewrite_dispatch_metadata(
        source, tampered, lambda metadata: metadata.update(resolved="fused")
    )

    with pytest.raises(ValueError, match="dispatch.*engagement"):
        GradientBoosting.load_model(tampered)


@pytest.mark.parametrize("field", ["n_rows", "n_threads", "max_realized_bins"])
def test_safe_load_rejects_dispatch_inputs_that_disagree_with_fit(tmp_path, field):
    X, y = _numeric_data(seed=60)
    model = GradientBoosting(
        **_core_params(), oblivious_kernel="unfused"
    ).fit(X, y)
    source = tmp_path / f"source-{field}.npz"
    tampered = tmp_path / f"tampered-{field}.npz"
    model.save_model(source)

    def mutate(metadata):
        metadata["inputs"][field] += 1
        inputs = metadata["inputs"]
        active_threads = min(
            inputs["n_threads"], inputs["n_active_features"]
        )
        metadata["scan_work"] = inputs["n_rows"] * (
            (inputs["n_active_features"] + active_threads - 1)
            // active_threads
        )

    _rewrite_dispatch_metadata(source, tampered, mutate)
    expected = {
        "n_rows": "row count",
        "n_threads": "thread count",
        "max_realized_bins": "realized bins",
    }[field]
    with pytest.raises(ValueError, match=f"dispatch.*{expected}"):
        GradientBoosting.load_model(tampered)


def test_safe_load_rejects_level_count_below_retained_tree_depth(tmp_path):
    X, y = _numeric_data(seed=62)
    model = GradientBoosting(
        **_core_params(), oblivious_kernel="unfused"
    ).fit(X, y)
    retained_levels = sum(tree.depth for tree in model.trees_)
    assert retained_levels > 1
    source = tmp_path / "source-count.npz"
    tampered = tmp_path / "tampered-count.npz"
    model.save_model(source)

    def mutate(metadata):
        metadata["unfused_level_count"] = retained_levels - 1
        metadata["engaged"] = True

    _rewrite_dispatch_metadata(source, tampered, mutate)
    with pytest.raises(ValueError, match="dispatch.*fitted trees"):
        GradientBoosting.load_model(tampered)


def test_safe_save_rejects_divergent_fitted_dispatch_views(tmp_path):
    X, y = _numeric_data(seed=63)
    model = GradientBoosting(
        **_core_params(), oblivious_kernel="unfused"
    ).fit(X, y)
    model.oblivious_kernel_dispatch_["reason"] = "user_forced_fused"

    with pytest.raises(ValueError, match="attribute disagrees with auto_params"):
        model.save_model(tmp_path / "divergent.npz")


def test_forced_unfused_fit_and_predict_restore_ambient_thread_mask():
    available = int(numba.config.NUMBA_NUM_THREADS)
    if available < 3:
        pytest.skip("explicit fused-lane dispatch requires more than two threads")
    X, y = _numeric_data(seed=61)
    previous = numba.get_num_threads()
    ambient = min(available, 4)
    try:
        numba.set_num_threads(ambient)
        model = GradientBoosting(
            **_core_params(thread_count=3), oblivious_kernel="unfused"
        ).fit(X, y)
        assert numba.get_num_threads() == ambient
        model.predict_raw(X)
        assert numba.get_num_threads() == ambient
    finally:
        numba.set_num_threads(previous)
