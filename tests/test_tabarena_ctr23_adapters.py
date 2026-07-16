"""Integration checks for the CTR23-only comparator callback sidecar."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CHIMERABOOST_CHECKOUT = Path("/Users/kmedved/.cache/chimeraboost-v0.14.1")
BENCHMARK_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("autogluon", "catboost", "tabarena")
)
requires_local_benchmark_stack = pytest.mark.skipif(
    not BENCHMARK_DEPS_AVAILABLE or not CHIMERABOOST_CHECKOUT.is_dir(),
    reason="the pinned local TabArena/ChimeraBoost benchmark stack is unavailable",
)


@requires_local_benchmark_stack
def test_ctr23_callback_sidecar_is_isolated_observable_and_persistable(tmp_path):
    from autogluon.tabular.models import CatBoostModel

    from benchmarks.run_tabarena_regression_same_machine import (
        activate_chimeraboost_checkout,
    )

    activate_chimeraboost_checkout(CHIMERABOOST_CHECKOUT)
    from chimeraboost import ChimeraBoostRegressor

    from benchmarks.tabarena_ctr23_adapters import (
        CTR23ComparatorCatBoostModel,
        CTR23ComparatorChimeraBoostModel,
        CTR23_TIME_CALLBACK_AUDIT_KEY,
    )

    rng = np.random.default_rng(17)
    X = pd.DataFrame(
        {"x": rng.normal(size=260), "z": rng.normal(size=260)}
    )
    y = pd.Series(2.0 * X["x"] - X["z"] + rng.normal(scale=0.2, size=260))
    X_train, X_val = X.iloc[:220], X.iloc[220:]
    y_train, y_val = y.iloc[:220], y.iloc[220:]

    original_chimera_fit = ChimeraBoostRegressor.fit
    catboost_globals = CatBoostModel._fit.__globals__
    original_catboost_callback = catboost_globals["TimeCheckCallback"]

    for model_cls, engine in (
        (CTR23ComparatorChimeraBoostModel, "chimeraboost"),
        (CTR23ComparatorCatBoostModel, "catboost"),
    ):
        model_path = tmp_path / engine
        model = model_cls(
            path=f"{model_path}/",
            name="Probe",
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            hyperparameters={},
        )
        model.fit(
            X=X_train,
            y=y_train,
            X_val=X_val,
            y_val=y_val,
            time_limit=30.0,
            num_cpus=2,
            num_gpus=0,
        )
        info = model.get_info()
        audit = info[CTR23_TIME_CALLBACK_AUDIT_KEY]
        assert set(audit) == {
            "schema_version",
            "kind",
            "engine",
            "time_limit_seconds",
            "time_callback_instrumented",
            "time_callback_instance_count",
            "time_callback_call_count",
            "time_callback_hit",
        }
        assert audit["schema_version"] == 1
        assert audit["kind"] == "darkofit_ctr23_time_callback_audit"
        assert audit["engine"] == engine
        assert 0.0 < audit["time_limit_seconds"] <= 30.0
        assert audit["time_callback_instrumented"] is True
        assert audit["time_callback_instance_count"] == 1
        assert audit["time_callback_call_count"] >= 1
        assert audit["time_callback_hit"] is False

        # The published comparator schema remains unchanged; the audit is a
        # separate sidecar and the fit-local callback class is not persisted.
        assert info["comparator_fit"]["schema_version"] == 1
        assert "time_callback_hit" not in info["comparator_fit"]
        predictions = model.predict(X_val)
        saved_path = model.save(verbose=False)
        loaded = model_cls.load(path=saved_path, verbose=False)
        assert np.array_equal(predictions, loaded.predict(X_val))
        assert loaded.get_info()[CTR23_TIME_CALLBACK_AUDIT_KEY] == audit

    assert ChimeraBoostRegressor.fit is original_chimera_fit
    assert catboost_globals["TimeCheckCallback"] is original_catboost_callback


@requires_local_benchmark_stack
def test_chimera_callback_audit_covers_both_validation_selected_lanes(tmp_path):
    from benchmarks.run_tabarena_regression_same_machine import (
        activate_chimeraboost_checkout,
    )

    activate_chimeraboost_checkout(CHIMERABOOST_CHECKOUT)
    from benchmarks.tabarena_ctr23_adapters import (
        CTR23ComparatorChimeraBoostModel,
        CTR23_TIME_CALLBACK_AUDIT_KEY,
    )

    rng = np.random.default_rng(19)
    values = rng.normal(size=(1_250, 5))
    X = pd.DataFrame(values, columns=["a", "b", "c", "d", "e"])
    y = pd.Series(
        2.0 * X["a"] - X["b"] + X["c"] * X["d"]
        + rng.normal(scale=0.3, size=len(X))
    )
    model = CTR23ComparatorChimeraBoostModel(
        path=f"{tmp_path}/",
        name="LaneProbe",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        hyperparameters={},
    )
    model.fit(
        X=X.iloc[:1_100],
        y=y.iloc[:1_100],
        X_val=X.iloc[1_100:],
        y_val=y.iloc[1_100:],
        time_limit=30.0,
        num_cpus=2,
        num_gpus=0,
    )
    info = model.get_info()
    assert info["comparator_fit"]["linear_selection_performed"] is True
    audit = info[CTR23_TIME_CALLBACK_AUDIT_KEY]
    assert audit["time_callback_instance_count"] == 2
    assert audit["time_callback_call_count"] >= 2
    assert audit["time_callback_hit"] is False
