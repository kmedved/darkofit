"""Benchmark-only AutoGluon adapters with preprocessing instrumentation.

This module is separate from the run script so AutoGluon can pickle child model
classes by their stable module name.  Nothing here is part of DarkoFit's public
package API.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

from benchmarks.tabarena_adapter import DarkoFitModel
from benchmarks.preprocessing_instrumentation import capture_preprocessing
from benchmarks.run_tabarena_same_machine_performance import (
    CHIMERA_REGRESSOR_PRODUCT_DEFAULTS,
)

if TYPE_CHECKING:
    import pandas as pd


class _PreprocessingInfoMixin:
    """Expose benchmark-only preprocessing totals through AutoGluon get_info."""

    preprocessing_package: str

    def _fit_with_preprocessing_capture(self, fit, *args, **kwargs):
        with capture_preprocessing(self.preprocessing_package) as captured:
            result = fit(*args, **kwargs)
        self.preprocessing_fit_transform_seconds = float(captured["seconds"])
        self.preprocessing_fit_transform_calls = int(captured["calls"])
        return result

    def get_info(self, *args, **kwargs):
        info = super().get_info(*args, **kwargs)
        info["preprocessing_fit_transform_seconds"] = float(
            getattr(self, "preprocessing_fit_transform_seconds", 0.0)
        )
        info["preprocessing_fit_transform_calls"] = int(
            getattr(self, "preprocessing_fit_transform_calls", 0)
        )
        info["preprocessing_instrumentation"] = (
            "FeaturePreprocessor.fit_transform cumulative wall time"
        )
        return info


class SameMachineDarkoFitModel(_PreprocessingInfoMixin, DarkoFitModel):
    """DarkoFit TabArena adapter with child-level preprocessing telemetry."""

    ag_key = "DARKOPERF"
    ag_name = "DarkoFitSameMachine"
    preprocessing_package = "darkofit"

    def _fit(self, *args, **kwargs):
        return self._fit_with_preprocessing_capture(
            super()._fit, *args, **kwargs
        )


class LocalChimeraBoostModel(AbstractModel):
    """Local ChimeraBoost 0.14.1 default adapted to the TabArena contract."""

    ag_key = "CHIMERAPERF"
    ag_name = "ChimeraBoostSameMachine"
    seed_name = "random_state"

    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._cat_col_names = list(
                X.select_dtypes(include="category").columns
            )
        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        sample_weight=None,
        **kwargs,
    ) -> None:
        del num_gpus
        started = time.time()
        from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

        model_cls = (
            ChimeraBoostRegressor
            if self.problem_type == "regression"
            else ChimeraBoostClassifier
        )
        X = self.preprocess(X, y=y, is_train=True)
        params = self._get_model_params()
        params["thread_count"] = max(1, int(num_cpus))
        self.model = model_cls(**params)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val = self.preprocess(X_val)
            eval_set = (X_val, y_val)

        fit_kwargs = {}
        if time_limit is not None:
            deadline = started + 0.95 * time_limit

            def time_stop(iteration, train_loss, val_loss, model):
                del iteration, train_loss, val_loss, model
                return time.time() >= deadline

            fit_kwargs["callbacks"] = time_stop
        self.model.fit(
            X,
            y,
            cat_features=self._cat_col_names or None,
            eval_set=eval_set,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    def _get_default_auxiliary_params(self) -> dict:
        auxiliary = super()._get_default_auxiliary_params()
        auxiliary.update({"valid_raw_types": ["int", "float", "category"]})
        return auxiliary

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        return ResourceManager.get_cpu_count(only_physical_cores=True), 0

    def get_info(self, *args, **kwargs):
        info = super().get_info(*args, **kwargs)
        info["benchmark_package"] = "chimeraboost"
        info["benchmark_package_version"] = os.environ.get(
            "DARKOFIT_BENCH_CHIMERA_VERSION"
        )
        info["benchmark_source_commit"] = os.environ.get(
            "DARKOFIT_BENCH_CHIMERA_COMMIT"
        )
        model = getattr(self, "model", None)
        info["benchmark_regressor_product_parameters"] = {
            name: getattr(model, name, None)
            for name in CHIMERA_REGRESSOR_PRODUCT_DEFAULTS
        }
        return info


class SameMachineChimeraBoostModel(
    _PreprocessingInfoMixin, LocalChimeraBoostModel
):
    """ChimeraBoost adapter with child-level preprocessing telemetry."""

    preprocessing_package = "chimeraboost"

    def _fit(self, *args, **kwargs):
        return self._fit_with_preprocessing_capture(
            super()._fit, *args, **kwargs
        )
