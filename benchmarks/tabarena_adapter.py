"""AutoGluon model adapter used to benchmark DarkoFit with TabArena.

This module intentionally lives with the benchmark tooling instead of the
``darkofit`` package: importing DarkoFit must not require AutoGluon or TabArena.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.utils.config_utils import ConfigGenerator


class DarkoFitModel(AbstractModel):
    """Expose DarkoFit through TabArena's AutoGluon model contract."""

    ag_key = "DARKO"
    ag_name = "DarkoFit"
    seed_name = "random_state"

    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Preserve native categories and record their integer positions."""
        X = super()._preprocess(X, **kwargs)
        if is_train:
            categorical_columns = set(X.select_dtypes(include="category").columns)
            self._categorical_indices = [
                index
                for index, column in enumerate(X.columns)
                if column in categorical_columns
            ]
        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        sample_weight=None,
        sample_weight_val=None,
        **kwargs,
    ) -> None:
        """Fit DarkoFit using TabArena's validation split and resources.

        The three-dataset smoke is bounded by the configured iteration cap and
        early stopping. DarkoFit does not yet expose a wall-clock callback, so
        ``time_limit`` is recorded but cannot be enforced inside a boosting
        round; that hook is required before a submission-grade full run.
        """
        del num_gpus

        if self.problem_type == "regression":
            from darkofit import DarkoRegressor

            model_cls = DarkoRegressor
        else:
            from darkofit import DarkoClassifier

            model_cls = DarkoClassifier

        X = self.preprocess(X, y=y, is_train=True)
        params = self._get_model_params()
        params["thread_count"] = max(1, int(num_cpus))
        self.model = model_cls(**params)
        self._tabarena_time_limit = time_limit

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val = self.preprocess(X_val)
            eval_set = (X_val, y_val)

        self.model.fit(
            X,
            y,
            cat_features=self._categorical_indices or None,
            eval_set=eval_set,
            sample_weight=sample_weight,
            eval_sample_weight=sample_weight_val,
        )

    def _set_default_params(self) -> None:
        defaults = {
            "iterations": 1000,
            "early_stopping": True,
            "tree_mode": "catboost",
            "diagnostic_warnings": "never",
        }
        for parameter, value in defaults.items():
            self._set_default_param_value(parameter, value)

    def _get_default_auxiliary_params(self) -> dict:
        auxiliary = super()._get_default_auxiliary_params()
        auxiliary.update({"valid_raw_types": ["int", "float", "category"]})
        return auxiliary

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        return ResourceManager.get_cpu_count(only_physical_cores=True), 0

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        """Return the default plus a conservative, valid DarkoFit HPO space."""
        from autogluon.common.space import Categorical, Int, Real

        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(
            model_cls=cls,
            manual_configs=[{}],
            search_space={
                "learning_rate": Real(0.03, 0.2, log=True),
                "depth": Int(4, 8),
                "l2_leaf_reg": Real(0.1, 10.0, log=True),
                "min_child_weight": Real(0.0, 8.0),
                "subsample": Real(0.6, 1.0),
                "colsample": Real(0.6, 1.0),
                "max_bins": Categorical(128, 254),
                "cat_smoothing": Real(0.1, 10.0, log=True),
                "ts_permutations": Int(1, 4),
                "ordered_boosting": Categorical(False, True),
            },
        )
