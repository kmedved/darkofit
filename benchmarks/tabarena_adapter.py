"""AutoGluon model adapter used to benchmark DarkoFit with TabArena.

This module intentionally lives with the benchmark tooling instead of the
``darkofit`` package: importing DarkoFit must not require AutoGluon or TabArena.
"""

from __future__ import annotations

from copy import deepcopy
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

        ``time_limit`` is enforced as a soft monotonic deadline between
        completed boosting rounds. A single tree plus final model bookkeeping
        may overrun the deadline, so a small safety margin is reserved.
        """
        del num_gpus

        from darkofit.callbacks import WallClockStopper

        deadline = None
        if time_limit is not None:
            time_limit = float(time_limit)
            deadline = WallClockStopper(
                time_limit,
                safety_margin=min(5.0, 0.05 * time_limit),
            )

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
            callbacks=deadline,
        )

        core_model = self.model.model_
        best_iteration = int(self.model.best_n_estimators_)
        resolved_learning_rate = float(self.model.learning_rate_)
        selected_tree_mode = str(core_model.tree_mode_)
        refit_params = self.model.get_refit_params(strategy="exact")
        refit_params["use_best_model"] = False
        refit_params["refit"] = False
        refit_param_names = (
            "iterations",
            "learning_rate",
            "tree_mode",
            "early_stopping",
            "early_stopping_rounds",
            "use_best_model",
            "refit",
            "depth",
            "num_leaves",
            "l2_leaf_reg",
            "min_child_samples",
            "min_child_weight",
            "cat_smoothing",
        )
        self.params_trained.update(
            {name: refit_params[name] for name in refit_param_names}
        )

        linear_residual_active = bool(
            getattr(self.model, "linear_residual_active_", False)
        )
        stop_reason = str(core_model.stop_reason_)
        tree_mode_selection = getattr(
            self.model, "tree_mode_selection_", None
        )
        deadline_hit = stop_reason == "time_limit"
        if tree_mode_selection is not None and deadline is not None:
            deadline_hit = bool(deadline.deadline_hit)
        fitted_metadata = {
            "iterations_requested": int(params["iterations"]),
            "iterations_attempted": int(core_model.iterations_attempted_),
            "rounds_completed": int(core_model.rounds_completed_),
            "rounds_retained": int(core_model.best_iteration_),
            "best_iteration": best_iteration,
            "resolved_learning_rate": resolved_learning_rate,
            "requested_tree_mode": str(params["tree_mode"]),
            "selected_tree_mode": selected_tree_mode,
            "selected_lane": (
                "linear_residual" if linear_residual_active else "boosting"
            ),
            "linear_residual_active": linear_residual_active,
            "early_stopping_rounds": (
                None
                if core_model.early_stopping_rounds_ is None
                else int(core_model.early_stopping_rounds_)
            ),
            "stop_reason": stop_reason,
            "wall_clock_limit_seconds": time_limit,
            "wall_clock_safety_margin_seconds": (
                None if deadline is None else float(deadline.safety_margin)
            ),
            "wall_clock_effective_seconds": (
                None if deadline is None else float(deadline.effective_seconds)
            ),
            "wall_clock_elapsed_seconds": (
                None if deadline is None else float(deadline.elapsed_seconds)
            ),
            "deadline_hit": deadline_hit,
            "deadline_is_soft": deadline is not None,
        }
        if tree_mode_selection is not None:
            fitted_metadata["tree_mode_selection"] = deepcopy(
                tree_mode_selection
            )
        self._fit_metadata["darkofit_fit"] = fitted_metadata

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
