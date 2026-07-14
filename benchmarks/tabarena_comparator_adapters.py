"""Audited adapters for the same-machine TabArena comparator campaign.

The native classes inherit the benchmarked implementations without changing
their defaults or fit paths.  The ordinal classes add the exact source-frozen,
target-free representation used by the earlier DarkoFit mechanism screen.  All
six classes add only JSON-safe fitted telemetry under ``comparator_fit`` and a
representation audit under ``benchmark_representation``.

The classes live in an importable module because AutoGluon persists child model
classes by module and qualified name when it pickles a TabArena bag.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from autogluon.tabular.models import CatBoostModel
from tabarena.models.chimeraboost.model import ChimeraBoostModel

try:
    from benchmarks.tabarena_adapter import DarkoFitModel
    from benchmarks.tabarena_screen_adapters import (
        AIRFOIL_CHILD_CODE_VALUES,
        AIRFOIL_COLUMNS,
        DIAMONDS_CHILD_CODE_RANKS,
        DIAMONDS_COLUMNS,
        DIAMONDS_ORDERS,
        MIAMI_COLUMNS,
        ORDINAL_COMPACT_DOMAINS,
        REPRESENTATION_METADATA_KEY,
    )
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    from tabarena_adapter import DarkoFitModel
    from tabarena_screen_adapters import (
        AIRFOIL_CHILD_CODE_VALUES,
        AIRFOIL_COLUMNS,
        DIAMONDS_CHILD_CODE_RANKS,
        DIAMONDS_COLUMNS,
        DIAMONDS_ORDERS,
        MIAMI_COLUMNS,
        ORDINAL_COMPACT_DOMAINS,
        REPRESENTATION_METADATA_KEY,
    )

if TYPE_CHECKING:
    import pandas as pd


COMPARATOR_METADATA_KEY = "comparator_fit"


def _json_safe(value: Any, *, field: str) -> Any:
    """Return a strict JSON value or fail rather than persisting an opaque repr."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{field} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{field} contains a non-string mapping key")
            normalized[key] = _json_safe(item, field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RuntimeError(
        f"{field} contains unsupported value type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _safe_mapping(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    normalized = _json_safe(value, field=field)
    if not isinstance(normalized, dict):  # Defensive: the input contract is a map.
        raise RuntimeError(f"{field} did not normalize to a mapping")
    json.dumps(normalized, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return normalized


def _feature_schema_digest(columns: list[str]) -> str:
    if (
        not columns
        or any(not isinstance(column, str) for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise RuntimeError("feature schema must contain unique string names")
    payload = json.dumps(
        columns,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fit_argument(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    position: int,
    default: Any,
) -> Any:
    """Read a common official ``_fit`` argument without changing its dispatch."""
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else default


def _record_comparator_metadata(model: Any, metadata: Mapping[str, Any]) -> None:
    normalized = _safe_mapping(metadata, field=COMPARATOR_METADATA_KEY)
    model._fit_metadata[COMPARATOR_METADATA_KEY] = normalized


class _NativeRepresentationMixin:
    """Record the engine's unchanged native child-training representation."""

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        input_columns = list(X.columns)
        if (
            not input_columns
            or any(not isinstance(column, str) for column in input_columns)
            or len(set(input_columns)) != len(input_columns)
            or input_columns != list(self.features or [])
        ):
            raise RuntimeError(
                "native comparator input must match the declared unique string schema"
            )
        categorical_columns = [
            str(column)
            for column in X.select_dtypes(include="category").columns
        ]

        super()._fit(X, y, *args, **kwargs)

        fitted_columns = list(getattr(self, "_features_internal", []))
        if (
            not fitted_columns
            or any(not isinstance(column, str) for column in fitted_columns)
            or len(set(fitted_columns)) != len(fitted_columns)
        ):
            raise RuntimeError("native comparator fitted feature schema is invalid")
        fitted_set = set(fitted_columns)
        dropped_columns = [
            column for column in input_columns if column not in fitted_set
        ]
        if fitted_columns != [
            column for column in input_columns if column not in dropped_columns
        ]:
            raise RuntimeError(
                "native comparator fitted schema must be an order-preserving subset"
            )
        dropped_unique_counts = [
            int(X[column].nunique(dropna=False)) for column in dropped_columns
        ]
        if any(count != 1 for count in dropped_unique_counts):
            raise RuntimeError(
                "native comparator may drop only child-training-fold constants"
            )
        fitted_categorical_columns = [
            column for column in categorical_columns if column in fitted_set
        ]
        self._fit_metadata[REPRESENTATION_METADATA_KEY] = {
            "schema_version": 2,
            "kind": "native",
            "fit_scope": "comparator_child_training_fold",
            "feature_alignment_policy": "autogluon_child_drop_unique",
            "target_used_by_representation": bool(fitted_categorical_columns),
            "input_feature_count": len(input_columns),
            "output_feature_count": len(fitted_columns),
            "external_feature_schema_sha256": _feature_schema_digest(
                input_columns
            ),
            "fitted_feature_schema_sha256": _feature_schema_digest(
                fitted_columns
            ),
            "categorical_input_columns": categorical_columns,
            "fitted_categorical_input_columns": fitted_categorical_columns,
            "dropped_constant_input_columns": dropped_columns,
            "dropped_constant_input_unique_counts": dropped_unique_counts,
        }


class _SafeOrdinalRepresentationMixin:
    """Apply the source-frozen safe ordinal mapping before an engine fits.

    The schema, category domains, physical Airfoil values, and declared
    Diamonds ranks are imported from the already-audited screen adapter.  No
    mapping is learned from row order or the target, and any unknown schema,
    category domain, or value fails closed.
    """

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._screen_representation_fit_active = True
        self._screen_representation_fit_calls = 0
        self._screen_representation_eval_transform_calls = 0
        self._screen_representation_eval_unknown_counts: list[int] = []
        try:
            super()._fit(X, y, *args, **kwargs)
        finally:
            self._screen_representation_fit_active = False
        if self._screen_representation_fit_calls != 1:
            raise RuntimeError(
                "safe comparator representation must be fitted exactly once"
            )
        metadata = self._representation_metadata()
        metadata.update(
            {
                "fit_scope": "child_training_rows_only",
                "target_used_by_representation": False,
                "fit_calls": self._screen_representation_fit_calls,
                "eval_transform_calls_during_fit": (
                    self._screen_representation_eval_transform_calls
                ),
                "eval_unknown_counts": list(
                    self._screen_representation_eval_unknown_counts
                ),
            }
        )
        self._fit_metadata[REPRESENTATION_METADATA_KEY] = metadata

    def _preprocess(
        self,
        X: pd.DataFrame,
        is_train: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        is_catboost = isinstance(self, CatBoostModel)
        if is_catboost:
            # CatBoost's model-specific preprocessing compacts a categorical
            # column to the categories observed in this child fold.  The frozen
            # ordinal contract instead validates the complete source-declared
            # compact domain, so run AutoGluon's generic feature alignment,
            # apply the frozen mapping, and then give the numeric result to the
            # unmodified CatBoost preprocessing implementation.
            X = super(CatBoostModel, self)._preprocess(
                X, is_train=is_train, **kwargs
            )
        else:
            X = super()._preprocess(X, is_train=is_train, **kwargs)
        if is_train:
            if self._screen_representation_fit_calls:
                raise RuntimeError(
                    "safe comparator representation cannot be refitted"
                )
            self._fit_representation(X)
            self._screen_representation_fit_calls += 1
        elif not self._screen_representation_fit_calls:
            raise RuntimeError(
                "safe comparator representation used before child training fit"
            )
        transformed, unknown_count = self._transform_representation(X)
        if not is_train and self._screen_representation_fit_active:
            self._screen_representation_eval_transform_calls += 1
            self._screen_representation_eval_unknown_counts.append(
                int(unknown_count)
            )
        if is_catboost:
            transformed = CatBoostModel._preprocess(
                self, transformed, is_train=is_train, **kwargs
            )
        return transformed

    def _fit_representation(self, X: pd.DataFrame) -> None:
        import pandas as pd

        self._representation_input_columns = list(X.columns)
        columns = tuple(str(column) for column in X.columns)
        categorical = [
            str(column)
            for column in X.columns
            if isinstance(X[column].dtype, pd.CategoricalDtype)
        ]
        if columns == AIRFOIL_COLUMNS and categorical == ["attack-angle"]:
            self._representation_domain = "airfoil_attack_angle_numeric"
            self._representation_rules = {
                "attack-angle": {
                    str(code): value
                    for code, value in enumerate(AIRFOIL_CHILD_CODE_VALUES)
                }
            }
        elif columns == DIAMONDS_COLUMNS and categorical == list(DIAMONDS_ORDERS):
            self._representation_domain = "diamonds_declared_orders"
            self._representation_rules = {
                column: {
                    str(code): rank
                    for code, rank in enumerate(
                        DIAMONDS_CHILD_CODE_RANKS[column]
                    )
                }
                for column in DIAMONDS_ORDERS
            }
        elif columns == MIAMI_COLUMNS and categorical == []:
            self._representation_domain = "miami_avno60plus_binary"
            self._representation_rules = {
                "avno60plus": {"0": 0, "0.0": 0, "1": 1, "1.0": 1}
            }
        else:
            raise RuntimeError(
                "safe ordinal comparator input does not match a predeclared "
                f"schema: columns={columns!r}, categoricals={categorical!r}"
            )
        self._representation_compact_domains = ORDINAL_COMPACT_DOMAINS[
            self._representation_domain
        ]
        self._validate_compact_category_domains(X)
        self._representation_categorical_positions = [
            self._representation_input_columns.index(column)
            for column in self._representation_rules
        ]
        self._representation_observed_counts = []
        for column in self._representation_rules:
            values = X[column].astype(object)
            self._representation_observed_counts.append(
                int(values.nunique(dropna=True))
            )
        frozen_schema = {
            "domain": self._representation_domain,
            "columns": columns,
            "rules": {
                column: [[key, value] for key, value in rule.items()]
                for column, rule in self._representation_rules.items()
            },
            "compact_category_domains": {
                column: list(domain)
                for column, domain in self._representation_compact_domains.items()
            },
        }
        self._representation_schema_sha256 = hashlib.sha256(
            json.dumps(
                frozen_schema,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self._representation_output_columns = list(X.columns)

    def _validate_compact_category_domains(self, X: pd.DataFrame) -> None:
        import pandas as pd

        for column, expected in self._representation_compact_domains.items():
            if not isinstance(X[column].dtype, pd.CategoricalDtype):
                raise RuntimeError(
                    f"safe ordinal {column!r} lost its categorical dtype"
                )
            observed = tuple(X[column].cat.categories.tolist())
            if observed != expected:
                raise RuntimeError(
                    f"safe ordinal {column!r} compact category domain changed: "
                    f"observed={observed!r}, expected={expected!r}"
                )

    def _transform_representation(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        if list(X.columns) != self._representation_input_columns:
            raise RuntimeError("safe ordinal input schema changed after fitting")
        self._validate_compact_category_domains(X)
        transformed = X.copy()
        for column, rule in self._representation_rules.items():
            values = X[column].astype(object)
            missing = values.isna().to_numpy(dtype=bool, copy=False)
            encoded = np.full(len(values), np.nan, dtype=np.float64)
            for index, value in enumerate(values):
                if missing[index]:
                    continue
                key = str(value)
                if key not in rule:
                    raise RuntimeError(
                        f"safe ordinal {column!r} contains undeclared value {key!r}"
                    )
                encoded[index] = rule[key]
            transformed[column] = encoded

        # DarkoFit records integer categorical positions while ChimeraBoost
        # records names.  Clear both after the target-free numeric transform so
        # neither engine target-encodes the restored ordinal values.  CatBoost
        # independently derives its Pool categorical list from ``transformed``.
        self._categorical_indices = []
        self._cat_col_names = []
        return transformed, 0

    def _representation_metadata(self) -> dict[str, Any]:
        return {
            "kind": "safe_ordinal",
            "domain": self._representation_domain,
            "mapping_source": "source_frozen_before_campaign",
            "input_feature_count": len(self._representation_input_columns),
            "output_feature_count": len(self._representation_output_columns),
            "categorical_input_positions": list(
                self._representation_categorical_positions
            ),
            "observed_training_category_counts": list(
                self._representation_observed_counts
            ),
            "compact_category_domains": {
                column: list(domain)
                for column, domain in self._representation_compact_domains.items()
            },
            "category_schema_sha256": self._representation_schema_sha256,
            "missing_policy": "numeric_nan",
            "unknown_policy": "fail_closed",
            "remaining_native_target_stat_positions": [],
        }


class _DarkoFitTelemetryMixin:
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        num_cpus = _fit_argument(
            args, kwargs, name="num_cpus", position=3, default=1
        )
        num_gpus = _fit_argument(
            args, kwargs, name="num_gpus", position=4, default=0
        )
        super()._fit(X, y, *args, **kwargs)

        fitted = dict(self._fit_metadata.get("darkofit_fit", {}))
        if not fitted:
            raise RuntimeError("DarkoFit comparator fit metadata is missing")
        resolved_params = dict(self.model.get_params(deep=False))
        resolved_params["iterations"] = int(fitted["iterations_requested"])
        resolved_params["learning_rate"] = float(
            fitted["resolved_learning_rate"]
        )
        resolved_params["tree_mode"] = str(fitted["selected_tree_mode"])
        core = self.model.model_
        auto_structure = getattr(core, "_auto_structure_params_", {}).get(
            "resolved", {}
        )
        for name in (
            "depth",
            "num_leaves",
            "l2_leaf_reg",
            "min_child_samples",
            "min_child_weight",
            "cat_smoothing",
        ):
            if name in auto_structure:
                resolved_params[name] = auto_structure[name]
        resolved_params["thread_count"] = int(core.n_threads_)

        metadata = {
            "schema_version": 1,
            "engine": "darkofit",
            "iterations_requested": int(fitted["iterations_requested"]),
            "best_iteration": int(fitted["best_iteration"]),
            "rounds_retained": int(fitted["rounds_retained"]),
            "resolved_params": _safe_mapping(
                resolved_params, field="comparator_fit.resolved_params"
            ),
            "num_cpus": int(num_cpus),
            "num_gpus": float(num_gpus),
            "requested_tree_mode": str(fitted["requested_tree_mode"]),
            "selected_tree_mode": str(fitted["selected_tree_mode"]),
            "selected_lane": str(fitted["selected_lane"]),
            "resolved_learning_rate": float(
                fitted["resolved_learning_rate"]
            ),
            "iterations_attempted": int(fitted["iterations_attempted"]),
            "rounds_completed": int(fitted["rounds_completed"]),
            "stop_reason": str(fitted["stop_reason"]),
            "wall_clock_limit_seconds": fitted[
                "wall_clock_limit_seconds"
            ],
            "wall_clock_safety_margin_seconds": fitted[
                "wall_clock_safety_margin_seconds"
            ],
            "wall_clock_effective_seconds": fitted[
                "wall_clock_effective_seconds"
            ],
            "wall_clock_elapsed_seconds": fitted[
                "wall_clock_elapsed_seconds"
            ],
            "deadline_hit": bool(fitted["deadline_hit"]),
            "deadline_is_soft": bool(fitted["deadline_is_soft"]),
        }
        _record_comparator_metadata(self, metadata)


def _selected_chimera_core(model: Any) -> Any:
    members = getattr(model, "estimators_", None)
    if members is not None:
        if len(members) != 1:
            raise RuntimeError(
                "comparator telemetry requires a single ChimeraBoost member"
            )
        model = members[0]
    core = getattr(model, "model_", None)
    if core is None:
        raise RuntimeError("fitted ChimeraBoost core model is missing")
    return core


def _infer_chimera_stop_reason(
    *,
    attempted: int,
    retained: int,
    requested: int,
    time_limit: float | None,
    linear_selection_performed: bool,
) -> str | None:
    """Infer only outcomes proven by all fits hidden behind the wrapper."""
    if linear_selection_performed and time_limit is not None:
        # The official auto lane fits constant then linear candidates under one
        # absolute callback but persists only the winner. The selected core's
        # history cannot prove whether the discarded candidate hit the shared
        # deadline and thereby affected lane selection.
        return None
    if attempted > retained:
        return "early_stopping"
    if retained >= requested:
        return "iteration_limit"
    if time_limit is None:
        return "no_legal_split"
    return None


class _ChimeraBoostTelemetryMixin:
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        time_limit = _fit_argument(
            args, kwargs, name="time_limit", position=2, default=None
        )
        num_cpus = _fit_argument(
            args, kwargs, name="num_cpus", position=3, default=1
        )
        num_gpus = _fit_argument(
            args, kwargs, name="num_gpus", position=4, default=0
        )
        super()._fit(X, y, *args, **kwargs)

        core = _selected_chimera_core(self.model)
        retained = int(len(core.trees_))
        valid_history = list(getattr(core, "valid_history_", []))
        attempted = len(valid_history) if valid_history else retained
        requested = int(core.n_estimators)
        active_linear = bool(
            core.linear_leaves
            and getattr(core, "_centers_std_", None) is not None
        )
        selection_value = getattr(
            self.model, "linear_leaves_selected_", None
        )
        selection_performed = selection_value is not None
        stop_reason = _infer_chimera_stop_reason(
            attempted=attempted,
            retained=retained,
            requested=requested,
            time_limit=time_limit,
            linear_selection_performed=selection_performed,
        )

        resolved_params = dict(self.model.get_params(deep=False))
        for name in (
            "n_estimators",
            "depth",
            "l2_leaf_reg",
            "max_bins",
            "subsample",
            "colsample",
            "cat_smoothing",
            "cat_n_permutations",
            "early_stopping_rounds",
            "min_child_weight",
            "random_state",
            "ordered_boosting",
            "cat_combinations",
            "leaf_estimation_iterations",
            "linear_lambda",
        ):
            resolved_params[name] = getattr(core, name)
        resolved_params["learning_rate"] = float(core.lr_)
        resolved_params["linear_leaves"] = active_linear
        resolved_params["thread_count"] = int(core.n_threads_)

        metadata = {
            "schema_version": 1,
            "engine": "chimeraboost",
            "iterations_requested": requested,
            "best_iteration": retained,
            "rounds_retained": retained,
            "resolved_params": _safe_mapping(
                resolved_params, field="comparator_fit.resolved_params"
            ),
            "num_cpus": int(num_cpus),
            "num_gpus": float(num_gpus),
            "selected_lane": "linear" if active_linear else "constant",
            "linear_leaves_selected": active_linear,
            "linear_selection_performed": selection_performed,
            "resolved_learning_rate": float(core.lr_),
            "iterations_attempted": int(attempted),
            "stop_reason": stop_reason,
            "stop_reason_inferred": stop_reason is not None,
        }
        _record_comparator_metadata(self, metadata)


def _longest_metric_history(value: Any) -> int:
    if isinstance(value, Mapping):
        return max((_longest_metric_history(item) for item in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


class _CatBoostTelemetryMixin:
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        time_limit = _fit_argument(
            args, kwargs, name="time_limit", position=2, default=None
        )
        num_cpus = _fit_argument(
            args, kwargs, name="num_cpus", position=4, default=-1
        )
        num_gpus = _fit_argument(
            args, kwargs, name="num_gpus", position=3, default=0
        )
        requested_params = dict(self._get_model_params())
        super()._fit(X, y, *args, **kwargs)

        resolved_params = dict(self.model.get_all_params())
        requested = int(requested_params["iterations"])
        tree_count = int(self.model.tree_count_)
        raw_best = int(self.model.get_best_iteration())
        best_iteration = raw_best + 1 if raw_best >= 0 else tree_count
        attempted = _longest_metric_history(self.model.get_evals_result())
        if attempted == 0:
            attempted = tree_count
        # AutoGluon's official adapter forwards this fitted child allocation as
        # CatBoost's ``thread_count`` but CatBoost omits it from
        # ``get_all_params()``.  Retain the exact effective value explicitly.
        resolved_params["thread_count"] = int(num_cpus)
        resolved_learning_rate = float(resolved_params["learning_rate"])
        if attempted >= requested:
            stop_reason = "iteration_limit"
        elif time_limit is None and len(X) * int(X.shape[1]) <= 5_000_000:
            stop_reason = "early_stopping"
        else:
            # Official AutoGluon callbacks do not persist which of early stop,
            # time, or memory requested termination.
            stop_reason = None

        metadata = {
            "schema_version": 1,
            "engine": "catboost",
            "iterations_requested": requested,
            "best_iteration": int(best_iteration),
            "rounds_retained": tree_count,
            "resolved_params": _safe_mapping(
                resolved_params, field="comparator_fit.resolved_params"
            ),
            "num_cpus": int(num_cpus),
            "num_gpus": float(num_gpus),
            "tree_count": tree_count,
            "catboost_best_iteration_zero_based": raw_best,
            "resolved_learning_rate": resolved_learning_rate,
            "iterations_attempted": int(attempted),
            "stop_reason": stop_reason,
            "stop_reason_inferred": stop_reason is not None,
        }
        _record_comparator_metadata(self, metadata)


class ComparatorDarkoFitModel(
    _NativeRepresentationMixin,
    _DarkoFitTelemetryMixin,
    DarkoFitModel,
):
    """Official DarkoFit TabArena defaults plus comparator telemetry."""


class ComparatorOrdinalDarkoFitModel(
    _SafeOrdinalRepresentationMixin,
    _DarkoFitTelemetryMixin,
    DarkoFitModel,
):
    """DarkoFit with the source-frozen safe ordinal representation."""


class ComparatorChimeraBoostModel(
    _NativeRepresentationMixin,
    _ChimeraBoostTelemetryMixin,
    ChimeraBoostModel,
):
    """Official TabArena ChimeraBoost defaults plus comparator telemetry."""


class ComparatorOrdinalChimeraBoostModel(
    _SafeOrdinalRepresentationMixin,
    _ChimeraBoostTelemetryMixin,
    ChimeraBoostModel,
):
    """ChimeraBoost with the source-frozen safe ordinal representation."""


class ComparatorCatBoostModel(
    _NativeRepresentationMixin,
    _CatBoostTelemetryMixin,
    CatBoostModel,
):
    """Official AutoGluon CatBoost defaults plus comparator telemetry."""


class ComparatorOrdinalCatBoostModel(
    _SafeOrdinalRepresentationMixin,
    _CatBoostTelemetryMixin,
    CatBoostModel,
):
    """CatBoost with the source-frozen safe ordinal representation."""


__all__ = [
    "COMPARATOR_METADATA_KEY",
    "ComparatorCatBoostModel",
    "ComparatorChimeraBoostModel",
    "ComparatorDarkoFitModel",
    "ComparatorOrdinalCatBoostModel",
    "ComparatorOrdinalChimeraBoostModel",
    "ComparatorOrdinalDarkoFitModel",
    "REPRESENTATION_METADATA_KEY",
]
