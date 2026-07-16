"""CTR23-only comparator adapters with observable time callbacks.

The existing comparator adapters intentionally keep their version-1 fitted
metadata stable for the published same-machine campaign.  This module adds a
separate, campaign-specific audit sidecar without changing model parameters,
callback return values, or the legacy ``comparator_fit`` schema.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from autogluon.tabular.models import CatBoostModel

try:
    from benchmarks.tabarena_comparator_adapters import (
        ComparatorCatBoostModel,
        ComparatorChimeraBoostModel,
        _fit_argument,
        _safe_mapping,
    )
except ModuleNotFoundError:  # Direct execution from ``benchmarks``.
    from tabarena_comparator_adapters import (
        ComparatorCatBoostModel,
        ComparatorChimeraBoostModel,
        _fit_argument,
        _safe_mapping,
    )

if TYPE_CHECKING:
    import pandas as pd


CTR23_TIME_CALLBACK_AUDIT_KEY = "ctr23_time_callback_audit"
_AUDIT_KIND = "darkofit_ctr23_time_callback_audit"


def _positive_time_limit(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("CTR23 comparator time limit must be finite and positive")
    return result


def _record_time_callback_audit(
    model: Any,
    *,
    engine: str,
    time_limit: float | None,
    instrumented: bool,
    instance_count: int,
    call_count: int,
    hit: bool,
) -> None:
    metadata = _safe_mapping(
        {
            "schema_version": 1,
            "kind": _AUDIT_KIND,
            "engine": engine,
            "time_limit_seconds": time_limit,
            "time_callback_instrumented": instrumented,
            "time_callback_instance_count": instance_count,
            "time_callback_call_count": call_count,
            "time_callback_hit": hit,
        },
        field=CTR23_TIME_CALLBACK_AUDIT_KEY,
    )
    model._fit_metadata[CTR23_TIME_CALLBACK_AUDIT_KEY] = metadata


class _CTR23ChimeraTimeAuditMixin:
    """Observe the exact callback shared by ChimeraBoost lane candidates."""

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        time_limit = _positive_time_limit(
            _fit_argument(
                args,
                kwargs,
                name="time_limit",
                position=2,
                default=None,
            )
        )
        audit = {"instrumented": False, "calls": 0, "hit": False}
        model_ids: set[int] = set()
        patched_cls = None
        original_fit = None
        if time_limit is not None:
            from chimeraboost import ChimeraBoostRegressor

            patched_cls = ChimeraBoostRegressor
            original_fit = ChimeraBoostRegressor.fit

            def audited_fit(model, *fit_args: Any, **fit_kwargs: Any):
                callback = fit_kwargs.get("callbacks")
                if callback is None:
                    raise RuntimeError(
                        "ChimeraBoost time-limit callback was not installed"
                    )
                audit["instrumented"] = True

                def audited_callback(*callback_args: Any, **callback_kwargs: Any):
                    result = callback(*callback_args, **callback_kwargs)
                    audit["calls"] += 1
                    if len(callback_args) >= 4:
                        model_ids.add(id(callback_args[3]))
                    if bool(result):
                        audit["hit"] = True
                    return result

                fit_kwargs["callbacks"] = audited_callback
                return original_fit(model, *fit_args, **fit_kwargs)

            ChimeraBoostRegressor.fit = audited_fit
        try:
            super()._fit(X, y, *args, **kwargs)
        finally:
            if patched_cls is not None:
                patched_cls.fit = original_fit

        if time_limit is not None and not audit["instrumented"]:
            raise RuntimeError("ChimeraBoost time callback was not instrumented")
        _record_time_callback_audit(
            self,
            engine="chimeraboost",
            time_limit=time_limit,
            instrumented=bool(audit["instrumented"]),
            instance_count=len(model_ids),
            call_count=int(audit["calls"]),
            hit=bool(audit["hit"]),
        )


class _CTR23CatBoostTimeAuditMixin:
    """Observe AutoGluon's exact CatBoost time callback binding."""

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        time_limit = _positive_time_limit(
            _fit_argument(
                args,
                kwargs,
                name="time_limit",
                position=2,
                default=None,
            )
        )
        audit = {
            "instrumented": False,
            "instances": 0,
            "calls": 0,
            "hit": False,
        }
        fit_globals = CatBoostModel._fit.__globals__
        original_callback_cls = None
        if time_limit is not None:
            original_callback_cls = fit_globals.get("TimeCheckCallback")
            if not isinstance(original_callback_cls, type):
                raise RuntimeError("CatBoost time callback class is unavailable")

            class AuditedTimeCheckCallback(original_callback_cls):
                def __init__(self, *callback_args: Any, **callback_kwargs: Any):
                    super().__init__(*callback_args, **callback_kwargs)
                    audit["instrumented"] = True
                    audit["instances"] += 1

                def after_iteration(self, info):
                    result = super().after_iteration(info)
                    audit["calls"] += 1
                    if not bool(result):
                        audit["hit"] = True
                    return result

            fit_globals["TimeCheckCallback"] = AuditedTimeCheckCallback
        try:
            super()._fit(X, y, *args, **kwargs)
        finally:
            if original_callback_cls is not None:
                fit_globals["TimeCheckCallback"] = original_callback_cls

        if time_limit is not None and (
            not audit["instrumented"]
            or audit["instances"] != 1
            or audit["calls"] < 1
        ):
            raise RuntimeError("CatBoost time callback was not instrumented")
        _record_time_callback_audit(
            self,
            engine="catboost",
            time_limit=time_limit,
            instrumented=bool(audit["instrumented"]),
            instance_count=int(audit["instances"]),
            call_count=int(audit["calls"]),
            hit=bool(audit["hit"]),
        )


class CTR23ComparatorChimeraBoostModel(
    _CTR23ChimeraTimeAuditMixin,
    ComparatorChimeraBoostModel,
):
    """ChimeraBoost default plus CTR23-only callback observability."""


class CTR23ComparatorCatBoostModel(
    _CTR23CatBoostTimeAuditMixin,
    ComparatorCatBoostModel,
):
    """CatBoost default plus CTR23-only callback observability."""


__all__ = [
    "CTR23ComparatorCatBoostModel",
    "CTR23ComparatorChimeraBoostModel",
    "CTR23_TIME_CALLBACK_AUDIT_KEY",
]
