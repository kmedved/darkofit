"""Scoring adapters for stepwise tuning."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
from sklearn.base import is_classifier
from sklearn.metrics import get_scorer, log_loss, mean_pinball_loss


@dataclass
class ResolvedScorer:
    name: str
    greater_is_better: bool
    kind: str
    scorer: object = None

    def __call__(self, estimator, X, y, sample_weight=None):
        score = _score_by_kind(self, estimator, X, y, sample_weight)
        loss = -score if self.greater_is_better else score
        return float(score), float(loss)


def resolve_scorer(estimator, scoring=None, greater_is_better=None):
    if scoring is None:
        return _default_scorer(estimator)

    if isinstance(scoring, str):
        gib = True if greater_is_better is None else bool(greater_is_better)
        return ResolvedScorer(scoring, gib, "sklearn")

    gib = True if greater_is_better is None else bool(greater_is_better)
    return ResolvedScorer(
        getattr(scoring, "__name__", "custom_scorer"), gib, "custom", scoring
    )


def score_estimator(estimator, scorer, X_valid, y_valid, sample_weight_valid=None):
    return scorer(estimator, X_valid, y_valid, sample_weight_valid)


def _default_scorer(estimator):
    if is_classifier(estimator):
        return ResolvedScorer("neg_log_loss", True, "default")

    loss = getattr(estimator, "loss", "RMSE")
    if loss == "MAE":
        return ResolvedScorer("neg_mean_absolute_error", True, "default")
    if loss == "Quantile":
        return ResolvedScorer("neg_mean_pinball_loss", True, "default")
    return ResolvedScorer("neg_root_mean_squared_error", True, "default")


def _score_by_kind(resolved, estimator, X, y, sample_weight):
    if resolved.kind == "default":
        if resolved.name == "neg_log_loss":
            return _classifier_neg_log_loss(estimator, X, y, sample_weight)
        if resolved.name == "neg_mean_absolute_error":
            return _neg_mae(estimator, X, y, sample_weight)
        if resolved.name == "neg_mean_pinball_loss":
            return _neg_pinball(estimator, X, y, sample_weight)
        return _neg_rmse(estimator, X, y, sample_weight)
    if resolved.kind == "sklearn":
        scorer = get_scorer(resolved.name)
        return scorer(estimator, X, y, sample_weight=sample_weight)
    if resolved.kind == "custom":
        return _call_custom_scorer(resolved.scorer, estimator, X, y, sample_weight)
    raise ValueError(f"unknown scorer kind {resolved.kind!r}")


def _classifier_neg_log_loss(estimator, X, y, sample_weight):
    proba = estimator.predict_proba(X)
    labels = getattr(estimator, "classes_", None)
    return -log_loss(y, proba, sample_weight=sample_weight, labels=labels)


def _neg_rmse(estimator, X, y, sample_weight):
    resid = np.asarray(y, dtype=np.float64) - estimator.predict(X)
    mse = np.average(resid * resid, weights=sample_weight)
    return -float(np.sqrt(mse))


def _neg_mae(estimator, X, y, sample_weight):
    err = np.abs(np.asarray(y, dtype=np.float64) - estimator.predict(X))
    return -float(np.average(err, weights=sample_weight))


def _neg_pinball(estimator, X, y, sample_weight):
    alpha = getattr(estimator, "alpha", 0.5)
    return -float(mean_pinball_loss(
        y, estimator.predict(X), alpha=alpha, sample_weight=sample_weight
    ))


def _call_custom_scorer(scoring, estimator, X, y, sample_weight):
    try:
        signature = inspect.signature(scoring)
    except (TypeError, ValueError):
        return scoring(estimator, X, y, sample_weight=sample_weight)

    params = signature.parameters
    values = params.values()
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in values)
    if "sample_weight" in params or accepts_kwargs:
        return scoring(estimator, X, y, sample_weight=sample_weight)
    positional = [
        p for p in params.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) >= 4:
        return scoring(estimator, X, y, sample_weight)
    return scoring(estimator, X, y)
