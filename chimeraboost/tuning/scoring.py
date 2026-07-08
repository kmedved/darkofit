"""Scoring adapters for stepwise tuning."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import numpy as np
from sklearn.base import is_classifier
from sklearn.metrics import get_scorer, log_loss, mean_pinball_loss

from ..losses import VECTOR_LOSSES


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
        if scoring in {
            "neg_gaussian_nll",
            "neg_lognormal_nll",
            "neg_student_t_nll",
            "neg_poisson_nll",
            "neg_negative_binomial_nll",
            "neg_distributional_nll",
        }:
            return ResolvedScorer(scoring, gib, "default")
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
    if loss in VECTOR_LOSSES:
        default = {
            "Gaussian": "neg_gaussian_nll",
            "LogNormal": "neg_lognormal_nll",
            "StudentT": "neg_student_t_nll",
            "Poisson": "neg_poisson_nll",
            "NegativeBinomial": "neg_negative_binomial_nll",
        }.get(loss, "neg_distributional_nll")
        return ResolvedScorer(default, True, "default")
    if loss == "MAE":
        return ResolvedScorer("neg_mean_absolute_error", True, "default")
    if loss == "Quantile":
        return ResolvedScorer("neg_mean_pinball_loss", True, "default")
    return ResolvedScorer("neg_root_mean_squared_error", True, "default")


def _score_by_kind(resolved, estimator, X, y, sample_weight):
    if resolved.kind == "default":
        if resolved.name == "neg_log_loss":
            return _classifier_neg_log_loss(estimator, X, y, sample_weight)
        if resolved.name == "neg_gaussian_nll":
            return _neg_gaussian_nll(estimator, X, y, sample_weight)
        if resolved.name == "neg_lognormal_nll":
            return _neg_lognormal_nll(estimator, X, y, sample_weight)
        if resolved.name == "neg_student_t_nll":
            return _neg_student_t_nll(estimator, X, y, sample_weight)
        if resolved.name == "neg_poisson_nll":
            return _neg_poisson_nll(estimator, X, y, sample_weight)
        if resolved.name == "neg_negative_binomial_nll":
            return _neg_negative_binomial_nll(estimator, X, y, sample_weight)
        if resolved.name == "neg_distributional_nll":
            return _neg_distributional_nll(estimator, X, y, sample_weight)
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


def _neg_gaussian_nll(estimator, X, y, sample_weight):
    mu, sigma = estimator.predict_dist(X)
    y_arr = np.asarray(y, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), np.exp(-15.0))
    rho = np.log(sigma)
    z = np.clip(
        (y_arr - np.asarray(mu, dtype=np.float64)) / sigma,
        -1000.0,
        1000.0,
    )
    nll = rho + 0.5 * z * z + 0.5 * np.log(2.0 * np.pi)
    return -float(np.average(nll, weights=sample_weight))


def _neg_distributional_nll(estimator, X, y, sample_weight):
    model = getattr(estimator, "model_", None)
    loss = getattr(model, "loss_", None)
    if loss is None:
        raise ValueError("distributional scoring requires a fitted model")
    loss_name = getattr(loss, "name", getattr(estimator, "loss", None))
    if loss_name == "Gaussian":
        return _neg_gaussian_nll(estimator, X, y, sample_weight)
    if loss_name == "LogNormal":
        return _neg_lognormal_nll(estimator, X, y, sample_weight)
    if loss_name == "StudentT":
        return _neg_student_t_nll(estimator, X, y, sample_weight)
    if loss_name == "Poisson":
        return _neg_poisson_nll(estimator, X, y, sample_weight)
    if loss_name == "NegativeBinomial":
        return _neg_negative_binomial_nll(estimator, X, y, sample_weight)
    raw = model.predict_raw(X)
    F = np.ascontiguousarray(raw.T, dtype=np.float64)
    return -float(loss.eval_class_major(np.asarray(y, dtype=np.float64), F, sample_weight))


def _neg_lognormal_nll(estimator, X, y, sample_weight):
    m, s = estimator.predict_dist(X)
    y_arr = np.asarray(y, dtype=np.float64)
    s = np.maximum(np.asarray(s, dtype=np.float64), np.exp(-15.0))
    u = np.log(y_arr)
    z = np.clip((u - np.asarray(m, dtype=np.float64)) / s, -1000.0, 1000.0)
    nll = np.log(s) + 0.5 * z * z + 0.5 * np.log(2.0 * np.pi) + u
    return -float(np.average(nll, weights=sample_weight))


def _neg_student_t_nll(estimator, X, y, sample_weight):
    mu, scale, nu_arr = estimator.predict_dist(X)
    y_arr = np.asarray(y, dtype=np.float64)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), np.exp(-15.0))
    nu = float(np.asarray(nu_arr, dtype=np.float64)[0])
    const = (
        0.5 * np.log(nu * np.pi)
        + math.lgamma(nu / 2.0)
        - math.lgamma((nu + 1.0) / 2.0)
    )
    z = np.clip((y_arr - np.asarray(mu, dtype=np.float64)) / scale, -1e150, 1e150)
    nll = np.log(scale) + const + 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
    return -float(np.average(nll, weights=sample_weight))


def _neg_poisson_nll(estimator, X, y, sample_weight):
    (lam,) = estimator.predict_dist(X)
    y_arr = np.asarray(y, dtype=np.float64)
    lam = np.maximum(np.asarray(lam, dtype=np.float64), np.exp(-15.0))
    eta = np.log(lam)
    lgamma = np.fromiter(
        (math.lgamma(float(v) + 1.0) for v in y_arr),
        dtype=np.float64,
        count=y_arr.size,
    )
    nll = lam - y_arr * eta + lgamma
    return -float(np.average(nll, weights=sample_weight))


def _neg_negative_binomial_nll(estimator, X, y, sample_weight):
    mu, alpha = estimator.predict_dist(X)
    y_arr = np.asarray(y, dtype=np.float64)
    mu = np.maximum(np.asarray(mu, dtype=np.float64), np.exp(-15.0))
    alpha = np.maximum(np.asarray(alpha, dtype=np.float64), 1e-12)
    r = 1.0 / alpha
    log_r = np.log(r)
    log_r_mu = np.log(r + mu)
    lgamma_y_r = np.fromiter(
        (math.lgamma(float(yi + ri)) for yi, ri in zip(y_arr, r)),
        dtype=np.float64,
        count=y_arr.size,
    )
    lgamma_r = np.fromiter(
        (math.lgamma(float(ri)) for ri in r),
        dtype=np.float64,
        count=y_arr.size,
    )
    lgamma_y = np.fromiter(
        (math.lgamma(float(yi) + 1.0) for yi in y_arr),
        dtype=np.float64,
        count=y_arr.size,
    )
    nll = (
        -lgamma_y_r
        + lgamma_r
        + lgamma_y
        - r * (log_r - log_r_mu)
        - y_arr * (np.log(mu) - log_r_mu)
    )
    return -float(np.average(nll, weights=sample_weight))


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
