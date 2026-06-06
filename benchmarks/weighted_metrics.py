"""Weighted holdout metrics shared by revision benchmarks.

The revision-comparison harness reports both ordinary and weighted metrics. The
weighted versions use the exact same prediction arrays, but pass sample weights
through sklearn metrics (or an equivalent direct formula) so benchmark decisions
can be based on the intended row importance instead of only raw row counts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _as_weight(sample_weight):
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.size == 0:
        return None
    return w


def _normalize_proba(proba):
    p = np.asarray(proba, dtype=np.float64)
    p = np.clip(p, 1e-15, 1.0)
    row_sum = p.sum(axis=1, keepdims=True)
    return p / row_sum


def _multiclass_brier(y_true, proba, labels, sample_weight=None):
    labels = np.asarray(labels)
    y_true = np.asarray(y_true)
    p = _normalize_proba(proba)
    index = {label: i for i, label in enumerate(labels)}
    y_onehot = np.zeros_like(p, dtype=np.float64)
    for i, y in enumerate(y_true):
        if y in index:
            y_onehot[i, index[y]] = 1.0
    row_loss = np.sum((p - y_onehot) ** 2, axis=1)
    return float(np.average(row_loss, weights=_as_weight(sample_weight)))


def regression_metrics(y_true, pred, sample_weight=None) -> dict[str, float]:
    """Return RMSE/MAE/R2, optionally weighted."""
    w = _as_weight(sample_weight)
    mse = mean_squared_error(y_true, pred, sample_weight=w)
    return {
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, pred, sample_weight=w)),
        "r2": float(r2_score(y_true, pred, sample_weight=w)),
    }


def quantile_metrics(y_true, pred, alpha, sample_weight=None) -> dict[str, float]:
    """Return pinball loss and empirical coverage for one quantile level."""
    w = _as_weight(sample_weight)
    y_true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = y_true - pred
    pinball = np.maximum(alpha * err, (alpha - 1.0) * err)
    covered = (y_true <= pred).astype(np.float64)
    return {
        "pinball": float(np.average(pinball, weights=w)),
        "coverage": float(np.average(covered, weights=w)),
    }


def classification_metrics(
    y_true,
    pred,
    proba,
    *,
    labels: Any = None,
    sample_weight=None,
) -> dict[str, float]:
    """Return accuracy/F1/log-loss/Brier, optionally weighted.

    ``labels`` should match the probability-column order. Passing the fitted
    estimator's ``classes_`` avoids ambiguous log-loss behavior when a small test
    split happens to omit one class.
    """
    w = _as_weight(sample_weight)
    labels = np.asarray(labels if labels is not None else np.unique(y_true))
    p = _normalize_proba(proba)
    return {
        "accuracy": float(accuracy_score(y_true, pred, sample_weight=w)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", sample_weight=w)),
        "log_loss": float(log_loss(y_true, p, labels=labels, sample_weight=w)),
        "brier": _multiclass_brier(y_true, p, labels, sample_weight=w),
    }


def metric_bundle(
    task,
    y_true,
    pred,
    proba=None,
    labels=None,
    sample_weight=None,
    alpha=None,
):
    """Return unweighted and weighted metrics in flat CSV-friendly keys."""
    if task == "regression":
        unweighted = regression_metrics(y_true, pred)
        weighted = regression_metrics(y_true, pred, sample_weight=sample_weight)
        primary = "weighted_rmse" if _as_weight(sample_weight) is not None else "rmse"
        primary_value = weighted["rmse"] if primary == "weighted_rmse" else unweighted["rmse"]
    elif task == "quantile":
        if alpha is None:
            raise ValueError("quantile metrics require alpha")
        unweighted = quantile_metrics(y_true, pred, float(alpha))
        weighted = quantile_metrics(
            y_true, pred, float(alpha), sample_weight=sample_weight)
        primary = (
            "weighted_pinball"
            if _as_weight(sample_weight) is not None
            else "pinball"
        )
        primary_value = (
            weighted["pinball"] if primary == "weighted_pinball"
            else unweighted["pinball"]
        )
    else:
        unweighted = classification_metrics(y_true, pred, proba, labels=labels)
        weighted = classification_metrics(
            y_true, pred, proba, labels=labels, sample_weight=sample_weight
        )
        primary = "weighted_log_loss" if _as_weight(sample_weight) is not None else "log_loss"
        primary_value = weighted["log_loss"] if primary == "weighted_log_loss" else unweighted["log_loss"]

    out = {
        "primary_metric": primary,
        "primary_value": float(primary_value),
    }
    out.update(unweighted)
    out.update({f"weighted_{k}": v for k, v in weighted.items()})
    return out
