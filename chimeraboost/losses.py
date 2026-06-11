"""Loss functions for ChimeraBoost.

Each loss provides:
  init(y)            -> scalar raw score to start every prediction from
  grad_hess(y, raw)  -> (gradient, hessian) of the loss wrt the raw score
  eval(y, raw)       -> scalar loss value (for early stopping / logging)

Raw scores are the additive model output *before* any link function.
For regression the raw score is the prediction itself; for binary
classification it is the log-odds, turned into a probability by a sigmoid.
"""

import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------
# Weighted statistical helpers
# --------------------------------------------------------------------------

def _weighted_median(values, weights):
    """Weighted nearest-rank median.

    When *weights* is None this delegates to ``np.median`` so the
    ``sample_weight=None`` code path is bitwise identical to the original.
    """
    if weights is None:
        return float(np.median(values)) if values.size else 0.0
    if not values.size:
        return 0.0
    order = np.argsort(values)
    sv, sw = values[order], weights[order]
    cumw = np.cumsum(sw)
    idx = min(int(np.searchsorted(cumw, cumw[-1] * 0.5)), len(sv) - 1)
    return float(sv[idx])


def _weighted_quantile(values, weights, alpha):
    """Weighted nearest-rank quantile at level *alpha*.

    When *weights* is None this delegates to ``np.quantile`` so the
    ``sample_weight=None`` code path is bitwise identical to the original.
    """
    if weights is None:
        return float(np.quantile(values, alpha)) if values.size else 0.0
    if not values.size:
        return 0.0
    order = np.argsort(values)
    sv, sw = values[order], weights[order]
    cumw = np.cumsum(sw)
    idx = min(int(np.searchsorted(cumw, cumw[-1] * alpha)), len(sv) - 1)
    return float(sv[idx])


def _sigmoid(z):
    # Numerically stable logistic.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@njit(cache=True, parallel=True)
def _logloss_grad_hess_into(y, raw, sample_weight, grad_out, hess_out):
    for i in prange(raw.shape[0]):
        z = raw[i]
        if z >= 0.0:
            p = 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            p = ez / (1.0 + ez)
        grad = p - y[i]
        hess = p * (1.0 - p)
        if hess < 1e-6:
            hess = 1e-6
        if sample_weight is not None:
            w = sample_weight[i]
            grad *= w
            hess *= w
        grad_out[i] = grad
        hess_out[i] = hess


@njit(cache=True, parallel=True)
def _logloss_eval(y, raw, sample_weight):
    # The reduction variables are updated exactly once per iteration,
    # unconditionally: numba's parfor reduction analysis is fragile with
    # reductions inside branches (the "unexpected cycle in lookup()"
    # assertion), and 1.0 * ce == ce keeps the unweighted sums bitwise
    # identical to the plain accumulation.
    total = 0.0
    weight_total = 0.0
    for i in prange(raw.shape[0]):
        z = raw[i]
        if z >= 0.0:
            p = 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            p = ez / (1.0 + ez)
        if p < 1e-9:
            p = 1e-9
        elif p > 1.0 - 1e-9:
            p = 1.0 - 1e-9
        ce = -(y[i] * np.log(p) + (1.0 - y[i]) * np.log(1.0 - p))
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
        total += w * ce
        weight_total += w
    return total / weight_total


class RMSE:
    """Squared-error regression. grad = pred - y, hess = 1."""

    name = "RMSE"
    is_classification = False
    adjusts_leaves = False
    constant_hessian = True

    def init(self, y, sample_weight=None):
        return float(np.average(y, weights=sample_weight))

    def grad_hess(self, y, raw):
        grad = raw - y
        hess = np.ones_like(raw)
        return grad, hess

    def grad_hess_into(self, y, raw, sample_weight, grad_out, hess_out):
        np.subtract(raw, y, out=grad_out)
        hess_out.fill(1.0)
        if sample_weight is not None:
            grad_out *= sample_weight
            hess_out *= sample_weight

    def eval(self, y, raw, sample_weight=None):
        return float(np.sqrt(np.average((raw - y) ** 2, weights=sample_weight)))

    def transform(self, raw):
        return raw


class Logloss:
    """Binary cross-entropy. raw = log-odds, p = sigmoid(raw)."""

    name = "Logloss"
    is_classification = True
    adjusts_leaves = False
    constant_hessian = False

    def init(self, y, sample_weight=None):
        p = np.clip(np.average(y, weights=sample_weight), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    def grad_hess(self, y, raw):
        p = _sigmoid(raw)
        grad = p - y
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess

    def grad_hess_into(self, y, raw, sample_weight, grad_out, hess_out):
        _logloss_grad_hess_into(y, raw, sample_weight, grad_out, hess_out)

    def eval(self, y, raw, sample_weight=None):
        return float(_logloss_eval(y, raw, sample_weight))

    def transform(self, raw):
        return _sigmoid(raw)


class MAE:
    """Mean absolute error. grad = sign(pred - y), hess = 1 (constant).

    Uses the gradient-step approximation for leaf values rather than exact
    medians, which is simple and converges fine with enough small steps.
    """

    name = "MAE"
    is_classification = False
    adjusts_leaves = True   # sign gradients only pick structure; set leaf = median
    constant_hessian = True

    def leaf_value(self, residuals, weights=None):
        return _weighted_median(residuals, weights)

    def init(self, y, sample_weight=None):
        return _weighted_median(y, sample_weight)

    def grad_hess(self, y, raw):
        grad = np.sign(raw - y)
        hess = np.ones_like(raw)
        return grad, hess

    def grad_hess_into(self, y, raw, sample_weight, grad_out, hess_out):
        np.sign(raw - y, out=grad_out)
        hess_out.fill(1.0)
        if sample_weight is not None:
            grad_out *= sample_weight
            hess_out *= sample_weight

    def eval(self, y, raw, sample_weight=None):
        return float(np.average(np.abs(raw - y), weights=sample_weight))

    def transform(self, raw):
        return raw


class Quantile:
    """Pinball loss for quantile regression at level `alpha` in (0, 1)."""

    name = "Quantile"
    is_classification = False
    adjusts_leaves = True
    constant_hessian = True

    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)

    def leaf_value(self, residuals, weights=None):
        return _weighted_quantile(residuals, weights, self.alpha)

    def init(self, y, sample_weight=None):
        return _weighted_quantile(y, sample_weight, self.alpha)

    def grad_hess(self, y, raw):
        a = self.alpha
        grad = np.where(y >= raw, -a, 1.0 - a)
        hess = np.ones_like(raw)
        return grad, hess

    def grad_hess_into(self, y, raw, sample_weight, grad_out, hess_out):
        a = self.alpha
        grad_out[:] = np.where(y >= raw, -a, 1.0 - a)
        hess_out.fill(1.0)
        if sample_weight is not None:
            grad_out *= sample_weight
            hess_out *= sample_weight

    def eval(self, y, raw, sample_weight=None):
        r = y - raw
        pinball = np.maximum(self.alpha * r, (self.alpha - 1.0) * r)
        return float(np.average(pinball, weights=sample_weight))

    def transform(self, raw):
        return raw


def _softmax(F):
    z = F - F.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def _softmax_class_major(F):
    z = F - F.max(axis=0, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=0, keepdims=True)


@njit(cache=True, parallel=True)
def _softmax_class_major_grad_hess_into(Y, F, sample_weight, grad_out, hess_out):
    K, n = F.shape
    for i in prange(n):
        max_f = F[0, i]
        for k in range(1, K):
            if F[k, i] > max_f:
                max_f = F[k, i]

        denom = 0.0
        for k in range(K):
            p = np.exp(F[k, i] - max_f)
            grad_out[k, i] = p
            denom += p

        for k in range(K):
            p = grad_out[k, i] / denom
            grad = p - Y[k, i]
            hess = p * (1.0 - p)
            if hess < 1e-6:
                hess = 1e-6
            if sample_weight is not None:
                w = sample_weight[i]
                grad *= w
                hess *= w
            grad_out[k, i] = grad
            hess_out[k, i] = hess


@njit(cache=True)
def _softmax_class_major_eval(Y, F, sample_weight):
    K, n = F.shape
    total = 0.0
    weight_total = 0.0
    for i in range(n):
        max_f = F[0, i]
        for k in range(1, K):
            if F[k, i] > max_f:
                max_f = F[k, i]

        denom = 0.0
        true_exp = 0.0
        for k in range(K):
            e = np.exp(F[k, i] - max_f)
            denom += e
            if Y[k, i] != 0.0:
                true_exp = e

        p = true_exp / denom
        if p < 1e-12:
            p = 1e-12
        elif p > 1.0:
            p = 1.0
        ce = -np.log(p)
        # Unconditional reduction updates; see _logloss_eval.
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
        total += w * ce
        weight_total += w
    return total / weight_total


@njit(cache=True)
def _softmax_class_major_eval_labels(labels, F, sample_weight):
    K, n = F.shape
    total = 0.0
    weight_total = 0.0
    for i in range(n):
        max_f = F[0, i]
        for k in range(1, K):
            if F[k, i] > max_f:
                max_f = F[k, i]

        denom = 0.0
        true_label = labels[i]
        true_exp = 0.0
        for k in range(K):
            e = np.exp(F[k, i] - max_f)
            denom += e
            if k == true_label:
                true_exp = e

        p = true_exp / denom
        if p < 1e-12:
            p = 1e-12
        elif p > 1.0:
            p = 1.0
        ce = -np.log(p)
        # Unconditional reduction updates; see _logloss_eval.
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
        total += w * ce
        weight_total += w
    return total / weight_total


class MultiSoftmax:
    """Multinomial logistic loss. Operates on raw scores F of shape (n, K)."""

    name = "MultiClass"
    is_classification = True
    constant_hessian = False

    def __init__(self, n_classes):
        self.K = int(n_classes)

    def init(self, Y, sample_weight=None):  # Y one-hot (n, K)
        p = np.clip(np.average(Y, axis=0, weights=sample_weight), 1e-6, 1.0)
        return np.log(p)  # (K,)

    def init_class_major(self, Y, sample_weight=None):  # Y one-hot (K, n)
        p = np.clip(np.average(Y, axis=1, weights=sample_weight), 1e-6, 1.0)
        return np.log(p)  # (K,)

    def grad_hess(self, Y, F):  # F (n, K)
        P = _softmax(F)
        grad = P - Y
        hess = np.maximum(P * (1.0 - P), 1e-6)
        return grad, hess

    def grad_hess_class_major(self, Y, F):  # F/Y (K, n)
        P = _softmax_class_major(F)
        grad = P - Y
        hess = np.maximum(P * (1.0 - P), 1e-6)
        return grad, hess

    def grad_hess_class_major_into(self, Y, F, sample_weight, grad_out, hess_out):
        _softmax_class_major_grad_hess_into(
            Y, F, sample_weight, grad_out, hess_out
        )

    def eval(self, Y, F, sample_weight=None):
        P = np.clip(_softmax(F), 1e-12, 1.0)
        row_ce = -np.sum(Y * np.log(P), axis=1)
        return float(np.average(row_ce, weights=sample_weight))

    def eval_class_major(self, Y, F, sample_weight=None):
        return float(_softmax_class_major_eval(Y, F, sample_weight))

    def eval_class_major_labels(self, labels, F, sample_weight=None):
        return float(_softmax_class_major_eval_labels(labels, F, sample_weight))

    def transform(self, F):
        return _softmax(F)


LOSSES = {"RMSE": RMSE, "Logloss": Logloss, "MAE": MAE, "Quantile": Quantile}
