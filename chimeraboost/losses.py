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


def _sigmoid(z):
    # Numerically stable logistic.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class RMSE:
    """Squared-error regression. grad = pred - y, hess = 1."""

    name = "RMSE"
    is_classification = False
    adjusts_leaves = False

    def init(self, y):
        return float(np.mean(y))

    def grad_hess(self, y, raw):
        grad = raw - y
        hess = np.ones_like(raw)
        return grad, hess

    def eval(self, y, raw):
        return float(np.sqrt(np.mean((raw - y) ** 2)))

    def transform(self, raw):
        return raw


class Logloss:
    """Binary cross-entropy. raw = log-odds, p = sigmoid(raw)."""

    name = "Logloss"
    is_classification = True
    adjusts_leaves = False

    def init(self, y):
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    def grad_hess(self, y, raw):
        p = _sigmoid(raw)
        grad = p - y
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess

    def eval(self, y, raw):
        p = np.clip(_sigmoid(raw), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

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

    def leaf_value(self, residuals):
        return float(np.median(residuals)) if residuals.size else 0.0

    def init(self, y):
        return float(np.median(y))

    def grad_hess(self, y, raw):
        grad = np.sign(raw - y)
        hess = np.ones_like(raw)
        return grad, hess

    def eval(self, y, raw):
        return float(np.mean(np.abs(raw - y)))

    def transform(self, raw):
        return raw


class Quantile:
    """Pinball loss for quantile regression at level `alpha` in (0, 1)."""

    name = "Quantile"
    is_classification = False
    adjusts_leaves = True

    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)

    def leaf_value(self, residuals):
        return float(np.quantile(residuals, self.alpha)) if residuals.size else 0.0

    def init(self, y):
        return float(np.quantile(y, self.alpha))

    def grad_hess(self, y, raw):
        a = self.alpha
        grad = np.where(y >= raw, -a, 1.0 - a)
        hess = np.ones_like(raw)
        return grad, hess

    def eval(self, y, raw):
        r = y - raw
        return float(np.mean(np.maximum(self.alpha * r, (self.alpha - 1.0) * r)))

    def transform(self, raw):
        return raw


def _softmax(F):
    z = F - F.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


class MultiSoftmax:
    """Multinomial logistic loss. Operates on raw scores F of shape (n, K)."""

    name = "MultiClass"
    is_classification = True

    def __init__(self, n_classes):
        self.K = int(n_classes)

    def init(self, Y):  # Y one-hot (n, K)
        p = np.clip(Y.mean(axis=0), 1e-6, 1.0)
        return np.log(p)  # (K,)

    def grad_hess(self, Y, F):  # F (n, K)
        P = _softmax(F)
        grad = P - Y
        hess = np.maximum(P * (1.0 - P), 1e-6)
        return grad, hess

    def eval(self, Y, F):
        P = np.clip(_softmax(F), 1e-12, 1.0)
        return float(-np.mean(np.sum(Y * np.log(P), axis=1)))

    def transform(self, F):
        return _softmax(F)


LOSSES = {"RMSE": RMSE, "Logloss": Logloss, "MAE": MAE, "Quantile": Quantile}

