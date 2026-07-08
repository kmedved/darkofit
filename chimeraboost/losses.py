"""Loss functions for ChimeraBoost.

Each scalar loss provides:
  init(y)            -> scalar raw score to start every prediction from
  grad_hess(y, raw)  -> (gradient, hessian) of the loss wrt the raw score
  eval(y, raw)       -> scalar loss value (for early stopping / logging)

Distributional vector losses registered in ``VECTOR_LOSSES`` provide the
class-major training protocol (``init_class_major``,
``grad_hess_class_major_into``, ``eval_class_major``) plus a prediction
protocol over sample-major ``predict_raw`` output: ``mean_from_raw``,
``params_from_raw``, ``variance_from_raw``, ``interval_from_raw``, and
``sample_from_raw``.  The booster and sklearn wrapper duck-type on those
methods rather than importing a formal ABC.

Raw scores are the additive model output *before* any link function.
For regression the raw score is the prediction itself; for binary
classification it is the log-odds, turned into a probability by a sigmoid.
"""

import math

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
        if not np.isfinite(self.alpha) or not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be finite and in (0, 1)")

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


_GAUSS_RHO_MIN = -15.0
_GAUSS_RHO_MAX = 15.0
_GAUSS_Z_CLIP = 10.0
_GAUSS_EVAL_Z_GUARD = 1000.0
_GAUSS_INIT_RHO_MIN = -10.0
_GAUSS_INIT_RHO_MAX = 10.0
_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)
_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


@njit(cache=True, parallel=True)
def _gaussian_nll_grad_hess_into(y, F, sample_weight, grad_out, hess_out):
    n = F.shape[1]
    for i in prange(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                grad_out[0, i] = 0.0
                grad_out[1, i] = 0.0
                hess_out[0, i] = 0.0
                hess_out[1, i] = 0.0
                continue

        r = F[1, i]
        if r < _GAUSS_RHO_MIN:
            r = _GAUSS_RHO_MIN
        elif r > _GAUSS_RHO_MAX:
            r = _GAUSS_RHO_MAX
        sigma = np.exp(r)
        inv_var = 1.0 / (sigma * sigma)
        z = (y[i] - F[0, i]) / sigma
        if z < -_GAUSS_Z_CLIP:
            z = -_GAUSS_Z_CLIP
        elif z > _GAUSS_Z_CLIP:
            z = _GAUSS_Z_CLIP

        grad_out[0, i] = w * (-(z / sigma))
        hess_out[0, i] = w * inv_var
        grad_out[1, i] = w * (1.0 - z * z)
        hess_out[1, i] = w * 2.0


@njit(cache=True)
def _gaussian_nll_eval(y, F, sample_weight):
    n = F.shape[1]
    total = 0.0
    weight_total = 0.0
    for i in range(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                continue

        r = F[1, i]
        if r < _GAUSS_RHO_MIN:
            r = _GAUSS_RHO_MIN
        elif r > _GAUSS_RHO_MAX:
            r = _GAUSS_RHO_MAX
        sigma = np.exp(r)
        z = (y[i] - F[0, i]) / sigma
        if z < -_GAUSS_EVAL_Z_GUARD:
            z = -_GAUSS_EVAL_Z_GUARD
        elif z > _GAUSS_EVAL_Z_GUARD:
            z = _GAUSS_EVAL_Z_GUARD
        total += w * (_HALF_LOG_2PI + r + 0.5 * z * z)
        weight_total += w
    return total / weight_total


@njit(cache=True)
def _gaussian_crps(y, F, sample_weight):
    n = F.shape[1]
    total = 0.0
    weight_total = 0.0
    for i in range(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                continue

        r = F[1, i]
        if r < _GAUSS_RHO_MIN:
            r = _GAUSS_RHO_MIN
        elif r > _GAUSS_RHO_MAX:
            r = _GAUSS_RHO_MAX
        sigma = np.exp(r)
        z = (y[i] - F[0, i]) / sigma
        phi = math.exp(-0.5 * z * z) * _INV_SQRT_2PI
        cdf = 0.5 * (1.0 + math.erf(z / _SQRT2))
        crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - _INV_SQRT_PI)
        total += w * crps
        weight_total += w
    return total / weight_total


class GaussianNLL:
    """Heteroscedastic Gaussian negative log-likelihood.

    Raw scores are class-major during training: row 0 is the mean and row 1 is
    log standard deviation. Hessians are the Fisher diagonal, so vector-tree
    Newton leaf values are natural-gradient steps.
    """

    name = "Gaussian"
    distribution_name = "gaussian"
    is_classification = False
    adjusts_leaves = False
    constant_hessian = False
    n_outputs = 2
    default_eval_metric = "nll"
    supported_eval_metrics = ("nll", "crps")
    target_standardization = True
    target_standardization_basis = "target"
    calibration_targets = ("scale",)
    scale_param_index = 1
    interval_support = True
    sample_support = True

    def __init__(self, hessian_mode="natural"):
        if hessian_mode != "natural":
            raise ValueError("hessian_mode='natural' is the only supported mode")
        self.hessian_mode = hessian_mode

    def validate_target(self, y):
        return None

    def init_class_major(self, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64)
        weights = sample_weight
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            positive = weights > 0.0
            if not np.any(positive):
                raise ValueError("sample_weight must have positive total weight")
            y = y[positive]
            weights = weights[positive]
        mu0 = float(np.average(y, weights=weights))
        var0 = float(np.average((y - mu0) ** 2, weights=weights))
        rho0 = 0.5 * np.log(max(var0, 1e-12))
        rho0 = float(np.clip(rho0, _GAUSS_INIT_RHO_MIN, _GAUSS_INIT_RHO_MAX))
        return np.array([mu0, rho0], dtype=np.float64)

    def grad_hess_class_major_into(self, y, F, sample_weight, grad_out, hess_out):
        _gaussian_nll_grad_hess_into(y, F, sample_weight, grad_out, hess_out)

    def eval_class_major(self, y, F, sample_weight=None):
        return float(_gaussian_nll_eval(y, F, sample_weight))

    def crps_class_major(self, y, F, sample_weight=None):
        return float(_gaussian_crps(y, F, sample_weight))

    @staticmethod
    def mean_and_sigma(raw):
        rho = np.clip(raw[:, 1], _GAUSS_RHO_MIN, _GAUSS_RHO_MAX)
        return raw[:, 0], np.exp(rho)

    def mean_from_raw(self, raw):
        return np.asarray(raw)[:, 0].copy()

    def mean_from_params(self, mu, sigma):
        return np.asarray(mu, dtype=np.float64).copy()

    def params_from_raw(self, raw):
        mu, sigma = GaussianNLL.mean_and_sigma(np.asarray(raw))
        return mu.copy(), sigma

    def variance_from_raw(self, raw):
        return self.variance_from_params(*self.params_from_raw(raw))

    def variance_from_params(self, mu, sigma):
        return sigma * sigma

    def interval_from_raw(self, raw, alpha):
        return self.interval_from_params(*self.params_from_raw(raw), alpha)

    def interval_from_params(self, mu, sigma, alpha):
        from statistics import NormalDist
        zq = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
        return mu - zq * sigma, mu + zq * sigma

    def sample_from_raw(self, raw, rng, n_samples):
        return self.sample_from_params(*self.params_from_raw(raw), rng, n_samples)

    def sample_from_params(self, mu, sigma, rng, n_samples):
        n_samples = int(n_samples)
        return rng.normal(
            mu[:, None], sigma[:, None], size=(mu.shape[0], n_samples)
        )

    def transform(self, raw):
        return raw


_T_RHO_MIN = -15.0
_T_RHO_MAX = 15.0
_T_INIT_RHO_MIN = -10.0
_T_INIT_RHO_MAX = 10.0
_T_EVAL_Z_GUARD = 1e150


@njit(cache=True, parallel=True)
def _student_t_nll_grad_hess_into(
    y, F, sample_weight, nu, grad_out, hess_out
):
    n = F.shape[1]
    fisher_mu_coef = (nu + 1.0) / (nu + 3.0)
    fisher_rho = 2.0 * nu / (nu + 3.0)
    for i in prange(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                grad_out[0, i] = 0.0
                grad_out[1, i] = 0.0
                hess_out[0, i] = 0.0
                hess_out[1, i] = 0.0
                continue

        r = F[1, i]
        if r < _T_RHO_MIN:
            r = _T_RHO_MIN
        elif r > _T_RHO_MAX:
            r = _T_RHO_MAX
        scale = np.exp(r)
        z = (y[i] - F[0, i]) / scale
        if (
            not np.isfinite(z)
            or z < -_T_EVAL_Z_GUARD
            or z > _T_EVAL_Z_GUARD
        ):
            inv_z = 1.0 / z
            inv_z2 = inv_z * inv_z
            z2_ratio = 1.0 / (1.0 + nu * inv_z2)
            z_over_denom = inv_z * z2_ratio
        else:
            z2 = z * z
            denom = nu + z2
            z2_ratio = z2 / denom
            z_over_denom = z / denom
        grad_out[0, i] = w * (-((nu + 1.0) * z_over_denom) / scale)
        hess_out[0, i] = w * fisher_mu_coef / (scale * scale)
        grad_out[1, i] = w * (1.0 - (nu + 1.0) * z2_ratio)
        hess_out[1, i] = w * fisher_rho


@njit(cache=True)
def _student_t_nll_eval(y, F, sample_weight, nu, nll_const):
    n = F.shape[1]
    total = 0.0
    weight_total = 0.0
    for i in range(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                continue

        r = F[1, i]
        if r < _T_RHO_MIN:
            r = _T_RHO_MIN
        elif r > _T_RHO_MAX:
            r = _T_RHO_MAX
        scale = np.exp(r)
        z = (y[i] - F[0, i]) / scale
        if z < -_T_EVAL_Z_GUARD:
            z = -_T_EVAL_Z_GUARD
        elif z > _T_EVAL_Z_GUARD:
            z = _T_EVAL_Z_GUARD
        total += w * (r + nll_const + 0.5 * (nu + 1.0) * np.log1p(z * z / nu))
        weight_total += w
    return total / weight_total


class StudentTNLL:
    """Fixed-nu Student-t negative log-likelihood."""

    name = "StudentT"
    distribution_name = "student_t"
    is_classification = False
    adjusts_leaves = False
    constant_hessian = False
    n_outputs = 2
    default_eval_metric = "nll"
    supported_eval_metrics = ("nll",)
    target_standardization = True
    target_standardization_basis = "target"
    calibration_targets = ("scale",)
    scale_param_index = 1
    interval_support = True
    sample_support = True

    def __init__(self, nu=6.0, hessian_mode="natural"):
        nu = float(nu)
        if not nu > 2.0:
            raise ValueError("StudentT requires nu > 2 (finite variance)")
        if hessian_mode != "natural":
            raise ValueError("hessian_mode='natural' is the only supported mode")
        self.nu = nu
        self.hessian_mode = hessian_mode
        self._nll_const = (
            0.5 * math.log(nu * math.pi)
            + math.lgamma(nu / 2.0)
            - math.lgamma((nu + 1.0) / 2.0)
        )

    def validate_target(self, y):
        return None

    def init_class_major(self, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64)
        weights = sample_weight
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            positive = weights > 0.0
            if not np.any(positive):
                raise ValueError("sample_weight must have positive total weight")
            y = y[positive]
            weights = weights[positive]
        mu0 = _weighted_median(y, weights)
        mad = _weighted_median(np.abs(y - mu0), weights)
        scale0 = 1.4826 * mad * math.sqrt((self.nu - 2.0) / self.nu)
        if not scale0 > 0.0:
            var0 = float(np.average((y - mu0) ** 2, weights=weights))
            scale0 = math.sqrt(max(var0, 1e-12) * (self.nu - 2.0) / self.nu)
        rho0 = float(np.clip(math.log(max(scale0, 1e-12)), _T_INIT_RHO_MIN, _T_INIT_RHO_MAX))
        return np.array([mu0, rho0], dtype=np.float64)

    def grad_hess_class_major_into(self, y, F, sample_weight, grad_out, hess_out):
        _student_t_nll_grad_hess_into(
            y, F, sample_weight, self.nu, grad_out, hess_out
        )

    def eval_class_major(self, y, F, sample_weight=None):
        return float(
            _student_t_nll_eval(y, F, sample_weight, self.nu, self._nll_const)
        )

    def mean_from_raw(self, raw):
        return np.asarray(raw)[:, 0].copy()

    def mean_from_params(self, mu, scale, nu):
        return np.asarray(mu, dtype=np.float64).copy()

    def params_from_raw(self, raw):
        raw = np.asarray(raw)
        rho = np.clip(raw[:, 1], _T_RHO_MIN, _T_RHO_MAX)
        scale = np.exp(rho)
        return raw[:, 0].copy(), scale, np.full(raw.shape[0], self.nu)

    def scale_calibration_arrays(self, y, params):
        return np.asarray(y, dtype=np.float64), params[0], params[1]

    def variance_from_raw(self, raw):
        return self.variance_from_params(*self.params_from_raw(raw))

    def variance_from_params(self, mu, scale, nu):
        return scale * scale * (self.nu / (self.nu - 2.0))

    def interval_from_raw(self, raw, alpha):
        return self.interval_from_params(*self.params_from_raw(raw), alpha)

    def interval_from_params(self, mu, scale, nu, alpha):
        from scipy.stats import t as _t
        q = float(_t.ppf(1.0 - float(alpha) / 2.0, df=self.nu))
        return mu - q * scale, mu + q * scale

    def sample_from_raw(self, raw, rng, n_samples):
        return self.sample_from_params(*self.params_from_raw(raw), rng, n_samples)

    def sample_from_params(self, mu, scale, nu, rng, n_samples):
        draws = rng.standard_t(self.nu, size=(mu.shape[0], int(n_samples)))
        return mu[:, None] + scale[:, None] * draws

    def transform(self, raw):
        return raw


@njit(cache=True, parallel=True)
def _lognormal_nll_grad_hess_into(y, F, sample_weight, grad_out, hess_out):
    n = F.shape[1]
    for i in prange(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                grad_out[0, i] = 0.0
                grad_out[1, i] = 0.0
                hess_out[0, i] = 0.0
                hess_out[1, i] = 0.0
                continue

        u = np.log(y[i])
        r = F[1, i]
        if r < _GAUSS_RHO_MIN:
            r = _GAUSS_RHO_MIN
        elif r > _GAUSS_RHO_MAX:
            r = _GAUSS_RHO_MAX
        sigma = np.exp(r)
        inv_var = 1.0 / (sigma * sigma)
        z = (u - F[0, i]) / sigma
        if z < -_GAUSS_Z_CLIP:
            z = -_GAUSS_Z_CLIP
        elif z > _GAUSS_Z_CLIP:
            z = _GAUSS_Z_CLIP
        grad_out[0, i] = w * (-(z / sigma))
        hess_out[0, i] = w * inv_var
        grad_out[1, i] = w * (1.0 - z * z)
        hess_out[1, i] = w * 2.0


@njit(cache=True)
def _lognormal_nll_eval(y, F, sample_weight):
    n = F.shape[1]
    total = 0.0
    weight_total = 0.0
    for i in range(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                continue

        u = np.log(y[i])
        r = F[1, i]
        if r < _GAUSS_RHO_MIN:
            r = _GAUSS_RHO_MIN
        elif r > _GAUSS_RHO_MAX:
            r = _GAUSS_RHO_MAX
        sigma = np.exp(r)
        z = (u - F[0, i]) / sigma
        if z < -_GAUSS_EVAL_Z_GUARD:
            z = -_GAUSS_EVAL_Z_GUARD
        elif z > _GAUSS_EVAL_Z_GUARD:
            z = _GAUSS_EVAL_Z_GUARD
        total += w * (_HALF_LOG_2PI + r + 0.5 * z * z + u)
        weight_total += w
    return total / weight_total


class LogNormalNLL:
    """LogNormal negative log-likelihood over strictly positive targets."""

    name = "LogNormal"
    distribution_name = "lognormal"
    is_classification = False
    adjusts_leaves = False
    constant_hessian = False
    n_outputs = 2
    default_eval_metric = "nll"
    supported_eval_metrics = ("nll",)
    target_standardization = True
    target_standardization_basis = "log_target"
    calibration_targets = ("scale",)
    scale_param_index = 1
    interval_support = True
    sample_support = True

    def __init__(self, hessian_mode="natural"):
        if hessian_mode != "natural":
            raise ValueError("hessian_mode='natural' is the only supported mode")
        self.hessian_mode = hessian_mode

    def validate_target(self, y):
        y = np.asarray(y, dtype=np.float64)
        if np.any(y <= 0.0):
            raise ValueError("loss='LogNormal' requires strictly positive targets")

    def preprocessing_target(self, y):
        return np.log(np.asarray(y, dtype=np.float64))

    def standardization_target(self, y):
        return np.log(np.asarray(y, dtype=np.float64))

    def transform_target(self, y, mean, scale):
        z = (self.standardization_target(y) - float(mean)) / float(scale)
        return np.exp(np.clip(z, -700.0, 700.0))

    def init_class_major(self, y, sample_weight=None):
        u = np.log(np.asarray(y, dtype=np.float64))
        weights = sample_weight
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            positive = weights > 0.0
            if not np.any(positive):
                raise ValueError("sample_weight must have positive total weight")
            u = u[positive]
            weights = weights[positive]
        m0 = float(np.average(u, weights=weights))
        var0 = float(np.average((u - m0) ** 2, weights=weights))
        rho0 = 0.5 * np.log(max(var0, 1e-12))
        rho0 = float(np.clip(rho0, _GAUSS_INIT_RHO_MIN, _GAUSS_INIT_RHO_MAX))
        return np.array([m0, rho0], dtype=np.float64)

    def grad_hess_class_major_into(self, y, F, sample_weight, grad_out, hess_out):
        _lognormal_nll_grad_hess_into(y, F, sample_weight, grad_out, hess_out)

    def eval_class_major(self, y, F, sample_weight=None):
        return float(_lognormal_nll_eval(y, F, sample_weight))

    def params_from_raw(self, raw):
        raw = np.asarray(raw)
        rho = np.clip(raw[:, 1], _GAUSS_RHO_MIN, _GAUSS_RHO_MAX)
        return raw[:, 0].copy(), np.exp(rho)

    def scale_calibration_arrays(self, y, params):
        return np.log(np.asarray(y, dtype=np.float64)), params[0], params[1]

    def mean_from_raw(self, raw):
        m, s = self.params_from_raw(raw)
        return self.mean_from_params(m, s)

    def mean_from_params(self, m, s):
        exponent = np.clip(m + 0.5 * s * s, -745.0, 700.0)
        return np.exp(exponent)

    def variance_from_raw(self, raw):
        return self.variance_from_params(*self.params_from_raw(raw))

    def variance_from_params(self, m, s):
        s2 = s * s
        log_expm1_s2 = np.empty_like(s2)
        positive_small = (s2 > 0.0) & (s2 < 50.0)
        zero = s2 <= 0.0
        log_expm1_s2[positive_small] = np.log(np.expm1(s2[positive_small]))
        log_expm1_s2[zero] = -np.inf
        log_expm1_s2[~(positive_small | zero)] = s2[~(positive_small | zero)]
        log_variance = log_expm1_s2 + 2.0 * m + s2
        variance = np.exp(np.clip(log_variance, -745.0, 700.0))
        variance[zero] = 0.0
        return variance

    def interval_from_raw(self, raw, alpha):
        return self.interval_from_params(*self.params_from_raw(raw), alpha)

    def interval_from_params(self, m, s, alpha):
        from statistics import NormalDist
        zq = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
        lo = np.exp(np.clip(m - zq * s, -745.0, 700.0))
        hi = np.exp(np.clip(m + zq * s, -745.0, 700.0))
        return lo, hi

    def sample_from_raw(self, raw, rng, n_samples):
        return self.sample_from_params(*self.params_from_raw(raw), rng, n_samples)

    def sample_from_params(self, m, s, rng, n_samples):
        return rng.lognormal(
            m[:, None], s[:, None], size=(m.shape[0], int(n_samples))
        )

    def transform(self, raw):
        return raw


_POIS_ETA_MIN = -15.0
_POIS_ETA_MAX = 15.0
_POIS_HESS_FLOOR = 1e-6


@njit(cache=True, parallel=True)
def _poisson_nll_grad_hess_into(y, F, sample_weight, grad_out, hess_out):
    n = F.shape[1]
    for i in prange(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                grad_out[0, i] = 0.0
                hess_out[0, i] = 0.0
                continue
        eta = F[0, i]
        if eta < _POIS_ETA_MIN:
            eta = _POIS_ETA_MIN
        elif eta > _POIS_ETA_MAX:
            eta = _POIS_ETA_MAX
        lam = np.exp(eta)
        h = lam
        if h < _POIS_HESS_FLOOR:
            h = _POIS_HESS_FLOOR
        grad_out[0, i] = w * (lam - y[i])
        hess_out[0, i] = w * h


@njit(cache=True)
def _poisson_nll_eval(y, F, sample_weight):
    n = F.shape[1]
    total = 0.0
    weight_total = 0.0
    for i in range(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                continue
        eta = F[0, i]
        if eta < _POIS_ETA_MIN:
            eta = _POIS_ETA_MIN
        elif eta > _POIS_ETA_MAX:
            eta = _POIS_ETA_MAX
        lam = np.exp(eta)
        total += w * (lam - y[i] * eta + math.lgamma(y[i] + 1.0))
        weight_total += w
    return total / weight_total


class PoissonNLL:
    """Poisson negative log-likelihood for nonnegative integer counts."""

    name = "Poisson"
    distribution_name = "poisson"
    is_classification = False
    adjusts_leaves = False
    constant_hessian = False
    n_outputs = 1
    default_eval_metric = "nll"
    supported_eval_metrics = ("nll",)
    calibration_targets = ("mean",)
    mean_param_index = 0
    interval_support = True
    sample_support = True

    def __init__(self):
        pass

    def validate_target(self, y):
        y = np.asarray(y, dtype=np.float64)
        if np.any(y < 0.0):
            raise ValueError("loss='Poisson' requires nonnegative targets")
        if np.any(np.abs(y - np.rint(y)) > 1e-8):
            raise ValueError("loss='Poisson' requires integer counts")

    def init_class_major(self, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64)
        weights = sample_weight
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            positive = weights > 0.0
            if not np.any(positive):
                raise ValueError("sample_weight must have positive total weight")
            y = y[positive]
            weights = weights[positive]
        lam0 = max(float(np.average(y, weights=weights)), 1e-12)
        return np.array([math.log(lam0)], dtype=np.float64)

    def grad_hess_class_major_into(self, y, F, sample_weight, grad_out, hess_out):
        _poisson_nll_grad_hess_into(y, F, sample_weight, grad_out, hess_out)

    def eval_class_major(self, y, F, sample_weight=None):
        return float(_poisson_nll_eval(y, F, sample_weight))

    def mean_from_raw(self, raw):
        eta = np.clip(np.asarray(raw)[:, 0], _POIS_ETA_MIN, _POIS_ETA_MAX)
        return np.exp(eta)

    def mean_from_params(self, lam):
        return np.asarray(lam, dtype=np.float64).copy()

    def params_from_raw(self, raw):
        return (self.mean_from_raw(raw),)

    def variance_from_raw(self, raw):
        return self.mean_from_raw(raw)

    def variance_from_params(self, lam):
        return lam

    def interval_from_raw(self, raw, alpha):
        return self.interval_from_params(*self.params_from_raw(raw), alpha)

    def interval_from_params(self, lam, alpha):
        from scipy.stats import poisson as _poisson
        return _poisson.ppf(float(alpha) / 2.0, lam), _poisson.ppf(
            1.0 - float(alpha) / 2.0, lam
        )

    def sample_from_raw(self, raw, rng, n_samples):
        return self.sample_from_params(*self.params_from_raw(raw), rng, n_samples)

    def sample_from_params(self, lam, rng, n_samples):
        return rng.poisson(
            lam[:, None], size=(lam.shape[0], int(n_samples))
        ).astype(np.float64)

    def transform(self, raw):
        return raw


_NB_ETA_MIN = -15.0
_NB_ETA_MAX = 15.0
_NB_R_MIN = 1e-6
_NB_R_MAX = 1e6
_NB_HESS_FLOOR = 1e-6


@njit(cache=True, parallel=True)
def _nb_global_nll_grad_hess_into(y, F, sample_weight, r, grad_out, hess_out):
    n = F.shape[1]
    for i in prange(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                grad_out[0, i] = 0.0
                hess_out[0, i] = 0.0
                continue
        eta = F[0, i]
        if eta < _NB_ETA_MIN:
            eta = _NB_ETA_MIN
        elif eta > _NB_ETA_MAX:
            eta = _NB_ETA_MAX
        mu = np.exp(eta)
        denom = r + mu
        h = r * mu / denom
        if h < _NB_HESS_FLOOR:
            h = _NB_HESS_FLOOR
        grad_out[0, i] = w * (r * (mu - y[i]) / denom)
        hess_out[0, i] = w * h


@njit(cache=True)
def _nb_global_nll_eval(y, F, sample_weight, r):
    n = F.shape[1]
    total = 0.0
    weight_total = 0.0
    log_r = math.log(r)
    lgamma_r = math.lgamma(r)
    for i in range(n):
        if sample_weight is None:
            w = 1.0
        else:
            w = sample_weight[i]
            if w <= 0.0:
                continue
        eta = F[0, i]
        if eta < _NB_ETA_MIN:
            eta = _NB_ETA_MIN
        elif eta > _NB_ETA_MAX:
            eta = _NB_ETA_MAX
        mu = np.exp(eta)
        log_r_mu = math.log(r + mu)
        nll = (
            -math.lgamma(y[i] + r)
            + lgamma_r
            + math.lgamma(y[i] + 1.0)
            - r * (log_r - log_r_mu)
            - y[i] * (eta - log_r_mu)
        )
        total += w * nll
        weight_total += w
    return total / weight_total


def _golden_section_minimize_python(func, lower, upper, iterations=48, tol=1e-5):
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0
    a = float(lower)
    b = float(upper)
    h = b - a
    if h <= tol:
        x = 0.5 * (a + b)
        return x, float(func(x))
    c = a + inv_phi2 * h
    d = a + inv_phi * h
    yc = float(func(c))
    yd = float(func(d))
    for _ in range(int(iterations)):
        if h <= tol:
            break
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = inv_phi * h
            c = a + inv_phi2 * h
            yc = float(func(c))
        else:
            a = c
            c = d
            yc = yd
            h = inv_phi * h
            d = a + inv_phi * h
            yd = float(func(d))
    x = 0.5 * (a + b)
    return x, float(func(x))


class NegativeBinomialNLL:
    """NB2 negative log-likelihood with global dispersion."""

    name = "NegativeBinomial"
    distribution_name = "negative_binomial"
    is_classification = False
    adjusts_leaves = False
    constant_hessian = False
    n_outputs = 1
    default_eval_metric = "nll"
    supported_eval_metrics = ("nll",)
    calibration_targets = ("mean", "dispersion")
    mean_param_index = 0
    dispersion_param_index = 1
    interval_support = True
    sample_support = True

    def __init__(self, global_dispersion=True, r=None):
        if not bool(global_dispersion):
            raise ValueError(
                "NegativeBinomial heterodispersion is not implemented yet; "
                "use global_dispersion=True"
            )
        self.global_dispersion = True
        self.fixed_r = None if r is None else float(r)
        if self.fixed_r is not None and not self.fixed_r > 0.0:
            raise ValueError("NegativeBinomial r must be positive")
        self.state_ = {}

    def validate_target(self, y):
        y = np.asarray(y, dtype=np.float64)
        if np.any(y < 0.0):
            raise ValueError("loss='NegativeBinomial' requires nonnegative targets")
        if np.any(np.abs(y - np.rint(y)) > 1e-8):
            raise ValueError("loss='NegativeBinomial' requires integer counts")

    def _initial_r(self, y, weights):
        mu0 = max(float(np.average(y, weights=weights)), 1e-12)
        var0 = float(np.average((y - mu0) ** 2, weights=weights))
        if var0 <= mu0:
            return _NB_R_MAX
        alpha0 = max((var0 - mu0) / (mu0 * mu0), 1e-4)
        return float(np.clip(1.0 / alpha0, _NB_R_MIN, _NB_R_MAX))

    def init_class_major(self, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64)
        weights = sample_weight
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            positive = weights > 0.0
            if not np.any(positive):
                raise ValueError("sample_weight must have positive total weight")
            y_fit = y[positive]
            weights_fit = weights[positive]
        else:
            y_fit = y
            weights_fit = None
        mu0 = max(float(np.average(y_fit, weights=weights_fit)), 1e-12)
        r0 = self.fixed_r if self.fixed_r is not None else self._initial_r(y_fit, weights_fit)
        self.state_ = {
            "r": float(r0),
            "source": "explicit" if self.fixed_r is not None else "method_of_moments",
            "r_path": [{"round": 0, "r": float(r0), "source": "init"}],
        }
        return np.array([math.log(mu0)], dtype=np.float64)

    def _current_r(self):
        return float(self.state_.get("r", self.fixed_r or _NB_R_MAX))

    def refresh_state(self, y, F, sample_weight, iteration, *, force=False):
        if self.fixed_r is not None:
            return False
        if not force and int(iteration) % 25 != 0:
            return False
        current = self._current_r()
        log_current = math.log(max(current, _NB_R_MIN))
        lower = max(math.log(_NB_R_MIN), log_current - 6.0)
        upper = min(math.log(_NB_R_MAX), log_current + 6.0)

        def objective(log_r):
            return _nb_global_nll_eval(y, F, sample_weight, math.exp(log_r))

        best_log_r, _ = _golden_section_minimize_python(objective, lower, upper)
        r = float(np.clip(math.exp(best_log_r), _NB_R_MIN, _NB_R_MAX))
        self.state_["r"] = r
        self.state_["source"] = "profile_nll"
        self.state_.setdefault("r_path", []).append({
            "round": int(iteration),
            "r": r,
            "source": "refresh" if not force else "final_refresh",
        })
        return True

    def grad_hess_class_major_into(self, y, F, sample_weight, grad_out, hess_out):
        _nb_global_nll_grad_hess_into(
            y, F, sample_weight, self._current_r(), grad_out, hess_out
        )

    def eval_class_major(self, y, F, sample_weight=None):
        return float(_nb_global_nll_eval(y, F, sample_weight, self._current_r()))

    def mean_from_raw(self, raw):
        eta = np.clip(np.asarray(raw)[:, 0], _NB_ETA_MIN, _NB_ETA_MAX)
        return np.exp(eta)

    def mean_from_params(self, mu, alpha):
        return np.asarray(mu, dtype=np.float64).copy()

    def params_from_raw(self, raw):
        mu = self.mean_from_raw(raw)
        alpha = np.full(mu.shape[0], 1.0 / self._current_r(), dtype=np.float64)
        return mu, alpha

    def variance_from_raw(self, raw):
        return self.variance_from_params(*self.params_from_raw(raw))

    def variance_from_params(self, mu, alpha):
        return mu + alpha * mu * mu

    def interval_from_raw(self, raw, alpha):
        return self.interval_from_params(*self.params_from_raw(raw), alpha)

    def interval_from_params(self, mu, alpha, alpha_interval):
        from scipy.stats import nbinom as _nbinom
        r = 1.0 / np.maximum(alpha, 1e-300)
        p = r / (r + mu)
        return _nbinom.ppf(float(alpha_interval) / 2.0, r, p), _nbinom.ppf(
            1.0 - float(alpha_interval) / 2.0, r, p
        )

    def sample_from_raw(self, raw, rng, n_samples):
        return self.sample_from_params(*self.params_from_raw(raw), rng, n_samples)

    def sample_from_params(self, mu, alpha, rng, n_samples):
        r = 1.0 / np.maximum(alpha, 1e-300)
        p = r / (r + mu)
        return rng.negative_binomial(
            r[:, None], p[:, None], size=(mu.shape[0], int(n_samples))
        ).astype(np.float64)

    def transform(self, raw):
        return raw


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
VECTOR_LOSSES = {
    "Gaussian": GaussianNLL,
    "LogNormal": LogNormalNLL,
    "StudentT": StudentTNLL,
    "Poisson": PoissonNLL,
    "NegativeBinomial": NegativeBinomialNLL,
}
