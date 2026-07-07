"""Data-dependent default policies for ChimeraBoost.

The learning-rate constants are CatBoost CPU auto-learning-rate coefficients
for boost-from-average losses, transplanted as an initial transparent policy.
ChimeraBoost uses Kish effective sample size instead of CatBoost's raw row
count to expose sample-weight concentration. ChimeraBoost's CatBoost-mode RMSE
learner also applies a measured LR-only correction for materially weighted
regression data; all-ones weights keep the unweighted path.
"""

import math

import numpy as np


AUTO_LR_RULE = "catboost-transplant-v2"
AUTO_LR_MIN = 0.005
AUTO_LR_MAX = 0.5
AUTO_LR_FEATURE_RATIO_REFERENCE = 0.05
AUTO_LR_FEATURE_MULTIPLIER_MIN = 0.85
CATBOOST_WEIGHTED_RMSE_LR_MULTIPLIER = 1.543
LIGHTGBM_UNWEIGHTED_LR_MULTIPLIERS = {
    "RMSE": 0.370462,
    "Logloss": 0.421916,
    "MultiClass": 0.4,
}

_LR_COEFS = {
    ("RMSE", True): (0.157, -4.062, -0.610),
    ("RMSE", False): (0.158, -4.287, -0.813),
    ("Logloss", True): (0.246, -5.127, -0.451),
    ("Logloss", False): (0.408, -7.299, -0.928),
    ("MultiClass", True): (0.020, -2.364, -0.382),
    ("MultiClass", False): (0.051, -2.889, -0.845),
}

_FALLBACK_LOSS = {
    "MAE": ("RMSE", 1.5),
    "Quantile": ("RMSE", 1.5),
    "Gaussian": ("RMSE", 1.0),
}


def is_auto_learning_rate(learning_rate):
    """Return True when the public learning_rate value requests auto mode."""
    return learning_rate is None or learning_rate == "auto"


def effective_sample_size(sample_weight, n_samples):
    """Kish effective sample size for normalized or raw sample weights."""
    if sample_weight is None:
        return float(n_samples)
    w = np.asarray(sample_weight, dtype=np.float64)
    denom = float(np.dot(w, w))
    if denom <= 0.0:
        return 0.0
    return float((w.sum() ** 2) / denom)


def _raw_auto_learning_rate(
    loss_name,
    n_eff,
    iterations,
    use_best_model,
    tree_mode,
    max_leaves,
    n_eff_fraction=1.0,
    p_model=None,
):
    base_loss, multiplier = _FALLBACK_LOSS.get(loss_name, (loss_name, 1.0))
    key = (base_loss, bool(use_best_model))
    if key not in _LR_COEFS:
        base_loss, multiplier = "RMSE", 1.0
        key = (base_loss, bool(use_best_model))
    a, b, c = _LR_COEFS[key]
    n_ref = max(float(n_eff), 2.0)
    t_ref = max(float(iterations), 1.0)
    lr = math.exp(a * math.log(n_ref) + b) * (t_ref / 1000.0) ** c
    lr *= multiplier
    if (
        loss_name == "RMSE"
        and tree_mode in {"catboost", "oblivious"}
        and float(n_eff_fraction) < 0.99
    ):
        lr *= CATBOOST_WEIGHTED_RMSE_LR_MULTIPLIER
    if tree_mode == "lightgbm":
        leaves = max(float(max_leaves), 1.0)
        lr *= (31.0 / leaves) ** 0.25
        if float(n_eff_fraction) >= 0.99:
            lr *= LIGHTGBM_UNWEIGHTED_LR_MULTIPLIERS.get(base_loss, 0.4)
    lr *= feature_count_learning_rate_multiplier(n_eff, p_model)
    return float(lr)


def feature_count_learning_rate_multiplier(n_eff, p_model):
    """Return a small bounded LR shrinkage for high-dimensional expansions."""
    if p_model is None:
        return 1.0
    p = max(float(p_model), 0.0)
    if p <= 0.0:
        return 1.0
    ratio = p / max(float(n_eff), 1.0)
    if ratio <= AUTO_LR_FEATURE_RATIO_REFERENCE:
        return 1.0
    multiplier = (AUTO_LR_FEATURE_RATIO_REFERENCE / ratio) ** 0.05
    return float(np.clip(multiplier, AUTO_LR_FEATURE_MULTIPLIER_MIN, 1.0))


def auto_learning_rate_details(
    loss_name,
    n_eff,
    iterations,
    use_best_model,
    tree_mode,
    max_leaves,
    n_eff_fraction=1.0,
    p_model=None,
):
    """Return the automatic learning rate plus clipping diagnostics.

    The fitted CatBoost source formula is:
        exp(a * log(n) + b) * (iterations / 1000) ** c

    ``n_eff`` replaces raw n to respect sample-weight concentration.
    """
    raw = _raw_auto_learning_rate(
        loss_name,
        n_eff,
        iterations,
        use_best_model,
        tree_mode,
        max_leaves,
        n_eff_fraction=n_eff_fraction,
        p_model=p_model,
    )
    clipped = float(np.clip(raw, AUTO_LR_MIN, AUTO_LR_MAX))
    clip_bound = None
    if raw < AUTO_LR_MIN:
        clip_bound = "min"
    elif raw > AUTO_LR_MAX:
        clip_bound = "max"
    feature_ratio = (
        None
        if p_model is None
        else float(max(float(p_model), 0.0) / max(float(n_eff), 1.0))
    )
    feature_multiplier = feature_count_learning_rate_multiplier(n_eff, p_model)
    details = {
        "resolved": round(clipped, 6),
        "raw_auto": raw,
        "p_model": None if p_model is None else int(p_model),
        "feature_ratio": feature_ratio,
        "feature_multiplier": feature_multiplier,
        "feature_shrinkage_active": bool(feature_multiplier < 1.0),
        "clipped": clip_bound is not None,
        "clip_bound": clip_bound,
        "clip_min": AUTO_LR_MIN,
        "clip_max": AUTO_LR_MAX,
    }
    if loss_name == "Gaussian":
        details["loss_coefficient_source"] = "rmse_coefs_for_gaussian"
    return details


def auto_learning_rate(
    loss_name,
    n_eff,
    iterations,
    use_best_model,
    tree_mode,
    max_leaves,
    n_eff_fraction=1.0,
    p_model=None,
):
    """Resolve a CatBoost-form automatic learning rate."""
    return auto_learning_rate_details(
        loss_name,
        n_eff,
        iterations,
        use_best_model,
        tree_mode,
        max_leaves,
        n_eff_fraction=n_eff_fraction,
        p_model=p_model,
    )["resolved"]


def resolve_learning_rate(
    learning_rate,
    *,
    loss_name,
    n_eff,
    iterations,
    use_best_model,
    tree_mode,
    max_leaves,
    n_eff_fraction=1.0,
    p_model=None,
):
    """Resolve explicit or automatic learning_rate values."""
    return resolve_learning_rate_details(
        learning_rate,
        loss_name=loss_name,
        n_eff=n_eff,
        iterations=iterations,
        use_best_model=use_best_model,
        tree_mode=tree_mode,
        max_leaves=max_leaves,
        n_eff_fraction=n_eff_fraction,
        p_model=p_model,
    )["resolved"]


def resolve_learning_rate_details(
    learning_rate,
    *,
    loss_name,
    n_eff,
    iterations,
    use_best_model,
    tree_mode,
    max_leaves,
    n_eff_fraction=1.0,
    p_model=None,
):
    """Resolve explicit or automatic learning_rate values with metadata."""
    if is_auto_learning_rate(learning_rate):
        details = auto_learning_rate_details(
            loss_name,
            n_eff,
            iterations,
            use_best_model,
            tree_mode,
            max_leaves,
            n_eff_fraction=n_eff_fraction,
            p_model=p_model,
        )
        details.update({
            "source": "auto",
            "input": learning_rate,
            "rule": AUTO_LR_RULE,
        })
        return details
    try:
        lr = float(learning_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "learning_rate must be a number, None, or 'auto'"
        ) from exc
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    return {
        "resolved": lr,
        "raw_auto": None,
        "source": "explicit",
        "input": lr,
        "rule": "explicit",
        "p_model": None if p_model is None else int(p_model),
        "feature_ratio": None,
        "feature_multiplier": 1.0,
        "feature_shrinkage_active": False,
        "clipped": False,
        "clip_bound": None,
        "clip_min": AUTO_LR_MIN,
        "clip_max": AUTO_LR_MAX,
    }
