"""Small public-input validation helpers shared across estimator layers."""

import operator

import numpy as np


def n_features_from_array_like(X, *, name="X"):
    """Return the second dimension for a 2D array-like without coercing dtype."""
    shape = getattr(X, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"{name} must be a 2-dimensional array")
        return int(shape[1])
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-dimensional array")
    return int(arr.shape[1])


def n_samples_from_array_like(X, *, name="X"):
    """Return the first dimension for a 2D array-like without coercing dtype."""
    shape = getattr(X, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"{name} must be a 2-dimensional array")
        return int(shape[0])
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-dimensional array")
    return int(arr.shape[0])


def validate_target_vector(y, n_samples, *, name="y", dtype=None):
    """Return a 1D target vector with a row count matching X."""
    arr = np.asarray(y, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-dimensional array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if arr.shape[0] != int(n_samples):
        raise ValueError(f"{name} must have shape ({int(n_samples)},)")
    if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def normalize_cat_features(cat_features, n_features=None):
    """Normalize categorical column indices to a tuple with clear validation."""
    if cat_features is None:
        return ()
    if isinstance(cat_features, (str, bytes)):
        raise ValueError("cat_features must contain integer column indices")

    arr = np.asarray(cat_features)
    values = [arr.item()] if arr.ndim == 0 else arr.ravel().tolist()
    normalized = []
    seen = set()
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "cat_features must contain integer column indices"
            )
        try:
            idx = operator.index(value)
        except TypeError as exc:
            raise ValueError(
                "cat_features must contain integer column indices"
            ) from exc
        if idx < 0:
            raise ValueError("cat_features indices must be nonnegative")
        if n_features is not None and idx >= int(n_features):
            raise ValueError(
                f"cat_features index {idx} is out of bounds for "
                f"{int(n_features)} input features"
            )
        if idx not in seen:
            normalized.append(int(idx))
            seen.add(idx)
    return tuple(normalized)
