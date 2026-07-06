"""Small public-input validation helpers shared across estimator layers."""

import operator
import warnings

import numpy as np
from sklearn.exceptions import DataConversionWarning


def n_features_from_array_like(X, *, name="X"):
    """Return the second dimension for a 2D array-like without coercing dtype."""
    shape = getattr(X, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"{name} must be a 2-dimensional array")
        if int(shape[1]) == 0:
            raise ValueError(f"{name} must contain at least one feature")
        return int(shape[1])
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-dimensional array")
    if int(arr.shape[1]) == 0:
        raise ValueError(f"{name} must contain at least one feature")
    return int(arr.shape[1])


def array_like_to_numpy(X, dtype):
    """Coerce array-like objects, preserving pandas nullable missing values."""
    if hasattr(X, "to_numpy"):
        try:
            return X.to_numpy(dtype=dtype, na_value=np.nan)
        except TypeError:
            return X.to_numpy(dtype=dtype)
    return np.asarray(X, dtype=dtype)


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
    if arr.ndim == 2 and arr.shape[1] == 1:
        warnings.warn(
            f"A column-vector {name} was passed when a 1-dimensional array "
            "was expected. Raveling to shape (n_samples,).",
            DataConversionWarning,
            stacklevel=2,
        )
        arr = arr.ravel()
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-dimensional array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if arr.shape[0] != int(n_samples):
        raise ValueError(f"{name} must have shape ({int(n_samples)},)")
    if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def normalize_random_state_seed(random_state, *, name="random_state"):
    """Return an integer seed, None, or raise a public error for invalid input."""
    if random_state is None:
        return None
    try:
        return int(operator.index(random_state))
    except TypeError:
        pass
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, np.iinfo(np.int32).max))
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, np.iinfo(np.int32).max))
    raise ValueError(
        f"{name} must be None, an integer seed, numpy.random.RandomState, "
        "or numpy.random.Generator"
    )


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
