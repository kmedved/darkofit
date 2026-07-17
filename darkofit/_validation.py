"""Small public-input validation helpers shared across estimator layers."""

import operator
import warnings

import numpy as np
from sklearn.exceptions import DataConversionWarning

try:
    from numpy.exceptions import ComplexWarning as _ComplexWarning
except ImportError:  # NumPy < 1.25
    from numpy import ComplexWarning as _ComplexWarning


def reject_masked_array(X, *, name="X"):
    """Reject masked arrays before NumPy can silently discard their mask."""
    if np.ma.isMaskedArray(X):
        raise TypeError(
            f"Masked arrays are not supported for {name}. Convert with "
            f"{name}.filled(np.nan); NaN is treated as missing."
        )


def _array_shape(X, *, name):
    reject_masked_array(X, name=name)
    shape = getattr(X, "shape", None)
    if shape is None:
        shape = np.asarray(X).shape
    if len(shape) != 2:
        raise ValueError(
            f"Expected a 2D array for {name}; got {len(shape)}D array instead. "
            "Reshape your data with array.reshape(-1, 1) for a single feature "
            "or array.reshape(1, -1) for a single sample."
        )
    return int(shape[0]), int(shape[1])


def n_features_from_array_like(X, *, name="X"):
    """Return the second dimension for a 2D array-like without coercing dtype."""
    n_samples, n_features = _array_shape(X, name=name)
    if n_samples == 0:
        raise ValueError(
            f"{name} has 0 sample(s) (shape=(0, {n_features})) while a "
            "minimum of 1 is required; pass at least one sample and at least "
            "one row."
        )
    if n_features == 0:
        raise ValueError(
            f"{name} has 0 feature(s) (shape=({n_samples}, 0)) while a "
            "minimum of 1 is required; pass at least one feature."
        )
    return n_features


def array_like_to_numpy(X, dtype=None):
    """Coerce frame-like objects while preserving nullable missing values."""
    reject_masked_array(X)
    to_numpy = getattr(X, "to_numpy", None)
    if to_numpy is not None:
        if dtype is None:
            return np.asarray(to_numpy())
        kwargs = {"dtype": dtype}
        try:
            return to_numpy(na_value=np.nan, **kwargs)
        except TypeError:
            return np.asarray(to_numpy(), dtype=dtype)
        except ValueError as exc:
            if "read-only" not in str(exc):
                raise
            if hasattr(X, "where") and hasattr(X, "isna"):
                return X.where(~X.isna(), np.nan).to_numpy(dtype=dtype)
            return np.array(to_numpy(), dtype=dtype, copy=True)
    return np.asarray(X, dtype=dtype)


def n_samples_from_array_like(X, *, name="X"):
    """Return the first dimension for a 2D array-like without coercing dtype."""
    n_samples, n_features = _array_shape(X, name=name)
    if n_samples == 0:
        raise ValueError(
            f"{name} has 0 sample(s) (shape=(0, {n_features})) while a "
            "minimum of 1 is required; pass at least one sample and at least "
            "one row."
        )
    return n_samples


def feature_names_from_input(X):
    """Return flat scalar column names for pandas, Polars, or PyArrow input."""
    names = getattr(X, "column_names", None)
    if names is None:
        names = getattr(X, "columns", None)
    if names is None:
        return None
    try:
        values = list(names)
        arr = np.asarray(values, dtype=object)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    if any(
        isinstance(value, (tuple, list, np.ndarray))
        for value in arr.tolist()
    ):
        return None
    if not all(isinstance(value, str) for value in arr.tolist()):
        return None
    return arr


def _contains_complex(values):
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        return True
    if arr.dtype != object:
        return False
    return any(
        isinstance(value, (complex, np.complexfloating))
        for value in arr.ravel()
    )


def _numeric_column(X, index):
    iloc = getattr(X, "iloc", None)
    if iloc is not None:
        return iloc[:, int(index)]
    arr = array_like_to_numpy(X)
    return arr[:, int(index)]


def _describe_nonnumeric_columns(X, indices):
    names = feature_names_from_input(X)
    if names is None:
        return []
    bad = []
    for index in indices:
        try:
            array_like_to_numpy(_numeric_column(X, index), np.float64)
        except (TypeError, ValueError):
            bad.append(repr(str(names[int(index)])))
    return bad


def resolve_cat_features(cat_features, X, n_features=None):
    """Resolve categorical names/indices and reject unknown or duplicate entries."""
    if n_features is None:
        n_features = n_features_from_array_like(X)
    if cat_features is None:
        return ()
    if isinstance(cat_features, (str, bytes)):
        values = [cat_features]
    else:
        arr = np.asarray(cat_features, dtype=object)
        values = [arr.item()] if arr.ndim == 0 else arr.ravel().tolist()
    names = feature_names_from_input(X)
    name_to_index = (
        {} if names is None
        else {str(value): index for index, value in enumerate(names)}
    )
    resolved = []
    for value in values:
        if isinstance(value, str):
            if names is None:
                raise ValueError(
                    "string cat_features require input with named columns"
                )
            if value not in name_to_index:
                raise ValueError(
                    f"cat_features contains unknown column name {value!r}"
                )
            if list(names).count(value) != 1:
                raise ValueError(
                    f"cat_features column name {value!r} is not unique"
                )
            resolved.append(name_to_index[value])
        else:
            resolved.append(value)
    return normalize_cat_features(resolved, n_features)


def coerce_feature_matrix(
    X,
    cat_features=(),
    *,
    name="X",
    check_infinite=True,
    resolve_names=False,
):
    """Validate and coerce a public feature matrix without lossy conversion."""
    reject_masked_array(X, name=name)
    if hasattr(X, "tocoo") and hasattr(X, "format"):
        raise ValueError("sparse matrices are not supported; pass a dense array")
    n_features = n_features_from_array_like(X, name=name)
    if resolve_names:
        cat_features = resolve_cat_features(cat_features, X, n_features)
    else:
        cat_features = normalize_cat_features(cat_features, n_features)
    cat_set = set(cat_features)
    numeric_indices = [
        index for index in range(n_features) if index not in cat_set
    ]

    if cat_features:
        arr = array_like_to_numpy(X, object)
        numeric_raw = arr[:, numeric_indices]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", _ComplexWarning)
                numeric = np.asarray(numeric_raw, dtype=np.float64)
        except _ComplexWarning as exc:
            if _contains_complex(numeric_raw):
                raise ValueError("Complex data not supported.") from exc
            numeric = np.asarray(numeric_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            if _contains_complex(numeric_raw):
                raise ValueError("Complex data not supported.") from exc
            bad = _describe_nonnumeric_columns(X, numeric_indices)
            if bad:
                raise ValueError(
                    f"{name} could not be converted to numeric: column(s) "
                    f"{', '.join(bad)} are non-numeric. Declare them in "
                    "cat_features or encode them first."
                ) from exc
            raise
    else:
        raw = array_like_to_numpy(X)
        if _contains_complex(raw):
            raise ValueError("Complex data not supported.")
        try:
            arr = array_like_to_numpy(X, np.float64)
        except (TypeError, ValueError) as exc:
            bad = _describe_nonnumeric_columns(X, numeric_indices)
            if bad:
                raise ValueError(
                    f"{name} could not be converted to numeric: column(s) "
                    f"{', '.join(bad)} are non-numeric. Declare them in "
                    "cat_features or encode them first."
                ) from exc
            raise
        numeric = arr
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array for {name}.")
    if check_infinite and np.isinf(numeric).any():
        raise ValueError(
            f"{name} contains infinity. NaN is accepted as missing, but "
            "infinity must be clipped or cleaned."
        )
    return arr, cat_features, n_features


def sklearn_assume_finite():
    """Return scikit-learn's serving-time finite-check escape hatch."""
    try:
        from sklearn import get_config

        return bool(get_config().get("assume_finite", False))
    except Exception:
        return False


def validate_feature_names(
    expected_names,
    X,
    *,
    name="X",
    fitted_name="this estimator",
):
    """Match sklearn's named/unnamed warnings and ordered-name enforcement."""
    actual_names = feature_names_from_input(X)
    if expected_names is None and actual_names is None:
        return
    if expected_names is None:
        warnings.warn(
            f"{name} has feature names, but {fitted_name} was fitted without "
            "feature names.",
            UserWarning,
            stacklevel=3,
        )
        return
    if actual_names is None:
        warnings.warn(
            f"{fitted_name} was fitted with feature names, but {name} was "
            "passed without feature names.",
            UserWarning,
            stacklevel=3,
        )
        return
    expected_names = np.asarray(expected_names, dtype=object)
    if not np.array_equal(actual_names, expected_names):
        raise ValueError(
            f"The feature names of {name} do not match those seen during fit. "
            "Columns must match in name and order; DarkoFit does not reorder "
            "them automatically."
        )


def validate_target_vector(y, n_samples, *, name="y", dtype=None):
    """Return a 1D target vector with a row count matching X."""
    if y is None:
        raise ValueError(
            "This estimator requires y to be passed, but the target y is None."
        )
    raw = np.asarray(y)
    if _contains_complex(raw):
        raise ValueError("Complex data not supported.")
    arr = np.asarray(y, dtype=dtype)
    if arr.ndim == 2 and arr.shape[1] == 1:
        warnings.warn(
            f"A column-vector {name} was passed when a 1d array was expected. "
            "Please change the shape of y to (n_samples,).",
            DataConversionWarning,
            stacklevel=2,
        )
        arr = arr.ravel()
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a 1d array")
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
        if idx in seen:
            raise ValueError(
                f"cat_features contains duplicate index {int(idx)}"
            )
        normalized.append(int(idx))
        seen.add(idx)
    return tuple(normalized)
