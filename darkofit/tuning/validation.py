"""Validation splitting and fit-payload slicing for tuning."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    check_cv,
)
from sklearn.utils import _safe_indexing


def make_cv_splits(
    X,
    y,
    *,
    cv=5,
    groups=None,
    classifier=False,
    random_state=None,
    validation_fraction=0.2,
    sample_weight=None,
):
    """Return validated train/validation index pairs.

    Grouped splits are owned by the tuner; wrappers receive explicit eval sets
    and therefore should not create hidden validation splits.
    """
    y = np.asarray(y)
    if cv is None:
        splits = list(_holdout_splitter(
            groups=groups,
            classifier=classifier,
            random_state=random_state,
            validation_fraction=validation_fraction,
        ).split(X, y, groups=groups))
        splits = _normalize_cv_splits(splits)
        validate_cv_splits(
            splits, y, groups=groups, classifier=classifier,
            sample_weight=sample_weight,
        )
        return splits

    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape[0] != y.shape[0]:
            raise ValueError("groups must have the same length as y")
        splitter = _group_splitter(
            cv, classifier=classifier, random_state=random_state
        )
        splits = _materialize_cv_splits(splitter, X, y, groups)
    else:
        splitter = _plain_splitter(cv, y, classifier=classifier, random_state=random_state)
        splits = _materialize_cv_splits(splitter, X, y, None)

    splits = _normalize_cv_splits(splits)
    validate_cv_splits(
        splits, y, groups=groups, classifier=classifier,
        sample_weight=sample_weight,
    )
    return splits


def _materialize_cv_splits(splitter_or_splits, X, y, groups):
    if hasattr(splitter_or_splits, "split"):
        if groups is None:
            return list(splitter_or_splits.split(X, y))
        return list(splitter_or_splits.split(X, y, groups=groups))
    return list(splitter_or_splits)


def _normalize_cv_splits(splits):
    return [
        (_normalize_cv_index_array("train", train_idx),
         _normalize_cv_index_array("validation", valid_idx))
        for train_idx, valid_idx in splits
    ]


def _normalize_cv_index_array(name, idx):
    arr = np.asarray(idx)
    if arr.ndim == 1 and arr.size > 0:
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"CV {name} indices must be integer")
    return arr.astype(np.int64, copy=False)


def validate_cv_splits(
    splits, y, *, groups=None, classifier=False, sample_weight=None
):
    y = np.asarray(y)
    n_samples = int(y.shape[0])
    all_classes = np.unique(y) if classifier else None
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    positive_mass_classes = None
    if w is not None:
        if w.shape != y.shape:
            raise ValueError("sample_weight must have the same length as y")
        if not np.all(np.isfinite(w)):
            raise ValueError("sample_weight must contain only finite values")
        if np.any(w < 0.0):
            raise ValueError("sample_weight must be nonnegative")
        if float(np.sum(w)) <= 0.0:
            raise ValueError("sample_weight must have positive total weight")
        if classifier:
            positive_mass_classes = [
                cls for cls in all_classes if float(np.sum(w[y == cls])) > 0.0
            ]
    for train_idx, valid_idx in splits:
        _validate_cv_index_array("train", train_idx, n_samples)
        _validate_cv_index_array("validation", valid_idx, n_samples)
        if train_idx.size == 0 or valid_idx.size == 0:
            raise ValueError("CV splits must have non-empty train and validation folds")
        if np.intersect1d(train_idx, valid_idx, assume_unique=True).size:
            raise ValueError("CV train and validation indices must be disjoint")
        if w is not None:
            train_mass = float(np.sum(w[train_idx]))
            valid_mass = float(np.sum(w[valid_idx]))
            if train_mass <= 0.0 or valid_mass <= 0.0:
                raise ValueError(
                    "CV splits must assign positive sample_weight mass to "
                    "both training and validation folds"
                )
            if classifier:
                for cls in positive_mass_classes:
                    train_class_mass = float(np.sum(w[train_idx][y[train_idx] == cls]))
                    valid_class_mass = float(np.sum(w[valid_idx][y[valid_idx] == cls]))
                    if train_class_mass <= 0.0 or valid_class_mass <= 0.0:
                        raise ValueError(
                            "CV splits must assign positive sample_weight mass "
                            "for each positive-mass class to both training and "
                            "validation folds"
                        )
        if groups is not None:
            train_groups = set(np.asarray(groups)[train_idx])
            valid_groups = set(np.asarray(groups)[valid_idx])
            if train_groups.intersection(valid_groups):
                raise ValueError("groups cannot appear in both train and validation folds")
        if classifier:
            train_classes = np.unique(y[train_idx])
            if train_classes.size != all_classes.size:
                raise ValueError(
                    "each classification training fold must contain all classes"
                )


def _validate_cv_index_array(name, idx, n_samples):
    if idx.ndim != 1:
        raise ValueError(f"CV {name} indices must be a 1-dimensional array")
    if np.any((idx < 0) | (idx >= n_samples)):
        raise ValueError(f"CV {name} indices are out of bounds")
    if np.unique(idx).shape[0] != idx.shape[0]:
        raise ValueError(f"CV {name} indices must be unique")


def slice_fit_payload(X, y, train_idx, valid_idx, sample_weight=None):
    X_train = _safe_indexing(X, train_idx)
    y_train = _safe_indexing(y, train_idx)
    X_valid = _safe_indexing(X, valid_idx)
    y_valid = _safe_indexing(y, valid_idx)
    if sample_weight is None:
        return X_train, y_train, X_valid, y_valid, None, None
    w = np.asarray(sample_weight, dtype=np.float64)
    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        w[train_idx],
        w[valid_idx],
    )


def validation_mass(valid_idx, sample_weight=None):
    if sample_weight is None:
        return float(len(valid_idx))
    return float(np.sum(np.asarray(sample_weight, dtype=np.float64)[valid_idx]))


def _plain_splitter(cv, y, *, classifier, random_state):
    if not isinstance(cv, int):
        return check_cv(cv=cv, y=y, classifier=classifier)
    if classifier:
        _, counts = np.unique(y, return_counts=True)
        n_splits = int(cv)
        min_class_count = int(np.min(counts))
        if n_splits < 2:
            raise ValueError("cv must be at least 2 for classification")
        if n_splits > min_class_count:
            raise ValueError(
                "cv cannot exceed the number of samples in the least "
                f"populated class ({min_class_count})"
            )
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    return KFold(n_splits=int(cv), shuffle=True, random_state=random_state)


def _group_splitter(cv, *, classifier, random_state):
    if not isinstance(cv, int):
        return cv
    if classifier:
        return StratifiedGroupKFold(
            n_splits=int(cv), shuffle=True, random_state=random_state
        )
    return GroupKFold(n_splits=int(cv))


def _holdout_splitter(*, groups, classifier, random_state, validation_fraction):
    if groups is not None:
        return GroupShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
    if classifier:
        return StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
    return ShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_state,
    )
