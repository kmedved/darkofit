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
        validate_cv_splits(splits, y, groups=groups, classifier=classifier)
        return [(np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
                for tr, va in splits]

    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape[0] != y.shape[0]:
            raise ValueError("groups must have the same length as y")
        splitter = _group_splitter(cv, classifier=classifier, random_state=random_state)
        splits = list(splitter.split(X, y, groups=groups))
    else:
        splitter = _plain_splitter(cv, y, classifier=classifier, random_state=random_state)
        splits = list(splitter.split(X, y))

    validate_cv_splits(splits, y, groups=groups, classifier=classifier)
    return [(np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
            for tr, va in splits]


def validate_cv_splits(splits, y, *, groups=None, classifier=False):
    y = np.asarray(y)
    all_classes = np.unique(y) if classifier else None
    for train_idx, valid_idx in splits:
        if train_idx.size == 0 or valid_idx.size == 0:
            raise ValueError("CV splits must have non-empty train and validation folds")
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
        n_splits = max(2, min(int(cv), int(np.min(counts))))
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
