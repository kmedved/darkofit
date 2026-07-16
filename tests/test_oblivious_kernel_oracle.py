"""Readable Phase-0 oracle for future oblivious-kernel consolidation."""

from __future__ import annotations

import numpy as np
import pytest

from darkofit.tree import build_oblivious_tree


def _reference_oblivious_tree(
    X_binned,
    grad,
    hess,
    n_bins,
    *,
    max_depth,
    l2,
    learning_rate,
    min_gain,
    min_child_weight,
    feature_mask,
    row_indices,
):
    """Plain Python shared-split tree with DarkoFit's empty-child semantics."""
    X_binned = np.asarray(X_binned)
    grad = np.asarray(grad, dtype=np.float64)
    hess = np.asarray(hess, dtype=np.float64)
    selected_rows = (
        np.arange(X_binned.shape[0], dtype=np.int64)
        if row_indices is None
        else np.asarray(row_indices, dtype=np.int64)
    )
    leaf = np.zeros(X_binned.shape[0], dtype=np.int64)
    split_features = []
    split_thresholds = []
    split_gains = []

    for depth in range(max_depth):
        n_leaves = 1 << depth
        best_feature = 0
        best_threshold = -1
        best_gain = -np.inf
        for feature in range(X_binned.shape[1]):
            if not feature_mask[feature]:
                continue
            for threshold in range(int(n_bins[feature]) - 1):
                gain = 0.0
                legal = True
                any_nonempty = False
                for leaf_id in range(n_leaves):
                    total_gradient = 0.0
                    total_hessian = 0.0
                    left_gradient = 0.0
                    left_hessian = 0.0
                    for row in selected_rows:
                        if leaf[row] != leaf_id:
                            continue
                        total_gradient += grad[row]
                        total_hessian += hess[row]
                        if X_binned[row, feature] <= threshold:
                            left_gradient += grad[row]
                            left_hessian += hess[row]
                    if total_hessian <= 0.0:
                        continue
                    any_nonempty = True
                    right_hessian = total_hessian - left_hessian
                    # A structurally empty child is legal for a shared split:
                    # that leaf contributes no gain while productive sibling
                    # leaves may still use the level's common split.
                    if left_hessian <= 0.0 or right_hessian <= 0.0:
                        continue
                    if (
                        left_hessian < min_child_weight
                        or right_hessian < min_child_weight
                    ):
                        legal = False
                        break
                    left_denom = left_hessian + l2
                    right_denom = right_hessian + l2
                    parent_denom = total_hessian + l2
                    if min(left_denom, right_denom, parent_denom) <= 0.0:
                        legal = False
                        break
                    right_gradient = total_gradient - left_gradient
                    gain += (
                        left_gradient * left_gradient / left_denom
                        + right_gradient * right_gradient / right_denom
                        - total_gradient * total_gradient / parent_denom
                    )
                if legal and any_nonempty and gain > best_gain:
                    best_feature = feature
                    best_threshold = threshold
                    best_gain = gain

        if best_threshold < 0 or best_gain <= min_gain:
            break
        split_features.append(best_feature)
        split_thresholds.append(best_threshold)
        split_gains.append(best_gain)
        leaf = (leaf << 1) + (
            X_binned[:, best_feature] > best_threshold
        ).astype(np.int64)

    n_leaves = 1 << len(split_features)
    leaf_gradient = np.zeros(n_leaves, dtype=np.float64)
    leaf_hessian = np.zeros(n_leaves, dtype=np.float64)
    for row in selected_rows:
        leaf_gradient[leaf[row]] += grad[row]
        leaf_hessian[leaf[row]] += hess[row]
    values = np.zeros(n_leaves, dtype=np.float64)
    positive = leaf_hessian > 0.0
    values[positive] = (
        -learning_rate
        * leaf_gradient[positive]
        / (leaf_hessian[positive] + l2)
    )
    return {
        "split_features": np.asarray(split_features, dtype=np.int64),
        "split_thresholds": np.asarray(split_thresholds, dtype=np.int64),
        "split_gains": np.asarray(split_gains, dtype=np.float64),
        "leaf": leaf,
        "leaf_gradient": leaf_gradient,
        "leaf_hessian": leaf_hessian,
        "values": values,
    }


def _oracle_data():
    index = np.arange(96, dtype=np.int64)
    X_binned = np.column_stack(
        [
            (index * 3 + 1) % 7,
            (index * 5 + index // 4) % 9,
            (index * 7 + index // 3) % 8,
            (index * 11 + 2) % 6,
            (index * 13 + index // 5) % 10,
        ]
    ).astype(np.uint16)
    grad = (
        ((index * 17 + 3) % 31 - 15) / 7.0
        + (X_binned[:, 1] > 4) * 0.75
        - (X_binned[:, 4] > 6) * 0.5
    ).astype(np.float64)
    hess = (0.5 + ((index * 19 + 1) % 11) / 10.0).astype(np.float64)
    return (
        X_binned,
        grad,
        hess,
        np.asarray([7, 9, 8, 6, 10], dtype=np.int64),
    )


@pytest.mark.parametrize(
    "feature_indices,row_indices",
    [
        (None, None),
        (np.asarray([0, 2, 4], dtype=np.int64), None),
        (None, np.arange(0, 96, 2, dtype=np.int64)),
        (
            np.asarray([1, 3, 4], dtype=np.int64),
            np.flatnonzero((np.arange(96) % 3) != 1).astype(np.int64),
        ),
    ],
)
def test_oblivious_builder_matches_readable_oracle(
    feature_indices, row_indices
):
    X_binned, grad, hess, n_bins = _oracle_data()
    feature_mask = np.zeros(X_binned.shape[1], dtype=np.int64)
    if feature_indices is None:
        feature_mask[:] = 1
    else:
        feature_mask[feature_indices] = 1
    kwargs = {
        "max_depth": 4,
        "l2": 2.5,
        "learning_rate": 0.1,
        "min_gain": 1e-10,
        "min_child_weight": 0.4,
        "feature_mask": feature_mask,
        "row_indices": row_indices,
    }
    expected = _reference_oblivious_tree(
        X_binned, grad, hess, n_bins, **kwargs
    )
    tree, leaf, leaf_gradient, leaf_hessian = build_oblivious_tree(
        X_binned,
        grad,
        hess,
        n_bins,
        kwargs["max_depth"],
        kwargs["l2"],
        kwargs["learning_rate"],
        min_gain=kwargs["min_gain"],
        min_child_weight=kwargs["min_child_weight"],
        feature_mask=feature_mask,
        feature_indices=feature_indices,
        row_indices=row_indices,
        level_histogram_subtraction=False,
        return_training_state=True,
    )

    assert np.array_equal(tree.splits_feat, expected["split_features"])
    assert np.array_equal(tree.splits_thr, expected["split_thresholds"])
    assert np.array_equal(leaf, expected["leaf"])
    assert np.array_equal(tree.apply(X_binned), expected["leaf"])
    np.testing.assert_allclose(
        tree.gains, expected["split_gains"], rtol=0.0, atol=1e-12
    )
    assert np.array_equal(leaf_gradient, expected["leaf_gradient"])
    assert np.array_equal(leaf_hessian, expected["leaf_hessian"])
    assert np.array_equal(tree.values, expected["values"])
    assert np.array_equal(tree.predict(X_binned), expected["values"][leaf])
