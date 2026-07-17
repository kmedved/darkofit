"""Exact interventional TreeSHAP for oblivious DarkoFit forests.

The exact coalition-enumeration algorithm in this module is adapted from
ChimeraBoost commit ff6f248d09f92d608ed8cc366463b61f1af04acc,
Copyright 2026 Nathan Walker, under the Apache License, Version 2.0.

DarkoFit modifications include row-major binned inputs, its local-linear tree
representation, original-feature grouping for every fitted preprocessing
column, support for linear features beyond the split set, and separate safe
packing/validation boundaries.
"""

import operator

import numpy as np
from numba import njit, prange

from .tree import ObliviousTree


SHAP_BACKGROUND_SIZE = 200
SHAP_MAX_PLAYERS = 16


def normalize_max_background(value):
    """Return a positive integer background cap."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("max_background must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError("max_background must be a positive integer") from exc
    if value < 1:
        raise ValueError("max_background must be at least 1")
    return int(value)


def factorials(n):
    """Return 0! through n! as float64 Shapley coalition weights."""
    values = np.empty(int(n) + 1, dtype=np.float64)
    values[0] = 1.0
    for index in range(1, len(values)):
        values[index] = values[index - 1] * index
    return values


def pack_oblivious_shap_forest(trees):
    """Pack constant/local-linear oblivious trees for the SHAP kernel."""
    if any(type(tree) is not ObliviousTree for tree in trees):
        raise NotImplementedError(
            "TreeSHAP currently supports only oblivious trees"
        )
    n_trees = len(trees)
    max_depth = max((tree.depth for tree in trees), default=0)
    depths = np.asarray([tree.depth for tree in trees], dtype=np.int64)
    features = np.zeros((n_trees, max(max_depth, 1)), dtype=np.int64)
    thresholds = np.zeros((n_trees, max(max_depth, 1)), dtype=np.int64)
    linear_counts = np.zeros(n_trees, dtype=np.int64)
    feature_offsets = np.zeros(n_trees + 1, dtype=np.int64)
    coefficient_offsets = np.zeros(n_trees + 1, dtype=np.int64)
    linear_feature_chunks = []
    coefficient_chunks = []
    linear_bin_values = None

    for index, tree in enumerate(trees):
        if tree.depth:
            features[index, : tree.depth] = tree.splits_feat
            thresholds[index, : tree.depth] = tree.splits_thr
        if tree.linear_coefficients is None:
            tree_linear_features = np.empty(0, dtype=np.int64)
            tree_coefficients = np.asarray(tree.values, dtype=np.float64)
        else:
            tree_linear_features = np.asarray(
                tree.linear_features, dtype=np.int64
            )
            tree_coefficients = np.asarray(
                tree.linear_coefficients, dtype=np.float64
            ).reshape(-1)
            candidate = np.asarray(tree.linear_bin_values, dtype=np.float64)
            if linear_bin_values is None:
                linear_bin_values = candidate
            elif (
                linear_bin_values is not candidate
                and not np.array_equal(
                    linear_bin_values, candidate, equal_nan=True
                )
            ):
                raise ValueError(
                    "linear trees do not share fitted bin values"
                )
        linear_counts[index] = len(tree_linear_features)
        linear_feature_chunks.append(tree_linear_features)
        coefficient_chunks.append(tree_coefficients)
        feature_offsets[index + 1] = (
            feature_offsets[index] + len(tree_linear_features)
        )
        coefficient_offsets[index + 1] = (
            coefficient_offsets[index] + len(tree_coefficients)
        )

    linear_features = (
        np.concatenate(linear_feature_chunks)
        if linear_feature_chunks
        else np.empty(0, dtype=np.int64)
    )
    coefficients = (
        np.concatenate(coefficient_chunks)
        if coefficient_chunks
        else np.empty(0, dtype=np.float64)
    )
    if linear_bin_values is None:
        linear_bin_values = np.zeros((1, 1), dtype=np.float64)
    return (
        depths,
        features,
        thresholds,
        linear_counts,
        feature_offsets,
        linear_features,
        coefficient_offsets,
        coefficients,
        np.ascontiguousarray(linear_bin_values),
    )


def max_original_players(trees, feature_map):
    """Maximum distinct original features used by any retained tree."""
    feature_map = np.asarray(feature_map, dtype=np.int64)
    maximum = 0
    for tree in trees:
        if type(tree) is not ObliviousTree:
            raise NotImplementedError(
                "TreeSHAP currently supports only oblivious trees"
            )
        if tree.depth == 0:
            continue
        internal = list(np.asarray(tree.splits_feat, dtype=np.int64))
        if tree.linear_coefficients is not None:
            internal.extend(
                np.asarray(tree.linear_features, dtype=np.int64).tolist()
            )
        maximum = max(maximum, len({int(feature_map[item]) for item in internal}))
    return maximum


@njit(cache=True, parallel=True)
def shap_forest_linear(
    X_binned,
    background_binned,
    depths,
    features,
    thresholds,
    linear_counts,
    feature_offsets,
    linear_features,
    coefficient_offsets,
    coefficients,
    linear_bin_values,
    feature_map,
    n_original_features,
    factorial_values,
):
    """Exact empirical interventional SHAP in original feature space.

    Each tree's distinct original features are coalition players. For every
    explained/background row pair, features inside a coalition take the
    explained value and all others take the background value. Exact Shapley
    marginals are averaged across the empirical background and summed across
    trees. Linear-leaf intercepts and slopes participate in the same game.
    """
    n_rows = X_binned.shape[0]
    n_background = background_binned.shape[0]
    n_trees = depths.shape[0]
    contributions = np.zeros((n_rows, n_original_features), dtype=np.float64)
    inverse_background = 1.0 / n_background

    for row in prange(n_rows):
        for tree in range(n_trees):
            depth = depths[tree]
            if depth == 0:
                continue
            linear_count = linear_counts[tree]
            feature_base = feature_offsets[tree]
            coefficient_base = coefficient_offsets[tree]

            players = np.empty(depth + linear_count, dtype=np.int64)
            level_player = np.empty(depth, dtype=np.int64)
            player_count = 0
            for level in range(depth):
                original = feature_map[features[tree, level]]
                player = -1
                for candidate in range(player_count):
                    if players[candidate] == original:
                        player = candidate
                        break
                if player < 0:
                    players[player_count] = original
                    player = player_count
                    player_count += 1
                level_player[level] = player

            linear_player = np.empty(linear_count, dtype=np.int64)
            for index in range(linear_count):
                internal = linear_features[feature_base + index]
                original = feature_map[internal]
                player = -1
                for candidate in range(player_count):
                    if players[candidate] == original:
                        player = candidate
                        break
                if player < 0:
                    players[player_count] = original
                    player = player_count
                    player_count += 1
                linear_player[index] = player

            subset_count = 1 << player_count
            explained_bits = np.empty(depth, dtype=np.int64)
            for level in range(depth):
                explained_bits[level] = (
                    1
                    if X_binned[row, features[tree, level]]
                    > thresholds[tree, level]
                    else 0
                )
            explained_linear = np.empty(linear_count, dtype=np.float64)
            for index in range(linear_count):
                internal = linear_features[feature_base + index]
                value = linear_bin_values[
                    internal, X_binned[row, internal]
                ]
                explained_linear[index] = value if np.isfinite(value) else 0.0

            coalition_values = np.empty(subset_count, dtype=np.float64)
            background_bits = np.empty(depth, dtype=np.int64)
            background_linear = np.empty(linear_count, dtype=np.float64)
            for background_row in range(n_background):
                for level in range(depth):
                    background_bits[level] = (
                        1
                        if background_binned[
                            background_row, features[tree, level]
                        ] > thresholds[tree, level]
                        else 0
                    )
                for index in range(linear_count):
                    internal = linear_features[feature_base + index]
                    value = linear_bin_values[
                        internal, background_binned[background_row, internal]
                    ]
                    background_linear[index] = (
                        value if np.isfinite(value) else 0.0
                    )

                for mask in range(subset_count):
                    leaf = 0
                    for level in range(depth):
                        if (mask >> level_player[level]) & 1:
                            bit = explained_bits[level]
                        else:
                            bit = background_bits[level]
                        leaf = leaf * 2 + bit
                    coefficient = (
                        coefficient_base + leaf * (1 + linear_count)
                    )
                    value = coefficients[coefficient]
                    for index in range(linear_count):
                        if (mask >> linear_player[index]) & 1:
                            linear_value = explained_linear[index]
                        else:
                            linear_value = background_linear[index]
                        value += (
                            coefficients[coefficient + 1 + index]
                            * linear_value
                        )
                    coalition_values[mask] = value

                for player in range(player_count):
                    player_bit = 1 << player
                    contribution = 0.0
                    for mask in range(subset_count):
                        if (mask >> player) & 1:
                            continue
                        size = 0
                        remaining = mask
                        while remaining:
                            size += remaining & 1
                            remaining >>= 1
                        weight = (
                            factorial_values[size]
                            * factorial_values[player_count - size - 1]
                            / factorial_values[player_count]
                        )
                        contribution += weight * (
                            coalition_values[mask | player_bit]
                            - coalition_values[mask]
                        )
                    contributions[row, players[player]] += (
                        contribution * inverse_background
                    )
    return contributions
