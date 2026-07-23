from __future__ import annotations

import numpy as np
import pytest

from darkofit.preprocessing import FeaturePreprocessor


def _data():
    X = np.empty((7, 3), dtype=object)
    X[:, 0] = ["a", "a", "b", "b", "c", "c", None]
    X[:, 1] = [1.0, 3.0, 10.0, 14.0, 20.0, 1000.0, 7.0]
    X[:, 2] = np.arange(7, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, len(X))
    return X, y


def test_group_centered_means_are_weighted_target_free_and_bounded() -> None:
    X, y = _data()
    weight = np.ones(len(X), dtype=np.float64)
    weight[5] = 0.0
    prep = FeaturePreprocessor(
        max_bins=32,
        random_state=3,
        group_centered_pairs=[(1, 0), (2, 0)],
    )

    transformed = prep.fit_transform(
        X, [y], [0], sample_weight=weight
    )

    assert transformed.shape[1] == 2 + 2 + 1
    assert prep.group_centered_pairs_ == [(1, 0), (2, 0)]
    assert len(prep.group_centered_means_) == 2
    assert prep.group_centered_means_[0].tolist() == pytest.approx(
        [2.0, 12.0, 20.0, 7.0]
    )
    assert prep.group_centered_global_means_[0] == pytest.approx(55.0 / 6.0)
    assert prep.feature_map_.tolist() == [1, 2, 1, 2, 0]

    changed_target = FeaturePreprocessor(
        max_bins=32,
        random_state=3,
        group_centered_pairs=[(1, 0), (2, 0)],
    )
    changed_target.fit_transform(X, [-100.0 * y], [0], sample_weight=weight)
    for observed, expected in zip(
        changed_target.group_centered_means_, prep.group_centered_means_
    ):
        np.testing.assert_array_equal(observed, expected)


def test_group_centered_transform_uses_global_mean_for_unseen_category() -> None:
    X, y = _data()
    prep = FeaturePreprocessor(
        max_bins=32,
        random_state=2,
        group_centered_pairs=[(1, 0)],
    )
    prep.fit_transform(X, [y], [0])

    X_new = np.array(
        [["never_seen", 100.0, 0.0], [None, 8.0, 0.0]], dtype=object
    )
    codes = prep._codes_for_transform(X_new)
    block = prep._group_centered_block(X_new, codes)

    assert codes[0, 0] == -1
    assert block[0, 0] == pytest.approx(
        100.0 - prep.group_centered_global_means_[0]
    )
    assert block[1, 0] == pytest.approx(1.0)
    assert prep.transform(X_new).shape == (2, 4)


@pytest.mark.parametrize(
    "pairs, message",
    [
        ([(0, 1)], "one numeric and one categorical"),
        ([(1, 0), (1, 0)], "must be unique"),
        ([(1,)], "must contain"),
    ],
)
def test_group_centered_pair_contract_is_loud(pairs, message) -> None:
    X, y = _data()
    prep = FeaturePreprocessor(group_centered_pairs=pairs)

    with pytest.raises(ValueError, match=message):
        prep.fit_transform(X, [y], [0])
