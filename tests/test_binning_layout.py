from __future__ import annotations

import numpy as np

from darkofit.binning import Binner, _coerce_transform_block


def test_transform_block_preserves_both_contiguous_layouts():
    base = np.arange(60, dtype=np.float64).reshape(12, 5)
    for block in (np.ascontiguousarray(base), np.asfortranarray(base)):
        coerced = _coerce_transform_block(block)
        assert coerced is block
        assert np.shares_memory(coerced, block)


def test_transform_block_copies_genuinely_strided_input():
    base = np.arange(120, dtype=np.float64).reshape(12, 10)
    block = base[:, ::2]
    assert not block.flags.c_contiguous
    assert not block.flags.f_contiguous

    coerced = _coerce_transform_block(block)

    assert coerced.flags.c_contiguous
    assert not np.shares_memory(coerced, block)
    np.testing.assert_array_equal(coerced, block)


def test_binner_is_exact_across_c_f_and_strided_layouts():
    rng = np.random.default_rng(20260717)
    base = rng.normal(size=(500, 12))
    base[::31, 2] = np.nan
    fitted = Binner(max_bins=64).fit(base[:, ::2])
    expected = fitted.transform(np.ascontiguousarray(base[:, ::2]))

    for block in (
        np.asfortranarray(base[:, ::2]),
        base[:, ::2],
    ):
        actual = fitted.transform(block)
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.c_contiguous
