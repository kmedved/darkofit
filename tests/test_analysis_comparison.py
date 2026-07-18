"""Regression tests for portable campaign-analysis comparisons."""

from __future__ import annotations

import pytest


def test_analysis_comparison_allows_platform_float_rounding(
    assert_analysis_equal,
):
    stored = {"metric": [1.0]}
    regenerated = {"metric": [1.0 + 2e-15]}

    assert_analysis_equal(stored, regenerated)


@pytest.mark.parametrize(
    ("stored", "regenerated"),
    [
        (1, True),
        (1, 1.0),
        ([1], (1,)),
    ],
)
def test_analysis_comparison_rejects_type_changes(
    assert_analysis_equal,
    stored,
    regenerated,
):
    with pytest.raises(AssertionError, match=r"result: .* != .*"):
        assert_analysis_equal(stored, regenerated)
