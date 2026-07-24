from __future__ import annotations

import numpy as np

from benchmarks import run_declared_ordinal_transform_microbenchmark as bench


def test_benchmark_representations_have_equal_targets_and_values():
    categorical = bench._frame(128, categorical=True)
    generic = categorical.astype({
        name: object for name in bench.CATEGORIES
    })
    assert categorical.shape == generic.shape == (128, 7)
    assert np.array_equal(bench._target(categorical), bench._target(generic))
    for name in categorical.columns:
        assert np.array_equal(
            categorical[name].astype(object).to_numpy(),
            generic[name].to_numpy(),
        )


def test_benchmark_geometric_mean():
    assert bench._geometric_mean([0.5, 2.0]) == 1.0
