"""Tests for the DarkoFit phase profiler helper."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import profile_darkofit_phases as phases  # noqa: E402


def test_phase_profiler_rejects_auto_tree_mode():
    with pytest.raises(SystemExit):
        phases.parse_args(["--tree-mode", "auto"])


def test_phase_profiler_fit_disables_best_model(monkeypatch):
    seen_kwargs = []

    class FakeRegressor:
        def __init__(self, **kwargs):
            seen_kwargs.append(kwargs)
            self.timing_ = {phase: 0.0 for phase in phases.PHASES}
            self.best_iteration_ = int(kwargs["iterations"])

        def fit(self, *args, **kwargs):
            return self

    monkeypatch.setattr(phases, "DarkoRegressor", FakeRegressor)
    monkeypatch.setitem(phases.bench.SIZE_SAMPLES, "tiny", 20)
    monkeypatch.setattr(
        phases.bench,
        "_split_for_task",
        lambda X, y, task, seed: (X[:16], X[16:], y[:16], y[16:]),
    )
    monkeypatch.setattr(
        phases.bench,
        "_validation_split",
        lambda X, y, task, seed: (X[:12], X[12:], y[:12], y[12:]),
    )

    spec = SimpleNamespace(
        name="synthetic",
        task="regression",
        builder=lambda n, rng: (
            np.zeros((n, 2), dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            None,
        ),
    )
    args = SimpleNamespace(
        data_seed=0,
        iterations=3,
        learning_rate=0.1,
        depth=2,
        num_leaves=None,
        min_child_samples=1,
        min_gain_to_split=0.0,
        no_ordered_boosting=True,
        tree_mode="catboost",
        repeat=1,
    )

    row = phases._run_fit(spec, "tiny", thread_count=1, args=args, seed=0)

    assert seen_kwargs[0]["use_best_model"] is False
    assert row["iterations_run"] == 3

