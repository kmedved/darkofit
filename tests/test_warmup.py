"""The public warmup must cover defaults without changing caller state."""

import os
from importlib import import_module
from pathlib import Path
import subprocess
import sys
import threading

import numba
import numpy as np

from darkofit import DarkoClassifier, warmup


REPO_ROOT = Path(__file__).resolve().parents[1]
warmup_module = import_module("darkofit.warmup")


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_warmup_compiles_representative_default_path_kernels():
    elapsed = warmup()
    assert elapsed > 0.0

    from darkofit.binning import _bin_rows_into
    from darkofit.flat_model import (
        _flat_oblivious_add,
        _flat_oblivious_add_parallel,
        _flat_oblivious_class_add_parallel,
    )
    from darkofit.losses import (
        _logloss_grad_hess_into,
        _softmax_class_major_grad_hess_into,
    )
    from darkofit.target_encoding import _ordered_ts
    from darkofit.tree import (
        _build_histograms_unit_hess_and_best_split,
        _update_leaves_with_split_serial,
        ordered_leaf_update_inplace,
    )

    kernels = (
        _bin_rows_into,
        _logloss_grad_hess_into,
        _softmax_class_major_grad_hess_into,
        _ordered_ts,
        _update_leaves_with_split_serial,
        ordered_leaf_update_inplace,
    )
    if numba.config.NUMBA_NUM_THREADS > 1:
        kernels += (
            _flat_oblivious_add,
            _flat_oblivious_add_parallel,
            _flat_oblivious_class_add_parallel,
            _build_histograms_unit_hess_and_best_split,
        )
    for kernel in kernels:
        assert kernel.signatures, f"{kernel.py_func.__name__} was not compiled"
    if numba.config.NUMBA_NUM_THREADS > 4:
        assert any(
            signature[4].layout == signature[5].layout == "C"
            for signature in _build_histograms_unit_hess_and_best_split.signatures
        ), "full-machine contiguous-histogram signature was not compiled"


def test_background_warmup_returns_daemon_thread_and_finishes():
    thread = warmup(background=True)
    assert isinstance(thread, threading.Thread)
    assert thread.daemon
    thread.join(timeout=300)
    assert not thread.is_alive()


def test_warmup_env_dispatch(monkeypatch):
    calls = []

    def fake_warmup(**kwargs):
        calls.append(kwargs)
        return "result"

    monkeypatch.setattr(warmup_module, "warmup", fake_warmup)
    assert warmup_module._warmup_from_env(None) is None
    assert warmup_module._warmup_from_env("") is None
    assert warmup_module._warmup_from_env("0") is None
    assert warmup_module._warmup_from_env(" 0 ") is None
    assert warmup_module._warmup_from_env("1") == "result"
    assert warmup_module._warmup_from_env("background") == "result"
    assert calls == [{}, {"background": True}]


def test_warmup_preserves_rng_threads_and_deterministic_output():
    np.random.seed(90210)
    rng_before = np.random.get_state()
    threads_before = numba.get_num_threads()

    X = np.random.default_rng(7).standard_normal((300, 4))
    y = (X[:, 0] > 0.0).astype(np.int64)
    params = {
        "iterations": 20,
        "learning_rate": 0.1,
        "thread_count": min(2, numba.config.NUMBA_NUM_THREADS),
        "random_state": 3,
        "diagnostic_warnings": "never",
    }
    reference = DarkoClassifier(**params).fit(X, y).predict_proba(X)
    numba.set_num_threads(threads_before)

    warmup()

    _assert_rng_state_equal(rng_before, np.random.get_state())
    assert numba.get_num_threads() == threads_before
    candidate = DarkoClassifier(**params).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(reference, candidate)


def test_ordinary_import_does_not_run_warmup(tmp_path):
    cache = tmp_path / "numba-cache"
    cache.mkdir()
    env = os.environ.copy()
    env.pop("DARKOFIT_WARMUP", None)
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = str(REPO_ROOT)
    code = (
        "import darkofit; "
        "from darkofit.tree import "
        "_build_histograms_unit_hess_and_best_split as kernel; "
        "assert not kernel.signatures"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert not list(cache.rglob("*.nbc"))
