"""The public warmup must cover defaults without changing caller state."""

import os
from importlib import import_module
from pathlib import Path
import subprocess
import sys
import threading

import numba
import numpy as np
import pytest

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


def test_background_warmup_is_single_flight_and_safe_with_immediate_fit(
    tmp_path,
):
    cache = tmp_path / "numba-cache"
    cache.mkdir()
    env = os.environ.copy()
    env["DARKOFIT_WARMUP"] = "0"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["NUMBA_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = str(REPO_ROOT)
    code = """
import threading
import numpy as np
from darkofit import DarkoRegressor, warmup

first = warmup(background=True)
second = warmup(background=True)
assert first is second
assert isinstance(first, threading.Thread)
assert first.daemon

X = np.random.default_rng(7).normal(size=(4000, 8))
y = X[:, 0] - 0.3 * X[:, 1]
params = dict(
    iterations=20,
    learning_rate=0.1,
    thread_count=4,
    random_state=0,
    diagnostic_warnings="never",
)
model = DarkoRegressor(**params).fit(X, y)
prediction = model.predict(X[:8])
assert prediction.shape == (8,)

first.join(timeout=300)
assert not first.is_alive()
reference = DarkoRegressor(**params).fit(X, y).predict(X[:8])
np.testing.assert_array_equal(prediction, reference)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_background_env_is_safe_with_immediate_fit(tmp_path):
    cache = tmp_path / "numba-cache"
    cache.mkdir()
    env = os.environ.copy()
    env["DARKOFIT_WARMUP"] = "background"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["NUMBA_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = str(REPO_ROOT)
    code = """
import numpy as np
import darkofit

X = np.random.default_rng(7).normal(size=(4000, 8))
y = X[:, 0] - 0.3 * X[:, 1]
model = darkofit.DarkoRegressor(
    iterations=20,
    learning_rate=0.1,
    thread_count=4,
    random_state=0,
    diagnostic_warnings="never",
).fit(X, y)
assert model.predict(X[:8]).shape == (8,)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_background_warmup_rejects_active_workqueue():
    numba.get_num_threads()
    if numba.threading_layer() != "workqueue":
        pytest.skip("active Numba layer is thread-safe")
    with pytest.raises(RuntimeError, match="workqueue.*initialized"):
        warmup(background=True)


def test_warmup_env_dispatch(monkeypatch):
    calls = []

    def fake_warmup(**kwargs):
        calls.append(kwargs)
        return "result"

    monkeypatch.setattr(warmup_module, "warmup", fake_warmup)
    for value in (None, "", "0", " 0 ", "false", " FALSE ", "off", "No"):
        assert warmup_module._warmup_from_env(value) is None
    for value in ("1", " true ", "ON", "yes"):
        assert warmup_module._warmup_from_env(value) == "result"
    for value in ("background", " Thread ", "BG"):
        assert warmup_module._warmup_from_env(value) == "result"
    with pytest.warns(RuntimeWarning, match="unrecognized DARKOFIT_WARMUP"):
        assert warmup_module._warmup_from_env("sometimes") is None
    assert calls == [
        {},
        {},
        {},
        {},
        {"background": True},
        {"background": True},
        {"background": True},
    ]


def test_warmup_env_background_failure_warns_and_skips(monkeypatch):
    def unsafe_warmup(**kwargs):
        assert kwargs == {"background": True}
        raise RuntimeError("unsafe background")

    monkeypatch.setattr(warmup_module, "warmup", unsafe_warmup)
    with pytest.warns(RuntimeWarning, match="was skipped.*unsafe background"):
        assert warmup_module._warmup_from_env("background") is None


def test_numba_thread_masks_are_thread_local():
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("thread-local mask check requires at least two threads")
    original = numba.get_num_threads()
    main_threads = min(2, numba.config.NUMBA_NUM_THREADS)
    worker_threads = min(3, numba.config.NUMBA_NUM_THREADS)
    observed = []

    try:
        numba.set_num_threads(main_threads)

        def set_worker_threads():
            observed.append(numba.get_num_threads())
            numba.set_num_threads(worker_threads)
            observed.append(numba.get_num_threads())

        thread = threading.Thread(target=set_worker_threads)
        thread.start()
        thread.join()

        assert observed == [numba.config.NUMBA_NUM_THREADS, worker_threads]
        assert numba.get_num_threads() == main_threads
    finally:
        numba.set_num_threads(original)


def test_darkofit_thread_setup_waits_for_background_single_flight():
    from darkofit._numba_runtime import start_background_warmup
    from darkofit.booster import _apply_thread_count

    background_started = threading.Event()
    release_background = threading.Event()
    foreground_started = threading.Event()
    foreground_finished = threading.Event()
    original = numba.get_num_threads()

    def held_background():
        background_started.set()
        assert release_background.wait(timeout=30)

    def apply_foreground_threads():
        foreground_started.set()
        _apply_thread_count(min(2, numba.config.NUMBA_NUM_THREADS))
        foreground_finished.set()

    background = start_background_warmup(held_background)
    foreground = threading.Thread(target=apply_foreground_threads)
    try:
        assert background_started.wait(timeout=30)
        foreground.start()
        assert foreground_started.wait(timeout=30)
        assert not foreground_finished.wait(timeout=0.1)
    finally:
        release_background.set()
        background.join(timeout=30)
        if foreground.ident is not None:
            foreground.join(timeout=30)
        numba.set_num_threads(original)
    assert not background.is_alive()
    assert not foreground.is_alive()
    assert foreground_finished.is_set()


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


@pytest.mark.parametrize("env_value", [None, "false", " OFF ", "No"])
def test_ordinary_import_does_not_run_warmup(tmp_path, env_value):
    cache = tmp_path / "numba-cache"
    cache.mkdir()
    env = os.environ.copy()
    if env_value is None:
        env.pop("DARKOFIT_WARMUP", None)
    else:
        env["DARKOFIT_WARMUP"] = env_value
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
