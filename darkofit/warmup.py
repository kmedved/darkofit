"""Pre-compile DarkoFit's default-path Numba kernels.

Fresh Python workers otherwise pay Numba's compile or cache-load cost inside
their first fit and prediction. ``warmup()`` moves that work to an explicit
startup phase by running three deterministic, tiny fits. It intentionally
does not cover opt-in distributional, SHAP, local-linear-leaf, or
non-oblivious paths.

The public API and environment dispatch are adapted from ChimeraBoost's
Apache-2.0 warmup helper at commit
851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d. See ``NOTICE``.
"""

import threading
import time

import numba
import numpy as np

from .sklearn_api import DarkoClassifier, DarkoRegressor


def warmup(verbose=False, background=False):
    """Compile or load kernels used by DarkoFit's default estimators.

    The warmup covers scalar regression, binary and multiclass
    classification, categorical ordered target statistics, validation
    prediction, oblivious-tree construction, and constant-leaf packed
    prediction. Call it during worker startup, outside latency that should
    represent a real fit or prediction.

    Set ``DARKOFIT_WARMUP=1`` to run this function while importing
    :mod:`darkofit`, or ``DARKOFIT_WARMUP=background`` to start it in a
    daemon thread. Unset, empty, and ``0`` values leave ordinary imports
    unchanged.

    Parameters
    ----------
    verbose : bool, default False
        Print cumulative per-stage timings.
    background : bool, default False
        Run in a daemon thread and return it immediately.

    Returns
    -------
    float or threading.Thread
        Elapsed wall-clock seconds, or the started thread in background mode.
    """
    if background:
        thread = threading.Thread(
            target=warmup,
            kwargs={"verbose": verbose},
            name="darkofit-warmup",
            daemon=True,
        )
        thread.start()
        return thread

    started = time.perf_counter()
    rng = np.random.default_rng(0)
    previous_threads = numba.get_num_threads()
    # More than four threads selects the separate contiguous histogram
    # buffers used by full-machine default fits. The compiled signature is
    # identical above that boundary, so cap the tiny synthetic work at 18.
    warmup_threads = min(18, numba.config.NUMBA_NUM_THREADS)

    def log(stage):
        if verbose:
            elapsed = time.perf_counter() - started
            print(f"darkofit.warmup: {stage} ({elapsed:.2f}s)")

    common = {
        "iterations": 2,
        "learning_rate": 0.1,
        "thread_count": warmup_threads,
        "random_state": 0,
        "diagnostic_warnings": "never",
    }

    try:
        # Binary classification activates ordered target statistics and the
        # ordered leaf-update path. The explicit validation set also compiles
        # the validation-prediction lane.
        n_rows = 320
        numeric = rng.standard_normal((n_rows, 3))
        category = rng.integers(0, 3, size=n_rows).astype(np.float64)
        X = np.column_stack((numeric, category))
        y_binary = (X[:, 0] + X[:, 1] > 0.0).astype(np.int64)
        binary = DarkoClassifier(**common)
        binary.fit(
            X[32:],
            y_binary[32:],
            cat_features=[3],
            eval_set=(X[:32], y_binary[:32]),
        )
        binary.predict_proba(X[:8])
        log("binary + categorical + validation")

        # Default CatBoost-mode multiclass uses scalar per-class oblivious
        # trees and the class-major packed prediction kernel.
        y_multiclass = np.digitize(X[:, 0], (-0.5, 0.5))
        multiclass = DarkoClassifier(**common)
        multiclass.fit(X[:, :3], y_multiclass)
        multiclass.predict_proba(X[:8, :3])
        log("multiclass")

        # Scalar regression matches the path used by the basketball default:
        # CatBoost mode, constant leaves, and ordered boosting resolved off.
        y_regression = X[:, 0] + 0.1 * rng.standard_normal(n_rows)
        regression = DarkoRegressor(**common)
        regression.fit(X[:, :3], y_regression)
        regression.predict(X[:8, :3])
        if warmup_threads > 1:
            # Packed scalar prediction switches kernels at 8,192 rows.
            # Touch the row-parallel twin without adding another fit.
            large_batch = np.tile(X[:8, :3], (1024, 1))
            regression.predict(large_batch)
        log("scalar regression")
    finally:
        numba.set_num_threads(previous_threads)

    return time.perf_counter() - started


def _warmup_from_env(value):
    """Dispatch the opt-in ``DARKOFIT_WARMUP`` import setting."""
    if not value or value.strip() == "0":
        return None
    if value.strip().lower() in {"background", "thread", "bg"}:
        return warmup(background=True)
    return warmup()
