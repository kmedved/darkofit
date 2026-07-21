import numba
import numpy as np

from darkofit.losses import _logloss_eval


def test_logloss_eval_is_bitwise_deterministic_across_calls_and_thread_counts():
    rng = np.random.default_rng(20260721)
    y = rng.integers(0, 2, size=10_003).astype(np.float64)
    raw = rng.normal(size=y.shape[0])
    weights = rng.uniform(0.25, 2.0, size=y.shape[0])
    previous_threads = numba.get_num_threads()
    available_threads = int(numba.config.NUMBA_NUM_THREADS)
    try:
        values = []
        for thread_count in sorted({1, min(4, available_threads), available_threads}):
            numba.set_num_threads(thread_count)
            values.extend(
                float(_logloss_eval(y, raw, weights)).hex()
                for _ in range(20)
            )
        assert len(set(values)) == 1
    finally:
        numba.set_num_threads(previous_threads)
