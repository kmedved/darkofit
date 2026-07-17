"""Coordinate DarkoFit's opt-in background Numba warmup."""

from contextlib import contextmanager
import threading


_CONDITION = threading.Condition()
_BACKGROUND_THREAD = None
_THREAD_STATE = threading.local()


def _run_background_warmup(target, kwargs):
    global _BACKGROUND_THREAD
    _THREAD_STATE.in_background_warmup = True
    try:
        target(**kwargs)
    finally:
        _THREAD_STATE.in_background_warmup = False
        with _CONDITION:
            if _BACKGROUND_THREAD is threading.current_thread():
                _BACKGROUND_THREAD = None
            _CONDITION.notify_all()


def start_background_warmup(target, *, kwargs=None, preflight=None):
    """Start one daemon warmup, returning an existing run when present."""
    global _BACKGROUND_THREAD
    with _CONDITION:
        if _BACKGROUND_THREAD is not None:
            return _BACKGROUND_THREAD
        if preflight is not None:
            preflight()
        thread = threading.Thread(
            target=_run_background_warmup,
            args=(target, dict(kwargs or {})),
            name="darkofit-warmup",
            daemon=True,
        )
        _BACKGROUND_THREAD = thread
        try:
            thread.start()
        except BaseException:
            _BACKGROUND_THREAD = None
            _CONDITION.notify_all()
            raise
        return thread


@contextmanager
def numba_thread_setup():
    """Prevent foreground DarkoFit work from overlapping background warmup."""
    with _CONDITION:
        if not getattr(_THREAD_STATE, "in_background_warmup", False):
            while _BACKGROUND_THREAD is not None:
                _CONDITION.wait()
        yield
