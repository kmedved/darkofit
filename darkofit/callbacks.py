"""Fit-time callbacks for observing and stopping boosting loops."""

from dataclasses import dataclass
import math
import time
from typing import Optional


@dataclass(frozen=True)
class BoostingProgress:
    """Immutable snapshot taken immediately before a boosting attempt.

    The scores, when present, describe the last productive round.  Callbacks
    are checked before gradient computation and tree construction, so a
    callback can stop a fit before it builds its first tree.
    """

    next_iteration: int
    iterations_attempted: int
    rounds_completed: int
    last_train_score: Optional[float]
    last_validation_score: Optional[float]


class WallClockStopper:
    """Stop between boosting rounds after a monotonic wall-clock budget.

    The limit is intentionally soft: a tree already being built is allowed to
    finish, and the deadline is checked before the next attempt.  Construction
    starts the clock so preprocessing performed before the first callback also
    consumes the supplied budget.
    """

    stop_reason = "time_limit"

    def __init__(self, time_limit_seconds, safety_margin=0.0):
        seconds = float(time_limit_seconds)
        safety_margin = float(safety_margin)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("time_limit_seconds must be finite and nonnegative")
        if not math.isfinite(safety_margin) or safety_margin < 0.0:
            raise ValueError(
                "safety_margin must be finite and nonnegative"
            )
        self._seconds = seconds
        self._safety_margin = safety_margin
        self._effective_seconds = max(0.0, seconds - safety_margin)
        self._started_at = time.monotonic()
        self._deadline = self._started_at + self._effective_seconds
        self._deadline_hit = False

    @property
    def seconds(self):
        return self._seconds

    @property
    def safety_margin(self):
        return self._safety_margin

    @property
    def effective_seconds(self):
        return self._effective_seconds

    @property
    def elapsed_seconds(self):
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def deadline_hit(self):
        return self._deadline_hit

    def check_deadline(self):
        """Refresh and return deadline state at a safe work boundary."""
        if time.monotonic() >= self._deadline:
            self._deadline_hit = True
        return self._deadline_hit

    def __call__(self, progress):
        del progress
        return self.check_deadline()


def _normalize_callbacks(callbacks):
    """Snapshot and validate the fit-only callback collection."""
    if callbacks is None:
        return ()
    if callable(callbacks):
        normalized = (callbacks,)
    else:
        try:
            normalized = tuple(callbacks)
        except TypeError as exc:
            raise TypeError(
                "callbacks must be callable or an iterable of callables"
            ) from exc
    for index, callback in enumerate(normalized):
        if not callable(callback):
            raise TypeError(f"callbacks[{index}] must be callable")
    return normalized


def _callback_stop_reason(callbacks, progress):
    """Run callbacks in order and return the first explicit stop reason."""
    for callback in callbacks:
        if callback(progress) is True:
            reason = str(getattr(callback, "stop_reason", "callback"))
            return reason if reason else "callback"
    return None
