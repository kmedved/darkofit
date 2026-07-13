"""Pure benchmark-only timing helpers for feature preprocessing."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Mapping


_ACTIVE_PREPROCESSING_CAPTURE = ContextVar(
    "same_machine_preprocessing_capture", default=None
)


@contextmanager
def capture_preprocessing(package: str):
    """Capture cumulative instrumented ``fit_transform`` time for one fit."""
    state = {"package": package, "seconds": 0.0, "calls": 0}
    token = _ACTIVE_PREPROCESSING_CAPTURE.set(state)
    try:
        yield state
    finally:
        _ACTIVE_PREPROCESSING_CAPTURE.reset(token)


@contextmanager
def instrument_feature_preprocessors(
    preprocessors: Mapping[str, type],
    *,
    clock=time.perf_counter,
):
    """Temporarily time each package's ``FeaturePreprocessor.fit_transform``.

    The active per-model capture is a context variable, so calls made by both
    constant- and linear-leaf selection lanes accumulate into the same child
    model total. Originals are restored even when a fit raises.
    """
    originals = []
    if len(set(preprocessors.values())) != len(preprocessors):
        raise ValueError("each package must provide a distinct preprocessor class")
    try:
        for package, preprocessor in preprocessors.items():
            original = preprocessor.fit_transform
            if getattr(original, "_same_machine_timed", False):
                raise RuntimeError(
                    f"{preprocessor.__name__}.fit_transform is already instrumented"
                )

            @wraps(original)
            def timed(self, *args, __original=original, __package=package, **kwargs):
                started = clock()
                try:
                    return __original(self, *args, **kwargs)
                finally:
                    elapsed = clock() - started
                    state = _ACTIVE_PREPROCESSING_CAPTURE.get()
                    if state is not None and state["package"] == __package:
                        state["seconds"] += elapsed
                        state["calls"] += 1

            timed._same_machine_timed = True
            originals.append((preprocessor, original))
            preprocessor.fit_transform = timed
        yield
    finally:
        for preprocessor, original in reversed(originals):
            preprocessor.fit_transform = original
