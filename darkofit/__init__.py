"""DarkoFit: machine learning for tabular data in pure Python.

The current gradient-boosting engine borrows several ideas from CatBoost:
  * Ordered target statistics for categorical features (anti-leakage encoding)
  * Oblivious / symmetric trees (fast, strongly regularized -> good defaults)
  * Histogram-based quantized splitting (numba accelerated)

Public API:
  >>> from darkofit import DarkoRegressor, DarkoClassifier
  >>> model = DarkoClassifier().fit(X, y, cat_features=[0, 3])
  >>> proba = model.predict_proba(X_test)
"""

import os as _os

# Single source of truth for the package version (pyproject reads this).
# Defined before submodule imports so they may reference it safely.
__version__ = "0.12.0"

from .sklearn_api import (
    DarkoRegressor,
    DarkoClassifier,
)
from .callbacks import BoostingProgress, WallClockStopper
from .warmup import warmup, _warmup_from_env

# Opt-in startup compilation for fresh workers. Ordinary imports remain cold.
_warmup_from_env(_os.environ.get("DARKOFIT_WARMUP"))

__all__ = [
    "DarkoRegressor",
    "DarkoClassifier",
    "BoostingProgress",
    "WallClockStopper",
    "warmup",
]
