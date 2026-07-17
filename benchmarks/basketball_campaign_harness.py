"""Forward-only timing gates for new basketball campaigns.

The historical ``basketball_harness.py`` is bound byte-for-byte into frozen
campaign artifacts. New protocols import this module for corrected gate
semantics so old evidence remains reproducible without rebinding its support
manifest.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


DEFAULT_TIMING_REPETITIONS = 3
MAX_PAIRED_RATIO_IQR_OVER_MEDIAN = 0.10


def paired_ratio_summary(
    numerator_values: Iterable[float],
    denominator_values: Iterable[float],
    *,
    repetitions: int = DEFAULT_TIMING_REPETITIONS,
    max_iqr_over_median: float = MAX_PAIRED_RATIO_IQR_OVER_MEDIAN,
    median_seconds_budget: float | None = None,
) -> dict[str, Any]:
    """Summarize stability from within-block arm ratios.

    The input series must come from reciprocal fresh-worker blocks measured
    under the same conditions. An optional seconds-scale budget constrains the
    numerator median but never participates in the stability calculation.
    """
    numerator = np.asarray(list(numerator_values), dtype=np.float64)
    denominator = np.asarray(list(denominator_values), dtype=np.float64)
    expected = int(repetitions)
    if numerator.shape != (expected,) or denominator.shape != (expected,):
        raise RuntimeError(
            f"paired timing requires exactly {expected} values per arm"
        )
    if (
        not np.all(np.isfinite(numerator))
        or not np.all(np.isfinite(denominator))
        or np.any(numerator <= 0.0)
        or np.any(denominator <= 0.0)
    ):
        raise RuntimeError("paired timing values must be positive and finite")
    max_iqr_over_median = float(max_iqr_over_median)
    if not math.isfinite(max_iqr_over_median) or max_iqr_over_median < 0.0:
        raise ValueError("max_iqr_over_median must be finite and nonnegative")
    if median_seconds_budget is not None:
        median_seconds_budget = float(median_seconds_budget)
        if (
            not math.isfinite(median_seconds_budget)
            or median_seconds_budget <= 0.0
        ):
            raise ValueError(
                "median_seconds_budget must be positive and finite"
            )

    ratios = numerator / denominator
    median_ratio = float(np.median(ratios))
    q1, q3 = (
        float(value) for value in np.percentile(ratios, [25.0, 75.0])
    )
    iqr = q3 - q1
    iqr_over_median = iqr / median_ratio
    numerator_median = float(np.median(numerator))
    return {
        "repetitions": expected,
        "numerator_values_seconds": numerator.tolist(),
        "denominator_values_seconds": denominator.tolist(),
        "paired_ratios": ratios.tolist(),
        "median_ratio": median_ratio,
        "q1_ratio": q1,
        "q3_ratio": q3,
        "iqr_ratio": iqr,
        "iqr_over_median": iqr_over_median,
        "max_iqr_over_median": max_iqr_over_median,
        "stable": iqr_over_median <= max_iqr_over_median,
        "numerator_median_seconds": numerator_median,
        "median_seconds_budget": median_seconds_budget,
        "seconds_budget_passed": (
            None
            if median_seconds_budget is None
            else numerator_median <= median_seconds_budget
        ),
    }
