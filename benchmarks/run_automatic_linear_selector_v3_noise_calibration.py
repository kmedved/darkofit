#!/usr/bin/env python3
"""Record selector-v3 noise statistics on the spent non-Protein M6 grid."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import m6_quality_rule_v3 as rule
    import run_automatic_linear_selector_v2_m6_engagement as foundation
except ImportError:  # pragma: no cover
    from benchmarks import m6_quality_rule_v3 as rule
    from benchmarks import (
        run_automatic_linear_selector_v2_m6_engagement as foundation,
    )


RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_PATH = RUNNER_PATH.with_name(
    "automatic_linear_selector_v3_noise_calibration.md"
)
REGRESSION_DATASETS = (
    "diabetes_resampled",
    "friedman_numeric",
    "wide_numeric_reg",
    "categorical_reg",
)
SELECTOR_FIELDS = {
    "version",
    "requested",
    "fit_random_state_seed",
    "eligible",
    "resolved_linear_leaves",
    "final_booster_linear_leaves",
    "final_linear_leaves_active",
    "reason",
    "minimum_relative_improvement",
    "minimum_gain_z",
    "split",
    "constant_validation_rmse",
    "linear_validation_rmse",
    "relative_validation_improvement",
    "paired_mse_gain",
    "paired_mse_gain_standard_error",
    "paired_mse_gain_z",
    "selection_fits",
    "selection_total_seconds",
}


def expected_identities():
    return tuple(
        (dataset, size, seed, weight_mode)
        for size in rule.SIZES
        for dataset in REGRESSION_DATASETS
        for seed in rule.SEEDS
        for weight_mode in rule.WEIGHT_MODES
    )


def configure_foundation() -> None:
    foundation.IDENTITY = (
        "automatic-linear-selector-v3-noise-calibration-2se-20260723"
    )
    foundation.MECHANISM_ID = "automatic_linear_selector_v3"
    foundation.INSPECTION_INDEX = 2
    foundation.RUNNER_PATH = RUNNER_PATH
    foundation.PROTOCOL_PATH = PROTOCOL_PATH
    foundation.SELECTOR_CONTRACT_PATH = PROTOCOL_PATH
    foundation._SELECTOR_FIELDS = SELECTOR_FIELDS
    foundation.expected_identities = expected_identities


def main() -> None:
    configure_foundation()
    foundation.main(sys.argv[1:])


if __name__ == "__main__":
    main()
