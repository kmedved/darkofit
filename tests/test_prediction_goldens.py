"""Stable public prediction goldens for Phase-0 optimization safety."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks import prediction_goldens


GOLDEN_PATH = Path(__file__).with_name("golden_predictions.json")


@pytest.fixture(scope="module")
def expected_goldens():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def observed_goldens():
    return prediction_goldens.collect_prediction_goldens()


def test_prediction_golden_manifest_is_complete(
    expected_goldens, observed_goldens
):
    assert expected_goldens["schema_version"] == prediction_goldens.SCHEMA_VERSION
    assert expected_goldens["stable_decimals"] == (
        prediction_goldens.STABLE_DECIMALS
    )
    assert expected_goldens["baseline_darkofit_head"] == (
        "3295f70c231d4f7947e13a13ad77e3f2c19b3fe0"
    )
    assert expected_goldens["case_count"] == 12
    assert observed_goldens["case_count"] == 12
    assert set(expected_goldens["cases"]) == set(observed_goldens["cases"])

    kinds = {case["kind"] for case in expected_goldens["cases"].values()}
    assert kinds == {
        "regression",
        "regression_categorical",
        "binary_categorical",
        "multiclass",
        "distributional",
    }
    modes = {case["tree_mode"] for case in expected_goldens["cases"].values()}
    assert modes == {"catboost", "lightgbm", "hybrid", "depthwise"}
    losses = {
        case["loss"]
        for case in expected_goldens["cases"].values()
        if case["kind"] == "distributional"
    }
    assert losses == {
        "Gaussian",
        "LogNormal",
        "StudentT",
        "Poisson",
        "NegativeBinomial",
    }


def test_prediction_goldens_match_stable_outputs(
    expected_goldens, observed_goldens
):
    strict = os.environ.get("DARKOFIT_STRICT_GOLDENS") == "1"
    for case_name, expected_case in expected_goldens["cases"].items():
        observed_case = observed_goldens["cases"][case_name]
        assert observed_case["kind"] == expected_case["kind"]
        assert observed_case["tree_mode"] == expected_case["tree_mode"]
        assert set(observed_case["outputs"]) == set(expected_case["outputs"])
        for output_name, expected in expected_case["outputs"].items():
            observed = observed_case["outputs"][output_name]
            assert observed["shape"] == expected["shape"], (
                case_name,
                output_name,
            )
            assert observed["stable_decimals"] == expected["stable_decimals"], (
                case_name,
                output_name,
            )
            assert observed["stable_sha256"] == expected["stable_sha256"], (
                case_name,
                output_name,
            )
            stable_decimals = int(expected["stable_decimals"])
            np.testing.assert_allclose(
                observed["stable_values"],
                expected["stable_values"],
                rtol=0.0,
                atol=0.5 * 10 ** (-stable_decimals),
                err_msg=f"{case_name}/{output_name}",
            )
            if strict:
                assert observed["exact_sha256"] == expected["exact_sha256"], (
                    case_name,
                    output_name,
                )
