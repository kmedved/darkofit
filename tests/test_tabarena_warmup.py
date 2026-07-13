import json

import numba
import numpy as np
import pytest

import darkofit.flat_model as flat_model
from benchmarks.tabarena_warmup import warmup_tabarena_regression


def _assert_legacy_rng_state_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _stable_stage_metadata(metadata):
    stable = []
    for stage in metadata["stages"]:
        item = {
            key: value for key, value in stage.items() if key != "fit_seconds"
        }
        item["prediction_batches"] = [
            {
                key: value
                for key, value in batch.items()
                if key != "predict_seconds"
            }
            for batch in stage["prediction_batches"]
        ]
        stable.append(item)
    return stable


def test_tabarena_regression_warmup_matches_frozen_protocol_and_is_deterministic(
    monkeypatch,
):
    np.random.seed(90210)
    rng_before = np.random.get_state()
    threads_before = numba.get_num_threads()

    calls = []
    serial_kernel = flat_model._flat_oblivious_add
    parallel_kernel = flat_model._flat_oblivious_add_parallel

    def record_serial(*args):
        calls.append(("flat_serial", int(args[0].shape[0])))
        return serial_kernel(*args)

    def record_parallel(*args):
        calls.append(("flat_parallel", int(args[0].shape[0])))
        return parallel_kernel(*args)

    monkeypatch.setattr(flat_model, "_flat_oblivious_add", record_serial)
    monkeypatch.setattr(
        flat_model, "_flat_oblivious_add_parallel", record_parallel
    )

    first = warmup_tabarena_regression(thread_count=2)
    second = warmup_tabarena_regression(thread_count=2)

    _assert_legacy_rng_state_equal(rng_before, np.random.get_state())
    assert numba.get_num_threads() == threads_before
    assert first["schema_version"] == 2
    assert first["clock"] == "time.monotonic_ns"
    assert first["duration_seconds"] >= 0.0
    assert first["config"] == {
        "iterations": 5,
        "loss": "RMSE",
        "tree_mode": "catboost",
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "max_bins": 128,
        "learning_rate": 0.1,
        "ts_permutations": 1,
        "ordered_boosting": "auto",
        "sampling": "uniform",
        "linear_residual": False,
        "early_stopping": False,
        "use_best_model": False,
        "diagnostic_warnings": "never",
        "random_state": 20260713,
        "thread_count": 2,
    }
    assert [stage["name"] for stage in first["stages"]] == [
        "numeric",
        "categorical",
    ]
    assert _stable_stage_metadata(first) == _stable_stage_metadata(second)

    numeric, categorical = first["stages"]
    assert numeric["categorical_features"] == []
    assert categorical["categorical_features"] == [12]
    for stage in first["stages"]:
        assert stage["train_rows"] == 2048
        assert stage["validation_rows"] == 512
        assert stage["fit_seconds"] >= 0.0
        assert stage["iterations_fitted"] == 5
        assert stage["tree_depths"] == [6, 6, 6, 6, 6]
        assert stage["resolved_learning_rate"] == 0.1
        assert stage["resolved_tree_mode"] == "catboost"
        assert stage["resolved_ordered_boosting"] is False
        assert stage["resolved_thread_count"] == 2
        assert stage["flat_ensemble_type"] == "FlatObliviousEnsemble"
        assert stage["flat_prediction_router_selected"] is True
        assert (
            stage["prediction_parallel_min_rows"]
            == flat_model._PARALLEL_MIN_ROWS
        )
        serial, parallel = stage["prediction_batches"]
        assert serial["name"] == "serial_subthreshold"
        assert serial["route"] == "flat_serial"
        expected_features = 12 + bool(stage["categorical_features"])
        assert serial["input_shape"] == [
            flat_model._PARALLEL_MIN_ROWS - 1,
            expected_features,
        ]
        assert serial["prediction_shape"] == [flat_model._PARALLEL_MIN_ROWS - 1]
        assert parallel["name"] == "parallel_at_threshold"
        assert parallel["route"] == "flat_parallel"
        assert parallel["input_shape"] == [
            flat_model._PARALLEL_MIN_ROWS,
            expected_features,
        ]
        assert parallel["prediction_shape"] == [flat_model._PARALLEL_MIN_ROWS]
        for batch in stage["prediction_batches"]:
            assert batch["predict_seconds"] >= 0.0
            assert len(batch["prediction_sha256"]) == 64

    expected_calls = [
        ("flat_serial", flat_model._PARALLEL_MIN_ROWS - 1),
        ("flat_parallel", flat_model._PARALLEL_MIN_ROWS),
    ] * 4
    assert calls == expected_calls

    # allow_nan=False also rejects non-finite timing values.
    json.dumps(first, allow_nan=False)


def test_tabarena_regression_warmup_preserves_one_thread_public_route():
    metadata = warmup_tabarena_regression(thread_count=1)

    assert metadata["config"]["thread_count"] == 1
    for stage in metadata["stages"]:
        assert stage["resolved_thread_count"] == 1
        assert stage["flat_prediction_router_selected"] is False
        assert [batch["route"] for batch in stage["prediction_batches"]] == [
            "tree_loop",
            "tree_loop",
        ]


@pytest.mark.parametrize("thread_count", [0, -1])
def test_tabarena_regression_warmup_rejects_nonpositive_threads(thread_count):
    with pytest.raises(ValueError, match="at least 1"):
        warmup_tabarena_regression(thread_count=thread_count)


@pytest.mark.parametrize("thread_count", [True, 1.5, "2"])
def test_tabarena_regression_warmup_rejects_noninteger_threads(thread_count):
    with pytest.raises(TypeError, match="positive integer"):
        warmup_tabarena_regression(thread_count=thread_count)
