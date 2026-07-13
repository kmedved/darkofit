"""Optional AutoGluon integration tests for the TabArena adapter."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("autogluon.core")

from benchmarks.tabarena_adapter import DarkoFitModel  # noqa: E402
from darkofit import DarkoRegressor  # noqa: E402


FIXED_HYPERPARAMETERS = {
    "iterations": 5,
    "learning_rate": 0.1,
    "depth": 3,
    "l2_leaf_reg": 3.0,
    "max_bins": 128,
    "tree_mode": "catboost",
    "ts_permutations": 1,
    "early_stopping": False,
    "random_state": 7,
}


@pytest.fixture
def regression_data():
    """Return deterministic train/validation data with a native category."""
    n_rows = 72
    x = np.linspace(-3.0, 3.0, n_rows)
    category_code = np.arange(n_rows) % 3
    X = pd.DataFrame(
        {
            "x": x,
            "x2": np.sin(x),
            "category": pd.Series(
                np.asarray(["a", "b", "c"])[category_code],
                dtype="category",
            ),
        }
    )
    y = pd.Series(2.0 * x + 0.4 * np.sin(3.0 * x) + 0.2 * category_code)
    split = 54
    return (
        X.iloc[:split].reset_index(drop=True),
        y.iloc[:split].reset_index(drop=True),
        X.iloc[split:].reset_index(drop=True),
        y.iloc[split:].reset_index(drop=True),
    )


def _new_model(tmp_path, name):
    return DarkoFitModel(
        path=str(tmp_path),
        name=name,
        problem_type="regression",
        hyperparameters=dict(FIXED_HYPERPARAMETERS),
    )


def _fit_kwargs(regression_data, *, time_limit):
    X, y, X_val, y_val = regression_data
    return {
        "X": X,
        "y": y,
        "X_val": X_val,
        "y_val": y_val,
        "time_limit": time_limit,
        "num_cpus": 1,
        "num_gpus": 0,
    }


def test_regression_fit_records_refit_params_and_persistent_metadata(
    tmp_path,
    regression_data,
):
    model = _new_model(tmp_path, "darkofit_metadata")
    fit_kwargs = _fit_kwargs(regression_data, time_limit=None)

    model.fit(**fit_kwargs)

    constructor_parameters = inspect.signature(DarkoRegressor.__init__).parameters
    assert set(model.params_trained) <= set(constructor_parameters)
    assert set(model.params_trained) == {
        "iterations",
        "learning_rate",
        "tree_mode",
        "early_stopping",
        "early_stopping_rounds",
        "use_best_model",
        "refit",
        "depth",
        "num_leaves",
        "l2_leaf_reg",
        "min_child_samples",
        "min_child_weight",
        "cat_smoothing",
    }
    assert model.params_trained["iterations"] == 5
    assert model.params_trained["learning_rate"] == 0.1
    assert model.params_trained["tree_mode"] == "catboost"
    assert model.params_trained["early_stopping"] is False
    assert model.params_trained["early_stopping_rounds"] is None
    assert model.params_trained["use_best_model"] is False
    assert model.params_trained["refit"] is False
    assert model.params_trained["depth"] == 3
    assert model.params_trained["l2_leaf_reg"] == 3.0
    assert type(model.params_trained["iterations"]) is int
    assert type(model.params_trained["learning_rate"]) is float
    assert type(model.params_trained["tree_mode"]) is str

    metadata = model.get_fit_metadata()["darkofit_fit"]
    assert metadata == {
        "iterations_requested": 5,
        "iterations_attempted": 5,
        "rounds_completed": 5,
        "rounds_retained": 5,
        "best_iteration": 5,
        "resolved_learning_rate": 0.1,
        "requested_tree_mode": "catboost",
        "selected_tree_mode": "catboost",
        "selected_lane": "boosting",
        "linear_residual_active": False,
        "early_stopping_rounds": None,
        "stop_reason": "iteration_limit",
        "wall_clock_limit_seconds": None,
        "wall_clock_safety_margin_seconds": None,
        "wall_clock_effective_seconds": None,
        "wall_clock_elapsed_seconds": None,
        "deadline_hit": False,
        "deadline_is_soft": False,
    }
    expected_types = {
        "iterations_requested": int,
        "iterations_attempted": int,
        "rounds_completed": int,
        "rounds_retained": int,
        "best_iteration": int,
        "resolved_learning_rate": float,
        "requested_tree_mode": str,
        "selected_tree_mode": str,
        "selected_lane": str,
        "linear_residual_active": bool,
        "stop_reason": str,
        "deadline_hit": bool,
        "deadline_is_soft": bool,
    }
    for field, expected_type in expected_types.items():
        assert type(metadata[field]) is expected_type
    assert metadata["early_stopping_rounds"] is None

    assert model.get_info()["hyperparameters_fit"] == model.params_trained
    assert model.get_info()["darkofit_fit"] == metadata

    X_val = regression_data[2]
    predictions_before = model.predict(X_val)
    saved_path = model.save(verbose=False)
    loaded = DarkoFitModel.load(saved_path, verbose=False)

    assert loaded.params_trained == model.params_trained
    assert loaded.get_fit_metadata()["darkofit_fit"] == metadata
    assert loaded.get_info()["darkofit_fit"] == metadata
    np.testing.assert_array_equal(loaded.predict(X_val), predictions_before)


def test_autogluon_refit_template_uses_all_rows_without_new_selection_split(
    tmp_path,
    regression_data,
):
    model = _new_model(tmp_path, "darkofit_selection")
    model.fit(**_fit_kwargs(regression_data, time_limit=None))

    X, y, X_val, y_val = regression_data
    X_full = pd.concat([X, X_val], ignore_index=True)
    y_full = pd.concat([y, y_val], ignore_index=True)
    template = model.convert_to_refit_full_template()
    template.fit(
        X=X_full,
        y=y_full,
        time_limit=None,
        num_cpus=1,
        num_gpus=0,
    )

    validation = template.model.model_.auto_params_["validation_split"]
    assert validation["source"] == "none"
    assert validation["train_n_samples"] == len(X_full)
    assert validation["eval_n_samples"] is None
    assert template.model.early_stopping is False
    assert template.model.early_stopping_rounds is None
    assert template.model.use_best_model is False


def test_zero_time_limit_stops_before_first_boosting_attempt(
    tmp_path,
    regression_data,
):
    model = _new_model(tmp_path, "darkofit_zero_limit")
    fit_kwargs = _fit_kwargs(regression_data, time_limit=0.0)

    # AbstractModel.fit rejects an already exhausted budget before dispatching
    # to a child. Calling the initialized adapter hook directly verifies the
    # child-level zero-budget contract without a scheduler timing race.
    model.initialize(**fit_kwargs)
    model._register_fit_metadata(**fit_kwargs)
    model._fit(**fit_kwargs)

    metadata = model.get_fit_metadata()["darkofit_fit"]
    assert metadata["iterations_attempted"] == 0
    assert metadata["rounds_completed"] == 0
    assert metadata["rounds_retained"] == 0
    assert metadata["best_iteration"] == 0
    assert metadata["stop_reason"] == "time_limit"
    assert metadata["wall_clock_limit_seconds"] == 0.0
    assert metadata["wall_clock_safety_margin_seconds"] == 0.0
    assert metadata["wall_clock_effective_seconds"] == 0.0
    assert type(metadata["wall_clock_elapsed_seconds"]) is float
    assert metadata["wall_clock_elapsed_seconds"] >= 0.0
    assert metadata["deadline_hit"] is True
    assert metadata["deadline_is_soft"] is True
    assert model.params_trained["iterations"] == 0


def test_tabarena_single_bag_persists_all_eight_child_metadata_blocks(tmp_path):
    pytest.importorskip("tabarena")
    from tabarena.contexts import TabArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    config = {
        **FIXED_HYPERPARAMETERS,
        "iterations": 2,
        "early_stopping": True,
        "use_best_model": True,
    }
    generator = ConfigGenerator(
        model_cls=DarkoFitModel,
        manual_configs=[config],
        search_space={},
    )
    experiments = generator.generate_all_bag_experiments(
        num_random_configs=0,
        name_id_suffix="_metadata_preflight",
        add_seed="fold-wise",
        fold_fitting_strategy="sequential_local",
        time_limit=120,
    )
    context = TabArenaContext()
    jobs = context.build_jobs(
        experiments,
        task_ids=[363698],
        split_indices=["r0f0"],
    )
    assert len(jobs) == 1
    records = context.run_jobs(
        jobs,
        expname=str(tmp_path / "bag"),
        register=False,
        debug_mode=True,
    )
    assert len(records) == 1

    info = records[0]["method_metadata"]["info"]
    children = info["children_info"]
    assert set(children) == {f"S1F{index}" for index in range(1, 9)}
    required_refit_keys = set(next(iter(children.values()))["hyperparameters_fit"])
    assert required_refit_keys
    for child in children.values():
        fitted = child["darkofit_fit"]
        assert fitted["iterations_requested"] == 2
        assert 0 <= fitted["rounds_retained"] <= fitted["rounds_completed"] <= 2
        assert fitted["resolved_learning_rate"] == 0.1
        assert fitted["selected_tree_mode"] == "catboost"
        assert fitted["selected_lane"] == "boosting"
        assert fitted["stop_reason"] in {
            "iteration_limit",
            "early_stopping",
            "no_split",
            "time_limit",
        }
        assert set(child["hyperparameters_fit"]) == required_refit_keys
    assert set(info["bagged_info"]["child_hyperparameters_fit"]) == (
        required_refit_keys
    )
