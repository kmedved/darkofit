#!/usr/bin/env python3
"""Generate the Phase-0 public-prediction golden suite.

The stable digest rounds public outputs to twelve decimal places so the normal
CI gate remains portable across supported Python, NumPy, SciPy, Numba, CPU, and
OS combinations. The artifact also records an exact byte digest. Set
``DARKOFIT_STRICT_GOLDENS=1`` in a controlled before/after optimization lane to
make the test enforce that exact digest as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from darkofit import DarkoClassifier, DarkoRegressor  # noqa: E402


SCHEMA_VERSION = 1
STABLE_DECIMALS = 12
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "golden_predictions.json"
PREDICTION_ROWS = 12


def _numeric_data(n_rows: int = 112) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(n_rows, dtype=np.int64)
    columns = [
        ((index * 7 + 3) % 29 - 14) / 4.0,
        ((index * 11 + 1) % 31 - 15) / 5.0,
        ((index * 13 + index // 3) % 23 - 11) / 3.0,
        ((index * 5 + index // 7) % 19 - 9) / 2.0,
        ((index * 17 + 2) % 37 - 18) / 6.0,
    ]
    X = np.column_stack(columns).astype(np.float64)
    y = (
        0.75 * X[:, 0]
        - 0.4 * X[:, 1]
        + 0.2 * X[:, 2] * X[:, 3]
        + ((index * 3) % 7 - 3) / 10.0
    )
    return X, y.astype(np.float64)


def _categorical_data(n_rows: int = 112) -> tuple[pd.DataFrame, np.ndarray]:
    index = np.arange(n_rows, dtype=np.int64)
    teams = np.asarray(["ATL", "BOS", "CHI", "DEN", "NYK"], dtype=object)
    roles = np.asarray(["guard", "wing", "big"], dtype=object)
    team_code = (index * 3 + index // 5) % len(teams)
    role_code = (index + index // 4) % len(roles)
    X = pd.DataFrame(
        {
            "usage": ((index * 7) % 31 - 15) / 5.0,
            "age": 19.0 + ((index * 11) % 18),
            "team": pd.Series(teams[team_code], dtype="string"),
            "role": pd.Categorical(roles[role_code], categories=roles.tolist()),
        }
    )
    y = (
        0.35 * X["usage"].to_numpy()
        + 0.04 * X["age"].to_numpy()
        + 0.3 * team_code
        - 0.2 * role_code
        + ((index * 5) % 9 - 4) / 15.0
    )
    return X, np.asarray(y, dtype=np.float64)


def _common_regressor_params(tree_mode: str) -> dict[str, Any]:
    params = {
        "iterations": 14,
        "learning_rate": 0.1,
        "depth": 3,
        "l2_leaf_reg": 3.0,
        "max_bins": 16,
        "subsample": 1.0,
        "colsample": 1.0,
        "min_child_samples": 2,
        "thread_count": 1,
        "random_state": 2026,
        "ordered_boosting": False,
        "early_stopping": False,
        "tree_mode": tree_mode,
        "diagnostic_warnings": "never",
    }
    if tree_mode in {"lightgbm", "hybrid"}:
        params["num_leaves"] = 7
    return params


def _common_classifier_params(tree_mode: str) -> dict[str, Any]:
    params = _common_regressor_params(tree_mode)
    params.pop("diagnostic_warnings")
    return params


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    shape = np.asarray(array.shape, dtype="<i8")
    return hashlib.sha256(shape.tobytes() + array.tobytes()).hexdigest()


def _stable_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("prediction golden output contains a non-finite value")
    stable = np.round(array, decimals=STABLE_DECIMALS)
    stable[stable == 0.0] = 0.0
    return stable


def _record(value: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    stable = _stable_array(array)
    return {
        "shape": list(array.shape),
        "stable_sha256": _array_digest(stable),
        "exact_sha256": _array_digest(array),
        "stable_values": stable.tolist(),
    }


def _repeatable_output(function: Callable[[], Any]) -> Any:
    first = function()
    second = function()
    if isinstance(first, tuple):
        if not isinstance(second, tuple) or len(first) != len(second):
            raise RuntimeError("prediction golden tuple output changed shape")
        for left, right in zip(first, second):
            if not np.array_equal(left, right):
                raise RuntimeError("prediction golden output is not repeatable")
    elif not np.array_equal(first, second):
        raise RuntimeError("prediction golden output is not repeatable")
    return first


def _record_model_outputs(model: Any, X_predict: Any) -> dict[str, Any]:
    outputs = {
        "predict": _record(
            _repeatable_output(lambda: np.asarray(model.predict(X_predict)))
        ),
        "predict_raw": _record(
            _repeatable_output(
                lambda: np.asarray(model.model_.predict_raw(X_predict))
            )
        ),
    }
    if isinstance(model, DarkoClassifier):
        outputs["predict_proba"] = _record(
            _repeatable_output(
                lambda: np.asarray(model.predict_proba(X_predict))
            )
        )
    if isinstance(model, DarkoRegressor) and model.loss in {
        "Gaussian",
        "LogNormal",
        "StudentT",
        "Poisson",
        "NegativeBinomial",
    }:
        params = _repeatable_output(lambda: model.predict_dist(X_predict))
        for index, value in enumerate(params):
            outputs[f"predict_dist_{index}"] = _record(value)
        outputs["predict_variance"] = _record(
            _repeatable_output(lambda: model.predict_variance(X_predict))
        )
        interval = _repeatable_output(
            lambda: model.predict_interval(X_predict, alpha=0.2)
        )
        outputs["predict_interval_lower"] = _record(interval[0])
        outputs["predict_interval_upper"] = _record(interval[1])
        outputs["sample"] = _record(
            _repeatable_output(
                lambda: model.sample(
                    X_predict, n_samples=3, random_state=8675309
                )
            )
        )
    return outputs


def collect_prediction_goldens() -> dict[str, Any]:
    X_numeric, y_regression = _numeric_data()
    split = len(X_numeric) - PREDICTION_ROWS
    X_train = X_numeric[:split]
    X_predict = X_numeric[split:]
    y_train = y_regression[:split]
    cases: dict[str, dict[str, Any]] = {}

    for tree_mode in ("catboost", "lightgbm", "hybrid", "depthwise"):
        name = f"numeric_regression_{tree_mode}"
        params = _common_regressor_params(tree_mode)
        model = DarkoRegressor(**params).fit(X_train, y_train)
        cases[name] = {
            "kind": "regression",
            "tree_mode": tree_mode,
            "params": params,
            "outputs": _record_model_outputs(model, X_predict),
        }

    X_categorical, y_categorical = _categorical_data()
    X_cat_train = X_categorical.iloc[:split]
    X_cat_predict = X_categorical.iloc[split:]
    cat_features = [2, 3]
    cat_params = _common_regressor_params("catboost")
    cat_params.update({"ordered_boosting": True, "ts_permutations": 2})
    cat_model = DarkoRegressor(**cat_params).fit(
        X_cat_train,
        y_categorical[:split],
        cat_features=cat_features,
    )
    cases["categorical_regression_catboost"] = {
        "kind": "regression_categorical",
        "tree_mode": "catboost",
        "params": cat_params,
        "cat_features": cat_features,
        "outputs": _record_model_outputs(cat_model, X_cat_predict),
    }

    index = np.arange(split, dtype=np.int64)
    y_binary = (((index * 7 + index // 5) % 13) >= 6).astype(np.int64)
    binary_params = _common_classifier_params("catboost")
    binary_params.update({"ordered_boosting": True, "ts_permutations": 2})
    binary_model = DarkoClassifier(**binary_params).fit(
        X_cat_train, y_binary, cat_features=cat_features
    )
    cases["categorical_binary_catboost"] = {
        "kind": "binary_categorical",
        "tree_mode": "catboost",
        "params": binary_params,
        "cat_features": cat_features,
        "outputs": _record_model_outputs(binary_model, X_cat_predict),
    }

    y_multiclass = ((index // 2 + index % 5) % 3).astype(np.int64)
    multiclass_params = _common_classifier_params("lightgbm")
    multiclass_model = DarkoClassifier(**multiclass_params).fit(
        X_train, y_multiclass
    )
    cases["numeric_multiclass_lightgbm"] = {
        "kind": "multiclass",
        "tree_mode": "lightgbm",
        "params": multiclass_params,
        "outputs": _record_model_outputs(multiclass_model, X_predict),
    }

    full_index = np.arange(len(X_numeric), dtype=np.int64)
    distribution_targets = {
        "Gaussian": y_regression,
        "StudentT": y_regression + ((full_index % 11) == 0) * 2.5,
        "LogNormal": 1.0 + np.abs(y_regression) + (full_index % 7) / 10.0,
        "Poisson": ((full_index * 7 + full_index // 4) % 9).astype(np.float64),
        "NegativeBinomial": (
            (full_index * 11 + full_index // 3) % 13
        ).astype(np.float64),
    }
    distribution_params = {
        "Gaussian": {},
        "StudentT": {"nu": 5.0},
        "LogNormal": {},
        "Poisson": {},
        "NegativeBinomial": {"r": 2.5},
    }
    for loss, target in distribution_targets.items():
        params = _common_regressor_params("lightgbm")
        params.update({"loss": loss, "dist_params": distribution_params[loss]})
        model = DarkoRegressor(**params).fit(X_train, target[:split])
        cases[f"distributional_{loss.lower()}"] = {
            "kind": "distributional",
            "tree_mode": "lightgbm",
            "loss": loss,
            "params": params,
            "outputs": _record_model_outputs(model, X_predict),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "stable_decimals": STABLE_DECIMALS,
        "baseline_darkofit_head": "3295f70c231d4f7947e13a13ad77e3f2c19b3fe0",
        "case_count": len(cases),
        "cases": cases,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the generated artifact; otherwise print a summary only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = collect_prediction_goldens()
    if args.write:
        _atomic_write_json(args.output, payload)
        print(f"wrote {args.output}")
    else:
        print(
            json.dumps(
                {
                    "case_count": payload["case_count"],
                    "case_names": sorted(payload["cases"]),
                    "output": str(args.output),
                    "written": False,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
