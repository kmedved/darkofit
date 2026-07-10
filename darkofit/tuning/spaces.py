"""Stepwise search spaces for DarkoFit tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from ..losses import VECTOR_LOSSES


@dataclass(frozen=True)
class SpaceContext:
    estimator_params: Mapping[str, Any]
    has_categoricals: bool
    classifier: bool
    tree_modes: tuple[str, ...] = ("catboost", "lightgbm")


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    lane: str | None
    n_trials: int | None
    tunable: tuple[str, ...]
    suggest: Callable[[object, SpaceContext, "LaneState"], dict]


@dataclass
class LaneState:
    tree_mode: str
    fixed_params: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    best_loss: float = np.inf
    best_score: float = -np.inf
    best_fold_iterations: list[int] = field(default_factory=list)
    best_fold_learning_rates: list[float] = field(default_factory=list)


DEFAULT_PHASES = (
    "probe",
    "structure",
    "sampling_regularization",
    "learning_rate",
    "binning_categorical",
)

KNOWN_PHASES = frozenset((
    "joint_compact",
    "probe",
    "structure",
    "sampling_regularization",
    "split_noise",
    "learning_rate",
    "binning_categorical",
))


def phase_names(phases):
    if phases == "auto":
        return DEFAULT_PHASES
    if isinstance(phases, str):
        names = (phases,)
    else:
        names = tuple(phases)
    _validate_phase_names(names)
    return names


def make_phase_spec(name, lane, n_trials):
    _validate_phase_names((name,))
    suggest = {
        "joint_compact": suggest_joint_compact,
        "probe": suggest_probe,
        "structure": suggest_structure,
        "sampling_regularization": suggest_sampling_regularization,
        "split_noise": suggest_split_noise,
        "learning_rate": suggest_learning_rate,
        "binning_categorical": suggest_binning_categorical,
    }[name]
    tunable = {
        "joint_compact": ("tree_mode", "depth", "num_leaves", "l2_leaf_reg",
                          "min_child_samples", "min_child_weight",
                          "min_gain_to_split", "colsample"),
        "probe": (),
        "structure": ("depth", "num_leaves", "l2_leaf_reg",
                      "min_child_samples", "min_child_weight",
                      "min_gain_to_split"),
        "sampling_regularization": ("sampling", "subsample", "colsample",
                                    "top_rate", "other_rate", "l2_leaf_reg"),
        "split_noise": ("random_strength",),
        "learning_rate": ("learning_rate", "iterations"),
        "binning_categorical": ("max_bins", "cat_smoothing"),
    }[name]
    return PhaseSpec(name, lane, None if n_trials is None else int(n_trials),
                     tunable, suggest)


def _validate_phase_names(names):
    unknown = [name for name in names if name not in KNOWN_PHASES]
    if unknown:
        valid = ", ".join(sorted(KNOWN_PHASES))
        raise ValueError(
            f"unknown tuning phase {unknown[0]!r}; expected one of: {valid}"
        )


def suggest_joint_compact(trial, context, state):
    tree_modes = tuple(context.tree_modes or ("catboost", "lightgbm"))
    tree_mode = trial.suggest_categorical("tree_mode", list(tree_modes))
    prefix = f"joint_{tree_mode}"
    if tree_mode == "catboost":
        return {
            "tree_mode": "catboost",
            "depth": trial.suggest_int(f"{prefix}_depth", 3, 8),
            "num_leaves": None,
            "ordered_boosting": "auto",
            "l2_leaf_reg": trial.suggest_float(
                f"{prefix}_l2_leaf_reg", 0.5, 30.0, log=True
            ),
            "min_child_weight": trial.suggest_float(
                f"{prefix}_min_child_weight", 1.0, 100.0, log=True
            ),
            "colsample": trial.suggest_float(f"{prefix}_colsample", 0.6, 1.0),
        }
    return {
        "tree_mode": tree_mode,
        "ordered_boosting": "auto",
        "num_leaves": trial.suggest_categorical(
            f"{prefix}_num_leaves", [7, 15, 31, 63, 127]
        ),
        "depth": trial.suggest_categorical(
            f"{prefix}_depth", [-1, 3, 4, 5, 6, 8, 10]
        ),
        "min_child_samples": trial.suggest_int(
            f"{prefix}_min_child_samples", 5, 200, log=True
        ),
        "min_child_weight": trial.suggest_float(
            f"{prefix}_min_child_weight", 1e-2, 100.0, log=True
        ),
        "min_gain_to_split": trial.suggest_float(
            f"{prefix}_min_gain_to_split", 0.0, 1.0
        ),
        "l2_leaf_reg": trial.suggest_float(
            f"{prefix}_l2_leaf_reg", 0.1, 30.0, log=True
        ),
        "colsample": trial.suggest_float(f"{prefix}_colsample", 0.5, 1.0),
    }


def suggest_probe(trial, context, state):
    return {"tree_mode": state.tree_mode}


def suggest_structure(trial, context, state):
    prefix = f"{state.tree_mode}_structure"
    if state.tree_mode == "catboost":
        return {
            "tree_mode": "catboost",
            "depth": trial.suggest_int(f"{prefix}_depth", 3, 8),
            "num_leaves": None,
            "ordered_boosting": "auto",
            "l2_leaf_reg": trial.suggest_float(f"{prefix}_l2_leaf_reg", 0.5, 30.0, log=True),
            "min_child_weight": trial.suggest_float(f"{prefix}_min_child_weight", 1.0, 100.0, log=True),
        }
    return {
        "tree_mode": state.tree_mode,
        "ordered_boosting": "auto",
        "num_leaves": trial.suggest_categorical(f"{prefix}_num_leaves", [7, 15, 31, 63, 127]),
        "depth": trial.suggest_categorical(f"{prefix}_depth", [-1, 3, 4, 5, 6, 8, 10]),
        "min_child_samples": trial.suggest_int(f"{prefix}_min_child_samples", 5, 200, log=True),
        "min_child_weight": trial.suggest_float(f"{prefix}_min_child_weight", 1e-2, 100.0, log=True),
        "min_gain_to_split": trial.suggest_float(f"{prefix}_min_gain_to_split", 0.0, 1.0),
        "l2_leaf_reg": trial.suggest_float(f"{prefix}_l2_leaf_reg", 0.1, 30.0, log=True),
    }


def suggest_sampling_regularization(trial, context, state):
    prefix = f"{state.tree_mode}_sampling"
    if context.estimator_params.get("loss") in VECTOR_LOSSES:
        params = {
            "tree_mode": state.tree_mode,
            "sampling": "uniform",
            "bootstrap_type": "none",
            "bagging_temperature": 0.0,
            "mvs_reg": 1.0,
            "random_strength": 0.0,
            "subsample": trial.suggest_float(
                f"{prefix}_subsample", 0.6, 1.0
            ),
            "colsample": trial.suggest_float(
                f"{prefix}_colsample", 0.5, 1.0
            ),
            "l2_leaf_reg": trial.suggest_float(
                f"{prefix}_l2_leaf_reg", 0.1, 30.0, log=True
            ),
        }
        if context.estimator_params.get("loss") == "StudentT":
            params["dist_params"] = {
                "nu": trial.suggest_categorical(
                    f"{prefix}_student_t_nu", [3.0, 4.0, 6.0, 10.0, 30.0]
                )
            }
        return params
    sampling = trial.suggest_categorical(
        f"{prefix}_sampling", ["uniform", "goss", "mvs"]
    )
    bootstrap_type = trial.suggest_categorical(
        f"{prefix}_bootstrap_type", ["none", "bayesian"]
    )
    params = {
        "tree_mode": state.tree_mode,
        "sampling": sampling,
        "bootstrap_type": bootstrap_type,
        "bagging_temperature": 0.0,
        "mvs_reg": 1.0,
        "random_strength": 0.0,
        "colsample": trial.suggest_float(f"{prefix}_colsample", 0.5, 1.0),
        "l2_leaf_reg": trial.suggest_float(f"{prefix}_l2_leaf_reg", 0.1, 30.0, log=True),
    }
    if bootstrap_type == "bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            f"{prefix}_bagging_temperature", 0.0, 1.0
        )
    if sampling == "goss":
        params.update(
            subsample=1.0,
            top_rate=trial.suggest_float(f"{prefix}_top_rate", 0.1, 0.4),
            other_rate=trial.suggest_float(f"{prefix}_other_rate", 0.05, 0.4),
        )
    elif sampling == "mvs":
        params.update(
            subsample=trial.suggest_float(f"{prefix}_subsample", 0.5, 1.0),
            mvs_reg=trial.suggest_float(f"{prefix}_mvs_reg", 0.1, 10.0, log=True),
        )
    else:
        params.update(subsample=trial.suggest_float(f"{prefix}_subsample", 0.6, 1.0))
    return params


def suggest_split_noise(trial, context, state):
    prefix = f"{state.tree_mode}_split_noise"
    return {
        "tree_mode": state.tree_mode,
        "random_strength": trial.suggest_float(
            f"{prefix}_random_strength", 0.0, 2.0
        ),
    }


def suggest_learning_rate(trial, context, state):
    prefix = f"{state.tree_mode}_learning"
    anchor_lr = _median_or_default(state.best_fold_learning_rates, 0.05)
    anchor_rounds = int(_median_or_default(
        state.best_fold_iterations,
        context.estimator_params.get("iterations", 1000),
    ))
    mult = trial.suggest_float(f"{prefix}_lr_multiplier", 0.35, 2.5, log=True)
    rounds_mult = trial.suggest_float(f"{prefix}_rounds_multiplier", 0.75, 2.0)
    return {
        "tree_mode": state.tree_mode,
        "learning_rate": float(np.clip(anchor_lr * mult, 0.005, 0.5)),
        "iterations": max(10, int(round(anchor_rounds * rounds_mult))),
    }


def suggest_binning_categorical(trial, context, state):
    prefix = f"{state.tree_mode}_binning"
    params = {
        "tree_mode": state.tree_mode,
        "max_bins": trial.suggest_categorical(f"{prefix}_max_bins", [64, 128, 254, 512]),
    }
    if context.has_categoricals:
        params["cat_smoothing"] = trial.suggest_float(
            f"{prefix}_cat_smoothing", 0.3, 20.0, log=True
        )
    return params


def _median_or_default(values, default):
    if not values:
        return float(default)
    return float(np.median(np.asarray(values, dtype=np.float64)))
