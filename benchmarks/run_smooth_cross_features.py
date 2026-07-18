#!/usr/bin/env python3
"""Run the spent smooth-task full-budget cross-feature development screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import secrets
import stat
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
CHIMERA_ROOT = ROOT.parent / "chimeraboost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = {
    361251: "grid_stability",
    361258: "kin8nm",
    361623: "space_ga",
}
TASK_FEATURE_COUNTS = {
    361251: 12,
    361258: 8,
    361623: 6,
}
FOLDS = tuple(range(3, 10))
RANDOM_STATE = 4
THREADS = 6
TOP_NUMERIC_FEATURES = 6
EXPECTED_CHIMERA_HEAD = "851ab7fa79fbb2a7f698fbc1a00952e1bd18c62d"
PROTOCOL = ROOT / "benchmarks" / "smooth_cross_features_protocol.md"
PARTITION = ROOT / "benchmarks" / "ctr23_partition.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "smooth_cross_features.json"
EXACT_FIELDS = (
    "best_prefix_tree_count",
    "fingerprinted_tree_count",
    "best_validation_rmse",
    "test_rmse",
    "prediction_sha256",
    "validation_history_sha256",
    "borders_sha256",
    "model_sha256",
)
_CHIMERA_MODULE_NAME = "_darkofit_smooth_frozen_chimeraboost"
_CHIMERA_IMPORT_LOCK = threading.Lock()
_CHIMERA_MODULE = None
_CHIMERA_MODULES = None
_CHIMERA_REGRESSOR = None
FROZEN_EVIDENCE_SHA256 = (
    "9a9e558e75ba2a01f08fe91d10c85e882181048cff24ca62dcade163546941ac"
)
FROZEN_ARTIFACT_SHA256 = (
    "9b265ff3993bbc374e3300afa0088b55e33855c401c1213ceb9e061ae729bb39"
)


def _private_chimera_provenance_is_valid(
    package,
    initializer,
    module,
    regressor,
    private_modules,
):
    if not isinstance(private_modules, dict):
        return False
    source_name = getattr(module, "__file__", None)
    regressor_name = getattr(regressor, "__module__", None)
    regressor_module = (
        private_modules.get(regressor_name)
        if isinstance(regressor_name, str)
        else None
    )
    regressor_source_name = getattr(regressor_module, "__file__", None)
    regressor_init_code = getattr(
        getattr(regressor, "__init__", None),
        "__code__",
        None,
    )
    if (
        private_modules.get(_CHIMERA_MODULE_NAME) is not module
        or not isinstance(source_name, (str, os.PathLike))
        or Path(source_name).resolve() != initializer.resolve()
        or not isinstance(regressor, type)
        or not isinstance(regressor_name, str)
        or not regressor_name.startswith(f"{_CHIMERA_MODULE_NAME}.")
        or regressor_module is None
        or getattr(module, "ChimeraBoostRegressor", None) is not regressor
        or not isinstance(regressor_source_name, (str, os.PathLike))
        or not Path(regressor_source_name).resolve().is_relative_to(
            package.resolve()
        )
        or regressor_init_code is None
        or Path(regressor_init_code.co_filename).resolve()
        != Path(regressor_source_name).resolve()
    ):
        return False
    for name, loaded in private_modules.items():
        loaded_spec = getattr(loaded, "__spec__", None)
        loaded_loader = getattr(loaded_spec, "loader", None)
        loaded_file = getattr(loaded, "__file__", None)
        loaded_origin = getattr(loaded_spec, "origin", None)
        if (
            sys.modules.get(name) is not loaded
            or getattr(loaded, "__name__", None) != name
            or getattr(loaded_spec, "name", None) != name
            or type(loaded_loader) is not importlib.machinery.SourceFileLoader
            or getattr(loaded_loader, "name", None) != name
            or not isinstance(loaded_file, (str, os.PathLike))
            or not isinstance(loaded_origin, (str, os.PathLike))
            or not isinstance(
                getattr(loaded_loader, "path", None),
                (str, os.PathLike),
            )
            or Path(loaded_file).resolve() != Path(loaded_origin).resolve()
            or Path(loaded_file).resolve()
            != Path(loaded_loader.path).resolve()
            or not Path(loaded_file).resolve().is_relative_to(
                package.resolve()
            )
        ):
            return False
    return True


def _chimera_regressor_class():
    """Load the frozen sibling package without changing global import order."""
    global _CHIMERA_MODULE, _CHIMERA_MODULES, _CHIMERA_REGRESSOR

    package = CHIMERA_ROOT / "chimeraboost"
    initializer = package / "__init__.py"
    with _CHIMERA_IMPORT_LOCK:
        installed = sys.modules.get(_CHIMERA_MODULE_NAME)
        if _CHIMERA_MODULE is None:
            occupied = [
                name
                for name in sys.modules
                if name == _CHIMERA_MODULE_NAME
                or name.startswith(f"{_CHIMERA_MODULE_NAME}.")
            ]
            if occupied:
                raise RuntimeError(
                    "imported chimeraboost from the wrong checkout: "
                    "private module slot was already occupied"
                )
            spec = importlib.util.spec_from_file_location(
                _CHIMERA_MODULE_NAME,
                initializer,
                submodule_search_locations=[str(package)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"cannot load frozen chimeraboost package: {initializer}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[_CHIMERA_MODULE_NAME] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                for name in tuple(sys.modules):
                    if name == _CHIMERA_MODULE_NAME or name.startswith(
                        f"{_CHIMERA_MODULE_NAME}."
                    ):
                        sys.modules.pop(name, None)
                raise
            regressor = getattr(module, "ChimeraBoostRegressor", None)
            private_modules = {
                name: loaded
                for name, loaded in sys.modules.items()
                if name == _CHIMERA_MODULE_NAME
                or name.startswith(f"{_CHIMERA_MODULE_NAME}.")
            }
            if not _private_chimera_provenance_is_valid(
                package,
                initializer,
                module,
                regressor,
                private_modules,
            ):
                for name in tuple(sys.modules):
                    if name == _CHIMERA_MODULE_NAME or name.startswith(
                        f"{_CHIMERA_MODULE_NAME}."
                    ):
                        sys.modules.pop(name, None)
                raise RuntimeError(
                    "imported chimeraboost from the wrong checkout"
                )
            _CHIMERA_MODULE = module
            _CHIMERA_MODULES = private_modules
            _CHIMERA_REGRESSOR = regressor
        elif (
            installed is not _CHIMERA_MODULE
            or _CHIMERA_MODULES is None
            or {
                name
                for name in sys.modules
                if name == _CHIMERA_MODULE_NAME
                or name.startswith(f"{_CHIMERA_MODULE_NAME}.")
            }
            != set(_CHIMERA_MODULES)
            or any(
                sys.modules.get(name) is not loaded
                for name, loaded in _CHIMERA_MODULES.items()
            )
            or not _private_chimera_provenance_is_valid(
                package,
                initializer,
                _CHIMERA_MODULE,
                _CHIMERA_REGRESSOR,
                _CHIMERA_MODULES,
            )
        ):
            raise RuntimeError(
                "imported chimeraboost from the wrong checkout: "
                "private module slot changed"
            )
        return _CHIMERA_REGRESSOR


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_mutable_symlink_output_parents(path: Path) -> None:
    for directory in (path.parent, *path.parent.parents):
        if directory.is_symlink() and os.access(directory.parent, os.W_OK):
            raise RuntimeError(
                f"refusing symlink output directory: {directory}"
            )


def _create_missing_directories(
    directory: Path,
    created: list[tuple[Path, tuple[int, int]]],
) -> None:
    missing = []
    current = directory
    while not current.exists():
        if current.is_symlink():
            raise RuntimeError(
                f"refusing symlink output directory: {current}"
            )
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(
            f"output parent is not a directory: {current}"
        )
    for current in reversed(missing):
        try:
            current.mkdir()
        except FileExistsError:
            if not current.is_dir() or current.is_symlink():
                raise
        else:
            metadata = current.lstat()
            created.append(
                (current, (metadata.st_dev, metadata.st_ino))
            )


def _remove_owned_empty_directories(
    directories: list[tuple[Path, tuple[int, int]]],
) -> None:
    for directory, identity in reversed(directories):
        try:
            current = directory.lstat()
            if (
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == identity
            ):
                directory.rmdir()
        except OSError:
            pass


def _assert_output_parent_identity(
    path: Path,
    identity: tuple[int, int],
) -> None:
    _reject_mutable_symlink_output_parents(path)
    try:
        current = path.parent.lstat()
    except OSError as exc:
        raise RuntimeError(f"output parent changed: {path.parent}") from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        raise RuntimeError(f"output parent changed: {path.parent}")


def _open_output_parent(path: Path) -> tuple[int, tuple[int, int]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptor = os.open(path.parent, flags)
    try:
        current = os.fstat(descriptor)
        identity = (current.st_dev, current.st_ino)
        if not stat.S_ISDIR(current.st_mode):
            raise RuntimeError(
                f"output parent is not a directory: {path.parent}"
            )
        _assert_output_parent_identity(path, identity)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _temporary_at(
    directory_descriptor: int,
    output_name: str,
) -> tuple[int, str, tuple[int, int]]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(128):
        name = f".{output_name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                name,
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        current = os.fstat(descriptor)
        return descriptor, name, (current.st_dev, current.st_ino)
    raise FileExistsError(
        f"unable to reserve temporary output for {output_name}"
    )


def _stat_at(directory_descriptor: int, name: str) -> os.stat_result:
    return os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _exists_at(directory_descriptor: int, name: str) -> bool:
    try:
        _stat_at(directory_descriptor, name)
    except FileNotFoundError:
        return False
    return True


def _unlink_if_owned_at(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int],
) -> None:
    try:
        current = _stat_at(directory_descriptor, name)
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == identity
    ):
        os.unlink(name, dir_fd=directory_descriptor)


def _atomic_create(path: Path, value: bytes) -> None:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    _reject_mutable_symlink_output_parents(path)
    created_directories: list[tuple[Path, tuple[int, int]]] = []
    parent_descriptor = None
    parent_identity = None
    temporary_name = None
    identity = None
    created = False
    try:
        _create_missing_directories(path.parent, created_directories)
        parent_descriptor, parent_identity = _open_output_parent(path)
        if _exists_at(parent_descriptor, path.name):
            raise FileExistsError(
                f"refusing to replace existing output: {path}"
            )
        descriptor, temporary_name, identity = _temporary_at(
            parent_descriptor,
            path.name,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_output_parent_identity(path, parent_identity)
        current = _stat_at(parent_descriptor, temporary_name)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
        ):
            raise RuntimeError(
                "temporary output changed before publication: "
                f"{path.parent / temporary_name}"
            )
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        created = True
        _assert_output_parent_identity(path, parent_identity)
        current = _stat_at(parent_descriptor, path.name)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
        ):
            raise RuntimeError(f"published output changed: {path}")
        _unlink_if_owned_at(
            parent_descriptor,
            temporary_name,
            identity,
        )
        _assert_output_parent_identity(path, parent_identity)
    except BaseException:
        if parent_descriptor is not None and identity is not None:
            if created:
                try:
                    _unlink_if_owned_at(
                        parent_descriptor,
                        path.name,
                        identity,
                    )
                except OSError:
                    pass
            if temporary_name is not None:
                try:
                    _unlink_if_owned_at(
                        parent_descriptor,
                        temporary_name,
                        identity,
                    )
                except OSError:
                    pass
        _remove_owned_empty_directories(created_directories)
        raise
    finally:
        if parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass


def _array_sha256(value, dtype=None) -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_state(path: Path, *, expected_head=None):
    head = _git(path, "rev-parse", "HEAD")
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"source tree is not clean: {path}")
    if expected_head is not None and head != expected_head:
        raise RuntimeError(f"unexpected source head: {head} != {expected_head}")
    return {
        "path": str(path),
        "head": head,
        "branch": _git(path, "branch", "--show-current"),
        "clean": True,
    }


def _partition_boundary():
    payload = PARTITION.read_bytes()
    partition = json.loads(payload)
    confirmation = set(partition["confirmation_task_ids"])
    lockbox = set(partition["lockbox_task_ids"])
    task_ids = set(TASKS)
    if not task_ids <= confirmation or task_ids & lockbox:
        raise RuntimeError("smooth cross-feature task boundary changed")
    for task_id in task_ids:
        row = partition["task_allocation_metadata"][str(task_id)]
        if row["has_categorical"] != 0.0 or row["has_missing_features"] != 0.0:
            raise RuntimeError("smooth cross-feature task profile changed")
    return {
        "partition_sha256": hashlib.sha256(payload).hexdigest(),
        "confirmation_task_ids": sorted(confirmation),
        "lockbox_task_ids": sorted(lockbox),
        "lockbox_data_used": False,
        "default_promotion_authorized": False,
    }


def candidate_pairs(importances, categorical_indices, n_features):
    """Return deterministic top-six numeric diff/product pair declarations."""
    if (
        not isinstance(n_features, int)
        or isinstance(n_features, bool)
        or n_features < 0
    ):
        raise RuntimeError("feature count must be a nonnegative integer")
    try:
        supplied = np.asarray(importances, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "feature importances must be finite and one-dimensional "
            "and match the feature count"
        ) from exc
    if (
        supplied.shape != (n_features,)
        or not np.all(np.isfinite(supplied))
    ):
        raise RuntimeError(
            "feature importances must be finite and one-dimensional "
            "and match the feature count"
        )
    if categorical_indices is None:
        categorical_values = []
    else:
        try:
            categorical_values = list(categorical_indices)
        except TypeError as exc:
            raise RuntimeError(
                "categorical indices must be an iterable of feature indices"
            ) from exc
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= n_features
        for index in categorical_values
    ) or len(set(categorical_values)) != len(categorical_values):
        raise RuntimeError("categorical indices are invalid")
    categorical = set(categorical_values)
    numeric = [index for index in range(n_features) if index not in categorical]
    if len(numeric) < 2:
        return []
    importance = supplied
    top = sorted(numeric, key=lambda index: (-importance[index], index))[
        :TOP_NUMERIC_FEATURES
    ]
    return [
        (top[left], top[right], operation)
        for left in range(len(top))
        for right in range(left + 1, len(top))
        for operation in ("diff", "prod")
    ]


def augment_numeric_crosses(X, pairs):
    """Append declared numeric crosses without consulting the target."""
    array = np.asarray(X)
    if array.ndim != 2:
        raise RuntimeError("cross-feature input must be two-dimensional")
    try:
        numeric = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cross-feature input must be numeric") from exc
    try:
        declared_pairs = list(pairs)
    except TypeError as exc:
        raise RuntimeError("cross-feature pairs must be iterable") from exc
    normalized = []
    for pair in declared_pairs:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 3
            or not isinstance(pair[0], int)
            or isinstance(pair[0], bool)
            or not isinstance(pair[1], int)
            or isinstance(pair[1], bool)
            or pair[0] < 0
            or pair[1] < 0
            or pair[0] >= numeric.shape[1]
            or pair[1] >= numeric.shape[1]
            or pair[0] == pair[1]
            or pair[2] not in {"diff", "prod"}
        ):
            raise RuntimeError("cross-feature pair declaration is invalid")
        normalized.append(tuple(pair))
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("cross-feature pair declarations repeat")
    columns = [array]
    for left, right, operation in normalized:
        a = numeric[:, left]
        b = numeric[:, right]
        value = a - b if operation == "diff" else a * b
        columns.append(value[:, None])
    return np.hstack(columns)


def _load_task(task_id):
    import openml

    task = openml.tasks.get_task(task_id, download_splits=True)
    dataset = task.get_dataset()
    X, y, categorical, _names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    if any(bool(value) for value in categorical):
        raise RuntimeError(f"{TASKS[task_id]} unexpectedly has categoricals")
    if X.shape[1] != TASK_FEATURE_COUNTS[task_id]:
        raise RuntimeError(f"{TASKS[task_id]} feature count changed")
    X_array = np.asarray(X, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(y_array)):
        raise RuntimeError(f"{TASKS[task_id]} unexpectedly has nonfinite data")
    if task.get_split_dimensions() != (1, 10, 1):
        raise RuntimeError(f"{TASKS[task_id]} split dimensions changed")
    return task, X, y, {
        "task_id": int(task_id),
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": str(dataset.name),
        "target_name": str(task.target_name),
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "X_sha256": _array_sha256(X_array, "<f8"),
        "y_sha256": _array_sha256(y_array, "<f8"),
    }


def _update_array_hash(digest, label, value):
    digest.update(label.encode("utf-8"))
    if value is None:
        digest.update(b"<none>")
        return
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _model_sha256(core, tree_limit=None):
    digest = hashlib.sha256()
    _update_array_hash(digest, "init", np.asarray([core.init_], dtype="<f8"))
    trees = core.trees_ if tree_limit is None else core.trees_[:tree_limit]
    for index, tree in enumerate(trees):
        digest.update(str(index).encode("ascii"))
        _update_array_hash(digest, "splits_feat", tree.splits_feat)
        _update_array_hash(digest, "splits_thr", tree.splits_thr)
        _update_array_hash(digest, "values", tree.values)
        _update_array_hash(
            digest,
            "linear_features",
            getattr(tree, "linear_features", getattr(tree, "lin_feats", None)),
        )
        _update_array_hash(
            digest,
            "linear_coefficients",
            getattr(
                tree, "linear_coefficients", getattr(tree, "lin_coef", None)
            ),
        )
    return digest.hexdigest()


def _fingerprint(model, prediction, y_test, *, tree_limit=None):
    core = model.model_
    history = np.asarray(core.valid_history_, dtype=np.float64)
    best_rounds = int(np.argmin(history)) + 1
    if tree_limit is None:
        tree_limit = len(core.trees_)
    borders = np.concatenate(
        [
            np.asarray(border, dtype=np.float64)
            for border in core.prep_.binner_.borders_
        ]
    )
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "actual_retained_tree_count": int(len(core.trees_)),
        "best_prefix_tree_count": best_rounds,
        "fingerprinted_tree_count": int(tree_limit),
        "best_validation_rmse": float(np.min(history)),
        "test_rmse": float(
            mean_squared_error(np.asarray(y_test, dtype=np.float64), prediction)
            ** 0.5
        ),
        "prediction_sha256": _array_sha256(prediction, "<f8"),
        "validation_history_sha256": _array_sha256(history, "<f8"),
        "borders_sha256": _array_sha256(borders, "<f8"),
        "model_sha256": _model_sha256(core, tree_limit=tree_limit),
    }


def _staged_prediction_at(model, X, rounds):
    prediction = None
    for index, staged in enumerate(model.staged_predict(X), start=1):
        if index == rounds:
            prediction = np.asarray(staged, dtype=np.float64)
            break
    if prediction is None:
        raise RuntimeError(f"model did not yield staged round {rounds}")
    return prediction


def _darko_model(*, linear_leaves):
    from darkofit import DarkoRegressor

    return DarkoRegressor(
        iterations=2000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=1.0,
        max_bins=128,
        min_child_weight=1.0,
        linear_leaves=bool(linear_leaves),
        early_stopping=True,
        use_best_model=True,
        refit=False,
        random_state=RANDOM_STATE,
        thread_count=THREADS,
    )


def _fit(model, X_train, y_train, X_validation, y_validation):
    started = time.perf_counter_ns()
    model.fit(
        X_train,
        y_train,
        eval_set=(X_validation, y_validation),
    )
    return (time.perf_counter_ns() - started) / 1e9


def _evaluate_fold(task, X, y, task_id, fold):
    outer_train, outer_test = task.get_train_test_split_indices(
        repeat=0, fold=fold, sample=0
    )
    subtrain, validation = next(
        ShuffleSplit(
            n_splits=1, test_size=0.20, random_state=RANDOM_STATE
        ).split(X.iloc[outer_train])
    )
    fit_indices = np.asarray(outer_train)[subtrain]
    validation_indices = np.asarray(outer_train)[validation]
    eval_X = X.iloc[validation_indices]
    eval_y = y.iloc[validation_indices]

    base_candidates = []
    darko_fit_seconds = 0.0
    for linear in (False, True):
        model = _darko_model(linear_leaves=linear)
        darko_fit_seconds += _fit(
            model,
            X.iloc[fit_indices],
            y.iloc[fit_indices],
            eval_X,
            eval_y,
        )
        base_candidates.append(model)
    base = min(
        base_candidates,
        key=lambda model: (
            float(model.best_score_),
            bool(model.linear_leaves),
        ),
    )
    base_linear = bool(base.linear_leaves)
    pairs = candidate_pairs(base.feature_importances_, (), X.shape[1])
    augmented = augment_numeric_crosses(X, pairs)
    crossed = _darko_model(linear_leaves=base_linear)
    darko_fit_seconds += _fit(
        crossed,
        augmented[fit_indices],
        y.iloc[fit_indices],
        augmented[validation_indices],
        eval_y,
    )
    cross_selected = float(crossed.best_score_) < float(base.best_score_)
    selected = crossed if cross_selected else base
    selected_prediction = selected.predict(
        augmented[outer_test] if cross_selected else X.iloc[outer_test]
    )
    base_prediction = base.predict(X.iloc[outer_test])

    ChimeraBoostRegressor = _chimera_regressor_class()

    chimera = ChimeraBoostRegressor(
        n_estimators=2000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=1.0,
        max_bins=128,
        min_child_weight=1.0,
        linear_leaves=None,
        cross_features=None,
        selection_rounds=None,
        early_stopping=True,
        random_state=RANDOM_STATE,
        thread_count=THREADS,
    )
    chimera_fit_seconds = _fit(
        chimera,
        X.iloc[fit_indices],
        y.iloc[fit_indices],
        eval_X,
        eval_y,
    )
    chimera_actual_prediction = chimera.predict(X.iloc[outer_test])
    chimera_best_rounds = int(np.argmin(chimera.model_.valid_history_)) + 1
    chimera_prediction = _staged_prediction_at(
        chimera, X.iloc[outer_test], chimera_best_rounds
    )

    y_test = y.iloc[outer_test]
    base_fingerprint = _fingerprint(base, base_prediction, y_test)
    selected_fingerprint = _fingerprint(
        selected, selected_prediction, y_test
    )
    chimera_fingerprint = _fingerprint(
        chimera,
        chimera_prediction,
        y_test,
        tree_limit=chimera_best_rounds,
    )
    chimera_actual_fingerprint = _fingerprint(
        chimera,
        chimera_actual_prediction,
        y_test,
        tree_limit=len(chimera.model_.trees_),
    )
    mismatches = [
        field
        for field in EXACT_FIELDS
        if selected_fingerprint[field] != chimera_fingerprint[field]
    ]
    chimera_pairs = list(chimera.cross_pairs_ or ())
    selected_pairs = pairs if cross_selected else []
    if (
        bool(chimera.linear_leaves_selected_) != base_linear
        or bool(chimera.cross_features_selected_) != cross_selected
        or chimera_pairs != selected_pairs
        or mismatches
    ):
        raise RuntimeError(
            f"external/native cross mismatch on {task_id}/{fold}: "
            f"fields={mismatches}"
        )
    return {
        "task_id": int(task_id),
        "dataset_name": TASKS[task_id],
        "fold": int(fold),
        "outer_train_index_sha256": _array_sha256(outer_train, "<i8"),
        "outer_test_index_sha256": _array_sha256(outer_test, "<i8"),
        "fit_index_sha256": _array_sha256(fit_indices, "<i8"),
        "validation_index_sha256": _array_sha256(
            validation_indices, "<i8"
        ),
        "base_linear_selected": base_linear,
        "cross_selected": cross_selected,
        "candidate_cross_pairs": [list(pair) for pair in pairs],
        "selected_cross_pairs": [list(pair) for pair in selected_pairs],
        "base": base_fingerprint,
        "selected": selected_fingerprint,
        "chimera": chimera_fingerprint,
        "chimera_actual": chimera_actual_fingerprint,
        "external_native_exact": True,
        "darko_total_fit_seconds": float(darko_fit_seconds),
        "chimera_total_fit_seconds": float(chimera_fit_seconds),
    }


def _geomean(values):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise RuntimeError("geomean requires finite positive values")
    return float(np.exp(np.mean(np.log(values))))


def _valid_candidate_pairs(pairs, *, n_features) -> bool:
    if not isinstance(pairs, list):
        return False
    normalized = []
    ranked_features = []
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 3
            or not isinstance(pair[0], int)
            or isinstance(pair[0], bool)
            or not isinstance(pair[1], int)
            or isinstance(pair[1], bool)
            or pair[0] < 0
            or pair[1] < 0
            or pair[0] == pair[1]
            or pair[0] >= n_features
            or pair[1] >= n_features
            or pair[2] not in {"diff", "prod"}
        ):
            return False
        normalized.append(tuple(pair))
        for feature in pair[:2]:
            if feature not in ranked_features:
                ranked_features.append(feature)
    if len(ranked_features) != TOP_NUMERIC_FEATURES:
        return False
    expected = [
        (ranked_features[left], ranked_features[right], operation)
        for left in range(len(ranked_features))
        for right in range(left + 1, len(ranked_features))
        for operation in ("diff", "prod")
    ]
    return normalized == expected


def _positive_float(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(
            f"smooth cross-feature {label} must be finite and positive"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(
            f"smooth cross-feature {label} must be finite and positive"
        )
    return result


def _positive_int(value, label):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise RuntimeError(
            f"smooth cross-feature {label} must be a positive integer"
        )
    return value


def _validate_fingerprint(record, label, *, fingerprinted_prefix):
    if not isinstance(record, dict):
        raise RuntimeError(f"smooth cross-feature {label} fingerprint is invalid")
    actual = _positive_int(
        record.get("actual_retained_tree_count"),
        f"{label} retained tree count",
    )
    best = _positive_int(
        record.get("best_prefix_tree_count"),
        f"{label} best-prefix tree count",
    )
    fingerprinted = _positive_int(
        record.get("fingerprinted_tree_count"),
        f"{label} fingerprinted tree count",
    )
    if best > actual or fingerprinted > actual:
        raise RuntimeError(
            f"smooth cross-feature {label} tree-count ledger is invalid"
        )
    if fingerprinted_prefix and fingerprinted != best:
        raise RuntimeError(
            f"smooth cross-feature {label} prefix ledger is invalid"
        )
    if not fingerprinted_prefix and fingerprinted != actual:
        raise RuntimeError(
            f"smooth cross-feature {label} retained-model ledger is invalid"
        )
    _positive_float(
        record.get("best_validation_rmse"),
        f"{label} validation RMSE",
    )
    _positive_float(record.get("test_rmse"), f"{label} test RMSE")
    for field in (
        "prediction_sha256",
        "validation_history_sha256",
        "borders_sha256",
        "model_sha256",
    ):
        if not isinstance(record.get(field), str) or not record[field]:
            raise RuntimeError(
                f"smooth cross-feature {label} {field} is invalid"
            )
    return actual, best, fingerprinted


def analyze(rows):
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("task_id"), int)
        or isinstance(row.get("task_id"), bool)
        or not isinstance(row.get("fold"), int)
        or isinstance(row.get("fold"), bool)
        for row in rows
    ):
        raise RuntimeError("smooth cross-feature coordinate ledger is invalid")
    expected = {(task_id, fold) for task_id in TASKS for fold in FOLDS}
    actual = {(row["task_id"], row["fold"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("smooth cross-feature coordinate set is incomplete")
    for row in rows:
        task_id = int(row["task_id"])
        if row.get("dataset_name") != TASKS[task_id]:
            raise RuntimeError("smooth cross-feature dataset identity changed")
        if row.get("external_native_exact") is not True:
            raise RuntimeError("external/native cross parity is not exact")
        if not isinstance(row.get("cross_selected"), bool) or not isinstance(
            row.get("base_linear_selected"), bool
        ):
            raise RuntimeError("smooth cross-feature selector ledger is invalid")
        base_counts = _validate_fingerprint(
            row.get("base"),
            "base",
            fingerprinted_prefix=False,
        )
        selected_counts = _validate_fingerprint(
            row.get("selected"),
            "selected",
            fingerprinted_prefix=False,
        )
        chimera_counts = _validate_fingerprint(
            row.get("chimera"),
            "normalized ChimeraBoost",
            fingerprinted_prefix=True,
        )
        chimera_actual_counts = _validate_fingerprint(
            row.get("chimera_actual"),
            "actual ChimeraBoost",
            fingerprinted_prefix=False,
        )
        if base_counts[0] != base_counts[1] or (
            selected_counts[0] != selected_counts[1]
        ):
            raise RuntimeError(
                "smooth cross-feature DarkoFit retention ledger changed"
            )
        if (
            chimera_counts[:2] != chimera_actual_counts[:2]
            or any(
                row["chimera"].get(field)
                != row["chimera_actual"].get(field)
                for field in (
                    "best_validation_rmse",
                    "validation_history_sha256",
                    "borders_sha256",
                )
            )
        ):
            raise RuntimeError(
                "smooth cross-feature ChimeraBoost retention ledger changed"
            )
        if chimera_counts[0] == chimera_counts[1] and any(
            row["chimera"].get(field) != row["chimera_actual"].get(field)
            for field in EXACT_FIELDS
        ):
            raise RuntimeError(
                "smooth cross-feature ChimeraBoost retained model changed"
            )
        if not _valid_candidate_pairs(
            row.get("candidate_cross_pairs"),
            n_features=TASK_FEATURE_COUNTS[task_id],
        ):
            raise RuntimeError("smooth cross-feature candidate pair ledger changed")
        expected_pairs = (
            row["candidate_cross_pairs"] if row["cross_selected"] else []
        )
        if row["selected_cross_pairs"] != expected_pairs:
            raise RuntimeError("smooth cross-feature selected pair ledger changed")
        base_best = float(row["base"]["best_validation_rmse"])
        selected_best = float(row["selected"]["best_validation_rmse"])
        if (
            not math.isfinite(base_best)
            or not math.isfinite(selected_best)
            or base_best <= 0.0
            or selected_best <= 0.0
            or row["cross_selected"] != (selected_best < base_best)
        ):
            raise RuntimeError("smooth cross-feature selection decision changed")
        if not row["cross_selected"] and any(
            field not in row["base"]
            or field not in row["selected"]
            or row["base"][field] != row["selected"][field]
            for field in EXACT_FIELDS
        ):
            raise RuntimeError("smooth cross-feature declined selection changed")
        mismatches = [
            field
            for field in EXACT_FIELDS
            if field not in row["selected"]
            or field not in row["chimera"]
            or row["selected"][field] != row["chimera"][field]
        ]
        if mismatches:
            raise RuntimeError(
                "external/native cross parity fields differ: "
                f"{mismatches}"
            )
        for field in ("darko_total_fit_seconds", "chimera_total_fit_seconds"):
            _positive_float(row.get(field), f"{field} ledger")

    datasets = {}
    for task_id, name in TASKS.items():
        task_rows = [row for row in rows if row["task_id"] == task_id]
        ratios = [
            row["selected"]["test_rmse"] / row["base"]["test_rmse"]
            for row in task_rows
        ]
        datasets[name] = {
            "task_id": task_id,
            "geomean_ratio": _geomean(ratios),
            "worst_split_ratio": float(max(ratios)),
            "cross_selected_count": int(
                sum(row["cross_selected"] for row in task_rows)
            ),
            "linear_selected_count": int(
                sum(row["base_linear_selected"] for row in task_rows)
            ),
        }
    dataset_ratios = [
        record["geomean_ratio"] for record in datasets.values()
    ]
    leave_one_out = {
        name: _geomean(
            [
                record["geomean_ratio"]
                for other, record in datasets.items()
                if other != name
            ]
        )
        for name in datasets
    }
    return {
        "claim_tier": "development_diagnostic_only",
        "fresh_claim_eligible": False,
        "external_native_exact": True,
        "coordinate_count": len(rows),
        "equal_dataset_geomean_ratio": _geomean(dataset_ratios),
        "worst_dataset_ratio": float(max(dataset_ratios)),
        "worst_split_ratio": float(
            max(
                row["selected"]["test_rmse"] / row["base"]["test_rmse"]
                for row in rows
            )
        ),
        "leave_one_out_equal_dataset_ratios": leave_one_out,
        "datasets": datasets,
        "cross_selected_coordinates": int(
            sum(row["cross_selected"] for row in rows)
        ),
        "linear_selected_coordinates": int(
            sum(row["base_linear_selected"] for row in rows)
        ),
        "summed_darko_fit_seconds": float(
            sum(row["darko_total_fit_seconds"] for row in rows)
        ),
        "summed_chimera_fit_seconds": float(
            sum(row["chimera_total_fit_seconds"] for row in rows)
        ),
        "timing_claim_eligible": False,
    }


def _is_hex_digest(value, length) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value) -> bool:
    return _is_hex_digest(value, 40)


def _is_sha256(value) -> bool:
    return _is_hex_digest(value, 64)


def _valid_aware_timestamp(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _analysis_equal(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(
            left,
            right,
            rel_tol=1e-14,
            abs_tol=1e-15,
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _analysis_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _analysis_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def validate_artifact(artifact, *, require_frozen=True):
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 1
        or isinstance(artifact.get("schema_version"), bool)
        or not _valid_aware_timestamp(artifact.get("created_at"))
    ):
        raise RuntimeError("smooth cross-feature artifact has an unknown schema")
    protocol = artifact.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("name") != "smooth_cross_features"
        or protocol.get("path") != str(PROTOCOL.relative_to(ROOT))
        or protocol.get("sha256") != _sha256(PROTOCOL)
        or protocol.get("folds") != list(FOLDS)
    ):
        raise RuntimeError("smooth cross-feature protocol ledger changed")
    if protocol.get("tasks") != {
        str(task_id): name for task_id, name in TASKS.items()
    }:
        raise RuntimeError("smooth cross-feature protocol task ledger changed")

    if artifact.get("partition_boundary") != _partition_boundary():
        raise RuntimeError("smooth cross-feature partition ledger changed")
    sources = artifact.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "darkofit",
        "chimeraboost",
    }:
        raise RuntimeError("smooth cross-feature source ledger changed")
    for name, source in sources.items():
        if (
            not isinstance(source, dict)
            or source.get("clean") is not True
            or not isinstance(source.get("path"), str)
            or not Path(source["path"]).is_absolute()
            or not isinstance(source.get("branch"), str)
            or not source["branch"]
            or not _is_git_sha(source.get("head"))
        ):
            raise RuntimeError(
                f"smooth cross-feature {name} source ledger changed"
            )
    if sources["chimeraboost"]["head"] != EXPECTED_CHIMERA_HEAD:
        raise RuntimeError(
            "smooth cross-feature ChimeraBoost source ledger changed"
        )

    tasks = artifact.get("tasks")
    if not isinstance(tasks, dict):
        raise RuntimeError("smooth cross-feature task metadata changed")
    if set(tasks) != {str(task_id) for task_id in TASKS}:
        raise RuntimeError("smooth cross-feature task metadata changed")
    for task_id in TASKS:
        metadata = tasks[str(task_id)]
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("task_id"), int)
            or isinstance(metadata.get("task_id"), bool)
            or metadata["task_id"] != task_id
            or metadata.get("dataset_name") != TASKS[task_id]
            or not isinstance(metadata.get("features"), int)
            or isinstance(metadata.get("features"), bool)
            or metadata["features"] != TASK_FEATURE_COUNTS[task_id]
            or not isinstance(metadata.get("rows"), int)
            or isinstance(metadata.get("rows"), bool)
            or metadata["rows"] <= 0
            or not isinstance(metadata.get("dataset_id"), int)
            or isinstance(metadata.get("dataset_id"), bool)
            or metadata["dataset_id"] <= 0
            or not isinstance(metadata.get("target_name"), str)
            or not metadata["target_name"]
            or not _is_sha256(metadata.get("X_sha256"))
            or not _is_sha256(metadata.get("y_sha256"))
        ):
            raise RuntimeError(
                f"smooth cross-feature task {task_id} metadata changed"
            )

    rows = artifact.get("results")
    if not isinstance(rows, list):
        raise RuntimeError("smooth cross-feature result ledger changed")
    for row in rows:
        if not isinstance(row, dict) or any(
            not _is_sha256(row.get(field))
            for field in (
                "outer_train_index_sha256",
                "outer_test_index_sha256",
                "fit_index_sha256",
                "validation_index_sha256",
            )
        ):
            raise RuntimeError("smooth cross-feature split ledger changed")
        for fingerprint_name in (
            "base",
            "selected",
            "chimera",
            "chimera_actual",
        ):
            fingerprint = row.get(fingerprint_name)
            if not isinstance(fingerprint, dict) or any(
                not _is_sha256(fingerprint.get(field))
                for field in (
                    "prediction_sha256",
                    "validation_history_sha256",
                    "borders_sha256",
                    "model_sha256",
                )
            ):
                raise RuntimeError(
                    "smooth cross-feature fingerprint ledger changed"
                )
    frozen_evidence = {
        "protocol": protocol,
        "source_revisions": {
            name: {
                "head": source["head"],
                "branch": source["branch"],
                "clean": source["clean"],
            }
            for name, source in sources.items()
        },
        "partition_boundary": artifact["partition_boundary"],
        "tasks": tasks,
        "results": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "darko_total_fit_seconds",
                    "chimera_total_fit_seconds",
                }
            }
            for row in rows
        ],
    }
    if _json_sha256(frozen_evidence) != FROZEN_EVIDENCE_SHA256:
        raise RuntimeError(
            "smooth cross-feature frozen evidence ledger changed"
        )
    analysis = analyze(rows)
    stored_analysis = artifact.get("analysis")
    if (
        not isinstance(stored_analysis, dict)
        or stored_analysis.get("claim_tier")
        != "development_diagnostic_only"
        or stored_analysis.get("fresh_claim_eligible") is not False
        or stored_analysis.get("timing_claim_eligible") is not False
    ):
        raise RuntimeError("smooth cross-feature stored analysis changed")
    if not _analysis_equal(stored_analysis, analysis):
        raise RuntimeError("smooth cross-feature stored analysis is not reproducible")
    if (
        require_frozen
        and _json_sha256(artifact) != FROZEN_ARTIFACT_SHA256
    ):
        raise RuntimeError("smooth cross-feature frozen artifact changed")
    return analysis


def run(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    protocol_sha256 = _sha256(PROTOCOL)
    darko_source = _source_state(ROOT)
    chimera_source = _source_state(
        CHIMERA_ROOT, expected_head=EXPECTED_CHIMERA_HEAD
    )
    partition = _partition_boundary()
    rows = []
    tasks = {}
    for task_id in TASKS:
        task, X, y, metadata = _load_task(task_id)
        tasks[str(task_id)] = metadata
        for fold in FOLDS:
            rows.append(_evaluate_fold(task, X, y, task_id, fold))
    analysis = analyze(rows)
    if _source_state(ROOT) != darko_source:
        raise RuntimeError("DarkoFit source changed during campaign")
    if (
        _source_state(
            CHIMERA_ROOT, expected_head=EXPECTED_CHIMERA_HEAD
        )
        != chimera_source
    ):
        raise RuntimeError("ChimeraBoost source changed during campaign")
    if _sha256(PROTOCOL) != protocol_sha256:
        raise RuntimeError("smooth cross-feature protocol changed during campaign")
    if _partition_boundary() != partition:
        raise RuntimeError(
            "smooth cross-feature partition changed during campaign"
        )
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "smooth_cross_features",
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": protocol_sha256,
            "tasks": {
                str(task_id): name for task_id, name in TASKS.items()
            },
            "folds": list(FOLDS),
        },
        "sources": {
            "darkofit": darko_source,
            "chimeraboost": chimera_source,
        },
        "partition_boundary": partition,
        "tasks": tasks,
        "results": rows,
        "analysis": analysis,
    }
    validate_artifact(artifact, require_frozen=False)
    _atomic_create(
        output,
        (
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = run(args.output.resolve())
    print(json.dumps(artifact["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
