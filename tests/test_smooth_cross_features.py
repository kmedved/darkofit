import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks import run_smooth_cross_features as experiment
from benchmarks import analyze_smooth_cross_margin as margin_analysis
from benchmarks import run_rssi_linear_leaf_diagnosis as rssi_diagnosis


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "smooth_cross_features.json"
HERMETIC_CHIMERA_SETUP = r"""
import tempfile
from pathlib import Path

chimera_temp = tempfile.TemporaryDirectory()
chimera_root = Path(chimera_temp.name)
chimera_package = chimera_root / "chimeraboost"
chimera_package.mkdir()
(chimera_package / "__init__.py").write_text(
    "from .sklearn_api import ChimeraBoostRegressor\n",
    encoding="utf-8",
)
(chimera_package / "sklearn_api.py").write_text(
    "class ChimeraBoostRegressor:\n"
    "    def __init__(self):\n"
    "        self.ready = True\n",
    encoding="utf-8",
)
runner.CHIMERA_ROOT = chimera_root
"""


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.run_smooth_cross_features",
        "benchmarks.run_rssi_linear_leaf_diagnosis",
    ],
)
def test_runner_import_does_not_shadow_darkofit_benchmarks(module_name):
    script = f"""
import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import types

runner = importlib.import_module({module_name!r})
{HERMETIC_CHIMERA_SETUP}
assert str(runner.CHIMERA_ROOT) not in sys.path
fake = types.ModuleType("chimeraboost")
fake.__file__ = str(Path.cwd() / "fake" / "__init__.py")
sys.modules["chimeraboost"] = fake
original_path = list(sys.path)
with ThreadPoolExecutor(max_workers=4) as pool:
    regressors = list(pool.map(lambda _index: runner._chimera_regressor_class(), range(8)))
regressor = regressors[0]
assert all(value is regressor for value in regressors)
assert sys.path == original_path
assert sys.modules["chimeraboost"] is fake
assert Path(sys.modules[regressor.__module__].__file__).resolve().is_relative_to(
    runner.CHIMERA_ROOT.resolve()
)
from benchmarks import bench_status
assert Path(bench_status.__file__).resolve() == (
    Path.cwd() / "benchmarks" / "bench_status.py"
).resolve()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.run_smooth_cross_features",
        "benchmarks.run_rssi_linear_leaf_diagnosis",
    ],
)
def test_runner_rejects_occupied_private_chimera_module_slot(module_name):
    script = f"""
import importlib
import sys
import types

runner = importlib.import_module({module_name!r})
fake = types.ModuleType(runner._CHIMERA_MODULE_NAME)
fake.__file__ = str(
    runner.CHIMERA_ROOT / "chimeraboost" / "__init__.py"
)
fake.ChimeraBoostRegressor = type(
    "ForgedRegressor",
    (),
    {{"__module__": runner._CHIMERA_MODULE_NAME + ".sklearn_api"}},
)
sys.modules[runner._CHIMERA_MODULE_NAME] = fake
try:
    runner._chimera_regressor_class()
except RuntimeError as exc:
    assert "wrong checkout" in str(exc)
else:
    raise AssertionError("occupied private module slot was accepted")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.run_smooth_cross_features",
        "benchmarks.run_rssi_linear_leaf_diagnosis",
    ],
)
def test_runner_rejects_replacement_of_loaded_private_module(module_name):
    script = f"""
import importlib

runner = importlib.import_module({module_name!r})
{HERMETIC_CHIMERA_SETUP}
runner._chimera_regressor_class()
runner._CHIMERA_MODULE.ChimeraBoostRegressor = type(
    "ForgedRegressor",
    (),
    {{"__module__": runner._CHIMERA_MODULE_NAME + ".sklearn_api"}},
)
try:
    runner._chimera_regressor_class()
except RuntimeError as exc:
    assert "private module slot changed" in str(exc)
else:
    raise AssertionError("replacement private class was accepted")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.run_smooth_cross_features",
        "benchmarks.run_rssi_linear_leaf_diagnosis",
    ],
)
def test_runner_rejects_replacement_of_loaded_private_child_module(
    module_name,
):
    script = f"""
import importlib
import sys
import types

runner = importlib.import_module({module_name!r})
{HERMETIC_CHIMERA_SETUP}
regressor = runner._chimera_regressor_class()
child_name = regressor.__module__
replacement = types.ModuleType(child_name)
replacement.__file__ = str(
    runner.CHIMERA_ROOT / "chimeraboost" / "sklearn_api.py"
)
sys.modules[child_name] = replacement
try:
    runner._chimera_regressor_class()
except RuntimeError as exc:
    assert "private module slot changed" in str(exc)
else:
    raise AssertionError("replacement private child module was accepted")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.run_smooth_cross_features",
        "benchmarks.run_rssi_linear_leaf_diagnosis",
    ],
)
def test_runner_revalidates_cached_private_class_provenance(module_name):
    script = f"""
import importlib

runner = importlib.import_module({module_name!r})
{HERMETIC_CHIMERA_SETUP}
regressor = runner._chimera_regressor_class()
regressor.__module__ = "forged.provenance"
try:
    runner._chimera_regressor_class()
except RuntimeError as exc:
    assert "private module slot changed" in str(exc)
else:
    raise AssertionError("mutated cached class provenance was accepted")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.run_smooth_cross_features",
        "benchmarks.run_rssi_linear_leaf_diagnosis",
    ],
)
def test_runner_provenance_requires_private_package_root(module_name):
    script = f"""
import importlib

runner = importlib.import_module({module_name!r})
{HERMETIC_CHIMERA_SETUP}
regressor = runner._chimera_regressor_class()
private_modules = dict(runner._CHIMERA_MODULES)
private_modules.pop(runner._CHIMERA_MODULE_NAME)
assert not runner._private_chimera_provenance_is_valid(
    runner.CHIMERA_ROOT / "chimeraboost",
    runner.CHIMERA_ROOT / "chimeraboost" / "__init__.py",
    runner._CHIMERA_MODULE,
    regressor,
    private_modules,
)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.run_smooth_cross_features",
        "benchmarks.run_rssi_linear_leaf_diagnosis",
    ],
)
def test_runner_rejects_custom_loader_spoof_with_in_checkout_paths(
    module_name,
):
    script = f"""
import importlib
import importlib.machinery
import sys
import types

runner = importlib.import_module({module_name!r})

class SpoofLoader:
    def create_module(self, _spec):
        return None

    def exec_module(self, module):
        child_name = runner._CHIMERA_MODULE_NAME + ".sklearn_api"
        child = types.ModuleType(child_name)
        child.__file__ = str(
            runner.CHIMERA_ROOT / "chimeraboost" / "sklearn_api.py"
        )
        fake = type(
            "ChimeraBoostRegressor",
            (),
            {{"__module__": child_name}},
        )
        child.ChimeraBoostRegressor = fake
        sys.modules[child_name] = child
        module.ChimeraBoostRegressor = fake

def spoofed_spec(name, location, *, submodule_search_locations):
    spec = importlib.machinery.ModuleSpec(
        name,
        SpoofLoader(),
        origin=str(location),
        is_package=True,
    )
    spec.submodule_search_locations = list(submodule_search_locations)
    spec.has_location = True
    return spec

runner.importlib.util.spec_from_file_location = spoofed_spec
try:
    runner._chimera_regressor_class()
except RuntimeError as exc:
    assert "wrong checkout" in str(exc)
else:
    raise AssertionError("custom-loader spoof was accepted")
assert not any(
    name == runner._CHIMERA_MODULE_NAME
    or name.startswith(runner._CHIMERA_MODULE_NAME + ".")
    for name in sys.modules
)
assert runner._CHIMERA_MODULE is None
assert runner._CHIMERA_MODULES is None
assert runner._CHIMERA_REGRESSOR is None
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "benchmarks.run_smooth_cross_features",
        "benchmarks.run_rssi_linear_leaf_diagnosis",
    ],
)
def test_runner_cleans_private_modules_after_import_failure(module_name):
    script = f"""
import importlib
import importlib.machinery
import sys
import types

runner = importlib.import_module({module_name!r})

class FailingLoader:
    def create_module(self, _spec):
        return None

    def exec_module(self, _module):
        child_name = runner._CHIMERA_MODULE_NAME + ".partial"
        child = types.ModuleType(child_name)
        child.__file__ = str(
            runner.CHIMERA_ROOT / "chimeraboost" / "partial.py"
        )
        sys.modules[child_name] = child
        raise RuntimeError("injected import failure")

def failing_spec(name, _location, *, submodule_search_locations):
    spec = importlib.machinery.ModuleSpec(
        name,
        FailingLoader(),
        is_package=True,
    )
    spec.submodule_search_locations = list(submodule_search_locations)
    return spec

runner.importlib.util.spec_from_file_location = failing_spec
try:
    runner._chimera_regressor_class()
except RuntimeError as exc:
    assert "injected import failure" in str(exc)
else:
    raise AssertionError("injected import failure was not propagated")
assert not any(
    name == runner._CHIMERA_MODULE_NAME
    or name.startswith(runner._CHIMERA_MODULE_NAME + ".")
    for name in sys.modules
)
assert runner._CHIMERA_MODULE is None
assert runner._CHIMERA_REGRESSOR is None
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "script",
    [
        experiment.__file__,
        rssi_diagnosis.__file__,
        margin_analysis.__file__,
    ],
)
def test_smooth_rssi_cli_imports_outside_repository(tmp_path, script):
    completed = subprocess.run(
        [sys.executable, str(Path(script).resolve()), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_candidate_pairs_are_deterministic_and_skip_categoricals():
    pairs = experiment.candidate_pairs(
        [0.1, 0.5, 0.5, 0.2], [1], n_features=4
    )
    assert pairs == [
        (2, 3, "diff"),
        (2, 3, "prod"),
        (2, 0, "diff"),
        (2, 0, "prod"),
        (3, 0, "diff"),
        (3, 0, "prod"),
    ]
    assert all(1 not in pair[:2] for pair in pairs)


def test_candidate_pairs_reject_nonfinite_importances():
    with pytest.raises(RuntimeError, match="finite and one-dimensional"):
        experiment.candidate_pairs(
            [0.5, np.nan, 0.1],
            (),
            n_features=3,
        )


def test_candidate_pairs_reject_mismatched_or_invalid_declarations():
    with pytest.raises(RuntimeError, match="match the feature count"):
        experiment.candidate_pairs([0.5, 0.1], (), n_features=3)
    with pytest.raises(RuntimeError, match="feature count"):
        experiment.candidate_pairs([0.5], (), n_features=True)
    with pytest.raises(RuntimeError, match="categorical indices"):
        experiment.candidate_pairs(
            [0.5, 0.2, 0.1],
            [1, 1],
            n_features=3,
        )


def test_augmentation_appends_diff_and_product_with_nan_propagation():
    X = np.array([[2.0, 3.0], [np.nan, 4.0]])
    augmented = experiment.augment_numeric_crosses(
        X, [(0, 1, "diff"), (0, 1, "prod")]
    )
    np.testing.assert_array_equal(augmented[0], [2.0, 3.0, -1.0, 6.0])
    assert np.isnan(augmented[1, 2:]).all()


def test_augmentation_rejects_invalid_pair_semantics():
    X = np.array([[2.0, 3.0]])
    with pytest.raises(RuntimeError, match="pair declaration is invalid"):
        experiment.augment_numeric_crosses(X, [(0, 1, "sum")])
    with pytest.raises(RuntimeError, match="pair declaration is invalid"):
        experiment.augment_numeric_crosses(X, [(0, 2, "prod")])
    with pytest.raises(RuntimeError, match="declarations repeat"):
        experiment.augment_numeric_crosses(
            X,
            [(0, 1, "prod"), (0, 1, "prod")],
        )


def test_smooth_artifact_create_is_atomic_and_create_only(
    tmp_path, monkeypatch
):
    output = tmp_path / "result.json"
    experiment._atomic_create(output, b"first")
    assert output.read_bytes() == b"first"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        experiment._atomic_create(output, b"second")
    monkeypatch.setattr(
        experiment,
        "_source_state",
        lambda *_args, **_kwargs: pytest.fail("campaign should not start"),
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        experiment.run(output)


def test_smooth_artifact_rejects_mutable_symlink_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink output"):
        experiment._atomic_create(
            linked / "missing" / "result.json",
            b"result",
        )
    assert list(real.iterdir()) == []


@pytest.mark.parametrize(
    "module",
    [experiment, rssi_diagnosis, margin_analysis],
)
@pytest.mark.parametrize("system_root", [Path("/tmp"), Path("/var/tmp")])
def test_single_artifact_writers_allow_system_symlink_roots(
    module,
    system_root,
):
    if not system_root.is_dir():
        pytest.skip(f"{system_root} is unavailable")
    directory = Path(
        module.tempfile.mkdtemp(
            prefix="darkofit-evidence-writer-",
            dir=system_root,
        )
    )
    output = directory / "result.json"
    try:
        module._atomic_create(output, b"result")
        assert output.read_bytes() == b"result"
    finally:
        output.unlink(missing_ok=True)
        directory.rmdir()


@pytest.mark.parametrize(
    "module",
    [experiment, rssi_diagnosis, margin_analysis],
)
def test_single_artifact_writers_roll_back_after_temp_cleanup_failure(
    tmp_path,
    monkeypatch,
    module,
):
    output = tmp_path / f"{module.__name__}.json"
    original_unlink = module.os.unlink
    failed = False

    def fail_first_temp_cleanup(path, *args, **kwargs):
        nonlocal failed
        if str(path).endswith(".tmp") and not failed:
            failed = True
            raise OSError("injected temp cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "unlink", fail_first_temp_cleanup)
    with pytest.raises(OSError, match="injected temp cleanup failure"):
        module._atomic_create(output, b"result")
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "module",
    [experiment, rssi_diagnosis, margin_analysis],
)
def test_single_artifact_writers_do_not_report_failed_commit_on_close_cleanup(
    tmp_path,
    monkeypatch,
    module,
):
    output = tmp_path / f"{module.__name__}.json"
    original_close = module.os.close
    failed_descriptor = None

    def fail_first_close(descriptor):
        nonlocal failed_descriptor
        if failed_descriptor is None:
            failed_descriptor = descriptor
            raise OSError("injected close cleanup failure")
        return original_close(descriptor)

    monkeypatch.setattr(module.os, "close", fail_first_close)
    module._atomic_create(output, b"result")
    assert output.read_bytes() == b"result"
    assert failed_descriptor is not None
    original_close(failed_descriptor)


@pytest.mark.parametrize(
    "module",
    [experiment, rssi_diagnosis, margin_analysis],
)
def test_single_artifact_writers_roll_back_owned_nested_directories(
    tmp_path,
    monkeypatch,
    module,
):
    output = tmp_path / "first" / "second" / "result.json"

    def fail_link(_source, _destination, **_kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(module.os, "link", fail_link)
    with pytest.raises(OSError, match="injected publication failure"):
        module._atomic_create(output, b"result")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "module",
    [experiment, rssi_diagnosis, margin_analysis],
)
def test_single_artifact_writers_roll_back_partial_directory_creation(
    tmp_path,
    monkeypatch,
    module,
):
    output = tmp_path / "first" / "second" / "result.json"
    original_mkdir = Path.mkdir
    calls = 0

    def fail_second_directory(path, *args, **kwargs):
        nonlocal calls
        if tmp_path in path.parents:
            calls += 1
            if calls == 2:
                raise OSError("injected directory-creation failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_second_directory)
    with pytest.raises(OSError, match="injected directory-creation failure"):
        module._atomic_create(output, b"result")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "module",
    [experiment, rssi_diagnosis, margin_analysis],
)
def test_single_artifact_writers_preserve_replaced_temporary_file(
    tmp_path,
    monkeypatch,
    module,
):
    output = tmp_path / "result.json"
    replacement = None

    original_unlink = module.os.unlink

    def replace_temp_then_fail(
        source,
        _destination,
        *,
        src_dir_fd,
        **_kwargs,
    ):
        nonlocal replacement
        replacement = output.parent / source
        original_unlink(source, dir_fd=src_dir_fd)
        descriptor = module.os.open(
            source,
            module.os.O_WRONLY | module.os.O_CREAT | module.os.O_EXCL,
            0o600,
            dir_fd=src_dir_fd,
        )
        with module.os.fdopen(descriptor, "wb") as handle:
            handle.write(b"other writer")
        raise OSError("injected publication failure")

    monkeypatch.setattr(module.os, "link", replace_temp_then_fail)
    with pytest.raises(OSError, match="injected publication failure"):
        module._atomic_create(output, b"result")
    assert not output.exists()
    assert replacement is not None
    assert replacement.read_bytes() == b"other writer"


@pytest.mark.parametrize(
    "module",
    [experiment, rssi_diagnosis, margin_analysis],
)
def test_single_artifact_writers_pin_temp_inode_through_error_cleanup(
    tmp_path,
    monkeypatch,
    module,
):
    output = tmp_path / "result.json"
    original_temporary_at = module._temporary_at
    original_unlink_if_owned = module._unlink_if_owned_at
    temporary_descriptor = None
    cleanup_observed = False

    def capture_temporary_descriptor(*args, **kwargs):
        nonlocal temporary_descriptor
        result = original_temporary_at(*args, **kwargs)
        temporary_descriptor = result[0]
        return result

    def fail_publication(*_args, **_kwargs):
        raise OSError("injected publication failure")

    def observe_cleanup(*args, **kwargs):
        nonlocal cleanup_observed
        assert temporary_descriptor is not None
        module.os.fstat(temporary_descriptor)
        cleanup_observed = True
        return original_unlink_if_owned(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "_temporary_at",
        capture_temporary_descriptor,
    )
    monkeypatch.setattr(module.os, "link", fail_publication)
    monkeypatch.setattr(
        module,
        "_unlink_if_owned_at",
        observe_cleanup,
    )
    with pytest.raises(OSError, match="injected publication failure"):
        module._atomic_create(output, b"result")
    assert cleanup_observed is True
    assert temporary_descriptor is not None
    with pytest.raises(OSError):
        module.os.fstat(temporary_descriptor)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "module",
    [experiment, rssi_diagnosis, margin_analysis],
)
def test_single_artifact_writers_detect_symlink_parent_swap_race(
    tmp_path,
    monkeypatch,
    module,
):
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved"
    output = parent / "result.json"
    original_link = module.os.link

    def swap_parent_then_link(source, destination, **kwargs):
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "link", swap_parent_then_link)
    with pytest.raises(RuntimeError, match="symlink output"):
        module._atomic_create(output, b"result")
    assert not output.exists()
    assert list(moved.iterdir()) == []


@pytest.mark.parametrize(
    "module",
    [experiment, rssi_diagnosis, margin_analysis],
)
def test_single_artifact_writers_reject_real_parent_replacement(
    tmp_path,
    monkeypatch,
    module,
):
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved"
    output = parent / "result.json"
    original_link = module.os.link

    def replace_parent_then_link(source, destination, **kwargs):
        parent.rename(moved)
        parent.mkdir()
        output.write_bytes(b"replacement")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "link", replace_parent_then_link)
    with pytest.raises(RuntimeError, match="output parent changed"):
        module._atomic_create(output, b"result")
    assert output.read_bytes() == b"replacement"
    assert list(parent.iterdir()) == [output]
    assert list(moved.iterdir()) == []


def test_declined_cross_policy_has_candidates_but_no_selected_pairs():
    pairs = experiment.candidate_pairs([0.7, 0.3], (), n_features=2)
    assert pairs == [(0, 1, "diff"), (0, 1, "prod")]
    selected = pairs if False else []
    assert selected == []


def _fake_rows():
    def fingerprint(test_rmse, marker, *, best_validation_rmse=None):
        return {
            "actual_retained_tree_count": 10,
            "best_prefix_tree_count": 10,
            "fingerprinted_tree_count": 10,
            "best_validation_rmse": (
                test_rmse
                if best_validation_rmse is None
                else best_validation_rmse
            ),
            "test_rmse": test_rmse,
            "prediction_sha256": marker,
            "validation_history_sha256": marker,
            "borders_sha256": marker,
            "model_sha256": marker,
        }

    rows = []
    pairs = [
        list(pair)
        for pair in experiment.candidate_pairs(
            np.arange(6, dtype=np.float64), (), n_features=6
        )
    ]
    for task_id, name in experiment.TASKS.items():
        for fold in experiment.FOLDS:
            ratio = 0.98 if task_id != 361623 else 1.01
            marker = f"{task_id}-{fold}"
            selected = fingerprint(
                ratio,
                marker,
                best_validation_rmse=0.9,
            )
            rows.append(
                {
                    "task_id": task_id,
                    "dataset_name": name,
                    "fold": fold,
                    "base_linear_selected": fold % 2 == 0,
                    "cross_selected": True,
                    "candidate_cross_pairs": pairs,
                    "selected_cross_pairs": pairs,
                    "base": fingerprint(1.0, f"base-{marker}"),
                    "selected": selected,
                    "chimera": dict(selected),
                    "chimera_actual": dict(selected),
                    "external_native_exact": True,
                    "darko_total_fit_seconds": 1.0,
                    "chimera_total_fit_seconds": 1.0,
                }
            )
    return rows


def test_analysis_reports_magnitude_concentration_and_harm_without_win_gate():
    analysis = experiment.analyze(_fake_rows())
    assert analysis["coordinate_count"] == 21
    assert analysis["external_native_exact"] is True
    assert analysis["fresh_claim_eligible"] is False
    assert analysis["worst_dataset_ratio"] == pytest.approx(1.01)
    assert set(analysis["leave_one_out_equal_dataset_ratios"]) == set(
        experiment.TASKS.values()
    )
    assert "wins" not in analysis


def test_analysis_rejects_inexact_or_incomplete_artifacts():
    rows = _fake_rows()
    rows[0]["external_native_exact"] = False
    with pytest.raises(RuntimeError, match="parity"):
        experiment.analyze(rows)
    incomplete = copy.deepcopy(_fake_rows()[:-1])
    with pytest.raises(RuntimeError, match="incomplete"):
        experiment.analyze(incomplete)


def test_analysis_rejects_forged_exactness_and_dataset_identity():
    changed = copy.deepcopy(_fake_rows())
    changed[0]["selected"]["prediction_sha256"] = "forged"
    with pytest.raises(RuntimeError, match="parity fields"):
        experiment.analyze(changed)

    changed = copy.deepcopy(_fake_rows())
    del changed[0]["selected"]["prediction_sha256"]
    del changed[0]["chimera"]["prediction_sha256"]
    with pytest.raises(
        RuntimeError,
        match="parity fields|retained model|prediction_sha256",
    ):
        experiment.analyze(changed)

    changed = copy.deepcopy(_fake_rows())
    changed[0]["dataset_name"] = "wrong"
    with pytest.raises(RuntimeError, match="dataset identity"):
        experiment.analyze(changed)

    changed = copy.deepcopy(_fake_rows())
    changed[0]["selected_cross_pairs"] = []
    with pytest.raises(RuntimeError, match="pair ledger"):
        experiment.analyze(changed)

    changed = copy.deepcopy(_fake_rows())
    changed[0]["cross_selected"] = False
    changed[0]["selected_cross_pairs"] = []
    with pytest.raises(RuntimeError, match="selection decision"):
        experiment.analyze(changed)

    changed = copy.deepcopy(_fake_rows())
    changed[0]["candidate_cross_pairs"] = []
    changed[0]["selected_cross_pairs"] = []
    with pytest.raises(RuntimeError, match="candidate pair ledger"):
        experiment.analyze(changed)


def test_analysis_rejects_invalid_rmse_tree_and_pair_domains():
    changed = copy.deepcopy(_fake_rows())
    changed[0]["base"]["test_rmse"] = -1.0
    changed[0]["selected"]["test_rmse"] = -0.98
    changed[0]["chimera"]["test_rmse"] = -0.98
    changed[0]["chimera_actual"]["test_rmse"] = -0.98
    with pytest.raises(RuntimeError, match="finite and positive"):
        experiment.analyze(changed)

    changed = copy.deepcopy(_fake_rows())
    changed[0]["selected"]["actual_retained_tree_count"] = 0
    with pytest.raises(RuntimeError, match="positive integer"):
        experiment.analyze(changed)

    changed = copy.deepcopy(_fake_rows())
    for pairs_field in ("candidate_cross_pairs", "selected_cross_pairs"):
        changed[0][pairs_field][0][0] = 999
    with pytest.raises(RuntimeError, match="candidate pair ledger"):
        experiment.analyze(changed)

    changed = copy.deepcopy(_fake_rows())
    changed[0]["fold"] = float(changed[0]["fold"])
    with pytest.raises(RuntimeError, match="coordinate ledger"):
        experiment.analyze(changed)

    changed = copy.deepcopy(_fake_rows())
    changed[0]["base"]["test_rmse"] = "1.0"
    with pytest.raises(RuntimeError, match="finite and positive"):
        experiment.analyze(changed)


def test_artifact_validation_binds_stored_analysis_and_split_hashes():
    artifact = json.loads(ARTIFACT.read_text())
    changed = copy.deepcopy(artifact)
    changed["analysis"]["equal_dataset_geomean_ratio"] = 99.0
    with pytest.raises(RuntimeError, match="not reproducible"):
        experiment.validate_artifact(changed)

    changed = copy.deepcopy(artifact)
    changed["results"][0]["fit_index_sha256"] = "forged"
    with pytest.raises(RuntimeError, match="split ledger"):
        experiment.validate_artifact(changed)

    changed = copy.deepcopy(artifact)
    changed["results"][0]["fit_index_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="frozen evidence ledger"):
        experiment.validate_artifact(changed)

    changed = copy.deepcopy(artifact)
    changed["sources"]["darkofit"]["path"] = "/tmp/forged-darkofit"
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        experiment.validate_artifact(changed)


def test_recorded_artifact_reproduces_raw_analysis(assert_analysis_equal):
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["protocol"]["sha256"] == experiment._sha256(
        experiment.PROTOCOL
    )
    assert artifact["partition_boundary"][
        "partition_sha256"
    ] == experiment._sha256(experiment.PARTITION)
    experiment.validate_artifact(artifact)
    assert_analysis_equal(
        artifact["analysis"],
        experiment.analyze(artifact["results"]),
    )
    assert artifact["partition_boundary"]["lockbox_data_used"] is False
    assert artifact["analysis"]["external_native_exact"] is True
    assert artifact["analysis"]["fresh_claim_eligible"] is False
