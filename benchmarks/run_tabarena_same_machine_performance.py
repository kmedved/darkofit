"""Run the conditional same-machine DarkoFit/ChimeraBoost performance panel.

This driver is intentionally separate from the quality-confirmation runner.
Run it only if the frozen remaining-nine candidate passes every quality gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.preprocessing_instrumentation import (  # noqa: E402
    instrument_feature_preprocessors,
)
from benchmarks.run_tabarena_regression_remaining9 import (
    FROZEN_CANDIDATE,
    TASK_IDS,
    TASK_SPLIT_COUNTS,
)  # noqa: E402


SPLIT_INDICES = ["r0f0", "r1f1", "r2f2"]
REGISTERED_FOLDS = [0, 4, 8]
TIME_LIMIT_SECONDS = 3_600
EXPECTED_REGISTERED_ROWS = len(TASK_IDS) * len(SPLIT_INDICES)
EXPECTED_JOBS = 3 * EXPECTED_REGISTERED_ROWS
FROZEN_CHIMERA_VERSION = "0.14.1"
FROZEN_CHIMERA_COMMIT = "07995af9e2b6212a41975a49931ee20af8f2cc14"


def resolve_chimera_repo(
    explicit: Path = None,
    *,
    darkofit_repo: Path = None,
) -> Path:
    """Resolve an explicit checkout or discover the sibling repository."""
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
    else:
        root = (
            darkofit_repo.resolve()
            if darkofit_repo is not None
            else Path(__file__).resolve().parents[1]
        )
        candidate = root.parent / "chimeraboost"
    if not (candidate / "chimeraboost" / "__init__.py").is_file():
        raise RuntimeError(
            f"{candidate} is not a ChimeraBoost source checkout; pass --chimera-repo"
        )
    if not (candidate / ".git").exists():
        raise RuntimeError(f"{candidate} is not a Git checkout")
    return candidate


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def chimera_source_provenance(repo: Path) -> dict:
    """Require the frozen clean ChimeraBoost source and return its identity."""
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != FROZEN_CHIMERA_COMMIT:
        raise RuntimeError(
            f"ChimeraBoost checkout is {commit}, expected {FROZEN_CHIMERA_COMMIT}"
        )
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise RuntimeError("ChimeraBoost checkout must be clean for this comparison")
    return {
        "repository": str(repo),
        "commit": commit,
        "version_expected": FROZEN_CHIMERA_VERSION,
        "dirty": False,
    }


def darkofit_source_provenance(repo: Path) -> dict:
    """Require a clean DarkoFit checkout and return its source identity."""
    commit = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise RuntimeError("DarkoFit checkout must be clean for this comparison")
    import darkofit

    return {
        "darkofit_repository": str(repo),
        "darkofit_commit": commit,
        "darkofit_dirty": False,
        "darkofit_package": "darkofit",
        "darkofit_version_imported": darkofit.__version__,
        "darkofit_module_file": str(Path(darkofit.__file__).resolve()),
    }


def runtime_provenance() -> dict:
    """Return stable machine/runtime fields needed to interpret timings."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def import_local_chimeraboost(repo: Path):
    """Import ChimeraBoost from the validated checkout, never site-packages."""
    loaded = sys.modules.get("chimeraboost")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", ""))
        if not loaded_file or not _is_within(loaded_file, repo):
            raise RuntimeError(
                f"chimeraboost was already imported from {loaded_file}; "
                f"restart and use the checkout at {repo}"
            )
    repo_string = str(repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    importlib.invalidate_caches()
    package = importlib.import_module("chimeraboost")
    package_file = Path(package.__file__).resolve()
    if not _is_within(package_file, repo):
        raise RuntimeError(
            f"import resolved to {package_file}, outside requested checkout {repo}"
        )
    version = getattr(package, "__version__", None)
    if version != FROZEN_CHIMERA_VERSION:
        raise RuntimeError(
            f"imported chimeraboost {version!r}, expected {FROZEN_CHIMERA_VERSION}"
        )
    return package


def validate_registered_splits(
    results,
    *,
    task_split_counts: Mapping[str, tuple[int, int]] = TASK_SPLIT_COUNTS,
    registered_folds: Sequence[int] = REGISTERED_FOLDS,
) -> None:
    """Require the exact 27 non-imputed registered comparison coordinates."""
    records = (
        results.to_dict("records")
        if hasattr(results, "to_dict")
        else list(results)
    )
    selected = [
        row
        for row in records
        if row.get("dataset") in task_split_counts
        and row.get("method") == "CHIMERA (default)"
        and int(row.get("fold", -1)) in registered_folds
    ]
    expected = len(task_split_counts) * len(registered_folds)
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} registered performance rows, got {len(selected)}"
        )
    seen = set()
    for row in selected:
        if bool(row.get("imputed")):
            raise RuntimeError("registered performance coverage contains imputed rows")
        key = (row["dataset"], int(row["fold"]))
        if row.get("problem_type") != "regression" or row.get("metric") != "rmse":
            raise RuntimeError(
                f"registered performance row {key} is not regression/rmse"
            )
        if key in seen:
            raise RuntimeError(f"duplicate registered performance row for {key}")
        seen.add(key)
    expected_keys = {
        (dataset, fold)
        for dataset in task_split_counts
        for fold in registered_folds
    }
    missing = sorted(expected_keys - seen)
    if missing:
        raise RuntimeError(f"missing registered performance rows: {missing}")


def _experiments(model_cls, manual_config: dict, suffix: str):
    from tabarena.utils.config_utils import ConfigGenerator

    generator = ConfigGenerator(
        model_cls=model_cls,
        manual_configs=[dict(manual_config)],
        search_space={},
    )
    return generator.generate_all_bag_experiments(
        num_random_configs=0,
        name_id_suffix=suffix,
        add_seed="fold-wise",
        fold_fitting_strategy="sequential_local",
        time_limit=TIME_LIMIT_SECONDS,
    )


def build_experiments() -> list:
    """Build the three immutable product configurations."""
    from benchmarks.same_machine_performance_adapters import (
        SameMachineChimeraBoostModel,
        SameMachineDarkoFitModel,
    )

    experiments = []
    experiments.extend(
        _experiments(
            SameMachineDarkoFitModel,
            {},
            "_same_machine_darkofit_default",
        )
    )
    experiments.extend(
        _experiments(
            SameMachineDarkoFitModel,
            FROZEN_CANDIDATE,
            "_same_machine_darkofit_candidate",
        )
    )
    experiments.extend(
        _experiments(
            SameMachineChimeraBoostModel,
            {},
            "_same_machine_chimera_default",
        )
    )
    if len(experiments) != 3:
        raise RuntimeError(f"expected 3 experiments, built {len(experiments)}")
    return experiments


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/tabarena-same-machine-performance-20260712"),
    )
    parser.add_argument("--chimera-repo", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    chimera_repo = resolve_chimera_repo(args.chimera_repo)
    provenance = chimera_source_provenance(chimera_repo)
    provenance.update(darkofit_source_provenance(REPO_ROOT))
    provenance["runtime"] = runtime_provenance()
    chimeraboost = import_local_chimeraboost(chimera_repo)
    provenance.update(
        {
            "package": "chimeraboost",
            "version_imported": chimeraboost.__version__,
            "module_file": str(Path(chimeraboost.__file__).resolve()),
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "split_indices": list(SPLIT_INDICES),
            "candidate": dict(FROZEN_CANDIDATE),
        }
    )
    os.environ["DARKOFIT_BENCH_CHIMERA_VERSION"] = chimeraboost.__version__
    os.environ["DARKOFIT_BENCH_CHIMERA_COMMIT"] = provenance["commit"]

    from tabarena.contexts import TabArenaContext

    context = TabArenaContext()
    registered = context.load_results(methods=["ChimeraBoost"])
    validate_registered_splits(registered)
    jobs = context.build_jobs(
        build_experiments(),
        task_ids=TASK_IDS,
        split_indices=SPLIT_INDICES,
    )
    if len(jobs) != EXPECTED_JOBS:
        raise RuntimeError(f"expected {EXPECTED_JOBS} jobs, built {len(jobs)}")

    print(
        f"validated {EXPECTED_REGISTERED_ROWS} registered rows; built {len(jobs)} "
        f"jobs; ChimeraBoost {chimeraboost.__version__} {provenance['commit'][:7]}"
    )
    if args.dry_run:
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_started = time.perf_counter()
    chimeraboost.warmup()
    provenance["chimeraboost_warmup_seconds"] = time.perf_counter() - warmup_started
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    from chimeraboost.preprocessing import (
        FeaturePreprocessor as ChimeraFeaturePreprocessor,
    )
    from darkofit.preprocessing import FeaturePreprocessor as DarkoFeaturePreprocessor

    with instrument_feature_preprocessors(
        {
            "darkofit": DarkoFeaturePreprocessor,
            "chimeraboost": ChimeraFeaturePreprocessor,
        }
    ):
        results = context.run_jobs(
            jobs,
            expname=str(output_dir / "experiments"),
            new_result_prefix="[Same-machine regression performance] ",
            debug_mode=True,
        )
    print(f"SAME_MACHINE_PERFORMANCE_COMPLETE {len(results)} {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
