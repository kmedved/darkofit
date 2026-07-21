#!/usr/bin/env python3
"""Analyze attempt-2 M3b evidence with self-worker RSS provenance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    from . import run_m3b_ensemble_v3_r2 as runner
except ImportError:  # direct script execution
    import run_m3b_ensemble_v3_r2 as runner


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
BASE_ANALYZER_PATH = BENCH_DIR / "analyze_m3b_ensemble_v3.py"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load_isolated_base():
    name = "_darkofit_m3b_r2_base_analyzer"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, BASE_ANALYZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the bound attempt-1 M3b analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.runner = runner
    return module


_base = _load_isolated_base()
_base_validate_artifact = _base.validate_artifact
_base_build_gate = _base.build_gate
_base_build_final_result = _base.build_final_result
_base_render_note = _base.render_note


def validate_artifact(*args, **kwargs):
    artifact = _base_validate_artifact(*args, **kwargs)
    if artifact.get("rss_scope") != runner.RSS_SCOPE or any(
        row.get("rss_scope") != runner.RSS_SCOPE for row in artifact["rows"]
    ):
        raise RuntimeError("M3b attempt-2 RSS scope provenance is invalid")
    return artifact


def build_gate(*args, **kwargs):
    gate = _base_build_gate(*args, **kwargs)
    gate["rss_scope"] = runner.RSS_SCOPE
    return gate


def build_final_result(*args, **kwargs):
    gate_path = args[1] if len(args) > 1 else kwargs["gate_path"]
    gate = _base._load_json(Path(gate_path))
    if gate.get("rss_scope") != runner.RSS_SCOPE:
        raise RuntimeError("M3b attempt-2 gate has the wrong RSS scope")
    result = _base_build_final_result(*args, **kwargs)
    result["rss_scope"] = runner.RSS_SCOPE
    return result


def render_note(result):
    note = _base_render_note(result)
    marker = "This is spent private development evidence."
    return note.replace(
        marker,
        marker + " Peak RSS uses the attempt-2 self-worker-process scope; no child "
        "model workers exist in this sequential campaign.",
    )


_base.validate_artifact = validate_artifact
_base.build_gate = build_gate
_base.build_final_result = build_final_result
_base.render_note = render_note


def __getattr__(name: str):
    return getattr(_base, name)


def main() -> int:
    return _base.main()


if __name__ == "__main__":
    raise SystemExit(main())
