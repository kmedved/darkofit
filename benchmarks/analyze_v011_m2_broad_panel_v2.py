"""Verify and analyze the frozen 14-CPU v0.11 M2 successor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:  # Direct execution from a clean checkout.
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from benchmarks import analyze_v011_m2_broad_panel as _v1
from benchmarks import run_v011_m2_broad_panel_v2 as campaign


def main(argv: Sequence[str] | None = None) -> int:
    with campaign.configured_successor():
        return _v1.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
