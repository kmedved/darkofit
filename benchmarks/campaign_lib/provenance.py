"""Behavior-exact provenance primitives for prospective campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Hash the bytes captured by one complete file read."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash the compact canonical JSON form used by Panel 3 records."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def is_sha256(value: Any) -> bool:
    """Return whether *value* is a lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build an object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_float(value: str) -> float:
    """Parse a finite JSON floating-point number."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number: {value}")
    return result


def strict_json_int(value: str) -> int:
    """Parse a signed 64-bit JSON integer."""
    result = int(value)
    if not -(2**63) <= result <= 2**63 - 1:
        raise ValueError(f"out-of-range JSON integer: {value}")
    return result


def strict_json_loads(encoded: str) -> Any:
    """Decode JSON with the exact strict Panel 3 numeric grammar."""
    return json.loads(
        encoded,
        object_pairs_hook=strict_json_object,
        parse_float=strict_json_float,
        parse_int=strict_json_int,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def git_output(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> str:
    """Return stripped text output from a Git command."""
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=check,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_is_ancestor(
    repository: Path,
    ancestor: str,
    descendant: str,
) -> bool:
    """Check Git ancestry without exposing command output."""
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repository,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def require_single_create_only_history(
    git: Callable[..., str],
    source_head: str,
    execution_head: str,
    expected_path: str,
    *,
    error_message: str,
    describe_observed_paths: bool = False,
) -> None:
    """Require one committed artifact addition and no intermediate changes."""
    final_change = git(
        "diff",
        "--name-status",
        f"{source_head}..{execution_head}",
    ).splitlines()
    commits = filter(
        None,
        git(
            "rev-list",
            f"{source_head}..{execution_head}",
        ).splitlines(),
    )
    touched_by_commit = [
        {
            value
            for value in git(
                "diff-tree",
                "--root",
                "--no-commit-id",
                "--name-only",
                "-r",
                "-m",
                commit,
            ).splitlines()
            if value
        }
        for commit in commits
    ]
    nonempty = [paths for paths in touched_by_commit if paths]
    if (
        final_change == [f"A\t{expected_path}"]
        and len(nonempty) == 1
        and nonempty[0] == {expected_path}
    ):
        return

    if describe_observed_paths:
        observed_paths = {
            value
            for paths in touched_by_commit
            for value in paths
        }
        observed_paths.update(
            value
            for change in final_change
            for value in change.split("\t")[1:]
        )
        error_message = f"{error_message}: {sorted(observed_paths)}"
    raise RuntimeError(error_message)


def artifact_path(path: Path, *, root: Path) -> str:
    """Render in-repository paths relatively and external paths absolutely."""
    absolute = path.expanduser().absolute()
    try:
        return str(absolute.relative_to(root))
    except ValueError:
        return str(absolute)
