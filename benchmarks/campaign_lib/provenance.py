"""Behavior-exact provenance primitives for prospective campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
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


def artifact_path(path: Path, *, root: Path) -> str:
    """Render in-repository paths relatively and external paths absolutely."""
    absolute = path.expanduser().absolute()
    try:
        return str(absolute.relative_to(root))
    except ValueError:
        return str(absolute)
