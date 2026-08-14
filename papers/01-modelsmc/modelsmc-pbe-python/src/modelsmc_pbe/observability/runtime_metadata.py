"""Reproducibility metadata captured once when a run starts."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import torch

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def slug(value: str) -> str:
    """Convert a human run name into a bounded directory-name component."""

    cleaned = _SAFE_NAME.sub("-", value.strip()).strip("-._").lower()
    return cleaned[:64] or "run"


def git_state(cwd: Path) -> dict[str, Any]:
    """Read the revision, branch, and dirty flag without failing the run."""

    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    revision = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "revision": revision,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def runtime_snapshot() -> dict[str, Any]:
    """Capture platform and dependency versions used by an experiment."""

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    for distribution in ("modelsmc-pbe", "numpy", "pydantic", "rich"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": versions,
    }
