"""Small shell-free subprocess boundary shared by all external adapters."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .errors import AdapterProcessError, AdapterUnavailableError


def require_file(path: Path | str | None, description: str) -> Path:
    if path is None:
        raise AdapterUnavailableError(f"missing {description}")
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise AdapterUnavailableError(f"{description} is not a file: {resolved}")
    return resolved


def require_directory(path: Path | str | None, description: str) -> Path:
    if path is None:
        raise AdapterUnavailableError(f"missing {description}")
    resolved = Path(path).expanduser()
    if not resolved.is_dir():
        raise AdapterUnavailableError(f"{description} is not a directory: {resolved}")
    return resolved


def require_executable(value: str | Path | None, description: str) -> str:
    if value is None or not str(value):
        raise AdapterUnavailableError(f"missing {description}")
    executable = str(Path(value).expanduser())
    candidate = Path(executable)
    if candidate.is_absolute() or candidate.parent != Path("."):
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise AdapterUnavailableError(f"{description} is not executable: {candidate}")
        return str(candidate)
    if shutil.which(executable) is None:
        raise AdapterUnavailableError(f"{description} is not on PATH: {executable}")
    return executable


def run_checked(
    argv: Sequence[str | Path],
    *,
    cwd: Path | str | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    command = tuple(str(value) for value in argv)
    process_environment = dict(os.environ)
    if environment:
        process_environment.update(environment)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )
    if completed.returncode:
        raise AdapterProcessError(command, completed.returncode, completed.stderr)
    return completed
