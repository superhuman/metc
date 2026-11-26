"""Explicit adapter failures for unavailable runtimes and invalid outputs."""

from __future__ import annotations


class AdapterError(RuntimeError):
    """Base class for public system-adapter failures."""


class AdapterUnavailableError(AdapterError):
    """The declared external runtime or an exact required asset is absent."""


class AdapterInputError(AdapterError):
    """Canonical pair input violates the adapter contract."""


class AdapterOutputError(AdapterError):
    """An external producer emitted malformed or misaligned output."""


class AdapterProcessError(AdapterError):
    """An isolated producer subprocess exited unsuccessfully."""

    def __init__(self, argv: tuple[str, ...], returncode: int, stderr: str):
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"external producer exited with {returncode}: {stderr.strip()}"
        )
