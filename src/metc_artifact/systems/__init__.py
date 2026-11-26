"""Pinned annotation-system provenance and preflight metadata."""

from .models import (
    AssetKind,
    AssetRequirement,
    CommandTemplate,
    DatasetBinding,
    EnvironmentRequirement,
    Provenance,
    RuntimeEnvironment,
    SystemRole,
    SystemSpec,
)
from .registry import SYSTEMS, system_spec, systems_for_dataset
from .runner import (
    PreflightIssue,
    PreparedInvocation,
    SystemNotReadyError,
    SystemRunner,
    UnsupportedSystemExecutionError,
)

__all__ = [
    "SYSTEMS",
    "AssetKind",
    "AssetRequirement",
    "CommandTemplate",
    "DatasetBinding",
    "EnvironmentRequirement",
    "PreflightIssue",
    "PreparedInvocation",
    "Provenance",
    "RuntimeEnvironment",
    "SystemNotReadyError",
    "SystemRole",
    "SystemRunner",
    "SystemSpec",
    "UnsupportedSystemExecutionError",
    "system_spec",
    "systems_for_dataset",
]
