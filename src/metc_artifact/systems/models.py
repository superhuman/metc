"""Pure data contracts for independently executed annotation systems.

Nothing in this package imports a baseline, model, or corpus library.  Every
producer is described as an external command so incompatible Python and native
runtime stacks never share an interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SystemRole(StrEnum):
    ANNOTATOR = "annotator"
    TEACHER = "teacher"
    REFERENCE_CONSTRUCTOR = "reference_constructor"


class AssetKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"


@dataclass(frozen=True)
class EnvironmentRequirement:
    """One environment requirement, optionally satisfied by alternatives."""

    name: str
    any_of: tuple[str, ...]
    description: str
    secret: bool = False


@dataclass(frozen=True)
class RuntimeEnvironment:
    """The external runtime in which one producer must execute."""

    name: str
    executable_environment_variable: str
    python_version: str | None
    dependencies: tuple[str, ...] = ()
    default_executable: str | None = None
    required_environment: tuple[EnvironmentRequirement, ...] = ()
    fixed_environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AssetRequirement:
    """A file or directory required before a producer can run."""

    name: str
    kind: AssetKind
    description: str
    environment_variable: str | None = None
    default_relative_path: str | None = None
    sha256: str | None = None
    dataset_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandTemplate:
    """A shell-free argv template.

    Tokens use ``str.format`` names.  The runner always supplies
    ``executable``, ``dataset_id``, ``experiment_code``, ``language``, and all
    resolved asset names.  Callers supply the names in ``required_values``.
    """

    argv: tuple[str, ...]
    required_values: tuple[str, ...] = ()
    input_files: tuple[str, ...] = ()
    working_directory: str | None = None


@dataclass(frozen=True)
class DatasetBinding:
    dataset_id: str
    command_values: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Provenance:
    repository: str | None
    revision: str | None
    package_version: str | None = None
    license: str | None = None
    local_modifications: tuple[str, ...] = ()
    evaluated_configuration: tuple[tuple[str, str], ...] = ()
    historical_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    annotation_key: str
    display_name: str
    role: SystemRole
    runtime: RuntimeEnvironment
    command: CommandTemplate
    bindings: tuple[DatasetBinding, ...]
    assets: tuple[AssetRequirement, ...]
    provenance: Provenance
    availability_note: str

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(binding.dataset_id for binding in self.bindings)

    def binding_for(self, dataset_id: str) -> DatasetBinding:
        for binding in self.bindings:
            if binding.dataset_id == dataset_id:
                return binding
        raise ValueError(f"{self.system_id!r} does not apply to {dataset_id!r}")
