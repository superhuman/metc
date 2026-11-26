"""Preflight historical producer metadata without executing it.

Fresh paper-system execution is implemented only by :mod:`metc_artifact.e2e`.
The command templates in the registry document provenance and runtime
requirements; they are not a second supported execution interface.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from metc_artifact.schema import dataset_spec

from .models import AssetKind, SystemSpec
from .registry import system_spec


@dataclass(frozen=True)
class PreflightIssue:
    category: str
    name: str
    detail: str


@dataclass(frozen=True)
class PreparedInvocation:
    system_id: str
    dataset_id: str
    argv: tuple[str, ...]
    working_directory: Path | None
    environment: Mapping[str, str] = field(repr=False, compare=False)


class SystemNotReadyError(RuntimeError):
    def __init__(self, system_id: str, dataset_id: str, issues: Sequence[PreflightIssue]):
        self.system_id = system_id
        self.dataset_id = dataset_id
        self.issues = tuple(issues)
        details = "; ".join(f"{issue.category}:{issue.name}: {issue.detail}" for issue in issues)
        super().__init__(f"{system_id}/{dataset_id} is not ready: {details}")


class UnsupportedSystemExecutionError(RuntimeError):
    """Raised when the metadata-only runner is asked to execute a system."""


class SystemRunner:
    """Resolve registry metadata for diagnostics; never launch a paper system."""

    def __init__(
        self,
        *,
        artifact_root: Path | str,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.environment = dict(os.environ if environment is None else environment)

    def _resolve_executable(
        self, spec: SystemSpec, environment: Mapping[str, str], issues: list[PreflightIssue]
    ) -> str:
        variable = spec.runtime.executable_environment_variable
        executable = environment.get(variable)
        if not executable and spec.runtime.default_executable:
            executable = str(self.artifact_root / spec.runtime.default_executable)
        if not executable:
            issues.append(
                PreflightIssue(
                    "environment",
                    variable,
                    f"set this to the executable for {spec.runtime.name}",
                )
            )
            return ""

        candidate = Path(executable).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            if not candidate.is_file():
                issues.append(PreflightIssue("executable", variable, f"not a file: {candidate}"))
            elif not os.access(candidate, os.X_OK):
                issues.append(PreflightIssue("executable", variable, f"not executable: {candidate}"))
            return str(candidate)

        if shutil.which(executable, path=environment.get("PATH")) is None:
            issues.append(PreflightIssue("executable", variable, f"not found on PATH: {executable}"))
        return executable

    def _resolve_assets(
        self,
        spec: SystemSpec,
        dataset_id: str,
        values: Mapping[str, str | Path],
        environment: Mapping[str, str],
        issues: list[PreflightIssue],
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for asset in spec.assets:
            if asset.dataset_ids and dataset_id not in asset.dataset_ids:
                continue
            raw = values.get(asset.name)
            if raw is None and asset.environment_variable:
                raw = environment.get(asset.environment_variable)
            if raw is None and asset.default_relative_path:
                raw = self.artifact_root / asset.default_relative_path
            if raw is None:
                hint = (
                    f"set {asset.environment_variable} or pass values[{asset.name!r}]"
                    if asset.environment_variable
                    else f"pass values[{asset.name!r}]"
                )
                issues.append(PreflightIssue("asset", asset.name, f"{asset.description}; {hint}"))
                continue

            path = Path(raw).expanduser()
            resolved[asset.name] = str(path)
            if asset.kind == AssetKind.FILE and not path.is_file():
                issues.append(PreflightIssue("asset", asset.name, f"file not found: {path}"))
            if asset.kind == AssetKind.DIRECTORY and not path.is_dir():
                issues.append(PreflightIssue("asset", asset.name, f"directory not found: {path}"))
        return resolved

    def prepare(
        self,
        system_id: str,
        dataset_id: str,
        *,
        values: Mapping[str, str | Path] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> PreparedInvocation:
        """Validate and render a historical invocation for provenance inspection."""
        spec = system_spec(system_id)
        dataset = dataset_spec(dataset_id)
        binding = spec.binding_for(dataset.dataset_id)
        supplied = dict(values or {})
        process_environment = dict(self.environment)
        if environment:
            process_environment.update({key: str(value) for key, value in environment.items()})
        process_environment.update(dict(spec.runtime.fixed_environment))
        issues: list[PreflightIssue] = []

        executable = self._resolve_executable(spec, process_environment, issues)
        for requirement in spec.runtime.required_environment:
            if not any(process_environment.get(name) for name in requirement.any_of):
                alternatives = ", ".join(requirement.any_of)
                issues.append(
                    PreflightIssue(
                        "environment",
                        requirement.name,
                        f"{requirement.description}; set one of: {alternatives}",
                    )
                )

        assets = self._resolve_assets(
            spec, dataset.dataset_id, supplied, process_environment, issues
        )
        for name in spec.command.required_values:
            if name not in supplied:
                issues.append(PreflightIssue("value", name, "required command value is missing"))
        for name in spec.command.input_files:
            raw = supplied.get(name)
            if raw is not None and not Path(raw).expanduser().is_file():
                issues.append(PreflightIssue("input", name, f"file not found: {raw}"))

        context: dict[str, str] = {
            "executable": executable,
            "dataset_id": dataset.dataset_id,
            "experiment_code": dataset.experiment_code,
            "language": dataset.language,
            **dict(binding.command_values),
            **assets,
            **{key: str(value) for key, value in supplied.items()},
        }

        argv: tuple[str, ...] = ()
        working_directory: Path | None = None
        if not issues:
            try:
                argv = tuple(token.format_map(context) for token in spec.command.argv)
                if spec.command.working_directory:
                    working_directory = Path(
                        spec.command.working_directory.format_map(context)
                    ).expanduser()
            except KeyError as error:
                issues.append(
                    PreflightIssue("template", str(error.args[0]), "command placeholder is unresolved")
                )

        if working_directory is not None and not working_directory.is_dir():
            issues.append(
                PreflightIssue(
                    "working_directory",
                    str(working_directory),
                    "directory does not exist",
                )
            )

        if issues:
            raise SystemNotReadyError(spec.system_id, dataset.dataset_id, issues)
        return PreparedInvocation(
            system_id=spec.system_id,
            dataset_id=dataset.dataset_id,
            argv=argv,
            working_directory=working_directory,
            environment=process_environment,
        )

    def run(
        self,
        system_id: str,
        dataset_id: str,
        *,
        values: Mapping[str, str | Path] | None = None,
        environment: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> None:
        """Refuse execution through the legacy metadata interface.

        Keeping this explicit failure is safer than silently launching a
        historical command that bypasses hydration, canonical materialization,
        and fresh-versus-frozen comparison.
        """
        del values, environment, timeout, check
        raise UnsupportedSystemExecutionError(
            f"{system_id}/{dataset_id} cannot be run through the metadata registry; "
            "use reproduce.py or reproduce_e2e.py for canonical E2E execution"
        )
