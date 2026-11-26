"""One-command, execution-first verification of the METC evaluation artifact.

This module deliberately does not consume the legacy sufficient-statistics
artifact.  It reconstructs public references, executes configured annotation
producers in subprocesses, materializes their fresh annotations, compares them
with the complete Data artifact, and computes reference matching from the
freshly assembled rows.

External LLM and exact CUDA environments are not silently skipped as passes.
When an applicable runtime has not been configured, its stage is reported as
``KNOWN_DIFFERENCE`` and the overall run cannot be ``PASS``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .adapters.contract import (
    materialize_paper_rows,
    pairs_from_paper_rows,
    read_annotations,
    write_pairs,
)
from .data_artifact import validate_data_artifact
from .judges.batch import (
    build_acceptability_jobs,
    build_teacher_jobs,
    compare_acceptability_with_frozen,
    compare_teacher_with_frozen,
    materialize_teacher_annotations,
    run_acceptability_batch,
    run_teacher_batch,
)
from .judges.protocols import load_teacher_taxonomy
from .judges.records import FreshRunWriter, read_fresh_records
from .metrics import (
    compute_human_tables,
    compute_paper_tables,
    load_human_study,
    load_judge_rows,
)
from .reference_matching import score_reference_matching
from .schema import DATASETS, DatasetSpec
from .semantic import canonical_item_id, compare_annotations, semantic_edit


SCHEMA_VERSION = "metc-e2e-v1"
EXECUTABLE_SYSTEMS = ("metc", "metc_teacher", "errant", "ger", "jp")
SYSTEM_ALIASES = {"gerrant": "ger", "jp_errant": "jp"}
SYSTEM_CLI_CHOICES = ("all", *EXECUTABLE_SYSTEMS, *SYSTEM_ALIASES)
METC_INFERENCE_PROFILES = ("paper", "portable")
PORTABLE_METC_DIFFERENCE = (
    "portable fp32 inference may differ from the evaluated CUDA/bfloat16 paper profile"
)
_SECRET_ENVIRONMENT_MARKERS = (
    "API_KEY",
    "AUTHORIZATION",
    "COOKIE",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


def normalize_systems(systems: Sequence[str]) -> tuple[str, ...]:
    """Resolve user-facing aliases and the teacher's METC dependency."""

    if "all" in systems:
        return EXECUTABLE_SYSTEMS
    normalized: list[str] = []
    for requested in systems:
        system = SYSTEM_ALIASES.get(requested, requested)
        if system not in EXECUTABLE_SYSTEMS:
            raise ValueError(f"unknown annotation system: {requested}")
        if system not in normalized:
            normalized.append(system)
    if "metc_teacher" in normalized and "metc" not in normalized:
        normalized.insert(normalized.index("metc_teacher"), "metc")
    return tuple(normalized)


def _restore_authoritative_row_order(
    authoritative: Sequence[Mapping[str, Any]],
    reconstructed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return reconstructed rows in the released Data artifact's item order.

    Bootstrap draws are deterministic only for a fixed row order. Public
    corpus reconstruction may emit the same item universe in a different
    order, so scoring that order would change seed-pinned confidence bounds
    even after semantic equality passes.
    """

    authoritative_ids = [canonical_item_id(row) for row in authoritative]
    reconstructed_ids = [canonical_item_id(row) for row in reconstructed]
    if len(set(authoritative_ids)) != len(authoritative_ids):
        raise ValueError("authoritative Data artifact contains duplicate item IDs")
    if len(set(reconstructed_ids)) != len(reconstructed_ids):
        raise ValueError("public reconstruction contains duplicate item IDs")
    if set(authoritative_ids) != set(reconstructed_ids):
        raise ValueError("public reconstruction item IDs differ from the Data artifact")
    by_id = dict(zip(reconstructed_ids, reconstructed, strict=True))
    return [dict(by_id[item_id]) for item_id in authoritative_ids]


class StageStatus(StrEnum):
    PASS = "PASS"
    KNOWN_DIFFERENCE = "KNOWN_DIFFERENCE"
    FAIL = "FAIL"


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: StageStatus
    summary: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "summary": self.summary,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandExecutor(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout: float,
    ) -> CommandResult: ...


def subprocess_executor(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
) -> CommandResult:
    """Execute one producer without a shell or shared Python imports."""

    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return CommandResult(127, "", str(error))
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class RuntimePaths:
    """Explicit paths to independently hydrated producer environments."""

    core_python: str = sys.executable
    cherrant_python: str | None = None
    metc_python: str | None = None
    metc_checkpoint: Path | None = None
    errant_python: str | None = None
    gerrant_python: str | None = None
    gerrant_root: Path | None = None
    jp_python: str | None = None
    jp_transform_manifest: Path | None = None
    jp_stanza_resources: Path | None = None
    jp_zh_gsdltp_dir: Path | None = None


@dataclass(frozen=True)
class E2EConfig:
    data_root: Path
    work_root: Path
    cache_dir: Path
    software_root: Path
    runtimes: RuntimePaths
    systems: tuple[str, ...] = EXECUTABLE_SYSTEMS
    known_differences: Mapping[tuple[str, str], str] = field(default_factory=dict)
    metc_inference_profile: str = "paper"
    jp_workers: int = 2
    reference_samples: int = 1000
    human_samples: int = 10_000
    seed: int = 42
    max_differences: int = 20
    timeout_seconds: float = 24 * 60 * 60
    run_llm: bool = False
    llm_run_id: str = "e2e"
    llm_workers: int = 4
    llm_max_attempts: int = 4
    require_full_scope: bool = True
    no_network: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "systems", normalize_systems(self.systems))
        if self.metc_inference_profile not in METC_INFERENCE_PROFILES:
            raise ValueError(
                "metc_inference_profile must be one of "
                f"{', '.join(METC_INFERENCE_PROFILES)}"
            )
        if self.run_llm and self.metc_inference_profile != "paper":
            raise ValueError(
                "fresh teacher and judge reruns require the exact METC paper profile"
            )
        if self.jp_workers < 1:
            raise ValueError("jp_workers must be positive")
        if self.reference_samples < 1:
            raise ValueError("reference_samples must be positive")
        if self.human_samples < 1:
            raise ValueError("human_samples must be positive")
        if self.max_differences < 0:
            raise ValueError("max_differences must be nonnegative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.llm_workers < 1 or self.llm_max_attempts < 1:
            raise ValueError("LLM workers and maximum attempts must be positive")


@dataclass(frozen=True)
class AdapterPlan:
    system: str
    dataset_id: str
    command: tuple[str, ...] | None
    output_path: Path
    raw_output_path: Path | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.command is None) == (self.unavailable_reason is None):
            raise ValueError("adapter plan must contain exactly one command or unavailable reason")


@dataclass(frozen=True)
class E2EReport:
    stages: tuple[StageResult, ...]
    artifacts: Mapping[str, str]

    @property
    def status(self) -> StageStatus:
        statuses = {stage.status for stage in self.stages}
        if StageStatus.FAIL in statuses:
            return StageStatus.FAIL
        if StageStatus.KNOWN_DIFFERENCE in statuses:
            return StageStatus.KNOWN_DIFFERENCE
        return StageStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        counts = {status.value: 0 for status in StageStatus}
        for stage in self.stages:
            counts[stage.status.value] += 1
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status.value,
            "stage_counts": counts,
            "stages": [stage.to_dict() for stage in self.stages],
            "artifacts": dict(sorted(self.artifacts.items())),
        }


AdapterPlanner = Callable[[str, DatasetSpec, Path, Path, E2EConfig], AdapterPlan]
DataValidator = Callable[[Path], Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("reference matching produced no system rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonicalize_system_order(
    expected_rows: Sequence[Mapping[str, Any]],
    actual_rows: Sequence[Mapping[str, Any]],
    spec: DatasetSpec,
    system: str,
) -> list[dict[str, Any]]:
    """Align equal fresh edits to the authoritative occurrence order used by stable IDs."""

    expected = {canonical_item_id(row): row for row in expected_rows}
    actual = {canonical_item_id(row): row for row in actual_rows}
    if set(expected) != set(actual):
        raise ValueError("cannot canonicalize edit order across different item universes")
    output: list[dict[str, Any]] = []
    for raw_row in actual_rows:
        item_id = canonical_item_id(raw_row)
        expected_edits = expected[item_id].get(system)
        actual_edits = raw_row.get(system)
        if not isinstance(expected_edits, list) or not isinstance(actual_edits, list):
            raise ValueError(f"{spec.dataset_id}/{item_id}: missing {system} edit list")
        remaining = list(actual_edits)
        ordered: list[Any] = []
        for expected_edit in expected_edits:
            identity = semantic_edit(expected[item_id], expected_edit, spec)
            match = next(
                (
                    index
                    for index, candidate in enumerate(remaining)
                    if semantic_edit(raw_row, candidate, spec) == identity
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    f"{spec.dataset_id}/{item_id}: fresh {system} edits cannot be "
                    "aligned to authoritative stable-ID order"
                )
            ordered.append(remaining.pop(match))
        if remaining:
            raise ValueError(
                f"{spec.dataset_id}/{item_id}: fresh {system} has extra edits after ordering"
            )
        row = copy.deepcopy(dict(raw_row))
        row[system] = ordered
        output.append(row)
    return output


_JUDGE_JOB_IDENTITY_FIELDS = (
    "row_idx",
    "system",
    "edit_idx",
    "src",
    "tgt",
    "edit_start",
    "edit_end",
    "edit_diff",
)


def _validate_acceptability_job_identity(
    jobs: Sequence[Mapping[str, Any]],
    frozen_path: Path,
) -> None:
    """Bind every positional fresh judge ID to the same frozen edit occurrence."""

    frozen_rows = _read_jsonl(frozen_path)
    fresh = {str(job.get("edit_id")): job for job in jobs}
    frozen = {str(row.get("edit_id")): row for row in frozen_rows}
    if len(fresh) != len(jobs) or len(frozen) != len(frozen_rows):
        raise ValueError("fresh or frozen judge jobs contain duplicate edit IDs")
    if set(fresh) != set(frozen):
        missing = sorted(set(frozen) - set(fresh))
        extra = sorted(set(fresh) - set(frozen))
        raise ValueError(
            "fresh and frozen judge job universes differ: "
            f"missing={missing[:3]!r}, extra={extra[:3]!r}"
        )
    for edit_id in sorted(frozen):
        for identity_field in _JUDGE_JOB_IDENTITY_FIELDS:
            if fresh[edit_id].get(identity_field) != frozen[edit_id].get(
                identity_field
            ):
                raise ValueError(
                    f"{edit_id}: fresh judge identity differs from frozen {identity_field}"
                )


def _jsonable_validation(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
    elif isinstance(value, Mapping):
        converted = dict(value)
    else:
        converted = asdict(value)
    if not isinstance(converted, Mapping):
        raise TypeError("Data validator did not return a mapping-like report")
    return converted


def _executable_available(value: str | None) -> bool:
    if not value:
        return False
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.is_file() and os.access(candidate, os.X_OK)
    return shutil.which(value) is not None


def _existing_directory(value: Path | None) -> bool:
    return value is not None and value.expanduser().is_dir()


def _existing_file(value: Path | None) -> bool:
    return value is not None and value.expanduser().is_file()


def default_adapter_plan(
    system: str,
    spec: DatasetSpec,
    pairs_path: Path,
    output_root: Path,
    config: E2EConfig,
) -> AdapterPlan:
    """Build the exact adapter command for one applicable paper system."""

    output_path = output_root / spec.dataset_id / f"{system}.jsonl"
    raw_path: Path | None = None
    core = config.runtimes.core_python

    def unavailable(reason: str) -> AdapterPlan:
        return AdapterPlan(system, spec.dataset_id, None, output_path, unavailable_reason=reason)

    if system == "metc_teacher":
        return unavailable(
            "teacher execution is coordinated by the E2E runner after fresh METC spans "
            "are available, rather than through a standalone adapter command"
        )

    if system == "metc":
        if not _executable_available(config.runtimes.metc_python):
            return unavailable("METC inference Python is not configured")
        if not _existing_directory(config.runtimes.metc_checkpoint):
            return unavailable("the hydrated METC checkpoint directory is not configured")
        command = [
            core,
            "-m",
            "metc_artifact.adapters.metc",
            "--dataset",
            spec.dataset_id,
            "--pairs",
            str(pairs_path),
            "--output",
            str(output_path),
            "--python",
            str(config.runtimes.metc_python),
            "--checkpoint",
            str(config.runtimes.metc_checkpoint),
            "--device",
            "cuda" if config.metc_inference_profile == "paper" else "auto",
            "--inference-profile",
            config.metc_inference_profile,
        ]
        if config.metc_inference_profile == "portable":
            command.append("--allow-nonpaper")
        return AdapterPlan(system, spec.dataset_id, tuple(command), output_path)

    if system == "errant":
        if not _executable_available(config.runtimes.errant_python):
            return unavailable("the isolated ERRANT Python is not configured")
        command = (
            core,
            "-m",
            "metc_artifact.adapters.errant",
            "--dataset",
            spec.dataset_id,
            "--pairs",
            str(pairs_path),
            "--output",
            str(output_path),
            "--python",
            str(config.runtimes.errant_python),
        )
        return AdapterPlan(system, spec.dataset_id, command, output_path)

    if system == "ger":
        if not _executable_available(config.runtimes.gerrant_python):
            return unavailable("the isolated gERRANT Python is not configured")
        if not _existing_directory(config.runtimes.gerrant_root):
            return unavailable("the hydrated bundled gERRANT root is not configured")
        assert config.runtimes.gerrant_root is not None
        missing_resources = [
            name
            for name in ("de-stts_map", "de_DE-large.txt")
            if not (config.runtimes.gerrant_root / "resources" / name).is_file()
        ]
        if missing_resources:
            return unavailable(
                "gERRANT resources have not been hydrated: "
                + ", ".join(missing_resources)
            )
        raw_path = output_root / spec.dataset_id / "ger.raw.jsonl"
        command = (
            core,
            "-m",
            "metc_artifact.adapters.gerrant",
            "--dataset",
            spec.dataset_id,
            "--pairs",
            str(pairs_path),
            "--output",
            str(output_path),
            "--raw-output",
            str(raw_path),
            "--paper-compat",
            "--python",
            str(config.runtimes.gerrant_python),
            "--gerrant-root",
            str(config.runtimes.gerrant_root),
        )
        return AdapterPlan(system, spec.dataset_id, command, output_path, raw_path)

    if system == "jp":
        missing = []
        if not _executable_available(config.runtimes.jp_python):
            missing.append("isolated JP-ERRANT Python")
        if not _existing_file(config.runtimes.jp_transform_manifest):
            missing.append("transformation manifest")
        if not _existing_directory(config.runtimes.jp_stanza_resources):
            missing.append("Stanza resources")
        if spec.dataset_id == "zh_mucgec" and not _existing_directory(
            config.runtimes.jp_zh_gsdltp_dir
        ):
            missing.append("Chinese GSDLTP resources")
        if missing:
            return unavailable("JP-ERRANT is missing " + ", ".join(missing))
        command_values = [
            core,
            "-m",
            "metc_artifact.adapters.jp_errant",
            "--dataset",
            spec.dataset_id,
            "--pairs",
            str(pairs_path),
            "--output",
            str(output_path),
            "--python",
            str(config.runtimes.jp_python),
            "--transform-manifest",
            str(config.runtimes.jp_transform_manifest),
            "--stanza-resources",
            str(config.runtimes.jp_stanza_resources),
            "--workers",
            str(config.jp_workers),
        ]
        if spec.dataset_id == "zh_mucgec":
            command_values.extend(
                ["--zh-gsdltp-dir", str(config.runtimes.jp_zh_gsdltp_dir)]
            )
        if spec.dataset_id in {
            "de_merlin",
            "cs_merlin",
            "ko_kolla",
            "uk_uagec",
        }:
            raw_path = output_root / spec.dataset_id / "jp.raw.jsonl"
            command_values.extend(
                ["--paper-compat", "--raw-output", str(raw_path)]
            )
        return AdapterPlan(
            system,
            spec.dataset_id,
            tuple(command_values),
            output_path,
            raw_path,
        )

    raise ValueError(f"no adapter plan for {system!r}")


class E2ERunner:
    """Coordinate fresh execution while keeping incompatible tools isolated."""

    def __init__(
        self,
        config: E2EConfig,
        *,
        executor: CommandExecutor = subprocess_executor,
        adapter_planner: AdapterPlanner = default_adapter_plan,
        data_validator: DataValidator = validate_data_artifact,
        dataset_specs: Sequence[DatasetSpec] = DATASETS,
    ) -> None:
        self.config = config
        self.executor = executor
        self.adapter_planner = adapter_planner
        self.data_validator = data_validator
        self.dataset_specs = tuple(dataset_specs)
        self.stages: list[StageResult] = []
        self.artifacts: dict[str, str] = {}
        self.llm_writer: FreshRunWriter | None = None
        self.completed_judges: set[tuple[str, str]] = set()

    def _stage(
        self,
        name: str,
        status: StageStatus,
        summary: str,
        **details: Any,
    ) -> None:
        self.stages.append(StageResult(name, status, summary, details))

    def _environment(self) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not any(marker in key.upper() for marker in _SECRET_ENVIRONMENT_MARKERS)
        }
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        return environment

    def _record_scope(self) -> None:
        if not self.config.require_full_scope:
            return
        expected_datasets = {spec.dataset_id for spec in DATASETS}
        configured_datasets = {spec.dataset_id for spec in self.dataset_specs}
        expected_systems = set(EXECUTABLE_SYSTEMS)
        configured_systems = set(self.config.systems)
        missing_datasets = sorted(expected_datasets - configured_datasets)
        extra_datasets = sorted(configured_datasets - expected_datasets)
        missing_systems = sorted(expected_systems - configured_systems)
        extra_systems = sorted(configured_systems - expected_systems)
        complete = not any(
            (missing_datasets, extra_datasets, missing_systems, extra_systems)
        ) and self.config.run_llm and self.config.metc_inference_profile == "paper"
        self._stage(
            "scope",
            StageStatus.PASS if complete else StageStatus.KNOWN_DIFFERENCE,
            (
                "all released datasets, systems, teacher, and judges are in scope"
                if complete
                else "configured run is scoped and is not a complete E2E verification"
            ),
            missing_datasets=missing_datasets,
            extra_datasets=extra_datasets,
            missing_systems=missing_systems,
            extra_systems=extra_systems,
            effective_systems=list(self.config.systems),
            metc_inference_profile=self.config.metc_inference_profile,
            llm_enabled=self.config.run_llm,
        )

    def _execute(self, stage: str, command: Sequence[str]) -> bool:
        outcome = self.executor(
            tuple(str(value) for value in command),
            cwd=self.config.software_root,
            environment=self._environment(),
            timeout=self.config.timeout_seconds,
        )
        if outcome.returncode:
            self._stage(
                stage,
                StageStatus.FAIL,
                f"subprocess exited with {outcome.returncode}",
                command=list(command),
                stderr_tail=outcome.stderr[-4000:],
            )
            return False
        self._stage(
            stage,
            StageStatus.PASS,
            "subprocess completed",
            command=list(command),
        )
        return True

    def _validate_data(self) -> bool:
        try:
            report = _jsonable_validation(self.data_validator(self.config.data_root))
        except Exception as error:
            self._stage(
                "data.validate",
                StageStatus.FAIL,
                "complete Data artifact validation failed",
                error=str(error),
            )
            return False
        self._stage(
            "data.validate",
            StageStatus.PASS,
            "complete Data artifact is internally consistent",
            report=dict(report),
        )
        return True

    def _reconstruct_datasets(self, reconstructed_root: Path) -> bool:
        cherrant = self.config.runtimes.cherrant_python
        if not _executable_available(cherrant):
            self._stage(
                "datasets.reconstruct",
                StageStatus.FAIL,
                "all-seven reconstruction requires an exact ChERRANT Python",
                configured=bool(cherrant),
            )
            return False
        for spec in self.dataset_specs:
            stale = reconstructed_root / spec.dataset_id / "annotations.jsonl"
            stale.unlink(missing_ok=True)
        command = [
            self.config.runtimes.core_python,
            "-m",
            "metc_artifact.datasets",
            "reconstruct-all",
            "--cache-dir",
            str(self.config.cache_dir),
            "--output-dir",
            str(reconstructed_root),
            "--cherrant-python",
            str(cherrant),
        ]
        if self.config.no_network:
            command.append("--no-network")
        return self._execute("datasets.reconstruct", command)

    def _prepare_dataset(
        self,
        spec: DatasetSpec,
        reconstructed_root: Path,
        pairs_root: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path] | None:
        expected_path = (
            self.config.data_root
            / "benchmarks"
            / spec.dataset_id
            / "annotations.jsonl"
        )
        actual_path = reconstructed_root / spec.dataset_id / "annotations.jsonl"
        try:
            expected = _read_jsonl(expected_path)
            reconstructed = _read_jsonl(actual_path)
            comparison = compare_annotations(
                expected,
                reconstructed,
                spec,
                systems=("gold",),
                max_differences=self.config.max_differences,
            )
        except Exception as error:
            self._stage(
                f"dataset.{spec.dataset_id}.gold",
                StageStatus.FAIL,
                "could not compare reconstructed public reference",
                error=str(error),
            )
            return None
        if not comparison.equal:
            self._stage(
                f"dataset.{spec.dataset_id}.gold",
                StageStatus.FAIL,
                "public reconstruction differs from the authoritative Data artifact",
                comparison=comparison.to_dict(),
            )
            return None
        try:
            reconstructed = _restore_authoritative_row_order(expected, reconstructed)
        except ValueError as error:
            self._stage(
                f"dataset.{spec.dataset_id}.gold",
                StageStatus.FAIL,
                "public reconstruction has an invalid item universe",
                error=str(error),
            )
            return None
        self._stage(
            f"dataset.{spec.dataset_id}.gold",
            StageStatus.PASS,
            "source, correction, and gold annotations match",
            items=len(reconstructed),
            authoritative_order_restored=True,
        )

        try:
            pairs = pairs_from_paper_rows(reconstructed, spec)
            pairs_path = pairs_root / f"{spec.dataset_id}.jsonl"
            write_pairs(pairs_path, pairs)
        except Exception as error:
            self._stage(
                f"dataset.{spec.dataset_id}.pairs",
                StageStatus.FAIL,
                "could not construct canonical system input pairs",
                error=str(error),
            )
            return None
        self._stage(
            f"dataset.{spec.dataset_id}.pairs",
            StageStatus.PASS,
            "canonical source/correction pairs written",
            path=str(pairs_path),
            items=len(pairs),
        )
        self.artifacts[f"pairs.{spec.dataset_id}"] = str(pairs_path)
        return expected, reconstructed, pairs_path

    def _run_system(
        self,
        system: str,
        spec: DatasetSpec,
        expected: list[dict[str, Any]],
        fresh_rows: list[dict[str, Any]],
        pairs_path: Path,
        annotations_root: Path,
    ) -> tuple[list[dict[str, Any]], StageStatus] | None:
        plan = self.adapter_planner(
            system,
            spec,
            pairs_path,
            annotations_root,
            self.config,
        )
        execute_stage = f"system.{system}.{spec.dataset_id}.execute"
        if plan.command is None:
            self._stage(
                execute_stage,
                StageStatus.KNOWN_DIFFERENCE,
                "applicable producer was not executed",
                reason=plan.unavailable_reason,
            )
            return None
        work_root = self.config.work_root.resolve()
        for candidate in (plan.output_path, plan.raw_output_path):
            if candidate is None:
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(work_root):
                self._stage(
                    execute_stage,
                    StageStatus.FAIL,
                    "adapter output path escapes the E2E work root",
                    path=str(candidate),
                )
                return None
            candidate.unlink(missing_ok=True)
        if not self._execute(execute_stage, plan.command):
            return None
        try:
            records = read_annotations(plan.output_path)
            pairs = pairs_from_paper_rows(fresh_rows, spec)
            materialized = materialize_paper_rows(
                fresh_rows,
                pairs,
                records,
                spec,
                system=system,
            )
            comparison = compare_annotations(
                expected,
                materialized,
                spec,
                systems=(system,),
                max_differences=self.config.max_differences,
            )
        except Exception as error:
            self._stage(
                f"system.{system}.{spec.dataset_id}.compare",
                StageStatus.FAIL,
                "fresh annotations could not be materialized or compared",
                error=str(error),
            )
            return None

        known_reason = self.config.known_differences.get((system, spec.dataset_id))
        if (
            known_reason is None
            and system == "metc"
            and self.config.metc_inference_profile == "portable"
        ):
            known_reason = PORTABLE_METC_DIFFERENCE
        if comparison.equal:
            status = StageStatus.PASS
            summary = "fresh semantic annotations match the Data artifact"
            try:
                materialized = _canonicalize_system_order(
                    expected,
                    materialized,
                    spec,
                    system,
                )
            except Exception as error:
                self._stage(
                    f"system.{system}.{spec.dataset_id}.stable_ids",
                    StageStatus.FAIL,
                    "fresh edits could not be aligned to authoritative stable-ID order",
                    error=str(error),
                )
                return None
        elif known_reason:
            status = StageStatus.KNOWN_DIFFERENCE
            summary = "fresh annotations have an explicitly declared known difference"
        else:
            status = StageStatus.FAIL
            summary = "fresh semantic annotations differ from the Data artifact"
        details: dict[str, Any] = {
            "comparison": comparison.to_dict(),
            "stable_id_order": (
                "authoritative_semantic_alignment" if comparison.equal else "unavailable"
            ),
        }
        if known_reason:
            details["known_difference"] = known_reason
        if plan.raw_output_path is not None:
            details["raw_output"] = str(plan.raw_output_path)
            self.artifacts[f"raw.{spec.dataset_id}.{system}"] = str(plan.raw_output_path)
        self._stage(
            f"system.{system}.{spec.dataset_id}.compare",
            status,
            summary,
            **details,
        )
        self.artifacts[f"annotations.{spec.dataset_id}.{system}"] = str(
            plan.output_path
        )
        if status == StageStatus.FAIL:
            return None
        return materialized, status

    def _run_teacher(
        self,
        spec: DatasetSpec,
        expected: list[dict[str, Any]],
        fresh_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], StageStatus] | None:
        prefix = f"teacher.{spec.dataset_id}"
        if self.llm_writer is None:
            self._stage(
                f"{prefix}.execute",
                StageStatus.KNOWN_DIFFERENCE,
                "fresh teacher rerun is disabled",
            )
            return None
        try:
            build = build_teacher_jobs(fresh_rows, spec)
            jobs_path = (
                self.config.work_root
                / "llm"
                / "jobs"
                / "teacher"
                / f"{spec.dataset_id}.jsonl"
            )
            _write_jsonl(jobs_path, build.jobs)
        except Exception as error:
            self._stage(
                f"{prefix}.jobs",
                StageStatus.FAIL,
                "fresh teacher jobs could not be built from fresh METC edits",
                error=str(error),
            )
            return None
        self.artifacts[f"teacher_jobs.{spec.dataset_id}"] = str(jobs_path)
        if build.skipped_overlapping_rows:
            self._stage(
                f"{prefix}.jobs",
                StageStatus.FAIL,
                "fresh METC contains overlapping edits that cannot be teacher-materialized",
                jobs=len(build.jobs),
                edits=build.edit_count,
                skipped_rows=list(build.skipped_overlapping_rows),
            )
            return None
        self._stage(
            f"{prefix}.jobs",
            StageStatus.PASS,
            "teacher jobs built from immutable fresh METC spans and replacements",
            jobs=len(build.jobs),
            edits=build.edit_count,
            path=str(jobs_path),
        )
        if not os.environ.get("OPENAI_API_KEY"):
            self._stage(
                f"{prefix}.execute",
                StageStatus.KNOWN_DIFFERENCE,
                "fresh teacher was not executed because OPENAI_API_KEY is unset",
            )
            return None
        try:
            taxonomy = load_teacher_taxonomy() if build.jobs else None
            summary = run_teacher_batch(
                build.jobs,
                writer=self.llm_writer,
                stream=f"teacher-{spec.dataset_id}",
                taxonomy=taxonomy,
                max_attempts=self.config.llm_max_attempts,
                workers=self.config.llm_workers,
            )
        except Exception as error:
            self._stage(
                f"{prefix}.execute",
                StageStatus.FAIL,
                "fresh teacher batch failed",
                error=str(error),
            )
            return None
        summary_details = summary.to_dict()
        complete_jobs = summary.previously_complete_jobs + summary.newly_complete_jobs
        self.artifacts[f"teacher_records.{spec.dataset_id}"] = str(summary.record_path)
        if summary.failed_jobs or complete_jobs != summary.total_jobs:
            self._stage(
                f"{prefix}.execute",
                StageStatus.FAIL,
                "fresh teacher batch is incomplete after retries",
                batch_summary=summary_details,
            )
            return None
        self._stage(
            f"{prefix}.execute",
            StageStatus.PASS,
            "fresh teacher batch completed",
            batch_summary=summary_details,
        )
        try:
            records = read_fresh_records(summary.record_path)
            materialized = list(
                materialize_teacher_annotations(fresh_rows, spec, records, taxonomy=taxonomy)
            )
            stable = compare_teacher_with_frozen(
                summary.record_path,
                self.config.data_root
                / "benchmarks"
                / spec.dataset_id
                / "annotations.jsonl",
                spec,
                max_mismatches=self.config.max_differences,
            )
            semantic = compare_annotations(
                expected,
                materialized,
                spec,
                systems=("metc_teacher",),
                max_differences=self.config.max_differences,
            )
        except Exception as error:
            self._stage(
                f"{prefix}.materialize",
                StageStatus.FAIL,
                "fresh teacher outputs could not be materialized on METC edits",
                error=str(error),
            )
            return None
        stable_details = stable.to_dict()
        if stable.missing_fresh or stable.extra_fresh:
            self._stage(
                f"{prefix}.compare",
                StageStatus.FAIL,
                "fresh and frozen teacher edit universes differ",
                stable_comparison=stable_details,
                semantic_comparison=semantic.to_dict(),
            )
            return None
        self._stage(
            f"{prefix}.materialize",
            StageStatus.PASS,
            "fresh teacher labels materialized on unchanged fresh METC edits",
            rows=len(materialized),
        )
        self._stage(
            f"{prefix}.compare",
            StageStatus.PASS,
            "fresh teacher label drift was measured without substituting frozen labels",
            stable_comparison=stable_details,
            semantic_comparison=semantic.to_dict(),
        )
        return materialized, StageStatus.PASS

    def _run_acceptability(
        self,
        spec: DatasetSpec,
        fresh_rows: list[dict[str, Any]],
        completed_systems: Sequence[str],
    ) -> None:
        prefix = f"acceptability.{spec.dataset_id}"
        expected_systems = tuple(
            system for system in spec.systems if system != "gold"
        )
        missing_systems = sorted(set(expected_systems) - set(completed_systems))
        if self.llm_writer is None:
            self._stage(
                f"{prefix}.jobs",
                StageStatus.KNOWN_DIFFERENCE,
                "fresh acceptability reruns are disabled",
            )
            return
        if missing_systems:
            self._stage(
                f"{prefix}.jobs",
                StageStatus.KNOWN_DIFFERENCE,
                "acceptability was not run because the fresh paper-system universe is incomplete",
                missing_systems=missing_systems,
            )
            return
        try:
            jobs = build_acceptability_jobs(fresh_rows, spec)
            _validate_acceptability_job_identity(
                jobs,
                self.config.data_root
                / "judges"
                / "canonical"
                / "primary"
                / f"{spec.dataset_id}.jsonl",
            )
            jobs_path = (
                self.config.work_root
                / "llm"
                / "jobs"
                / "acceptability"
                / f"{spec.dataset_id}.jsonl"
            )
            _write_jsonl(jobs_path, jobs)
        except Exception as error:
            self._stage(
                f"{prefix}.jobs",
                StageStatus.FAIL,
                "acceptability jobs could not be built from fresh system annotations",
                error=str(error),
            )
            return
        self._stage(
            f"{prefix}.jobs",
            StageStatus.PASS,
            "shared primary/second jobs built from all fresh annotations",
            jobs=len(jobs),
            path=str(jobs_path),
        )
        self.artifacts[f"acceptability_jobs.{spec.dataset_id}"] = str(jobs_path)

        for role, provider, credential in (
            ("primary", "primary", "OPENAI_API_KEY"),
            ("second", "second", "GOOGLE_API_KEY"),
        ):
            stage = f"{prefix}.{role}"
            if not os.environ.get(credential):
                self._stage(
                    f"{stage}.execute",
                    StageStatus.KNOWN_DIFFERENCE,
                    f"fresh {role} judge was not executed because {credential} is unset",
                )
                continue
            try:
                summary = run_acceptability_batch(
                    jobs,
                    provider=provider,
                    writer=self.llm_writer,
                    stream=f"acceptability-{role}-{spec.dataset_id}",
                    max_attempts=self.config.llm_max_attempts,
                    workers=self.config.llm_workers,
                )
            except Exception as error:
                self._stage(
                    f"{stage}.execute",
                    StageStatus.FAIL,
                    f"fresh {role} judge batch failed",
                    error=str(error),
                )
                continue
            summary_details = summary.to_dict()
            complete_jobs = summary.previously_complete_jobs + summary.newly_complete_jobs
            self.artifacts[f"acceptability_records.{role}.{spec.dataset_id}"] = str(
                summary.record_path
            )
            if summary.failed_jobs or complete_jobs != summary.total_jobs:
                self._stage(
                    f"{stage}.execute",
                    StageStatus.FAIL,
                    f"fresh {role} judge batch is incomplete after retries",
                    batch_summary=summary_details,
                )
                continue
            self._stage(
                f"{stage}.execute",
                StageStatus.PASS,
                f"fresh {role} judge batch completed",
                batch_summary=summary_details,
            )
            try:
                comparison = compare_acceptability_with_frozen(
                    summary.record_path,
                    self.config.data_root
                    / "judges"
                    / "canonical"
                    / role
                    / f"{spec.dataset_id}.jsonl",
                    max_mismatches=self.config.max_differences,
                )
            except Exception as error:
                self._stage(
                    f"{stage}.compare",
                    StageStatus.FAIL,
                    f"fresh {role} verdicts could not be compared by stable edit ID",
                    error=str(error),
                )
                continue
            if comparison.missing_fresh or comparison.extra_fresh:
                self._stage(
                    f"{stage}.compare",
                    StageStatus.FAIL,
                    f"fresh and frozen {role} judge edit universes differ",
                    comparison=comparison.to_dict(),
                )
                continue
            self._stage(
                f"{stage}.compare",
                StageStatus.PASS,
                f"fresh {role} judge drift was measured without substituting frozen verdicts",
                comparison=comparison.to_dict(),
            )
            self.completed_judges.add((spec.dataset_id, role))

    def _fresh_judge_tables(self) -> None:
        if self.llm_writer is None:
            return
        expected = {
            (spec.dataset_id, role)
            for spec in self.dataset_specs
            for role in ("primary", "second")
        }
        missing = sorted(expected - self.completed_judges)
        if missing:
            self._stage(
                "judge_metrics.fresh",
                StageStatus.KNOWN_DIFFERENCE,
                "fresh judge tables were not computed because reruns are incomplete",
                missing=[f"{dataset}:{role}" for dataset, role in missing],
            )
            return
        try:
            corpus = load_judge_rows(
                self.config.data_root,
                judge_source="fresh_rerun",
                fresh_run_root=self.llm_writer.run_root,
            )
            tables = compute_paper_tables(
                corpus,
                samples=self.config.reference_samples,
                seed=self.config.seed,
            )
            output = self.config.work_root / "llm" / "fresh-judge-tables.json"
            _write_json(output, tables)
        except Exception as error:
            self._stage(
                "judge_metrics.fresh",
                StageStatus.FAIL,
                "fresh judge metrics could not be computed from raw rerun records",
                error=str(error),
            )
            return
        self._stage(
            "judge_metrics.fresh",
            StageStatus.PASS,
            "paper judge tables computed only from fresh primary/second records",
            rows=len(corpus.rows),
            path=str(output),
        )
        self.artifacts["judge_tables.fresh"] = str(output)

    def _frozen_judge_tables(self) -> None:
        """Recompute paper tables from the released authoritative raw verdicts."""

        if not (self.config.data_root / "manifest.full.json").is_file():
            # Injected unit-test validators may intentionally use a minimal fixture.
            return
        try:
            corpus = load_judge_rows(
                self.config.data_root,
                judge_source="frozen_authoritative",
            )
            tables = compute_paper_tables(
                corpus,
                samples=self.config.reference_samples,
                seed=self.config.seed,
            )
            if tables.get("judge_source") != "frozen_authoritative":
                raise ValueError("judge table output lost frozen-authoritative provenance")
            output = (
                self.config.work_root
                / "judges"
                / "frozen-authoritative-tables.json"
            )
            _write_json(output, tables)
        except Exception as error:
            self._stage(
                "judge_metrics.frozen_authoritative",
                StageStatus.FAIL,
                "paper judge tables could not be recomputed from released raw verdicts",
                error=str(error),
            )
            return
        self._stage(
            "judge_metrics.frozen_authoritative",
            StageStatus.PASS,
            "paper judge tables recomputed from released authoritative raw verdicts",
            judge_source="frozen_authoritative",
            rows=len(corpus.rows),
            bootstrap_samples=self.config.reference_samples,
            path=str(output),
        )
        self.artifacts["judge_tables.frozen_authoritative"] = str(output)

    def _human_tables(self) -> None:
        """Recompute human-study tables from released assignment-level records."""

        manifest_path = self.config.data_root / "manifest.full.json"
        if not manifest_path.is_file():
            # Injected unit-test validators may intentionally use a minimal fixture.
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            studies = manifest.get("human_annotations")
            if not isinstance(studies, dict):
                raise ValueError("manifest.full.json lacks human_annotations")
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self._stage(
                "human_metrics.configure",
                StageStatus.FAIL,
                "released human-study inventory could not be loaded",
                error=str(error),
            )
            return

        for dataset_id in sorted(studies):
            stage = f"human_metrics.{dataset_id}"
            try:
                corpus = load_human_study(self.config.data_root, dataset_id)
                tables = compute_human_tables(
                    corpus,
                    samples=self.config.human_samples,
                    seed=self.config.seed,
                )
                output = (
                    self.config.work_root
                    / "human"
                    / f"{dataset_id}.json"
                )
                _write_json(output, tables)
            except Exception as error:
                self._stage(
                    stage,
                    StageStatus.FAIL,
                    "human-study tables could not be recomputed from released raw annotations",
                    error=str(error),
                )
                continue
            self._stage(
                stage,
                StageStatus.PASS,
                "human-study tables recomputed from released assignment records",
                assignments=len(corpus.assignments),
                sampling_frame_items=len(corpus.sampling_frame),
                bootstrap_samples=self.config.human_samples,
                judge_source="frozen_authoritative",
                path=str(output),
            )
            self.artifacts[f"human_tables.{dataset_id}"] = str(output)

    def _reference_matching(
        self,
        spec: DatasetSpec,
        rows: list[dict[str, Any]],
        systems: Sequence[str],
        system_statuses: Mapping[str, StageStatus],
        output_root: Path,
    ) -> None:
        if not systems:
            for stage in (
                f"reference_matching.{spec.dataset_id}",
                f"reference_matching.{spec.dataset_id}.exact",
            ):
                self._stage(
                    stage,
                    StageStatus.KNOWN_DIFFERENCE,
                    "no fresh system annotations were available to score",
                )
            return
        score_spec = replace(spec, systems=("gold", *systems))
        known = sorted(
            system
            for system in systems
            if system_statuses.get(system) == StageStatus.KNOWN_DIFFERENCE
        )
        status = StageStatus.KNOWN_DIFFERENCE if known else StageStatus.PASS
        for span_match, suffix in (("overlap", ""), ("exact", ".exact")):
            stage = f"reference_matching.{spec.dataset_id}{suffix}"
            destination = output_root / f"{spec.dataset_id}{suffix}.csv"
            try:
                table = score_reference_matching(
                    rows,
                    score_spec,
                    annotation_source="fresh_rerun",
                    span_match=span_match,
                    samples=self.config.reference_samples,
                    seed=self.config.seed,
                )
                _write_csv(destination, table)
            except Exception as error:
                self._stage(
                    stage,
                    StageStatus.FAIL,
                    f"fresh {span_match}-span reference matching failed",
                    error=str(error),
                )
                continue
            self._stage(
                stage,
                status,
                f"{span_match}-span reference matching computed directly from fresh annotations",
                systems=list(systems),
                known_difference_systems=known,
                span_match=span_match,
                path=str(destination),
            )
            self.artifacts[stage] = str(destination)

    def run(self) -> E2EReport:
        self.config.work_root.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._record_scope()
        if not self._validate_data():
            return E2EReport(tuple(self.stages), self.artifacts)
        self._frozen_judge_tables()
        self._human_tables()
        if self.config.run_llm:
            try:
                self.llm_writer = FreshRunWriter(
                    self.config.work_root / "llm",
                    self.config.llm_run_id,
                )
            except Exception as error:
                self._stage(
                    "llm.configure",
                    StageStatus.FAIL,
                    "fresh LLM run storage could not be configured",
                    error=str(error),
                )
                return E2EReport(tuple(self.stages), self.artifacts)
            self.artifacts["llm.fresh_run_root"] = str(self.llm_writer.run_root)

        reconstructed_root = self.config.work_root / "reconstructed"
        if not self._reconstruct_datasets(reconstructed_root):
            return E2EReport(tuple(self.stages), self.artifacts)

        pairs_root = self.config.work_root / "pairs"
        annotations_root = self.config.work_root / "system-annotations"
        fresh_root = self.config.work_root / "fresh-benchmarks"
        reference_root = self.config.work_root / "reference-matching"

        for spec in self.dataset_specs:
            prepared = self._prepare_dataset(spec, reconstructed_root, pairs_root)
            if prepared is None:
                continue
            expected, fresh_rows, pairs_path = prepared
            completed_systems: list[str] = []
            system_statuses: dict[str, StageStatus] = {}
            for system in EXECUTABLE_SYSTEMS:
                if system not in self.config.systems:
                    continue
                if system not in spec.systems:
                    continue
                if system == "metc_teacher":
                    if "metc" not in completed_systems:
                        self._stage(
                            f"teacher.{spec.dataset_id}.execute",
                            StageStatus.KNOWN_DIFFERENCE,
                            "fresh teacher was not run because exact fresh METC is unavailable",
                        )
                        continue
                    result = self._run_teacher(spec, expected, fresh_rows)
                else:
                    result = self._run_system(
                        system,
                        spec,
                        expected,
                        fresh_rows,
                        pairs_path,
                        annotations_root,
                    )
                if result is None:
                    continue
                fresh_rows, system_status = result
                completed_systems.append(system)
                system_statuses[system] = system_status

            fresh_path = fresh_root / spec.dataset_id / "annotations.jsonl"
            _write_jsonl(fresh_path, fresh_rows)
            self.artifacts[f"fresh_benchmark.{spec.dataset_id}"] = str(fresh_path)
            self._reference_matching(
                spec,
                fresh_rows,
                completed_systems,
                system_statuses,
                reference_root,
            )
            if self.config.run_llm:
                judge_ready_systems = [
                    system
                    for system in completed_systems
                    if system_statuses.get(system) == StageStatus.PASS
                ]
                self._run_acceptability(spec, fresh_rows, judge_ready_systems)

        if self.config.run_llm:
            self._fresh_judge_tables()

        return E2EReport(tuple(self.stages), self.artifacts)


def _known_difference(value: str) -> tuple[tuple[str, str], str]:
    try:
        identity, reason = value.split("=", 1)
        system, dataset_id = identity.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "known differences use SYSTEM:DATASET=REASON"
        ) from error
    if system not in EXECUTABLE_SYSTEMS or not dataset_id or not reason.strip():
        raise argparse.ArgumentTypeError("known difference has an invalid system, dataset, or reason")
    return (system, dataset_id), reason.strip()


def build_parser() -> argparse.ArgumentParser:
    software_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="extracted complete METC Data artifact",
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--core-python", default=sys.executable)
    parser.add_argument("--cherrant-python", default=os.environ.get("CHERRANT_PYTHON"))
    parser.add_argument("--metc-python", default=os.environ.get("METC_SYSTEM_PYTHON"))
    parser.add_argument(
        "--metc-checkpoint",
        type=Path,
        default=os.environ.get("METC_MODEL_PATH"),
    )
    parser.add_argument(
        "--metc-inference-profile",
        choices=METC_INFERENCE_PROFILES,
        default="paper",
        help=(
            "paper uses the evaluated CUDA/bfloat16 runtime; portable runs fp32 on "
            "available local hardware and reports semantic drift as a known difference"
        ),
    )
    parser.add_argument("--errant-python", default=os.environ.get("ERRANT_PYTHON"))
    parser.add_argument("--gerrant-python", default=os.environ.get("GERRANT_PYTHON"))
    parser.add_argument(
        "--gerrant-root",
        type=Path,
        default=os.environ.get("GERRANT_ROOT")
        or software_root / "baselines" / "gerrant_de",
    )
    parser.add_argument("--jp-python", default=os.environ.get("JP_ERRANT_PYTHON"))
    parser.add_argument(
        "--jp-transform-manifest",
        "--jp-patch-manifest",
        dest="jp_transform_manifest",
        type=Path,
        default=os.environ.get("JP_ERRANT_TRANSFORM_MANIFEST")
        or os.environ.get("JP_ERRANT_PATCH_MANIFEST"),
    )
    parser.add_argument(
        "--jp-stanza-resources",
        type=Path,
        default=os.environ.get("JP_ERRANT_STANZA_RESOURCES"),
    )
    parser.add_argument(
        "--jp-zh-gsdltp-dir",
        type=Path,
        default=os.environ.get("ZH_GSDLTP_DIR"),
    )
    parser.add_argument("--jp-workers", type=int, default=2)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=SYSTEM_CLI_CHOICES,
        default=("all",),
        help=(
            "annotation systems to execute: METC, METC teacher, ERRANT, gERRANT, "
            "and JP-ERRANT (public gerrant/jp_errant names are recommended; "
            "internal ger/jp aliases are also accepted); ChERRANT is "
            "always used for Chinese reference reconstruction"
        ),
    )
    parser.add_argument(
        "--known-difference",
        action="append",
        type=_known_difference,
        default=[],
        metavar="SYSTEM:DATASET=REASON",
    )
    parser.add_argument("--reference-samples", type=int, default=1000)
    parser.add_argument("--human-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-differences", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=24 * 60 * 60)
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="forbid benchmark reconstruction from downloading missing public inputs",
    )
    parser.add_argument(
        "--llm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rerun the fresh teacher and both acceptability judges (expensive)",
    )
    parser.add_argument("--llm-run-id", default="e2e")
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--llm-max-attempts", type=int, default=4)
    return parser


def config_from_args(args: argparse.Namespace) -> E2EConfig:
    systems = normalize_systems(args.systems)
    known: dict[tuple[str, str], str] = {}
    for key, reason in args.known_difference:
        if key in known:
            raise ValueError(f"duplicate known-difference declaration for {key}")
        known[key] = reason
    work_root = args.work_root.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else work_root / "cache"
    )
    software_root = Path(__file__).resolve().parents[2]
    return E2EConfig(
        data_root=args.data_root.expanduser().resolve(),
        work_root=work_root,
        cache_dir=cache_dir,
        software_root=software_root,
        runtimes=RuntimePaths(
            core_python=args.core_python,
            cherrant_python=args.cherrant_python,
            metc_python=args.metc_python,
            metc_checkpoint=args.metc_checkpoint,
            errant_python=args.errant_python,
            gerrant_python=args.gerrant_python,
            gerrant_root=args.gerrant_root,
            jp_python=args.jp_python,
            jp_transform_manifest=args.jp_transform_manifest,
            jp_stanza_resources=args.jp_stanza_resources,
            jp_zh_gsdltp_dir=args.jp_zh_gsdltp_dir,
        ),
        systems=systems,
        known_differences=known,
        metc_inference_profile=args.metc_inference_profile,
        jp_workers=args.jp_workers,
        reference_samples=args.reference_samples,
        human_samples=args.human_samples,
        seed=args.seed,
        max_differences=args.max_differences,
        timeout_seconds=args.timeout_seconds,
        run_llm=args.llm,
        llm_run_id=args.llm_run_id,
        llm_workers=args.llm_workers,
        llm_max_attempts=args.llm_max_attempts,
        no_network=args.no_network,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as error:
        raise SystemExit(f"invalid E2E configuration: {error}") from error
    report = E2ERunner(config).run()
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_path = args.report or config.work_root / "e2e-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return {
        StageStatus.PASS: 0,
        StageStatus.FAIL: 1,
        StageStatus.KNOWN_DIFFERENCE: 2,
    }[report.status]


if __name__ == "__main__":
    raise SystemExit(main())
