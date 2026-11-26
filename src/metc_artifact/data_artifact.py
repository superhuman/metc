"""Validation for the complete METC evaluation Data artifact.

The validator checks the scientific content, not only archive hashes.  It
reconstructs the expected stable judge-job universe from the released
benchmark annotations and verifies that every frozen judge and human record
refers to an exact released edit.
"""

from __future__ import annotations

import ast
import hashlib
import json
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .judge_jobs import EDIT_ID_VERSION, build_judge_jobs
from .judges.protocols import (
    ACCEPTABILITY_PROMPT,
    ACCEPTABILITY_SYSTEM_MESSAGE,
    TEACHER_PROMPT,
    TEACHER_SYSTEM_MESSAGE,
    sha256_text,
)
from .public_ids import SAMPLE_ID_VERSION, public_sample_id
from .schema import DATASETS
from .semantic import canonical_item_id, semantic_edit, source_text, target_text


QUALITY_LEVELS = frozenset({"Good", "Acceptable", "Bad"})
HUMAN_STUDY_DATASETS = frozenset({"en_conll14_ann0", "de_merlin"})
JUDGE_CONTRACTS = {
    "primary": {
        "model_name": "GPT-5.4",
        "recorded_backend_id": "openai_direct_gpt54_flex",
    },
    "second": {
        "model_name": "Gemini 3.5 Flash",
        "recorded_backend_id": "vai_gemini35_flash",
    },
}
TEACHER_CONTRACT = {
    "schema_version": "metc-teacher-v1",
    "recorded_backend_id": "openai_direct_gpt54_flex",
    "public_model": "gpt-5.4",
    "service_tier": "flex",
    "source_commit": "a39041e3",
    "system_message": TEACHER_SYSTEM_MESSAGE,
    "request_configuration": {
        "reasoning_effort": "low",
        "temperature_explicitly_set": False,
        "response_format_explicitly_set": False,
        "calls_per_nonempty_sentence": 1,
    },
}
TEACHER_REVIEWED_PROMPT_SHA256 = (
    "139c913ad45c9a4807fc1934e40cc35ccda06a97ec4f152d8683accf25aed370"
)
HUMAN_ASSIGNMENT_FIELDS = frozenset(
    {
        "assignment_order",
        "corrected_sentence",
        "dataset_id",
        "decorated_sentence",
        "edit_count",
        "edits",
        "item_id",
        "language",
        "pass_id",
        "row_idx",
        "sample_id",
        "session_unannotatable",
        "source_sentence",
        "status",
        "system",
    }
)
HUMAN_ASSIGNMENT_EDIT_FIELDS = frozenset(
    {
        "displayed_edit",
        "displayed_span",
        "edit_id",
        "edit_idx",
        "label",
        "label_confidence",
        "label_quality",
        "slot",
        "span_confidence",
        "span_quality",
        "unannotatable",
    }
)
HUMAN_FRAME_FIELDS = frozenset(
    {
        "corrected_sentence",
        "dataset_id",
        "decorated_sentence",
        "edit_count",
        "edits",
        "item_id",
        "language",
        "row_idx",
        "sample_id",
        "selected_for_human_study",
        "selection_rank",
        "source_sentence",
        "system",
    }
)
HUMAN_FRAME_EDIT_FIELDS = frozenset(
    {
        "displayed_edit",
        "displayed_span",
        "edit_id",
        "edit_idx",
        "label",
        "slot",
    }
)
GERRANT_RUNTIME_FIELDS = {
    "schema_version": "metc-gerrant-resources-v1",
    "license_expression": "GPL-2.0-only OR GPL-3.0-only",
    "upstream_url": ("https://www.j3e.de/ispell/igerman98/dict/igerman98-20161207.tar.bz2"),
}
GERRANT_RUNTIME_FILES = {
    "source_archive": {
        "path": "resources/gerrant/igerman98-20161207.tar.bz2",
        "size_bytes": 470_207,
        "sha256": "17296f03c5fea62d76ecc530ebe80f6adc430278f58d472dc1842d71612960a8",
    },
    "derived_dictionary": {
        "path": "resources/gerrant/de_DE-large.txt",
        "size_bytes": 4_778_232,
        "sha256": "577c06ebd6461c0a43486a9a538c1175fbc1f0d1a3985e9b4350a5ad1c5d9da6",
        "derivation": "igerman98-unmunch-v1",
    },
    "source_notice": {
        "path": "resources/gerrant/SOURCE.md",
        "size_bytes": 1_869,
        "sha256": "4b7dfe40dbb9148d263473b48e5a894eae1b89cf40bcee965932c01eb4e799c6",
    },
}
GERRANT_UPSTREAM_NOTICES = {
    "Copyright": {
        "path": "resources/gerrant/Copyright",
        "size_bytes": 760,
        "sha256": "03abf202c6e207d41f054083cafccbb174a30f6b330ea8dba3aeef0a279cb83e",
    },
    "COPYING_GPLv2": {
        "path": "resources/gerrant/COPYING_GPLv2",
        "size_bytes": 17_992,
        "sha256": "e7431ac20e815b6797369e802f9e4bb1b0082485060d8b096235f035d741bd89",
    },
    "COPYING_GPLv3": {
        "path": "resources/gerrant/COPYING_GPLv3",
        "size_bytes": 35_147,
        "sha256": "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
    },
    "README_de_DE.txt": {
        "path": "resources/gerrant/README_de_DE.txt",
        "size_bytes": 264,
        "sha256": "1a99ebf419bb04ef1b320227f35480abcea6f5b0489c7e43d394be4ff91cd65a",
        "encoding": "utf-8",
        "encoding_conversion": ("iso-8859-1 to utf-8; text and line endings preserved"),
        "source_member_sha256": (
            "9838bad33b988cf84c5abd4ad94b36e7f5b8ff292115a5a224c50f11eaa6afd0"
        ),
    },
}


class DataArtifactError(RuntimeError):
    """Raised when the complete Data artifact violates its public contract."""


@dataclass(frozen=True)
class DataValidationReport:
    benchmark_items: int
    benchmark_edits: int
    teacher_sentences: int
    teacher_edits: int
    canonical_judge_edits: int
    human_sampling_frame_items: int
    human_assignments: int
    human_edit_slots: int

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "valid": True,
            "benchmark_items": self.benchmark_items,
            "benchmark_edits": self.benchmark_edits,
            "teacher_sentences": self.teacher_sentences,
            "teacher_edits": self.teacher_edits,
            "canonical_judge_edits": self.canonical_judge_edits,
            "human_sampling_frame_items": self.human_sampling_frame_items,
            "human_assignments": self.human_assignments,
            "human_edit_slots": self.human_edit_slots,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataArtifactError(f"{context}: expected a JSON object")
    return value


def _require_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise DataArtifactError(f"{context}: expected an integer >= {minimum}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise DataArtifactError(
            f"{context}: fields differ from the public schema; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _safe_artifact_path(root: Path, relative: Any, context: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise DataArtifactError(f"{context}: invalid artifact path")
    if "\\" in relative:
        raise DataArtifactError(f"{context}: backslashes are not allowed")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise DataArtifactError(f"{context}: path must be normalized and relative")
    path = root.joinpath(*parsed.parts)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise DataArtifactError(f"{context}: missing file {relative}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DataArtifactError(f"{context}: artifact entry must be a regular file")
    return path


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DataArtifactError(f"{context}: expected a lowercase SHA-256")
    return value


def _verify_file_metadata(root: Path, metadata: Mapping[str, Any], context: str) -> Path:
    path = _safe_artifact_path(root, metadata.get("path"), context)
    expected_size = _require_int(metadata.get("size_bytes"), f"{context}.size_bytes")
    if path.stat().st_size != expected_size:
        raise DataArtifactError(
            f"{context}: size is {path.stat().st_size}, expected {expected_size}"
        )
    expected_digest = _require_sha256(metadata.get("sha256"), f"{context}.sha256")
    observed_digest = _sha256_file(path)
    if observed_digest != expected_digest:
        raise DataArtifactError(
            f"{context}: SHA-256 is {observed_digest}, expected {expected_digest}"
        )
    return path


def _validate_complete_inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("release_status") is None:
        return
    release_status = _require_mapping(manifest.get("release_status"), "release_status")
    if release_status.get("stage") != "complete":
        return

    required_flags = {
        "complete_benchmark_text",
        "complete_system_annotations",
        "complete_raw_teacher_caches",
        "complete_frozen_judges",
        "complete_deidentified_human_annotations",
        "complete_human_sampling_frames",
        "complete_runtime_resources",
    }
    if any(release_status.get(flag) is not True for flag in required_flags):
        raise DataArtifactError(
            "release_status: complete Data artifact has an incomplete coverage flag"
        )

    inventory = manifest.get("files")
    if not isinstance(inventory, list) or not inventory:
        raise DataArtifactError("files: complete Data artifact requires a nonempty flat inventory")
    declared: set[str] = set()
    for index, value in enumerate(inventory):
        context = f"files.{index}"
        metadata = _require_mapping(value, context)
        path = _verify_file_metadata(root, metadata, context)
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.full.json":
            raise DataArtifactError(
                "files: the self-referential manifest must not inventory itself"
            )
        if relative in declared:
            raise DataArtifactError(f"files: duplicate path {relative}")
        declared.add(relative)

    required_documents = {
        "CC-BY-4.0-NOTICE.txt",
        "CC-BY-SA-4.0-NOTICE.txt",
        "CONLL-2014-NOTICE.txt",
        "README.md",
        "LICENSE",
        "LICENSE-APACHE-2.0",
        "LICENSE-GPL-3.0",
        "DATASET_LICENSES.md",
    }
    if not required_documents.issubset(declared):
        missing = sorted(required_documents - declared)
        raise DataArtifactError(
            f"files: complete Data artifact is missing required documents {missing}"
        )

    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise DataArtifactError(f"files: symlink is not allowed: {relative}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise DataArtifactError(f"files: non-regular entry is not allowed: {relative}")
        actual.add(relative)
    expected = declared | {"manifest.full.json"}
    if actual != expected:
        raise DataArtifactError(
            "files: payload inventory mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _verify_exact_metadata(
    root: Path,
    value: Any,
    expected: Mapping[str, Any],
    context: str,
) -> None:
    metadata = _require_mapping(value, context)
    if dict(metadata) != dict(expected):
        raise DataArtifactError(f"{context}: metadata differs from the pinned contract")
    _verify_file_metadata(root, metadata, context)


def _validate_runtime_resources(root: Path, manifest: Mapping[str, Any]) -> None:
    release_status_value = manifest.get("release_status")
    if release_status_value is None:
        return
    release_status = _require_mapping(release_status_value, "release_status")
    if release_status.get("stage") != "complete":
        return

    resources = _require_mapping(manifest.get("runtime_resources"), "runtime_resources")
    if set(resources) != {"gerrant"}:
        raise DataArtifactError(
            "runtime_resources: complete Data artifact requires exactly gerrant"
        )
    gerrant = _require_mapping(resources["gerrant"], "runtime_resources.gerrant")
    expected_fields = {
        *GERRANT_RUNTIME_FIELDS,
        *GERRANT_RUNTIME_FILES,
        "upstream_notices",
    }
    if set(gerrant) != expected_fields:
        raise DataArtifactError("runtime_resources.gerrant: fields differ from the pinned contract")
    for field, expected in GERRANT_RUNTIME_FIELDS.items():
        if gerrant.get(field) != expected:
            raise DataArtifactError(
                f"runtime_resources.gerrant.{field}: value differs from the pinned contract"
            )
    for field, expected in GERRANT_RUNTIME_FILES.items():
        _verify_exact_metadata(
            root,
            gerrant.get(field),
            expected,
            f"runtime_resources.gerrant.{field}",
        )
    notices = _require_mapping(
        gerrant.get("upstream_notices"),
        "runtime_resources.gerrant.upstream_notices",
    )
    if set(notices) != set(GERRANT_UPSTREAM_NOTICES):
        raise DataArtifactError("runtime_resources.gerrant.upstream_notices: inventory differs")
    for name, expected in GERRANT_UPSTREAM_NOTICES.items():
        _verify_exact_metadata(
            root,
            notices.get(name),
            expected,
            f"runtime_resources.gerrant.upstream_notices.{name}",
        )


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataArtifactError(f"{context}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise DataArtifactError(f"{context}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _python_literal(path: Path, name: str, context: str) -> Any:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise DataArtifactError(f"{context}: cannot parse released Python source") from error
    for statement in module.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        if isinstance(target, ast.Name) and target.id == name and value is not None:
            try:
                return ast.literal_eval(value)
            except (ValueError, TypeError) as error:
                raise DataArtifactError(
                    f"{context}: {name} is not a literal public contract"
                ) from error
    raise DataArtifactError(f"{context}: missing literal {name}")


def _validate_protocol_provenance(
    manifest: Mapping[str, Any], software_root: Path
) -> None:
    protocol = _require_mapping(manifest.get("judge_protocol"), "judge_protocol")
    _require_exact_keys(
        protocol,
        frozenset(
            {
                "prompt_path_in_code_artifact",
                "prompt_sha256",
                "stable_edit_id_version",
            }
        ),
        "judge_protocol",
    )
    if protocol.get("stable_edit_id_version") != EDIT_ID_VERSION:
        raise DataArtifactError("judge_protocol: stable edit ID version differs from code")
    if protocol.get("prompt_path_in_code_artifact") != "prompts/acceptability_judge.py":
        raise DataArtifactError("judge_protocol: unexpected judge prompt path")
    judge_prompt_path = _safe_artifact_path(
        software_root,
        protocol["prompt_path_in_code_artifact"],
        "judge_protocol.prompt_path_in_code_artifact",
    )
    released_judge_prompt = _python_literal(
        judge_prompt_path, "JUDGE_PROMPT", "judge_protocol"
    )
    expected_prompt_hash = sha256_text(ACCEPTABILITY_PROMPT)
    if (
        released_judge_prompt != ACCEPTABILITY_PROMPT
        or protocol.get("prompt_sha256") != expected_prompt_hash
    ):
        raise DataArtifactError(
            "judge_protocol: prompt text or checksum differs from the rerun protocol"
        )
    if (
        _python_literal(judge_prompt_path, "JUDGE_SYSTEM_MESSAGE", "judge_protocol")
        != ACCEPTABILITY_SYSTEM_MESSAGE
    ):
        raise DataArtifactError("judge_protocol: system message differs from rerun code")
    if (
        _python_literal(judge_prompt_path, "PRIMARY_JUDGE_MODEL", "judge_protocol")
        != JUDGE_CONTRACTS["primary"]["model_name"]
        or _python_literal(
            judge_prompt_path, "SECOND_JUDGE_MODEL", "judge_protocol"
        )
        != JUDGE_CONTRACTS["second"]["model_name"]
    ):
        raise DataArtifactError("judge_protocol: released judge model names differ")

    judges = _require_mapping(manifest.get("judges"), "judges")
    if set(judges) != set(JUDGE_CONTRACTS):
        raise DataArtifactError("judges: expected exactly primary and second")
    for judge_name, expected in JUDGE_CONTRACTS.items():
        metadata = _require_mapping(judges[judge_name], f"judges.{judge_name}")
        for field, expected_value in expected.items():
            if metadata.get(field) != expected_value:
                raise DataArtifactError(
                    f"judges.{judge_name}.{field}: differs from the rerun contract"
                )

    teacher = _require_mapping(manifest.get("metc_teacher"), "metc_teacher")
    for field, expected_value in TEACHER_CONTRACT.items():
        if teacher.get(field) != expected_value:
            raise DataArtifactError(
                f"metc_teacher.{field}: differs from the recorded teacher protocol"
            )
    prompt = _require_mapping(teacher.get("prompt"), "metc_teacher.prompt")
    _require_exact_keys(
        prompt,
        frozenset(
            {
                "reviewed_source_path",
                "reviewed_source_sha256",
                "code_artifact_path",
                "code_artifact_sha256",
            }
        ),
        "metc_teacher.prompt",
    )
    if (
        prompt.get("reviewed_source_path")
        != "gec-annotations/scripts/metc_annotation/annotation/prompt.py"
        or prompt.get("reviewed_source_sha256") != TEACHER_REVIEWED_PROMPT_SHA256
        or prompt.get("code_artifact_path") != "prompts/annotation_prompt.py"
    ):
        raise DataArtifactError("metc_teacher.prompt: source provenance differs")
    teacher_prompt_path = _safe_artifact_path(
        software_root,
        prompt["code_artifact_path"],
        "metc_teacher.prompt.code_artifact_path",
    )
    # These digests identify the inputs to the frozen teacher run.
    _require_sha256(
        prompt.get("code_artifact_sha256"), "metc_teacher.prompt.code_artifact_sha256"
    )
    if (
        _python_literal(
            teacher_prompt_path, "_ANNOTATION_TEMPLATE", "metc_teacher.prompt"
        )
        != TEACHER_PROMPT
        or _python_literal(
            teacher_prompt_path, "ANNOTATION_SYSTEM_MESSAGE", "metc_teacher.prompt"
        )
        != TEACHER_SYSTEM_MESSAGE
        or _python_literal(
            teacher_prompt_path, "ANNOTATION_TEACHER_MODEL", "metc_teacher.prompt"
        )
        != JUDGE_CONTRACTS["primary"]["model_name"]
        or _python_literal(
            teacher_prompt_path, "ANNOTATION_REQUEST_SETTINGS", "metc_teacher.prompt"
        )
        != {"reasoning_effort": "low"}
    ):
        raise DataArtifactError("metc_teacher.prompt: released teacher protocol differs")

    taxonomy = _require_mapping(teacher.get("taxonomy"), "metc_teacher.taxonomy")
    _require_exact_keys(
        taxonomy,
        frozenset({"code_artifact_path", "sha256"}),
        "metc_teacher.taxonomy",
    )
    if taxonomy.get("code_artifact_path") != "prompts/taxonomy_v2.md":
        raise DataArtifactError("metc_teacher.taxonomy: unexpected taxonomy path")
    _safe_artifact_path(
        software_root,
        taxonomy["code_artifact_path"],
        "metc_teacher.taxonomy.code_artifact_path",
    )
    _require_sha256(taxonomy.get("sha256"), "metc_teacher.taxonomy.sha256")


def _benchmark_rows(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    int,
    int,
]:
    benchmark_manifest = _require_mapping(manifest.get("benchmarks"), "benchmarks")
    expected_ids = {spec.dataset_id for spec in DATASETS}
    if set(benchmark_manifest) != expected_ids:
        raise DataArtifactError(
            "benchmarks: dataset IDs differ from the seven-setting public contract"
        )

    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    expected_jobs: dict[str, dict[str, dict[str, Any]]] = {}
    item_total = 0
    edit_total = 0
    for spec in DATASETS:
        context = f"benchmarks.{spec.dataset_id}"
        metadata = _require_mapping(benchmark_manifest[spec.dataset_id], context)
        path = _verify_file_metadata(root, metadata, context)
        rows = _read_jsonl(path, context)
        if len(rows) != _require_int(metadata.get("items"), f"{context}.items"):
            raise DataArtifactError(f"{context}: item count does not match manifest")
        if tuple(metadata.get("systems", ())) != spec.systems:
            raise DataArtifactError(f"{context}: system inventory does not match code")
        if metadata.get("experiment_code") != spec.experiment_code:
            raise DataArtifactError(f"{context}: experiment code does not match code")
        if metadata.get("edit_format") != spec.edit_format:
            raise DataArtifactError(f"{context}: edit format does not match code")
        if metadata.get("evaluation_span_unit") != spec.evaluation_span_unit:
            raise DataArtifactError(f"{context}: span unit does not match code")

        seen_items: set[str] = set()
        observed_counts: Counter[str] = Counter()
        all_edits: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(rows, 1):
            try:
                item_id = canonical_item_id(row)
                source = source_text(row, spec)
                target = target_text(row, spec)
            except (KeyError, TypeError, ValueError) as error:
                raise DataArtifactError(
                    f"{context}:{row_number}: invalid benchmark item: {error}"
                ) from error
            if item_id in seen_items:
                raise DataArtifactError(f"{context}: duplicate item ID {item_id}")
            seen_items.add(item_id)
            row_idx = row.get("idx", row.get("sent_idx"))
            for system in spec.systems:
                raw_edits = row.get(system)
                if not isinstance(raw_edits, list):
                    raise DataArtifactError(
                        f"{context}/{item_id}: {system} annotations must be a list"
                    )
                observed_counts[system] += len(raw_edits)
                for edit_idx, raw_edit in enumerate(raw_edits):
                    try:
                        normalized = semantic_edit(row, raw_edit, spec)
                    except (KeyError, TypeError, ValueError) as error:
                        raise DataArtifactError(
                            f"{context}/{item_id}/{system}/{edit_idx}: {error}"
                        ) from error
                    edit_id = f"metc-edit-v1:{spec.experiment_code}:{row_idx}:{system}:{edit_idx}"
                    if edit_id in all_edits:
                        raise DataArtifactError(f"{context}: duplicate stable edit ID {edit_id}")
                    all_edits[edit_id] = {
                        "dataset_id": spec.dataset_id,
                        "item_id": item_id,
                        "row_idx": row_idx,
                        "system": system,
                        "edit_idx": edit_idx,
                        "label": normalized.native_label,
                        "start": normalized.start,
                        "end": normalized.end,
                        "source": source,
                        "target": target,
                    }

        declared_counts = _require_mapping(
            metadata.get("system_edit_counts"), f"{context}.system_edit_counts"
        )
        if dict(observed_counts) != dict(declared_counts):
            raise DataArtifactError(f"{context}: system edit counts do not match manifest")
        judge_build = build_judge_jobs(rows, spec)
        jobs = {job["edit_id"]: dict(job) for job in judge_build.jobs}
        if len(jobs) != len(judge_build.jobs):
            raise DataArtifactError(f"{context}: judge job IDs are not unique")
        unknown_jobs = set(jobs) - set(all_edits)
        if unknown_jobs:
            raise DataArtifactError(f"{context}: judge builder produced unknown edit IDs")
        for edit_id, job in jobs.items():
            released = all_edits[edit_id]
            job.update(
                {
                    "dataset_id": spec.dataset_id,
                    "item_id": released["item_id"],
                    "source": released["source"],
                    "target": released["target"],
                    "start": released["start"],
                    "end": released["end"],
                }
            )
        rows_by_dataset[spec.dataset_id] = rows
        expected_jobs[spec.dataset_id] = jobs
        item_total += len(rows)
        edit_total += sum(observed_counts.values())
    return rows_by_dataset, expected_jobs, item_total, edit_total


def _raw_label(raw_edit: Any, spec: Any) -> str:
    if spec.edit_format == "char_tuple":
        return str(raw_edit[3])
    return str(raw_edit.get("raw_label", ""))


def _validate_teacher(
    root: Path,
    manifest: Mapping[str, Any],
    rows_by_dataset: Mapping[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    metadata = _require_mapping(manifest.get("metc_teacher"), "metc_teacher")
    if metadata.get("schema_version") != "metc-teacher-v1":
        raise DataArtifactError("metc_teacher: unsupported schema version")
    if metadata.get("recorded_backend_id") != "openai_direct_gpt54_flex":
        raise DataArtifactError("metc_teacher: recorded backend ID is not authoritative")
    if metadata.get("public_model") != "gpt-5.4" or metadata.get("service_tier") != "flex":
        raise DataArtifactError("metc_teacher: public model mapping is incomplete")
    datasets = _require_mapping(metadata.get("datasets"), "metc_teacher.datasets")
    if set(datasets) != {spec.dataset_id for spec in DATASETS}:
        raise DataArtifactError("metc_teacher: dataset coverage is incomplete")

    sentence_total = 0
    edit_total = 0
    for spec in DATASETS:
        context = f"metc_teacher.datasets.{spec.dataset_id}"
        dataset_meta = _require_mapping(datasets[spec.dataset_id], context)
        path = _verify_file_metadata(root, dataset_meta, context)
        cache_rows = _read_jsonl(path, context)
        if len(cache_rows) != _require_int(dataset_meta.get("sentences"), f"{context}.sentences"):
            raise DataArtifactError(f"{context}: sentence count does not match manifest")
        benchmark = {canonical_item_id(row): row for row in rows_by_dataset[spec.dataset_id]}
        expected_ids = {item_id for item_id, row in benchmark.items() if row.get("metc")}
        observed_ids: set[str] = set()
        observed_edits = 0
        for line_number, cache_row in enumerate(cache_rows, 1):
            if "idx" not in cache_row:
                raise DataArtifactError(f"{context}:{line_number}: missing item ID")
            item_id = json.dumps(
                cache_row["idx"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if item_id in observed_ids or item_id not in benchmark:
                raise DataArtifactError(
                    f"{context}:{line_number}: duplicate or unknown item ID {item_id}"
                )
            observed_ids.add(item_id)
            labels = cache_row.get("labels")
            sublabels = cache_row.get("sublabels")
            if not isinstance(labels, list) or not isinstance(sublabels, list):
                raise DataArtifactError(f"{context}:{line_number}: malformed label arrays")
            row = benchmark[item_id]
            student = row.get("metc")
            teacher = row.get("metc_teacher")
            if (
                not isinstance(student, list)
                or not isinstance(teacher, list)
                or len(labels) != len(sublabels)
                or len(labels) != len(student)
                or len(teacher) != len(student)
            ):
                raise DataArtifactError(
                    f"{context}:{line_number}: cache and annotation lengths differ"
                )
            expected_labels = [
                f"{label}/{sublabel}" for label, sublabel in zip(labels, sublabels, strict=True)
            ]
            observed_labels = [_raw_label(edit, spec) for edit in teacher]
            if expected_labels != observed_labels:
                raise DataArtifactError(
                    f"{context}:{line_number}: cache labels differ from applied teacher labels"
                )
            for edit_idx, (student_edit, teacher_edit) in enumerate(
                zip(student, teacher, strict=True)
            ):
                student_semantic = semantic_edit(row, student_edit, spec)
                teacher_semantic = semantic_edit(row, teacher_edit, spec)
                if (
                    student_semantic.start,
                    student_semantic.end,
                    student_semantic.replacement,
                ) != (
                    teacher_semantic.start,
                    teacher_semantic.end,
                    teacher_semantic.replacement,
                ):
                    raise DataArtifactError(
                        f"{context}:{line_number}/{edit_idx}: teacher changed the student edit"
                    )
            observed_edits += len(labels)
        if observed_ids != expected_ids:
            raise DataArtifactError(f"{context}: nonempty-sentence coverage is incomplete")
        if observed_edits != _require_int(dataset_meta.get("edits"), f"{context}.edits"):
            raise DataArtifactError(f"{context}: edit count does not match manifest")
        sentence_total += len(cache_rows)
        edit_total += observed_edits
    totals = _require_mapping(metadata.get("totals"), "metc_teacher.totals")
    if totals != {"sentences": sentence_total, "edits": edit_total}:
        raise DataArtifactError("metc_teacher: totals do not match dataset records")
    return sentence_total, edit_total


def _legacy_judge_key(row: Mapping[str, Any], context: str) -> tuple[Any, ...]:
    fields = ("system", "row_idx", "label", "edit_diff")
    missing = [field for field in fields if field not in row]
    if missing:
        raise DataArtifactError(
            f"{context}: legacy judge row lacks identity fields {missing}"
        )
    return tuple(row[field] for field in fields)


def _valid_raw_verdict(row: Mapping[str, Any]) -> bool:
    return (
        row.get("span_quality") in QUALITY_LEVELS
        and row.get("label_quality") in QUALITY_LEVELS
    )


def _validate_raw_judge_identity(
    row: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    backend: str,
    context: str,
) -> None:
    if row.get("judge") != backend:
        raise DataArtifactError(f"{context}: raw judge backend differs from manifest")
    for field in ("system", "row_idx", "label"):
        if row.get(field) != job.get(field):
            raise DataArtifactError(
                f"{context}: stable judge row {field} differs from released edit"
            )
    if row.get("edit_idx") is not None and row.get("edit_idx") != job.get("edit_idx"):
        raise DataArtifactError(
            f"{context}: stable judge row edit_idx differs from released edit"
        )


def _materialize_raw_judge_rows(
    *,
    expected: Mapping[str, Mapping[str, Any]],
    base_rows: list[dict[str, Any]],
    overlay_rows: list[dict[str, Any]],
    backend: str,
    context: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, int]]:
    expected_by_legacy: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    id_to_legacy: dict[str, tuple[Any, ...]] = {}
    for edit_id, job in expected.items():
        key = _legacy_judge_key(job, f"{context}.expected.{edit_id}")
        expected_by_legacy[key].append(job)
        id_to_legacy[edit_id] = key

    overlay_by_id: dict[str, Mapping[str, Any]] = {}
    overlay_ids_by_legacy: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for line_number, row in enumerate(overlay_rows, 1):
        row_context = f"{context}.stable_id_overlay:{line_number}"
        edit_id = row.get("edit_id")
        if not isinstance(edit_id, str):
            raise DataArtifactError(f"{row_context}: missing stable edit ID")
        if edit_id in overlay_by_id:
            raise DataArtifactError(f"{context}: duplicate overlay edit ID {edit_id}")
        job = expected.get(edit_id)
        if job is None:
            raise DataArtifactError(f"{row_context}: unknown stable edit ID {edit_id}")
        if not _valid_raw_verdict(row):
            raise DataArtifactError(f"{row_context}: invalid overlay verdict")
        _validate_raw_judge_identity(
            row, job, backend=backend, context=row_context
        )
        overlay_by_id[edit_id] = row
        overlay_ids_by_legacy[id_to_legacy[edit_id]].add(edit_id)

    for key, observed_ids in overlay_ids_by_legacy.items():
        expected_ids = {str(job["edit_id"]) for job in expected_by_legacy[key]}
        if observed_ids != expected_ids:
            raise DataArtifactError(
                f"{context}: stable-ID overlay only partially covers legacy group"
            )

    base_by_legacy: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    base_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for line_number, row in enumerate(base_rows, 1):
        row_context = f"{context}.base:{line_number}"
        if row.get("judge") != backend:
            raise DataArtifactError(f"{row_context}: raw judge backend differs from manifest")
        edit_id = row.get("edit_id")
        if edit_id:
            if not isinstance(edit_id, str) or edit_id not in expected:
                raise DataArtifactError(f"{row_context}: unknown stable edit ID {edit_id!r}")
            _validate_raw_judge_identity(
                row, expected[edit_id], backend=backend, context=row_context
            )
            base_by_id[edit_id].append(row)
            continue
        key = _legacy_judge_key(row, row_context)
        if key not in expected_by_legacy:
            raise DataArtifactError(
                f"{row_context}: legacy identity is outside released judge jobs"
            )
        base_by_legacy[key].append(row)

    chosen: dict[str, Mapping[str, Any]] = {}
    replaced_base_rows = 0
    identical_retry_rows = 0
    for key, jobs in expected_by_legacy.items():
        expected_ids = [str(job["edit_id"]) for job in jobs]
        if key in overlay_ids_by_legacy:
            for edit_id in expected_ids:
                chosen[edit_id] = overlay_by_id[edit_id]
            replaced_base_rows += len(base_by_legacy.get(key, ()))
            replaced_base_rows += sum(
                len(base_by_id.get(edit_id, ())) for edit_id in expected_ids
            )
            continue

        stable_rows = {
            edit_id: base_by_id[edit_id]
            for edit_id in expected_ids
            if base_by_id.get(edit_id)
        }
        legacy_rows = base_by_legacy.get(key, [])
        if stable_rows:
            if legacy_rows:
                raise DataArtifactError(
                    f"{context}: base mixes stable and legacy rows in one group"
                )
            if set(stable_rows) != set(expected_ids):
                raise DataArtifactError(
                    f"{context}: stable-ID base only partially covers a legacy group"
                )
            for edit_id, rows in stable_rows.items():
                if len(rows) != 1:
                    raise DataArtifactError(f"{context}: duplicate base edit ID {edit_id}")
                if not _valid_raw_verdict(rows[0]):
                    raise DataArtifactError(
                        f"{context}: invalid uncovered base verdict {edit_id}"
                    )
                chosen[edit_id] = rows[0]
            continue

        if any(not _valid_raw_verdict(row) for row in legacy_rows):
            raise DataArtifactError(
                f"{context}: invalid uncovered legacy judge verdict"
            )
        if len(legacy_rows) < len(jobs):
            raise DataArtifactError(
                f"{context}: legacy judge verdict coverage is incomplete"
            )
        signatures = {
            (row["span_quality"], row["label_quality"]) for row in legacy_rows
        }
        if len(signatures) != 1:
            raise DataArtifactError(
                f"{context}: ambiguous legacy verdict group is not uniform"
            )
        for job in jobs:
            chosen[str(job["edit_id"])] = legacy_rows[0]
        identical_retry_rows += len(legacy_rows) - len(jobs)

    if set(chosen) != set(expected):
        raise DataArtifactError(f"{context}: materialized raw verdict coverage is incomplete")
    return chosen, {
        "expected_jobs": len(expected),
        "base_rows": len(base_rows),
        "overlay_rows": len(overlay_rows),
        "base_rows_replaced_by_overlay": replaced_base_rows,
        "identical_retry_rows_excluded": identical_retry_rows,
    }


def _validate_judges(
    root: Path,
    manifest: Mapping[str, Any],
    expected_jobs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> int:
    judge_manifest = _require_mapping(manifest.get("judges"), "judges")
    if set(judge_manifest) != {"primary", "second"}:
        raise DataArtifactError("judges: expected exactly primary and second")

    canonical_ids: dict[str, set[str]] = {}
    total = 0
    for judge_name in ("primary", "second"):
        judge_context = f"judges.{judge_name}"
        judge_meta = _require_mapping(judge_manifest[judge_name], judge_context)
        if judge_meta.get("model_name") != JUDGE_CONTRACTS[judge_name]["model_name"]:
            raise DataArtifactError(f"{judge_context}: model name differs from protocol")
        backend = judge_meta.get("recorded_backend_id")
        if backend != JUDGE_CONTRACTS[judge_name]["recorded_backend_id"]:
            raise DataArtifactError(f"{judge_context}: backend differs from protocol")
        datasets = _require_mapping(judge_meta.get("datasets"), f"{judge_context}.datasets")
        if set(datasets) != set(expected_jobs):
            raise DataArtifactError(f"{judge_context}: dataset coverage is incomplete")
        all_ids: set[str] = set()
        for dataset_id, expected in expected_jobs.items():
            context = f"{judge_context}.datasets.{dataset_id}"
            metadata = _require_mapping(datasets[dataset_id], context)
            canonical_meta = _require_mapping(metadata.get("canonical"), f"{context}.canonical")
            path = _verify_file_metadata(root, canonical_meta, f"{context}.canonical")
            rows = _read_jsonl(path, f"{context}.canonical")
            declared = _require_int(canonical_meta.get("edits"), f"{context}.canonical.edits")
            if len(rows) != declared or len(rows) != len(expected):
                raise DataArtifactError(f"{context}: canonical judge coverage is incomplete")
            if canonical_meta.get("contains_free_text_rationales") is not False:
                raise DataArtifactError(
                    f"{context}: canonical rows must not claim legacy rationale assignment"
                )
            observed: set[str] = set()
            canonical_by_id: dict[str, Mapping[str, Any]] = {}
            for line_number, row in enumerate(rows, 1):
                edit_id = row.get("edit_id")
                if not isinstance(edit_id, str) or edit_id not in expected:
                    raise DataArtifactError(
                        f"{context}:{line_number}: unknown stable edit ID {edit_id!r}"
                    )
                if edit_id in observed:
                    raise DataArtifactError(f"{context}: duplicate stable edit ID {edit_id}")
                observed.add(edit_id)
                canonical_by_id[edit_id] = row
                job = expected[edit_id]
                exact_fields = {
                    "row_idx": job["row_idx"],
                    "system": job["system"],
                    "edit_idx": job["edit_idx"],
                    "label": job["label"],
                    "edit_start": job["start"],
                    "edit_end": job["end"],
                    "src": job["source"],
                    "tgt": job["target"],
                    "edit_diff": job["edit_diff"],
                    "lang": job["lang"],
                    "judge": backend,
                }
                for field, expected_value in exact_fields.items():
                    if row.get(field) != expected_value:
                        raise DataArtifactError(
                            f"{context}:{line_number}: {field} does not match released edit"
                        )
                if row.get("span_quality") not in QUALITY_LEVELS:
                    raise DataArtifactError(
                        f"{context}:{line_number}: invalid span-quality verdict"
                    )
                if row.get("label_quality") not in QUALITY_LEVELS:
                    raise DataArtifactError(
                        f"{context}:{line_number}: invalid label-quality verdict"
                    )
                if row.get("errored") is not False:
                    raise DataArtifactError(
                        f"{context}:{line_number}: final canonical verdict is marked errored"
                    )
            if observed != set(expected):
                raise DataArtifactError(f"{context}: stable edit ID coverage is incomplete")
            all_ids.update(observed)

            raw = _require_mapping(metadata.get("raw"), f"{context}.raw")
            if set(raw) not in ({"base"}, {"base", "stable_id_overlay"}):
                raise DataArtifactError(
                    f"{context}.raw: expected base and optional stable_id_overlay"
                )
            raw_rows_by_name: dict[str, list[dict[str, Any]]] = {}
            for raw_name, raw_value in raw.items():
                raw_meta = _require_mapping(raw_value, f"{context}.raw.{raw_name}")
                raw_path = _verify_file_metadata(root, raw_meta, f"{context}.raw.{raw_name}")
                raw_rows = _read_jsonl(raw_path, f"{context}.raw.{raw_name}")
                if len(raw_rows) != _require_int(
                    raw_meta.get("rows"), f"{context}.raw.{raw_name}.rows"
                ):
                    raise DataArtifactError(f"{context}.raw.{raw_name}: row count mismatch")
                raw_rows_by_name[raw_name] = raw_rows
            materialized, observed_stats = _materialize_raw_judge_rows(
                expected=expected,
                base_rows=raw_rows_by_name["base"],
                overlay_rows=raw_rows_by_name.get("stable_id_overlay", []),
                backend=backend,
                context=f"{context}.raw",
            )
            for stat_name, observed_value in observed_stats.items():
                if (
                    _require_int(metadata.get(stat_name), f"{context}.{stat_name}")
                    != observed_value
                ):
                    raise DataArtifactError(
                        f"{context}.{stat_name}: differs from raw materialization"
                    )
            for edit_id, canonical_row in canonical_by_id.items():
                raw_row = materialized[edit_id]
                raw_verdict = {
                    "span_quality": raw_row.get("span_quality"),
                    "label_quality": raw_row.get("label_quality"),
                    "judge": raw_row.get("judge"),
                    "errored": bool(raw_row.get("errored", False)),
                }
                canonical_verdict = {
                    field: canonical_row.get(field)
                    for field in ("span_quality", "label_quality", "judge", "errored")
                }
                if canonical_verdict != raw_verdict:
                    raise DataArtifactError(
                        f"{context}: canonical verdict differs from raw materialization "
                        f"for {edit_id}"
                    )
        canonical_ids[judge_name] = all_ids
        total += len(all_ids)
    if canonical_ids["primary"] != canonical_ids["second"]:
        raise DataArtifactError("judges: primary and second stable-edit universes differ")
    return total


def _validate_human(
    root: Path,
    manifest: Mapping[str, Any],
    expected_jobs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[int, int, int]:
    human_manifest = _require_mapping(manifest.get("human_annotations"), "human_annotations")
    release_status_value = manifest.get("release_status")
    if release_status_value is not None:
        release_status = _require_mapping(release_status_value, "release_status")
        if release_status.get("complete_deidentified_human_annotations") is True:
            observed = set(human_manifest)
            if observed != HUMAN_STUDY_DATASETS:
                missing = sorted(HUMAN_STUDY_DATASETS - observed)
                unexpected = sorted(observed - HUMAN_STUDY_DATASETS)
                raise DataArtifactError(
                    "human_annotations: complete deidentified human annotations require "
                    f"exactly {sorted(HUMAN_STUDY_DATASETS)}; missing={missing}, "
                    f"unexpected={unexpected}"
                )
    if not set(human_manifest).issubset(expected_jobs):
        raise DataArtifactError("human_annotations: unknown benchmark")
    assignment_total = 0
    edit_slot_total = 0
    frame_item_total = 0
    for dataset_id, metadata_value in human_manifest.items():
        context = f"human_annotations.{dataset_id}"
        metadata = _require_mapping(metadata_value, context)
        path = _verify_file_metadata(root, metadata, context)
        rows = _read_jsonl(path, context)
        if len(rows) != _require_int(metadata.get("assignments"), f"{context}.assignments"):
            raise DataArtifactError(f"{context}: assignment count does not match manifest")
        expected = expected_jobs[dataset_id]
        frame_meta = _require_mapping(metadata.get("sampling_frame"), f"{context}.sampling_frame")
        frame_path = _verify_file_metadata(root, frame_meta, f"{context}.sampling_frame")
        frame_rows = _read_jsonl(frame_path, f"{context}.sampling_frame")
        if len(frame_rows) != _require_int(
            frame_meta.get("instrument_items"), f"{context}.sampling_frame.instrument_items"
        ):
            raise DataArtifactError(f"{context}: sampling-frame count does not match manifest")
        frame_samples: set[str] = set()
        selected_samples: set[str] = set()
        frame_systems: Counter[str] = Counter()
        selected_systems: Counter[str] = Counter()
        for rank, frame_row in enumerate(frame_rows):
            _require_exact_keys(
                frame_row,
                HUMAN_FRAME_FIELDS,
                f"{context}.sampling_frame:{rank + 1}",
            )
            if frame_row.get("dataset_id") != dataset_id:
                raise DataArtifactError(f"{context}.sampling_frame:{rank + 1}: wrong dataset")
            if frame_row.get("selection_rank") != rank:
                raise DataArtifactError(f"{context}.sampling_frame: ranks are not contiguous")
            sample_id = frame_row.get("sample_id")
            system = frame_row.get("system")
            row_idx = frame_row.get("row_idx")
            expected_sample_id = (
                public_sample_id(dataset_id, row_idx, system)
                if isinstance(system, str)
                else None
            )
            if (
                not isinstance(sample_id, str)
                or not sample_id.startswith(f"{SAMPLE_ID_VERSION}:")
                or sample_id != expected_sample_id
                or sample_id in frame_samples
            ):
                raise DataArtifactError(f"{context}.sampling_frame: duplicate sample ID")
            frame_samples.add(sample_id)
            edits = frame_row.get("edits")
            if not isinstance(edits, list) or frame_row.get("edit_count") != len(edits):
                raise DataArtifactError(f"{context}.sampling_frame/{sample_id}: bad edit count")
            for slot, edit in enumerate(edits, 1):
                if not isinstance(edit, Mapping) or edit.get("edit_id") not in expected:
                    raise DataArtifactError(
                        f"{context}.sampling_frame/{sample_id}: unknown edit ID"
                    )
                _require_exact_keys(
                    edit,
                    HUMAN_FRAME_EDIT_FIELDS,
                    f"{context}.sampling_frame/{sample_id}/edit:{slot}",
                )
                if edit.get("slot") != slot:
                    raise DataArtifactError(
                        f"{context}.sampling_frame/{sample_id}: edit slots are not contiguous"
                    )
                job = expected[str(edit["edit_id"])]
                if (
                    job["row_idx"] != frame_row.get("row_idx")
                    or job["system"] != system
                    or job["edit_idx"] != edit.get("edit_idx")
                    or job["label"] != edit.get("label")
                ):
                    raise DataArtifactError(
                        f"{context}.sampling_frame/{sample_id}: edit identity mismatch"
                    )
            frame_systems[str(system)] += 1
            if frame_row.get("selected_for_human_study") is True:
                selected_samples.add(sample_id)
                selected_systems[str(system)] += 1
            elif frame_row.get("selected_for_human_study") is not False:
                raise DataArtifactError(
                    f"{context}.sampling_frame/{sample_id}: selection flag is not boolean"
                )
        if len(selected_samples) != _require_int(
            frame_meta.get("selected_items"), f"{context}.sampling_frame.selected_items"
        ):
            raise DataArtifactError(f"{context}: selected frame count does not match manifest")
        if dict(sorted(frame_systems.items())) != frame_meta.get("system_item_counts"):
            raise DataArtifactError(f"{context}: frame system counts do not match manifest")
        if dict(sorted(selected_systems.items())) != frame_meta.get("selected_system_item_counts"):
            raise DataArtifactError(f"{context}: selected system counts do not match manifest")
        samples: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        statuses: Counter[str] = Counter()
        systems: Counter[str] = Counter()
        seen_orders: set[int] = set()
        for line_number, row in enumerate(rows, 1):
            _require_exact_keys(
                row, HUMAN_ASSIGNMENT_FIELDS, f"{context}:{line_number}"
            )
            if row.get("dataset_id") != dataset_id:
                raise DataArtifactError(f"{context}:{line_number}: wrong dataset ID")
            sample_id = row.get("sample_id")
            row_idx = row.get("row_idx")
            system = row.get("system")
            expected_sample_id = (
                public_sample_id(dataset_id, row_idx, system)
                if isinstance(system, str)
                else None
            )
            if (
                not isinstance(sample_id, str)
                or not sample_id.startswith(f"{SAMPLE_ID_VERSION}:")
                or sample_id != expected_sample_id
            ):
                raise DataArtifactError(f"{context}:{line_number}: invalid public sample ID")
            pass_id = row.get("pass_id")
            if not isinstance(pass_id, str) or not pass_id.startswith("pass-v1:"):
                raise DataArtifactError(f"{context}:{line_number}: invalid public pass ID")
            order = _require_int(row.get("assignment_order"), f"{context}.assignment_order")
            if order in seen_orders:
                raise DataArtifactError(f"{context}: duplicate assignment order {order}")
            seen_orders.add(order)
            status_value = row.get("status")
            if status_value not in {"scorable", "unannotatable", "incomplete"}:
                raise DataArtifactError(f"{context}:{line_number}: invalid assignment status")
            if not isinstance(row.get("session_unannotatable"), bool):
                raise DataArtifactError(
                    f"{context}:{line_number}: session_unannotatable must be boolean"
                )
            edits = row.get("edits")
            if not isinstance(edits, list) or row.get("edit_count") != len(edits):
                raise DataArtifactError(f"{context}:{line_number}: edits must be a list")
            for slot, edit in enumerate(edits, 1):
                if not isinstance(edit, Mapping):
                    raise DataArtifactError(f"{context}:{line_number}: invalid edit slot")
                _require_exact_keys(
                    edit,
                    HUMAN_ASSIGNMENT_EDIT_FIELDS,
                    f"{context}:{line_number}/edit:{slot}",
                )
                if edit.get("slot") != slot:
                    raise DataArtifactError(
                        f"{context}:{line_number}: edit slots are not contiguous"
                    )
                if not isinstance(edit.get("unannotatable"), bool):
                    raise DataArtifactError(
                        f"{context}:{line_number}: edit unannotatable flag must be boolean"
                    )
                for quality_field in ("span_quality", "label_quality"):
                    quality = edit.get(quality_field)
                    if quality is not None and quality not in QUALITY_LEVELS:
                        raise DataArtifactError(
                            f"{context}:{line_number}: invalid {quality_field}"
                        )
                edit_id = edit.get("edit_id")
                if not isinstance(edit_id, str) or edit_id not in expected:
                    raise DataArtifactError(
                        f"{context}:{line_number}: unknown human edit ID {edit_id!r}"
                    )
                job = expected[edit_id]
                if (
                    job["row_idx"] != row_idx
                    or job["system"] != system
                    or job["edit_idx"] != edit.get("edit_idx")
                    or job["label"] != edit.get("label")
                ):
                    raise DataArtifactError(
                        f"{context}:{line_number}: human edit does not match released edit"
                    )
            samples[sample_id].append(row)
            statuses[str(status_value)] += 1
            systems[str(system)] += 1
            edit_slot_total += len(edits)
        passes_per_sample = _require_int(
            metadata.get("passes_per_sample"), f"{context}.passes_per_sample", minimum=1
        )
        if len(samples) != _require_int(metadata.get("samples"), f"{context}.samples"):
            raise DataArtifactError(f"{context}: sample count does not match manifest")
        for sample_id, assignments in samples.items():
            pass_ids = {row.get("pass_id") for row in assignments}
            if len(assignments) != passes_per_sample or len(pass_ids) != passes_per_sample:
                raise DataArtifactError(
                    f"{context}/{sample_id}: expected {passes_per_sample} distinct passes"
                )
            signatures = {
                (
                    row.get("row_idx"),
                    row.get("system"),
                    tuple(edit.get("edit_id") for edit in row.get("edits", ())),
                )
                for row in assignments
            }
            if len(signatures) != 1:
                raise DataArtifactError(
                    f"{context}/{sample_id}: annotation passes target different edits"
                )
        if set(samples) != selected_samples:
            raise DataArtifactError(
                f"{context}: realized assignments differ from selected sampling-frame items"
            )
        if dict(sorted(statuses.items())) != metadata.get("status_counts"):
            raise DataArtifactError(f"{context}: status counts do not match manifest")
        if dict(sorted(systems.items())) != metadata.get("system_assignment_counts"):
            raise DataArtifactError(f"{context}: system counts do not match manifest")
        assignment_total += len(rows)
        frame_item_total += len(frame_rows)
    return frame_item_total, assignment_total, edit_slot_total


def validate_data_artifact(root: Path) -> DataValidationReport:
    """Validate one extracted complete Data artifact directory."""

    root = root.resolve()
    manifest_path = root / "manifest.full.json"
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataArtifactError(f"cannot read {manifest_path}: {error}") from error
    manifest = _require_mapping(manifest_value, "manifest")
    if manifest.get("schema_version") != "metc-data-v2":
        raise DataArtifactError("manifest: unsupported schema version")
    _validate_protocol_provenance(manifest, Path(__file__).resolve().parents[2])
    _validate_complete_inventory(root, manifest)
    _validate_runtime_resources(root, manifest)
    rows_by_dataset, expected_jobs, benchmark_items, benchmark_edits = _benchmark_rows(
        root, manifest
    )
    teacher_sentences, teacher_edits = _validate_teacher(root, manifest, rows_by_dataset)
    judge_edits = _validate_judges(root, manifest, expected_jobs)
    frame_items, human_assignments, human_edit_slots = _validate_human(
        root, manifest, expected_jobs
    )
    return DataValidationReport(
        benchmark_items=benchmark_items,
        benchmark_edits=benchmark_edits,
        teacher_sentences=teacher_sentences,
        teacher_edits=teacher_edits,
        canonical_judge_edits=judge_edits,
        human_sampling_frame_items=frame_items,
        human_assignments=human_assignments,
        human_edit_slots=human_edit_slots,
    )
