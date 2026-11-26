"""Resumable, execution-first batches for METC's public LLM protocols.

The complete Data artifact remains authoritative.  This module builds fresh
jobs from benchmark annotations, executes the released provider clients, and
stores every attempt below ``fresh_runs``.  Fresh records can be compared with
the frozen records by stable edit ID, but are never substituted for them.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from metc_artifact.adapters.mappings import map_metc
from metc_artifact.judge_jobs import build_judge_jobs, stable_edit_id
from metc_artifact.schema import DatasetSpec, dataset_spec
from metc_artifact.semantic import source_text, target_text

from .clients import (
    Transport,
    build_request_contract,
    run_google_acceptability,
    run_openai_acceptability,
    run_openai_teacher,
    urllib_json_transport,
)
from .protocols import load_teacher_taxonomy
from .records import (
    FreshRunWriter,
    StableComparison,
    compare_with_frozen,
    read_fresh_records,
)


GLOSS_ENTRY_COUNT = 3_616
GLOSS_ENTRIES_SHA256 = "24ec79277ec9a98b9c1c097be432a31e2a9e16e9499d2a1bd4ea05c8b44923ca"
DEFAULT_GLOSS_PATH = (
    Path(__file__).resolve().parents[3] / "evaluation" / "label_glosses.json"
)


class BatchError(ValueError):
    """Raised when batch inputs, resume state, or materialization are unsafe."""


@dataclass(frozen=True)
class TeacherJobBuild:
    jobs: tuple[dict[str, Any], ...]
    edit_count: int
    skipped_overlapping_rows: tuple[int | str, ...]


@dataclass(frozen=True)
class BatchRunSummary:
    run_kind: str
    provider: str
    stream: str
    total_jobs: int
    previously_complete_jobs: int
    attempted_jobs: int
    request_attempts: int
    newly_complete_jobs: int
    failed_jobs: int
    record_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "metc-fresh-batch-v1",
            "source": "fresh_rerun",
            "authoritative_frozen_output": False,
            "run_kind": self.run_kind,
            "provider": self.provider,
            "stream": self.stream,
            "total_jobs": self.total_jobs,
            "previously_complete_jobs": self.previously_complete_jobs,
            "attempted_jobs": self.attempted_jobs,
            "request_attempts": self.request_attempts,
            "newly_complete_jobs": self.newly_complete_jobs,
            "failed_jobs": self.failed_jobs,
            "complete_jobs": self.previously_complete_jobs + self.newly_complete_jobs,
            "record_path": str(self.record_path),
        }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        source = path.open(encoding="utf-8")
    except OSError as error:
        raise BatchError(f"cannot read JSONL input {path}: {error}") from error
    with source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BatchError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise BatchError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def load_label_glosses(
    path: Path = DEFAULT_GLOSS_PATH,
) -> dict[tuple[str, str, str], str]:
    """Load and checksum the exact gloss inventory used by the paper judges."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchError(f"cannot load released label glosses from {path}: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise BatchError(f"unsupported label-gloss inventory: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise BatchError(f"label-gloss inventory has no entries: {path}")
    observed_digest = _sha256_json(entries)
    if (
        len(entries) != GLOSS_ENTRY_COUNT
        or payload.get("entry_count") != GLOSS_ENTRY_COUNT
        or payload.get("entries_sha256") != GLOSS_ENTRIES_SHA256
        or observed_digest != GLOSS_ENTRIES_SHA256
    ):
        raise BatchError(
            "released label-gloss inventory differs from the paper-pinned "
            f"{GLOSS_ENTRY_COUNT}/{GLOSS_ENTRIES_SHA256} contract"
        )
    glosses: dict[tuple[str, str, str], str] = {}
    for position, entry in enumerate(entries, 1):
        if not isinstance(entry, Mapping):
            raise BatchError(f"label-gloss entry {position} is not an object")
        key_values = (entry.get("dataset"), entry.get("system"), entry.get("label"))
        description = entry.get("description")
        if not all(isinstance(value, str) and value for value in key_values):
            raise BatchError(f"label-gloss entry {position} has an invalid key")
        if not isinstance(description, str) or not description:
            raise BatchError(f"label-gloss entry {position} has no description")
        key = (str(key_values[0]), str(key_values[1]), str(key_values[2]))
        if key in glosses:
            raise BatchError(f"duplicate label-gloss key {key!r}")
        glosses[key] = description
    return glosses


def build_acceptability_jobs(
    rows: Iterable[Mapping[str, Any]],
    spec: DatasetSpec,
    *,
    gloss_path: Path = DEFAULT_GLOSS_PATH,
) -> tuple[dict[str, Any], ...]:
    """Build the exact shared job universe used by both acceptability judges."""

    glosses = load_label_glosses(gloss_path)
    build = build_judge_jobs(rows, spec)
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_job in build.jobs:
        job = dict(raw_job)
        key = (spec.experiment_code, str(job["system"]), str(job["label"]))
        try:
            job["label_gloss"] = glosses[key]
        except KeyError as error:
            raise BatchError(f"released gloss inventory has no entry for {key!r}") from error
        job["dataset_id"] = spec.dataset_id
        edit_id = str(job["edit_id"])
        if edit_id in seen:
            raise BatchError(f"duplicate acceptability job ID {edit_id}")
        seen.add(edit_id)
        jobs.append(job)
    return tuple(jobs)


def build_acceptability_jobs_from_file(
    benchmark_path: Path,
    spec: DatasetSpec,
    *,
    gloss_path: Path = DEFAULT_GLOSS_PATH,
) -> tuple[dict[str, Any], ...]:
    return build_acceptability_jobs(
        _read_jsonl(benchmark_path),
        spec,
        gloss_path=gloss_path,
    )


def _exact_int(value: Any, context: str) -> int:
    if type(value) is not int:
        raise BatchError(f"{context} must be an exact integer")
    return value


def _row_index(row: Mapping[str, Any], spec: DatasetSpec) -> int | str:
    value = row.get("idx", row.get("sent_idx"))
    if not isinstance(value, (int, str)) or isinstance(value, bool):
        raise BatchError(f"{spec.dataset_id}: row lacks a stable integer/string index")
    return value


def _teacher_edit(
    row: Mapping[str, Any],
    spec: DatasetSpec,
    raw_edit: Any,
    edit_idx: int,
) -> dict[str, Any]:
    row_idx = _row_index(row, spec)
    source = source_text(row, spec)
    if spec.edit_format == "char_tuple":
        if not isinstance(raw_edit, (list, tuple)) or len(raw_edit) != 5:
            raise BatchError(
                f"{spec.dataset_id}/{row_idx}/metc/{edit_idx}: expected a five-field edit"
            )
        start = _exact_int(raw_edit[0], "character edit start")
        end = _exact_int(raw_edit[1], "character edit end")
        replacement = str(raw_edit[4] or "")
        prompt_span = {"unit": "unicode_codepoint", "start": start, "end": end}
        paper_span = dict(prompt_span)
        source_fragment = source[start:end]
    elif spec.dataset_id == "uk_uagec":
        if not isinstance(raw_edit, Mapping):
            raise BatchError(f"{spec.dataset_id}/{row_idx}/metc/{edit_idx}: invalid token edit")
        start = _exact_int(raw_edit.get("orig_start"), "token edit start")
        end = _exact_int(raw_edit.get("orig_end"), "token edit end")
        replacement = raw_edit.get("replacement")
        if replacement is None:
            replacement = ""
        if not isinstance(replacement, str):
            raise BatchError("token edit replacement must be a string")
        tokens = [str(token) for token in row.get("orig_tokens", ())]
        source_fragment = " ".join(tokens[start:end])
        prompt_span = {"unit": "token", "start": start, "end": end}
        paper_span = dict(prompt_span)
    else:
        if not isinstance(raw_edit, Mapping):
            raise BatchError(f"{spec.dataset_id}/{row_idx}/metc/{edit_idx}: invalid token edit")
        paper_start = _exact_int(raw_edit.get("orig_start"), "paper token edit start")
        paper_end = _exact_int(raw_edit.get("orig_end"), "paper token edit end")
        start = _exact_int(raw_edit.get("char_start"), "teacher character edit start")
        end = _exact_int(raw_edit.get("char_end"), "teacher character edit end")
        replacement = raw_edit.get("char_repl")
        if not isinstance(replacement, str):
            raise BatchError(
                f"{spec.dataset_id}/{row_idx}/metc/{edit_idx}: fresh METC edit lacks char_repl"
            )
        source_fragment = source[start:end]
        recorded_fragment = raw_edit.get("char_src")
        if recorded_fragment is not None and recorded_fragment != source_fragment:
            raise BatchError(
                f"{spec.dataset_id}/{row_idx}/metc/{edit_idx}: char_src differs from fixed span"
            )
        prompt_span = {"unit": "unicode_codepoint", "start": start, "end": end}
        paper_span = {"unit": "token", "start": paper_start, "end": paper_end}
    if start < 0 or end < start:
        raise BatchError(f"{spec.dataset_id}/{row_idx}/metc/{edit_idx}: invalid fixed span")
    span_limit = len(row.get("orig_tokens", ())) if prompt_span["unit"] == "token" else len(source)
    if end > span_limit:
        raise BatchError(f"{spec.dataset_id}/{row_idx}/metc/{edit_idx}: span exceeds source")
    return {
        "edit_id": stable_edit_id(spec.experiment_code, row_idx, "metc", edit_idx),
        "edit_idx": edit_idx,
        "source": source_fragment,
        "target": replacement,
        "span": prompt_span,
        "paper_span": paper_span,
    }


def _numbered_sentence(
    row: Mapping[str, Any],
    spec: DatasetSpec,
    edits: Sequence[Mapping[str, Any]],
) -> str | None:
    if spec.dataset_id == "uk_uagec":
        tokens = [str(token) for token in row.get("orig_tokens", ())]
        output: list[str] = []
        previous = 0
        for position, edit in enumerate(edits, 1):
            span = edit["span"]
            start, end = int(span["start"]), int(span["end"])
            if start < previous:
                return None
            output.extend(tokens[previous:start])
            output.append(
                f"{{#{position} {' '.join(tokens[start:end])}=>{edit['target']}}}"
            )
            previous = end
        output.extend(tokens[previous:])
        return " ".join(output)

    source = source_text(row, spec)
    parts: list[str] = []
    previous = 0
    for position, edit in enumerate(edits, 1):
        span = edit["span"]
        start, end = int(span["start"]), int(span["end"])
        if start < previous:
            return None
        parts.append(source[previous:start])
        parts.append(f"{{#{position} {source[start:end]}=>{edit['target']}}}")
        previous = end
    parts.append(source[previous:])
    return "".join(parts)


def build_teacher_jobs(
    rows: Iterable[Mapping[str, Any]], spec: DatasetSpec
) -> TeacherJobBuild:
    """Turn fresh METC edits into exact one-call-per-sentence teacher jobs."""

    jobs: list[dict[str, Any]] = []
    skipped: list[int | str] = []
    seen_edit_ids: set[str] = set()
    edit_count = 0
    for row in rows:
        row_idx = _row_index(row, spec)
        raw_edits = row.get("metc")
        if not isinstance(raw_edits, list):
            raise BatchError(f"{spec.dataset_id}/{row_idx}: metc annotations must be a list")
        if not raw_edits:
            continue
        edits = [
            _teacher_edit(row, spec, raw_edit, edit_idx)
            for edit_idx, raw_edit in enumerate(raw_edits)
        ]
        edits.sort(
            key=lambda edit: (
                int(edit["span"]["start"]),
                int(edit["span"]["end"]),
                int(edit["edit_idx"]),
            )
        )
        numbered = _numbered_sentence(row, spec, edits)
        if numbered is None:
            skipped.append(row_idx)
            continue
        for edit in edits:
            edit_id = str(edit["edit_id"])
            if edit_id in seen_edit_ids:
                raise BatchError(f"duplicate teacher stable edit ID {edit_id}")
            seen_edit_ids.add(edit_id)
        job = {
            "job_id": f"metc-teacher-v1:{spec.experiment_code}:{row_idx}",
            "sentence_id": f"{spec.experiment_code}:{row_idx}",
            "dataset_id": spec.dataset_id,
            "row_idx": row_idx,
            "src": source_text(row, spec),
            "tgt": target_text(row, spec),
            "numbered": numbered,
            "edits": edits,
        }
        jobs.append(job)
        edit_count += len(edits)
    return TeacherJobBuild(tuple(jobs), edit_count, tuple(skipped))


def build_teacher_jobs_from_file(
    benchmark_path: Path, spec: DatasetSpec
) -> TeacherJobBuild:
    return build_teacher_jobs(_read_jsonl(benchmark_path), spec)


def _request_fingerprint(
    job: Mapping[str, Any],
    *,
    run_kind: str,
    provider: str,
    taxonomy: str | None,
) -> str:
    return _sha256_json(
        build_request_contract(
            job,
            run_kind=run_kind,
            provider=provider,
            taxonomy=taxonomy,
        )
    )


def _job_ids(job: Mapping[str, Any], run_kind: str) -> tuple[str, ...]:
    if run_kind == "acceptability":
        edit_id = job.get("edit_id")
        if not isinstance(edit_id, str):
            raise BatchError("acceptability job lacks edit_id")
        return (edit_id,)
    edits = job.get("edits")
    if not isinstance(edits, Sequence) or isinstance(edits, (str, bytes)):
        raise BatchError("teacher job lacks edits")
    edit_ids = tuple(
        str(edit["edit_id"])
        for edit in edits
        if isinstance(edit, Mapping) and isinstance(edit.get("edit_id"), str)
    )
    if len(edit_ids) != len(edits):
        raise BatchError("teacher job contains an invalid edit ID")
    return edit_ids


def _resume_state(
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, set[str]],
    *,
    run_kind: str,
    provider: str,
) -> tuple[set[str], dict[str, int]]:
    expected_ids = set().union(*expected.values()) if expected else set()
    complete: set[str] = set()
    attempts = {fingerprint: 0 for fingerprint in expected}
    for record_number, record in enumerate(records, 1):
        if record.get("run_kind") != run_kind or record.get("provider") != provider:
            raise BatchError(
                f"resume record {record_number} mixes another run kind/provider into the stream"
            )
        record_ids = set(record.get("stable_edit_ids", ()))
        if not record_ids <= expected_ids:
            raise BatchError(f"resume record {record_number} contains IDs outside current jobs")
        batch = record.get("batch")
        fingerprint = (
            batch.get("request_sha256") if isinstance(batch, Mapping) else None
        )
        if not isinstance(fingerprint, str) or fingerprint not in expected:
            raise BatchError(
                f"resume record {record_number} was built from different or unpinned job input"
            )
        if record_ids != expected[fingerprint]:
            raise BatchError(f"resume record {record_number} stable-ID set differs from its job")
        attempts[fingerprint] += 1
        if record.get("error") is None:
            outputs = record.get("outputs_by_edit_id")
            if not isinstance(outputs, Mapping) or set(outputs) != record_ids:
                raise BatchError(f"successful resume record {record_number} is incomplete")
            if fingerprint in complete:
                raise BatchError(f"resume stream contains duplicate successful job {fingerprint}")
            complete.add(fingerprint)
    return complete, attempts


def _credential(provider: str, explicit: str | None) -> str:
    environment = "OPENAI_API_KEY" if provider == "openai" else "GOOGLE_API_KEY"
    key = explicit or os.environ.get(environment)
    if not key:
        raise BatchError(f"{environment} is required for this fresh rerun")
    return key


def _run_batch(
    jobs: Sequence[Mapping[str, Any]],
    *,
    run_kind: str,
    provider: str,
    writer: FreshRunWriter,
    stream: str,
    runner: Callable[[Mapping[str, Any]], dict[str, Any]],
    max_attempts: int,
    workers: int,
    retry_delay_seconds: float,
    sleeper: Callable[[float], None],
    taxonomy: str | None = None,
) -> BatchRunSummary:
    if max_attempts < 1 or workers < 1 or retry_delay_seconds < 0:
        raise BatchError("max_attempts/workers must be positive and retry delay nonnegative")
    prepared = [
        (_request_fingerprint(job, run_kind=run_kind, provider=provider, taxonomy=taxonomy), job)
        for job in jobs
    ]
    expected = {fingerprint: set(_job_ids(job, run_kind)) for fingerprint, job in prepared}
    if len(expected) != len(prepared):
        raise BatchError("batch contains duplicate job payloads")
    path = writer.path_for(stream)
    previous_records = read_fresh_records(path) if path.is_file() else []
    complete, prior_attempts = _resume_state(
        previous_records, expected, run_kind=run_kind, provider=provider
    )
    pending = [entry for entry in prepared if entry[0] not in complete]

    def execute(fingerprint: str, job: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        job_sha256 = _sha256_json(job)
        records: list[dict[str, Any]] = []
        for local_attempt in range(1, max_attempts + 1):
            record = runner(job)
            record["batch"] = {
                "schema_version": "metc-fresh-batch-v1",
                "job_sha256": job_sha256,
                "request_sha256": fingerprint,
                "attempt": prior_attempts[fingerprint] + local_attempt,
                "max_attempts_this_invocation": max_attempts,
            }
            records.append(record)
            if record.get("error") is None:
                return records, True
            if local_attempt < max_attempts and retry_delay_seconds:
                sleeper(min(8.0, retry_delay_seconds * (1.5 ** (local_attempt - 1))))
        return records, False

    newly_complete = 0
    failed = 0
    request_attempts = 0

    def persist(outcome: tuple[list[dict[str, Any]], bool]) -> None:
        nonlocal newly_complete, failed, request_attempts
        records, succeeded = outcome
        for record in records:
            writer.append(stream, record)
            request_attempts += 1
        if succeeded:
            newly_complete += 1
        else:
            failed += 1

    if workers == 1:
        for fingerprint, job in pending:
            persist(execute(fingerprint, job))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Bound queued futures: a full judge run has 147,926 jobs, so
            # submitting the entire universe at once needlessly consumes large
            # amounts of memory before the first response is durable.
            window = workers * 4
            for offset in range(0, len(pending), window):
                futures = [
                    executor.submit(execute, fingerprint, job)
                    for fingerprint, job in pending[offset : offset + window]
                ]
                for future in as_completed(futures):
                    # Persist each completed job immediately so an interrupted
                    # run resumes from durable responses rather than restarting.
                    persist(future.result())
    return BatchRunSummary(
        run_kind=run_kind,
        provider=provider,
        stream=stream,
        total_jobs=len(jobs),
        previously_complete_jobs=len(complete),
        attempted_jobs=len(pending),
        request_attempts=request_attempts,
        newly_complete_jobs=newly_complete,
        failed_jobs=failed,
        record_path=path,
    )


def run_acceptability_batch(
    jobs: Sequence[Mapping[str, Any]],
    *,
    provider: str,
    writer: FreshRunWriter,
    stream: str,
    api_key: str | None = None,
    transport: Transport = urllib_json_transport,
    timeout: float = 180.0,
    max_attempts: int = 4,
    workers: int = 4,
    retry_delay_seconds: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> BatchRunSummary:
    """Run/resume one fresh acceptability judge without touching frozen output."""

    aliases = {"primary": "openai", "openai": "openai", "second": "google", "google": "google"}
    try:
        normalized = aliases[provider]
    except KeyError as error:
        raise BatchError("acceptability provider must be primary/openai or second/google") from error
    key = _credential(normalized, api_key)
    if normalized == "openai":
        def runner(job: Mapping[str, Any]) -> dict[str, Any]:
            return run_openai_acceptability(
                job,
                api_key=key,
                transport=transport,
                timeout=timeout,
            )
    else:
        def runner(job: Mapping[str, Any]) -> dict[str, Any]:
            return run_google_acceptability(
                job,
                api_key=key,
                transport=transport,
                timeout=timeout,
            )
    return _run_batch(
        jobs,
        run_kind="acceptability",
        provider=normalized,
        writer=writer,
        stream=stream,
        runner=runner,
        max_attempts=max_attempts,
        workers=workers,
        retry_delay_seconds=retry_delay_seconds,
        sleeper=sleeper,
    )


def run_teacher_batch(
    jobs: Sequence[Mapping[str, Any]],
    *,
    writer: FreshRunWriter,
    stream: str,
    api_key: str | None = None,
    taxonomy_path: Path | None = None,
    taxonomy: str | None = None,
    transport: Transport = urllib_json_transport,
    timeout: float = 180.0,
    max_attempts: int = 4,
    workers: int = 4,
    retry_delay_seconds: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> BatchRunSummary:
    """Run/resume the fresh OpenAI teacher over immutable METC edits."""

    key = _credential("openai", api_key)
    if jobs and taxonomy is None:
        taxonomy = load_teacher_taxonomy(taxonomy_path)
    return _run_batch(
        jobs,
        run_kind="annotation_teacher",
        provider="openai",
        writer=writer,
        stream=stream,
        runner=lambda job: run_openai_teacher(
            job,
            api_key=key,
            taxonomy=taxonomy,
            transport=transport,
            timeout=timeout,
        ),
        max_attempts=max_attempts,
        workers=workers,
        retry_delay_seconds=retry_delay_seconds,
        sleeper=sleeper,
        taxonomy=taxonomy,
    )


def compare_acceptability_with_frozen(
    fresh_path: Path,
    frozen_path: Path,
    *,
    max_mismatches: int = 100,
) -> StableComparison:
    return compare_with_frozen(
        read_fresh_records(fresh_path),
        _read_jsonl(frozen_path),
        fields=("span_quality", "label_quality"),
        max_mismatches=max_mismatches,
    )


def frozen_teacher_outputs(
    rows: Iterable[Mapping[str, Any]], spec: DatasetSpec
) -> dict[str, dict[str, str]]:
    """Index frozen teacher labels by the stable IDs of their fixed METC edits."""

    outputs: dict[str, dict[str, str]] = {}
    for row in rows:
        row_idx = _row_index(row, spec)
        student = row.get("metc")
        teacher = row.get("metc_teacher")
        if not isinstance(student, list) or not isinstance(teacher, list):
            raise BatchError(f"{spec.dataset_id}/{row_idx}: missing METC/teacher edit arrays")
        if len(student) != len(teacher):
            raise BatchError(f"{spec.dataset_id}/{row_idx}: teacher/student edit counts differ")
        for edit_idx, raw_edit in enumerate(teacher):
            if spec.edit_format == "char_tuple":
                if not isinstance(raw_edit, (list, tuple)) or len(raw_edit) != 5:
                    raise BatchError(f"{spec.dataset_id}/{row_idx}: malformed teacher edit")
                raw_label = str(raw_edit[3])
            else:
                if not isinstance(raw_edit, Mapping):
                    raise BatchError(f"{spec.dataset_id}/{row_idx}: malformed teacher edit")
                raw_label = str(raw_edit.get("raw_label", ""))
            label, separator, sublabel = raw_label.partition("/")
            if not separator or not label or not sublabel:
                raise BatchError(f"{spec.dataset_id}/{row_idx}: invalid teacher label {raw_label!r}")
            edit_id = stable_edit_id(spec.experiment_code, row_idx, "metc", edit_idx)
            outputs[edit_id] = {
                "label": label,
                "sublabel": sublabel,
                "raw_label": raw_label,
            }
    return outputs


def compare_teacher_with_frozen(
    fresh_path: Path,
    frozen_benchmark_path: Path,
    spec: DatasetSpec,
    *,
    max_mismatches: int = 100,
) -> StableComparison:
    return compare_with_frozen(
        read_fresh_records(fresh_path),
        frozen_teacher_outputs(_read_jsonl(frozen_benchmark_path), spec),
        fields=("raw_label",),
        max_mismatches=max_mismatches,
    )


def _matching_teacher_outputs(
    records: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    *,
    taxonomy: str | None,
) -> dict[str, Mapping[str, Any]]:
    expected = {
        _request_fingerprint(
            job,
            run_kind="annotation_teacher",
            provider="openai",
            taxonomy=taxonomy,
        ): set(_job_ids(job, "annotation_teacher"))
        for job in jobs
    }
    outputs: dict[str, Mapping[str, Any]] = {}
    successful_jobs: set[str] = set()
    for record_number, record in enumerate(records, 1):
        if record.get("run_kind") != "annotation_teacher" or record.get("provider") != "openai":
            raise BatchError(f"teacher record {record_number} has the wrong protocol/provider")
        batch = record.get("batch")
        fingerprint = (
            batch.get("request_sha256") if isinstance(batch, Mapping) else None
        )
        if not isinstance(fingerprint, str) or fingerprint not in expected:
            raise BatchError(f"teacher record {record_number} does not match current METC edits")
        if record.get("error") is not None:
            continue
        if fingerprint in successful_jobs:
            raise BatchError(f"duplicate successful teacher record for job {fingerprint}")
        record_outputs = record.get("outputs_by_edit_id")
        if not isinstance(record_outputs, Mapping) or set(record_outputs) != expected[fingerprint]:
            raise BatchError(f"successful teacher record {record_number} is incomplete")
        successful_jobs.add(fingerprint)
        for edit_id, output in record_outputs.items():
            if edit_id in outputs or not isinstance(output, Mapping):
                raise BatchError(f"duplicate or invalid teacher output for {edit_id}")
            outputs[str(edit_id)] = output
    missing_jobs = set(expected) - successful_jobs
    if missing_jobs:
        raise BatchError(f"teacher rerun is incomplete: {len(missing_jobs)} sentence jobs missing")
    return outputs


def materialize_teacher_annotations(
    rows: Iterable[Mapping[str, Any]],
    spec: DatasetSpec,
    fresh_records: Sequence[Mapping[str, Any]],
    *,
    taxonomy_path: Path | None = None,
    taxonomy: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Apply fresh labels to copies of the exact fixed METC spans/replacements."""

    original = [dict(row) for row in rows]
    build = build_teacher_jobs(original, spec)
    if build.skipped_overlapping_rows:
        raise BatchError(
            "cannot materialize teacher labels because fixed METC edits overlap on "
            f"{len(build.skipped_overlapping_rows)} rows"
        )
    if build.jobs and taxonomy is None:
        taxonomy = load_teacher_taxonomy(taxonomy_path)
    outputs = _matching_teacher_outputs(fresh_records, build.jobs, taxonomy=taxonomy)
    materialized: list[dict[str, Any]] = []
    for row in original:
        result = copy.deepcopy(row)
        row_idx = _row_index(row, spec)
        student = row.get("metc")
        if not isinstance(student, list):
            raise BatchError(f"{spec.dataset_id}/{row_idx}: metc annotations must be a list")
        teacher: list[Any] = []
        for edit_idx, raw_edit in enumerate(student):
            edit_id = stable_edit_id(spec.experiment_code, row_idx, "metc", edit_idx)
            try:
                output = outputs[edit_id]
                raw_label = str(output["raw_label"])
            except (KeyError, TypeError) as error:
                raise BatchError(f"missing validated teacher output for {edit_id}") from error
            mapped = map_metc(spec.dataset_id, raw_label)
            if spec.edit_format == "char_tuple":
                teacher.append(
                    [raw_edit[0], raw_edit[1], mapped, raw_label, raw_edit[4]]
                )
            else:
                teacher_edit = copy.deepcopy(dict(raw_edit))
                teacher_edit["raw_label"] = raw_label
                teacher_edit["target"] = mapped
                teacher.append(teacher_edit)
        result["metc_teacher"] = teacher
        materialized.append(result)
    return tuple(materialized)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def dataset_input(name: str) -> argparse.ArgumentParser:
        command = subparsers.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--benchmark", required=True, type=Path)
        return command

    build_acceptability = dataset_input("build-acceptability")
    build_acceptability.add_argument("--glosses", type=Path, default=DEFAULT_GLOSS_PATH)
    build_acceptability.add_argument("--output", required=True, type=Path)

    run_acceptability = dataset_input("run-acceptability")
    run_acceptability.add_argument(
        "--provider", choices=("primary", "second", "both"), default="both"
    )
    run_acceptability.add_argument("--glosses", type=Path, default=DEFAULT_GLOSS_PATH)
    run_acceptability.add_argument("--output-root", required=True, type=Path)
    run_acceptability.add_argument("--run-id", required=True)
    run_acceptability.add_argument("--workers", type=int, default=4)
    run_acceptability.add_argument("--max-attempts", type=int, default=4)
    run_acceptability.add_argument("--limit", type=int)

    build_teacher = dataset_input("build-teacher")
    build_teacher.add_argument("--output", required=True, type=Path)

    run_teacher = dataset_input("run-teacher")
    run_teacher.add_argument("--output-root", required=True, type=Path)
    run_teacher.add_argument("--run-id", required=True)
    run_teacher.add_argument("--taxonomy", type=Path)
    run_teacher.add_argument("--workers", type=int, default=4)
    run_teacher.add_argument("--max-attempts", type=int, default=4)
    run_teacher.add_argument("--limit", type=int)

    materialize = dataset_input("materialize-teacher")
    materialize.add_argument("--fresh-records", required=True, type=Path)
    materialize.add_argument("--output", required=True, type=Path)
    materialize.add_argument("--taxonomy", type=Path)

    compare_acceptability = subparsers.add_parser("compare-acceptability")
    compare_acceptability.add_argument("--fresh-records", required=True, type=Path)
    compare_acceptability.add_argument("--frozen-records", required=True, type=Path)

    compare_teacher = dataset_input("compare-teacher")
    compare_teacher.add_argument("--fresh-records", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare-acceptability":
        comparison = compare_acceptability_with_frozen(
            args.fresh_records, args.frozen_records
        )
        print(json.dumps(comparison.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0

    spec = dataset_spec(args.dataset)
    if args.command == "build-acceptability":
        jobs = build_acceptability_jobs_from_file(
            args.benchmark, spec, gloss_path=args.glosses
        )
        _write_jsonl(args.output, jobs)
        print(json.dumps({"dataset_id": spec.dataset_id, "jobs": len(jobs)}, sort_keys=True))
        return 0
    if args.command == "build-teacher":
        build = build_teacher_jobs_from_file(args.benchmark, spec)
        _write_jsonl(args.output, build.jobs)
        print(
            json.dumps(
                {
                    "dataset_id": spec.dataset_id,
                    "jobs": len(build.jobs),
                    "edits": build.edit_count,
                    "skipped_overlapping_rows": list(build.skipped_overlapping_rows),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-acceptability":
        jobs = list(
            build_acceptability_jobs_from_file(
                args.benchmark, spec, gloss_path=args.glosses
            )
        )
        if args.limit is not None:
            jobs = jobs[: args.limit]
        writer = FreshRunWriter(args.output_root, args.run_id)
        providers = ("primary", "second") if args.provider == "both" else (args.provider,)
        summaries = [
            run_acceptability_batch(
                jobs,
                provider=provider,
                writer=writer,
                stream=f"acceptability-{provider}-{spec.dataset_id}",
                workers=args.workers,
                max_attempts=args.max_attempts,
            ).to_dict()
            for provider in providers
        ]
        print(json.dumps(summaries, ensure_ascii=False, sort_keys=True))
        return 0 if all(summary["failed_jobs"] == 0 for summary in summaries) else 2
    if args.command == "run-teacher":
        build = build_teacher_jobs_from_file(args.benchmark, spec)
        jobs = list(build.jobs)
        if args.limit is not None:
            jobs = jobs[: args.limit]
        writer = FreshRunWriter(args.output_root, args.run_id)
        summary = run_teacher_batch(
            jobs,
            writer=writer,
            stream=f"teacher-{spec.dataset_id}",
            taxonomy_path=args.taxonomy,
            workers=args.workers,
            max_attempts=args.max_attempts,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0 if summary.failed_jobs == 0 else 2
    if args.command == "materialize-teacher":
        rows = _read_jsonl(args.benchmark)
        records = read_fresh_records(args.fresh_records)
        materialized = materialize_teacher_annotations(
            rows,
            spec,
            records,
            taxonomy_path=args.taxonomy,
        )
        _write_jsonl(args.output, materialized)
        print(json.dumps({"dataset_id": spec.dataset_id, "rows": len(materialized)}, sort_keys=True))
        return 0
    if args.command == "compare-teacher":
        comparison = compare_teacher_with_frozen(
            args.fresh_records, args.benchmark, spec
        )
        print(json.dumps(comparison.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
