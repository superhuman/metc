"""Fresh-only JSONL storage and stable-edit comparison with frozen outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class FreshRecordError(ValueError):
    """Raised when a rerun record or comparison input violates its schema."""


@dataclass(frozen=True)
class StableComparison:
    frozen_edits: int
    fresh_edits: int
    comparable_edits: int
    missing_fresh: tuple[str, ...]
    extra_fresh: tuple[str, ...]
    errored_records: int
    field_matches: Mapping[str, int]
    field_totals: Mapping[str, int]
    mismatches: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frozen_edits": self.frozen_edits,
            "fresh_edits": self.fresh_edits,
            "comparable_edits": self.comparable_edits,
            "missing_fresh": list(self.missing_fresh),
            "extra_fresh": list(self.extra_fresh),
            "errored_records": self.errored_records,
            "field_matches": dict(self.field_matches),
            "field_totals": dict(self.field_totals),
            "field_agreement": {
                field: (self.field_matches[field] / total if total else None)
                for field, total in self.field_totals.items()
            },
            "mismatches": [dict(row) for row in self.mismatches],
        }


def _validate_fresh_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != "metc-fresh-rerun-v1":
        raise FreshRecordError("unsupported fresh rerun record schema")
    if record.get("source") != "fresh_rerun":
        raise FreshRecordError("refusing to store a non-fresh record")
    if record.get("authoritative_frozen_output") is not False:
        raise FreshRecordError("fresh record must not be marked authoritative")
    edit_ids = record.get("stable_edit_ids")
    if not isinstance(edit_ids, list) or any(
        not isinstance(edit_id, str) or not edit_id.startswith("metc-edit-v1:")
        for edit_id in edit_ids
    ):
        raise FreshRecordError("fresh record has invalid stable edit IDs")
    outputs = record.get("outputs_by_edit_id")
    if not isinstance(outputs, Mapping) or not set(outputs) <= set(edit_ids):
        raise FreshRecordError("fresh outputs are not keyed by declared stable edit IDs")


class FreshRunWriter:
    """Append records below an enforced ``fresh_runs/<run_id>`` namespace."""

    def __init__(self, root: Path, run_id: str):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
            raise FreshRecordError("run_id must be a safe, nonempty path component")
        self.root = root
        self.run_id = run_id
        self.run_root = root / "fresh_runs" / run_id

    def path_for(self, stream: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", stream):
            raise FreshRecordError("stream must be a safe, nonempty path component")
        return self.run_root / f"{stream}.jsonl"

    def append(self, stream: str, record: Mapping[str, Any]) -> Path:
        _validate_fresh_record(record)
        path = self.path_for(stream)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        return path


def read_fresh_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FreshRecordError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise FreshRecordError(f"{path}:{line_number}: expected an object")
            _validate_fresh_record(record)
            records.append(record)
    return records


def _fresh_outputs(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    outputs: dict[str, Mapping[str, Any]] = {}
    errored = 0
    for record in records:
        _validate_fresh_record(record)
        if record.get("error") is not None:
            errored += 1
        for edit_id, output in record["outputs_by_edit_id"].items():
            if edit_id in outputs:
                raise FreshRecordError(f"duplicate fresh output for {edit_id}")
            if not isinstance(output, Mapping):
                raise FreshRecordError(f"fresh output for {edit_id} is not an object")
            outputs[edit_id] = output
    return outputs, errored


def _frozen_outputs(
    rows: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        iterable = rows.items()
    else:
        iterable = ((row.get("edit_id"), row) for row in rows)
    output: dict[str, Mapping[str, Any]] = {}
    for edit_id, row in iterable:
        if not isinstance(edit_id, str) or not edit_id.startswith("metc-edit-v1:"):
            raise FreshRecordError("frozen row lacks a stable edit_id")
        if edit_id in output:
            raise FreshRecordError(f"duplicate frozen output for {edit_id}")
        if not isinstance(row, Mapping):
            raise FreshRecordError(f"frozen output for {edit_id} is not an object")
        output[edit_id] = row
    return output


def compare_with_frozen(
    fresh_records: Iterable[Mapping[str, Any]],
    frozen_rows: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    fields: Sequence[str],
    max_mismatches: int = 100,
) -> StableComparison:
    """Compare fresh and authoritative outputs solely by stable edit ID."""

    if not fields or len(set(fields)) != len(fields):
        raise FreshRecordError("comparison fields must be a nonempty unique sequence")
    if max_mismatches < 0:
        raise FreshRecordError("max_mismatches must be nonnegative")
    fresh, errored = _fresh_outputs(fresh_records)
    frozen = _frozen_outputs(frozen_rows)
    fresh_ids = set(fresh)
    frozen_ids = set(frozen)
    shared = sorted(fresh_ids & frozen_ids)
    matches = {field: 0 for field in fields}
    totals = {field: 0 for field in fields}
    mismatches: list[Mapping[str, Any]] = []
    for edit_id in shared:
        fresh_output = fresh[edit_id]
        frozen_output = frozen[edit_id]
        for field in fields:
            totals[field] += 1
            actual = fresh_output.get(field)
            expected = frozen_output.get(field)
            if actual == expected:
                matches[field] += 1
            elif len(mismatches) < max_mismatches:
                mismatches.append(
                    {
                        "edit_id": edit_id,
                        "field": field,
                        "frozen": expected,
                        "fresh": actual,
                    }
                )
    return StableComparison(
        frozen_edits=len(frozen),
        fresh_edits=len(fresh),
        comparable_edits=len(shared),
        missing_fresh=tuple(sorted(frozen_ids - fresh_ids)),
        extra_fresh=tuple(sorted(fresh_ids - frozen_ids)),
        errored_records=errored,
        field_matches=matches,
        field_totals=totals,
        mismatches=tuple(mismatches),
    )
