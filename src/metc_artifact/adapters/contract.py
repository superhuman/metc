"""Canonical pair, normalized-annotation, and paper-materialization contracts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from metc_artifact.schema import DatasetSpec, dataset_spec
from metc_artifact.semantic import canonical_item_id, source_text, target_text

from .errors import AdapterInputError, AdapterOutputError


def _canonical_json_identity(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_item_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterInputError("item_id must be nonempty canonical JSON text")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise AdapterInputError(f"item_id is not JSON text: {value!r}") from error
    if _canonical_json_identity(decoded) != value:
        raise AdapterInputError(f"item_id is not canonical JSON text: {value!r}")
    return value


def _tokens(value: Any, field: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(token, str) for token in value):
        raise AdapterInputError(f"{field} must be an array of strings")
    return tuple(value)


@dataclass(frozen=True)
class PairRecord:
    dataset_id: str
    item_id: str
    source: str
    correction: str
    source_tokens: tuple[str, ...] | None = None
    correction_tokens: tuple[str, ...] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairRecord":
        try:
            spec = dataset_spec(str(value["dataset_id"]))
        except (KeyError, ValueError) as error:
            raise AdapterInputError(f"invalid dataset_id: {value.get('dataset_id')!r}") from error
        source = value.get("source")
        correction = value.get("correction")
        if not isinstance(source, str) or not isinstance(correction, str):
            raise AdapterInputError("source and correction must be strings")
        record = cls(
            dataset_id=spec.dataset_id,
            item_id=_validate_item_id(value.get("item_id")),
            source=source,
            correction=correction,
            source_tokens=_tokens(value.get("source_tokens"), "source_tokens"),
            correction_tokens=_tokens(value.get("correction_tokens"), "correction_tokens"),
        )
        record.validate_for(spec)
        return record

    def validate_for(self, spec: DatasetSpec) -> None:
        if self.dataset_id != spec.dataset_id:
            raise AdapterInputError(
                f"pair {self.item_id} belongs to {self.dataset_id}, expected {spec.dataset_id}"
            )
        if spec.edit_format == "token_dict":
            if self.source_tokens is None or self.correction_tokens is None:
                raise AdapterInputError(
                    f"{spec.dataset_id}/{self.item_id}: token arrays are required"
                )
            if " ".join(self.source_tokens) != self.source:
                raise AdapterInputError(
                    f"{spec.dataset_id}/{self.item_id}: source_tokens do not reproduce source"
                )
            if " ".join(self.correction_tokens) != self.correction:
                raise AdapterInputError(
                    f"{spec.dataset_id}/{self.item_id}: correction_tokens do not reproduce correction"
                )

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "item_id": self.item_id,
            "source": self.source,
            "correction": self.correction,
        }
        if self.source_tokens is not None:
            output["source_tokens"] = list(self.source_tokens)
        if self.correction_tokens is not None:
            output["correction_tokens"] = list(self.correction_tokens)
        return output


@dataclass(frozen=True, order=True)
class NormalizedEdit:
    start: int
    end: int
    replacement: str
    native_label: str
    mapped_label: str
    char_start: int | None = None
    char_end: int | None = None
    char_source: str | None = None
    char_replacement: str | None = None

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise AdapterOutputError("normalized edit start/end must be exact integers")
        if self.start < 0 or self.end < self.start:
            raise AdapterOutputError(f"invalid normalized span [{self.start}, {self.end})")
        if not all(
            isinstance(value, str)
            for value in (self.replacement, self.native_label, self.mapped_label)
        ):
            raise AdapterOutputError(
                "replacement, native_label, and mapped_label must be strings"
            )
        if (self.char_start is None) != (self.char_end is None):
            raise AdapterOutputError("char_start and char_end must be supplied together")
        if self.char_start is None:
            if self.char_source is not None or self.char_replacement is not None:
                raise AdapterOutputError(
                    "character source/replacement require a character span"
                )
            return
        if type(self.char_start) is not int or type(self.char_end) is not int:
            raise AdapterOutputError("character span boundaries must be exact integers")
        if self.char_start < 0 or self.char_end < self.char_start:
            raise AdapterOutputError(
                f"invalid character span [{self.char_start}, {self.char_end})"
            )
        if not isinstance(self.char_source, str) or not isinstance(
            self.char_replacement, str
        ):
            raise AdapterOutputError(
                "character source/replacement must be strings when a span is present"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NormalizedEdit":
        try:
            start = value["start"]
            end = value["end"]
        except KeyError as error:
            raise AdapterOutputError("normalized edit requires integer start/end") from error
        required_strings = ("replacement", "native_label", "mapped_label")
        if any(not isinstance(value.get(field), str) for field in required_strings):
            raise AdapterOutputError(
                "replacement, native_label, and mapped_label must be strings"
            )
        char_start = value.get("char_start")
        char_end = value.get("char_end")
        return cls(
            start=start,
            end=end,
            replacement=value["replacement"],
            native_label=value["native_label"],
            mapped_label=value["mapped_label"],
            char_start=char_start,
            char_end=char_end,
            char_source=value.get("char_source"),
            char_replacement=value.get("char_replacement"),
        )

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "replacement": self.replacement,
            "native_label": self.native_label,
            "mapped_label": self.mapped_label,
        }
        if self.char_start is not None:
            output.update(
                {
                    "char_start": self.char_start,
                    "char_end": self.char_end,
                    "char_source": self.char_source or "",
                    "char_replacement": self.char_replacement or "",
                }
            )
        return output


@dataclass(frozen=True)
class AnnotationRecord:
    dataset_id: str
    item_id: str
    system: str
    span_unit: str
    edits: tuple[NormalizedEdit, ...]

    def validate_for_pair(self, pair: PairRecord, spec: DatasetSpec) -> None:
        pair.validate_for(spec)
        if self.dataset_id != spec.dataset_id or pair.dataset_id != spec.dataset_id:
            raise AdapterOutputError(f"{pair.item_id}: annotation dataset mismatch")
        if self.item_id != pair.item_id:
            raise AdapterOutputError(
                f"annotation item {self.item_id} does not match pair {pair.item_id}"
            )
        if self.span_unit != spec.evaluation_span_unit:
            raise AdapterOutputError(f"{pair.item_id}: wrong annotation span unit")
        span_limit = (
            len(pair.source_tokens or ())
            if spec.evaluation_span_unit == "token"
            else len(pair.source)
        )
        for edit in self.edits:
            if edit.end > span_limit:
                raise AdapterOutputError(
                    f"{pair.item_id}: edit span [{edit.start}, {edit.end}) exceeds "
                    f"source limit {span_limit}"
                )
            if edit.char_start is not None:
                assert edit.char_end is not None
                if edit.char_end > len(pair.source):
                    raise AdapterOutputError(
                        f"{pair.item_id}: character span exceeds source length"
                    )
                if edit.char_source != pair.source[edit.char_start : edit.char_end]:
                    raise AdapterOutputError(
                        f"{pair.item_id}: character source does not match pair text"
                    )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnnotationRecord":
        try:
            spec = dataset_spec(str(value["dataset_id"]))
        except (KeyError, ValueError) as error:
            raise AdapterOutputError("annotation has invalid dataset_id") from error
        if value.get("span_unit") != spec.evaluation_span_unit:
            raise AdapterOutputError(
                f"{spec.dataset_id}: span_unit must be {spec.evaluation_span_unit!r}"
            )
        system = value.get("system")
        if not isinstance(system, str) or not system:
            raise AdapterOutputError("annotation system must be a nonempty string")
        raw_edits = value.get("edits")
        if not isinstance(raw_edits, list):
            raise AdapterOutputError("annotation edits must be an array")
        return cls(
            dataset_id=spec.dataset_id,
            item_id=_validate_item_id(value.get("item_id")),
            system=system,
            span_unit=spec.evaluation_span_unit,
            edits=tuple(NormalizedEdit.from_dict(edit) for edit in raw_edits),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "item_id": self.item_id,
            "system": self.system,
            "span_unit": self.span_unit,
            "edits": [edit.to_dict() for edit in self.edits],
        }


def pairs_from_paper_rows(
    rows: Iterable[Mapping[str, Any]], spec: DatasetSpec
) -> tuple[PairRecord, ...]:
    pairs: list[PairRecord] = []
    seen: set[str] = set()
    for row in rows:
        item_id = canonical_item_id(row)
        if item_id in seen:
            raise AdapterInputError(f"{spec.dataset_id}: duplicate item_id {item_id}")
        seen.add(item_id)
        source_tokens = (
            tuple(str(token) for token in row.get("orig_tokens", ()))
            if spec.edit_format == "token_dict"
            else None
        )
        correction_tokens = (
            tuple(str(token) for token in row.get(spec.target_tokens_key or "", ()))
            if spec.edit_format == "token_dict"
            else None
        )
        pair = PairRecord(
            dataset_id=spec.dataset_id,
            item_id=item_id,
            source=source_text(row, spec),
            correction=target_text(row, spec),
            source_tokens=source_tokens,
            correction_tokens=correction_tokens,
        )
        pair.validate_for(spec)
        pairs.append(pair)
    return tuple(pairs)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AdapterInputError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise AdapterInputError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def read_pairs(path: Path | str, dataset: str | DatasetSpec) -> tuple[PairRecord, ...]:
    spec = dataset_spec(dataset) if isinstance(dataset, str) else dataset
    pairs = tuple(PairRecord.from_dict(row) for row in _read_jsonl(Path(path)))
    seen: set[str] = set()
    for pair in pairs:
        pair.validate_for(spec)
        if pair.item_id in seen:
            raise AdapterInputError(f"{spec.dataset_id}: duplicate item_id {pair.item_id}")
        seen.add(pair.item_id)
    return pairs


def read_annotations(path: Path | str) -> tuple[AnnotationRecord, ...]:
    records = tuple(AnnotationRecord.from_dict(row) for row in _read_jsonl(Path(path)))
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = record.dataset_id, record.item_id, record.system
        if key in seen:
            raise AdapterOutputError(f"duplicate annotation record: {key}")
        seen.add(key)
    return records


def _write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
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


def write_pairs(path: Path | str, pairs: Iterable[PairRecord]) -> None:
    _write_jsonl(path, (pair.to_dict() for pair in pairs))


def write_annotations(path: Path | str, records: Iterable[AnnotationRecord]) -> None:
    _write_jsonl(path, (record.to_dict() for record in records))


def paper_edits(record: AnnotationRecord, spec: DatasetSpec) -> list[Any]:
    if record.dataset_id != spec.dataset_id:
        raise AdapterOutputError(
            f"record belongs to {record.dataset_id}, expected {spec.dataset_id}"
        )
    if record.span_unit != spec.evaluation_span_unit:
        raise AdapterOutputError(f"{record.item_id}: wrong span unit")
    if spec.edit_format == "char_tuple":
        return [
            [
                edit.start,
                edit.end,
                edit.mapped_label,
                edit.native_label,
                edit.replacement,
            ]
            for edit in record.edits
        ]

    output: list[dict[str, Any]] = []
    for edit in record.edits:
        item: dict[str, Any] = {
            "orig_start": edit.start,
            "orig_end": edit.end,
            "th1_start": -1,
            "th1_end": -1,
            "raw_label": edit.native_label,
            "target": edit.mapped_label,
            "replacement": edit.replacement,
        }
        if edit.char_start is not None:
            item.update(
                {
                    "char_start": edit.char_start,
                    "char_end": edit.char_end,
                    "char_src": edit.char_source or "",
                    "char_repl": edit.char_replacement or "",
                }
            )
        output.append(item)
    return output


def materialize_paper_rows(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[PairRecord],
    records: Sequence[AnnotationRecord],
    spec: DatasetSpec,
    *,
    system: str,
) -> list[dict[str, Any]]:
    """Attach normalized annotations after checking exact pair identity/text."""
    row_ids = [canonical_item_id(row) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise AdapterOutputError("paper rows contain duplicate item IDs")
    for pair in pairs:
        if pair.dataset_id != spec.dataset_id:
            raise AdapterOutputError(
                f"pair {pair.item_id} belongs to {pair.dataset_id}, expected {spec.dataset_id}"
            )
    pair_index = {pair.item_id: pair for pair in pairs}
    if len(pair_index) != len(pairs):
        raise AdapterOutputError("pairs contain duplicate item IDs")
    matching_records = [
        record
        for record in records
        if record.dataset_id == spec.dataset_id and record.system == system
    ]
    record_index = {
        record.item_id: record
        for record in matching_records
    }
    if len(record_index) != len(matching_records):
        raise AdapterOutputError("annotations contain duplicate item IDs")
    expected_ids = set(row_ids)
    if set(pair_index) != expected_ids:
        raise AdapterOutputError("pair item IDs do not match the paper rows")
    if set(record_index) != expected_ids:
        raise AdapterOutputError("annotation item IDs do not match the paper rows")

    output: list[dict[str, Any]] = []
    for raw_row in rows:
        row = copy.deepcopy(dict(raw_row))
        item_id = canonical_item_id(row)
        pair = pair_index[item_id]
        if pair.source != source_text(row, spec) or pair.correction != target_text(row, spec):
            raise AdapterOutputError(f"{spec.dataset_id}/{item_id}: pair text drift")
        record = record_index[item_id]
        record.validate_for_pair(pair, spec)
        row[system] = paper_edits(record, spec)
        output.append(row)
    return output
