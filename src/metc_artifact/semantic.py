"""Semantic normalization and equality checks for annotation files.

Reproduction is intentionally independent of JSON formatting and edit order.
Two outputs are equal when they contain the same benchmark items, exact source
and correction text, and the same multiset of edits for every requested system.
An edit is identified by its evaluation span, replacement, native label, and
mapped reference-taxonomy label.
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schema import DatasetSpec


@dataclass(frozen=True, order=True)
class SemanticEdit:
    span_unit: str
    start: int
    end: int
    replacement: str
    native_label: str
    mapped_label: str


@dataclass(frozen=True)
class ItemDifference:
    item_id: str
    field: str
    expected: Any
    actual: Any


@dataclass(frozen=True)
class AnnotationComparison:
    dataset_id: str
    expected_items: int
    actual_items: int
    systems: tuple[str, ...]
    difference_count: int
    differences: tuple[ItemDifference, ...]

    @property
    def equal(self) -> bool:
        return self.difference_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "equal": self.equal,
            "expected_items": self.expected_items,
            "actual_items": self.actual_items,
            "systems": list(self.systems),
            "difference_count": self.difference_count,
            "differences": [asdict(difference) for difference in self.differences],
        }


def _exact_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}: expected an integer")
    return value


def _bounded_span(
    start_value: Any,
    end_value: Any,
    *,
    limit: int,
    context: str,
) -> tuple[int, int]:
    start = _exact_int(start_value, f"{context}.start")
    end = _exact_int(end_value, f"{context}.end")
    if not 0 <= start <= end <= limit:
        raise ValueError(
            f"{context}: span [{start}, {end}) is outside [0, {limit}]"
        )
    return start, end


def canonical_item_id(row: Mapping[str, Any]) -> str:
    """Return a type-preserving JSON identity for one benchmark item."""

    if "idx" in row:
        value = row["idx"]
    elif "sent_idx" in row:
        value = row["sent_idx"]
    else:
        raise ValueError("benchmark row has neither 'idx' nor 'sent_idx'")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_text(row: Mapping[str, Any], spec: DatasetSpec) -> str:
    if spec.edit_format == "char_tuple":
        return str(row["src"])
    return " ".join(str(token) for token in row.get("orig_tokens", ()))


def target_text(row: Mapping[str, Any], spec: DatasetSpec) -> str:
    if spec.edit_format == "char_tuple":
        return str(row.get("tgt", row["src"]))
    if spec.target_tokens_key is None:
        raise ValueError(f"{spec.dataset_id}: token benchmark has no target token key")
    return " ".join(str(token) for token in row.get(spec.target_tokens_key, ()))


def _alignment_replacement(
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
    start: int,
    end: int,
) -> str:
    """Recover a replacement from a source/target token alignment.

    This is only a final fallback for historical rows that lack explicit target
    coordinates and replacement text. Exact producer outputs should carry one
    of those two representations directly.
    """

    opcodes = difflib.SequenceMatcher(
        a=list(source_tokens), b=list(target_tokens), autojunk=False
    ).get_opcodes()
    if start == end:
        before = 0
        after = len(target_tokens)
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "insert" and i1 == start:
                return " ".join(target_tokens[j1:j2])
            if i2 <= start:
                before = max(before, j2)
                continue
            if i1 < start < i2:
                if tag in {"replace", "delete"}:
                    return " ".join(target_tokens[j1:j2])
                position = j1 + (start - i1)
                before = after = position
                break
            if i1 >= start:
                after = min(after, j1)
                break
        return " ".join(target_tokens[before:after]) if after > before else ""

    output: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if i2 <= start or i1 >= end:
            continue
        if tag == "equal":
            overlap_start = max(i1, start)
            overlap_end = min(i2, end)
            target_start = j1 + overlap_start - i1
            target_end = j1 + overlap_end - i1
            output.extend(target_tokens[target_start:target_end])
        else:
            output.extend(target_tokens[j1:j2])
    return " ".join(output)


def _token_replacement(
    row: Mapping[str, Any], raw_edit: Mapping[str, Any], spec: DatasetSpec
) -> str:
    if spec.target_tokens_key is None:
        return ""
    raw_target_tokens = row.get(spec.target_tokens_key)
    if not isinstance(raw_target_tokens, Sequence) or isinstance(
        raw_target_tokens, (str, bytes)
    ):
        raise ValueError(f"{spec.dataset_id}: target is not a token sequence")
    target_tokens = [str(token) for token in raw_target_tokens]
    has_target_start = "th1_start" in raw_edit
    has_target_end = "th1_end" in raw_edit
    if has_target_start != has_target_end:
        raise ValueError(
            f"{spec.dataset_id}: token edit must carry both th1_start and th1_end"
        )
    if has_target_start:
        target_start = _exact_int(
            raw_edit["th1_start"], f"{spec.dataset_id}.target_span.start"
        )
        target_end = _exact_int(
            raw_edit["th1_end"], f"{spec.dataset_id}.target_span.end"
        )
        if (target_start, target_end) == (-1, -1):
            pass
        elif not 0 <= target_start <= target_end <= len(target_tokens):
            raise ValueError(
                f"{spec.dataset_id}.target_span: span [{target_start}, {target_end}) "
                f"is outside [0, {len(target_tokens)}]"
            )

    replacement = raw_edit.get("replacement")
    if replacement is not None:
        return str(replacement)
    char_replacement = raw_edit.get("char_repl")
    if char_replacement is not None:
        return str(char_replacement)
    if has_target_start and (target_start, target_end) != (-1, -1):
        return " ".join(target_tokens[target_start:target_end])

    source_tokens = [str(token) for token in row.get("orig_tokens", ())]
    return _alignment_replacement(
        source_tokens,
        target_tokens,
        int(raw_edit["orig_start"]),
        int(raw_edit["orig_end"]),
    )


def semantic_edit(
    row: Mapping[str, Any], raw_edit: Any, spec: DatasetSpec
) -> SemanticEdit:
    if spec.edit_format == "char_tuple":
        if not isinstance(raw_edit, (list, tuple)) or len(raw_edit) != 5:
            raise ValueError(
                f"{spec.dataset_id}: expected a five-element character edit, got {raw_edit!r}"
            )
        start_value, end_value, mapped_label, native_label, replacement = raw_edit
        source = row.get("src")
        if not isinstance(source, str):
            raise ValueError(f"{spec.dataset_id}: character benchmark source is not a string")
        start, end = _bounded_span(
            start_value,
            end_value,
            limit=len(source),
            context=f"{spec.dataset_id}.source_span",
        )
        return SemanticEdit(
            span_unit=spec.evaluation_span_unit,
            start=start,
            end=end,
            replacement=str(replacement or ""),
            native_label=str(native_label or ""),
            mapped_label=str(mapped_label or ""),
        )

    if not isinstance(raw_edit, Mapping):
        raise ValueError(f"{spec.dataset_id}: expected a token edit object, got {raw_edit!r}")
    source_tokens = row.get("orig_tokens")
    if not isinstance(source_tokens, Sequence) or isinstance(source_tokens, (str, bytes)):
        raise ValueError(f"{spec.dataset_id}: token benchmark source is not a token sequence")
    start, end = _bounded_span(
        raw_edit.get("orig_start"),
        raw_edit.get("orig_end"),
        limit=len(source_tokens),
        context=f"{spec.dataset_id}.source_span",
    )
    return SemanticEdit(
        span_unit=spec.evaluation_span_unit,
        start=start,
        end=end,
        replacement=_token_replacement(row, raw_edit, spec),
        native_label=str(raw_edit.get("raw_label") or ""),
        mapped_label=str(raw_edit.get("target") or ""),
    )


def semantic_edits(
    row: Mapping[str, Any], system: str, spec: DatasetSpec
) -> Counter[SemanticEdit]:
    raw = row.get(system)
    if raw is None:
        raise ValueError(
            f"{spec.dataset_id}/{canonical_item_id(row)}: system {system!r} is missing"
        )
    if not isinstance(raw, list):
        raise ValueError(
            f"{spec.dataset_id}/{canonical_item_id(row)}: system {system!r} is not a list"
        )
    return Counter(semantic_edit(row, edit, spec) for edit in raw)


def index_rows(
    rows: Iterable[Mapping[str, Any]], spec: DatasetSpec
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item_id = canonical_item_id(row)
        if item_id in output:
            raise ValueError(f"{spec.dataset_id}: duplicate item ID {item_id}")
        output[item_id] = row
    return output


def _counter_payload(counter: Counter[SemanticEdit]) -> list[dict[str, Any]]:
    payload = []
    for edit, count in sorted(counter.items()):
        item = asdict(edit)
        item["count"] = count
        payload.append(item)
    return payload


def compare_annotations(
    expected_rows: Iterable[Mapping[str, Any]],
    actual_rows: Iterable[Mapping[str, Any]],
    spec: DatasetSpec,
    *,
    systems: Iterable[str] | None = None,
    max_differences: int = 100,
) -> AnnotationComparison:
    """Compare two complete annotation datasets without serialization coupling."""

    if max_differences < 0:
        raise ValueError("max_differences must be nonnegative")
    expected = index_rows(expected_rows, spec)
    actual = index_rows(actual_rows, spec)
    checked_systems = tuple(systems or spec.systems)
    differences: list[ItemDifference] = []
    difference_count = 0

    def add(item_id: str, field: str, left: Any, right: Any) -> None:
        nonlocal difference_count
        difference_count += 1
        if len(differences) < max_differences:
            differences.append(ItemDifference(item_id, field, left, right))

    for item_id in sorted(set(expected) - set(actual)):
        add(item_id, "item", "present", "missing")
    for item_id in sorted(set(actual) - set(expected)):
        add(item_id, "item", "missing", "present")

    for item_id in sorted(set(expected) & set(actual)):
        expected_row = expected[item_id]
        actual_row = actual[item_id]
        expected_source = source_text(expected_row, spec)
        actual_source = source_text(actual_row, spec)
        if expected_source != actual_source:
            add(item_id, "source", expected_source, actual_source)
        expected_target = target_text(expected_row, spec)
        actual_target = target_text(actual_row, spec)
        if expected_target != actual_target:
            add(item_id, "correction", expected_target, actual_target)
        for system in checked_systems:
            expected_edits = semantic_edits(expected_row, system, spec)
            actual_edits = semantic_edits(actual_row, system, spec)
            if expected_edits != actual_edits:
                add(
                    item_id,
                    f"annotations.{system}",
                    _counter_payload(expected_edits),
                    _counter_payload(actual_edits),
                )

    return AnnotationComparison(
        dataset_id=spec.dataset_id,
        expected_items=len(expected),
        actual_items=len(actual),
        systems=checked_systems,
        difference_count=difference_count,
        differences=tuple(differences),
    )
