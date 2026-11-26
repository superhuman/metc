"""Deterministically construct the edit jobs used by both LLM judges."""

from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schema import DatasetSpec
from .semantic import source_text, target_text


EDIT_ID_VERSION = "metc-edit-v1"


@dataclass(frozen=True)
class JudgeJobBuild:
    jobs: tuple[dict[str, Any], ...]
    unrecoverable: Mapping[str, int]
    nonjudgeable: Mapping[str, int]
    whitespace_only: Mapping[str, int]


def stable_edit_id(
    dataset_code: str, row_idx: int | str, system: str, edit_idx: int
) -> str:
    if not dataset_code or ":" in dataset_code:
        raise ValueError(f"invalid dataset code for edit ID: {dataset_code!r}")
    if not system or ":" in system:
        raise ValueError(f"invalid system for edit ID: {system!r}")
    return f"{EDIT_ID_VERSION}:{dataset_code}:{row_idx}:{system}:{edit_idx}"


def _edit(raw_edit: Any, spec: DatasetSpec, index: int) -> dict[str, Any]:
    if spec.edit_format == "char_tuple":
        start, end, mapped, raw_label, replacement = raw_edit
        return {
            "start": int(start),
            "end": int(end),
            "raw_label": str(raw_label),
            "mapped": str(mapped),
            "replacement": str(replacement or ""),
            "raw_idx": index,
        }
    return {
        "start": int(raw_edit["orig_start"]),
        "end": int(raw_edit["orig_end"]),
        "raw_label": str(raw_edit.get("raw_label", "")),
        "mapped": str(raw_edit.get("target", "")),
        "replacement": None,
        "raw_idx": index,
    }


def _source_span(row: Mapping[str, Any], spec: DatasetSpec, edit: Mapping[str, Any]) -> str:
    if spec.edit_format == "char_tuple":
        return str(row["src"])[int(edit["start"]) : int(edit["end"])]
    return " ".join(
        str(token)
        for token in row.get("orig_tokens", ())[int(edit["start"]) : int(edit["end"])]
    )


def _target_via_alignment(
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
    start: int,
    end: int,
) -> str:
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


def _target_span(
    row: Mapping[str, Any],
    spec: DatasetSpec,
    edit: Mapping[str, Any],
    raw_edit: Any,
) -> tuple[str, bool]:
    if spec.edit_format == "char_tuple":
        return str(edit["replacement"] or ""), True
    if not isinstance(raw_edit, Mapping):
        return "", False

    th1 = row.get("th1_tokens")
    corrected_tokens = th1 if th1 is not None else row.get("cor_tokens")
    if th1 is not None:
        target_start = raw_edit.get("th1_start")
        target_end = raw_edit.get("th1_end")
        if (
            isinstance(target_start, int)
            and not isinstance(target_start, bool)
            and isinstance(target_end, int)
            and not isinstance(target_end, bool)
            and target_start >= 0
            and target_end >= 0
        ):
            return " ".join(str(token) for token in th1[target_start:target_end]), True
    if raw_edit.get("replacement") is not None:
        return str(raw_edit["replacement"]), True
    if corrected_tokens is None:
        return "", False
    return (
        _target_via_alignment(
            [str(token) for token in row.get("orig_tokens", ())],
            [str(token) for token in corrected_tokens],
            int(edit["start"]),
            int(edit["end"]),
        ),
        True,
    )


def edit_diff_text(source_span: str, target_span: str) -> str:
    source = source_span.strip()
    target = target_span.strip()
    if not source and not target:
        return "(insertion; see corrected sentence above)"
    if not source:
        return f"(insert) => {target!r}"
    if not target:
        return f"{source!r} => (delete)"
    return f"{source!r} => {target!r}"


def _char_edit_diff(source_span: str, target_span: str) -> str:
    if not source_span and not target_span:
        return "(insertion; see corrected sentence above)"
    if not source_span:
        return f"(insert) => {target_span!r}"
    if not target_span:
        return f"{source_span!r} => (delete)"
    return f"{source_span!r} => {target_span!r}"


def build_judge_jobs(
    rows: Iterable[Mapping[str, Any]], spec: DatasetSpec
) -> JudgeJobBuild:
    """Build the complete, pre-filtered paper judge universe for one setting."""

    jobs: list[dict[str, Any]] = []
    unrecoverable: Counter[str] = Counter()
    nonjudgeable: Counter[str] = Counter()
    whitespace: Counter[str] = Counter()
    for row in rows:
        source = source_text(row, spec)
        target = target_text(row, spec)
        row_idx = row.get("idx", row.get("sent_idx", -1))
        for system in spec.systems:
            raw_edits = row.get(system, ())
            for edit_idx, raw_edit in enumerate(raw_edits):
                edit = _edit(raw_edit, spec, edit_idx)
                label = edit["raw_label"] or edit["mapped"]
                source_span = _source_span(row, spec, edit)
                target_span, recoverable = _target_span(
                    row, spec, edit, raw_edit
                )
                if not recoverable:
                    unrecoverable[system] += 1
                    continue
                effective_source = source_span
                effective_target = target_span
                edit_diff = edit_diff_text(source_span, target_span)
                if (
                    edit_diff.startswith("(insertion; see")
                    and isinstance(raw_edit, Mapping)
                    and raw_edit.get("char_src") is not None
                ):
                    effective_source = str(raw_edit["char_src"])
                    effective_target = str(raw_edit.get("char_repl") or "")
                    edit_diff = _char_edit_diff(effective_source, effective_target)
                parts = edit_diff.split(" => ")
                if edit_diff.startswith("(insertion; see") or (
                    len(parts) == 2 and parts[0] == parts[1]
                ):
                    nonjudgeable[system] += 1
                    continue
                if effective_source.strip() == "" and effective_target.strip() == "":
                    whitespace[system] += 1
                    continue
                jobs.append(
                    {
                        "edit_id": stable_edit_id(
                            spec.experiment_code, row_idx, system, edit_idx
                        ),
                        "edit_idx": edit_idx,
                        "edit_start": edit["start"],
                        "edit_end": edit["end"],
                        "system": system,
                        "row_idx": row_idx,
                        "lang": spec.language,
                        "src": source,
                        "tgt": target,
                        "edit_diff": edit_diff,
                        "label": label,
                    }
                )
    return JudgeJobBuild(
        jobs=tuple(jobs),
        unrecoverable={system: unrecoverable[system] for system in spec.systems},
        nonjudgeable={system: nonjudgeable[system] for system in spec.systems},
        whitespace_only={system: whitespace[system] for system in spec.systems},
    )

