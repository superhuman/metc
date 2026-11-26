"""M2 parsing and CoNLL/KoLLA reference construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class M2Edit:
    start: int
    end: int
    label: str
    replacement: str
    annotator: str


@dataclass(frozen=True)
class M2Sentence:
    tokens: tuple[str, ...]
    edits: tuple[M2Edit, ...]


def parse_m2(path: Path) -> Iterator[M2Sentence]:
    tokens: tuple[str, ...] | None = None
    edits: list[M2Edit] = []
    with Path(path).open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if line.startswith("S "):
                if tokens is not None:
                    yield M2Sentence(tokens, tuple(edits))
                tokens = tuple(line[2:].split(" "))
                edits = []
            elif line.startswith("A "):
                if tokens is None:
                    raise ValueError(f"M2 edit appears before a sentence in {path}")
                span, *fields = line[2:].split("|||")
                if len(fields) < 2:
                    raise ValueError(f"malformed M2 edit in {path}: {line!r}")
                start_text, end_text = span.split()
                edits.append(
                    M2Edit(
                        int(start_text),
                        int(end_text),
                        fields[0],
                        fields[1],
                        fields[-1],
                    )
                )
        if tokens is not None:
            yield M2Sentence(tokens, tuple(edits))


def _real_edits(sentence: M2Sentence, annotator: str) -> list[M2Edit]:
    return [
        edit
        for edit in sentence.edits
        if edit.annotator == annotator and edit.start >= 0 and edit.label.lower() != "noop"
    ]


def _apply_edits(
    tokens: Iterable[str], edits: list[M2Edit], *, same_position_reverse: bool
) -> list[str]:
    corrected = list(tokens)
    indexed = list(enumerate(edits))
    if same_position_reverse:
        indexed.sort(key=lambda item: (item[1].start, item[1].end, item[0]), reverse=True)
    else:
        indexed.sort(key=lambda item: item[1].start, reverse=True)
    for _index, edit in indexed:
        replacement = edit.replacement.split() if edit.replacement not in {"", "-NONE-"} else []
        corrected[edit.start : edit.end] = replacement
    return corrected


def _insertion_offset(tokens: tuple[str, ...], index: int) -> int:
    if index <= 0:
        return 0
    if index >= len(tokens):
        return len(" ".join(tokens))
    return len(" ".join(tokens[:index])) + 1


def _token_span_to_chars(tokens: tuple[str, ...], start: int, end: int) -> tuple[int, int]:
    if start == end:
        point = _insertion_offset(tokens, start)
        return point, point
    char_start = _insertion_offset(tokens, start)
    char_end = len(" ".join(tokens[:end]))
    return char_start, char_end


def reconstruct_conll(path: Path, annotator: int) -> list[dict[str, object]]:
    """Build the paper's CoNLL reference for one official annotator."""
    rows: list[dict[str, object]] = []
    for sent_idx, sentence in enumerate(parse_m2(path)):
        edits = _real_edits(sentence, str(annotator))
        corrected = _apply_edits(sentence.tokens, edits, same_position_reverse=False)
        source_text = " ".join(sentence.tokens)
        target_text = " ".join(corrected)
        if source_text == target_text:
            continue
        gold: list[list[object]] = []
        for edit in edits:
            start, end = _token_span_to_chars(sentence.tokens, edit.start, edit.end)
            replacement = "" if edit.replacement == "-NONE-" else edit.replacement
            gold.append([start, end, edit.label, edit.label, replacement])
        rows.append({"sent_idx": sent_idx, "src": source_text, "tgt": target_text, "gold": gold})
    return rows


def reconstruct_kolla(path: Path) -> list[dict[str, object]]:
    """Build the KoLLA/KAGAS-derived reference using annotator 0."""
    rows: list[dict[str, object]] = []
    for idx, sentence in enumerate(parse_m2(path)):
        edits = _real_edits(sentence, "0")
        corrected = _apply_edits(sentence.tokens, edits, same_position_reverse=True)
        gold = [
            {
                "orig_start": edit.start,
                "orig_end": edit.end,
                "raw_label": edit.label,
                "target": edit.label,
                "replacement": "" if edit.replacement == "-NONE-" else edit.replacement,
            }
            for edit in edits
        ]
        rows.append(
            {
                "idx": idx,
                "orig_tokens": list(sentence.tokens),
                "cor_tokens": corrected,
                "gold": gold,
            }
        )
    return rows
