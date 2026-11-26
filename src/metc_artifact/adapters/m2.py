"""M2 parsing and span projection without importing any NLP tool stack."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .contract import NormalizedEdit, PairRecord
from .errors import AdapterOutputError


@dataclass(frozen=True)
class M2Edit:
    start: int
    end: int
    native_label: str
    replacement: str


@dataclass(frozen=True)
class M2Sentence:
    tokens: tuple[str, ...]
    edits: tuple[M2Edit, ...]


def parse_m2(path: Path | str) -> tuple[M2Sentence, ...]:
    sentences: list[M2Sentence] = []
    tokens: tuple[str, ...] | None = None
    edits: list[M2Edit] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\n")
            if line.startswith("S "):
                if tokens is not None:
                    sentences.append(M2Sentence(tokens, tuple(edits)))
                tokens = tuple(line[2:].split(" ")) if line[2:] else ()
                edits = []
            elif line.startswith("A "):
                if tokens is None:
                    raise AdapterOutputError(f"{path}:{line_number}: edit before sentence")
                parts = line[2:].split("|||")
                if len(parts) < 3:
                    raise AdapterOutputError(f"{path}:{line_number}: malformed M2 edit")
                try:
                    start, end = (int(value) for value in parts[0].split())
                except (ValueError, TypeError) as error:
                    raise AdapterOutputError(
                        f"{path}:{line_number}: malformed M2 span"
                    ) from error
                native_label = parts[1]
                if start < 0 or native_label.lower() == "noop":
                    continue
                replacement = "" if parts[2] == "-NONE-" else parts[2]
                edits.append(M2Edit(start, end, native_label, replacement))
    if tokens is not None:
        sentences.append(M2Sentence(tokens, tuple(edits)))
    return tuple(sentences)


def token_char_spans(tokens: Sequence[str], separator: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    position = 0
    for index, token in enumerate(tokens):
        if index:
            position += len(separator)
        spans.append((position, position + len(token)))
        position += len(token)
    return tuple(spans)


def _position_mapper(source: str, target: str) -> Callable[[int], int]:
    blocks = difflib.SequenceMatcher(a=source, b=target, autojunk=False).get_matching_blocks()

    def convert(position: int) -> int:
        for source_start, target_start, length in blocks:
            if position < source_start:
                return target_start
            if source_start <= position < source_start + length:
                return target_start + position - source_start
        return len(target)

    return convert


def _tool_char_span(sentence: M2Sentence, edit: M2Edit, separator: str) -> tuple[int, int]:
    spans = token_char_spans(sentence.tokens, separator)
    if edit.start < 0 or edit.end < edit.start or edit.end > len(spans):
        raise AdapterOutputError(
            f"M2 span [{edit.start}, {edit.end}) is outside {len(spans)} tokens"
        )
    if edit.start == edit.end:
        boundary = spans[edit.start][0] if edit.start < len(spans) else len(
            separator.join(sentence.tokens)
        )
        return boundary, boundary
    return spans[edit.start][0], spans[edit.end - 1][1]


def _char_to_token_span(
    token_spans: Sequence[tuple[int, int]],
    start: int,
    end: int,
    *,
    insertion_inside_token_uses_containing_boundary: bool = False,
) -> tuple[int, int]:
    if start == end:
        if insertion_inside_token_uses_containing_boundary:
            for index, (token_start, token_end) in enumerate(token_spans):
                if token_start < start < token_end:
                    return index, index
        for index, (token_start, _token_end) in enumerate(token_spans):
            if token_start >= start:
                return index, index
        return len(token_spans), len(token_spans)
    first: int | None = None
    last: int | None = None
    for index, (token_start, token_end) in enumerate(token_spans):
        if token_end > start and token_start < end:
            first = index if first is None else first
            last = index + 1
    if first is None:
        if not token_spans:
            return 0, 0
        closest = min(range(len(token_spans)), key=lambda i: abs(token_spans[i][0] - start))
        return closest, closest
    return first, last if last is not None else first


def normalized_m2_edits(
    pair: PairRecord,
    sentence: M2Sentence,
    *,
    span_unit: str,
    label_mapper: Callable[[str], str],
    tool_separator: str,
    insertion_inside_token_uses_containing_boundary: bool = False,
) -> tuple[NormalizedEdit, ...]:
    tool_source = tool_separator.join(sentence.tokens)
    to_source = _position_mapper(tool_source, pair.source)
    source_token_spans = (
        token_char_spans(pair.source_tokens or (), " ") if span_unit == "token" else ()
    )
    output: list[NormalizedEdit] = []
    for edit in sentence.edits:
        tool_start, tool_end = _tool_char_span(sentence, edit, tool_separator)
        char_start = to_source(tool_start)
        char_end = to_source(tool_end)
        if span_unit == "token":
            start, end = _char_to_token_span(
                source_token_spans,
                char_start,
                char_end,
                insertion_inside_token_uses_containing_boundary=(
                    insertion_inside_token_uses_containing_boundary
                ),
            )
        elif span_unit == "unicode_codepoint":
            start, end = char_start, char_end
        else:  # pragma: no cover - schema controls this
            raise AdapterOutputError(f"unsupported span unit: {span_unit}")
        output.append(
            NormalizedEdit(
                start=start,
                end=end,
                replacement=edit.replacement,
                native_label=edit.native_label,
                mapped_label=label_mapper(edit.native_label),
            )
        )
    return tuple(output)


_WHITESPACE = re.compile(r"\s+")


def source_key(value: str) -> str:
    return _WHITESPACE.sub("", value)


def align_m2_by_position(
    pairs: Sequence[PairRecord],
    sentences: Sequence[M2Sentence],
    *,
    allow_parser_normalization: bool = False,
) -> tuple[M2Sentence, ...]:
    if len(pairs) != len(sentences):
        raise AdapterOutputError(
            f"M2 sentence count {len(sentences)} differs from pair count {len(pairs)}"
        )
    for pair, sentence in zip(pairs, sentences, strict=True):
        tool_source = " ".join(sentence.tokens)
        if (
            not allow_parser_normalization
            and source_key(tool_source) != source_key(pair.source)
        ):
            raise AdapterOutputError(
                f"M2 source at item {pair.item_id} does not match canonical pair text"
            )
    return tuple(sentences)


def align_m2_historical_first_source(
    pairs: Sequence[PairRecord], sentences: Sequence[M2Sentence]
) -> tuple[M2Sentence, ...]:
    """Reproduce the German source-only first-hit materialization exactly."""

    aligned = align_m2_by_position(pairs, sentences)
    by_source: dict[str, M2Sentence] = {}
    for sentence in aligned:
        by_source.setdefault(source_key(" ".join(sentence.tokens)), sentence)
    output: list[M2Sentence] = []
    for pair in pairs:
        try:
            output.append(by_source[source_key(pair.source)])
        except KeyError as error:
            raise AdapterOutputError(
                f"no gERRANT M2 sentence matches pair {pair.item_id}"
            ) from error
    return tuple(output)
