"""UA-GEC test/gec-only reference reconstruction without a package dependency."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


_ANNOTATION = re.compile(r"\{([^{]*)=>(.*?)(:::[^:][^}]*)?\}")
_TOKEN = re.compile(r"\S+")
_NO_SUGGESTIONS = "NO_SUGGESTIONS"


@dataclass(frozen=True)
class InlineAnnotation:
    start: int
    end: int
    source: str
    replacement: str
    error_type: str


def _unescape(text: str) -> str:
    return text.replace("\\n", "\n")


def _parse_annotated(text: str) -> tuple[str, str, list[InlineAnnotation]]:
    original_parts: list[str] = []
    corrected_parts: list[str] = []
    annotations: list[InlineAnnotation] = []
    previous = 0
    source_length = 0
    for match in _ANNOTATION.finditer(text):
        prefix = text[previous : match.start()]
        source = _unescape(match.group(1))
        suggestion_text = match.group(2)
        if suggestion_text == _NO_SUGGESTIONS:
            replacement = source
        else:
            replacement = _unescape(suggestion_text.split("|", 1)[0])
        meta_text = match.group(3) or ""
        metadata = {
            key: value
            for field in meta_text.split(":::")[1:]
            for key, separator, value in (field.partition("="),)
            if separator
        }
        error_type = metadata.get("error_type")
        if error_type is None:
            raise ValueError(f"UA-GEC annotation has no error_type: {match.group(0)!r}")

        original_parts.extend((prefix, source))
        corrected_parts.extend((prefix, replacement))
        source_length += len(prefix)
        start = source_length
        source_length += len(source)
        annotations.append(InlineAnnotation(start, source_length, source, replacement, error_type))
        previous = match.end()
    suffix = text[previous:]
    original_parts.append(suffix)
    corrected_parts.append(suffix)
    return "".join(original_parts), "".join(corrected_parts), annotations


def _token_span(
    token_spans: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    token_start: int | None = None
    token_end: int | None = None
    for index, (start, end) in enumerate(token_spans):
        if end <= char_start:
            continue
        if start >= char_end:
            if token_start is None:
                token_start = index
                token_end = index
            break
        if token_start is None:
            token_start = index
        token_end = index + 1
    if token_start is None:
        token_start = len(token_spans)
        token_end = token_start
    assert token_end is not None
    return token_start, token_end


def _paragraph_reference(line: str) -> tuple[list[str], list[str], list[dict[str, object]]]:
    source, corrected, annotations = _parse_annotated(line)
    token_matches = list(_TOKEN.finditer(source))
    original_tokens = [match.group(0) for match in token_matches]
    token_spans = [(match.start(), match.end()) for match in token_matches]
    gold: list[dict[str, object]] = []
    for annotation in annotations:
        token_start, token_end = _token_span(token_spans, annotation.start, annotation.end)
        if token_start == token_end:
            replacement = annotation.replacement
        else:
            span_start = token_spans[token_start][0]
            span_end = token_spans[token_end - 1][1]
            replacement = (
                source[span_start : annotation.start]
                + annotation.replacement
                + source[annotation.end : span_end]
            )
        gold.append(
            {
                "orig_start": token_start,
                "orig_end": token_end,
                "raw_label": annotation.error_type,
                "target": annotation.error_type,
                "replacement": replacement,
                "char_start": annotation.start,
                "char_end": annotation.end,
                "char_src": annotation.source,
                "char_repl": annotation.replacement,
            }
        )
    return original_tokens, corrected.split(), gold


def _test_documents(root: Path) -> list[tuple[str, int]]:
    metadata_paths = list(Path(root).rglob("metadata.csv"))
    if len(metadata_paths) != 1:
        raise ValueError(f"expected one UA-GEC metadata.csv under {root}")
    documents: list[tuple[str, int]] = []
    with metadata_paths[0].open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["partition"] != "test":
                continue
            documents.extend((row["id"], int(value)) for value in row["annotator_id"].split())
    return documents


def _annotation_directory(root: Path) -> Path:
    candidates = [
        path
        for path in Path(root).rglob("annotated")
        if path.parent.name == "test" and path.parent.parent.name == "gec-only"
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one UA-GEC gec-only/test/annotated directory under {root}")
    return candidates[0]


def reconstruct_uagec(root: Path) -> list[dict[str, object]]:
    """Reconstruct the 2,127 paragraph records used by the paper."""
    annotation_dir = _annotation_directory(root)
    rows: list[dict[str, object]] = []
    for doc_id, annotator in _test_documents(root):
        path = annotation_dir / f"{doc_id}.a{annotator}.ann"
        raw = path.read_text(encoding="utf-8")
        source, corrected, _annotations = _parse_annotated(raw)
        source_lines = source.split("\n")
        corrected_lines = corrected.split("\n")
        if len(source_lines) != len(corrected_lines):
            continue
        raw_lines = raw.split("\n")
        if len(raw_lines) != len(source_lines):
            raise ValueError(f"unexpected escaped-newline structure in aligned UA-GEC file: {path}")
        for para_idx, (raw_line, source_line, corrected_line) in enumerate(
            zip(raw_lines, source_lines, corrected_lines, strict=True)
        ):
            if not source_line.strip() or not corrected_line.strip():
                continue
            original_tokens, corrected_tokens, gold = _paragraph_reference(raw_line)
            rows.append(
                {
                    "idx": len(rows),
                    "doc_id": doc_id,
                    "para_idx": para_idx,
                    "orig_tokens": original_tokens,
                    "cor_tokens": corrected_tokens,
                    "gold": gold,
                }
            )
    return rows
