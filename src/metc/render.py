"""Render classified annotations in METC's public output format."""

from __future__ import annotations

from collections.abc import Iterable

from .merge import _java_split_space
from .offsets import utf16_len, utf16_slice
from .types import Annotation


def render_classified(
    source: str, annotations: Iterable[Annotation], *, tokenized: bool = False
) -> str:
    ordered = sorted(
        annotations,
        key=lambda annotation: (annotation.start_token if tokenized else annotation.start) or 0,
    )
    if not ordered:
        return source
    return (
        _render_tokenized(source, ordered) if tokenized else _render_non_tokenized(source, ordered)
    )


def _marker(source: str, annotation: Annotation) -> str:
    return f"{{{source}=>{annotation.replacement}:::{annotation.label}/{annotation.sublabel}}}"


def _render_non_tokenized(source: str, annotations: list[Annotation]) -> str:
    result: list[str] = []
    last_end = 0
    for annotation in annotations:
        if annotation.start is None or annotation.end is None:
            raise ValueError("non-tokenized rendering received token offsets")
        if annotation.start < last_end:
            raise ValueError("annotations overlap or are not ordered")
        if annotation.start > last_end:
            result.append(utf16_slice(source, last_end, annotation.start))
        source_text = (
            utf16_slice(source, annotation.start, annotation.end)
            if annotation.start < annotation.end
            else ""
        )
        result.append(_marker(source_text, annotation))
        last_end = annotation.end
    if last_end < utf16_len(source):
        result.append(utf16_slice(source, last_end, utf16_len(source)))
    return "".join(result)


def _render_tokenized(source: str, annotations: list[Annotation]) -> str:
    tokens = _java_split_space(source)
    result = ""
    last_end = 0
    for annotation in annotations:
        if annotation.start_token is None or annotation.end_token is None:
            raise ValueError("tokenized rendering received character offsets")
        start = annotation.start_token
        end = annotation.end_token
        if start < last_end:
            raise ValueError("annotations overlap or are not ordered")
        for index in range(last_end, min(start, len(tokens))):
            if result:
                result += " "
            result += tokens[index]
        if result:
            result += " "
        source_text = " ".join(tokens[start:end]) if 0 <= start < end <= len(tokens) else ""
        result += _marker(source_text, annotation)
        last_end = end
    for index in range(last_end, len(tokens)):
        if result:
            result += " "
        result += tokens[index]
    return result
