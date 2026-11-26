"""Conversions between Python code-point indexes and Java UTF-16 offsets."""

from __future__ import annotations


def utf16_len(text: str) -> int:
    """Return the number of UTF-16 code units in ``text``."""

    return len(text.encode("utf-16-le")) // 2


def py_to_utf16_index(text: str, index: int) -> int:
    """Convert a Python string boundary to a UTF-16 boundary."""

    if not 0 <= index <= len(text):
        raise IndexError(f"Python index {index} is outside 0..{len(text)}")
    return utf16_len(text[:index])


def utf16_to_py_index(text: str, offset: int) -> int:
    """Convert a UTF-16 boundary to a Python string boundary.

    An offset in the middle of a surrogate pair is rejected. METC-generated
    offsets are always valid boundaries; rejecting invalid external offsets
    prevents silent corruption during rendering and annotation merging.
    """

    total = utf16_len(text)
    if not 0 <= offset <= total:
        raise IndexError(f"UTF-16 offset {offset} is outside 0..{total}")

    consumed = 0
    for index, char in enumerate(text):
        if consumed == offset:
            return index
        width = 2 if ord(char) > 0xFFFF else 1
        if consumed < offset < consumed + width:
            raise ValueError(f"UTF-16 offset {offset} splits a surrogate pair")
        consumed += width
    return len(text)


def utf16_slice(text: str, start: int, end: int) -> str:
    """Slice ``text`` using half-open UTF-16 offsets."""

    if end < start:
        raise ValueError(f"end {end} precedes start {start}")
    return text[utf16_to_py_index(text, start) : utf16_to_py_index(text, end)]
