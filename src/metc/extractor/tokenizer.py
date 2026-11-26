from typing import Iterator, Tuple

from .constants import (
    eos_chars,
    is_chinese_japanese_char,
    trailing_punct_after_eos,
    whitespace_and_newlines,
    word_delimiters,
)
from .emoji import is_emoji_char


def is_word_delimiter(char):
    return char in word_delimiters or is_emoji_char(char)


def is_finished_sentence(s: str, pos: int = None) -> bool:
    if pos is not None:
        s = s[: pos + 1]

    s = s.rstrip()

    if not s:
        return False

    while s and s[-1] in trailing_punct_after_eos:
        s = s[:-1].rstrip()

    return bool(s) and s[-1] in eos_chars


def is_numeric_context(text: str, index: int) -> bool:
    if 0 < index < len(text) - 1:
        prev_char = text[index - 1]
        next_char = text[index + 1]

        if prev_char.isdigit() and next_char.isdigit():
            return True
    return False


def is_mid_word_apostrophe(text: str, index: int) -> bool:
    """Check if the character at index is an apostrophe in the middle of a word."""
    return (
        text[index] in {"'", "\u2019"}
        and index != 0
        and text[index - 1] not in whitespace_and_newlines
        and index + 1 < len(text)
        and text[index + 1] not in whitespace_and_newlines
    )


def tokenize_text(text: str) -> Iterator[Tuple[str, int, int]]:
    lineStart = 0
    lineEnd = -1

    def findWordEndPosition():
        length = len(text)
        index = lineStart
        while index < length:
            ch = text[index]

            # Chinese/Japanese characters are treated as individual tokens
            # since these languages don't use spaces between words.
            if is_chinese_japanese_char(ch):
                if index == lineStart:
                    return index  # Single CJK char token
                else:
                    return index - 1  # End previous token before CJK char

            is_delimiter = is_word_delimiter(ch) or (
                ch == "." and is_finished_sentence(text, index)
            )

            if is_delimiter:
                if ch in {".", ","} and is_numeric_context(text, index):
                    index += 1
                    continue

                elif is_mid_word_apostrophe(text, index):
                    index += 1
                elif index - lineStart >= 1:
                    return index - 1
                else:
                    return index
            index += 1
        return -1

    while lineEnd < len(text) - 1:
        lineEnd = findWordEndPosition()
        if lineEnd == -1:
            lineEnd = len(text) - 1
        line = text[lineStart : lineEnd + 1]
        yield line, lineStart, lineEnd + 1
        lineStart = lineEnd + 1
