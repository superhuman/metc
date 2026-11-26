"""Exact Python port of METC's two-pass annotation merger."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable

from .offsets import utf16_len, utf16_slice
from .types import Annotation

MERGEABLE_SUBLABELS = frozenset(
    {
        "Adjective",
        "Adverb",
        "Negation",
        "Noun",
        "Pronoun",
        "Verb",
        "VerbTenseForm",
        "WordOrder",
    }
)
BROADER_SUBLABELS = frozenset(
    {
        "Adjective",
        "Adverb",
        "Agreement",
        "Conjunction",
        "Determiner",
        "Compound",
        "Compound_Trivial",
        "Compound_Moderate",
        "Compound_Complex",
        "Morphology",
        "Negation",
        "Noun",
        "Preposition",
        "Pronoun",
        "Misspelling",
        "Verb",
        "VerbTenseForm",
        "WordOrder",
    }
)
MULTI_WORD_REQUIRED_SUBLABELS = frozenset({"Conjunction", "Determiner", "Preposition"})
EXCLUDED_OVERLAP_WORDS = frozenset({"to"})


def _java_split_space(text: str) -> list[str]:
    """Match Java ``String.split(" ")`` for this pipeline."""

    parts = text.split(" ")
    if text == "":
        return parts
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def _java_is_whitespace(char: str) -> bool:
    """Match Java ``Character.isWhitespace`` for Unicode code points."""

    if char in {"\t", "\n", "\v", "\f", "\r", "\x1c", "\x1d", "\x1e", "\x1f"}:
        return True
    # Java intentionally excludes these three non-breaking space separators.
    if char in {"\u00a0", "\u2007", "\u202f"}:
        return False
    return unicodedata.category(char) in {"Zs", "Zl", "Zp"}


def _java_is_blank(text: str) -> bool:
    return all(_java_is_whitespace(char) for char in text)


def _java_trim(text: str) -> str:
    """Match Java ``String.trim`` (remove code points U+0020 and below)."""

    start = 0
    end = len(text)
    while start < end and ord(text[start]) <= 0x20:
        start += 1
    while end > start and ord(text[end - 1]) <= 0x20:
        end -= 1
    return text[start:end]


def _java_is_letter_or_digit_utf16_unit(code_unit: int) -> bool:
    """Match ``Character.isLetterOrDigit(char)`` for one UTF-16 code unit."""

    category = unicodedata.category(chr(code_unit))
    return category in {"Lu", "Ll", "Lt", "Lm", "Lo", "Nd"}


class AnnotationMerger:
    """Merge word-order-like pairs, then chained digit-formatting edits.

    This follows ``AnnotationMerger.java``. METC uses the multilingual gap
    limit (three untouched tokens) and granular compound labels by default.
    """

    def __init__(
        self,
        source: str,
        *,
        tokenized: bool = False,
        multilingual: bool = True,
        granular_compound: bool = True,
    ) -> None:
        self.source = source
        self.source_tokens = _java_split_space(source)
        self.tokenized = tokenized
        self.max_words_between = 3 if multilingual else 5
        self.granular_compound = granular_compound

    def merge(self, annotations: Iterable[Annotation]) -> list[Annotation]:
        ordered = list(annotations)
        return self._digit_formatting_pass(self._word_order_pass(ordered))

    def _word_order_pass(self, annotations: list[Annotation]) -> list[Annotation]:
        return self._walk(
            annotations,
            self._can_merge_as_word_order,
            self._create_merged_as_word_order,
            lockout_after_merge=True,
        )

    def _digit_formatting_pass(self, annotations: list[Annotation]) -> list[Annotation]:
        return self._walk(
            annotations,
            self._can_merge_as_digit_formatting,
            self._create_merged_as_digit_formatting,
            lockout_after_merge=False,
        )

    def _walk(
        self,
        annotations: list[Annotation],
        can_merge: Callable[[Annotation, Annotation, list[Annotation]], bool],
        merger: Callable[[Annotation, Annotation], Annotation],
        *,
        lockout_after_merge: bool,
    ) -> list[Annotation]:
        if len(annotations) < 2:
            return annotations
        all_annotations = sorted(annotations, key=self._position)
        result: list[Annotation] = []
        last = all_annotations[0]
        last_merged = False
        for current in all_annotations[1:]:
            if (not lockout_after_merge or not last_merged) and can_merge(
                last, current, all_annotations
            ):
                last = merger(last, current)
                last_merged = True
            else:
                result.append(last)
                last = current
                last_merged = False
        result.append(last)
        return result

    def _position(self, annotation: Annotation) -> int:
        value = annotation.start_token if self.tokenized else annotation.start
        return value if value is not None else 0

    def _can_merge_as_digit_formatting(
        self, first: Annotation, second: Annotation, all_annotations: list[Annotation]
    ) -> bool:
        return (
            self._has_digit(first)
            and self._has_digit(second)
            and self._is_gap_whitespace_only(first, second)
            and self._is_gap_untouched(first, second, all_annotations)
        )

    def _has_digit(self, annotation: Annotation) -> bool:
        return bool(
            re.search(r"[0-9]", self._extract_source(annotation))
            or re.search(r"[0-9]", annotation.replacement or "")
        )

    def _is_gap_whitespace_only(self, first: Annotation, second: Annotation) -> bool:
        if self.tokenized:
            return (
                first.end_token is not None
                and second.start_token is not None
                and first.end_token == second.start_token
            )
        if first.end is None or second.start is None or first.end > second.start:
            return False
        return all(
            _java_is_whitespace(char) for char in utf16_slice(self.source, first.end, second.start)
        )

    def _can_merge_as_word_order(
        self, first: Annotation, second: Annotation, all_annotations: list[Annotation]
    ) -> bool:
        distance = self._compute_distance(first, second)
        return (
            0 <= distance <= self.max_words_between
            and self._is_allowed_operation_pair(first, second)
            and self._is_eligible_pair(first, second)
            and self._passes_multi_word_check(first, second)
            and self._has_word_overlap(first, second)
            and self._is_gap_untouched(first, second, all_annotations)
        )

    def _compute_distance(self, first: Annotation, second: Annotation) -> int:
        if self.tokenized:
            if first.end_token is None or second.start_token is None:
                return -1
            return (
                second.start_token - first.end_token
                if first.end_token <= second.start_token
                else -1
            )
        if first.end is None or second.start is None:
            return -1
        if first.end > second.start:
            return -1
        return self._count_words(utf16_slice(self.source, first.end, second.start))

    @staticmethod
    def _count_words(text: str) -> int:
        return sum(not _java_is_blank(word) for word in _java_split_space(text))

    def _is_insertion(self, annotation: Annotation) -> bool:
        return (
            annotation.start_token == annotation.end_token
            if self.tokenized
            else annotation.start == annotation.end
        )

    @staticmethod
    def _is_deletion(annotation: Annotation) -> bool:
        return annotation.replacement is None or _java_is_blank(annotation.replacement)

    def _is_replacement(self, annotation: Annotation) -> bool:
        return not self._is_insertion(annotation) and not self._is_deletion(annotation)

    def _is_allowed_operation_pair(self, first: Annotation, second: Annotation) -> bool:
        first_insert = self._is_insertion(first)
        second_insert = self._is_insertion(second)
        first_delete = self._is_deletion(first)
        second_delete = self._is_deletion(second)
        if first_insert and second_insert:
            return False
        if first_delete and second_delete:
            return False
        if self._is_replacement(first) and self._is_replacement(second):
            return any(
                annotation.sublabel == "WordOrder"
                or self._is_compound_sublabel(annotation.sublabel)
                for annotation in (first, second)
            )
        return True

    @staticmethod
    def _is_compound_sublabel(sublabel: str | None) -> bool:
        return bool(sublabel and sublabel.startswith("Compound"))

    def _is_eligible_pair(self, first: Annotation, second: Annotation) -> bool:
        return (self._is_role_a(first) and self._is_role_b(second)) or (
            self._is_role_a(second) and self._is_role_b(first)
        )

    @staticmethod
    def _is_role_a(annotation: Annotation) -> bool:
        return annotation.sublabel in MERGEABLE_SUBLABELS

    @staticmethod
    def _is_role_b(annotation: Annotation) -> bool:
        return (
            annotation.sublabel in BROADER_SUBLABELS
            or annotation.label == "Punctuation"
            or (annotation.label == "Spelling" and annotation.sublabel == "Other")
            or (annotation.label == "Rephrasing" and annotation.sublabel == "Other")
        )

    def _passes_multi_word_check(self, first: Annotation, second: Annotation) -> bool:
        return all(
            annotation.sublabel not in MULTI_WORD_REQUIRED_SUBLABELS
            or self._meets_multi_word_requirement(annotation)
            for annotation in (first, second)
        )

    def _meets_multi_word_requirement(self, annotation: Annotation) -> bool:
        text = (
            self._extract_source(annotation)
            if self._is_deletion(annotation)
            else _java_trim(annotation.replacement or "")
        )
        return self._count_words(text) >= 2

    def _has_word_overlap(self, first: Annotation, second: Annotation) -> bool:
        first_replacement = self._is_replacement(first)
        second_replacement = self._is_replacement(second)
        if first_replacement and second_replacement:
            source_first = self._moved_words(
                self._filtered_words(self._extract_source(first)),
                self._filtered_words(first.replacement or ""),
            )
            target_first = self._moved_words(
                self._filtered_words(first.replacement or ""),
                self._filtered_words(self._extract_source(first)),
            )
            source_second = self._moved_words(
                self._filtered_words(self._extract_source(second)),
                self._filtered_words(second.replacement or ""),
            )
            target_second = self._moved_words(
                self._filtered_words(second.replacement or ""),
                self._filtered_words(self._extract_source(second)),
            )
            return self._substring_overlap(source_first, target_second) or self._substring_overlap(
                target_first, source_second
            )
        return self._substring_overlap(
            self._words_for_overlap(first, second),
            self._words_for_overlap(second, first),
        )

    def _words_for_overlap(self, annotation: Annotation, other: Annotation) -> set[str]:
        if self._is_insertion(annotation):
            return self._filtered_words(annotation.replacement or "")
        if self._is_deletion(annotation):
            return self._filtered_words(self._extract_source(annotation))
        source_words = self._filtered_words(self._extract_source(annotation))
        target_words = self._filtered_words(annotation.replacement or "")
        if self._is_deletion(other):
            return self._moved_words(target_words, source_words)
        return self._moved_words(source_words, target_words)

    @staticmethod
    def _moved_words(words: set[str], other_side: set[str]) -> set[str]:
        if not other_side:
            return words
        return {
            word
            for word in words
            if not any(word in other or other in word for other in other_side)
        }

    @staticmethod
    def _strip_punctuation(token: str) -> str:
        # Java indexes String values as UTF-16 code units and calls the ``char``
        # overload of Character.isLetterOrDigit.  In particular, numeric
        # characters outside category Nd are not digits, and each half of a
        # supplementary-plane character is a non-alphanumeric surrogate here.
        encoded = token.encode("utf-16-le", errors="surrogatepass")
        code_units = [
            encoded[index] | (encoded[index + 1] << 8) for index in range(0, len(encoded), 2)
        ]
        start = 0
        end = len(code_units)
        while start < end and not _java_is_letter_or_digit_utf16_unit(code_units[start]):
            start += 1
        while end > start and not _java_is_letter_or_digit_utf16_unit(code_units[end - 1]):
            end -= 1
        return encoded[start * 2 : end * 2].decode("utf-16-le", errors="surrogatepass")

    def _filtered_words(self, text: str | None) -> set[str]:
        words: set[str] = set()
        if text is None:
            return words
        for word in _java_split_space(text.lower()):
            trimmed = self._strip_punctuation(_java_trim(word))
            if not trimmed or _java_is_blank(trimmed):
                continue
            if trimmed in EXCLUDED_OVERLAP_WORDS:
                continue
            words.add(trimmed)
        return words

    @staticmethod
    def _substring_overlap(first: set[str], second: set[str]) -> bool:
        return any(a in b or b in a for a in first for b in second)

    def _is_gap_untouched(
        self, first: Annotation, second: Annotation, all_annotations: list[Annotation]
    ) -> bool:
        gap_start = first.end_token if self.tokenized else first.end
        gap_end = second.start_token if self.tokenized else second.start
        if gap_start is None or gap_end is None:
            return False
        if gap_start >= gap_end:
            return True
        for other in all_annotations:
            if other is first or other is second:
                continue
            other_start = other.start_token if self.tokenized else other.start
            position = other_start if other_start is not None else 0
            if gap_start <= position < gap_end:
                return False
        return True

    def _extract_source(self, annotation: Annotation) -> str:
        if self.tokenized:
            start = annotation.start_token
            end = annotation.end_token
            if (
                start is None
                or end is None
                or start < 0
                or end > len(self.source_tokens)
                or start >= end
            ):
                return ""
            return " ".join(self.source_tokens[start:end])
        start = annotation.start
        end = annotation.end
        if start is None or end is None or start < 0 or end > utf16_len(self.source) or start > end:
            return ""
        return utf16_slice(self.source, start, end)

    def _create_merged_as_word_order(self, first: Annotation, second: Annotation) -> Annotation:
        if (
            self._is_replacement(first)
            and self._is_replacement(second)
            and first.sublabel != "WordOrder"
            and second.sublabel != "WordOrder"
        ):
            sublabel = "Compound_Complex" if self.granular_compound else "Compound"
            return self._build_merged(first, second, "Other", sublabel)
        return self._build_merged(first, second, "SentenceStructure", "WordOrder")

    def _create_merged_as_digit_formatting(
        self, first: Annotation, second: Annotation
    ) -> Annotation:
        return self._build_merged(first, second, "Formatting", "Other")

    def _build_merged(
        self,
        first: Annotation,
        second: Annotation,
        label: str,
        sublabel: str,
    ) -> Annotation:
        if self.tokenized:
            assert first.start_token is not None and second.end_token is not None
            return Annotation.tokenized(
                first.start_token,
                second.end_token,
                label,
                sublabel,
                self._replacement_tokenized(first, second),
            )
        assert first.start is not None and second.end is not None
        return Annotation.non_tokenized(
            first.start,
            second.end,
            label,
            sublabel,
            self._replacement_non_tokenized(first, second),
        )

    def _replacement_tokenized(self, first: Annotation, second: Annotation) -> str:
        parts: list[str] = []
        if first.replacement and not _java_is_blank(first.replacement):
            parts.append(first.replacement)
        assert first.end_token is not None and second.start_token is not None
        if first.end_token < second.start_token <= len(self.source_tokens):
            between = " ".join(self.source_tokens[first.end_token : second.start_token])
            if between and not _java_is_blank(between):
                parts.append(between)
        if second.replacement and not _java_is_blank(second.replacement):
            parts.append(second.replacement)
        return " ".join(parts)

    def _replacement_non_tokenized(self, first: Annotation, second: Annotation) -> str:
        assert first.end is not None and second.start is not None
        between = utf16_slice(self.source, first.end, second.start)
        return (first.replacement or "") + between + (second.replacement or "")
