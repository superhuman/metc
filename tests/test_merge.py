from metc.merge import AnnotationMerger
from metc.offsets import utf16_len
from metc.types import Annotation


def test_non_tokenized_word_order_merge_matches_service_spec():
    source = "quickly he ran"
    annotations = [
        Annotation.non_tokenized(0, 8, "SentenceStructure", "WordOrder", ""),
        Annotation.non_tokenized(14, 14, "SentenceStructure", "WordOrder", " quickly"),
    ]
    assert AnnotationMerger(
        source, tokenized=False, multilingual=False, granular_compound=False
    ).merge(annotations) == [
        Annotation.non_tokenized(0, 14, "SentenceStructure", "WordOrder", "he ran quickly")
    ]


def test_word_order_merge_uses_utf16_offsets():
    source = "😂 quickly he ran"
    annotations = [
        Annotation.non_tokenized(3, 11, "SentenceStructure", "WordOrder", ""),
        Annotation.non_tokenized(
            utf16_len(source),
            utf16_len(source),
            "SentenceStructure",
            "WordOrder",
            " quickly",
        ),
    ]
    [merged] = AnnotationMerger(
        source, tokenized=False, multilingual=False, granular_compound=False
    ).merge(annotations)
    assert (merged.start, merged.end, merged.replacement) == (
        3,
        utf16_len(source),
        "he ran quickly",
    )


def test_multilingual_three_token_limit_is_inclusive():
    source = "quickly a b c ran"
    annotations = [
        Annotation.tokenized(0, 1, "SentenceStructure", "WordOrder", ""),
        Annotation.tokenized(4, 4, "SentenceStructure", "WordOrder", "quickly"),
    ]
    assert len(AnnotationMerger(source, tokenized=True).merge(annotations)) == 1


def test_word_order_pass_does_not_chain_three_annotations():
    source = "the cat sat on mat the"
    annotations = [
        Annotation.tokenized(0, 1, "SentenceStructure", "WordOrder", ""),
        Annotation.tokenized(2, 2, "Grammar", "Noun", "the"),
        Annotation.tokenized(4, 4, "SentenceStructure", "WordOrder", "the"),
    ]
    result = AnnotationMerger(
        source, tokenized=True, multilingual=False, granular_compound=False
    ).merge(annotations)
    assert len(result) == 2
    assert result[0].replacement == "cat the"


def test_two_insertions_and_two_deletions_do_not_merge():
    inserts = [
        Annotation.tokenized(1, 1, "SentenceStructure", "WordOrder", "quickly"),
        Annotation.tokenized(3, 3, "Grammar", "Adverb", "quickly"),
    ]
    assert len(AnnotationMerger("he ran to home", tokenized=True).merge(inserts)) == 2
    deletes = [
        Annotation.tokenized(1, 2, "SentenceStructure", "WordOrder", ""),
        Annotation.tokenized(3, 4, "Grammar", "Adverb", ""),
    ]
    assert len(AnnotationMerger("he quickly ran quickly home", tokenized=True).merge(deletes)) == 2


def test_digit_formatting_pass_chains_all_adjacent_chunks():
    source = "value ( 9,000,000 )"
    annotations = [
        Annotation.non_tokenized(7, 7, "Formatting", "Other", "9"),
        Annotation.non_tokenized(8, 17, "Formatting", "Other", "000"),
        Annotation.non_tokenized(18, 18, "Formatting", "Other", "000"),
    ]
    [merged] = AnnotationMerger(
        source, tokenized=False, multilingual=False, granular_compound=False
    ).merge(annotations)
    assert (merged.start, merged.end, merged.label, merged.sublabel) == (
        7,
        18,
        "Formatting",
        "Other",
    )
    assert merged.replacement == "9 000 000"


def test_annotation_inside_gap_blocks_pair_merge():
    source = "quickly . a b c ran"
    annotations = [
        Annotation.tokenized(0, 1, "SentenceStructure", "WordOrder", ""),
        Annotation.tokenized(1, 2, "Punctuation", "Other", ""),
        Annotation.tokenized(5, 5, "SentenceStructure", "WordOrder", "quickly"),
    ]
    assert len(AnnotationMerger(source, tokenized=True).merge(annotations)) == 3


def test_strip_punctuation_matches_java_letter_or_digit_categories():
    strip = AnnotationMerger._strip_punctuation

    # Python considers No (superscript two) and Nl (Roman numeral eight)
    # alphanumeric; Java Character.isLetterOrDigit(char) does not.
    assert strip("²Ⅷword²Ⅷ") == "word"
    # Both runtimes retain decimal digits (Nd), including non-ASCII ones.
    assert strip("٢word٢") == "٢word٢"
    # Only boundaries are stripped; punctuation-like interior units remain.
    assert strip("a²b") == "a²b"


def test_strip_punctuation_matches_java_utf16_char_semantics_for_astral_text():
    strip = AnnotationMerger._strip_punctuation

    # Java charAt returns the two surrogate units separately, so even astral
    # letters and decimal digits are non-alphanumeric at token boundaries.
    assert strip("𐐀word𐐀") == "word"  # DESERET CAPITAL LETTER LONG I
    assert strip("𐒠word𐒠") == "word"  # OSMANYA DIGIT ZERO
    # Astral units inside BMP-alphanumeric boundaries are not inspected.
    assert strip("a𐐀b") == "a𐐀b"


def test_java_non_alphanumeric_boundary_tokens_do_not_trigger_word_overlap():
    for word in ("²", "Ⅷ", "𐐀", "𐒠"):
        source = f"{word} stays"
        annotations = [
            Annotation.tokenized(0, 1, "Grammar", "Noun", ""),
            Annotation.tokenized(2, 2, "Grammar", "Noun", word),
        ]
        assert len(AnnotationMerger(source, tokenized=True).merge(annotations)) == 2


def test_java_decimal_digit_boundary_token_still_triggers_word_overlap():
    source = "٢ stays"
    annotations = [
        Annotation.tokenized(0, 1, "Grammar", "Noun", ""),
        Annotation.tokenized(2, 2, "Grammar", "Noun", "٢"),
    ]
    assert len(AnnotationMerger(source, tokenized=True).merge(annotations)) == 1
