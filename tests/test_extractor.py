from metc.extractor import (
    compute_annotated_edits_result,
    compute_annotated_edits_result_tokenized,
)
from metc.extractor.tokenizer import tokenize_text
from metc.offsets import utf16_slice


def test_marker_shape_and_simple_edit():
    result = compute_annotated_edits_result("She go.", "She went.")
    assert result["annotated"] == "She [EDIT] go=>went [/EDIT]."
    assert result["edits"] == [
        {
            "edit": "go=>went",
            "start": 4,
            "end": 6,
            "source": "go",
            "target": "went",
        }
    ]


def test_utf16_offsets_follow_emoji_prefix():
    source = "🇨🇴 i mean"
    result = compute_annotated_edits_result(source, "🇨🇴 I mean,")
    for edit in result["edits"]:
        assert utf16_slice(source, edit["start"], edit["end"]) == edit["source"]
    assert all(edit["start"] >= 5 for edit in result["edits"])


def test_cjk_and_korean_tokenization():
    assert [token for token, _, _ in tokenize_text("你好世界")] == list("你好世界")
    assert [token for token, _, _ in tokenize_text("한국어 텍스트")] == [
        "한국어",
        " ",
        "텍스트",
    ]


def test_cjk_insertion_is_zero_width():
    [edit] = compute_annotated_edits_result("你好", "你很好")["edits"]
    assert (edit["source"], edit["target"], edit["start"], edit["end"]) == (
        "",
        "很",
        1,
        1,
    )


def test_cleanup_semantic_resplits_unchanged_word():
    result = compute_annotated_edits_result(
        "Yeah its a weird looking bag",
        "Yeah, it's a weird-looking bag",
        cleanup_semantic=True,
    )
    assert [(edit["source"], edit["target"]) for edit in result["edits"]] == [
        (" its", ", it's"),
        ("weird looking", "weird-looking"),
    ]


def test_exotic_space_normalization_avoids_overlapping_edits():
    result = compute_annotated_edits_result("Wie\u00a0geth es dir?", "Wie geht es dir?")
    assert [(edit["source"], edit["target"]) for edit in result["edits"]] == [("geth", "geht")]


def test_tokenized_offsets_count_tokens():
    result = compute_annotated_edits_result_tokenized("( word", '" Word')
    assert [
        (edit["source"], edit["target"], edit["startToken"], edit["endToken"])
        for edit in result["edits"]
    ] == [("(", '"', 0, 1), ("word", "Word", 1, 2)]


def test_tokenized_offsets_preserve_interior_empty_tokens():
    result = compute_annotated_edits_result_tokenized("a  b", "a  c")
    assert [
        (edit["source"], edit["target"], edit["startToken"], edit["endToken"])
        for edit in result["edits"]
    ] == [("b", "c", 2, 3)]
