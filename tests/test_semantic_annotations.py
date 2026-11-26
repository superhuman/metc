from __future__ import annotations

import pytest

from metc_artifact.schema import dataset_spec
from metc_artifact.semantic import compare_annotations, semantic_edit


def test_semantic_comparison_ignores_rows_edit_order_and_json_shape() -> None:
    spec = dataset_spec("en")
    first = {
        "sent_idx": 1,
        "src": "She go.",
        "tgt": "She goes.",
        "gold": [[4, 6, "SVA", "SVA", "goes"]],
    }
    second = {
        "sent_idx": 2,
        "src": "A cats.",
        "tgt": "A cat.",
        "gold": [[2, 6, "Nn", "Nn", "cat"]],
    }
    actual_first = {
        "tgt": "She goes.",
        "gold": [[4, 6, "SVA", "SVA", "goes"]],
        "src": "She go.",
        "sent_idx": 1,
    }
    report = compare_annotations(
        [first, second], [second, actual_first], spec, systems=("gold",)
    )
    assert report.equal


def test_semantic_comparison_detects_replacement_and_label_changes() -> None:
    spec = dataset_spec("zh")
    expected = [{
        "idx": "1",
        "src": "你好",
        "tgt": "您好",
        "gold": [[1, 2, "PRON", "R:PRON", "您"]],
    }]
    actual = [{
        "idx": "1",
        "src": "你好",
        "tgt": "您好",
        "gold": [[1, 2, "PRON", "R:NOUN", "他"]],
    }]
    report = compare_annotations(expected, actual, spec, systems=("gold",))
    assert not report.equal
    assert report.differences[0].field == "annotations.gold"


def test_zero_difference_sample_limit_cannot_turn_mismatch_into_equality() -> None:
    spec = dataset_spec("zh")
    expected = [{
        "idx": "1",
        "src": "你好",
        "tgt": "您好",
        "gold": [[1, 2, "PRON", "R:PRON", "您"]],
    }]
    actual = [{
        "idx": "1",
        "src": "你好",
        "tgt": "您好",
        "gold": [[1, 2, "PRON", "R:NOUN", "他"]],
    }]

    report = compare_annotations(
        expected,
        actual,
        spec,
        systems=("gold",),
        max_differences=0,
    )

    assert report.equal is False
    assert report.difference_count == 1
    assert report.differences == ()


def test_token_replacement_is_derived_from_target_coordinates() -> None:
    spec = dataset_spec("de")
    row = {
        "idx": 1,
        "orig_tokens": ["Ich", "gehen"],
        "th1_tokens": ["Ich", "gehe"],
        "gold": [{
            "orig_start": 1,
            "orig_end": 2,
            "th1_start": 1,
            "th1_end": 2,
            "raw_label": "G_Verb/tns",
            "target": "G_Verb/tns",
        }],
    }
    report = compare_annotations([row], [row], spec, systems=("gold",))
    assert report.equal


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (True, 1),
        ("0", 1),
        (-1, 1),
        (0, 3),
        (2, 1),
    ],
)
def test_character_spans_require_true_bounded_integers(
    start: object, end: object
) -> None:
    spec = dataset_spec("en")
    row = {"idx": 1, "src": "ab", "tgt": "ab"}

    with pytest.raises(ValueError, match="source_span"):
        semantic_edit(row, [start, end, "mapped", "native", "x"], spec)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (False, 1),
        ("0", 1),
        (-1, 1),
        (0, 3),
        (2, 1),
    ],
)
def test_token_source_spans_require_true_bounded_integers(
    start: object, end: object
) -> None:
    spec = dataset_spec("de")
    row = {
        "idx": 1,
        "orig_tokens": ["a", "b"],
        "th1_tokens": ["a", "b"],
    }

    with pytest.raises(ValueError, match="source_span"):
        semantic_edit(
            row,
            {
                "orig_start": start,
                "orig_end": end,
                "replacement": "x",
                "raw_label": "native",
                "target": "mapped",
            },
            spec,
        )


@pytest.mark.parametrize(
    ("target_start", "target_end"),
    [
        (True, 1),
        ("0", 1),
        (-1, 0),
        (0, 3),
        (2, 1),
    ],
)
def test_token_target_spans_require_true_bounded_integers(
    target_start: object, target_end: object
) -> None:
    spec = dataset_spec("de")
    row = {
        "idx": 1,
        "orig_tokens": ["a", "b"],
        "th1_tokens": ["a", "b"],
    }

    with pytest.raises(ValueError, match="target_span"):
        semantic_edit(
            row,
            {
                "orig_start": 0,
                "orig_end": 1,
                "th1_start": target_start,
                "th1_end": target_end,
                "replacement": "x",
                "raw_label": "native",
                "target": "mapped",
            },
            spec,
        )


def test_token_target_span_requires_both_coordinates() -> None:
    spec = dataset_spec("de")
    row = {
        "idx": 1,
        "orig_tokens": ["a"],
        "th1_tokens": ["b"],
    }

    with pytest.raises(ValueError, match="both th1_start and th1_end"):
        semantic_edit(
            row,
            {
                "orig_start": 0,
                "orig_end": 1,
                "th1_start": 0,
                "raw_label": "native",
                "target": "mapped",
            },
            spec,
        )


def test_token_target_span_preserves_explicit_unavailable_sentinel() -> None:
    spec = dataset_spec("de")
    row = {
        "idx": 1,
        "orig_tokens": ["a"],
        "th1_tokens": ["b"],
    }

    edit = semantic_edit(
        row,
        {
            "orig_start": 0,
            "orig_end": 1,
            "th1_start": -1,
            "th1_end": -1,
            "replacement": "b",
            "raw_label": "native",
            "target": "mapped",
        },
        spec,
    )

    assert edit.replacement == "b"
