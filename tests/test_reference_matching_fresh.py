from __future__ import annotations

from metc_artifact.reference_matching import (
    Counts,
    MatchEdit,
    match_counts,
    score_reference_matching,
    spans_overlap,
)
from metc_artifact.schema import dataset_spec


def test_bipartite_matching_does_not_double_count_overlapping_edits() -> None:
    system = [MatchEdit(0, 2, "A"), MatchEdit(1, 3, "A")]
    reference = [MatchEdit(0, 3, "A")]

    assert match_counts(
        system, reference, span_predicate=spans_overlap, require_label=True
    ) == Counts(tp=1, fp=1, fn=0)


def test_reference_matching_scores_complete_annotations() -> None:
    spec = dataset_spec("en")
    row = {
        "idx": 0,
        "src": "a b",
        "tgt": "x b",
        **{system: [] for system in spec.systems},
    }
    row["gold"] = [[0, 1, "A", "gold-a", "x"]]
    row["metc"] = [[0, 1, "A", "metc-a", "x"]]
    row["metc_teacher"] = [[0, 1, "B", "teacher-b", "x"]]

    scores = {
        result["system"]: result
        for result in score_reference_matching(
            [row],
            spec,
            annotation_source="fresh_rerun",
            samples=10,
        )
    }

    assert all(result["annotation_source"] == "fresh_rerun" for result in scores.values())
    assert all(result["span_match"] == "overlap" for result in scores.values())
    assert scores["metc"]["h1_span_f05"] == 1.0
    assert scores["metc"]["h2_typed_f05"] == 1.0
    assert scores["metc_teacher"]["h1_span_f05"] == 1.0
    assert scores["metc_teacher"]["h2_typed_f05"] == 0.0


def test_reference_matching_rejects_ambiguous_annotation_source() -> None:
    spec = dataset_spec("en")

    try:
        score_reference_matching(
            [],
            spec,
            annotation_source="provided",
            samples=10,
        )
    except ValueError as error:
        assert "annotation_source" in str(error)
    else:
        raise AssertionError("ambiguous annotation provenance was accepted")
