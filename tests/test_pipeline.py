from metc.errors import MarkerError
from metc.model import count_well_formed_edits
from metc.pipeline import METC
from metc.types import EditPrediction


class FakeClassifier:
    @staticmethod
    def _predictions(text):
        return [
            EditPrediction(22, "Grammar", "VerbTenseForm", 0.99)
            for _ in range(count_well_formed_edits(text))
        ]

    def predict_marked(self, text, expected_edit_count=None):
        result = self._predictions(text)
        assert expected_edit_count == len(result)
        return result

    def predict_marked_batch(self, texts, expected_edit_counts=None):
        result = [self._predictions(text) for text in texts]
        assert expected_edit_counts == [len(item) for item in result]
        return result


def test_complete_non_tokenized_response():
    result = METC(classifier=FakeClassifier()).classify("She go.", "She went.")
    assert result.to_dict() == {
        "classified": "She {go=>went:::Grammar/VerbTenseForm}.",
        "annotations": [
            {
                "start": 4,
                "end": 6,
                "label": "Grammar",
                "sublabel": "VerbTenseForm",
                "replacement": "went",
            }
        ],
    }


def test_no_change_skips_model_and_roundtrips_source():
    result = METC(classifier=FakeClassifier()).classify("No change.", "No change.")
    assert result.classified == "No change."
    assert result.annotations == ()


def test_batch_preserves_input_order_and_no_edit_rows():
    results = METC(classifier=FakeClassifier()).classify_batch(
        [("She go.", "She went."), ("same", "same"), ("I walk.", "I walked.")]
    )
    assert [len(result.annotations) for result in results] == [1, 0, 1]
    assert results[1].classified == "same"


def test_tokenized_mode_returns_token_offsets():
    result = METC(classifier=FakeClassifier()).classify(
        "She go home", "She went home", tokenized=True
    )
    assert result.annotations[0].start_token == 1
    assert result.annotations[0].end_token == 2
    assert result.annotations[0].start is None


def test_tokenized_mode_preserves_interior_empty_tokens_when_rendering():
    result = METC(classifier=FakeClassifier()).classify("a  b", "a  c", tokenized=True)
    assert result.classified == "a  {b=>c:::Grammar/VerbTenseForm}"
    assert result.annotations[0].start_token == 2
    assert result.annotations[0].end_token == 3


def test_reserved_markers_in_user_text_are_rejected():
    try:
        METC(classifier=FakeClassifier()).classify("literal [EDIT] text", "literal text")
    except MarkerError:
        pass
    else:
        raise AssertionError("reserved marker must be rejected")
