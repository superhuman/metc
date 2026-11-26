from types import SimpleNamespace

import pytest
import torch

from metc.errors import CheckpointContractError, MarkerError
from metc.model import TokenClassifier, count_well_formed_edits
from metc.pipeline import METC


class FakeTokenizer:
    unk_token_id = 9
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __init__(self):
        self.last_padding = None
        self.last_texts = None

    def encode(self, text, add_special_tokens=False):
        return {"[EDIT]": [256000], "[/EDIT]": [256001]}.get(text, [9])

    def __call__(
        self,
        texts,
        *,
        max_length,
        truncation,
        padding,
        return_tensors,
    ):
        self.last_padding = padding
        self.last_texts = list(texts)
        rows = []
        for text in texts:
            pieces = text.replace("[EDIT]", " [EDIT] ").replace("[/EDIT]", " [/EDIT] ").split()
            ids = [10]
            for piece in pieces:
                ids.append(256000 if piece == "[EDIT]" else 256001 if piece == "[/EDIT]" else 20)
            ids.append(11)
            rows.append(ids[:max_length])
        width = max_length if padding == "max_length" else max(map(len, rows))
        input_ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(0).long()}


class FakeModel:
    def __init__(self, num_labels=42):
        self.config = SimpleNamespace(
            num_labels=num_labels, max_position_embeddings=8192, pad_token_id=0
        )

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask):
        logits = torch.full((*input_ids.shape, self.config.num_labels), -10.0)
        for row in range(input_ids.shape[0]):
            seen = 0
            for column in (input_ids[row] == 256000).nonzero().flatten().tolist():
                logits[row, column, 22 if seen == 0 else 37] = 10.0
                seen += 1
        return SimpleNamespace(logits=logits)


def make_classifier(max_length=512):
    return TokenClassifier(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        max_sequence_length=max_length,
    )


def test_contract_requires_exactly_42_labels():
    with pytest.raises(CheckpointContractError, match="42"):
        TokenClassifier(model=FakeModel(41), tokenizer=FakeTokenizer(), device="cpu")


def test_unknown_inference_profile_is_rejected():
    with pytest.raises(ValueError, match="inference_profile"):
        TokenClassifier(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            device="cpu",
            inference_profile="almost-paper",
        )


def test_paper_profile_fails_closed_without_cuda():
    with pytest.raises(CheckpointContractError, match="CUDA"):
        TokenClassifier(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            device="cpu",
            inference_profile="paper",
        )


def test_paper_profile_pins_fixed_length_shape(monkeypatch):
    monkeypatch.setattr(TokenClassifier, "_resolve_device", staticmethod(lambda _device: "cuda"))
    classifier = TokenClassifier(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="auto",
        inference_profile="paper",
    )
    metadata = classifier.inspect_checkpoint()
    assert metadata["precision"] == "bfloat16"
    assert metadata["attention_implementation"] == "flash_attention_2"
    assert metadata["inference_max_sequence_length"] == 512
    assert metadata["pretokenization_character_limit"] == 3584
    assert metadata["padding"] == "max_length"
    assert metadata["padding_side"] == "right"
    assert metadata["truncation_side"] == "right"
    assert metadata["truncated_edit_fallback"] == "Other/Compound_Complex"
    assert metadata["reference_compile"] is True


def test_predictions_are_taken_only_at_edit_open_tokens():
    predictions = make_classifier().predict_marked(
        "She [EDIT] go=>went [/EDIT] and [EDIT] recieve=>receive [/EDIT]."
    )
    assert [prediction.label_id for prediction in predictions] == [22, 37]
    assert predictions[0].label == "Grammar"
    assert predictions[0].sublabel == "VerbTenseForm"


def test_portable_profile_uses_dynamic_batch_padding():
    classifier = make_classifier()
    classifier.predict_marked("[EDIT] a=>b [/EDIT]")
    assert classifier.tokenizer.last_padding is True


def test_pickle_checkpoint_is_rejected_before_loading(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"not a safe checkpoint")
    with pytest.raises(CheckpointContractError, match="pickle-based"):
        TokenClassifier(tmp_path, device="cpu")


def test_missing_tokenizer_points_to_explicit_preparation_command(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"safe placeholder")

    with pytest.raises(CheckpointContractError, match=r"metc prepare-tokenizer --model"):
        TokenClassifier(tmp_path, device="cpu")


def test_batch_preserves_zero_edit_items_and_order():
    results = make_classifier().predict_marked_batch(
        ["plain text", "[EDIT] a=>b [/EDIT]", "plain again"], [0, 1, 0]
    )
    assert [len(result) for result in results] == [0, 1, 0]


def test_retained_open_marker_is_classified_when_close_marker_is_truncated():
    predictions = make_classifier(max_length=3).predict_marked("[EDIT] a=>b [/EDIT]")
    assert [prediction.label_id for prediction in predictions] == [22]


def test_truncated_open_markers_receive_serving_fallback_in_edit_order():
    predictions = make_classifier(max_length=4).predict_marked(
        "[EDIT] a=>b [/EDIT] [EDIT] c=>d [/EDIT]"
    )
    assert [prediction.label_id for prediction in predictions] == [22, 23]
    assert predictions[1].label == "Other"
    assert predictions[1].sublabel == "Compound_Complex"
    assert predictions[1].score == 0.0


def test_input_is_character_bounded_before_tokenization():
    classifier = make_classifier(max_length=2)
    annotated = "[EDIT] a=>b [/EDIT] trailing text"
    classifier.predict_marked(annotated)
    assert classifier.tokenizer.last_texts == [annotated[:14]]


def test_fallback_is_attached_to_extracted_edit_before_merge():
    result = METC(classifier=make_classifier(max_length=8)).classify(
        "She go home and he are here.",
        "She went home and he is here.",
    )
    assert [(item.label, item.sublabel) for item in result.annotations] == [
        ("Grammar", "VerbTenseForm"),
        ("Other", "Compound_Complex"),
    ]


def test_malformed_markers_are_rejected():
    with pytest.raises(MarkerError):
        count_well_formed_edits("[EDIT] no close")
