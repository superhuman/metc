"""Integration tests that run when the released checkpoint is available."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from metc import METC, TokenClassifier


@pytest.fixture(scope="module")
def checkpoint_path() -> Path:
    configured = os.environ.get("METC_MODEL_PATH")
    if not configured:
        pytest.skip("set METC_MODEL_PATH to run checkpoint integration tests")
    path = Path(configured)
    if not (path / "config.json").is_file():
        pytest.fail(f"METC_MODEL_PATH is not a checkpoint directory: {path}")
    return path


@pytest.fixture(scope="module")
def classifier(checkpoint_path: Path) -> TokenClassifier:
    return TokenClassifier(checkpoint_path, device="cpu")


def test_released_checkpoint_contract(classifier: TokenClassifier):
    metadata = classifier.inspect_checkpoint()
    assert metadata["architecture"] == "ModernBertForTokenClassification"
    assert metadata["num_labels"] == 42
    assert metadata["max_position_embeddings"] == 8192
    assert metadata["inference_max_sequence_length"] == 512
    assert metadata["pretokenization_character_limit"] == 3584
    assert metadata["padding_side"] == "right"
    assert metadata["truncation_side"] == "right"
    assert metadata["edit_open_token_id"] == 256000
    assert metadata["edit_close_token_id"] == 256001


@pytest.mark.parametrize(
    "text, expected_label_id",
    [
        ("She [EDIT]go=>went[/EDIT] to school yesterday.", 22),
        ("I [EDIT]dont=>don't[/EDIT] feel well today.", 37),
        ("The cats [EDIT]is=>are[/EDIT] sleeping.", 21),
        ("The meeting is at [EDIT]3pm=>3 p.m.[/EDIT].", 8),
    ],
)
def test_checkpoint_predictions_match_frozen_smoke_cases(
    classifier: TokenClassifier, text: str, expected_label_id: int
):
    [prediction] = classifier.predict_marked(text)
    assert prediction.label_id == expected_label_id


def test_end_to_end_multiscript_offsets(
    classifier: TokenClassifier,
):
    metc = METC(classifier=classifier)
    pairs = [
        ("She go.", "She went."),
        ("🇨🇴 i mean", "🇨🇴 I mean,"),
        ("你好世界", "你好世间"),
        ("한국어 문장 이다.", "한국어 문장이다."),
        ("Це моя сім’я.", "Це наша сім’я."),
    ]
    results = metc.classify_batch(pairs, cleanup_semantic=True)
    assert len(results) == len(pairs)
    assert all(result.annotations for result in results)
