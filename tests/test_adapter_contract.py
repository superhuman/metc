from __future__ import annotations

import json

import pytest

from metc_artifact.adapters import (
    AdapterInputError,
    AdapterOutputError,
    AnnotationRecord,
    NormalizedEdit,
    PairRecord,
    materialize_paper_rows,
    pairs_from_paper_rows,
)
from metc_artifact.adapters.mappings import (
    map_cherrant_gold,
    map_errant_en,
    map_gerrant_de,
    map_jp,
    map_metc,
)
from metc_artifact.schema import dataset_spec


def test_pair_contract_preserves_type_sensitive_item_identity_and_exact_text() -> None:
    number = PairRecord.from_dict(
        {
            "dataset_id": "en",
            "item_id": "1",
            "source": "She go.",
            "correction": "She goes.",
        }
    )
    string = PairRecord.from_dict(
        {
            "dataset_id": "en",
            "item_id": '"1"',
            "source": "She go.",
            "correction": "She goes.",
        }
    )
    assert number.item_id != string.item_id

    with pytest.raises(AdapterInputError, match="canonical JSON"):
        PairRecord.from_dict(
            {
                "dataset_id": "en",
                "item_id": " 1 ",
                "source": "a",
                "correction": "b",
            }
        )


def test_token_pair_and_normalized_record_materialize_to_paper_shape() -> None:
    spec = dataset_spec("de")
    row = {
        "idx": 7,
        "orig_tokens": ["Ich", "gehen"],
        "th1_tokens": ["Ich", "gehe"],
        "gold": [],
        "metc": [],
        "metc_teacher": [],
        "ger": [],
        "jp": [],
    }
    pairs = pairs_from_paper_rows([row], spec)
    assert pairs[0].item_id == "7"
    assert pairs[0].source == "Ich gehen"
    record = AnnotationRecord(
        dataset_id=spec.dataset_id,
        item_id="7",
        system="ger",
        span_unit="token",
        edits=(
            NormalizedEdit(
                start=1,
                end=2,
                replacement="gehe",
                native_label="R:VERB:SVA",
                mapped_label="G_Agr",
            ),
        ),
    )
    materialized = materialize_paper_rows([row], pairs, [record], spec, system="ger")
    assert materialized[0]["ger"] == [
        {
            "orig_start": 1,
            "orig_end": 2,
            "th1_start": -1,
            "th1_end": -1,
            "raw_label": "R:VERB:SVA",
            "target": "G_Agr",
            "replacement": "gehe",
        }
    ]
    assert row["ger"] == []


def test_frozen_mapping_helpers_match_paper_policies() -> None:
    assert map_metc("en_conll14_ann0", "Grammar/Verb") == "Wci"
    assert map_metc("de_merlin", "Grammar/Verb") == "G_Verb"
    assert map_metc("ko_kolla", "Spelling/Misspelling") == "R:SPELL"
    assert map_metc("uk_uagec", "Grammar/Preposition") == "G/Prep"
    assert map_metc("zh_mucgec", "Grammar/Noun") == "NOUN"
    assert map_errant_en("M:VERB") == "V0"
    assert map_gerrant_de("R:SPELL") == "O_Graph"
    assert map_jp("de_merlin", "R:ADP -> ADP") == "G_Prep"
    assert map_jp("ko_kolla", "R:NOUN -> NOUN") == "R:NOUN -> NOUN"
    assert map_jp("uk_uagec", "R:ADP -> ADP") == "G/Prep"
    assert map_jp("zh_mucgec", "R:PROPN -> NOUN") == "NOUN"
    assert map_cherrant_gold("R:PUNCT") == "PUNCT"


def test_pair_json_is_plain_data_not_python_identity() -> None:
    pair = PairRecord.from_dict(
        {
            "dataset_id": "zh",
            "item_id": json.dumps("doc-1", ensure_ascii=False),
            "source": "你好",
            "correction": "您好",
        }
    )
    assert pair.to_dict()["item_id"] == '"doc-1"'


def test_normalized_edits_reject_coerced_or_out_of_bounds_spans() -> None:
    with pytest.raises(AdapterOutputError, match="exact integers"):
        NormalizedEdit(
            start=True,
            end=1,
            replacement="x",
            native_label="R:NOUN",
            mapped_label="NOUN",
        )
    with pytest.raises(AdapterOutputError, match="exact integers"):
        NormalizedEdit.from_dict(
            {
                "start": 0.5,
                "end": 1,
                "replacement": "x",
                "native_label": "R:NOUN",
                "mapped_label": "NOUN",
            }
        )

    spec = dataset_spec("en")
    pair = PairRecord(spec.dataset_id, "0", "abc", "abcd")
    record = AnnotationRecord(
        dataset_id=spec.dataset_id,
        item_id="0",
        system="errant",
        span_unit=spec.evaluation_span_unit,
        edits=(NormalizedEdit(2, 4, "d", "R:NOUN", "Nn"),),
    )
    with pytest.raises(AdapterOutputError, match="exceeds source limit"):
        record.validate_for_pair(pair, spec)


def test_materializer_rejects_duplicate_pair_or_annotation_ids() -> None:
    spec = dataset_spec("en")
    row = {"idx": 0, "source": "a", "target": "b"}
    pair = PairRecord(spec.dataset_id, "0", "a", "b")
    record = AnnotationRecord(
        spec.dataset_id,
        "0",
        "errant",
        spec.evaluation_span_unit,
        (),
    )
    with pytest.raises(AdapterOutputError, match="pairs contain duplicate"):
        materialize_paper_rows(
            [row], [pair, pair], [record], spec, system="errant"
        )
    with pytest.raises(AdapterOutputError, match="annotations contain duplicate"):
        materialize_paper_rows(
            [row], [pair], [record, record], spec, system="errant"
        )
