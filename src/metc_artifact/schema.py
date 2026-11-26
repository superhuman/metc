"""Authoritative benchmark and annotation schemas used by the METC paper.

The historical experiment files use two edit encodings. English and Chinese
store character spans as five-element lists, while the other benchmarks store
token spans as dictionaries. This module names that contract in one place and
provides a stable public dataset identifier for every paper setting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    experiment_code: str
    filename: str
    language: str
    corpus: str
    edit_format: str
    evaluation_span_unit: str
    systems: tuple[str, ...]
    target_tokens_key: str | None = None
    legacy_ids: tuple[str, ...] = ()


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="en_conll14_ann0",
        experiment_code="en",
        filename="edits_by_sent.jsonl",
        language="English",
        corpus="CoNLL-2014 annotator 0",
        edit_format="char_tuple",
        evaluation_span_unit="unicode_codepoint",
        systems=("gold", "metc", "metc_teacher", "errant", "jp"),
    ),
    DatasetSpec(
        dataset_id="en_conll14_ann1",
        experiment_code="en1",
        filename="edits_by_sent_ann1.jsonl",
        language="English",
        corpus="CoNLL-2014 annotator 1",
        edit_format="char_tuple",
        evaluation_span_unit="unicode_codepoint",
        systems=("gold", "metc", "metc_teacher", "errant", "jp"),
    ),
    DatasetSpec(
        dataset_id="de_merlin",
        experiment_code="de",
        filename="edits_by_sent_de.jsonl",
        language="German",
        corpus="MERLIN German",
        edit_format="token_dict",
        evaluation_span_unit="token",
        target_tokens_key="th1_tokens",
        systems=("gold", "metc", "metc_teacher", "ger", "jp"),
    ),
    DatasetSpec(
        dataset_id="cs_merlin",
        experiment_code="cs",
        filename="edits_by_sent_cs.jsonl",
        language="Czech",
        corpus="MERLIN Czech",
        edit_format="token_dict",
        evaluation_span_unit="token",
        target_tokens_key="th1_tokens",
        systems=("gold", "metc", "metc_teacher", "jp"),
    ),
    DatasetSpec(
        dataset_id="ko_kolla",
        experiment_code="ko",
        filename="edits_by_sent_ko.jsonl",
        language="Korean",
        corpus="KoLLA",
        edit_format="token_dict",
        evaluation_span_unit="token",
        target_tokens_key="cor_tokens",
        systems=("gold", "metc", "metc_teacher", "jp"),
    ),
    DatasetSpec(
        dataset_id="uk_uagec",
        experiment_code="uk",
        filename="edits_by_sent_uk.jsonl",
        language="Ukrainian",
        corpus="UA-GEC",
        edit_format="token_dict",
        evaluation_span_unit="token",
        target_tokens_key="cor_tokens",
        systems=("gold", "metc", "metc_teacher", "jp"),
    ),
    DatasetSpec(
        dataset_id="zh_mucgec",
        experiment_code="zh",
        filename="edits_by_sent_zh.jsonl",
        language="Chinese",
        corpus="MuCGEC",
        edit_format="char_tuple",
        evaluation_span_unit="unicode_codepoint",
        systems=("gold", "metc", "metc_teacher", "jp"),
    ),
)


_BY_ID = {
    identifier: spec
    for spec in DATASETS
    for identifier in (spec.dataset_id, spec.experiment_code, *spec.legacy_ids)
}


def dataset_spec(identifier: str) -> DatasetSpec:
    """Resolve a public ID, experiment code, or documented legacy ID."""

    try:
        return _BY_ID[identifier]
    except KeyError as error:
        raise ValueError(f"unknown METC benchmark: {identifier!r}") from error
