from __future__ import annotations

import json
from pathlib import Path

import pytest

from metc_artifact.metrics import (
    JudgeMetricsInputError,
    JudgeVerdictCorpus,
    compute_judge_metrics,
    compute_paper_tables,
    load_judge_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _quality_rows() -> tuple[list[dict], list[dict]]:
    definitions = [
        # row_idx, system, edit_idx, primary span/label, second span/label
        (10, "metc", 0, "Good", "Good", "Good", "Good"),
        (10, "metc", 1, "Good", "Acceptable", "Good", "Good"),
        (10, "errant", 0, "Bad", "Good", "Good", "Good"),
        (11, "metc", 0, "Good", "Good", "Good", "Acceptable"),
        (11, "errant", 0, "Good", "Good", "Acceptable", "Good"),
    ]
    primary = []
    second = []
    for row_idx, system, edit_idx, p_span, p_label, s_span, s_label in definitions:
        edit_id = f"metc-edit-v1:en:{row_idx}:{system}:{edit_idx}"
        common = {
            "edit_id": edit_id,
            "edit_idx": edit_idx,
            "edit_start": edit_idx,
            "edit_end": edit_idx + 1,
            "edit_diff": f"'x{edit_idx}' => 'y{edit_idx}'",
            "errored": False,
            "label": "Grammar/Test",
            "row_idx": row_idx,
            "src": f"source {row_idx}",
            "system": system,
            "tgt": f"target {row_idx}",
        }
        primary.append(
            {
                **common,
                "judge": "openai_direct_gpt54_flex",
                "span_quality": p_span,
                "label_quality": p_label,
            }
        )
        second.append(
            {
                **common,
                "judge": "vai_gemini35_flash",
                "span_quality": s_span,
                "label_quality": s_label,
            }
        )
    return primary, second


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    _write_jsonl(
        root / "benchmarks/en_conll14_ann0/annotations.jsonl",
        [
            {"sent_idx": 11, "src": "source 11", "tgt": "target 11"},
            {"sent_idx": 10, "src": "source 10", "tgt": "target 10"},
        ],
    )
    primary, second = _quality_rows()
    _write_jsonl(
        root / "judges/canonical/primary/en_conll14_ann0.jsonl",
        list(reversed(primary)),
    )
    _write_jsonl(
        root / "judges/canonical/second/en_conll14_ann0.jsonl",
        second,
    )
    return root


def _fresh_record(row: dict, *, provider: str, model: str) -> dict:
    output = {
        "span_quality": row["span_quality"],
        "label_quality": row["label_quality"],
        "span_reason": "synthetic",
        "label_reason": "synthetic",
    }
    input_row = {
        key: value
        for key, value in row.items()
        if key not in {"judge", "span_quality", "label_quality", "errored"}
    }
    return {
        "schema_version": "metc-fresh-rerun-v1",
        "source": "fresh_rerun",
        "authoritative_frozen_output": False,
        "run_kind": "acceptability",
        "provider": provider,
        "model": model,
        "stable_edit_ids": [row["edit_id"]],
        "input": input_row,
        "outputs_by_edit_id": {row["edit_id"]: output},
        "error": None,
    }


def test_frozen_loader_joins_stable_edits_and_uses_benchmark_order(tmp_path: Path):
    corpus = load_judge_rows(_data_root(tmp_path), judge_source="frozen_authoritative")

    assert corpus.judge_source == "frozen_authoritative"
    assert len(corpus.rows) == 5
    row_11 = next(row for row in corpus.rows if row["row_idx"] == 11)
    row_10 = next(row for row in corpus.rows if row["row_idx"] == 10)
    assert row_10["item_order"] == 0
    assert row_11["item_order"] == 1
    assert row_10["primary_judge"] == "openai_direct_gpt54_flex"
    assert row_10["second_judge"] == "vai_gemini35_flash"


def test_raw_metrics_reconstruct_quality_adequacy_agreement_and_pairs(
    tmp_path: Path,
):
    corpus = load_judge_rows(_data_root(tmp_path), judge_source="frozen_authoritative")
    metrics = compute_judge_metrics(corpus, samples=100, seed=42)

    assert metrics["judge_source"] == "frozen_authoritative"
    assert metrics["n_edits"] == 5
    primary = metrics["acceptability"]["primary"]["en_conll14_ann0"]
    assert primary["metc"]["label"]["Good"]["count"] == 2
    assert primary["metc"]["label"]["Acceptable"]["count"] == 1
    assert primary["errant"]["span"]["Bad"]["pct"] == 50.0

    adequacy = metrics["sentence_adequacy"]
    assert adequacy["primary"]["en_conll14_ann0"]["metc"]["adequacy_pct"] == 75.0
    assert adequacy["primary"]["en_conll14_ann0"]["errant"]["adequacy_pct"] == 50.0
    assert adequacy["second"]["en_conll14_ann0"]["metc"]["adequacy_pct"] == 75.0
    assert adequacy["second"]["en_conll14_ann0"]["errant"]["adequacy_pct"] == 75.0

    agreement = metrics["judge_agreement"]["en_conll14_ann0"]
    assert agreement["overall"]["span_agreement_pct"] == 60.0
    assert agreement["overall"]["label_agreement_pct"] == 60.0
    assert agreement["systems"]["metc"]["label_agreement_pct"] == pytest.approx(100 / 3)
    assert agreement["systems"]["errant"]["span_agreement_pct"] == 0.0

    adequacy_pair = metrics["paired_sentence_adequacy"]["primary"]["en_conll14_ann0"]["comparisons"]
    assert adequacy_pair["metc_minus_errant"]["difference_points"] == 25.0
    assert adequacy_pair["errant_minus_metc"]["difference_points"] == -25.0
    label_pair = metrics["paired_label_good"]["primary"]["en_conll14_ann0"]
    assert label_pair["comparisons"]["metc_minus_errant"]["difference_points"] == pytest.approx(
        -100 / 3
    )
    assert label_pair["comparisons"]["metc_minus_errant"]["seed"] == 42


def test_metrics_and_tables_are_independent_of_raw_row_order(tmp_path: Path):
    corpus = load_judge_rows(_data_root(tmp_path), judge_source="frozen_authoritative")
    reversed_corpus = JudgeVerdictCorpus(
        judge_source=corpus.judge_source,
        rows=tuple(reversed(corpus.rows)),
    )

    forward = compute_paper_tables(corpus, samples=100, seed=42)
    reversed_result = compute_paper_tables(reversed_corpus, samples=100, seed=42)
    assert forward == reversed_result
    assert {row["judge_source"] for table in forward["tables"].values() for row in table} == {
        "frozen_authoritative"
    }
    compact = forward["tables"]["compact_summary"]
    metc = next(row for row in compact if row["system"] == "metc")
    assert metc["primary_adequacy_pct"] == 75.0
    assert metc["label_agreement_pct"] == pytest.approx(100 / 3)


def test_fresh_rerun_is_computed_separately_and_carries_fresh_provenance(
    tmp_path: Path,
):
    data_root = _data_root(tmp_path)
    primary, second = _quality_rows()
    run_root = tmp_path / "fresh-run"
    _write_jsonl(
        run_root / "primary.jsonl",
        [_fresh_record(row, provider="openai", model="gpt-5.4") for row in primary],
    )
    _write_jsonl(
        run_root / "second.jsonl",
        [_fresh_record(row, provider="google", model="gemini-3.5-flash") for row in second],
    )

    corpus = load_judge_rows(
        data_root,
        judge_source="fresh_rerun",
        fresh_run_root=run_root,
    )
    result = compute_paper_tables(corpus, samples=40, seed=42)

    assert corpus.judge_source == "fresh_rerun"
    assert result["judge_source"] == "fresh_rerun"
    assert (
        result["metrics"]["sentence_adequacy"]["primary"]["en_conll14_ann0"]["metc"]["adequacy_pct"]
        == 75.0
    )
    assert all(
        row["judge_source"] == "fresh_rerun" for table in result["tables"].values() for row in table
    )


def test_fresh_loader_rejects_stable_id_attached_to_a_different_edit(
    tmp_path: Path,
) -> None:
    data_root = _data_root(tmp_path)
    primary, second = _quality_rows()
    primary_records = [
        _fresh_record(row, provider="openai", model="gpt-5.4") for row in primary
    ]
    second_records = [
        _fresh_record(row, provider="google", model="gemini-3.5-flash")
        for row in second
    ]
    primary_records[0]["input"]["edit_start"] = 99
    second_records[0]["input"]["edit_start"] = 99
    run_root = tmp_path / "fresh-run"
    _write_jsonl(run_root / "primary.jsonl", primary_records)
    _write_jsonl(run_root / "second.jsonl", second_records)

    with pytest.raises(
        JudgeMetricsInputError,
        match="stable-ID identity differs from authoritative edit_start",
    ):
        load_judge_rows(
            data_root,
            judge_source="fresh_rerun",
            fresh_run_root=run_root,
        )


def test_loader_rejects_incomplete_cross_judge_universe(tmp_path: Path):
    data_root = _data_root(tmp_path)
    second_path = data_root / "judges/canonical/second/en_conll14_ann0.jsonl"
    rows = [json.loads(line) for line in second_path.read_text().splitlines()]
    _write_jsonl(second_path, rows[:-1])

    with pytest.raises(
        JudgeMetricsInputError, match="primary and second judge edit universes differ"
    ):
        load_judge_rows(data_root, judge_source="frozen_authoritative")
