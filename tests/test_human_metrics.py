from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from metc_artifact.metrics import (
    HumanMetricsInputError,
    HumanStudyCorpus,
    JudgeVerdictCorpus,
    compute_human_metrics,
    compute_human_tables,
    load_human_study,
)


def _edit(
    edit_id: str,
    *,
    span: str | None,
    label: str | None,
    unannotatable: bool = False,
) -> dict:
    return {
        "slot": 1,
        "edit_id": edit_id,
        "edit_idx": 0,
        "label": "Grammar/Test",
        "span_quality": span,
        "label_quality": label,
        "unannotatable": unannotatable,
    }


def _assignment(
    *,
    order: int,
    sample: str,
    pass_id: str,
    row_idx: int,
    system: str,
    span: str | None,
    label: str | None,
    unannotatable: bool = False,
) -> dict:
    edit_id = f"metc-edit-v1:de:{row_idx}:{system}:0"
    status = "unannotatable" if unannotatable else "scorable"
    return {
        "dataset_id": "de_merlin",
        "sample_id": sample,
        "assignment_order": order,
        "pass_id": pass_id,
        "item_id": str(row_idx),
        "row_idx": row_idx,
        "system": system,
        "edit_count": 1,
        "status": status,
        "session_unannotatable": False,
        "edits": [
            _edit(
                edit_id,
                span=span,
                label=label,
                unannotatable=unannotatable,
            )
        ],
    }


def _judge_row(row_idx: int, system: str, primary: tuple[str, str], second: tuple[str, str]):
    return {
        "dataset": "de_merlin",
        "item_id": str(row_idx),
        "item_order": row_idx,
        "edit_id": f"metc-edit-v1:de:{row_idx}:{system}:0",
        "edit_idx": 0,
        "system": system,
        "label": "Grammar/Test",
        "primary_span": primary[0],
        "primary_label": primary[1],
        "second_span": second[0],
        "second_label": second[1],
    }


def _corpus() -> HumanStudyCorpus:
    assignments = (
        _assignment(
            order=0,
            sample="m1",
            pass_id="pass-v1:1",
            row_idx=1,
            system="metc",
            span="Good",
            label="Good",
        ),
        _assignment(
            order=1,
            sample="g1",
            pass_id="pass-v1:1",
            row_idx=1,
            system="ger",
            span="Good",
            label="Good",
        ),
        _assignment(
            order=2,
            sample="m2",
            pass_id="pass-v1:1",
            row_idx=2,
            system="metc",
            span="Bad",
            label="Good",
        ),
        _assignment(
            order=3,
            sample="g2",
            pass_id="pass-v1:1",
            row_idx=3,
            system="ger",
            span="Good",
            label="Good",
        ),
        _assignment(
            order=4,
            sample="m1",
            pass_id="pass-v1:2",
            row_idx=1,
            system="metc",
            span="Good",
            label="Acceptable",
        ),
        _assignment(
            order=5,
            sample="g1",
            pass_id="pass-v1:2",
            row_idx=1,
            system="ger",
            span="Good",
            label="Good",
        ),
        _assignment(
            order=6,
            sample="m2",
            pass_id="pass-v1:2",
            row_idx=2,
            system="metc",
            span="Good",
            label="Good",
        ),
        _assignment(
            order=7,
            sample="g2",
            pass_id="pass-v1:2",
            row_idx=3,
            system="ger",
            span=None,
            label=None,
            unannotatable=True,
        ),
    )
    frame = tuple(
        {
            "dataset_id": "de_merlin",
            "sample_id": sample,
            "selection_rank": rank,
            "selected_for_human_study": True,
            "item_id": str(row_idx),
            "row_idx": row_idx,
            "system": system,
            "edit_count": 1,
            "edits": [
                {
                    "slot": 1,
                    "edit_id": f"metc-edit-v1:de:{row_idx}:{system}:0",
                    "edit_idx": 0,
                    "label": "Grammar/Test",
                }
            ],
        }
        for rank, (sample, row_idx, system) in enumerate(
            (("m1", 1, "metc"), ("m2", 2, "metc"), ("g1", 1, "ger"), ("g2", 3, "ger"))
        )
    )
    judges = (
        _judge_row(1, "metc", ("Good", "Good"), ("Good", "Acceptable")),
        _judge_row(2, "metc", ("Good", "Good"), ("Good", "Good")),
        _judge_row(1, "ger", ("Good", "Good"), ("Good", "Good")),
        _judge_row(3, "ger", ("Good", "Good"), ("Acceptable", "Good")),
    )
    return HumanStudyCorpus(
        dataset_id="de_merlin",
        annotator_pool_size=9,
        individual_annotator_ids_available=False,
        passes_per_sample=2,
        selection_rule="synthetic balanced sample",
        assignments=assignments,
        sampling_frame=frame,
        judge_rows=judges,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_raw_human_metrics_respect_passes_and_cluster_repeated_items() -> None:
    metrics = compute_human_metrics(_corpus(), samples=200, seed=42)

    assert metrics["human_source"] == "frozen_authoritative"
    assert metrics["study"]["annotator_pool_size"] == 9
    assert metrics["study"]["assignments"] == 8
    assert metrics["study"]["selected_samples"] == 4
    assert metrics["study"]["repeated_benchmark_items"] == 1
    assert metrics["study"]["unique_benchmark_items"] == 3

    iaa = metrics["inter_annotator_agreement"]
    assert iaa["n_double_scorable_edits"] == 3
    assert iaa["span_agreement_pct"] == pytest.approx(200 / 3)
    assert iaa["label_agreement_pct"] == pytest.approx(200 / 3)
    assert metrics["human_vs_judges"]["primary"]["n_human_verdicts"] == 7
    assert metrics["judge_vs_judge_on_human_edits"]["n_unique_edits"] == 4
    assert (
        metrics["judge_vs_judge_on_human_edits"]["weighting"]
        == "once_per_unique_scorable_edit"
    )

    assert metrics["human_edit_quality"]["metc"]["label"]["Good"]["pct"] == 75.0
    assert metrics["human_edit_quality"]["ger"]["label"]["Good"]["pct"] == 100.0
    assert metrics["sentence_adequacy"]["metc"]["adequacy_pct"] == 62.5
    assert metrics["sentence_adequacy"]["ger"]["adequacy_pct"] == 100.0

    comparison = metrics["system_comparisons"]["label_good"]
    independent = comparison["independent_human_sample"]["ger_minus_metc"]
    clustered = comparison["benchmark_item_clustered"]["ger_minus_metc"]
    assert independent["difference_points"] == 25.0
    assert independent["resampling_unit"] == "human_sample"
    assert clustered["difference_points"] == 25.0
    assert clustered["resampling_unit"] == "benchmark_item"
    assert clustered["n_clusters"] == 3
    assert clustered["n_shared_clusters"] == 1


def test_human_tables_include_compact_detailed_agreement_and_sensitivity_rows() -> None:
    result = compute_human_tables(_corpus(), samples=80, seed=42)

    assert result["schema_version"] == "metc-raw-human-tables-v1"
    tables = result["tables"]
    assert set(tables) == {
        "compact_summary",
        "edit_quality",
        "sentence_adequacy",
        "agreement",
        "system_comparisons",
        "adequacy_sensitivity",
    }
    compact = {row["system"]: row for row in tables["compact_summary"]}
    assert compact["metc"]["label_good_pct"] == 75.0
    assert compact["ger"]["adequacy_pct"] == 100.0
    assert any(
        row["comparison"] == "human_vs_primary" and row["system"] == "ALL"
        for row in tables["agreement"]
    )
    assert any(
        row["analysis"] == "benchmark_item_clustered"
        and row["comparison"] == "ger_minus_metc"
        for row in tables["system_comparisons"]
    )
    assert all(
        row["human_source"] == "frozen_authoritative"
        for table in tables.values()
        for row in table
    )


def test_loader_consumes_sampling_frame_and_manifest_annotator_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    edit_id = "metc-edit-v1:en:1:metc:0"
    frame = {
        "dataset_id": "en_conll14_ann0",
        "sample_id": "sample-1",
        "selection_rank": 0,
        "selected_for_human_study": True,
        "item_id": "1",
        "row_idx": 1,
        "system": "metc",
        "edit_count": 1,
        "edits": [
            {
                "slot": 1,
                "edit_id": edit_id,
                "edit_idx": 0,
                "label": "Grammar/Test",
            }
        ],
    }
    assignments = [
        {
            "dataset_id": "en_conll14_ann0",
            "sample_id": "sample-1",
            "assignment_order": pass_number - 1,
            "pass_id": f"pass-v1:{pass_number}",
            "item_id": "1",
            "row_idx": 1,
            "system": "metc",
            "edit_count": 1,
            "status": "scorable",
            "session_unannotatable": False,
            "edits": [
                {
                    "slot": 1,
                    "edit_id": edit_id,
                    "edit_idx": 0,
                    "label": "Grammar/Test",
                    "span_quality": "Good",
                    "label_quality": "Good",
                    "unannotatable": False,
                }
            ],
        }
        for pass_number in (1, 2)
    ]
    _write_jsonl(root / "human/en.jsonl", assignments)
    _write_jsonl(root / "human/frames/en.jsonl", [frame])
    (root / "manifest.full.json").write_text(
        json.dumps(
            {
                "human_annotations": {
                    "en_conll14_ann0": {
                        "path": "human/en.jsonl",
                        "assignments": 2,
                        "samples": 1,
                        "passes_per_sample": 2,
                        "annotator_pool_size": 12,
                        "individual_annotator_ids_available": False,
                        "sampling_frame": {
                            "path": "human/frames/en.jsonl",
                            "instrument_items": 1,
                            "selection_rule": "all fixture items",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    judge = {
        "dataset": "en_conll14_ann0",
        "item_id": "1",
        "item_order": 0,
        "edit_id": edit_id,
        "edit_idx": 0,
        "system": "metc",
        "label": "Grammar/Test",
        "primary_span": "Good",
        "primary_label": "Good",
        "second_span": "Good",
        "second_label": "Good",
    }
    monkeypatch.setattr(
        "metc_artifact.metrics.human.load_frozen_judge_rows",
        lambda _: JudgeVerdictCorpus(
            judge_source="frozen_authoritative",
            rows=(judge,),
        ),
    )

    corpus = load_human_study(root, "en_conll14_ann0")

    assert corpus.annotator_pool_size == 12
    assert corpus.passes_per_sample == 2
    assert len(corpus.assignments) == 2
    assert len(corpus.sampling_frame) == 1


def test_loader_rejects_duplicate_pass_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    edit_id = "metc-edit-v1:en:1:metc:0"
    frame = {
        "dataset_id": "en_conll14_ann0",
        "sample_id": "sample-1",
        "selection_rank": 0,
        "selected_for_human_study": True,
        "item_id": "1",
        "row_idx": 1,
        "system": "metc",
        "edit_count": 1,
        "edits": [
            {
                "slot": 1,
                "edit_id": edit_id,
                "edit_idx": 0,
                "label": "Grammar/Test",
            }
        ],
    }
    assignment = {
        "dataset_id": "en_conll14_ann0",
        "sample_id": "sample-1",
        "assignment_order": 0,
        "pass_id": "pass-v1:1",
        "item_id": "1",
        "row_idx": 1,
        "system": "metc",
        "edit_count": 1,
        "status": "scorable",
        "session_unannotatable": False,
        "edits": [
            {
                "slot": 1,
                "edit_id": edit_id,
                "edit_idx": 0,
                "label": "Grammar/Test",
                "span_quality": "Good",
                "label_quality": "Good",
                "unannotatable": False,
            }
        ],
    }
    second = {**assignment, "assignment_order": 1}
    _write_jsonl(root / "human/en.jsonl", [assignment, second])
    _write_jsonl(root / "human/frames/en.jsonl", [frame])
    (root / "manifest.full.json").write_text(
        json.dumps(
            {
                "human_annotations": {
                    "en_conll14_ann0": {
                        "path": "human/en.jsonl",
                        "assignments": 2,
                        "samples": 1,
                        "passes_per_sample": 2,
                        "annotator_pool_size": 12,
                        "individual_annotator_ids_available": False,
                        "sampling_frame": {
                            "path": "human/frames/en.jsonl",
                            "instrument_items": 1,
                            "selection_rule": "fixture",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "metc_artifact.metrics.human.load_frozen_judge_rows",
        lambda _: JudgeVerdictCorpus(
            judge_source="frozen_authoritative",
            rows=(),
        ),
    )

    with pytest.raises(HumanMetricsInputError, match="pass IDs are missing or duplicated"):
        load_human_study(root, "en_conll14_ann0")


@pytest.mark.skipif(
    not os.environ.get("METC_DATA_PATH"),
    reason="set METC_DATA_PATH to the extracted complete Data artifact",
)
def test_released_human_values_match_current_paper() -> None:
    root = Path(os.environ["METC_DATA_PATH"])
    expected = {
        "en_conll14_ann0": {
            "assignments": 1600,
            "pool": 12,
            "iaa_label": 65.14962593516209,
            "primary_label": 62.25392296718974,
            "second_label": 65.04992867332382,
            "judge_label": 74.53971593897948,
            "metc_label": 67.41973840665874,
            "metc_adequacy": 65.66091954022988,
        },
        "de_merlin": {
            "assignments": 400,
            "pool": None,
            "iaa_label": 64.17004048582996,
            "primary_label": 68.02721088435374,
            "second_label": 70.94266277939747,
            "judge_label": 78.13084112149534,
            "metc_label": 78.72340425531915,
            "metc_adequacy": 69.27083333333334,
        },
    }
    for dataset, values in expected.items():
        metrics = compute_human_metrics(
            load_human_study(root, dataset),
            samples=100,
            seed=42,
        )
        assert metrics["study"]["assignments"] == values["assignments"]
        assert metrics["study"]["annotator_pool_size"] == values["pool"]
        assert (
            metrics["inter_annotator_agreement"]["label_agreement_pct"]
            == values["iaa_label"]
        )
        assert (
            metrics["human_vs_judges"]["primary"]["label_agreement_pct"]
            == values["primary_label"]
        )
        assert (
            metrics["human_vs_judges"]["second"]["label_agreement_pct"]
            == values["second_label"]
        )
        assert (
            metrics["judge_vs_judge_on_human_edits"]["label_agreement_pct"]
            == values["judge_label"]
        )
        assert (
            metrics["human_edit_quality"]["metc"]["label"]["Good"]["pct"]
            == values["metc_label"]
        )
        assert (
            metrics["sentence_adequacy"]["metc"]["adequacy_pct"]
            == values["metc_adequacy"]
        )
