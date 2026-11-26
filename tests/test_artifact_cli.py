from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from metc_artifact.cli import main
from metc_artifact.metrics import JudgeVerdictCorpus
from metc_artifact.schema import DATASETS, dataset_spec


def _row(mapped_label: str = "mapped") -> dict:
    return {
        "idx": 0,
        "src": "a",
        "tgt": "b",
        "gold": [[0, 1, "mapped", "gold", "b"]],
        "metc": [[0, 1, mapped_label, "metc", "b"]],
        "metc_teacher": [],
        "errant": [],
        "jp": [],
    }


def _jsonl(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_compare_annotations_cli_uses_semantic_equality(
    tmp_path: Path, capsys
) -> None:
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    _jsonl(expected, _row())
    _jsonl(actual, _row())

    exit_code = main(
        [
            "compare-annotations",
            "--dataset",
            "en",
            "--expected",
            str(expected),
            "--actual",
            str(actual),
            "--systems",
            "metc",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["equal"] is True


def test_compare_annotations_cli_fails_on_changed_mapping(
    tmp_path: Path, capsys
) -> None:
    expected = tmp_path / "expected.jsonl"
    actual = tmp_path / "actual.jsonl"
    _jsonl(expected, _row())
    _jsonl(actual, _row("different"))

    exit_code = main(
        [
            "compare-annotations",
            "--dataset",
            "en",
            "--expected",
            str(expected),
            "--actual",
            str(actual),
            "--systems",
            "metc",
        ]
    )

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["equal"] is False


def test_reference_matching_cli_requires_explicit_provenance(
    tmp_path: Path, capsys
) -> None:
    spec = dataset_spec("en")
    benchmark = tmp_path / "benchmarks" / spec.dataset_id
    benchmark.mkdir(parents=True)
    row = {
        "idx": 0,
        "src": "a",
        "tgt": "b",
        **{system: [] for system in spec.systems},
    }
    row["gold"] = [[0, 1, "mapped", "gold", "b"]]
    row["metc"] = [[0, 1, "mapped", "metc", "b"]]
    _jsonl(benchmark / "annotations.jsonl", row)
    output = tmp_path / "tables"

    exit_code = main(
        [
            "reference-matching",
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--datasets",
            "en",
            "--annotation-source",
            "fresh_rerun",
            "--samples",
            "10",
        ]
    )

    assert exit_code == 0
    with (output / f"{spec.dataset_id}.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {row["annotation_source"] for row in rows} == {"fresh_rerun"}
    assert spec.dataset_id in capsys.readouterr().out


def test_judge_tables_cli_uses_frozen_authoritative_verdicts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    calls: dict = {}

    corpus = _judge_corpus({spec.dataset_id for spec in DATASETS})

    def load(data_root, *, judge_source):
        calls["load"] = (data_root, judge_source)
        return corpus

    def compute(corpus, *, samples, seed):
        calls["compute"] = (corpus, samples, seed)
        return {"judge_source": "frozen_authoritative", "tables": {}}

    monkeypatch.setattr("metc_artifact.cli.load_judge_rows", load)
    monkeypatch.setattr("metc_artifact.cli.compute_paper_tables", compute)

    exit_code = main(
        [
            "judge-tables",
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert calls["load"] == (tmp_path / "data", "frozen_authoritative")
    assert calls["compute"] == (corpus, 1000, 42)
    written = json.loads(
        (tmp_path / "out" / "frozen-authoritative-tables.json").read_text(encoding="utf-8")
    )
    assert written["judge_source"] == "frozen_authoritative"
    assert "judge_tables" in capsys.readouterr().out


def test_human_tables_cli_writes_one_table_per_released_study(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "manifest.full.json").write_text(
        json.dumps({"human_annotations": {"en_conll14_ann0": {}, "de_merlin": {}}}),
        encoding="utf-8",
    )
    seen: list[str] = []

    def load(root, dataset_id):
        assert root == data_root
        seen.append(dataset_id)
        return dataset_id

    monkeypatch.setattr("metc_artifact.cli.load_human_study", load)
    monkeypatch.setattr(
        "metc_artifact.cli.compute_human_tables",
        lambda corpus, *, samples, seed: {"dataset": corpus, "samples": samples},
    )

    exit_code = main(
        ["human-tables", "--data-root", str(data_root), "--output-dir", str(tmp_path / "out")]
    )

    assert exit_code == 0
    assert seen == ["de_merlin", "en_conll14_ann0"]
    for dataset_id in seen:
        written = json.loads(
            (tmp_path / "out" / f"{dataset_id}.json").read_text(encoding="utf-8")
        )
        assert written == {"dataset": dataset_id, "samples": 10_000}


def _judge_corpus(datasets: set[str]) -> JudgeVerdictCorpus:
    return JudgeVerdictCorpus(
        judge_source="frozen_authoritative",
        rows=tuple({"dataset": dataset} for dataset in sorted(datasets)),
    )


@pytest.mark.parametrize(
    ("datasets", "missing", "unexpected"),
    [
        ({"en_conll14_ann0"}, True, False),
        ({spec.dataset_id for spec in DATASETS} | {"xx_unknown"}, False, True),
    ],
)
def test_judge_tables_cli_rejects_incomplete_or_unexpected_datasets(
    tmp_path: Path, monkeypatch, capsys, datasets, missing, unexpected
) -> None:
    monkeypatch.setattr(
        "metc_artifact.cli.load_judge_rows",
        lambda data_root, *, judge_source: _judge_corpus(datasets),
    )
    monkeypatch.setattr(
        "metc_artifact.cli.compute_paper_tables",
        lambda *args, **kwargs: pytest.fail("tables must not be computed"),
    )

    exit_code = main(
        ["judge-tables", "--data-root", str(tmp_path), "--output-dir", str(tmp_path / "out")]
    )

    assert exit_code == 1
    assert not (tmp_path / "out").exists()
    report = json.loads(capsys.readouterr().err)
    assert bool(report["missing"]) is missing
    assert bool(report["unexpected"]) is unexpected


@pytest.mark.parametrize(
    "studies",
    [
        {},
        {"en_conll14_ann0": {}},
        {"en_conll14_ann0": {}, "de_merlin": {}, "../escape": {}},
    ],
)
def test_human_tables_cli_rejects_incomplete_or_unexpected_studies(
    tmp_path: Path, monkeypatch, capsys, studies
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "manifest.full.json").write_text(
        json.dumps({"human_annotations": studies}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "metc_artifact.cli.load_human_study",
        lambda *args: pytest.fail("studies must not be loaded"),
    )

    exit_code = main(
        ["human-tables", "--data-root", str(data_root), "--output-dir", str(tmp_path / "out")]
    )

    assert exit_code == 1
    assert not (tmp_path / "out").exists()
    assert "human studies" in capsys.readouterr().err
