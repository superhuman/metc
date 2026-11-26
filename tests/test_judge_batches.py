from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from metc_artifact.judges.batch import (
    GLOSS_ENTRIES_SHA256,
    GLOSS_ENTRY_COUNT,
    BatchError,
    build_acceptability_jobs,
    build_teacher_jobs,
    compare_acceptability_with_frozen,
    compare_teacher_with_frozen,
    load_label_glosses,
    main,
    materialize_teacher_annotations,
    run_acceptability_batch,
    run_teacher_batch,
)
from metc_artifact.judges.clients import HTTPResponse
from metc_artifact.judges.records import FreshRunWriter, read_fresh_records
from metc_artifact.metrics import load_judge_rows
from metc_artifact.schema import dataset_spec


def _openai_response(content: Mapping[str, Any], *, status: int = 200) -> HTTPResponse:
    return HTTPResponse(
        status=status,
        body=json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": json.dumps(content)}}
                ]
            }
        ),
        headers={"x-request-id": "mock-openai"},
    )


def _google_response(content: Mapping[str, Any]) -> HTTPResponse:
    return HTTPResponse(
        status=200,
        body=json.dumps(
            {
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(content)}]}}
                ]
            }
        ),
        headers={"x-request-id": "mock-google"},
    )


def _acceptability_row() -> dict[str, Any]:
    spec = dataset_spec("en")
    row: dict[str, Any] = {
        "sent_idx": 7,
        "src": "cats",
        "tgt": "cat",
        **{system: [] for system in spec.systems},
    }
    row["metc"] = [[0, 4, "Nn", "Grammar/Agreement", "cat"]]
    return row


def _teacher_row() -> dict[str, Any]:
    spec = dataset_spec("en")
    row: dict[str, Any] = {
        "sent_idx": 7,
        "src": "ab c",
        "tgt": "a b d",
        **{system: [] for system in spec.systems},
    }
    # Deliberately store edits out of positional order. The teacher prompt is
    # positional, while its stable IDs must still point to physical edit slots.
    row["metc"] = [
        [3, 4, "Nn", "Grammar/Agreement", "d"],
        [1, 1, "Mec", "Formatting/MissingSpace", " "],
    ]
    return row


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_gloss_loader_pins_exact_released_inventory() -> None:
    glosses = load_label_glosses()

    assert len(glosses) == GLOSS_ENTRY_COUNT == 3_616
    assert GLOSS_ENTRIES_SHA256 == (
        "24ec79277ec9a98b9c1c097be432a31e2a9e16e9499d2a1bd4ea05c8b44923ca"
    )
    assert glosses[("en", "metc", "Grammar/Agreement")] == (
        "Problem with grammatical agreement (number, person, or gender between a noun "
        "and its determiner, adjective, pronoun, or quantifier; not subject-verb agreement)."
    )


def test_both_judge_jobs_share_exact_stable_input_and_gloss(tmp_path: Path) -> None:
    spec = dataset_spec("en")
    [job] = build_acceptability_jobs([_acceptability_row()], spec)
    assert job["edit_id"] == "metc-edit-v1:en:7:metc:0"
    assert job["label_gloss"].startswith("Problem with grammatical agreement")

    primary_calls = 0

    def primary_transport(url, headers, payload, timeout):
        nonlocal primary_calls
        primary_calls += 1
        assert headers["Authorization"] == "Bearer test-openai-secret"
        if primary_calls == 1:
            return HTTPResponse(status=503, body='{"error":"retry"}', headers={})
        return _openai_response(
            {
                "span_quality": "Good",
                "label_quality": "Acceptable",
                "span_reason": "The insertion span is exact.",
                "label_reason": "The description fits.",
            }
        )

    writer = FreshRunWriter(tmp_path, "rerun-1")
    primary = run_acceptability_batch(
        [job],
        provider="primary",
        writer=writer,
        stream="acceptability-primary-en_conll14_ann0",
        api_key="test-openai-secret",
        transport=primary_transport,
        workers=1,
        max_attempts=2,
        retry_delay_seconds=0,
    )
    assert primary.request_attempts == 2
    assert primary.newly_complete_jobs == 1
    records = read_fresh_records(primary.record_path)
    assert [record["error"] is None for record in records] == [False, True]
    assert records[0]["raw_response"] == '{"error":"retry"}'
    assert records[1]["request"]["configuration"]["historical_backend_id"] == (
        "openai_direct_gpt54_flex"
    )
    assert "test-openai-secret" not in primary.record_path.read_text(encoding="utf-8")

    # A resumed invocation makes no provider call after a complete stable job.
    resumed = run_acceptability_batch(
        [job],
        provider="primary",
        writer=writer,
        stream="acceptability-primary-en_conll14_ann0",
        api_key="test-openai-secret",
        transport=lambda *_args: pytest.fail("completed job was called again"),
        workers=1,
        retry_delay_seconds=0,
    )
    assert resumed.previously_complete_jobs == 1
    assert resumed.attempted_jobs == 0

    def second_transport(url, headers, payload, timeout):
        assert headers["x-goog-api-key"] == "test-google-secret"
        return _google_response(
            {
                "span_quality": "Acceptable",
                "label_quality": "Good",
                "span_reason": "The span is defensible.",
                "label_reason": "The description is precise.",
            }
        )

    second = run_acceptability_batch(
        [job],
        provider="second",
        writer=writer,
        stream="acceptability-second-en_conll14_ann0",
        api_key="test-google-secret",
        transport=second_transport,
        workers=1,
        retry_delay_seconds=0,
    )
    [second_record] = read_fresh_records(second.record_path)
    assert second_record["input"] == job
    assert second_record["stable_edit_ids"] == [job["edit_id"]]
    assert second_record["request"]["configuration"][
        "historical_thinking_configuration"
    ]["status"] == "not_preserved"

    frozen = tmp_path / "frozen-primary.jsonl"
    _write_jsonl(
        frozen,
        [
            {
                "edit_id": job["edit_id"],
                "span_quality": "Good",
                "label_quality": "Good",
            }
        ],
    )
    comparison = compare_acceptability_with_frozen(primary.record_path, frozen)
    assert comparison.comparable_edits == 1
    assert comparison.field_matches == {"span_quality": 1, "label_quality": 0}
    assert comparison.errored_records == 1

    data_root = tmp_path / "data"
    _write_jsonl(
        data_root / "benchmarks/en_conll14_ann0/annotations.jsonl",
        [_acceptability_row()],
    )
    _write_jsonl(
        data_root / "judges/canonical/primary/en_conll14_ann0.jsonl",
        [
            {
                **job,
                "span_quality": "Good",
                "label_quality": "Good",
            }
        ],
    )
    corpus = load_judge_rows(
        data_root,
        judge_source="fresh_rerun",
        fresh_run_root=writer.run_root,
    )
    assert corpus.judge_source == "fresh_rerun"
    assert len(corpus.rows) == 1
    assert corpus.rows[0]["primary_span"] == "Good"
    assert corpus.rows[0]["second_span"] == "Acceptable"


def test_teacher_batch_preserves_fixed_edits_and_materializes_by_stable_id(
    tmp_path: Path,
) -> None:
    spec = dataset_spec("en")
    row = _teacher_row()
    build = build_teacher_jobs([row], spec)
    assert build.edit_count == 2
    assert build.skipped_overlapping_rows == ()
    [job] = build.jobs
    assert [edit["edit_id"] for edit in job["edits"]] == [
        "metc-edit-v1:en:7:metc:1",
        "metc-edit-v1:en:7:metc:0",
    ]
    assert job["numbered"] == "a{#1 => }b {#2 c=>d}"
    assert job["edits"][0]["span"] == {
        "unit": "unicode_codepoint",
        "start": 1,
        "end": 1,
    }

    def teacher_transport(url, headers, payload, timeout):
        return _openai_response(
            {
                "labels": ["Formatting", "Grammar"],
                "sublabels": ["MissingSpace", "Agreement"],
            }
        )

    writer = FreshRunWriter(tmp_path, "teacher-1")
    summary = run_teacher_batch(
        [job],
        writer=writer,
        stream="teacher-en",
        api_key="test-openai-secret",
        transport=teacher_transport,
        workers=1,
        retry_delay_seconds=0,
    )
    records = read_fresh_records(summary.record_path)
    [materialized] = materialize_teacher_annotations([row], spec, records)
    assert materialized["metc"] == row["metc"]
    assert materialized["metc_teacher"] == [
        [3, 4, "Nn", "Grammar/Agreement", "d"],
        [1, 1, "Mec", "Formatting/MissingSpace", " "],
    ]
    # Raw response, prompt/config, and immutable student spans are all durable.
    assert records[0]["raw_response"] is not None
    assert records[0]["prompt"]["sha256"]
    assert records[0]["request"]["configuration"]["fixed_student_spans"] is True
    assert records[0]["input"]["edits"] == job["edits"]

    frozen_row = dict(row)
    frozen_row["metc_teacher"] = materialized["metc_teacher"]
    frozen_path = tmp_path / "frozen-benchmark.jsonl"
    _write_jsonl(frozen_path, [frozen_row])
    comparison = compare_teacher_with_frozen(
        summary.record_path, frozen_path, spec
    )
    assert comparison.comparable_edits == 2
    assert comparison.field_matches == {"raw_label": 2}


def test_teacher_materialization_rejects_records_for_changed_fresh_spans(
    tmp_path: Path,
) -> None:
    spec = dataset_spec("en")
    row = _teacher_row()
    [job] = build_teacher_jobs([row], spec).jobs

    writer = FreshRunWriter(tmp_path, "teacher-1")
    summary = run_teacher_batch(
        [job],
        writer=writer,
        stream="teacher-en",
        api_key="secret",
        transport=lambda *_args: _openai_response(
            {
                "labels": ["Formatting", "Grammar"],
                "sublabels": ["MissingSpace", "Agreement"],
            }
        ),
        workers=1,
        retry_delay_seconds=0,
    )
    changed = _teacher_row()
    changed["metc"][0][0:2] = [2, 4]
    records = read_fresh_records(summary.record_path)
    with pytest.raises(BatchError, match="does not match current METC edits"):
        materialize_teacher_annotations([changed], spec, records)


def test_batch_requires_explicit_or_environment_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    [job] = build_acceptability_jobs([_acceptability_row()], dataset_spec("en"))
    writer = FreshRunWriter(tmp_path, "no-key")
    with pytest.raises(BatchError, match="OPENAI_API_KEY"):
        run_acceptability_batch(
            [job],
            provider="primary",
            writer=writer,
            stream="primary-en",
            api_key=None,
            workers=1,
            retry_delay_seconds=0,
        )


def test_resume_rejects_changed_provider_request_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    [job] = build_acceptability_jobs([_acceptability_row()], dataset_spec("en"))
    writer = FreshRunWriter(tmp_path, "request-contract")
    run_acceptability_batch(
        [job],
        provider="primary",
        writer=writer,
        stream="primary-en",
        api_key="secret",
        transport=lambda *_args: _openai_response(
            {
                "span_quality": "Good",
                "label_quality": "Good",
                "span_reason": "Exact.",
                "label_reason": "Exact.",
            }
        ),
        workers=1,
        retry_delay_seconds=0,
    )

    monkeypatch.setattr("metc_artifact.judges.clients.OPENAI_MODEL", "changed-model")
    with pytest.raises(BatchError, match="different or unpinned"):
        run_acceptability_batch(
            [job],
            provider="primary",
            writer=writer,
            stream="primary-en",
            api_key="secret",
            transport=lambda *_args: pytest.fail(
                "changed request contract must fail before a provider request"
            ),
            workers=1,
            retry_delay_seconds=0,
        )


def test_teacher_retries_use_one_taxonomy_and_resume_detects_changes(tmp_path: Path) -> None:
    [job] = build_teacher_jobs([_teacher_row()], dataset_spec("en")).jobs
    released = Path(__file__).resolve().parents[1] / "prompts" / "taxonomy_v2.md"
    taxonomy = tmp_path / "taxonomy.md"
    original_text = released.read_text(encoding="utf-8")
    taxonomy.write_text(original_text, encoding="utf-8")
    sent_prompts = []

    def transport(_url, _headers, payload, _timeout):
        sent_prompts.append(payload["messages"][1]["content"])
        if len(sent_prompts) == 1:
            taxonomy.write_text(original_text + "\nRevised example.\n", encoding="utf-8")
            return HTTPResponse(status=503, body="", headers={})
        return _openai_response(
            {"labels": ["Formatting", "Grammar"], "sublabels": ["MissingSpace", "Agreement"]}
        )

    options = {
        "writer": FreshRunWriter(tmp_path, "taxonomy-resume"),
        "stream": "teacher-en",
        "api_key": "test-openai-secret",
        "taxonomy_path": taxonomy,
        "workers": 1,
        "max_attempts": 2,
        "retry_delay_seconds": 0,
    }
    summary = run_teacher_batch([job], transport=transport, **options)
    assert summary.newly_complete_jobs == 1
    assert len(sent_prompts) == 2 and sent_prompts[0] == sent_prompts[1]
    assert original_text in sent_prompts[0]
    assert "Revised example." not in sent_prompts[1]
    records = read_fresh_records(summary.record_path)
    assert len({record["batch"]["request_sha256"] for record in records}) == 1

    def unexpected_provider_request(*_args):
        pytest.fail("a changed taxonomy must fail before a provider request")

    with pytest.raises(BatchError, match="different or unpinned"):
        run_teacher_batch([job], transport=unexpected_provider_request, **options)


def test_cli_builds_reviewable_job_files_without_calling_providers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    benchmark = tmp_path / "benchmark.jsonl"
    output = tmp_path / "acceptability-jobs.jsonl"
    _write_jsonl(benchmark, [_acceptability_row()])

    assert main(
        [
            "build-acceptability",
            "--dataset",
            "en",
            "--benchmark",
            str(benchmark),
            "--output",
            str(output),
        ]
    ) == 0
    [job] = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert job["edit_id"] == "metc-edit-v1:en:7:metc:0"
    assert job["label_gloss"].startswith("Problem with grammatical agreement")
    assert json.loads(capsys.readouterr().out) == {
        "dataset_id": "en_conll14_ann0",
        "jobs": 1,
    }
