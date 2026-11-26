from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from metc_artifact.e2e import (
    CommandResult,
    E2EConfig,
    E2ERunner,
    RuntimePaths,
    StageStatus,
    _canonicalize_system_order,
    _restore_authoritative_row_order,
    _validate_acceptability_job_identity,
    build_parser,
    config_from_args,
    default_adapter_plan,
    main,
    normalize_systems,
)
from metc_artifact.judges.batch import (
    build_acceptability_jobs,
    run_acceptability_batch,
    run_teacher_batch,
)
from metc_artifact.judges.clients import HTTPResponse
from metc_artifact.judges.records import FreshRunWriter
from metc_artifact.schema import dataset_spec


DATASET_ID = "en_conll14_ann0"


def _expected_row(*, metc_mapping: str = "Nn") -> dict:
    return {
        "sent_idx": 0,
        "src": "a",
        "tgt": "b",
        "gold": [[0, 1, "Nn", "Nn", "b"]],
        "metc": [[0, 1, metc_mapping, "Grammar/Agreement", "b"]],
        "metc_teacher": [[0, 1, "Nn", "Grammar/Agreement", "b"]],
        "errant": [[0, 1, "Nn", "R:NOUN:NUM", "b"]],
        "jp": [[0, 1, "Nn", "R:NOUN", "b"]],
    }


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _canonical_judges(data_root: Path) -> None:
    jobs = build_acceptability_jobs([_expected_row()], dataset_spec("en"))
    for role, judge in (
        ("primary", "openai_direct_gpt54_flex"),
        ("second", "vai_gemini35_flash"),
    ):
        _jsonl(
            data_root
            / "judges"
            / "canonical"
            / role
            / f"{DATASET_ID}.jsonl",
            [
                {
                    **job,
                    "judge": judge,
                    "errored": False,
                    "span_quality": "Good",
                    "label_quality": "Good",
                }
                for job in jobs
            ],
        )


def _executable(path: Path) -> str:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _runtime_paths(tmp_path: Path, *, metc: bool = True) -> RuntimePaths:
    stanza = tmp_path / "stanza"
    stanza.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    transform = tmp_path / "jp-transform.json"
    transform.write_text("{}\n", encoding="utf-8")
    return RuntimePaths(
        core_python=sys.executable,
        cherrant_python=_executable(tmp_path / "cherrant-python"),
        metc_python=_executable(tmp_path / "metc-python") if metc else None,
        metc_checkpoint=checkpoint if metc else None,
        errant_python=_executable(tmp_path / "errant-python"),
        jp_python=_executable(tmp_path / "jp-python"),
        jp_transform_manifest=transform,
        jp_stanza_resources=stanza,
    )


class FakeExecutor:
    def __init__(self, *, metc_mapping: str = "Nn") -> None:
        self.commands: list[tuple[str, ...]] = []
        self.metc_mapping = metc_mapping

    def __call__(self, argv, *, cwd, environment, timeout) -> CommandResult:
        command = tuple(argv)
        self.commands.append(command)
        module = command[command.index("-m") + 1]
        if module == "metc_artifact.datasets":
            output_root = Path(command[command.index("--output-dir") + 1])
            reconstructed = {
                "sent_idx": 0,
                "src": "a",
                "tgt": "b",
                "gold": [[0, 1, "Nn", "Nn", "b"]],
            }
            _jsonl(output_root / DATASET_ID / "annotations.jsonl", [reconstructed])
            return CommandResult(0)

        system_by_module = {
            "metc_artifact.adapters.metc": (
                "metc",
                "Grammar/Agreement",
                self.metc_mapping,
            ),
            "metc_artifact.adapters.errant": ("errant", "R:NOUN:NUM", "Nn"),
            "metc_artifact.adapters.jp_errant": ("jp", "R:NOUN", "Nn"),
        }
        system, native_label, mapped_label = system_by_module[module]
        output = Path(command[command.index("--output") + 1])
        _jsonl(
            output,
            [
                {
                    "dataset_id": DATASET_ID,
                    "item_id": "0",
                    "system": system,
                    "span_unit": "unicode_codepoint",
                    "edits": [
                        {
                            "start": 0,
                            "end": 1,
                            "replacement": "b",
                            "native_label": native_label,
                            "mapped_label": mapped_label,
                        }
                    ],
                }
            ],
        )
        return CommandResult(0)


def _config(
    tmp_path: Path,
    *,
    runtimes: RuntimePaths,
    systems: tuple[str, ...],
    known_differences=None,
    metc_inference_profile: str = "paper",
    run_llm: bool = False,
) -> E2EConfig:
    data_root = tmp_path / "data"
    _jsonl(
        data_root / "benchmarks" / DATASET_ID / "annotations.jsonl",
        [_expected_row()],
    )
    _canonical_judges(data_root)
    return E2EConfig(
        data_root=data_root,
        work_root=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        software_root=tmp_path,
        runtimes=runtimes,
        systems=systems,
        known_differences=known_differences or {},
        metc_inference_profile=metc_inference_profile,
        reference_samples=20,
        run_llm=run_llm,
        require_full_scope=False,
    )


def _valid_data(_path: Path) -> dict:
    return {"valid": True, "benchmark_items": 1}


def _openai_response(content: Mapping[str, Any]) -> HTTPResponse:
    return HTTPResponse(
        status=200,
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


def test_e2e_executes_materializes_compares_and_scores_fresh_annotations(
    tmp_path: Path,
) -> None:
    executor = FakeExecutor()
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("metc", "errant", "jp"),
    )
    report = E2ERunner(
        config,
        executor=executor,
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    assert report.status == StageStatus.PASS
    assert {command[command.index("-m") + 1] for command in executor.commands} == {
        "metc_artifact.datasets",
        "metc_artifact.adapters.metc",
        "metc_artifact.adapters.errant",
        "metc_artifact.adapters.jp_errant",
    }
    fresh_path = Path(report.artifacts[f"fresh_benchmark.{DATASET_ID}"])
    [fresh] = [json.loads(line) for line in fresh_path.read_text().splitlines()]
    assert {"gold", "metc", "errant", "jp"}.issubset(fresh)
    reference_path = Path(report.artifacts[f"reference_matching.{DATASET_ID}"])
    reference_text = reference_path.read_text(encoding="utf-8")
    assert "annotation_source,span_match,system" in reference_text
    assert "fresh_rerun,overlap" in reference_text
    assert "metc" in reference_text
    assert "errant" in reference_text
    assert "jp" in reference_text
    exact_reference_path = Path(
        report.artifacts[f"reference_matching.{DATASET_ID}.exact"]
    )
    exact_reference_text = exact_reference_path.read_text(encoding="utf-8")
    assert "annotation_source,span_match,system" in exact_reference_text
    assert "fresh_rerun,exact" in exact_reference_text
    assert "metc" in exact_reference_text
    assert "errant" in exact_reference_text
    assert "jp" in exact_reference_text
    exact_stage = next(
        stage
        for stage in report.stages
        if stage.stage == f"reference_matching.{DATASET_ID}.exact"
    )
    assert exact_stage.details["span_match"] == "exact"
    assert report.to_dict()["stage_counts"]["FAIL"] == 0


def test_external_producer_environment_strips_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "cloud-secret")
    monkeypatch.setenv("METC_SAFE_RUNTIME_FLAG", "kept")
    runner = E2ERunner(
        _config(
            tmp_path,
            runtimes=_runtime_paths(tmp_path),
            systems=(),
        ),
        data_validator=_valid_data,
        dataset_specs=(),
    )

    environment = runner._environment()

    assert "OPENAI_API_KEY" not in environment
    assert "GOOGLE_API_KEY" not in environment
    assert "AWS_SESSION_TOKEN" not in environment
    assert environment["METC_SAFE_RUNTIME_FLAG"] == "kept"


def test_partial_scope_cannot_report_complete_e2e_pass(tmp_path: Path) -> None:
    config = replace(
        _config(
            tmp_path,
            runtimes=_runtime_paths(tmp_path),
            systems=("metc",),
            run_llm=False,
        ),
        require_full_scope=True,
    )
    report = E2ERunner(
        config,
        executor=FakeExecutor(),
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    assert report.status == StageStatus.KNOWN_DIFFERENCE
    scope = next(stage for stage in report.stages if stage.stage == "scope")
    assert scope.status == StageStatus.KNOWN_DIFFERENCE
    assert scope.details["llm_enabled"] is False
    assert "errant" in scope.details["missing_systems"]
    assert "de_merlin" in scope.details["missing_datasets"]


def test_no_network_is_forwarded_to_public_dataset_reconstruction(
    tmp_path: Path,
) -> None:
    executor = FakeExecutor()
    config = replace(
        _config(
            tmp_path,
            runtimes=_runtime_paths(tmp_path),
            systems=(),
        ),
        no_network=True,
    )
    E2ERunner(
        config,
        executor=executor,
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    reconstruct = next(
        command
        for command in executor.commands
        if "metc_artifact.datasets" in command
    )
    assert "--no-network" in reconstruct


def test_cli_defaults_to_no_live_llm_requests(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--work-root",
            str(tmp_path / "work"),
        ]
    )

    assert args.llm is False
    assert config_from_args(args).run_llm is False
    assert args.metc_inference_profile == "paper"
    assert config_from_args(args).metc_inference_profile == "paper"


def test_cli_describes_data_artifact_without_schema_version() -> None:
    help_text = build_parser().format_help()

    assert "extracted complete METC Data artifact" in help_text
    assert "metc-data-v2 artifact" not in help_text


def test_user_system_aliases_and_teacher_dependency_are_normalized() -> None:
    assert normalize_systems(("gerrant", "jp_errant")) == ("ger", "jp")
    assert normalize_systems(("metc_teacher",)) == ("metc", "metc_teacher")


def test_public_rows_are_restored_to_authoritative_bootstrap_order() -> None:
    authoritative = [
        {"sent_idx": 9, "src": "a", "tgt": "b", "gold": []},
        {"sent_idx": 2, "src": "c", "tgt": "d", "gold": []},
    ]
    reconstructed = [authoritative[1], authoritative[0]]

    restored = _restore_authoritative_row_order(authoritative, reconstructed)

    assert [row["sent_idx"] for row in restored] == [9, 2]


def test_authoritative_order_restoration_rejects_duplicate_item_ids() -> None:
    authoritative = [{"sent_idx": 1}, {"sent_idx": 2}]
    reconstructed = [{"sent_idx": 1}, {"sent_idx": 1}]

    with pytest.raises(ValueError, match="duplicate item IDs"):
        _restore_authoritative_row_order(authoritative, reconstructed)


def test_no_llm_flag_never_launches_requests_even_when_keys_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "present-but-disabled")
    monkeypatch.setenv("GOOGLE_API_KEY", "present-but-disabled")
    monkeypatch.setattr(
        "metc_artifact.e2e.run_teacher_batch",
        lambda *_args, **_kwargs: pytest.fail("teacher request launched without --llm"),
    )
    monkeypatch.setattr(
        "metc_artifact.e2e.run_acceptability_batch",
        lambda *_args, **_kwargs: pytest.fail("judge request launched without --llm"),
    )
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("metc_teacher",),
        run_llm=False,
    )
    report = E2ERunner(
        config,
        executor=FakeExecutor(),
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    assert config.systems == ("metc", "metc_teacher")
    disabled = next(
        stage
        for stage in report.stages
        if stage.stage == f"teacher.{DATASET_ID}.execute"
    )
    assert disabled.status == StageStatus.KNOWN_DIFFERENCE
    assert "disabled" in disabled.summary


def test_fresh_edit_order_is_aligned_before_stable_ids_are_assigned() -> None:
    spec = dataset_spec("en")
    expected = [{
        "sent_idx": 0,
        "src": "ab",
        "tgt": "cd",
        "metc": [
            [0, 1, "Nn", "Grammar/Agreement", "c"],
            [1, 2, "V0", "Grammar/Verb", "d"],
        ],
    }]
    actual = [{**expected[0], "metc": list(reversed(expected[0]["metc"]))}]

    [aligned] = _canonicalize_system_order(expected, actual, spec, "metc")

    assert aligned["metc"] == expected[0]["metc"]


def test_judge_job_ids_fail_closed_when_edit_identity_is_reordered(
    tmp_path: Path,
) -> None:
    frozen = [
        {
            "edit_id": "metc-edit-v1:en:0:metc:0",
            "row_idx": 0,
            "system": "metc",
            "edit_idx": 0,
            "src": "ab",
            "tgt": "cd",
            "edit_start": 0,
            "edit_end": 1,
            "edit_diff": "'a' => 'c'",
        }
    ]
    path = tmp_path / "frozen.jsonl"
    _jsonl(path, frozen)
    reordered = [{**frozen[0], "edit_start": 1, "edit_end": 2}]

    with pytest.raises(ValueError, match="fresh judge identity differs"):
        _validate_acceptability_job_identity(reordered, path)


def test_e2e_recomputes_human_tables_from_released_assignments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=(),
    )
    (config.data_root / "manifest.full.json").write_text(
        json.dumps({"human_annotations": {DATASET_ID: {}}}) + "\n",
        encoding="utf-8",
    )
    corpus = SimpleNamespace(assignments=(object(), object()), sampling_frame=(object(),))
    captured: dict[str, Any] = {}

    def fake_load(data_root: Path, dataset_id: str):
        captured["load"] = (data_root, dataset_id)
        return corpus

    def fake_compute(value, *, samples: int, seed: int):
        captured["compute"] = (value, samples, seed)
        return {
            "schema_version": "metc-raw-human-tables-v1",
            "human_source": "frozen_authoritative",
            "dataset": DATASET_ID,
            "metrics": {},
            "tables": {},
        }

    monkeypatch.setattr("metc_artifact.e2e.load_human_study", fake_load)
    monkeypatch.setattr("metc_artifact.e2e.compute_human_tables", fake_compute)
    monkeypatch.setattr(E2ERunner, "_frozen_judge_tables", lambda _self: None)
    report = E2ERunner(
        config,
        executor=FakeExecutor(),
        data_validator=_valid_data,
        dataset_specs=(),
    ).run()

    assert report.status == StageStatus.PASS
    assert captured["load"] == (config.data_root, DATASET_ID)
    assert captured["compute"] == (corpus, 10_000, 42)
    output = Path(report.artifacts[f"human_tables.{DATASET_ID}"])
    assert json.loads(output.read_text(encoding="utf-8"))["human_source"] == (
        "frozen_authoritative"
    )


def test_e2e_always_recomputes_frozen_authoritative_judge_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=(),
    )
    (config.data_root / "manifest.full.json").write_text(
        json.dumps({"human_annotations": {}}) + "\n",
        encoding="utf-8",
    )
    corpus = SimpleNamespace(
        judge_source="frozen_authoritative",
        rows=(object(), object(), object()),
    )
    captured: dict[str, Any] = {}

    def fake_load(data_root: Path, *, judge_source: str):
        captured["load"] = (data_root, judge_source)
        return corpus

    def fake_compute(value, *, samples: int, seed: int):
        captured["compute"] = (value, samples, seed)
        return {
            "schema_version": "metc-raw-judge-tables-v1",
            "judge_source": "frozen_authoritative",
            "metrics": {"judge_source": "frozen_authoritative"},
            "tables": {},
        }

    monkeypatch.setattr("metc_artifact.e2e.load_judge_rows", fake_load)
    monkeypatch.setattr("metc_artifact.e2e.compute_paper_tables", fake_compute)
    report = E2ERunner(
        config,
        executor=FakeExecutor(),
        data_validator=_valid_data,
        dataset_specs=(),
    ).run()

    assert report.status == StageStatus.PASS
    assert captured["load"] == (config.data_root, "frozen_authoritative")
    assert captured["compute"] == (corpus, 20, 42)
    output = Path(report.artifacts["judge_tables.frozen_authoritative"])
    assert json.loads(output.read_text(encoding="utf-8"))["judge_source"] == (
        "frozen_authoritative"
    )
    stage = next(
        stage
        for stage in report.stages
        if stage.stage == "judge_metrics.frozen_authoritative"
    )
    assert stage.status == StageStatus.PASS
    assert stage.details["judge_source"] == "frozen_authoritative"


def test_llm_stages_are_explicitly_incomplete_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("metc", "metc_teacher", "errant", "jp"),
        run_llm=True,
    )

    report = E2ERunner(
        config,
        executor=FakeExecutor(),
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    assert report.status == StageStatus.KNOWN_DIFFERENCE
    teacher = next(
        stage
        for stage in report.stages
        if stage.stage == f"teacher.{DATASET_ID}.execute"
    )
    assert teacher.status == StageStatus.KNOWN_DIFFERENCE
    assert "OPENAI_API_KEY" in teacher.summary
    acceptability = next(
        stage
        for stage in report.stages
        if stage.stage == f"acceptability.{DATASET_ID}.jobs"
    )
    assert acceptability.status == StageStatus.KNOWN_DIFFERENCE
    assert acceptability.details["missing_systems"] == ["metc_teacher"]
    metrics = next(
        stage for stage in report.stages if stage.stage == "judge_metrics.fresh"
    )
    assert metrics.status == StageStatus.KNOWN_DIFFERENCE


def test_acceptability_gates_both_missing_provider_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    spec = dataset_spec("en")
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("metc", "metc_teacher", "errant", "jp"),
        run_llm=True,
    )
    runner = E2ERunner(
        config,
        executor=FakeExecutor(),
        data_validator=_valid_data,
        dataset_specs=(spec,),
    )
    runner.llm_writer = FreshRunWriter(config.work_root / "llm", "missing-keys")

    runner._run_acceptability(
        spec,
        [_expected_row()],
        ("metc", "metc_teacher", "errant", "jp"),
    )

    stages = {stage.stage: stage for stage in runner.stages}
    assert stages[f"acceptability.{DATASET_ID}.jobs"].status == StageStatus.PASS
    for role, credential in (
        ("primary", "OPENAI_API_KEY"),
        ("second", "GOOGLE_API_KEY"),
    ):
        stage = stages[f"acceptability.{DATASET_ID}.{role}.execute"]
        assert stage.status == StageStatus.KNOWN_DIFFERENCE
        assert credential in stage.summary


def test_e2e_runs_teacher_both_judges_and_metrics_only_from_fresh_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("metc", "metc_teacher", "errant", "jp"),
        run_llm=True,
    )
    spec = dataset_spec("en")
    frozen_jobs = build_acceptability_jobs([_expected_row()], spec)
    for role in ("primary", "second"):
        _jsonl(
            config.data_root
            / "judges"
            / "canonical"
            / role
            / f"{DATASET_ID}.jsonl",
            [
                {
                    **job,
                    "edit_id": job["edit_id"],
                    "span_quality": "Good",
                    "label_quality": "Good",
                }
                for job in frozen_jobs
            ],
        )

    def fake_teacher_batch(jobs, **kwargs):
        kwargs.update(
            {
                "api_key": "openai-secret",
                "workers": 1,
                "max_attempts": 1,
                "retry_delay_seconds": 0,
                "transport": lambda *_args: _openai_response(
                    {"labels": ["Grammar"], "sublabels": ["Agreement"]}
                ),
            }
        )
        return run_teacher_batch(jobs, **kwargs)

    def fake_acceptability_batch(jobs, *, provider, **kwargs):
        def primary_transport(*_args):
            return _openai_response(
                {
                    "span_quality": "Good",
                    "label_quality": "Good",
                    "span_reason": "Exact span.",
                    "label_reason": "Exact label.",
                }
            )

        def second_transport(*_args):
            return _google_response(
                {
                    "span_quality": "Good",
                    "label_quality": "Good",
                    "span_reason": "Exact span.",
                    "label_reason": "Exact label.",
                }
            )

        transport = primary_transport if provider == "primary" else second_transport
        kwargs.update(
            {
                "api_key": f"{provider}-secret",
                "workers": 1,
                "max_attempts": 1,
                "retry_delay_seconds": 0,
                "transport": transport,
            }
        )
        return run_acceptability_batch(jobs, provider=provider, **kwargs)

    monkeypatch.setattr(
        "metc_artifact.e2e.run_teacher_batch",
        fake_teacher_batch,
    )
    monkeypatch.setattr(
        "metc_artifact.e2e.run_acceptability_batch",
        fake_acceptability_batch,
    )

    report = E2ERunner(
        config,
        executor=FakeExecutor(),
        data_validator=_valid_data,
        dataset_specs=(spec,),
    ).run()

    assert report.status == StageStatus.PASS
    tables_path = Path(report.artifacts["judge_tables.fresh"])
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    assert tables["judge_source"] == "fresh_rerun"
    assert tables["metrics"]["judge_source"] == "fresh_rerun"
    assert tables["metrics"]["n_edits"] == len(frozen_jobs)
    assert all(
        stage.status == StageStatus.PASS
        for stage in report.stages
        if stage.stage.startswith(("teacher.", "acceptability.", "judge_metrics."))
    )


@pytest.mark.parametrize(
    ("known", "expected_status"),
    [
        ({}, StageStatus.FAIL),
        (
            {("metc", DATASET_ID): "documented fixture drift"},
            StageStatus.KNOWN_DIFFERENCE,
        ),
    ],
)
def test_semantic_mismatch_requires_explicit_known_difference(
    tmp_path: Path,
    known: dict,
    expected_status: StageStatus,
) -> None:
    executor = FakeExecutor(metc_mapping="DIFFERENT")
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("metc",),
        known_differences=known,
    )
    report = E2ERunner(
        config,
        executor=executor,
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    assert report.status == expected_status
    comparison = next(
        stage
        for stage in report.stages
        if stage.stage == f"system.metc.{DATASET_ID}.compare"
    )
    assert comparison.status == expected_status
    assert comparison.details["comparison"]["equal"] is False


def test_portable_metc_mismatch_is_reported_as_known_difference(
    tmp_path: Path,
) -> None:
    executor = FakeExecutor(metc_mapping="DIFFERENT")
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("metc",),
        metc_inference_profile="portable",
    )
    report = E2ERunner(
        config,
        executor=executor,
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    assert report.status == StageStatus.KNOWN_DIFFERENCE
    comparison = next(
        stage
        for stage in report.stages
        if stage.stage == f"system.metc.{DATASET_ID}.compare"
    )
    assert comparison.status == StageStatus.KNOWN_DIFFERENCE
    assert "portable fp32 inference" in comparison.details["known_difference"]


def test_portable_metc_adapter_plan_is_explicitly_nonpaper(
    tmp_path: Path,
) -> None:
    config = E2EConfig(
        data_root=tmp_path / "data",
        work_root=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        software_root=tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("metc",),
        metc_inference_profile="portable",
    )

    plan = default_adapter_plan(
        "metc",
        dataset_spec("en"),
        tmp_path / "pairs.jsonl",
        tmp_path / "output",
        config,
    )

    assert plan.command is not None
    assert plan.command[plan.command.index("--device") + 1] == "auto"
    assert plan.command[plan.command.index("--inference-profile") + 1] == "portable"
    assert "--allow-nonpaper" in plan.command


def test_portable_metc_profile_cannot_drive_live_llm_reruns(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="teacher and judge reruns require the exact METC paper profile",
    ):
        _config(
            tmp_path,
            runtimes=_runtime_paths(tmp_path),
            systems=("metc", "metc_teacher"),
            metc_inference_profile="portable",
            run_llm=True,
        )


def test_unavailable_exact_runtime_is_never_reported_as_pass(tmp_path: Path) -> None:
    executor = FakeExecutor()
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path, metc=False),
        systems=("metc",),
    )
    report = E2ERunner(
        config,
        executor=executor,
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    assert report.status == StageStatus.KNOWN_DIFFERENCE
    unavailable = next(
        stage
        for stage in report.stages
        if stage.stage == f"system.metc.{DATASET_ID}.execute"
    )
    assert unavailable.status == StageStatus.KNOWN_DIFFERENCE
    assert "not executed" in unavailable.summary
    reference = next(
        stage
        for stage in report.stages
        if stage.stage == f"reference_matching.{DATASET_ID}"
    )
    assert reference.status == StageStatus.KNOWN_DIFFERENCE


def test_successful_process_cannot_reuse_stale_adapter_output(tmp_path: Path) -> None:
    delegate = FakeExecutor()

    def executor(argv, *, cwd, environment, timeout) -> CommandResult:
        command = tuple(argv)
        module = command[command.index("-m") + 1]
        if module == "metc_artifact.datasets":
            return delegate(
                argv,
                cwd=cwd,
                environment=environment,
                timeout=timeout,
            )
        return CommandResult(0)

    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("metc",),
    )
    stale = (
        config.work_root
        / "system-annotations"
        / DATASET_ID
        / "metc.jsonl"
    )
    _jsonl(
        stale,
        [
            {
                "dataset_id": DATASET_ID,
                "item_id": "0",
                "system": "metc",
                "span_unit": "unicode_codepoint",
                "edits": [
                    {
                        "start": 0,
                        "end": 1,
                        "replacement": "b",
                        "native_label": "Grammar/Agreement",
                        "mapped_label": "Nn",
                    }
                ],
            }
        ],
    )

    report = E2ERunner(
        config,
        executor=executor,
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    assert report.status == StageStatus.FAIL
    comparison = next(
        stage
        for stage in report.stages
        if stage.stage == f"system.metc.{DATASET_ID}.compare"
    )
    assert comparison.status == StageStatus.FAIL
    assert "could not be materialized" in comparison.summary


def test_successful_reconstruction_process_cannot_reuse_stale_output(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        runtimes=_runtime_paths(tmp_path),
        systems=("errant",),
    )
    stale = (
        config.work_root
        / "reconstructed"
        / DATASET_ID
        / "annotations.jsonl"
    )
    _jsonl(
        stale,
        [
            {
                "sent_idx": 0,
                "src": "a",
                "tgt": "b",
                "gold": [[0, 1, "Nn", "Nn", "b"]],
            }
        ],
    )

    report = E2ERunner(
        config,
        executor=lambda *_args, **_kwargs: CommandResult(0),
        data_validator=_valid_data,
        dataset_specs=(dataset_spec("en"),),
    ).run()

    assert report.status == StageStatus.FAIL
    comparison = next(
        stage
        for stage in report.stages
        if stage.stage == f"dataset.{DATASET_ID}.gold"
    )
    assert comparison.status == StageStatus.FAIL
    assert "could not compare" in comparison.summary


@pytest.mark.parametrize(
    ("dataset", "paper_compat"),
    [
        ("en_conll14_ann0", False),
        ("en_conll14_ann1", False),
        ("de_merlin", True),
        ("cs_merlin", True),
        ("ko_kolla", True),
        ("uk_uagec", True),
        ("zh_mucgec", False),
    ],
)
def test_jp_plan_keeps_raw_sidecars_for_paper_compatibility(
    tmp_path: Path,
    dataset: str,
    paper_compat: bool,
) -> None:
    runtimes = _runtime_paths(tmp_path)
    if dataset == "zh_mucgec":
        gsdltp = tmp_path / "gsdltp"
        gsdltp.mkdir()
        runtimes = RuntimePaths(**{**runtimes.__dict__, "jp_zh_gsdltp_dir": gsdltp})
    config = E2EConfig(
        data_root=tmp_path / "data",
        work_root=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        software_root=tmp_path,
        runtimes=runtimes,
    )

    plan = default_adapter_plan(
        "jp",
        dataset_spec(dataset),
        tmp_path / "pairs.jsonl",
        tmp_path / "output",
        config,
    )

    assert plan.command is not None
    assert ("--paper-compat" in plan.command) is paper_compat
    assert ("--raw-output" in plan.command) is paper_compat
    assert (plan.raw_output_path is not None) is paper_compat


def test_cli_uses_distinct_exit_code_for_incomplete_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class KnownRunner:
        def __init__(self, _config) -> None:
            pass

        def run(self):
            from metc_artifact.e2e import E2EReport, StageResult

            return E2EReport(
                (
                    StageResult(
                        "system.metc.en_conll14_ann0.execute",
                        StageStatus.KNOWN_DIFFERENCE,
                        "CUDA unavailable",
                    ),
                ),
                {},
            )

    monkeypatch.setattr("metc_artifact.e2e.E2ERunner", KnownRunner)
    exit_code = main(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--work-root",
            str(tmp_path / "work"),
        ]
    )

    assert exit_code == 2
