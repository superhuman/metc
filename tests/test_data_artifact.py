from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import metc_artifact.data_artifact as data_artifact
from metc_artifact.data_artifact import (
    DataArtifactError,
    _validate_complete_inventory,
    _validate_human,
    _validate_runtime_resources,
    validate_data_artifact,
)
from metc_artifact.judge_jobs import EDIT_ID_VERSION, build_judge_jobs
from metc_artifact.judges.protocols import ACCEPTABILITY_PROMPT
from metc_artifact.schema import DATASETS


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _build_fixture(root: Path) -> Path:
    historical_prompt_digest = hashlib.sha256(b"frozen teacher prompt fixture").hexdigest()
    historical_taxonomy_digest = hashlib.sha256(b"frozen teacher taxonomy fixture").hexdigest()
    manifest: dict = {
        "schema_version": "metc-data-v2",
        "benchmarks": {},
        "metc_teacher": {
            **data_artifact.TEACHER_CONTRACT,
            "prompt": {
                "reviewed_source_path": (
                    "gec-annotations/scripts/metc_annotation/annotation/prompt.py"
                ),
                "reviewed_source_sha256": (data_artifact.TEACHER_REVIEWED_PROMPT_SHA256),
                "code_artifact_path": "prompts/annotation_prompt.py",
                "code_artifact_sha256": historical_prompt_digest,
            },
            "taxonomy": {
                "code_artifact_path": "prompts/taxonomy_v2.md",
                "sha256": historical_taxonomy_digest,
            },
            "datasets": {},
            "totals": {"sentences": 0, "edits": 0},
        },
        "judges": {
            "primary": {
                **data_artifact.JUDGE_CONTRACTS["primary"],
                "datasets": {},
            },
            "second": {
                **data_artifact.JUDGE_CONTRACTS["second"],
                "datasets": {},
            },
        },
        "human_annotations": {},
        "judge_protocol": {
            "prompt_path_in_code_artifact": "prompts/acceptability_judge.py",
            "prompt_sha256": hashlib.sha256(ACCEPTABILITY_PROMPT.encode("utf-8")).hexdigest(),
            "stable_edit_id_version": EDIT_ID_VERSION,
        },
    }
    for spec in DATASETS:
        if spec.edit_format == "char_tuple":
            row = {"idx": 0, "src": "a", "tgt": "b"}
            edit = [0, 1, "mapped", "native", "b"]
        else:
            row = {"idx": 0, "orig_tokens": ["a"], spec.target_tokens_key: ["b"]}
            edit = {
                "orig_start": 0,
                "orig_end": 1,
                "raw_label": "native",
                "target": "mapped",
                "replacement": "b",
            }
        for system in spec.systems:
            row[system] = [edit] if system == "gold" else []
        relative = Path("benchmarks") / spec.dataset_id / "annotations.jsonl"
        file_meta = _write_jsonl(root / relative, [row])
        file_meta["path"] = relative.as_posix()
        manifest["benchmarks"][spec.dataset_id] = {
            **file_meta,
            "items": 1,
            "experiment_code": spec.experiment_code,
            "edit_format": spec.edit_format,
            "evaluation_span_unit": spec.evaluation_span_unit,
            "systems": list(spec.systems),
            "system_edit_counts": {system: 1 if system == "gold" else 0 for system in spec.systems},
        }
        teacher_relative = Path("teacher") / "raw" / f"{spec.dataset_id}.jsonl"
        teacher_meta = _write_jsonl(root / teacher_relative, [])
        teacher_meta.update({"path": teacher_relative.as_posix(), "sentences": 0, "edits": 0})
        manifest["metc_teacher"]["datasets"][spec.dataset_id] = teacher_meta
        job = dict(build_judge_jobs([row], spec).jobs[0])
        for judge_name in ("primary", "second"):
            backend = data_artifact.JUDGE_CONTRACTS[judge_name]["recorded_backend_id"]
            canonical = {
                **job,
                "label_gloss": "A test gloss.",
                "span_quality": "Good",
                "label_quality": "Acceptable",
                "judge": backend,
                "errored": False,
            }
            canonical_relative = (
                Path("judges") / "canonical" / judge_name / f"{spec.dataset_id}.jsonl"
            )
            canonical_meta = _write_jsonl(root / canonical_relative, [canonical])
            canonical_meta.update(
                {
                    "path": canonical_relative.as_posix(),
                    "edits": 1,
                    "contains_free_text_rationales": False,
                }
            )
            raw_relative = Path("judges") / "raw" / judge_name / spec.dataset_id / "base.jsonl"
            raw_row = {field: job[field] for field in ("system", "row_idx", "label", "edit_diff")}
            raw_row.update(
                {
                    "span_quality": canonical["span_quality"],
                    "label_quality": canonical["label_quality"],
                    "judge": backend,
                }
            )
            raw_meta = _write_jsonl(root / raw_relative, [raw_row])
            raw_meta.update({"path": raw_relative.as_posix(), "rows": 1})
            manifest["judges"][judge_name]["datasets"][spec.dataset_id] = {
                "canonical": canonical_meta,
                "raw": {"base": raw_meta},
                "expected_jobs": 1,
                "base_rows": 1,
                "overlay_rows": 0,
                "base_rows_replaced_by_overlay": 0,
                "identical_retry_rows_excluded": 0,
            }
    (root / "manifest.full.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root


def test_complete_data_validator_checks_scientific_content(tmp_path: Path) -> None:
    report = validate_data_artifact(_build_fixture(tmp_path / "data"))

    assert report.benchmark_items == 7
    assert report.canonical_judge_edits == 14
    assert report.teacher_edits == 0


def test_complete_data_validator_rejects_wrong_edit_identity(tmp_path: Path) -> None:
    root = _build_fixture(tmp_path / "data")
    manifest_path = root / "manifest.full.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["judges"]["primary"]["datasets"]["en_conll14_ann0"]["canonical"]
    path = root / metadata["path"]
    row = json.loads(path.read_text(encoding="utf-8"))
    row["edit_start"] = 1
    rewritten = _write_jsonl(path, [row])
    metadata.update({key: rewritten[key] for key in ("size_bytes", "sha256")})
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(DataArtifactError, match="does not match released edit"):
        validate_data_artifact(root)


def test_complete_data_validator_rejects_raw_verdict_mutation(
    tmp_path: Path,
) -> None:
    root = _build_fixture(tmp_path / "data")
    manifest_path = root / "manifest.full.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["judges"]["primary"]["datasets"]["en_conll14_ann0"]["raw"]["base"]
    path = root / metadata["path"]
    row = json.loads(path.read_text(encoding="utf-8"))
    row["span_quality"] = "Bad"
    rewritten = _write_jsonl(path, [row])
    metadata.update({key: rewritten[key] for key in ("size_bytes", "sha256")})
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(DataArtifactError, match="differs from raw materialization"):
        validate_data_artifact(root)


def test_complete_data_validator_applies_complete_stable_id_overlay(
    tmp_path: Path,
) -> None:
    root = _build_fixture(tmp_path / "data")
    manifest_path = root / "manifest.full.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest["judges"]["primary"]["datasets"]["en_conll14_ann0"]
    canonical_path = root / dataset["canonical"]["path"]
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))

    base_metadata = dataset["raw"]["base"]
    base_path = root / base_metadata["path"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base["span_quality"] = "Bad"
    rewritten_base = _write_jsonl(base_path, [base])
    base_metadata.update({key: rewritten_base[key] for key in ("size_bytes", "sha256")})

    overlay = {
        "edit_id": canonical["edit_id"],
        "edit_idx": canonical["edit_idx"],
        "system": canonical["system"],
        "row_idx": canonical["row_idx"],
        "label": canonical["label"],
        "span_quality": canonical["span_quality"],
        "label_quality": canonical["label_quality"],
        "judge": canonical["judge"],
        "errored": False,
    }
    overlay_relative = Path("judges/raw/primary/en_conll14_ann0/stable_id_overlay.jsonl")
    overlay_metadata = _write_jsonl(root / overlay_relative, [overlay])
    overlay_metadata.update({"path": overlay_relative.as_posix(), "rows": 1})
    dataset["raw"]["stable_id_overlay"] = overlay_metadata
    dataset["overlay_rows"] = 1
    dataset["base_rows_replaced_by_overlay"] = 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    validate_data_artifact(root)

    overlay["label_quality"] = "Bad"
    rewritten_overlay = _write_jsonl(root / overlay_relative, [overlay])
    overlay_metadata.update({key: rewritten_overlay[key] for key in ("size_bytes", "sha256")})
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(DataArtifactError, match="differs from raw materialization"):
        validate_data_artifact(root)


def test_complete_data_validator_rejects_judge_prompt_checksum_mutation(
    tmp_path: Path,
) -> None:
    root = _build_fixture(tmp_path / "data")
    manifest_path = root / "manifest.full.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["judge_protocol"]["prompt_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(DataArtifactError, match="prompt text or checksum"):
        validate_data_artifact(root)


@pytest.mark.parametrize("section,key", [("prompt", "code_artifact_sha256"), ("taxonomy", "sha256")])
def test_complete_data_validator_rejects_malformed_frozen_provenance_hash(
    tmp_path: Path, section: str, key: str
) -> None:
    root = _build_fixture(tmp_path / "data")
    manifest_path = root / "manifest.full.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metc_teacher"][section][key] = "g" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(DataArtifactError, match=rf"metc_teacher\.{section}\.{key}: expected"):
        validate_data_artifact(root)


def test_complete_data_validator_enforces_flat_payload_inventory(
    tmp_path: Path,
) -> None:
    root = _build_fixture(tmp_path / "data")
    for name in (
        "CC-BY-4.0-NOTICE.txt",
        "CC-BY-SA-4.0-NOTICE.txt",
        "CONLL-2014-NOTICE.txt",
        "README.md",
        "LICENSE",
        "LICENSE-APACHE-2.0",
        "LICENSE-GPL-3.0",
        "DATASET_LICENSES.md",
    ):
        (root / name).write_text(f"{name}\n", encoding="utf-8")

    manifest_path = root / "manifest.full.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_status"] = {
        "stage": "complete",
        "complete_benchmark_text": True,
        "complete_system_annotations": True,
        "complete_raw_teacher_caches": True,
        "complete_frozen_judges": True,
        "complete_deidentified_human_annotations": True,
        "complete_human_sampling_frames": True,
        "complete_runtime_resources": True,
    }
    manifest["files"] = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    _validate_complete_inventory(root, manifest)

    (root / "unexpected.txt").write_text("not inventoried\n", encoding="utf-8")
    with pytest.raises(DataArtifactError, match="payload inventory mismatch"):
        _validate_complete_inventory(root, manifest)


def test_complete_data_validator_pins_gerrant_runtime_resource_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    fields = {
        "schema_version": "test-gerrant-v1",
        "license_expression": "GPL-test",
        "upstream_url": "https://example.test/source.tar.bz2",
    }

    def install(relative: str, data: bytes, extra: dict | None = None) -> dict:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {
            "path": relative,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            **(extra or {}),
        }

    files = {
        "source_archive": install("resources/gerrant/source.tar.bz2", b"archive"),
        "derived_dictionary": install(
            "resources/gerrant/dictionary.txt",
            b"dictionary",
            {"derivation": "test-derivation"},
        ),
        "source_notice": install("resources/gerrant/SOURCE.md", b"notice"),
    }
    notices = {
        "Copyright": install("resources/gerrant/Copyright", b"copyright"),
        "COPYING": install("resources/gerrant/COPYING", b"license"),
    }
    monkeypatch.setattr(data_artifact, "GERRANT_RUNTIME_FIELDS", fields)
    monkeypatch.setattr(data_artifact, "GERRANT_RUNTIME_FILES", files)
    monkeypatch.setattr(data_artifact, "GERRANT_UPSTREAM_NOTICES", notices)
    manifest = {
        "release_status": {
            "stage": "complete",
            "complete_runtime_resources": True,
        },
        "runtime_resources": {
            "gerrant": {
                **fields,
                **files,
                "upstream_notices": notices,
            }
        },
    }

    _validate_runtime_resources(root, manifest)

    dictionary = root / files["derived_dictionary"]["path"]
    dictionary.write_bytes(b"dicti0nary")
    with pytest.raises(DataArtifactError, match="SHA-256"):
        _validate_runtime_resources(root, manifest)


def test_complete_human_annotation_claim_accepts_exact_released_studies(
    tmp_path: Path,
) -> None:
    manifest = {
        "release_status": {"complete_deidentified_human_annotations": True},
        "human_annotations": {
            "en_conll14_ann0": {},
            "de_merlin": {},
        },
    }

    # Exact coverage passes the new completeness gate and reaches normal file
    # metadata validation.
    with pytest.raises(DataArtifactError, match="invalid artifact path"):
        _validate_human(tmp_path, manifest, {"en_conll14_ann0": {}, "de_merlin": {}})


@pytest.mark.parametrize(
    "human_datasets",
    [
        {"en_conll14_ann0": {}},
        {"en_conll14_ann0": {}, "de_merlin": {}, "cs_merlin": {}},
    ],
)
def test_complete_human_annotation_claim_rejects_inexact_study_coverage(
    tmp_path: Path,
    human_datasets: dict[str, dict],
) -> None:
    manifest = {
        "release_status": {"complete_deidentified_human_annotations": True},
        "human_annotations": human_datasets,
    }

    with pytest.raises(
        DataArtifactError,
        match="complete deidentified human annotations require exactly",
    ):
        _validate_human(
            tmp_path,
            manifest,
            {dataset_id: {} for dataset_id in human_datasets},
        )


@pytest.mark.parametrize(
    "location",
    [
        "assignment",
        "assignment_edit",
        "sampling_frame",
        "sampling_frame_edit",
    ],
)
def test_human_validator_rejects_unrecognized_fields(tmp_path: Path, location: str) -> None:
    root = tmp_path / "data"
    dataset_id = "en_conll14_ann0"
    row_idx = 1
    system = "metc"
    edit_id = "metc-edit-v1:en:1:metc:0"
    sample_id = data_artifact.public_sample_id(dataset_id, row_idx, system)
    frame_edit = {
        "slot": 1,
        "edit_id": edit_id,
        "edit_idx": 0,
        "label": "Grammar/Noun",
        "displayed_span": "{a=>b:::edit_1}",
        "displayed_edit": "Problem with a noun.",
    }
    assignment_edit = {
        **frame_edit,
        "unannotatable": False,
        "span_quality": "Good",
        "span_confidence": "High",
        "label_quality": "Good",
        "label_confidence": "High",
    }
    frame = {
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "selection_rank": 0,
        "selected_for_human_study": True,
        "item_id": "1",
        "row_idx": row_idx,
        "system": system,
        "edit_count": 1,
        "language": "English",
        "source_sentence": "a",
        "decorated_sentence": "{a=>b:::edit_1}",
        "corrected_sentence": "b",
        "edits": [frame_edit],
    }
    assignment = {
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "assignment_order": 0,
        "pass_id": "pass-v1:1",
        "item_id": "1",
        "row_idx": row_idx,
        "system": system,
        "edit_count": 1,
        "status": "scorable",
        "session_unannotatable": False,
        "language": "English",
        "source_sentence": "a",
        "decorated_sentence": "{a=>b:::edit_1}",
        "corrected_sentence": "b",
        "edits": [assignment_edit],
    }
    target = {
        "assignment": assignment,
        "assignment_edit": assignment_edit,
        "sampling_frame": frame,
        "sampling_frame_edit": frame_edit,
    }[location]
    target["annotator_uuid"] = "private"

    assignment_relative = Path("human/en_conll14_ann0.jsonl")
    assignment_metadata = _write_jsonl(root / assignment_relative, [assignment])
    assignment_metadata["path"] = assignment_relative.as_posix()
    frame_relative = Path("human/sampling_frames/en_conll14_ann0.jsonl")
    frame_metadata = _write_jsonl(root / frame_relative, [frame])
    frame_metadata.update(
        {
            "path": frame_relative.as_posix(),
            "instrument_items": 1,
            "selected_items": 1,
            "system_item_counts": {"metc": 1},
            "selected_system_item_counts": {"metc": 1},
        }
    )
    manifest = {
        "human_annotations": {
            dataset_id: {
                **assignment_metadata,
                "assignments": 1,
                "samples": 1,
                "passes_per_sample": 1,
                "sampling_frame": frame_metadata,
                "status_counts": {"scorable": 1},
                "system_assignment_counts": {"metc": 1},
            }
        }
    }
    expected = {
        dataset_id: {
            edit_id: {
                "row_idx": row_idx,
                "system": system,
                "edit_idx": 0,
                "label": "Grammar/Noun",
            }
        }
    }

    with pytest.raises(DataArtifactError, match="unexpected=.*annotator_uuid"):
        _validate_human(root, manifest, expected)
