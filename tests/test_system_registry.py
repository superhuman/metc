from __future__ import annotations

import tomllib
from pathlib import Path

from metc_artifact.hydration import PYTHON_VERSION
from metc_artifact.schema import DATASETS
from metc_artifact.systems import SYSTEMS, SystemRole, system_spec, systems_for_dataset


def test_registry_covers_every_evaluated_annotation_key_per_dataset() -> None:
    for dataset in DATASETS:
        producers = systems_for_dataset(dataset.dataset_id)
        observed = {producer.annotation_key for producer in producers}
        assert observed == set(dataset.systems)
        assert sum(producer.annotation_key == "gold" for producer in producers) == 1


def test_registry_has_all_isolated_producers() -> None:
    assert set(SYSTEMS) == {
        "metc",
        "metc_teacher",
        "errant",
        "gerrant",
        "jp_errant",
        "conll14_reference",
        "merlin_reference",
        "kolla_reference",
        "uagec_reference",
        "cherrant_reference",
    }
    executable_variables = {
        producer.runtime.executable_environment_variable for producer in SYSTEMS.values()
    }
    assert len(executable_variables) == len(SYSTEMS)
    for producer in SYSTEMS.values():
        configuration_keys = [
            key for key, _value in producer.provenance.evaluated_configuration
        ]
        assert len(configuration_keys) == len(set(configuration_keys))


def test_baseline_applicability_matches_paper_settings() -> None:
    assert system_spec("errant").dataset_ids == (
        "en_conll14_ann0",
        "en_conll14_ann1",
    )
    assert system_spec("gerrant").dataset_ids == ("de_merlin",)
    assert set(system_spec("jp_errant").dataset_ids) == {
        dataset.dataset_id for dataset in DATASETS
    }
    assert system_spec("cherrant_reference").dataset_ids == ("zh_mucgec",)


def test_provenance_pins_result_affecting_revisions_and_patches() -> None:
    assert system_spec("errant").provenance.revision == (
        "4a36d9bf7fa1d6109bea79b23fccc5d7731fabc1"
    )
    assert system_spec("gerrant").provenance.revision == (
        "92a86d0722536f4b2c6a4a10037b64f3955ad1b5"
    )
    jp = system_spec("jp_errant").provenance
    assert jp.revision == "839ac8641b15272c2fcb6bc74015e88dda4a8842"
    assert any("empty-table fallback" in modification for modification in jp.local_modifications)
    cherrant = system_spec("cherrant_reference")
    assert cherrant.role == SystemRole.REFERENCE_CONSTRUCTOR
    assert cherrant.provenance.revision == "39221934ee4ccf07801879136a75dd7e11319977"


def test_registry_contains_no_private_or_obsolete_release_claims() -> None:
    rendered = repr(SYSTEMS)
    for prohibited in (
        "internal " + "backend",
        "not " + "bundled",
        "pen" + "ding",
        "not yet " + "wired",
    ):
        assert prohibited not in rendered
    for producer in SYSTEMS.values():
        repository = producer.provenance.repository
        assert repository is None or repository.startswith("https://github.com/")
    assert "separate integration work" not in rendered


def test_registry_describes_current_teacher_and_reference_paths() -> None:
    teacher = system_spec("metc_teacher")
    [credential] = teacher.runtime.required_environment
    assert credential.any_of == ("OPENAI_API_KEY",)
    assert teacher.command.argv[:4] == (
        "{executable}",
        "-m",
        "metc_artifact.judges.batch",
        "run-teacher",
    )
    assert "annotation prompt" in teacher.availability_note
    assert "direct OpenAI client" in teacher.availability_note

    metc = system_spec("metc")
    assert "canonical E2E METC adapter" in metc.availability_note
    assert "materializes annotations" in metc.availability_note

    for system_id in (
        "conll14_reference",
        "merlin_reference",
        "kolla_reference",
        "uagec_reference",
        "cherrant_reference",
    ):
        assert "Implemented in the bundled public-dataset reconstruction path" in (
            system_spec(system_id).availability_note
        )

    assert "exact 2,879-edit paper materialization" in (
        system_spec("gerrant").availability_note
    )


def test_cherrant_registry_matches_hydrated_runtime_lock() -> None:
    lock_root = Path(__file__).resolve().parents[1] / "runtime-locks" / "cherrant"
    project = tomllib.loads((lock_root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((lock_root / "uv.lock").read_text(encoding="utf-8"))
    runtime = system_spec("cherrant_reference").runtime

    assert runtime.python_version == PYTHON_VERSION == "3.11.13"
    assert project["project"]["requires-python"] == lock["requires-python"] == "==3.11.*"
    assert runtime.python_version.startswith("3.11.")

    locked_versions = {
        package["name"]: package["version"]
        for package in lock["package"]
    }
    project_dependencies = tuple(project["project"]["dependencies"])
    assert runtime.dependencies == project_dependencies
    for dependency in runtime.dependencies:
        name, expected_version = dependency.split("==", 1)
        assert locked_versions[name] == expected_version


def test_metc_provenance_matches_the_hosted_checkpoint_contract() -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "model" / "checkpoint.json").read_text(encoding="utf-8"))
    provenance = system_spec("metc").provenance
    configuration = dict(provenance.evaluated_configuration)

    assert configuration["checkpoint_archive_sha256"] == contract["archive"]["sha256"]
    assert provenance.repository == contract["landing_url"]
