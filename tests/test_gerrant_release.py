from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baselines" / "gerrant_de"
MANIFEST = json.loads((BASELINE_DIR / "manifest.json").read_text(encoding="utf-8"))


def _load_preparer():
    path = BASELINE_DIR / "prepare_resources.py"
    spec = importlib.util.spec_from_file_location("gerrant_prepare_resources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_code_files_match_the_reviewed_manifest() -> None:
    assert set(MANIFEST["code_files"]) == {
        "align_text.py",
        "cat_rules_de.py",
        "gerrant_de.py",
        "prepare_resources.py",
        "rdlextra.py",
        "toolbox.py",
    }
    for name, expected in MANIFEST["code_files"].items():
        path = BASELINE_DIR / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        ast.parse(path.read_text(encoding="utf-8"), filename=name)


def test_problematic_resources_are_not_bundled() -> None:
    relative_files = {
        path.relative_to(BASELINE_DIR).as_posix()
        for path in BASELINE_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert "resources/de_DE-large.txt" not in relative_files
    assert "resources/de-stts_map" not in relative_files
    assert not any("stanza" in name.casefold() for name in relative_files)
    assert not any("jp_errant" in name.casefold() for name in relative_files)
    assert not any("gemma" in name.casefold() for name in relative_files)
    assert all(not resource["bundled"] for resource in MANIFEST["resources"].values())


def test_stts_compatibility_table_is_complete_and_exact() -> None:
    preparer = _load_preparer()
    rendered = preparer.render_stts_map()
    rows = rendered.decode("utf-8").splitlines()

    assert len(rows) == 56
    assert "NNE\tPNOUN" in rows
    assert "PROAV\tADV" in rows
    assert (
        hashlib.sha256(rendered).hexdigest()
        == MANIFEST["resources"]["de-stts_map"]["derived_sha256"]
    )


def test_pinned_resource_contract() -> None:
    preparer = _load_preparer()
    stts = MANIFEST["resources"]["de-stts_map"]
    dictionary = MANIFEST["resources"]["de_DE-large.txt"]
    assert stts["construction"].startswith("Author-created")
    assert stts["derived_rows"] == len(preparer.STTS_TO_UNIVERSAL)
    assert preparer.STTS_MAP_SHA256 == stts["derived_sha256"]
    assert (preparer.IGERMAN_URL, preparer.IGERMAN_SIZE, preparer.IGERMAN_SHA256) == (
        dictionary["source_url"],
        dictionary["source_size_bytes"],
        dictionary["source_sha256"],
    )
    assert (preparer.DICTIONARY_SIZE, preparer.DICTIONARY_SHA256) == (
        dictionary["derived_size_bytes"],
        dictionary["derived_sha256"],
    )


def test_wrong_local_resource_is_rejected(tmp_path: Path) -> None:
    preparer = _load_preparer()
    path = tmp_path / "wrong.txt"
    path.write_bytes(b"wrong")
    with pytest.raises(preparer.ResourceError, match="expected"):
        preparer._checked_bytes(path, expected_size=5, expected_sha256="0" * 64)


def test_verified_dictionary_seed_bypasses_host_dictionary_build(
    tmp_path: Path,
) -> None:
    preparer = _load_preparer()
    expected_stts = b"NN\tNOUN\n"
    dictionary = tmp_path / "de_DE-large.txt"
    dictionary.write_bytes(b"seeded dictionary\n")
    output = tmp_path / "resources"

    with (
        patch.object(preparer, "STTS_TO_UNIVERSAL", (("NN", "NOUN"),)),
        patch.object(preparer, "STTS_MAP_ROWS", 1),
        patch.object(
            preparer,
            "STTS_MAP_SHA256",
            hashlib.sha256(expected_stts).hexdigest(),
        ),
        patch.object(preparer, "DICTIONARY_SIZE", dictionary.stat().st_size),
        patch.object(
            preparer,
            "DICTIONARY_SHA256",
            hashlib.sha256(dictionary.read_bytes()).hexdigest(),
        ),
        patch.object(
            preparer,
            "render_dictionary",
            side_effect=AssertionError("dictionary build must not run"),
        ),
    ):
        preparer.prepare_resources(
            output,
            igerman_archive=None,
            dictionary=dictionary,
            force=False,
        )

    assert (output / "de-stts_map").read_bytes() == expected_stts
    assert (output / "de_DE-large.txt").read_bytes() == dictionary.read_bytes()


def test_dictionary_and_archive_arguments_are_mutually_exclusive() -> None:
    preparer = _load_preparer()

    with pytest.raises(SystemExit):
        preparer.parse_args(
            [
                "--igerman-archive",
                "igerman.tar.bz2",
                "--dictionary",
                "de_DE-large.txt",
            ]
        )


def test_upstream_mit_notice_and_modification_record_are_present() -> None:
    license_text = (BASELINE_DIR / "LICENSE-MIT").read_text(encoding="utf-8")
    modifications = (BASELINE_DIR / "MODIFICATIONS.md").read_text(encoding="utf-8")
    assert "Copyright (c) 2017 Christopher Bryant, Mariano Felice" in license_text
    assert "The above copyright notice and this permission notice" in license_text
    assert MANIFEST["upstream"]["commit"] in modifications


def test_historical_materialization_exclusion_is_explicit() -> None:
    regression = MANIFEST["paper_regression"]
    assert regression["generated_native_edits"] == 2_880
    assert regression["materialized_evaluation_edits"] == 2_879
    assert regression["historical_duplicate_source_excluded_edits"] == 1
    assert (
        regression["generated_native_edits"]
        - regression["historical_duplicate_source_excluded_edits"]
        == regression["materialized_evaluation_edits"]
    )
