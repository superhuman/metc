import hashlib
import json
import os
from pathlib import Path

import pytest

import metc.tokenizer_preparation as preparation
from metc.errors import CheckpointContractError


def _metadata(files: dict[str, bytes]) -> dict[str, tuple[int, str]]:
    return {
        name: (len(content), hashlib.sha256(content).hexdigest()) for name, content in files.items()
    }


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for relative, content in files.items():
        (root / relative).write_bytes(content)


def test_exact_existing_tokenizer_is_verified_without_reconstruction(tmp_path, monkeypatch):
    prepared = {
        "special_tokens_map.json": b"special",
        "tokenizer.json": b"tokenizer",
        "tokenizer_config.json": b"config",
    }
    _write_files(tmp_path, prepared)
    monkeypatch.setattr(
        preparation,
        "EXPECTED_PREPARED_TOKENIZER_FILES",
        _metadata(prepared),
    )

    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        preparation,
        "verify_archived_checkpoint_files",
        lambda path: calls.append(("archive", path)),
    )
    monkeypatch.setattr(
        preparation,
        "verify_checkpoint_files",
        lambda path: calls.append(("full", path)),
    )

    def fail_reconstruction(*_args, **_kwargs):
        raise AssertionError("an exact existing tokenizer must not be reconstructed")

    monkeypatch.setattr(preparation, "_download_upstream_files", fail_reconstruction)
    monkeypatch.setattr(preparation, "_reconstruct_tokenizer", fail_reconstruction)

    result = preparation.prepare_tokenizer(tmp_path)

    assert result == {
        "checkpoint": str(tmp_path.resolve()),
        "repository": preparation.UPSTREAM_REPOSITORY,
        "revision": preparation.UPSTREAM_REVISION,
        "status": "already_prepared",
    }
    assert calls == [("archive", tmp_path.resolve()), ("full", tmp_path.resolve())]


def test_existing_tokenizer_byte_drift_is_rejected_before_reconstruction(tmp_path, monkeypatch):
    expected = {"tokenizer.json": b"expected"}
    (tmp_path / "tokenizer.json").write_bytes(b"expEcted")
    monkeypatch.setattr(
        preparation,
        "EXPECTED_PREPARED_TOKENIZER_FILES",
        _metadata(expected),
    )
    monkeypatch.setattr(preparation, "verify_archived_checkpoint_files", lambda _path: None)

    reconstructed = False

    def record_reconstruction(*_args, **_kwargs):
        nonlocal reconstructed
        reconstructed = True

    monkeypatch.setattr(preparation, "_reconstruct_tokenizer", record_reconstruction)

    with pytest.raises(
        CheckpointContractError,
        match="existing tokenizer file does not match the evaluated checkpoint",
    ):
        preparation.prepare_tokenizer(tmp_path)

    assert not reconstructed


def test_source_hash_drift_is_rejected_before_reconstruction(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    source = tmp_path / "source"
    checkpoint.mkdir()
    source.mkdir()
    expected_source = {"tokenizer.json": b"upstream"}
    (source / "tokenizer.json").write_bytes(b"upstreAm")

    monkeypatch.setattr(preparation, "UPSTREAM_FILES", _metadata(expected_source))
    monkeypatch.setattr(
        preparation,
        "EXPECTED_PREPARED_TOKENIZER_FILES",
        {"tokenizer.json": (1, hashlib.sha256(b"x").hexdigest())},
    )
    monkeypatch.setattr(preparation, "verify_archived_checkpoint_files", lambda _path: None)

    reconstructed = False

    def record_reconstruction(*_args, **_kwargs):
        nonlocal reconstructed
        reconstructed = True

    monkeypatch.setattr(preparation, "_reconstruct_tokenizer", record_reconstruction)

    with pytest.raises(CheckpointContractError, match="SHA-256"):
        preparation.prepare_tokenizer(checkpoint, source_dir=source)

    assert not reconstructed


def test_download_accepts_hugging_face_cache_symlink_and_copies_bytes(tmp_path, monkeypatch):
    source_bytes = b"pinned upstream tokenizer bytes"
    blob = tmp_path / "blob"
    blob.write_bytes(source_bytes)
    cached = tmp_path / "cached-tokenizer.json"
    cached.symlink_to(blob)
    destination = tmp_path / "destination"
    destination.mkdir()

    monkeypatch.setattr(
        preparation,
        "UPSTREAM_FILES",
        {"tokenizer.json": _metadata({"tokenizer.json": source_bytes})["tokenizer.json"]},
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **_kwargs: str(cached),
    )

    preparation._download_upstream_files(destination)

    copied = destination / "tokenizer.json"
    assert copied.read_bytes() == source_bytes
    assert not copied.is_symlink()


def test_partial_exact_tokenizer_installs_only_missing_files(tmp_path, monkeypatch):
    prepared = {
        "special_tokens_map.json": b"special",
        "tokenizer.json": b"tokenizer",
        "tokenizer_config.json": b"config",
    }
    expected = _metadata(prepared)
    existing = tmp_path / "special_tokens_map.json"
    existing.write_bytes(prepared[existing.name])
    existing_inode = existing.stat().st_ino
    source = tmp_path / "source"
    source.mkdir()

    monkeypatch.setattr(preparation, "EXPECTED_PREPARED_TOKENIZER_FILES", expected)
    monkeypatch.setattr(preparation, "verify_archived_checkpoint_files", lambda _path: None)
    monkeypatch.setattr(preparation, "_copy_upstream_files", lambda _source, _output: None)
    monkeypatch.setattr(
        preparation,
        "_verify_file_set",
        lambda _root, _expected, *, description: None,
    )

    def reconstruct(_source: Path, output: Path) -> None:
        _write_files(output, prepared)

    monkeypatch.setattr(preparation, "_reconstruct_tokenizer", reconstruct)
    monkeypatch.setattr(
        preparation,
        "verify_prepared_tokenizer_files",
        lambda root: preparation._verify_file_set(
            root,
            expected,
            description="prepared tokenizer",
        ),
    )

    full_verifications: list[Path] = []

    def verify_full(root: Path) -> None:
        preparation._verify_file_set(root, expected, description="checkpoint")
        full_verifications.append(root)

    monkeypatch.setattr(preparation, "verify_checkpoint_files", verify_full)
    replaced: list[str] = []
    real_replace = os.replace

    def record_replace(source_path: Path, destination_path: Path) -> None:
        replaced.append(Path(destination_path).name)
        real_replace(source_path, destination_path)

    monkeypatch.setattr(preparation.os, "replace", record_replace)

    result = preparation.prepare_tokenizer(tmp_path, source_dir=source)

    assert result["status"] == "prepared"
    assert set(replaced) == {"tokenizer.json", "tokenizer_config.json"}
    assert existing.stat().st_ino == existing_inode
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == prepared
    assert full_verifications == [tmp_path.resolve()]


def test_prepare_tokenizer_cli_dispatches_and_prints_json(monkeypatch, capsys):
    calls = []
    payload = {
        "checkpoint": "/tmp/checkpoint",
        "repository": preparation.UPSTREAM_REPOSITORY,
        "revision": preparation.UPSTREAM_REVISION,
        "status": "prepared",
    }

    def fake_prepare(model, *, source_dir=None):
        calls.append((model, source_dir))
        return payload

    from metc import cli

    monkeypatch.setattr(cli, "prepare_tokenizer", fake_prepare)

    assert (
        cli.main(
            [
                "prepare-tokenizer",
                "--model",
                "/tmp/checkpoint",
                "--source-dir",
                "/tmp/upstream",
            ]
        )
        == 0
    )
    assert calls == [("/tmp/checkpoint", "/tmp/upstream")]
    assert json.loads(capsys.readouterr().out) == payload
