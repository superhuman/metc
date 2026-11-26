import hashlib
import json
from pathlib import Path

import pytest

import metc.checkpoint as checkpoint
import metc.tokenizer_preparation as tokenizer_preparation
from metc.errors import CheckpointContractError


def test_public_manifest_matches_runtime_checkpoint_contract():
    manifest_path = Path(__file__).resolve().parents[1] / "model" / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = {
        entry["path"]: (entry["size_bytes"], entry["sha256"]) for entry in manifest["files"]
    }
    assert observed == checkpoint.EXPECTED_CHECKPOINT_FILES
    assert manifest["parameter_count"] == checkpoint.EXPECTED_PARAMETER_COUNT
    assert manifest["distribution_terms"] == {
        "status": "approved",
        "required_documents": [
            "LICENSE",
            "MMBERT_NOTICE.md",
            "THIRD_PARTY_NOTICES.md",
        ],
    }
    assert manifest["archive"]["documents"] == [
        "LICENSE",
        "MMBERT_NOTICE.md",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
    ]


def test_archive_and_prepared_tokenizer_partition_the_runtime_checkpoint():
    manifest_path = Path(__file__).resolve().parents[1] / "model" / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archived = set(manifest["archive"]["model_files"])
    prepared = set(manifest["tokenizer_preparation"]["output_files"])

    assert len(archived) == 4
    assert len(prepared) == 3
    assert archived == set(checkpoint.EXPECTED_ARCHIVED_CHECKPOINT_FILES)
    assert prepared == set(checkpoint.EXPECTED_PREPARED_TOKENIZER_FILES)
    assert archived.isdisjoint(prepared)
    assert archived | prepared == set(checkpoint.EXPECTED_CHECKPOINT_FILES)
    assert manifest["schema_version"] == 4


def test_public_manifest_matches_tokenizer_preparation_contract():
    manifest_path = Path(__file__).resolve().parents[1] / "model" / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preparation = manifest["tokenizer_preparation"]
    source_files = {
        entry["path"]: (entry["size_bytes"], entry["sha256"])
        for entry in preparation["source_files"]
    }

    assert preparation["repository"] == tokenizer_preparation.UPSTREAM_REPOSITORY
    assert preparation["revision"] == tokenizer_preparation.UPSTREAM_REVISION
    assert source_files == tokenizer_preparation.UPSTREAM_FILES
    assert preparation["toolchain"] == {
        "tokenizers_version": tokenizer_preparation.EXPECTED_TOKENIZERS_VERSION,
        "transformers_version": tokenizer_preparation.EXPECTED_TRANSFORMERS_VERSION,
    }
    assert manifest["edit_open_token_id"] == tokenizer_preparation.EDIT_OPEN_TOKEN_ID
    assert manifest["edit_close_token_id"] == tokenizer_preparation.EDIT_CLOSE_TOKEN_ID


def test_checkpoint_file_verification_rejects_byte_drift(tmp_path, monkeypatch):
    expected = b"exact checkpoint bytes"
    monkeypatch.setattr(
        checkpoint,
        "EXPECTED_CHECKPOINT_FILES",
        {
            "tiny.bin": (
                len(expected),
                hashlib.sha256(expected).hexdigest(),
            )
        },
    )
    path = tmp_path / "tiny.bin"
    path.write_bytes(expected)
    checkpoint.verify_checkpoint_files(tmp_path)

    path.write_bytes(b"exact checkpoint bytez")
    with pytest.raises(CheckpointContractError, match="SHA-256"):
        checkpoint.verify_checkpoint_files(tmp_path)


def test_checkpoint_file_verification_rejects_symlinks(tmp_path, monkeypatch):
    expected = b"exact checkpoint bytes"
    target = tmp_path / "target.bin"
    target.write_bytes(expected)
    (tmp_path / "tiny.bin").symlink_to(target)
    monkeypatch.setattr(
        checkpoint,
        "EXPECTED_CHECKPOINT_FILES",
        {
            "tiny.bin": (
                len(expected),
                hashlib.sha256(expected).hexdigest(),
            )
        },
    )

    with pytest.raises(CheckpointContractError, match="must be a regular file"):
        checkpoint.verify_checkpoint_files(tmp_path)
