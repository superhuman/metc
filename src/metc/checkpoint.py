"""Exact file contract for the checkpoint evaluated in the METC paper."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

from .errors import CheckpointContractError


EXPECTED_PARAMETER_COUNT = 140_658_474
EXPECTED_ARCHIVED_CHECKPOINT_FILES = {
    "config.json": (
        4_135,
        "4685c3834028851896ec38e76c50dd5b98b8041b37ea796284403e929d737d2f",
    ),
    "model-00001-of-00002.safetensors": (
        498_849_256,
        "70068ffd2dedc0d6e6417104740445abb86ccce07becc0dd8cbd78b342647b0f",
    ),
    "model-00002-of-00002.safetensors": (
        63_798_736,
        "1619e3d815a586f0c548a15ff7e337078f0f967fd87b5ba5b0b0f5ff363476e1",
    ),
    "model.safetensors.index.json": (
        10_270,
        "3d38543ab44e32b1d1fefffbb5b887f278b7a6152785a3759c7b83953fb03785",
    ),
}
EXPECTED_PREPARED_TOKENIZER_FILES = {
    "special_tokens_map.json": (
        1_036,
        "d3efb945fcdc4dc561e2f4cf55559d71b27fd7458f466e67c7b9eb76e7769f58",
    ),
    "tokenizer.json": (
        34_363_555,
        "dbe152000b5455328301181b51d9347cf3df2c0a0a5ab23ec8c4c19c4ff2d6b4",
    ),
    "tokenizer_config.json": (
        46_801,
        "1e9e582f8b549d8b4b10e1ee995161aa89dc11685692e4a2b3622f7f773d3dc6",
    ),
}
EXPECTED_CHECKPOINT_FILES = {
    **EXPECTED_ARCHIVED_CHECKPOINT_FILES,
    **EXPECTED_PREPARED_TOKENIZER_FILES,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_inventory_sha256() -> str:
    """Return a stable digest of the expected path, size, and hash triples."""

    inventory: list[dict[str, Any]] = [
        {"path": path, "size_bytes": size, "sha256": digest}
        for path, (size, digest) in sorted(EXPECTED_CHECKPOINT_FILES.items())
    ]
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_files(
    model_path: Path,
    expected_files: dict[str, tuple[int, str]],
    *,
    description: str,
) -> None:
    for relative, (expected_size, expected_digest) in expected_files.items():
        path = model_path / relative
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            raise CheckpointContractError(f"{description} is missing {relative}")
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise CheckpointContractError(f"{description} file {relative} must be a regular file")
        observed_size = metadata.st_size
        if observed_size != expected_size:
            raise CheckpointContractError(
                f"{description} file {relative} has {observed_size} bytes; expected {expected_size}"
            )
        observed_digest = _sha256_file(path)
        if observed_digest != expected_digest:
            raise CheckpointContractError(
                f"{description} file {relative} has SHA-256 {observed_digest}; "
                f"expected {expected_digest}"
            )


def verify_archived_checkpoint_files(model_path: Path) -> None:
    """Verify the four model files distributed in the hosted archive."""

    _verify_files(
        model_path,
        EXPECTED_ARCHIVED_CHECKPOINT_FILES,
        description="checkpoint archive",
    )


def verify_prepared_tokenizer_files(model_path: Path) -> None:
    """Verify the exact tokenizer files prepared from the pinned upstream source."""

    _verify_files(
        model_path,
        EXPECTED_PREPARED_TOKENIZER_FILES,
        description="prepared tokenizer",
    )


def verify_checkpoint_files(model_path: Path) -> None:
    """Hash every file required at inference time and reject byte-level drift."""

    _verify_files(model_path, EXPECTED_CHECKPOINT_FILES, description="checkpoint")
