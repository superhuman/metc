"""Prepare the exact METC tokenizer from a pinned upstream mmBERT snapshot."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .checkpoint import (
    EXPECTED_PREPARED_TOKENIZER_FILES,
    verify_archived_checkpoint_files,
    verify_checkpoint_files,
    verify_prepared_tokenizer_files,
)
from .errors import CheckpointContractError


UPSTREAM_REPOSITORY = "jhu-clsp/mmBERT-small"
UPSTREAM_REVISION = "abc32620dd4f6ab06f5fbe905dc25f310618e09f"
UPSTREAM_FILES = {
    "config.json": (
        1_187,
        "3b0eea47c4734682370b66dbd73d655f1100387f683d7c69015f33aa1bac1be9",
    ),
    "special_tokens_map.json": (
        636,
        "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
    ),
    "tokenizer.json": (
        17_525_329,
        "197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356",
    ),
    "tokenizer_config.json": (
        46_440,
        "ea349495764b24570b4560fe2e4a557e5c0a47e6bce782c2409b1d6445cf8633",
    ),
}
EXPECTED_TRANSFORMERS_VERSION = "4.57.3"
EXPECTED_TOKENIZERS_VERSION = "0.22.0"
EDIT_OPEN = "[EDIT]"
EDIT_CLOSE = "[/EDIT]"
EDIT_OPEN_TOKEN_ID = 256000
EDIT_CLOSE_TOKEN_ID = 256001


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise CheckpointContractError(f"{description} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise CheckpointContractError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise CheckpointContractError(f"{description} must be a regular file: {path}")
    return metadata


def _require_directory(path: Path, *, description: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise CheckpointContractError(f"{description} does not exist: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise CheckpointContractError(f"{description} must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise CheckpointContractError(f"{description} must be a directory: {path}")


def _verify_file_set(
    root: Path,
    expected: dict[str, tuple[int, str]],
    *,
    description: str,
) -> None:
    for relative, (expected_size, expected_digest) in expected.items():
        path = root / relative
        metadata = _require_regular_file(path, description=f"{description} file {relative}")
        if metadata.st_size != expected_size:
            raise CheckpointContractError(
                f"{description} file {relative} has {metadata.st_size} bytes; "
                f"expected {expected_size}"
            )
        observed_digest = _sha256_file(path)
        if observed_digest != expected_digest:
            raise CheckpointContractError(
                f"{description} file {relative} has SHA-256 {observed_digest}; "
                f"expected {expected_digest}"
            )


def _download_upstream_files(destination: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - installation failure
        raise CheckpointContractError(
            "huggingface-hub is required to download the pinned tokenizer"
        ) from error

    for relative in sorted(UPSTREAM_FILES):
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=UPSTREAM_REPOSITORY,
                    revision=UPSTREAM_REVISION,
                    filename=relative,
                )
            )
        except Exception as error:
            raise CheckpointContractError(
                f"could not download {relative} from "
                f"{UPSTREAM_REPOSITORY}@{UPSTREAM_REVISION}: {error}"
            ) from error
        try:
            source = cached.stat()
        except FileNotFoundError as error:
            raise CheckpointContractError(
                f"downloaded upstream tokenizer file is missing: {cached}"
            ) from error
        if not stat.S_ISREG(source.st_mode):
            raise CheckpointContractError(
                f"downloaded upstream tokenizer path is not a regular file: {cached}"
            )
        output = destination / relative
        with cached.open("rb") as input_stream, output.open("xb") as output_stream:
            while block := input_stream.read(1024 * 1024):
                output_stream.write(block)
        os.chmod(output, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        if output.stat().st_size != source.st_size:
            raise CheckpointContractError(f"downloaded file changed while copying: {relative}")


def _copy_upstream_files(source_dir: Path, destination: Path) -> None:
    _require_directory(source_dir, description="upstream tokenizer source directory")
    for relative in sorted(UPSTREAM_FILES):
        source = source_dir / relative
        _require_regular_file(source, description=f"upstream tokenizer file {relative}")
        output = destination / relative
        with source.open("rb") as input_stream, output.open("xb") as output_stream:
            while block := input_stream.read(1024 * 1024):
                output_stream.write(block)
        os.chmod(output, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def _reconstruct_tokenizer(source_dir: Path, output_dir: Path) -> None:
    try:
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - installation failure
        raise CheckpointContractError(
            "transformers and tokenizers are required to prepare the tokenizer"
        ) from error

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise CheckpointContractError(
            "tokenizer preparation requires transformers "
            f"{EXPECTED_TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )
    if tokenizers.__version__ != EXPECTED_TOKENIZERS_VERSION:
        raise CheckpointContractError(
            "tokenizer preparation requires tokenizers "
            f"{EXPECTED_TOKENIZERS_VERSION}, found {tokenizers.__version__}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(source_dir), local_files_only=True)
    additions = tokenizer.add_special_tokens({"additional_special_tokens": [EDIT_OPEN, EDIT_CLOSE]})
    if additions != 2:
        raise CheckpointContractError(
            f"tokenizer preparation added {additions} vocabulary entries; expected 2"
        )
    observed_ids = (
        tokenizer.convert_tokens_to_ids(EDIT_OPEN),
        tokenizer.convert_tokens_to_ids(EDIT_CLOSE),
    )
    expected_ids = (EDIT_OPEN_TOKEN_ID, EDIT_CLOSE_TOKEN_ID)
    if observed_ids != expected_ids:
        raise CheckpointContractError(
            f"prepared edit marker IDs are {observed_ids}; expected {expected_ids}"
        )

    expected_saved = set(EXPECTED_PREPARED_TOKENIZER_FILES)
    first_pass_dir = output_dir.parent / "serialized-once"
    first_pass_dir.mkdir()
    first_saved = {Path(path).name for path in tokenizer.save_pretrained(first_pass_dir)}
    if first_saved != expected_saved:
        raise CheckpointContractError(
            "tokenizer preparation emitted an unexpected file inventory; "
            f"expected={sorted(expected_saved)}, observed={sorted(first_saved)}"
        )

    # The evaluated export loaded and saved the trained tokenizer once more.
    # Repeating that serialization step reproduces all three files byte-for-byte.
    exported_tokenizer = AutoTokenizer.from_pretrained(str(first_pass_dir), local_files_only=True)
    final_saved = {Path(path).name for path in exported_tokenizer.save_pretrained(output_dir)}
    if final_saved != expected_saved:
        raise CheckpointContractError(
            "tokenizer preparation emitted an unexpected final file inventory; "
            f"expected={sorted(expected_saved)}, observed={sorted(final_saved)}"
        )
    for relative in expected_saved:
        os.chmod(
            output_dir / relative,
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
        )


def _existing_tokenizer_state(model_path: Path) -> str:
    existing: list[str] = []
    for relative, (expected_size, expected_digest) in EXPECTED_PREPARED_TOKENIZER_FILES.items():
        path = model_path / relative
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise CheckpointContractError(
                f"existing tokenizer path must be a regular file, not replaced automatically: {path}"
            )
        observed_digest = _sha256_file(path)
        if metadata.st_size != expected_size or observed_digest != expected_digest:
            raise CheckpointContractError(
                f"existing tokenizer file does not match the evaluated checkpoint: {path}; "
                "remove all three tokenizer files only after reviewing the unexpected bytes"
            )
        existing.append(relative)
    if len(existing) == len(EXPECTED_PREPARED_TOKENIZER_FILES):
        return "already_prepared"
    return "partial" if existing else "missing"


def prepare_tokenizer(
    model_path: str | Path,
    *,
    source_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare and verify the exact tokenizer beside the released model weights."""

    destination = Path(model_path).expanduser().resolve()
    _require_directory(destination, description="checkpoint directory")
    verify_archived_checkpoint_files(destination)
    state = _existing_tokenizer_state(destination)
    if state == "already_prepared":
        verify_checkpoint_files(destination)
        return {
            "checkpoint": str(destination),
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "status": "already_prepared",
        }

    resolved_source = Path(source_dir).expanduser().resolve() if source_dir is not None else None
    with tempfile.TemporaryDirectory(
        prefix=".metc-tokenizer-prepare-",
        dir=destination.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        upstream = temporary_root / "upstream"
        generated = temporary_root / "generated"
        upstream.mkdir()
        generated.mkdir()
        if resolved_source is None:
            _download_upstream_files(upstream)
        else:
            _copy_upstream_files(resolved_source, upstream)
        _verify_file_set(upstream, UPSTREAM_FILES, description="upstream tokenizer")
        _reconstruct_tokenizer(upstream, generated)
        verify_prepared_tokenizer_files(generated)

        for relative in sorted(EXPECTED_PREPARED_TOKENIZER_FILES):
            destination_path = destination / relative
            if destination_path.exists():
                continue
            os.replace(generated / relative, destination_path)

    verify_checkpoint_files(destination)
    return {
        "checkpoint": str(destination),
        "repository": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_REVISION,
        "status": "prepared",
    }
