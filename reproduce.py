#!/usr/bin/env python3
"""Artifact safety helpers and the execution-first METC reproduction entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Sequence
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class ReproductionError(RuntimeError):
    """Raised when a verification stage fails."""


@dataclass(frozen=True)
class ArtifactReference:
    filename: str
    root_directory: str
    size_bytes: int
    sha256: str
    download_url: str | None
    members: frozenset[str] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReproductionError(f"cannot read artifact metadata {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ReproductionError(f"artifact metadata {path.name} must be a JSON object")
    return value


def _require_plain_filename(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ReproductionError(f"{field} must be a plain filename")
    if "\x00" in value or "\\" in value:
        raise ReproductionError(f"{field} contains an unsafe filename")
    return value


def _require_root_directory(value: object, *, field: str) -> str:
    root = _require_plain_filename(value, field=field)
    if root in {".", ".."}:
        raise ReproductionError(f"{field} must be a normal directory name")
    return root


def _require_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReproductionError(f"{field} must be a lowercase SHA-256")
    return value


def _require_size(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ReproductionError(f"{field} must be a positive integer")
    return value


def _require_initial_https_url(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ReproductionError(f"{field} must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ReproductionError(f"{field} must be a credential-free HTTPS URL")
    return value


def load_checkpoint_reference(
    path: Path = ROOT / "model" / "checkpoint.json",
) -> ArtifactReference:
    value = _read_json_object(path)
    if value.get("status") != "available":
        raise ReproductionError("checkpoint is not marked available")
    archive = value.get("archive")
    if not isinstance(archive, dict):
        raise ReproductionError("checkpoint archive metadata must be an object")
    documents = archive.get("documents")
    model_files = archive.get("model_files")
    if not isinstance(documents, list) or not all(isinstance(item, str) for item in documents):
        raise ReproductionError("checkpoint archive.documents must be a list of strings")
    if not isinstance(model_files, list) or not all(isinstance(item, str) for item in model_files):
        raise ReproductionError("checkpoint archive.model_files must be a list of strings")
    root = _require_root_directory(
        archive.get("root_directory"), field="checkpoint.archive.root_directory"
    )
    relative_members = [
        _require_plain_filename(item, field="checkpoint archive member")
        for item in [*documents, *model_files]
    ]
    if len(relative_members) != len(set(relative_members)):
        raise ReproductionError("checkpoint archive member list contains duplicates")
    return ArtifactReference(
        filename=_require_plain_filename(
            archive.get("filename"), field="checkpoint.archive.filename"
        ),
        root_directory=root,
        size_bytes=_require_size(archive.get("size_bytes"), field="checkpoint.archive.size_bytes"),
        sha256=_require_digest(archive.get("sha256"), field="checkpoint.archive.sha256"),
        download_url=_require_initial_https_url(
            value.get("download_url"), field="checkpoint.download_url"
        ),
        members=frozenset(f"{root}/{member}" for member in relative_members),
    )


def verify_artifact_file(path: Path, reference: ArtifactReference) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ReproductionError(f"missing {reference.filename}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReproductionError(f"artifact must be a regular file: {path}")
    if metadata.st_size != reference.size_bytes:
        raise ReproductionError(
            f"{reference.filename} has {metadata.st_size} bytes; expected {reference.size_bytes}"
        )
    observed = sha256_file(path)
    if observed != reference.sha256:
        raise ReproductionError(
            f"{reference.filename} has SHA-256 {observed}; expected {reference.sha256}"
        )
    return path


class _HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        parsed = urlsplit(new_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise urllib.error.HTTPError(
                request.full_url, code, "unsafe artifact redirect", headers, file_pointer
            )
        return super().redirect_request(request, file_pointer, code, message, headers, new_url)


def _open_https(url: str, *, timeout: float) -> BinaryIO:
    opener = urllib.request.build_opener(_HTTPSRedirectHandler())
    request = urllib.request.Request(url, headers={"User-Agent": "metc-reproduce/1"})
    return opener.open(request, timeout=timeout)


def obtain_download(
    reference: ArtifactReference,
    cache_dir: Path,
    *,
    no_network: bool,
    timeout: float = 60.0,
) -> Path:
    destination = cache_dir / "downloads" / f"{reference.sha256[:16]}-{reference.filename}"
    if destination.exists():
        try:
            verified = verify_artifact_file(destination, reference)
            print(f"[artifact] using verified cache: {reference.filename}", flush=True)
            return verified
        except ReproductionError:
            if no_network:
                raise ReproductionError(
                    f"cached {reference.filename} is invalid and --no-network was supplied"
                )
    elif no_network:
        raise ReproductionError(f"{reference.filename} is not cached and --no-network was supplied")
    if reference.download_url is None:
        raise ReproductionError(f"{reference.filename} has no download URL")

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{reference.filename}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    written = 0
    next_progress = 64 * 1024 * 1024
    print(
        f"[artifact] downloading {reference.filename} "
        f"({reference.size_bytes / (1024 * 1024):.1f} MiB)",
        flush=True,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            try:
                response = _open_https(reference.download_url, timeout=timeout)
                with response:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > reference.size_bytes:
                            raise ReproductionError(
                                f"downloaded {reference.filename} exceeds its declared size"
                            )
                        digest.update(block)
                        output.write(block)
                        if written >= next_progress:
                            percent = 100 * written / reference.size_bytes
                            print(
                                f"[artifact] {reference.filename}: {percent:.0f}%",
                                flush=True,
                            )
                            next_progress += 64 * 1024 * 1024
            except ReproductionError:
                raise
            except (OSError, urllib.error.URLError) as error:
                raise ReproductionError(f"download failed for {reference.filename}: {error}") from error
            output.flush()
            os.fsync(output.fileno())
        if written != reference.size_bytes or digest.hexdigest() != reference.sha256:
            raise ReproductionError(
                f"downloaded {reference.filename} failed size or SHA-256 verification"
            )
        os.replace(temporary, destination)
        verified = verify_artifact_file(destination, reference)
        print(f"[artifact] verified: {reference.filename}", flush=True)
        return verified
    finally:
        temporary.unlink(missing_ok=True)


def _validate_member_name(name: str, root_directory: str) -> None:
    if not name or "\x00" in name or "\\" in name:
        raise ReproductionError(f"unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or path.parts[0] != root_directory
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReproductionError(f"ZIP member is outside {root_directory!r}: {name!r}")


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    reference: ArtifactReference,
) -> Path:
    try:
        archive = zipfile.ZipFile(archive_path, mode="r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ReproductionError(f"invalid ZIP archive {reference.filename}: {error}") from error
    with archive:
        if archive.comment:
            raise ReproductionError(f"{reference.filename} contains an archive comment")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != sorted(names):
            raise ReproductionError(f"{reference.filename} members are not sorted")
        if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise ReproductionError(f"{reference.filename} contains duplicate or colliding members")
        if reference.members is not None and set(names) != reference.members:
            raise ReproductionError(f"{reference.filename} does not contain its exact declared inventory")

        maximum_uncompressed = max(reference.size_bytes * 4, 64 * 1024 * 1024)
        declared_total = 0
        for info in infos:
            _validate_member_name(info.filename, reference.root_directory)
            mode = info.external_attr >> 16
            declared_total += info.file_size
            if (
                info.is_dir()
                or info.flag_bits & 0x1
                or info.date_time != FIXED_ZIP_TIMESTAMP
                or info.create_system != 3
                or info.extra
                or info.comment
                or info.compress_type != zipfile.ZIP_DEFLATED
                or not stat.S_ISREG(mode)
                or stat.S_ISLNK(mode)
            ):
                raise ReproductionError(f"unsafe or non-deterministic ZIP member: {info.filename}")
        if declared_total > maximum_uncompressed:
            raise ReproductionError(f"{reference.filename} exceeds the extraction size limit")

        destination.mkdir(parents=True, exist_ok=False)
        observed_total = 0
        for info in infos:
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            member_size = 0
            try:
                with archive.open(info, mode="r") as source, target.open("xb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        member_size += len(block)
                        observed_total += len(block)
                        if observed_total > maximum_uncompressed:
                            raise ReproductionError(
                                f"{reference.filename} exceeded its extraction size limit"
                            )
                        output.write(block)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise ReproductionError(f"cannot extract {info.filename}: {error}") from error
            if member_size != info.file_size:
                raise ReproductionError(f"extracted size differs for {info.filename}")
            target.chmod(0o644)
    return destination / reference.root_directory


def main(argv: Sequence[str] | None = None) -> int:
    """Route the canonical reproduction command to execution-first E2E verification."""

    from hydrate_and_reproduce import main as execution_first_main

    return execution_first_main(argv)


if __name__ == "__main__":
    from metc_artifact.hydration import HydrationError

    try:
        raise SystemExit(main())
    except (ReproductionError, HydrationError) as error:
        raise SystemExit(f"METC REPRODUCTION FAILED: {error}") from error
