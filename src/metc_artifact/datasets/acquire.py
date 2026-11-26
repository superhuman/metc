"""Checksum-pinned acquisition of public benchmark sources."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from .catalog import GitArtifact, HttpArtifact, SourceArtifact, source_artifact


class AcquisitionError(RuntimeError):
    """A public source could not be acquired or did not match its pin."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise AcquisitionError(f"checksum mismatch for {path}: expected {expected}, got {actual}")


def _safe_member(name: str) -> PurePosixPath:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise AcquisitionError(f"archive contains unsafe member: {name!r}")
    return normalized


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            _safe_member(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise AcquisitionError(f"archive contains non-regular member: {member.name!r}")
        bundle.extractall(destination, filter="data")


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            _safe_member(member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise AcquisitionError(f"archive contains symlink: {member.filename!r}")
        bundle.extractall(destination)


def _download(source: HttpArtifact, cache_dir: Path, *, no_network: bool = False) -> Path:
    downloads = cache_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    destination = downloads / f"{source.sha256[:16]}-{source.filename}"
    if destination.exists():
        _check_sha256(destination, source.sha256)
        return destination
    if no_network:
        raise AcquisitionError(
            f"{source.source_id} is not cached and no-network mode was requested"
        )

    descriptor, temporary_name = tempfile.mkstemp(prefix="download-", dir=downloads)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            source.url,
            headers={"User-Agent": "metc-artifact-public-source-rebuilder/1"},
        )
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        _check_sha256(temporary, source.sha256)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _acquire_http(
    source: HttpArtifact,
    cache_dir: Path,
    *,
    no_network: bool = False,
) -> Path:
    archive = _download(source, cache_dir, no_network=no_network)
    if source.archive_kind == "file":
        return archive

    destination = cache_dir / "sources" / source.source_id
    marker = destination / ".metc-source.json"
    if destination.exists():
        if marker.exists():
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            if metadata.get("sha256") == source.sha256:
                return destination
        raise AcquisitionError(f"existing source directory is not the pinned source: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{source.source_id}-", dir=destination.parent))
    try:
        if source.archive_kind == "tar.gz":
            _extract_tar(archive, temporary)
        elif source.archive_kind == "zip":
            _extract_zip(archive, temporary)
        else:  # pragma: no cover - exhaustiveness guard
            raise AcquisitionError(f"unsupported archive kind: {source.archive_kind}")
        (temporary / ".metc-source.json").write_text(
            json.dumps(
                {"source_id": source.source_id, "url": source.url, "sha256": source.sha256},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _run_git(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _acquire_git(
    source: GitArtifact,
    cache_dir: Path,
    *,
    no_network: bool = False,
) -> Path:
    destination = cache_dir / "sources" / source.source_id
    if destination.exists():
        if not (destination / ".git").exists():
            raise AcquisitionError(
                f"existing source directory is not a git checkout: {destination}"
            )
        head = _run_git("rev-parse", "HEAD", cwd=destination)
        dirty = _run_git("status", "--porcelain", "--untracked-files=no", cwd=destination)
        if head != source.commit or dirty:
            raise AcquisitionError(
                f"existing checkout is not clean at {source.commit}: {destination}"
            )
        return destination
    if no_network:
        raise AcquisitionError(
            f"{source.source_id} is not cached and no-network mode was requested"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{source.source_id}-", dir=destination.parent))
    shutil.rmtree(temporary)
    try:
        _run_git("clone", "--no-checkout", "--filter=blob:none", source.url, str(temporary))
        _run_git("checkout", "--detach", source.commit, cwd=temporary)
        head = _run_git("rev-parse", "HEAD", cwd=temporary)
        if head != source.commit:
            raise AcquisitionError(f"git checkout resolved to {head}, expected {source.commit}")
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def acquire_source(
    source: str | SourceArtifact,
    cache_dir: Path,
    *,
    no_network: bool = False,
) -> Path:
    """Acquire one source and return its verified file or checkout directory."""
    artifact = source_artifact(source) if isinstance(source, str) else source
    cache_dir = Path(cache_dir).resolve()
    if isinstance(artifact, HttpArtifact):
        return _acquire_http(artifact, cache_dir, no_network=no_network)
    return _acquire_git(artifact, cache_dir, no_network=no_network)


def acquire_sources(
    source_ids: Iterable[str],
    cache_dir: Path,
    *,
    no_network: bool = False,
) -> dict[str, Path]:
    """Acquire each requested source once, preserving caller order."""
    return {
        source_id: acquire_source(source_id, cache_dir, no_network=no_network)
        for source_id in source_ids
    }
