"""Tests for the one-command reproduction entrypoint."""

from __future__ import annotations

import hashlib
import io
import stat
import sys
import zipfile
from pathlib import Path

import pytest

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

import reproduce  # noqa: E402


def _reference(
    payload: bytes,
    *,
    filename: str = "artifact.zip",
    root: str = "artifact",
    members: frozenset[str] | None = None,
) -> reproduce.ArtifactReference:
    return reproduce.ArtifactReference(
        filename=filename,
        root_directory=root,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        download_url="https://example.test/artifact.zip",
        members=members,
    )


def _write_zip(
    path: Path,
    members: list[tuple[str, bytes, int | None]],
) -> bytes:
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, payload, mode in members:
            info = zipfile.ZipInfo(name, date_time=reproduce.FIXED_ZIP_TIMESTAMP)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (mode if mode is not None else stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
    return path.read_bytes()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/a.zip",
        "https://user" + chr(64) + "example.test/a.zip",
        "https://example.test/a.zip?token=secret",
        "https://example.test/a.zip#fragment",
    ],
)
def test_initial_artifact_url_must_be_clean_https(url: str):
    with pytest.raises(reproduce.ReproductionError, match="credential-free HTTPS"):
        reproduce._require_initial_https_url(url, field="test.url")


def test_reproduce_exposes_no_legacy_statistics_replay() -> None:
    for name in (
        "legacy_statistics_review",
        "run_review",
        "compare_report_directories",
        "load_results_reference",
    ):
        assert not hasattr(reproduce, name)


def test_download_reuses_verified_cache_without_network(tmp_path: Path, monkeypatch):
    payload = b"verified artifact"
    reference = _reference(payload)
    cached = tmp_path / "downloads" / f"{reference.sha256[:16]}-{reference.filename}"
    cached.parent.mkdir()
    cached.write_bytes(payload)

    def fail_open(*args, **kwargs):
        raise AssertionError("network must not be used")

    monkeypatch.setattr(reproduce, "_open_https", fail_open)
    assert reproduce.obtain_download(reference, tmp_path, no_network=True) == cached


def test_no_network_rejects_corrupt_cache(tmp_path: Path):
    reference = _reference(b"expected")
    cached = tmp_path / "downloads" / f"{reference.sha256[:16]}-{reference.filename}"
    cached.parent.mkdir()
    cached.write_bytes(b"corrupt!")
    with pytest.raises(reproduce.ReproductionError, match="cached artifact.zip is invalid"):
        reproduce.obtain_download(reference, tmp_path, no_network=True)


def test_download_is_verified_before_replacing_cache(tmp_path: Path, monkeypatch):
    payload = b"new verified bytes"
    reference = _reference(payload)
    cached = tmp_path / "downloads" / f"{reference.sha256[:16]}-{reference.filename}"
    cached.parent.mkdir()
    cached.write_bytes(b"old invalid bytes!")
    monkeypatch.setattr(
        reproduce,
        "_open_https",
        lambda url, timeout: io.BytesIO(payload),
    )
    assert reproduce.obtain_download(reference, tmp_path, no_network=False).read_bytes() == payload
    assert not list(cached.parent.glob("*.part"))


def test_failed_download_does_not_replace_existing_cache(tmp_path: Path, monkeypatch):
    reference = _reference(b"expected bytes")
    cached = tmp_path / "downloads" / f"{reference.sha256[:16]}-{reference.filename}"
    cached.parent.mkdir()
    cached.write_bytes(b"existing bad")
    monkeypatch.setattr(
        reproduce,
        "_open_https",
        lambda url, timeout: io.BytesIO(b"wrong payload"),
    )
    with pytest.raises(reproduce.ReproductionError, match="failed size or SHA-256"):
        reproduce.obtain_download(reference, tmp_path, no_network=False)
    assert cached.read_bytes() == b"existing bad"


def test_safe_extract_accepts_exact_regular_inventory(tmp_path: Path):
    archive_path = tmp_path / "good.zip"
    payload = _write_zip(
        archive_path,
        [
            ("artifact/a.txt", b"a", None),
            ("artifact/b.txt", b"b", None),
        ],
    )
    reference = _reference(
        payload,
        members=frozenset({"artifact/a.txt", "artifact/b.txt"}),
    )
    root = reproduce.safe_extract_zip(archive_path, tmp_path / "out", reference)
    assert (root / "a.txt").read_bytes() == b"a"
    assert (root / "b.txt").read_bytes() == b"b"


@pytest.mark.parametrize(
    "members, message",
    [
        ([("artifact/../escape.txt", b"x", None)], "outside"),
        (
            [
                ("artifact/A.txt", b"a", None),
                ("artifact/a.txt", b"b", None),
            ],
            "duplicate or colliding",
        ),
        ([("wrong/a.txt", b"x", None)], "outside"),
        ([("artifact/link", b"x", stat.S_IFLNK | 0o777)], "unsafe or non-deterministic"),
    ],
)
def test_safe_extract_rejects_unsafe_members(tmp_path: Path, members, message: str):
    archive_path = tmp_path / "bad.zip"
    payload = _write_zip(archive_path, members)
    reference = _reference(payload)
    with pytest.raises(reproduce.ReproductionError, match=message):
        reproduce.safe_extract_zip(archive_path, tmp_path / "out", reference)


def test_safe_extract_rejects_inventory_mismatch(tmp_path: Path):
    archive_path = tmp_path / "wrong-inventory.zip"
    payload = _write_zip(archive_path, [("artifact/a.txt", b"a", None)])
    reference = _reference(payload, members=frozenset({"artifact/b.txt"}))
    with pytest.raises(reproduce.ReproductionError, match="exact declared inventory"):
        reproduce.safe_extract_zip(archive_path, tmp_path / "out", reference)

