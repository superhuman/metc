from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from metc_artifact.datasets import AcquisitionError
from metc_artifact.datasets.acquire import _acquire_git, _acquire_http, _extract_zip
from metc_artifact.datasets.catalog import (
    RECIPES,
    SOURCES,
    GitArtifact,
    HttpArtifact,
    LicenseTerms,
)


def test_catalog_covers_every_paper_setting_with_immutable_pins() -> None:
    assert tuple(RECIPES) == (
        "en_conll14_ann0",
        "en_conll14_ann1",
        "de_merlin",
        "cs_merlin",
        "ko_kolla",
        "uk_uagec",
        "zh_mucgec",
    )
    assert SOURCES["conll14_test"].sha256 == (
        "d45784ef928b47521d2fb30f4fa24e59d316357f1567eae922ce02560ba6cf04"
    )
    assert SOURCES["ua_gec"].commit == "4757f72f192c4a41e4c8fb1d9690a948f87cf6d6"
    assert SOURCES["mucgec"].commit == "39221934ee4ccf07801879136a75dd7e11319977"
    assert RECIPES["cs_merlin"].expected_items == 6_590
    assert RECIPES["zh_mucgec"].expected_gold_edits == 4_191


def _direct_source(payload: bytes) -> HttpArtifact:
    return HttpArtifact(
        source_id="fixture",
        url="https://example.test/fixture.txt",
        filename="fixture.txt",
        sha256=hashlib.sha256(payload).hexdigest(),
        archive_kind="file",
        official_page="https://example.test",
        version="test",
        license=LicenseTerms(
            "test", "https://example.test/terms", None, "official-download-only", "test"
        ),
    )


def _git_source() -> GitArtifact:
    return GitArtifact(
        source_id="git-fixture",
        url="https://example.test/fixture.git",
        commit="0" * 40,
        official_page="https://example.test",
        version="test",
        license=LicenseTerms(
            "test", "https://example.test/terms", None, "official-download-only", "test"
        ),
    )


def test_http_acquisition_verifies_before_caching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"pinned public bytes"
    monkeypatch.setattr(
        "metc_artifact.datasets.acquire.urllib.request.urlopen",
        lambda _request: io.BytesIO(payload),
    )
    path = _acquire_http(_direct_source(payload), tmp_path)
    assert path.read_bytes() == payload
    assert path.name.startswith(hashlib.sha256(payload).hexdigest()[:16])


def test_http_acquisition_rejects_wrong_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "metc_artifact.datasets.acquire.urllib.request.urlopen",
        lambda _request: io.BytesIO(b"wrong"),
    )
    with pytest.raises(AcquisitionError, match="checksum mismatch"):
        _acquire_http(_direct_source(b"expected"), tmp_path)
    assert not list((tmp_path / "downloads").glob("*.txt"))


def test_no_network_rejects_uncached_http_source(tmp_path: Path) -> None:
    with pytest.raises(AcquisitionError, match="not cached.*no-network"):
        _acquire_http(_direct_source(b"expected"), tmp_path, no_network=True)


def test_no_network_rejects_uncached_git_source(tmp_path: Path) -> None:
    with pytest.raises(AcquisitionError, match="not cached.*no-network"):
        _acquire_git(_git_source(), tmp_path, no_network=True)


def test_zip_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(AcquisitionError, match="unsafe member"):
        _extract_zip(archive, tmp_path / "out")
