"""Local-only tests for runtime hydration."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from metc_artifact import hydration


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_packaged_jp_transform_is_checksum_attested() -> None:
    software = Path(__file__).resolve().parents[1]
    transform = software / "runtime-tools" / "transform_jp_errant.py"
    assert hydration.sha256_file(transform) == hydration.JP_TRANSFORM_SHA256
    assert hydration.JP_TRANSFORMATIONS == {
        "parallel_legacy_true": True,
        "zh_explicit_gsdltp_models": True,
        "zh_empty_shape_table_fallback": True,
    }


def test_hydration_environment_does_not_forward_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("GOOGLE_API_TOKEN", "google-secret")
    monkeypatch.setenv("REVIEW_COOKIE", "cookie-secret")
    monkeypatch.setenv("SAFE_SETTING", "retained")

    environment = hydration._clean_environment(
        {
            "INJECTED_PASSWORD": "also-secret",
            "UV_PROJECT_ENVIRONMENT": "/tmp/runtime",
        }
    )

    assert environment["SAFE_SETTING"] == "retained"
    assert environment["UV_PROJECT_ENVIRONMENT"] == "/tmp/runtime"
    assert "OPENAI_API_KEY" not in environment
    assert "GOOGLE_API_TOKEN" not in environment
    assert "REVIEW_COOKIE" not in environment
    assert "INJECTED_PASSWORD" not in environment
    assert os.environ["OPENAI_API_KEY"] == "openai-secret"


def test_jp_transform_rejects_unpinned_source(tmp_path: Path) -> None:
    software = Path(__file__).resolve().parents[1]
    transform = software / "runtime-tools" / "transform_jp_errant.py"
    source = tmp_path / "jp_errant"
    source.mkdir()
    (source / "__init__.py").write_bytes(b"not the pinned upstream source\n")

    completed = subprocess.run(
        (sys.executable, transform, "--source-root", tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "upstream SHA-256" in completed.stderr


def test_stanza_verifier_hashes_files_and_can_select_from_superset(tmp_path: Path) -> None:
    resources = b'{"resources":"pinned"}\n'
    model = b"exact model bytes"
    root = tmp_path / "models"
    (root / "en" / "tokenize").mkdir(parents=True)
    (root / "resources.json").write_bytes(resources)
    (root / "en" / "tokenize" / "combined.pt").write_bytes(model)
    (root / "unrelated.txt").write_bytes(b"extra")
    manifest = {
        "resources_json": {
            "size_bytes": len(resources),
            "sha256": _digest(resources),
        },
        "files": {
            "en/tokenize/combined.pt": [len(model), _digest(model)],
        },
    }

    hydration._verify_stanza(root, manifest, exact_inventory=False)
    with pytest.raises(hydration.HydrationError, match="extra=.*unrelated"):
        hydration._verify_stanza(root, manifest)

    (root / "en" / "tokenize" / "combined.pt").write_bytes(b"exact model bytez")
    with pytest.raises(hydration.HydrationError, match="SHA-256"):
        hydration._verify_stanza(root, manifest, exact_inventory=False)


def test_cold_stanza_hydration_discards_unpinned_default_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = b'{"resources":"pinned"}\n'
    model = b"exact model bytes"
    manifest = {
        "schema_version": "metc-stanza-models-v1",
        "stanza_version": "1.10.1",
        "resources_version": "1.10.0",
        "resources_json": {
            "size_bytes": len(resources),
            "sha256": _digest(resources),
        },
        "files": {
            "en/tokenize/combined.pt": [len(model), _digest(model)],
        },
    }
    software = tmp_path / "software"
    locks = software / "runtime-locks"
    tools = software / "runtime-tools"
    locks.mkdir(parents=True)
    tools.mkdir()
    (locks / "stanza-models.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (tools / "download_stanza.py").write_text("# test downloader\n", encoding="utf-8")

    def fake_download(
        argv: list[str | Path] | tuple[str | Path, ...],
        *,
        cwd: Path,
        environment: dict[str, str] | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, environment, timeout
        output = Path(argv[argv.index("--output") + 1])
        (output / "en" / "tokenize").mkdir(parents=True)
        (output / "resources.json").write_bytes(resources)
        (output / "en" / "tokenize" / "combined.pt").write_bytes(model)
        (output / "en" / "default.zip").write_bytes(b"unused default bundle")
        (output / "en" / "ner").mkdir()
        (output / "en" / "ner" / "ontonotes.pt").write_bytes(b"unused model")
        return subprocess.CompletedProcess([str(value) for value in argv], 0, "", "")

    monkeypatch.setattr(hydration, "_run_checked", fake_download)

    root = hydration.hydrate_stanza(
        software_root=software,
        cache_dir=tmp_path / "cache",
        jp_python=Path(sys.executable),
        no_network=False,
    )

    assert (root / "resources.json").read_bytes() == resources
    assert (root / "en" / "tokenize" / "combined.pt").read_bytes() == model
    assert not (root / "en" / "default.zip").exists()
    assert not (root / "en" / "ner").exists()
    hydration._verify_stanza(root, manifest)


def _write_tiny_gsdltp(path: Path, payloads: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_gsdltp_extracts_only_attested_runtime_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = b"token model"
    pos = b"pos model"
    token_source = "root/charlm/saved_models/tokenize/token.pt"
    pos_source = "root/charlm/saved_models/pos/pos.pt"
    archive = tmp_path / "models.tar.gz"
    _write_tiny_gsdltp(
        archive,
        {
            token_source: token,
            pos_source: pos,
            "root/unused.pt": b"not extracted",
        },
    )
    monkeypatch.setattr(hydration, "GSDLTP_ARCHIVE_SIZE", archive.stat().st_size)
    monkeypatch.setattr(hydration, "GSDLTP_ARCHIVE_SHA256", hydration.sha256_file(archive))
    monkeypatch.setattr(
        hydration,
        "GSDLTP_ARCHIVE_FILES",
        {
            token_source: ("tokenize/token.pt", len(token), _digest(token)),
            pos_source: ("pos/pos.pt", len(pos), _digest(pos)),
        },
    )

    root = hydration.hydrate_gsdltp(
        cache_dir=tmp_path / "cache",
        no_network=True,
        archive=archive,
    )
    assert (root / "tokenize" / "token.pt").read_bytes() == token
    assert (root / "pos" / "pos.pt").read_bytes() == pos
    assert not (root / "unused.pt").exists()


def test_gsdltp_rejects_archive_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "models.tar.gz"
    source_name = "root/tokenize/token.pt"
    with tarfile.open(archive, mode="w:gz") as bundle:
        link = tarfile.TarInfo(source_name)
        link.type = tarfile.SYMTYPE
        link.linkname = "../../escape"
        bundle.addfile(link)
    monkeypatch.setattr(hydration, "GSDLTP_ARCHIVE_SIZE", archive.stat().st_size)
    monkeypatch.setattr(hydration, "GSDLTP_ARCHIVE_SHA256", hydration.sha256_file(archive))
    monkeypatch.setattr(
        hydration,
        "GSDLTP_ARCHIVE_FILES",
        {source_name: ("tokenize/token.pt", 0, _digest(b""))},
    )

    with pytest.raises(hydration.HydrationError, match="unsafe GSDLTP"):
        hydration.hydrate_gsdltp(
            cache_dir=tmp_path / "cache",
            no_network=True,
            archive=archive,
        )


def test_runtime_marker_rejects_lock_drift(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    marker = {
        "schema_version": hydration.SCHEMA_VERSION,
        "runtime": "errant",
        "python_version": hydration.PYTHON_VERSION,
        "lock_sha256": "old",
    }
    (root / ".metc-runtime.json").write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(hydration.HydrationError, match="does not match"):
        hydration._verify_runtime(root, "errant", "new")


def test_jp_source_install_ignores_root_uv_build_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    software = tmp_path / "software"
    project = software / "runtime-locks" / "jp_errant"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='runtime'\n", encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    source = tmp_path / "jp-source"
    source.mkdir()
    commands: list[tuple[str, ...]] = []

    def execute(argv, **kwargs):
        commands.append(tuple(str(value) for value in argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(hydration, "_run_checked", execute)
    monkeypatch.setattr(
        hydration,
        "_verify_runtime",
        lambda root, runtime, lock_digest: hydration._python_executable(root),
    )

    hydration.hydrate_runtime(
        "jp_errant",
        software_root=software,
        cache_dir=tmp_path / "cache",
        uv=tmp_path / "uv",
        no_network=False,
        jp_source=source,
    )

    install = next(command for command in commands if command[1:3] == ("pip", "install"))
    assert "--no-config" in install


def test_runtime_hydration_bootstraps_managed_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    software = tmp_path / "software"
    project = software / "runtime-locks" / "errant"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='runtime'\n", encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def execute(argv, **kwargs):
        calls.append(
            (
                tuple(str(value) for value in argv),
                dict(kwargs.get("environment", {})),
            )
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(hydration, "_run_checked", execute)
    monkeypatch.setattr(
        hydration,
        "_verify_runtime",
        lambda root, runtime, lock_digest: hydration._python_executable(root),
    )

    hydration.hydrate_runtime(
        "errant",
        software_root=software,
        cache_dir=tmp_path / "cache",
        uv=tmp_path / "uv",
        no_network=False,
    )

    install_index = next(
        index for index, (command, _) in enumerate(calls) if command[1:3] == ("python", "install")
    )
    sync_index = next(index for index, (command, _) in enumerate(calls) if command[1] == "sync")
    install, install_environment = calls[install_index]
    sync, sync_environment = calls[sync_index]
    expected_install_dir = str(tmp_path / "cache" / "python")

    assert install_index < sync_index
    assert install[-1] == hydration.PYTHON_VERSION
    assert install[install.index("--install-dir") + 1] == expected_install_dir
    assert install_environment["UV_PYTHON_INSTALL_DIR"] == expected_install_dir
    assert sync[sync.index("--python") + 1] == hydration.PYTHON_VERSION
    assert "--managed-python" in sync
    assert sync_environment["UV_PYTHON_INSTALL_DIR"] == expected_install_dir


def test_selected_hydration_skips_unrequested_producers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def runtime(name: str, **kwargs) -> Path:
        calls.append(name)
        return tmp_path / f"{name}-python"

    monkeypatch.setattr(hydration, "hydrate_runtime", runtime)
    monkeypatch.setattr(
        hydration,
        "hydrate_ltp",
        lambda **kwargs: tmp_path / "ltp",
    )
    monkeypatch.setattr(
        hydration,
        "hydrate_jp_source",
        lambda **kwargs: pytest.fail("ERRANT-only must not hydrate JP-ERRANT"),
    )
    monkeypatch.setattr(
        hydration,
        "hydrate_gerrant_root",
        lambda **kwargs: pytest.fail("ERRANT-only must not hydrate gERRANT"),
    )

    hydrated = hydration.hydrate_selected(
        software_root=Path(__file__).resolve().parents[1],
        cache_dir=tmp_path / "cache",
        uv=sys.executable,
        systems=("errant",),
        accept_ltp_terms=True,
    )

    assert calls == ["errant", "cherrant"]
    assert hydrated.errant_python == tmp_path / "errant-python"
    assert hydrated.gerrant_python is None
    assert hydrated.jp_python is None
    assert hydrated.cherrant_python == tmp_path / "cherrant-python"


def test_ltp_hydration_requires_explicit_terms_acceptance(tmp_path: Path) -> None:
    with pytest.raises(
        hydration.HydrationError,
        match="--accept-ltp-terms",
    ):
        hydration.hydrate_ltp(
            cache_dir=tmp_path / "cache",
            cherrant_python=Path(sys.executable),
            no_network=True,
        )


def test_gerrant_uncached_offline_requires_both_derived_resources(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    (data_root / "resources" / "gerrant").mkdir(parents=True)

    with pytest.raises(
        hydration.HydrationError,
        match="Data artifact provides the exact dictionary but not.*STTS",
    ):
        hydration.hydrate_gerrant_root(
            software_root=Path(__file__).resolve().parents[1],
            cache_dir=tmp_path / "cache",
            gerrant_python=Path(sys.executable),
            no_network=True,
            data_root=data_root,
        )
