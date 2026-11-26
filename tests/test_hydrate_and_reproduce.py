"""The one-command wrapper must require a real E2E report artifact."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "hydrate_and_reproduce",
    SOFTWARE_ROOT / "hydrate_and_reproduce.py",
)
assert SPEC is not None and SPEC.loader is not None
hydrate_and_reproduce = importlib.util.module_from_spec(SPEC)
sys.modules["hydrate_and_reproduce"] = hydrate_and_reproduce
SPEC.loader.exec_module(hydrate_and_reproduce)


def _args(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    return SimpleNamespace(
        data_root=data,
        data_archive=None,
        data_sha256=None,
        data_size_bytes=None,
        data_root_directory=None,
        data_uncompressed_size_bytes=None,
        work_root=tmp_path / "work",
        resume=False,
        cache_dir=tmp_path / "cache",
        checkpoint_archive=None,
        checkpoint_dir=tmp_path / "checkpoint",
        tokenizer_source=None,
        metc_inference_profile="paper",
        gerrant_resources=None,
        stanza_resources=None,
        gsdltp_archive=None,
        accept_ltp_terms=True,
        uv="uv",
        no_network=False,
        jp_workers=2,
        systems=("all",),
        llm=False,
        llm_run_id="test",
        llm_workers=1,
        llm_max_attempts=1,
        reference_samples=10,
        seed=42,
        timeout_seconds=60,
        report=None,
    )


def _runtimes(tmp_path: Path):
    return SimpleNamespace(
        errant_python=tmp_path / "errant-python",
        gerrant_python=tmp_path / "gerrant-python",
        gerrant_root=tmp_path / "gerrant-root",
        jp_python=tmp_path / "jp-python",
        jp_transform_manifest=tmp_path / "jp-transform.json",
        jp_stanza_resources=tmp_path / "stanza",
        jp_zh_gsdltp_dir=tmp_path / "gsdltp",
        cherrant_python=tmp_path / "cherrant-python",
        cherrant_hf_home=tmp_path / "ltp",
        to_dict=lambda: {},
    )


def _errant_runtimes(tmp_path: Path):
    return SimpleNamespace(
        errant_python=tmp_path / "errant-python",
        gerrant_python=None,
        gerrant_root=None,
        jp_python=None,
        jp_transform_manifest=None,
        jp_stanza_resources=None,
        jp_zh_gsdltp_dir=None,
        cherrant_python=tmp_path / "cherrant-python",
        cherrant_hf_home=tmp_path / "ltp",
        to_dict=lambda: {
            "errant_python": str(tmp_path / "errant-python"),
            "cherrant_python": str(tmp_path / "cherrant-python"),
            "cherrant_hf_home": str(tmp_path / "ltp"),
        },
    )


def _patch_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        hydrate_and_reproduce,
        "_validate_metc_runtime",
        lambda systems, inference_profile: None,
    )
    monkeypatch.setattr(hydrate_and_reproduce, "_verify_data_root", lambda path: path)
    monkeypatch.setattr(
        hydrate_and_reproduce,
        "_hydrate_checkpoint",
        lambda **kwargs: tmp_path / "checkpoint",
    )
    monkeypatch.setattr(
        hydrate_and_reproduce,
        "hydrate_selected",
        lambda **kwargs: _runtimes(tmp_path),
    )


def _write_data_zip(path: Path, payload: bytes = b"{}") -> bytes:
    with zipfile.ZipFile(path, mode="w") as bundle:
        info = zipfile.ZipInfo(
            "metc-data/manifest.full.json",
            date_time=hydrate_and_reproduce.reproduce.FIXED_ZIP_TIMESTAMP,
        )
        info.create_system = 3
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        bundle.writestr(info, payload)
    return path.read_bytes()


def test_exit_zero_without_report_is_not_a_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        hydrate_and_reproduce.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    with pytest.raises(
        hydrate_and_reproduce.HydrationError,
        match="without producing",
    ):
        hydrate_and_reproduce.run(_args(tmp_path))


def test_known_difference_return_code_keeps_machine_readable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)

    def execute(command, **kwargs):
        report = Path(command[command.index("--report") + 1])
        report.write_text('{"status":"KNOWN_DIFFERENCE"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(hydrate_and_reproduce.subprocess, "run", execute)
    assert hydrate_and_reproduce.run(_args(tmp_path)) == 2
    assert (tmp_path / "work" / "e2e-report.json").is_file()


def test_no_network_is_forwarded_to_execution_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)
    args = _args(tmp_path)
    args.no_network = True

    def execute(command, **kwargs):
        assert command[-1] == "--no-network"
        report = Path(command[command.index("--report") + 1])
        report.write_text('{"status":"KNOWN_DIFFERENCE"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(hydrate_and_reproduce.subprocess, "run", execute)
    assert hydrate_and_reproduce.run(args) == 2


def test_wrapper_defaults_to_no_live_llm_requests() -> None:
    args = hydrate_and_reproduce.build_parser().parse_args([])

    assert args.llm is False
    assert args.resume is False
    assert args.metc_inference_profile == "paper"


def test_dirty_work_root_is_rejected_without_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)
    args = _args(tmp_path)
    args.work_root.mkdir()
    (args.work_root / "unrelated.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(
        hydrate_and_reproduce.HydrationError,
        match="work directory is not empty",
    ):
        hydrate_and_reproduce.run(args)


def test_resume_requires_prior_hydration_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)
    args = _args(tmp_path)
    args.resume = True
    args.work_root.mkdir()
    (args.work_root / "partial.txt").write_text("partial\n", encoding="utf-8")

    with pytest.raises(
        hydrate_and_reproduce.HydrationError,
        match="requires the prior hydration report",
    ):
        hydrate_and_reproduce.run(args)


def test_resume_reuses_exact_run_metadata_and_same_llm_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)
    args = _args(tmp_path)
    commands: list[list[str]] = []

    def execute(command, **kwargs):
        commands.append(command)
        report = Path(command[command.index("--report") + 1])
        report.write_text('{"status":"KNOWN_DIFFERENCE"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(hydrate_and_reproduce.subprocess, "run", execute)
    assert hydrate_and_reproduce.run(args) == 2

    args.resume = True
    assert hydrate_and_reproduce.run(args) == 2
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--llm-run-id") + 1] == "test"


def test_resume_rejects_changed_run_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)
    args = _args(tmp_path)

    def execute(command, **kwargs):
        report = Path(command[command.index("--report") + 1])
        report.write_text('{"status":"KNOWN_DIFFERENCE"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(hydrate_and_reproduce.subprocess, "run", execute)
    assert hydrate_and_reproduce.run(args) == 2

    args.resume = True
    args.llm_run_id = "different"
    with pytest.raises(
        hydrate_and_reproduce.HydrationError,
        match="resume configuration differs.*run",
    ):
        hydrate_and_reproduce.run(args)


def test_errant_only_does_not_hydrate_checkpoint_or_unselected_systems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.systems = ("errant",)
    monkeypatch.setattr(
        hydrate_and_reproduce,
        "_validate_metc_runtime",
        lambda systems, inference_profile: None,
    )
    monkeypatch.setattr(hydrate_and_reproduce, "_verify_data_root", lambda path: path)
    monkeypatch.setattr(
        hydrate_and_reproduce,
        "_hydrate_checkpoint",
        lambda **kwargs: pytest.fail("ERRANT-only must not hydrate the METC checkpoint"),
    )

    def hydrate(**kwargs):
        assert kwargs["systems"] == ("errant",)
        assert kwargs["data_root"] == args.data_root
        return _errant_runtimes(tmp_path)

    monkeypatch.setattr(hydrate_and_reproduce, "hydrate_selected", hydrate)

    def execute(command, **kwargs):
        assert "--errant-python" in command
        assert "--metc-checkpoint" not in command
        assert "--gerrant-python" not in command
        assert "--jp-python" not in command
        report = Path(command[command.index("--report") + 1])
        report.write_text('{"status":"KNOWN_DIFFERENCE"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(hydrate_and_reproduce.subprocess, "run", execute)
    assert hydrate_and_reproduce.run(args) == 2


def test_wrapper_describes_data_artifact_without_schema_version() -> None:
    help_text = hydrate_and_reproduce.build_parser().format_help()

    assert "extracted complete METC Data artifact" in help_text
    assert "metc-data-v2 artifact" not in help_text


def test_wrapper_passes_effective_aliases_and_teacher_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)
    args = _args(tmp_path)
    args.systems = ("gerrant", "jp_errant", "metc_teacher")

    def execute(command, **kwargs):
        start = command.index("--systems") + 1
        end = command.index("--metc-inference-profile")
        assert command[start:end] == ["ger", "jp", "metc", "metc_teacher"]
        report = Path(command[command.index("--report") + 1])
        report.write_text('{"status":"KNOWN_DIFFERENCE"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(hydrate_and_reproduce.subprocess, "run", execute)
    assert hydrate_and_reproduce.run(args) == 2
    hydration = json.loads(
        (tmp_path / "work" / "hydration-report.json").read_text(encoding="utf-8")
    )
    assert hydration["effective_systems"] == ["ger", "jp", "metc", "metc_teacher"]


def test_data_zip_is_hash_checked_safely_extracted_and_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "metc-data.zip"
    body = json.dumps({"schema_version": "metc-data-v2"}).encode()
    payload = _write_data_zip(archive, body)
    args = _args(tmp_path)
    args.data_root = None
    args.data_archive = archive
    args.data_sha256 = hashlib.sha256(payload).hexdigest()
    args.data_size_bytes = len(payload)
    args.data_root_directory = "metc-data"
    args.data_uncompressed_size_bytes = len(body)
    seen: list[Path] = []

    def validate(path: Path) -> Path:
        seen.append(path)
        assert (path / "manifest.full.json").is_file()
        return path

    monkeypatch.setattr(hydrate_and_reproduce, "_verify_data_root", validate)
    root = hydrate_and_reproduce._obtain_data_root(args, tmp_path / "cache")
    assert root.name == args.data_sha256
    assert seen[-1] == root


def test_sibling_pinned_data_zip_is_discovered_and_extracted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    software_root = tmp_path / "software"
    software_root.mkdir()
    archive = tmp_path / "metc-data.zip"
    payload = _write_data_zip(archive)
    reference = hydrate_and_reproduce.DataReference(
        hydrate_and_reproduce.reproduce.ArtifactReference(
            filename=archive.name,
            root_directory="metc-data",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            download_url=None,
        ),
        2,
    )
    args = _args(tmp_path)
    args.data_root = None
    args.data_archive = None
    seen: list[Path] = []

    def validate(path: Path) -> Path:
        seen.append(path)
        assert (path / "manifest.full.json").read_bytes() == b"{}"
        return path

    monkeypatch.setattr(hydrate_and_reproduce, "ROOT", software_root)
    monkeypatch.setattr(hydrate_and_reproduce, "_load_data_reference", lambda: reference)
    monkeypatch.setattr(hydrate_and_reproduce, "_verify_data_root", validate)
    monkeypatch.setattr(
        hydrate_and_reproduce.reproduce,
        "obtain_download",
        lambda *args, **kwargs: pytest.fail("sibling Data ZIP should be used"),
    )

    extracted = hydrate_and_reproduce._obtain_data_root(args, tmp_path / "cache")
    assert extracted == tmp_path / "cache" / "data" / reference.artifact.sha256
    assert seen[-1] == extracted


def test_missing_companion_data_has_no_network_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    software_root = tmp_path / "metc-code"
    software_root.mkdir()
    reference = hydrate_and_reproduce.DataReference(
        hydrate_and_reproduce.reproduce.ArtifactReference(
            filename="metc-data.zip",
            root_directory="metc-data",
            size_bytes=1,
            sha256="0" * 64,
            download_url=None,
        ),
        1,
    )
    args = _args(tmp_path)
    args.data_root = None
    args.data_archive = None

    monkeypatch.setattr(hydrate_and_reproduce, "ROOT", software_root)
    monkeypatch.setattr(hydrate_and_reproduce, "_load_data_reference", lambda: reference)
    monkeypatch.setattr(
        hydrate_and_reproduce.reproduce,
        "obtain_download",
        lambda *args, **kwargs: pytest.fail("Data must never be downloaded from GitHub"),
    )

    with pytest.raises(
        hydrate_and_reproduce.HydrationError,
        match="companion Data artifact not found",
    ):
        hydrate_and_reproduce._obtain_data_root(args, tmp_path / "cache")


@pytest.mark.parametrize(
    ("size_delta", "digest"),
    [
        (1, None),
        (0, "0" * 64),
    ],
)
def test_data_zip_requires_exact_size_and_sha256(
    tmp_path: Path,
    size_delta: int,
    digest: str | None,
) -> None:
    archive = tmp_path / "metc-data.zip"
    payload = _write_data_zip(archive)
    args = _args(tmp_path)
    args.data_root = None
    args.data_archive = archive
    args.data_sha256 = digest or hashlib.sha256(payload).hexdigest()
    args.data_size_bytes = len(payload) + size_delta
    args.data_root_directory = "metc-data"
    args.data_uncompressed_size_bytes = 2

    with pytest.raises(
        hydrate_and_reproduce.reproduce.ReproductionError,
        match=r"(has \d+ bytes|has SHA-256)",
    ):
        hydrate_and_reproduce._obtain_data_root(args, tmp_path / "cache")


def test_data_zip_enforces_pinned_uncompressed_size(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "metc-data.zip"
    payload = _write_data_zip(archive)
    reference = hydrate_and_reproduce.DataReference(
        hydrate_and_reproduce.reproduce.ArtifactReference(
            filename=archive.name,
            root_directory="metc-data",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            download_url=None,
        ),
        1,
    )
    with pytest.raises(
        hydrate_and_reproduce.HydrationError,
        match="declared uncompressed size",
    ):
        hydrate_and_reproduce._safe_extract_data_zip(
            archive,
            tmp_path / "extracted",
            reference,
        )


def test_metc_runtime_validation_is_skipped_for_other_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hydrate_and_reproduce.importlib.metadata,
        "version",
        lambda name: pytest.fail(f"unexpected METC runtime lookup: {name}"),
    )
    hydrate_and_reproduce._validate_metc_runtime(("errant", "ger", "jp"), "paper")


def test_metc_paper_runtime_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hydrate_and_reproduce.importlib.metadata,
        "version",
        lambda name: hydrate_and_reproduce.PAPER_METC_DISTRIBUTIONS[name],
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            is_bf16_supported=lambda: False,
        ),
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setattr(
        hydrate_and_reproduce.importlib,
        "import_module",
        lambda name: torch if name == "torch" else object(),
    )
    with pytest.raises(
        hydrate_and_reproduce.HydrationError,
        match="no usable CUDA device",
    ):
        hydrate_and_reproduce._validate_metc_runtime(("metc",), "paper")


def test_metc_paper_runtime_accepts_exact_supported_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hydrate_and_reproduce.importlib.metadata,
        "version",
        lambda name: hydrate_and_reproduce.PAPER_METC_DISTRIBUTIONS[name],
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
        ),
        version=SimpleNamespace(cuda="12.8"),
    )
    monkeypatch.setattr(
        hydrate_and_reproduce.importlib,
        "import_module",
        lambda name: torch if name == "torch" else object(),
    )
    hydrate_and_reproduce._validate_metc_runtime(("all",), "paper")


def test_metc_portable_runtime_requires_only_locked_core_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    looked_up: list[str] = []

    def version(name: str) -> str:
        looked_up.append(name)
        return hydrate_and_reproduce.PORTABLE_METC_DISTRIBUTIONS[name]

    imported: list[str] = []
    monkeypatch.setattr(hydrate_and_reproduce.importlib.metadata, "version", version)
    monkeypatch.setattr(
        hydrate_and_reproduce.importlib,
        "import_module",
        lambda name: imported.append(name) or object(),
    )

    hydrate_and_reproduce._validate_metc_runtime(("metc",), "portable")

    assert set(looked_up) == set(hydrate_and_reproduce.PORTABLE_METC_DISTRIBUTIONS)
    assert imported == ["torch"]


def test_portable_profile_is_forwarded_and_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)
    args = _args(tmp_path)
    args.metc_inference_profile = "portable"

    def execute(command, **kwargs):
        index = command.index("--metc-inference-profile")
        assert command[index + 1] == "portable"
        report = Path(command[command.index("--report") + 1])
        report.write_text('{"status":"KNOWN_DIFFERENCE"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(hydrate_and_reproduce.subprocess, "run", execute)
    assert hydrate_and_reproduce.run(args) == 2
    hydration = json.loads(
        (tmp_path / "work" / "hydration-report.json").read_text(encoding="utf-8")
    )
    assert hydration["metc_inference_profile"] == "portable"


def test_portable_profile_rejects_live_llm_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_setup(monkeypatch, tmp_path)
    args = _args(tmp_path)
    args.metc_inference_profile = "portable"
    args.llm = True

    with pytest.raises(
        hydrate_and_reproduce.HydrationError,
        match="require --metc-inference-profile paper",
    ):
        hydrate_and_reproduce.run(args)


def test_reproduce_py_routes_to_execution_first_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hydrate_and_reproduce, "main", lambda argv=None: 17)
    assert hydrate_and_reproduce.reproduce.main(["--anything"]) == 17
