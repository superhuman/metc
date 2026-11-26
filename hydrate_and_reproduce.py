#!/usr/bin/env python3
"""Hydrate pinned public dependencies, then execute the full METC E2E reproduction."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Sequence

import reproduce
from metc.checkpoint import verify_checkpoint_files
from metc.tokenizer_preparation import prepare_tokenizer
from metc_artifact.data_artifact import validate_data_artifact
from metc_artifact.e2e import (
    METC_INFERENCE_PROFILES,
    SYSTEM_CLI_CHOICES,
    normalize_systems,
)
from metc_artifact.hydration import HydrationError, hydrate_selected


ROOT = Path(__file__).resolve().parent
MAX_DATA_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
PAPER_METC_DISTRIBUTIONS = {
    "accelerate": "1.12.0",
    "flash-attn": "2.8.2",
    "torch": "2.9.0",
    "transformers": "4.57.3",
}
PORTABLE_METC_DISTRIBUTIONS = {
    "safetensors": "0.6.2",
    "tokenizers": "0.22.0",
    "torch": "2.9.0",
    "transformers": "4.57.3",
}


@dataclass(frozen=True)
class DataReference:
    artifact: reproduce.ArtifactReference
    expanded_size_bytes: int


def _metc_selected(systems: Sequence[str]) -> bool:
    return "all" in systems or "metc" in systems


def _validate_metc_runtime(
    systems: Sequence[str],
    inference_profile: str,
) -> None:
    if not _metc_selected(systems):
        return
    if inference_profile not in METC_INFERENCE_PROFILES:
        raise HydrationError(f"unknown METC inference profile: {inference_profile}")
    if inference_profile == "portable":
        instruction = "run the Code artifact with its locked `uv run --frozen` environment"
        for distribution, expected in PORTABLE_METC_DISTRIBUTIONS.items():
            try:
                observed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as error:
                raise HydrationError(
                    f"METC portable runtime is missing {distribution}=={expected}; {instruction}"
                ) from error
            if observed != expected:
                raise HydrationError(
                    f"METC portable runtime has {distribution}=={observed}, expected "
                    f"{expected}; {instruction}"
                )
        try:
            importlib.import_module("torch")
        except (ImportError, OSError) as error:
            raise HydrationError(
                f"METC portable runtime cannot load torch: {error}; {instruction}"
            ) from error
        return
    instruction = (
        "install the pinned paper runtime with `uv sync --extra paper --frozen` "
        "on an NVIDIA host with CUDA 12.8 and BF16 support"
    )
    for distribution, expected in PAPER_METC_DISTRIBUTIONS.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise HydrationError(
                f"METC paper runtime is missing {distribution}=={expected}; {instruction}"
            ) from error
        if observed != expected:
            raise HydrationError(
                f"METC paper runtime has {distribution}=={observed}, expected "
                f"{expected}; {instruction}"
            )
    try:
        torch = importlib.import_module("torch")
        importlib.import_module("flash_attn")
    except (ImportError, OSError) as error:
        raise HydrationError(
            f"METC paper runtime cannot load torch/flash-attn: {error}; {instruction}"
        ) from error
    if not torch.cuda.is_available():
        raise HydrationError(f"METC paper runtime has no usable CUDA device; {instruction}")
    if torch.version.cuda != "12.8":
        raise HydrationError(
            f"METC paper runtime uses CUDA {torch.version.cuda!r}, expected '12.8'; {instruction}"
        )
    if not torch.cuda.is_bf16_supported():
        raise HydrationError(f"METC paper runtime has no CUDA BF16 support; {instruction}")


def _verify_data_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    validate_data_artifact(root)
    return root


def _manifest_sha256(data_root: Path) -> str | None:
    manifest = data_root / "manifest.full.json"
    return reproduce.sha256_file(manifest) if manifest.is_file() else None


def _prepare_work_root(work_root: Path, *, resume: bool) -> None:
    if resume:
        if not work_root.is_dir() or not any(work_root.iterdir()):
            raise HydrationError(
                f"--resume requires an existing non-empty work directory: {work_root}"
            )
        hydration_report = work_root / "hydration-report.json"
        try:
            metadata = hydration_report.lstat()
        except FileNotFoundError as error:
            raise HydrationError(
                f"--resume requires the prior hydration report: {hydration_report}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise HydrationError(
                f"prior hydration report is not a regular file: {hydration_report}"
            )
        return
    if work_root.exists() and any(work_root.iterdir()):
        raise HydrationError(
            f"work directory is not empty; select a clean --work-root: {work_root}"
        )
    work_root.mkdir(parents=True, exist_ok=True)


def _record_or_verify_hydration(
    path: Path,
    expected: dict[str, object],
    *,
    resume: bool,
) -> None:
    if resume:
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise HydrationError(f"cannot read prior hydration report {path}: {error}") from error
        if observed != expected:
            observed_mapping = observed if isinstance(observed, dict) else {}
            differing = sorted(
                key
                for key in set(observed_mapping) | set(expected)
                if observed_mapping.get(key) != expected.get(key)
            )
            raise HydrationError(
                "resume configuration differs from the original run in: " + ", ".join(differing)
            )
        return
    path.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_data_reference(
    path: Path = ROOT / "release_artifacts.json",
) -> DataReference:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        data = value["data"]
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError, KeyError) as error:
        raise HydrationError(
            "the Code artifact has no pinned Data archive metadata; pass --data-root "
            "or --data-archive with --data-sha256, --data-size-bytes, and "
            "--data-root-directory, and --data-uncompressed-size-bytes"
        ) from error
    required = {
        "status",
        "delivery",
        "filename",
        "root_directory",
        "size_bytes",
        "sha256",
        "uncompressed_size_bytes",
    }
    if not isinstance(data, dict) or set(data) != required:
        raise HydrationError("release_artifacts.json data fields do not match the contract")
    if data.get("status") != "available":
        raise HydrationError("the complete Data artifact is not marked available")
    if data.get("delivery") != "companion_release_asset":
        raise HydrationError("the Data artifact delivery mode is not supported")
    expanded = data.get("uncompressed_size_bytes")
    if (
        not isinstance(expanded, int)
        or isinstance(expanded, bool)
        or expanded <= 0
        or expanded > MAX_DATA_EXPANDED_BYTES
    ):
        raise HydrationError("data.uncompressed_size_bytes is invalid")
    artifact = reproduce.ArtifactReference(
        filename=reproduce._require_plain_filename(data.get("filename"), field="data.filename"),
        root_directory=reproduce._require_root_directory(
            data.get("root_directory"),
            field="data.root_directory",
        ),
        size_bytes=reproduce._require_size(data.get("size_bytes"), field="data.size_bytes"),
        sha256=reproduce._require_digest(data.get("sha256"), field="data.sha256"),
        download_url=None,
    )
    return DataReference(artifact, expanded)


def _explicit_data_reference(args: argparse.Namespace) -> DataReference | None:
    supplied = (
        args.data_sha256,
        args.data_size_bytes,
        args.data_root_directory,
        args.data_uncompressed_size_bytes,
    )
    if not any(value is not None for value in supplied):
        return None
    if args.data_archive is None:
        raise HydrationError("explicit Data metadata is only valid together with --data-archive")
    if not all(value is not None for value in supplied):
        raise HydrationError(
            "explicit Data metadata requires --data-sha256, --data-size-bytes, "
            "--data-root-directory, and --data-uncompressed-size-bytes together"
        )
    archive = args.data_archive.expanduser().resolve()
    digest = reproduce._require_digest(args.data_sha256, field="--data-sha256")
    size = reproduce._require_size(args.data_size_bytes, field="--data-size-bytes")
    root = reproduce._require_root_directory(
        args.data_root_directory,
        field="--data-root-directory",
    )
    expanded = reproduce._require_size(
        args.data_uncompressed_size_bytes,
        field="--data-uncompressed-size-bytes",
    )
    if expanded > MAX_DATA_EXPANDED_BYTES:
        raise HydrationError(f"--data-uncompressed-size-bytes exceeds {MAX_DATA_EXPANDED_BYTES}")
    return DataReference(
        reproduce.ArtifactReference(
            filename=archive.name,
            root_directory=root,
            size_bytes=size,
            sha256=digest,
            download_url=None,
        ),
        expanded,
    )


def _safe_extract_data_zip(
    archive_path: Path,
    destination: Path,
    reference: DataReference,
) -> Path:
    artifact = reference.artifact
    try:
        archive = zipfile.ZipFile(archive_path, mode="r")
    except (OSError, zipfile.BadZipFile) as error:
        raise HydrationError(f"invalid Data ZIP: {error}") from error
    with archive:
        if archive.comment:
            raise HydrationError("Data ZIP contains an archive comment")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != sorted(names):
            raise HydrationError("Data ZIP members are not sorted")
        if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise HydrationError("Data ZIP contains duplicate or colliding members")
        declared_total = 0
        for info in infos:
            try:
                reproduce._validate_member_name(info.filename, artifact.root_directory)
            except reproduce.ReproductionError as error:
                raise HydrationError(str(error)) from error
            mode = info.external_attr >> 16
            declared_total += info.file_size
            if (
                info.is_dir()
                or info.flag_bits & 0x1
                or info.date_time != reproduce.FIXED_ZIP_TIMESTAMP
                or info.create_system != 3
                or info.extra
                or info.comment
                or info.compress_type != zipfile.ZIP_DEFLATED
                or not stat.S_ISREG(mode)
                or stat.S_ISLNK(mode)
            ):
                raise HydrationError(
                    f"unsafe or non-deterministic Data ZIP member: {info.filename}"
                )
        if declared_total != reference.expanded_size_bytes:
            raise HydrationError(
                "Data ZIP declared uncompressed size does not match its pinned value"
            )

        destination.mkdir(parents=True, exist_ok=False)
        observed_total = 0
        for info in infos:
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            member_size = 0
            try:
                with archive.open(info, mode="r") as source, target.open("xb") as output:
                    while block := source.read(1024 * 1024):
                        member_size += len(block)
                        observed_total += len(block)
                        if observed_total > reference.expanded_size_bytes:
                            raise HydrationError("Data ZIP exceeded its extraction size limit")
                        output.write(block)
            except HydrationError:
                raise
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise HydrationError(f"cannot extract {info.filename}: {error}") from error
            if member_size != info.file_size:
                raise HydrationError(f"extracted size differs for {info.filename}")
            target.chmod(0o644)
    return destination / artifact.root_directory


def _discover_data_root() -> Path | None:
    candidates = {
        path.resolve() for path in (ROOT / "metc-data", ROOT.parent / "metc-data") if path.is_dir()
    }
    if len(candidates) > 1:
        raise HydrationError("multiple extracted Data artifacts found; select one with --data-root")
    return next(iter(candidates), None)


def _discover_data_archive(reference: DataReference) -> Path | None:
    filename = reference.artifact.filename
    candidates = {
        path.resolve() for path in (ROOT / filename, ROOT.parent / filename) if path.is_file()
    }
    if len(candidates) > 1:
        raise HydrationError("multiple Data ZIPs found; select one with --data-archive")
    return next(iter(candidates), None)


def _obtain_data_root(args: argparse.Namespace, cache_dir: Path) -> Path:
    if args.data_root is not None:
        if any(
            value is not None
            for value in (
                args.data_sha256,
                args.data_size_bytes,
                args.data_root_directory,
                args.data_uncompressed_size_bytes,
            )
        ):
            raise HydrationError("Data ZIP metadata cannot be combined with --data-root")
        return _verify_data_root(args.data_root)
    discovered = _discover_data_root()
    if discovered is not None and args.data_archive is None:
        return _verify_data_root(discovered)

    explicit = _explicit_data_reference(args)
    reference = explicit or _load_data_reference()
    artifact = reference.artifact
    archive = (
        args.data_archive.expanduser().resolve()
        if args.data_archive is not None
        else _discover_data_archive(reference)
    )
    if archive is None:
        raise HydrationError(
            "companion Data artifact not found; place metc-data.zip beside "
            "the extracted metc-code directory, extract metc-data beside it, "
            "or pass --data-archive/--data-root"
        )
    reproduce.verify_artifact_file(archive, artifact)

    destination = cache_dir / "data" / artifact.sha256
    lock_path = destination.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            return _verify_data_root(destination)
        container = Path(tempfile.mkdtemp(prefix=".metc-data-", dir=destination.parent))
        try:
            extracted = _safe_extract_data_zip(
                archive,
                container / "archive",
                reference,
            )
            _verify_data_root(extracted)
            extracted.replace(destination)
        finally:
            if container.exists():
                shutil.rmtree(container)
        return _verify_data_root(destination)


def _hydrate_checkpoint(
    *,
    cache_dir: Path,
    checkpoint_archive: Path | None,
    checkpoint_dir: Path | None,
    tokenizer_source: Path | None,
    no_network: bool,
) -> Path:
    if checkpoint_dir is not None:
        root = checkpoint_dir.expanduser().resolve()
        verify_checkpoint_files(root)
        return root

    reference = reproduce.load_checkpoint_reference()
    destination = cache_dir / "checkpoints" / reference.sha256
    lock_path = destination.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            verify_checkpoint_files(destination)
            return destination
        if checkpoint_archive is None:
            archive = reproduce.obtain_download(reference, cache_dir, no_network=no_network)
        else:
            archive = reproduce.verify_artifact_file(
                checkpoint_archive.expanduser().resolve(),
                reference,
            )
        if no_network and tokenizer_source is None:
            raise HydrationError(
                "uncached checkpoint hydration with --no-network requires --tokenizer-source"
            )
        container = Path(tempfile.mkdtemp(prefix=".metc-checkpoint-", dir=destination.parent))
        try:
            extracted = reproduce.safe_extract_zip(
                archive,
                container / "archive",
                reference,
            )
            prepare_tokenizer(
                extracted,
                source_dir=(
                    tokenizer_source.expanduser().resolve()
                    if tokenizer_source is not None
                    else None
                ),
            )
            verify_checkpoint_files(extracted)
            extracted.replace(destination)
        finally:
            if container.exists():
                shutil.rmtree(container)
        verify_checkpoint_files(destination)
        return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group()
    data.add_argument(
        "--data-root",
        type=Path,
        help="extracted complete METC Data artifact",
    )
    data.add_argument("--data-archive", type=Path, help="local complete metc-data.zip")
    parser.add_argument("--data-sha256")
    parser.add_argument("--data-size-bytes", type=int)
    parser.add_argument("--data-root-directory")
    parser.add_argument("--data-uncompressed-size-bytes", type=int)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path.cwd() / "metc-e2e-work",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume the same LLM run in an existing work root after verifying "
            "its Data, checkpoint, runtimes, systems, and run configuration"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "metc-e2e",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--checkpoint-archive", type=Path)
    checkpoint.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--tokenizer-source", type=Path)
    parser.add_argument(
        "--metc-inference-profile",
        choices=METC_INFERENCE_PROFILES,
        default="paper",
        help=(
            "paper requires the evaluated CUDA/bfloat16 runtime; portable runs fp32 "
            "locally and reports METC drift as a known difference"
        ),
    )
    parser.add_argument("--gerrant-resources", type=Path)
    parser.add_argument("--stanza-resources", type=Path)
    parser.add_argument("--gsdltp-archive", type=Path)
    parser.add_argument(
        "--accept-ltp-terms",
        action="store_true",
        help=(
            "confirm that you reviewed the nonstandard HIT-SCIR LTP terms and "
            "that your use of the pinned LTP/small model is permitted"
        ),
    )
    parser.add_argument("--uv", default=os.environ.get("UV", "uv"))
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="require every runtime, model, checkpoint, tokenizer, and dataset source in cache",
    )
    parser.add_argument("--jp-workers", type=int, default=2)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=SYSTEM_CLI_CHOICES,
        default=("all",),
        help=(
            "annotation systems to execute: METC, METC teacher, ERRANT, gERRANT, "
            "and JP-ERRANT (public gerrant/jp_errant names are recommended; "
            "internal ger/jp aliases are also accepted); ChERRANT is "
            "always used for Chinese reference reconstruction"
        ),
    )
    parser.add_argument(
        "--llm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "rerun the teacher and both judges using API credentials from the "
            "environment (expensive)"
        ),
    )
    parser.add_argument("--llm-run-id", default="e2e")
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--llm-max-attempts", type=int, default=4)
    parser.add_argument("--reference-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=24 * 60 * 60)
    parser.add_argument("--report", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    if sys.version_info[:2] != (3, 12):
        raise HydrationError(
            f"the Code artifact requires CPython 3.12, got {sys.version.split()[0]}"
        )
    systems = normalize_systems(args.systems)
    if args.llm and args.metc_inference_profile != "paper":
        raise HydrationError(
            "fresh teacher and judge reruns require --metc-inference-profile paper"
        )
    _validate_metc_runtime(systems, args.metc_inference_profile)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.work_root.expanduser().resolve()
    _prepare_work_root(work_root, resume=args.resume)
    data_root = _obtain_data_root(args, cache_dir)

    checkpoint = None
    if _metc_selected(systems):
        checkpoint = _hydrate_checkpoint(
            cache_dir=cache_dir,
            checkpoint_archive=args.checkpoint_archive,
            checkpoint_dir=args.checkpoint_dir,
            tokenizer_source=args.tokenizer_source,
            no_network=args.no_network,
        )
    runtimes = hydrate_selected(
        software_root=ROOT,
        cache_dir=cache_dir,
        uv=args.uv,
        systems=systems,
        no_network=args.no_network,
        gerrant_resources=args.gerrant_resources,
        stanza_resources=args.stanza_resources,
        gsdltp_archive=args.gsdltp_archive,
        data_root=data_root,
        accept_ltp_terms=args.accept_ltp_terms,
    )
    hydration_report = {
        "schema_version": "metc-review-hydration-v2",
        "data": {
            "root": str(data_root),
            "manifest_sha256": _manifest_sha256(data_root),
        },
        "checkpoint": (
            {
                "root": str(checkpoint),
                "archive_sha256": reproduce.load_checkpoint_reference().sha256,
            }
            if checkpoint is not None
            else None
        ),
        "effective_systems": list(systems),
        "metc_inference_profile": args.metc_inference_profile,
        "runtimes": runtimes.to_dict(),
        "run": {
            "jp_workers": args.jp_workers,
            "llm": args.llm,
            "llm_max_attempts": args.llm_max_attempts,
            "llm_run_id": args.llm_run_id,
            "llm_workers": args.llm_workers,
            "no_network": args.no_network,
            "accepted_ltp_terms": args.accept_ltp_terms,
            "reference_samples": args.reference_samples,
            "seed": args.seed,
            "timeout_seconds": args.timeout_seconds,
        },
    }
    hydration_report_path = work_root / "hydration-report.json"
    _record_or_verify_hydration(
        hydration_report_path,
        hydration_report,
        resume=args.resume,
    )

    report = (
        args.report.expanduser().resolve()
        if args.report is not None
        else work_root / "e2e-report.json"
    )
    command = [
        sys.executable,
        str(ROOT / "reproduce_e2e.py"),
        "--data-root",
        str(data_root),
        "--work-root",
        str(work_root),
        "--cache-dir",
        str(cache_dir / "public-sources"),
        "--report",
        str(report),
        "--core-python",
        sys.executable,
        "--cherrant-python",
        str(runtimes.cherrant_python),
        "--jp-workers",
        str(args.jp_workers),
        "--systems",
        *systems,
        "--metc-inference-profile",
        args.metc_inference_profile,
        "--reference-samples",
        str(args.reference_samples),
        "--seed",
        str(args.seed),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--llm-run-id",
        args.llm_run_id,
        "--llm-workers",
        str(args.llm_workers),
        "--llm-max-attempts",
        str(args.llm_max_attempts),
        "--llm" if args.llm else "--no-llm",
    ]
    if checkpoint is not None:
        command.extend(("--metc-python", sys.executable, "--metc-checkpoint", str(checkpoint)))
    if runtimes.errant_python is not None:
        command.extend(("--errant-python", str(runtimes.errant_python)))
    if runtimes.gerrant_python is not None and runtimes.gerrant_root is not None:
        command.extend(
            (
                "--gerrant-python",
                str(runtimes.gerrant_python),
                "--gerrant-root",
                str(runtimes.gerrant_root),
            )
        )
    if (
        runtimes.jp_python is not None
        and runtimes.jp_transform_manifest is not None
        and runtimes.jp_stanza_resources is not None
        and runtimes.jp_zh_gsdltp_dir is not None
    ):
        command.extend(
            (
                "--jp-python",
                str(runtimes.jp_python),
                "--jp-transform-manifest",
                str(runtimes.jp_transform_manifest),
                "--jp-stanza-resources",
                str(runtimes.jp_stanza_resources),
                "--jp-zh-gsdltp-dir",
                str(runtimes.jp_zh_gsdltp_dir),
            )
        )
    if args.no_network:
        command.append("--no-network")
    if runtimes.cherrant_hf_home is None:
        raise HydrationError("ChERRANT model hydration did not produce an HF cache")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "HF_HOME": str(runtimes.cherrant_hf_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        shell=False,
    )
    if not report.is_file() or report.stat().st_size == 0:
        raise HydrationError(
            f"E2E command exited with {completed.returncode} without producing {report}"
        )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HydrationError, reproduce.ReproductionError) as error:
        raise SystemExit(f"METC E2E HYDRATION FAILED: {error}") from error
