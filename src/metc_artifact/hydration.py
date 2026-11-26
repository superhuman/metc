"""Fail-closed hydration of the external runtimes used by the paper.

The Code artifact does not redistribute third-party model packs or the
unlicensed JP-ERRANT checkout.  This module obtains them from their public,
pinned sources into a user-owned cache, verifies them, and returns only
explicit paths for the execution-first E2E runner.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .adapters.gerrant import validate_gerrant_root


SCHEMA_VERSION = "metc-hydration-v1"
PYTHON_VERSION = "3.11.13"
RUNTIME_NAMES = ("errant", "gerrant", "jp_errant", "cherrant")
_SECRET_ENVIRONMENT_MARKERS = (
    "API_KEY",
    "AUTHORIZATION",
    "COOKIE",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)

JP_REPOSITORY = "https://github.com/open-writing-evaluation/jp_errant_bea.git"
JP_COMMIT = "839ac8641b15272c2fcb6bc74015e88dda4a8842"
JP_TRANSFORMATIONS = {
    "parallel_legacy_true": True,
    "zh_explicit_gsdltp_models": True,
    "zh_empty_shape_table_fallback": True,
}
JP_UPSTREAM_FILES = {
    "jp_errant/__init__.py": "2b1001022c66f3ed716e31190915f58a67901d6f1767c95913a8b65cd63c0a2e",
    "jp_errant/commands/parallel_to_m2.py": (
        "446436358a81f720618dde207212ccf7c4f427254c2018fbe97b577a8a1d62c2"
    ),
    "jp_errant/zh/char_similar.py": (
        "87758ac2b31f29024cd1bf8078ddd72cd1fdf2dc67d756caf92cf9996ddd4b3a"
    ),
}
JP_PATCHED_FILES = {
    "jp_errant/__init__.py": "d6a57e200b096b44ca48201674482624d1dc503e2dd2dc0ab66c0ed37adaef4d",
    "jp_errant/commands/parallel_to_m2.py": (
        "fe01a6db7ba0de0216207e957ba57ad6d2101f057e3ea478a4b13146a82e642c"
    ),
    "jp_errant/zh/char_similar.py": (
        "94c761ef599e8cc1ae31a5ea707b1ebb8142bd790d8a43d5d565596498700a1b"
    ),
}
JP_TRANSFORM_SHA256 = "91d0e8026db672224db96e7e19a752a3bb46846a40a4e42272e330669ab129d9"

GSDLTP_ARCHIVE_URL = (
    "https://zenodo.org/api/records/15096936/files/UD_Chinese-GSDLTP.tar.gz/content"
)
GSDLTP_ARCHIVE_SIZE = 353_304_777
GSDLTP_ARCHIVE_SHA256 = "bcf6673629dd8574402d1eb9124a1e19a12f75d299abe4617d84e757404d91ed"
GSDLTP_ARCHIVE_FILES = {
    (
        "UD_Chinese-GSDLTP/UD_Chinese-GSDSimp_model-charlm/saved_models/"
        "tokenize/zh-hans_gsdsimp_tokenizer.pt"
    ): (
        "tokenize/zh-hans_gsdsimp_tokenizer.pt",
        1_427_059,
        "341a8af214d4d750de3b79975c82956620412bfe668f893277ce02587d1a175a",
    ),
    (
        "UD_Chinese-GSDLTP/UD_Chinese-GSDSimp_model-charlm/saved_models/"
        "pos/zh-hans_gsdsimp_charlm_tagger.pt"
    ): (
        "pos/zh-hans_gsdsimp_charlm_tagger.pt",
        33_573_184,
        "9ace01ec8db8eedc0577ae0e400e6bef795ed0d4892ae02c593631b37594dc2f",
    ),
}

GERRANT_RESOURCES = {
    "de-stts_map": (
        539,
        "840a7d37c2364b3a5f628900a58c4cc513014d47de6d3a96414fa43c9231c869",
    ),
    "de_DE-large.txt": (
        4_778_232,
        "577c06ebd6461c0a43486a9a538c1175fbc1f0d1a3985e9b4350a5ad1c5d9da6",
    ),
}

LTP_REPOSITORY = "LTP/small"
LTP_REVISION = "dea87bf0da3c5d054bbd426a78d7fa9efd347799"
LTP_TERMS_URL = "https://github.com/HIT-SCIR/ltp#开源协议"

REQUIRED_DISTRIBUTIONS: Mapping[str, Mapping[str, str]] = {
    "errant": {
        "click": "8.3.1",
        "errant": "3.0.0",
        "spacy": "3.8.14",
        "en-core-web-sm": "3.8.0",
        "rapidfuzz": "3.14.5",
    },
    "gerrant": {
        "click": "8.3.1",
        "spacy": "3.8.14",
        "de-core-news-sm": "3.8.0",
        "nltk": "3.9.4",
        "regex": "2026.6.28",
    },
    "jp_errant": {
        "click": "8.3.1",
        "jp-errant": "3.0.1",
        "errant": "3.0.0",
        "stanza": "1.10.1",
        "torch": "2.2.2",
        "transformers": "4.49.0",
        "numpy": "1.26.4",
        "scipy": "1.15.2",
        "pandas": "3.0.3",
        "pypinyin": "0.53.0",
    },
    "cherrant": {
        "ltp": "4.2.14",
        "ltp-core": "0.1.4",
        "torch": "2.12.1",
        "transformers": "4.40.0",
        "opencc": "1.3.1",
        "python-levenshtein": "0.12.1",
        "pypinyin": "0.55.0",
        "huggingface-hub": "0.22.2",
    },
}


class HydrationError(RuntimeError):
    """A pinned runtime or model could not be obtained and verified."""


@dataclass(frozen=True)
class PinnedAsset:
    filename: str
    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class HydratedRuntimes:
    errant_python: Path | None = None
    gerrant_python: Path | None = None
    gerrant_root: Path | None = None
    jp_python: Path | None = None
    jp_transform_manifest: Path | None = None
    jp_stanza_resources: Path | None = None
    jp_zh_gsdltp_dir: Path | None = None
    cherrant_python: Path | None = None
    cherrant_hf_home: Path | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            key: str(value) for key, value in sorted(self.__dict__.items()) if value is not None
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_regular_file(
    path: Path,
    *,
    size: int | None = None,
    sha256: str | None = None,
) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise HydrationError(f"required file is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HydrationError(f"required path is not a regular file: {path}")
    if size is not None and metadata.st_size != size:
        raise HydrationError(f"{path} has {metadata.st_size} bytes; expected {size}")
    if sha256 is not None:
        observed = sha256_file(path)
        if observed != sha256:
            raise HydrationError(f"{path} has SHA-256 {observed}; expected {sha256}")
    return path


def _clean_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    if extra:
        environment.update(extra)
    environment = {
        key: value
        for key, value in environment.items()
        if not any(marker in key.upper() for marker in _SECRET_ENVIRONMENT_MARKERS)
    }
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _run_checked(
    argv: Sequence[str | Path],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    timeout: float = 60 * 60,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [str(value) for value in argv],
            cwd=cwd,
            env=dict(environment or _clean_environment()),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HydrationError(f"could not run {argv[0]}: {error}") from error
    if result.returncode:
        detail = "\n".join((result.stderr or result.stdout).splitlines()[-30:])
        raise HydrationError(
            f"command exited with {result.returncode}: {' '.join(str(value) for value in argv)}"
            + (f"\n{detail}" if detail else "")
        )
    return result


def _python_executable(root: Path) -> Path:
    if os.name == "nt":  # pragma: no cover - paper runtimes were Linux/macOS
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _runtime_lock_digest(project: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("pyproject.toml", "uv.lock"):
        path = _require_regular_file(project / filename)
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _distribution_snapshot(python: Path, *, cwd: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json,re;"
        "canonical=lambda s:re.sub(r'[-_.]+','-',s.lower());"
        "print(json.dumps({canonical(d.metadata['Name']):d.version for d in m.distributions()},"
        "sort_keys=True))"
    )
    result = _run_checked((python, "-c", code), cwd=cwd)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise HydrationError(f"runtime package inventory is not JSON: {python}") from error
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(version, str) for key, version in value.items()
    ):
        raise HydrationError(f"runtime package inventory is malformed: {python}")
    return value


def _verify_runtime(root: Path, runtime: str, lock_digest: str) -> Path:
    marker_path = _require_regular_file(root / ".metc-runtime.json")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HydrationError(f"invalid runtime marker: {marker_path}") from error
    expected = {
        "schema_version": SCHEMA_VERSION,
        "runtime": runtime,
        "python_version": PYTHON_VERSION,
        "lock_sha256": lock_digest,
    }
    if marker != expected:
        raise HydrationError(f"runtime marker does not match the lock: {root}")
    python = _python_executable(root)
    if not python.exists() or not os.access(python, os.X_OK):
        raise HydrationError(f"runtime Python is missing or not executable: {python}")
    version = _run_checked(
        (python, "-c", "import platform; print(platform.python_version())"),
        cwd=root,
    ).stdout.strip()
    if version != PYTHON_VERSION:
        raise HydrationError(f"{runtime} uses Python {version}; expected {PYTHON_VERSION}")
    installed = _distribution_snapshot(python, cwd=root)
    for name, expected_version in REQUIRED_DISTRIBUTIONS[runtime].items():
        observed = installed.get(name)
        if observed != expected_version:
            raise HydrationError(
                f"{runtime} distribution {name} is {observed!r}; expected {expected_version}"
            )
    return python


def _runtime_marker(runtime: str, lock_digest: str) -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime": runtime,
        "python_version": PYTHON_VERSION,
        "lock_sha256": lock_digest,
    }


def _with_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    return stream


def hydrate_runtime(
    runtime: str,
    *,
    software_root: Path,
    cache_dir: Path,
    uv: str | Path,
    no_network: bool,
    jp_source: Path | None = None,
) -> Path:
    if runtime not in RUNTIME_NAMES:
        raise HydrationError(f"unknown runtime: {runtime}")
    project = software_root / "runtime-locks" / runtime
    lock_digest = _json_digest(
        {
            "python_version": PYTHON_VERSION,
            "runtime_lock_sha256": _runtime_lock_digest(project),
        }
    )
    if runtime == "jp_errant":
        if jp_source is None:
            raise HydrationError("JP-ERRANT runtime requires its transformed source")
        lock_digest = _json_digest(
            {
                "runtime_lock_sha256": lock_digest,
                "transform_sha256": JP_TRANSFORM_SHA256,
            }
        )
    destination = cache_dir / "runtimes" / f"{runtime}-{lock_digest[:16]}"
    lock_stream = _with_lock(destination.with_suffix(".lock"))
    try:
        if destination.exists():
            return _verify_runtime(destination, runtime, lock_digest)
        if no_network:
            raise HydrationError(f"{runtime} runtime is not cached and --no-network was supplied")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{runtime}-", dir=destination.parent))
        try:
            python_install_dir = cache_dir / "python"
            environment = _clean_environment(
                {
                    "UV_PROJECT_ENVIRONMENT": str(temporary),
                    "UV_PYTHON_INSTALL_DIR": str(python_install_dir),
                }
            )
            _run_checked(
                (
                    uv,
                    "python",
                    "install",
                    "--no-bin",
                    "--install-dir",
                    python_install_dir,
                    PYTHON_VERSION,
                ),
                cwd=software_root,
                environment=environment,
            )
            _run_checked(
                (
                    uv,
                    "sync",
                    "--project",
                    project,
                    "--frozen",
                    "--no-install-project",
                    "--python",
                    PYTHON_VERSION,
                    "--managed-python",
                ),
                cwd=software_root,
                environment=environment,
            )
            python = _python_executable(temporary)
            if runtime == "jp_errant":
                assert jp_source is not None
                _run_checked(
                    (
                        uv,
                        "pip",
                        "install",
                        "--no-config",
                        "--python",
                        python,
                        "--no-deps",
                        "--no-build-isolation",
                        jp_source,
                    ),
                    cwd=software_root,
                    environment=_clean_environment(),
                )
                _run_checked(
                    (
                        python,
                        "-c",
                        "import jp_errant,jp_errant.zh.char_similar;"
                        "print(jp_errant.__version__ if hasattr(jp_errant,'__version__') else 'ok')",
                    ),
                    cwd=software_root,
                )
            _write_json(
                temporary / ".metc-runtime.json",
                _runtime_marker(runtime, lock_digest),
            )
            _verify_runtime(temporary, runtime, lock_digest)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return _verify_runtime(destination, runtime, lock_digest)
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def _verify_jp_source(root: Path, transform_path: Path) -> Path:
    marker_path = _require_regular_file(root / ".metc-transform.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "metc-jp-transform-v1",
        "repository": JP_REPOSITORY,
        "upstream_commit": JP_COMMIT,
        "transform_sha256": JP_TRANSFORM_SHA256,
        "transformations": JP_TRANSFORMATIONS,
        "transformed_files": JP_PATCHED_FILES,
    }
    if marker != expected:
        raise HydrationError(f"JP-ERRANT transform marker differs: {marker_path}")
    _require_regular_file(transform_path, sha256=JP_TRANSFORM_SHA256)
    for relative, digest in JP_PATCHED_FILES.items():
        _require_regular_file(root / relative, sha256=digest)
    return marker_path


def hydrate_jp_source(
    *,
    software_root: Path,
    cache_dir: Path,
    no_network: bool,
) -> tuple[Path, Path]:
    transform_path = software_root / "runtime-tools" / "transform_jp_errant.py"
    _require_regular_file(transform_path, sha256=JP_TRANSFORM_SHA256)
    key = _json_digest({"commit": JP_COMMIT, "transform": JP_TRANSFORM_SHA256})
    destination = cache_dir / "sources" / f"jp-errant-{key[:16]}"
    lock_stream = _with_lock(destination.with_suffix(".lock"))
    try:
        if destination.exists():
            return destination, _verify_jp_source(destination, transform_path)
        if no_network:
            raise HydrationError("JP-ERRANT source is not cached and --no-network was supplied")
        if shutil.which("git") is None:
            raise HydrationError("git is required to hydrate JP-ERRANT")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".jp-errant-", dir=destination.parent))
        shutil.rmtree(temporary)
        try:
            _run_checked(
                (
                    "git",
                    "clone",
                    "--no-checkout",
                    "--filter=blob:none",
                    JP_REPOSITORY,
                    temporary,
                ),
                cwd=destination.parent,
            )
            _run_checked(
                ("git", "checkout", "--detach", JP_COMMIT),
                cwd=temporary,
            )
            head = _run_checked(("git", "rev-parse", "HEAD"), cwd=temporary).stdout.strip()
            dirty = _run_checked(
                ("git", "status", "--porcelain", "--untracked-files=no"),
                cwd=temporary,
            ).stdout.strip()
            if head != JP_COMMIT or dirty:
                raise HydrationError("JP-ERRANT checkout is not the clean pinned commit")
            for relative, digest in JP_UPSTREAM_FILES.items():
                _require_regular_file(temporary / relative, sha256=digest)
            _run_checked(
                (
                    sys.executable,
                    transform_path,
                    "--source-root",
                    temporary,
                ),
                cwd=temporary,
            )
            for relative, digest in JP_PATCHED_FILES.items():
                _require_regular_file(temporary / relative, sha256=digest)
            shutil.rmtree(temporary / ".git")
            marker = {
                "schema_version": "metc-jp-transform-v1",
                "repository": JP_REPOSITORY,
                "upstream_commit": JP_COMMIT,
                "transform_sha256": JP_TRANSFORM_SHA256,
                "transformations": JP_TRANSFORMATIONS,
                "transformed_files": JP_PATCHED_FILES,
            }
            _write_json(temporary / ".metc-transform.json", marker)
            _verify_jp_source(temporary, transform_path)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination, _verify_jp_source(destination, transform_path)
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


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
                request.full_url, code, "unsafe hydration redirect", headers, file_pointer
            )
        return super().redirect_request(request, file_pointer, code, message, headers, new_url)


def _download_asset(asset: PinnedAsset, cache_dir: Path, *, no_network: bool) -> Path:
    parsed = urlsplit(asset.url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise HydrationError(f"asset URL is not a credential-free HTTPS URL: {asset.url}")
    destination = cache_dir / "downloads" / f"{asset.sha256[:16]}-{asset.filename}"
    if destination.exists():
        return _require_regular_file(
            destination,
            size=asset.size_bytes,
            sha256=asset.sha256,
        )
    if no_network:
        raise HydrationError(f"{asset.filename} is not cached and --no-network was supplied")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{asset.filename}-",
        suffix=".part",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    written = 0
    try:
        opener = urllib.request.build_opener(_HTTPSRedirectHandler())
        request = urllib.request.Request(asset.url, headers={"User-Agent": "metc-hydration/1"})
        with os.fdopen(descriptor, "wb") as output:
            try:
                with opener.open(request, timeout=60) as response:
                    while block := response.read(1024 * 1024):
                        written += len(block)
                        if written > asset.size_bytes:
                            raise HydrationError(f"{asset.filename} exceeds its declared size")
                        digest.update(block)
                        output.write(block)
            except HydrationError:
                raise
            except (OSError, urllib.error.URLError) as error:
                raise HydrationError(f"could not download {asset.url}: {error}") from error
            output.flush()
            os.fsync(output.fileno())
        if written != asset.size_bytes or digest.hexdigest() != asset.sha256:
            raise HydrationError(f"{asset.filename} failed size or SHA-256 verification")
        temporary.replace(destination)
        return _require_regular_file(
            destination,
            size=asset.size_bytes,
            sha256=asset.sha256,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _verify_gsdltp(root: Path) -> Path:
    expected = {output: (size, digest) for output, size, digest in GSDLTP_ARCHIVE_FILES.values()}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    if actual != set(expected):
        raise HydrationError(
            f"GSDLTP inventory differs: missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    for relative, (size, digest) in expected.items():
        _require_regular_file(root / relative, size=size, sha256=digest)
    return root


def hydrate_gsdltp(
    *,
    cache_dir: Path,
    no_network: bool,
    archive: Path | None = None,
) -> Path:
    asset = PinnedAsset(
        "UD_Chinese-GSDLTP.tar.gz",
        GSDLTP_ARCHIVE_URL,
        GSDLTP_ARCHIVE_SIZE,
        GSDLTP_ARCHIVE_SHA256,
    )
    destination = cache_dir / "models" / f"gsdltp-{asset.sha256[:16]}"
    lock_stream = _with_lock(destination.with_suffix(".lock"))
    try:
        if destination.exists():
            return _verify_gsdltp(destination)
        source = (
            _require_regular_file(
                archive.expanduser().resolve(),
                size=asset.size_bytes,
                sha256=asset.sha256,
            )
            if archive is not None
            else _download_asset(asset, cache_dir, no_network=no_network)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".gsdltp-", dir=destination.parent))
        try:
            with tarfile.open(source, mode="r:gz") as bundle:
                members = bundle.getmembers()
                total = 0
                by_name = {member.name: member for member in members}
                for member in members:
                    path = PurePosixPath(member.name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or not (member.isfile() or member.isdir())
                        or member.issym()
                        or member.islnk()
                        or member.isdev()
                    ):
                        raise HydrationError(f"unsafe GSDLTP archive member: {member.name!r}")
                    if member.isfile():
                        total += member.size
                        if total > 1024 * 1024 * 1024:
                            raise HydrationError("GSDLTP expanded size exceeds 1 GiB")
                if not set(GSDLTP_ARCHIVE_FILES) <= set(by_name):
                    raise HydrationError("GSDLTP archive is missing the selected charlm models")
                for source_name, (output_name, size, digest) in GSDLTP_ARCHIVE_FILES.items():
                    member = by_name[source_name]
                    if member.size != size:
                        raise HydrationError(f"GSDLTP member {source_name} has the wrong size")
                    input_stream = bundle.extractfile(member)
                    if input_stream is None:
                        raise HydrationError(f"could not open GSDLTP member {source_name}")
                    output = temporary / output_name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with input_stream, output.open("xb") as destination_stream:
                        shutil.copyfileobj(input_stream, destination_stream)
                    _require_regular_file(output, size=size, sha256=digest)
            _write_json(
                temporary / ".metc-hydration.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "source": asset.url,
                    "archive_sha256": asset.sha256,
                },
            )
            _verify_gsdltp(temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return _verify_gsdltp(destination)
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def _load_stanza_manifest(software_root: Path) -> dict[str, Any]:
    path = software_root / "runtime-locks" / "stanza-models.json"
    _require_regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HydrationError("Stanza model manifest is invalid JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "metc-stanza-models-v1"
        or value.get("stanza_version") != "1.10.1"
        or value.get("resources_version") != "1.10.0"
        or not isinstance(value.get("files"), dict)
    ):
        raise HydrationError("Stanza model manifest does not match the paper runtime")
    return value


def _verify_stanza(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    exact_inventory: bool = True,
) -> Path:
    files = manifest["files"]
    expected = set(files) | {"resources.json"}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    if not expected <= actual or (exact_inventory and actual != expected):
        raise HydrationError(
            f"Stanza inventory differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    resources = manifest["resources_json"]
    _require_regular_file(
        root / "resources.json",
        size=resources["size_bytes"],
        sha256=resources["sha256"],
    )
    for relative, metadata in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(metadata, list)
            or len(metadata) != 2
            or type(metadata[0]) is not int
            or not isinstance(metadata[1], str)
        ):
            raise HydrationError("Stanza model manifest contains a malformed file entry")
        _require_regular_file(root / relative, size=metadata[0], sha256=metadata[1])
    return root


def _copy_stanza_inventory(
    source: Path,
    destination: Path,
    manifest: Mapping[str, Any],
) -> None:
    selected = set(manifest["files"]) | {"resources.json"}
    for relative in sorted(selected):
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, output)


def hydrate_stanza(
    *,
    software_root: Path,
    cache_dir: Path,
    jp_python: Path,
    no_network: bool,
    seed: Path | None = None,
) -> Path:
    manifest = _load_stanza_manifest(software_root)
    key = _json_digest(manifest)
    destination = cache_dir / "models" / f"stanza-{key[:16]}"
    lock_stream = _with_lock(destination.with_suffix(".lock"))
    try:
        if destination.exists():
            return _verify_stanza(destination, manifest)
        if seed is None and no_network:
            raise HydrationError("Stanza models are not cached and --no-network was supplied")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".stanza-", dir=destination.parent))
        try:
            if seed is not None:
                source = _verify_stanza(
                    seed.expanduser().resolve(),
                    manifest,
                    exact_inventory=False,
                )
            else:
                downloader = software_root / "runtime-tools" / "download_stanza.py"
                _require_regular_file(downloader)
                downloaded = temporary / ".download"
                _run_checked(
                    (jp_python, downloader, "--output", downloaded),
                    cwd=software_root,
                    timeout=4 * 60 * 60,
                )
                source = _verify_stanza(
                    downloaded,
                    manifest,
                    exact_inventory=False,
                )
            _copy_stanza_inventory(source, temporary, manifest)
            if seed is None:
                shutil.rmtree(downloaded)
            _write_json(
                temporary / ".metc-hydration.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "manifest_sha256": key,
                },
            )
            _verify_stanza(temporary, manifest)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return _verify_stanza(destination, manifest)
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def _verify_gerrant_resources(root: Path) -> Path:
    actual = {
        path.name for path in root.iterdir() if path.is_file() and not path.name.startswith(".")
    }
    if actual != set(GERRANT_RESOURCES):
        raise HydrationError(
            f"gERRANT resource inventory differs: "
            f"missing={sorted(set(GERRANT_RESOURCES) - actual)}, "
            f"extra={sorted(actual - set(GERRANT_RESOURCES))}"
        )
    for filename, (size, digest) in GERRANT_RESOURCES.items():
        _require_regular_file(root / filename, size=size, sha256=digest)
    return root


def hydrate_gerrant_root(
    *,
    software_root: Path,
    cache_dir: Path,
    gerrant_python: Path,
    no_network: bool,
    resource_seed: Path | None = None,
    data_root: Path | None = None,
) -> Path:
    bundled = software_root / "baselines" / "gerrant_de"
    manifest = json.loads(
        _require_regular_file(bundled / "manifest.json").read_text(encoding="utf-8")
    )
    key = _json_digest(
        {
            "manifest": manifest,
            "resources": GERRANT_RESOURCES,
        }
    )
    destination = cache_dir / "baselines" / f"gerrant-{key[:16]}"
    lock_stream = _with_lock(destination.with_suffix(".lock"))
    try:
        if destination.exists():
            validate_gerrant_root(destination)
            return destination
        if resource_seed is None and no_network:
            raise HydrationError(
                "gERRANT resources are not cached and --no-network was supplied; "
                "the complete Data artifact provides the exact dictionary but not "
                "the separately sourced STTS mapping, so pass --gerrant-resources "
                "with both exact derived files for an uncached offline run"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".gerrant-", dir=destination.parent))
        try:
            shutil.copytree(
                bundled,
                temporary,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("resources", "__pycache__", "*.pyc"),
            )
            resources = temporary / "resources"
            resources.mkdir()
            if resource_seed is not None:
                seed = _verify_gerrant_resources(resource_seed.expanduser().resolve())
                for filename in sorted(GERRANT_RESOURCES):
                    shutil.copyfile(seed / filename, resources / filename)
            else:
                dictionary = None
                if data_root is not None:
                    data_resources = data_root.expanduser().resolve() / "resources" / "gerrant"
                    if data_resources.exists():
                        dictionary = _require_regular_file(
                            data_resources / "de_DE-large.txt",
                            size=GERRANT_RESOURCES["de_DE-large.txt"][0],
                            sha256=GERRANT_RESOURCES["de_DE-large.txt"][1],
                        )
                if dictionary is None:
                    missing_tools = [
                        executable
                        for executable in ("make", "unmunch")
                        if shutil.which(executable) is None
                    ]
                    if missing_tools:
                        raise HydrationError(
                            "gERRANT resource construction requires system tools "
                            + ", ".join(missing_tools)
                            + "; install Hunspell (which provides unmunch), use the "
                            "complete Data artifact, or pass --gerrant-resources with "
                            "the two verified derived files"
                        )
                command: list[str | Path] = [
                    gerrant_python,
                    temporary / "prepare_resources.py",
                    "--output-dir",
                    resources,
                ]
                if dictionary is not None:
                    command.extend(("--dictionary", dictionary))
                _run_checked(
                    command,
                    cwd=temporary,
                    timeout=60 * 60,
                )
            _verify_gerrant_resources(resources)
            validate_gerrant_root(temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        validate_gerrant_root(destination)
        return destination
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def _verify_ltp_cache(root: Path) -> Path:
    snapshot = root / "hub" / "models--LTP--small" / "snapshots" / LTP_REVISION
    if not snapshot.is_dir():
        raise HydrationError(f"pinned LTP/small snapshot {LTP_REVISION} is missing under {root}")
    required = ("config.json",)
    for filename in required:
        candidate = snapshot / filename
        if not candidate.exists():
            raise HydrationError(f"LTP/small snapshot is missing {filename}")
    marker = _require_regular_file(root / ".metc-hydration.json")
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value != {
        "schema_version": SCHEMA_VERSION,
        "repository": LTP_REPOSITORY,
        "revision": LTP_REVISION,
    }:
        raise HydrationError("LTP/small hydration marker differs")
    return root


def hydrate_ltp(
    *,
    cache_dir: Path,
    cherrant_python: Path,
    no_network: bool,
    accept_terms: bool = False,
) -> Path:
    if not accept_terms:
        raise HydrationError(
            "LTP/small has nonstandard upstream terms. Review "
            f"{LTP_TERMS_URL} and rerun with --accept-ltp-terms only if your "
            "use is permitted."
        )
    destination = cache_dir / "models" / f"ltp-small-{LTP_REVISION[:16]}"
    lock_stream = _with_lock(destination.with_suffix(".lock"))
    try:
        if destination.exists():
            return _verify_ltp_cache(destination)
        if no_network:
            raise HydrationError("LTP/small is not cached and --no-network was supplied")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".ltp-small-", dir=destination.parent))
        try:
            environment = _clean_environment({"HF_HOME": str(temporary)})
            _run_checked(
                (
                    cherrant_python,
                    "-c",
                    f"from ltp import LTP;LTP('{LTP_REPOSITORY}@{LTP_REVISION}');print('hydrated')",
                ),
                cwd=destination.parent,
                environment=environment,
                timeout=2 * 60 * 60,
            )
            _write_json(
                temporary / ".metc-hydration.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "repository": LTP_REPOSITORY,
                    "revision": LTP_REVISION,
                },
            )
            _verify_ltp_cache(temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return _verify_ltp_cache(destination)
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def hydrate_selected(
    *,
    software_root: Path,
    cache_dir: Path,
    uv: str | Path,
    systems: Sequence[str],
    no_network: bool = False,
    gerrant_resources: Path | None = None,
    stanza_resources: Path | None = None,
    gsdltp_archive: Path | None = None,
    data_root: Path | None = None,
    accept_ltp_terms: bool = False,
) -> HydratedRuntimes:
    software_root = software_root.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    uv_path = Path(uv).expanduser()
    if uv_path.parent == Path("."):
        resolved = shutil.which(str(uv_path))
        if resolved is None:
            raise HydrationError("uv is required to construct the pinned runtimes")
        uv_path = Path(resolved)
    if not uv_path.is_file() or not os.access(uv_path, os.X_OK):
        raise HydrationError(f"uv is not executable: {uv_path}")

    requested = set(systems)
    unknown = requested - {"metc", "metc_teacher", "errant", "ger", "jp"}
    if unknown:
        raise HydrationError(f"unknown selected systems: {sorted(unknown)}")

    errant_python = None
    if "errant" in requested:
        errant_python = hydrate_runtime(
            "errant",
            software_root=software_root,
            cache_dir=cache_dir,
            uv=uv_path,
            no_network=no_network,
        )

    gerrant_python = None
    gerrant_root = None
    if "ger" in requested:
        gerrant_python = hydrate_runtime(
            "gerrant",
            software_root=software_root,
            cache_dir=cache_dir,
            uv=uv_path,
            no_network=no_network,
        )
        gerrant_root = hydrate_gerrant_root(
            software_root=software_root,
            cache_dir=cache_dir,
            gerrant_python=gerrant_python,
            no_network=no_network,
            resource_seed=gerrant_resources,
            data_root=data_root,
        )

    jp_python = None
    jp_manifest = None
    stanza_root = None
    gsdltp_root = None
    if "jp" in requested:
        jp_source, jp_manifest = hydrate_jp_source(
            software_root=software_root,
            cache_dir=cache_dir,
            no_network=no_network,
        )
        jp_python = hydrate_runtime(
            "jp_errant",
            software_root=software_root,
            cache_dir=cache_dir,
            uv=uv_path,
            no_network=no_network,
            jp_source=jp_source,
        )
        stanza_root = hydrate_stanza(
            software_root=software_root,
            cache_dir=cache_dir,
            jp_python=jp_python,
            no_network=no_network,
            seed=stanza_resources,
        )
        gsdltp_root = hydrate_gsdltp(
            cache_dir=cache_dir,
            no_network=no_network,
            archive=gsdltp_archive,
        )

    # ChERRANT reconstructs the Chinese gold reference in every E2E run,
    # independent of which annotation systems the user selects.
    cherrant_python = hydrate_runtime(
        "cherrant",
        software_root=software_root,
        cache_dir=cache_dir,
        uv=uv_path,
        no_network=no_network,
    )
    ltp_root = hydrate_ltp(
        cache_dir=cache_dir,
        cherrant_python=cherrant_python,
        no_network=no_network,
        accept_terms=accept_ltp_terms,
    )
    return HydratedRuntimes(
        errant_python=errant_python,
        gerrant_python=gerrant_python,
        gerrant_root=gerrant_root,
        jp_python=jp_python,
        jp_transform_manifest=jp_manifest,
        jp_stanza_resources=stanza_root,
        jp_zh_gsdltp_dir=gsdltp_root,
        cherrant_python=cherrant_python,
        cherrant_hf_home=ltp_root,
    )


def hydrate_all(
    *,
    software_root: Path,
    cache_dir: Path,
    uv: str | Path,
    no_network: bool = False,
    gerrant_resources: Path | None = None,
    stanza_resources: Path | None = None,
    gsdltp_archive: Path | None = None,
    data_root: Path | None = None,
    accept_ltp_terms: bool = False,
) -> HydratedRuntimes:
    """Hydrate every paper annotation producer and reference constructor."""

    return hydrate_selected(
        software_root=software_root,
        cache_dir=cache_dir,
        uv=uv,
        systems=("metc", "metc_teacher", "errant", "ger", "jp"),
        no_network=no_network,
        gerrant_resources=gerrant_resources,
        stanza_resources=stanza_resources,
        gsdltp_archive=gsdltp_archive,
        data_root=data_root,
        accept_ltp_terms=accept_ltp_terms,
    )
