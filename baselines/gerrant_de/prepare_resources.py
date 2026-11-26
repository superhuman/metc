#!/usr/bin/env python3
"""Fetch and reconstruct the two non-bundled German ERRANT resources."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence
from urllib.request import Request, urlopen


IGERMAN_URL = "https://www.j3e.de/ispell/igerman98/dict/igerman98-20161207.tar.bz2"
IGERMAN_SIZE = 470_207
IGERMAN_SHA256 = "17296f03c5fea62d76ecc530ebe80f6adc430278f58d472dc1842d71612960a8"
DICTIONARY_SHA256 = "577c06ebd6461c0a43486a9a538c1175fbc1f0d1a3985e9b4350a5ad1c5d9da6"
DICTIONARY_SIZE = 4_778_232

STTS_MAP_SHA256 = "840a7d37c2364b3a5f628900a58c4cc513014d47de6d3a96414fa43c9231c869"
STTS_MAP_ROWS = 56
STTS_TO_UNIVERSAL = (
    ("ADJA", "ADJ"),
    ("ADJD", "ADJ"),
    ("ADV", "ADV"),
    ("APPR", "ADP"),
    ("APPRART", "ADP"),
    ("APPO", "ADP"),
    ("APZR", "ADP"),
    ("ART", "DET"),
    ("CARD", "NUM"),
    ("FM", "NOUN"),
    ("ITJ", "INTJ"),
    ("KOUI", "SCONJ"),
    ("KOUS", "SCONJ"),
    ("KON", "CONJ"),
    ("KOKOM", "ADP"),
    ("NN", "NOUN"),
    ("NNE", "PNOUN"),
    ("NE", "PNOUN"),
    ("PDS", "PRON"),
    ("PDAT", "DET"),
    ("PIS", "PRON"),
    ("PIAT", "DET"),
    ("PIDAT", "ADJ"),
    ("PROAV", "ADV"),
    ("PPER", "PRON"),
    ("PPOSS", "PRON"),
    ("PPOSAT", "DET"),
    ("PRELS", "PRON"),
    ("PRELAT", "DET"),
    ("PRF", "PRON"),
    ("PWS", "PRON"),
    ("PWAT", "DET"),
    ("PWAV", "PRON"),
    ("PROP", "ADV"),
    ("PTKZU", "PART"),
    ("PTKNEG", "PART"),
    ("PTKVZ", "ADP"),
    ("PTKANT", "INTJ"),
    ("PTKA", "ADV"),
    ("TRUNC", "*"),
    ("VVFIN", "VERB"),
    ("VVIMP", "VERB"),
    ("VVINF", "VERB"),
    ("VVIZU", "VERB"),
    ("VVPP", "VERB"),
    ("VAFIN", "AUX"),
    ("VAIMP", "AUX"),
    ("VAINF", "AUX"),
    ("VAPP", "AUX"),
    ("VMFIN", "AUX"),
    ("VMINF", "AUX"),
    ("VMPP", "AUX"),
    ("XY", "X"),
    ("$,", "PUNCT"),
    ("$.", "PUNCT"),
    ("$(", "PUNCT"),
)

MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_EXPANDED_BYTES = 128 * 1024 * 1024
USER_AGENT = "metc-artifact-resource-preparer/1.0"


class ResourceError(RuntimeError):
    """Raised when a source or derived resource differs from the pinned input."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_bytes(path: Path, *, expected_size: int, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    if len(data) != expected_size:
        raise ResourceError(f"{path}: expected {expected_size} bytes, found {len(data)}")
    observed = sha256_bytes(data)
    if observed != expected_sha256:
        raise ResourceError(f"{path}: expected SHA-256 {expected_sha256}, found {observed}")
    return data


def _download(url: str, destination: Path, *, expected_size: int, expected_sha256: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned HTTPS URL and digest
            data = response.read(expected_size + 1)
    except OSError as error:
        raise ResourceError(f"could not download {url}: {error}") from error
    if len(data) != expected_size:
        raise ResourceError(f"{url}: expected {expected_size} bytes, downloaded {len(data)}")
    observed = sha256_bytes(data)
    if observed != expected_sha256:
        raise ResourceError(f"{url}: expected SHA-256 {expected_sha256}, found {observed}")
    destination.write_bytes(data)
    return data


def _obtain_source(
    supplied: Path | None,
    *,
    url: str,
    temporary_path: Path,
    expected_size: int,
    expected_sha256: str,
) -> Path:
    if supplied is not None:
        _checked_bytes(
            supplied,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        return supplied
    _download(
        url,
        temporary_path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    return temporary_path


def render_stts_map() -> bytes:
    """Render the reviewed functional STTS-to-universal compatibility table."""

    tags = [tag for tag, _ in STTS_TO_UNIVERSAL]
    if len(STTS_TO_UNIVERSAL) != STTS_MAP_ROWS or len(tags) != len(set(tags)):
        raise ResourceError(
            "embedded STTS map must contain "
            f"{STTS_MAP_ROWS} unique rows, found {len(STTS_TO_UNIVERSAL)}"
        )
    rendered = "".join(f"{tag}\t{universal}\n" for tag, universal in STTS_TO_UNIVERSAL).encode(
        "utf-8"
    )
    observed = sha256_bytes(rendered)
    if observed != STTS_MAP_SHA256:
        raise ResourceError(
            f"embedded STTS map expected SHA-256 {STTS_MAP_SHA256}, found {observed}"
        )
    return rendered


def _safe_extract(archive_path: Path, destination: Path) -> Path:
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ResourceError(f"igerman98 archive has too many members: {len(members)}")
        expanded = 0
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != "igerman98-20161207"
                or not (member.isfile() or member.isdir())
            ):
                raise ResourceError(f"unsafe igerman98 archive member: {member.name!r}")
            if member.isfile():
                expanded += member.size
                if expanded > MAX_ARCHIVE_EXPANDED_BYTES:
                    raise ResourceError("igerman98 archive exceeds the expanded-size limit")
        archive.extractall(destination, members=members)  # noqa: S202 - members validated above
    root = destination / "igerman98-20161207"
    if not root.is_dir():
        raise ResourceError("igerman98 archive did not contain its expected root directory")
    return root


def render_dictionary(archive_path: Path, temporary_root: Path) -> bytes:
    """Run Boyd's documented igerman98 expansion and verify its exact output."""

    source_root = _safe_extract(archive_path, temporary_root)
    if shutil.which("make") is None:
        raise ResourceError("the 'make' executable is required to build igerman98")
    if shutil.which("unmunch") is None:
        raise ResourceError("Hunspell's 'unmunch' executable is required")
    subprocess.run(
        ["make", "hunspell/de_DE.aff", "hunspell/de_DE.dic"],
        cwd=source_root,
        check=True,
    )
    completed = subprocess.run(
        ["unmunch", "hunspell/de_DE.dic", "hunspell/de_DE.aff"],
        cwd=source_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    text = completed.stdout.decode("iso-8859-1")
    lines = [
        line for line in text.splitlines() if not line.startswith(("-", "\t")) and "/" not in line
    ]
    rendered = ("\n".join(lines) + "\n").encode("utf-8")
    observed = sha256_bytes(rendered)
    if observed != DICTIONARY_SHA256:
        raise ResourceError(
            f"derived dictionary expected SHA-256 {DICTIONARY_SHA256}, found {observed}"
        )
    return rendered


def prepare_resources(
    output_dir: Path,
    *,
    igerman_archive: Path | None,
    dictionary: Path | None,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (output_dir / "de-stts_map", output_dir / "de_DE-large.txt")
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not force:
        raise ResourceError(
            "refusing to overwrite existing resources without --force: " + ", ".join(existing)
        )

    with tempfile.TemporaryDirectory(prefix="gerrant-resources-") as temporary:
        temp = Path(temporary)
        stts_data = render_stts_map()
        if dictionary is not None:
            dictionary_data = _checked_bytes(
                dictionary,
                expected_size=DICTIONARY_SIZE,
                expected_sha256=DICTIONARY_SHA256,
            )
        else:
            archive_path = _obtain_source(
                igerman_archive,
                url=IGERMAN_URL,
                temporary_path=temp / "igerman98-20161207.tar.bz2",
                expected_size=IGERMAN_SIZE,
                expected_sha256=IGERMAN_SHA256,
            )
            dictionary_data = render_dictionary(archive_path, temp / "expanded")

    for path, data in zip(outputs, (stts_data, dictionary_data), strict=True):
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_bytes(data)
        temporary_path.replace(path)
        print(f"wrote {path} ({len(data)} bytes, sha256={sha256_bytes(data)})")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("resources"))
    dictionary_source = parser.add_mutually_exclusive_group()
    dictionary_source.add_argument("--igerman-archive", type=Path)
    dictionary_source.add_argument(
        "--dictionary",
        type=Path,
        help="prebuilt de_DE-large.txt; exact size and SHA-256 are required",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare_resources(
            args.output_dir,
            igerman_archive=args.igerman_archive,
            dictionary=args.dictionary,
            force=args.force,
        )
    except (ResourceError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"resource preparation failed: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
