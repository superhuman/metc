#!/usr/bin/env python3
"""Apply the paper's hash-gated JP-ERRANT compatibility transformations.

The upstream checkout is not redistributed.  This script accepts only the
three exact upstream files evaluated in the paper, changes fixed line ranges,
and then verifies the exact transformed bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Transformation:
    relative_path: str
    upstream_sha256: str
    transformed_sha256: str
    start_line: int
    end_line: int
    replacement: tuple[str, ...]


TRANSFORMATIONS = (
    Transformation(
        relative_path="jp_errant/__init__.py",
        upstream_sha256=(
            "2b1001022c66f3ed716e31190915f58a67901d6f1767c95913a8b65cd63c0a2e"
        ),
        transformed_sha256=(
            "d6a57e200b096b44ca48201674482624d1dc503e2dd2dc0ab66c0ed37adaef4d"
        ),
        start_line=23,
        end_line=35,
        replacement=(
            "            # GSDLTP (GSD treebank + LTP-style segmentation) Stanza model,\n",
            "            # Zenodo 10.5281/zenodo.15096936.\n",
            "            import os\n",
            '            model_root = os.environ.get("ZH_GSDLTP_DIR")\n',
            "            if not model_root:\n",
            '                raise RuntimeError("ZH_GSDLTP_DIR is required for Chinese")\n',
            "            nlp = stanza.Pipeline(\n",
            '                lang="zh-hans",\n',
            '                processors="tokenize,pos",\n',
            '                tokenize_model_path=f"{model_root}/tokenize/zh-hans_gsdsimp_tokenizer.pt",\n',
            '                pos_model_path=f"{model_root}/pos/zh-hans_gsdsimp_charlm_tagger.pt",\n',
            "                verbose=False,\n",
            "            )\n",
        ),
    ),
    Transformation(
        relative_path="jp_errant/commands/parallel_to_m2.py",
        upstream_sha256=(
            "446436358a81f720618dde207212ccf7c4f427254c2018fbe97b577a8a1d62c2"
        ),
        transformed_sha256=(
            "fe01a6db7ba0de0216207e957ba57ad6d2101f057e3ea478a4b13146a82e642c"
        ),
        start_line=13,
        end_line=13,
        replacement=('    annotator = jp_errant.load(args.lang, legacy=True)\n',),
    ),
    Transformation(
        relative_path="jp_errant/zh/char_similar.py",
        upstream_sha256=(
            "87758ac2b31f29024cd1bf8078ddd72cd1fdf2dc67d756caf92cf9996ddd4b3a"
        ),
        transformed_sha256=(
            "94c761ef599e8cc1ae31a5ea707b1ebb8142bd790d8a43d5d565596498700a1b"
        ),
        start_line=18,
        end_line=18,
        replacement=(
            "    if os.path.exists(path):\n",
            "        df = pd.read_csv(path)\n",
            "    else:\n",
            "        # The glyph-shape table is on a gated Drive link. An empty table\n",
            "        # preserves the public phonetic path without inventing private data.\n",
            "        df = pd.DataFrame(\n",
            "            columns=['CharA', 'CharB', 'similarity']\n",
            "        )\n",
        ),
    ),
)


class TransformationError(RuntimeError):
    """The checkout is not the pinned source or the output is not exact."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_regular_file(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise TransformationError(f"required upstream file is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TransformationError(f"upstream path is not a regular file: {path}")
    return path.read_bytes()


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_transformations(source_root: Path) -> None:
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise TransformationError(f"source root is not a directory: {root}")

    planned: list[tuple[Path, bytes, str]] = []
    for transformation in TRANSFORMATIONS:
        path = root.joinpath(*transformation.relative_path.split("/"))
        upstream = _require_regular_file(path)
        observed = _sha256(upstream)
        if observed != transformation.upstream_sha256:
            raise TransformationError(
                f"{transformation.relative_path} has upstream SHA-256 {observed}; "
                f"expected {transformation.upstream_sha256}"
            )
        lines = upstream.splitlines(keepends=True)
        start = transformation.start_line - 1
        end = transformation.end_line
        if start < 0 or end > len(lines) or start >= end:
            raise TransformationError(
                f"invalid line range for {transformation.relative_path}"
            )
        transformed = b"".join(
            (
                *lines[:start],
                *(line.encode("utf-8") for line in transformation.replacement),
                *lines[end:],
            )
        )
        transformed_digest = _sha256(transformed)
        if transformed_digest != transformation.transformed_sha256:
            raise TransformationError(
                f"{transformation.relative_path} transforms to SHA-256 "
                f"{transformed_digest}; expected {transformation.transformed_sha256}"
            )
        planned.append((path, transformed, transformation.transformed_sha256))

    for path, transformed, _ in planned:
        _atomic_write(path, transformed)
    for path, _, expected in planned:
        observed = _sha256(_require_regular_file(path))
        if observed != expected:
            raise TransformationError(
                f"written file has SHA-256 {observed}; expected {expected}: {path}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    apply_transformations(args.source_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TransformationError as error:
        raise SystemExit(f"JP-ERRANT TRANSFORM FAILED: {error}") from error
