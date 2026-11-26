"""German gERRANT adapter with exact historical source-first materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from metc_artifact.schema import DatasetSpec, dataset_spec

from .contract import AnnotationRecord, PairRecord, read_pairs, write_annotations
from .errors import AdapterOutputError, AdapterUnavailableError
from .m2 import align_m2_by_position, normalized_m2_edits, parse_m2, source_key
from .mappings import map_gerrant_de
from .process import require_directory, require_executable, require_file, run_checked


EXPECTED_COMMIT = "92a86d0722536f4b2c6a4a10037b64f3955ad1b5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_gerrant_root(root: Path | str) -> tuple[Path, Path]:
    directory = require_directory(root, "bundled gERRANT root")
    manifest_path = require_file(directory / "manifest.json", "gERRANT manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdapterUnavailableError("gERRANT manifest is invalid JSON") from error
    if manifest.get("upstream", {}).get("commit") != EXPECTED_COMMIT:
        raise AdapterUnavailableError("gERRANT manifest has the wrong upstream commit")
    code_files = manifest.get("code_files")
    if not isinstance(code_files, dict) or "gerrant_de.py" not in code_files:
        raise AdapterUnavailableError("gERRANT manifest has no pinned entrypoint hash")
    for filename, expected in code_files.items():
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise AdapterUnavailableError("gERRANT manifest has malformed code hashes")
        path = require_file(directory / filename, f"gERRANT code file {filename}")
        if _sha256(path) != expected:
            raise AdapterUnavailableError(f"gERRANT code hash mismatch: {filename}")
    resources = manifest.get("resources", {})
    for filename, key in (
        ("de-stts_map", "de-stts_map"),
        ("de_DE-large.txt", "de_DE-large.txt"),
    ):
        path = require_file(directory / "resources" / filename, f"gERRANT resource {filename}")
        try:
            expected = resources[key]["derived_sha256"]
        except (KeyError, TypeError) as error:
            raise AdapterUnavailableError(
                f"gERRANT manifest has no hash for resource {filename}"
            ) from error
        if not isinstance(expected, str):
            raise AdapterUnavailableError(
                f"gERRANT manifest has malformed hash for resource {filename}"
            )
        if _sha256(path) != expected:
            raise AdapterUnavailableError(f"gERRANT resource hash mismatch: {filename}")
    return directory, require_file(directory / "gerrant_de.py", "gERRANT entrypoint")


def apply_gerrant_paper_compat(
    pairs: Sequence[PairRecord],
    raw_records: Sequence[AnnotationRecord],
) -> tuple[AnnotationRecord, ...]:
    """Reproduce the paper's historical source-only first-hit join explicitly.

    Raw fresh output remains available to the caller.  The compatibility view
    deliberately selects the lowest numeric benchmark ID for duplicate German
    source strings; this reproduces the one known collision (IDs 786 and 916)
    independently of JSONL row order.
    """

    if len(pairs) != len(raw_records):
        raise AdapterOutputError(
            f"gERRANT raw record count {len(raw_records)} differs from pair count "
            f"{len(pairs)}"
        )
    pair_by_id = {pair.item_id: pair for pair in pairs}
    record_by_id = {record.item_id: record for record in raw_records}
    if len(pair_by_id) != len(pairs) or len(record_by_id) != len(raw_records):
        raise AdapterOutputError("gERRANT paper compatibility requires unique item IDs")
    if set(pair_by_id) != set(record_by_id):
        raise AdapterOutputError("gERRANT raw records do not match the pair item IDs")

    def numeric_id(item_id: str) -> int:
        value = json.loads(item_id)
        if isinstance(value, bool) or not isinstance(value, int):
            raise AdapterOutputError(
                "gERRANT paper compatibility requires numeric German benchmark IDs"
            )
        return value

    first_by_source: dict[str, AnnotationRecord] = {}
    for item_id in sorted(pair_by_id, key=numeric_id):
        first_by_source.setdefault(
            source_key(pair_by_id[item_id].source), record_by_id[item_id]
        )

    output: list[AnnotationRecord] = []
    for pair in pairs:
        record = AnnotationRecord(
            dataset_id=record_by_id[pair.item_id].dataset_id,
            item_id=pair.item_id,
            system="ger",
            span_unit=record_by_id[pair.item_id].span_unit,
            edits=first_by_source[source_key(pair.source)].edits,
        )
        record.validate_for_pair(pair, dataset_spec(pair.dataset_id))
        output.append(record)
    return tuple(output)


def run_gerrant(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    python: str | Path,
    gerrant_root: str | Path,
    environment: Mapping[str, str] | None = None,
    validate_provenance: bool = True,
) -> tuple[AnnotationRecord, ...]:
    if spec.dataset_id != "de_merlin":
        raise AdapterOutputError(f"gERRANT does not apply to {spec.dataset_id}")
    executable = require_executable(python, "gERRANT isolated Python executable")
    if validate_provenance:
        root, entrypoint = validate_gerrant_root(gerrant_root)
    else:
        root = require_directory(gerrant_root, "gERRANT root")
        entrypoint = require_file(root / "gerrant_de.py", "gERRANT entrypoint")
    for pair in pairs:
        pair.validate_for(spec)
    if not pairs:
        return ()
    with tempfile.TemporaryDirectory(prefix="gerrant-adapter-") as temporary:
        working = Path(temporary)
        source = working / "source.txt"
        correction = working / "correction.txt"
        output = working / "gerrant.m2"
        source.write_text("".join(pair.source + "\n" for pair in pairs), encoding="utf-8")
        correction.write_text(
            "".join(pair.correction + "\n" for pair in pairs), encoding="utf-8"
        )
        run_checked(
            (
                executable,
                entrypoint,
                "-orig",
                source,
                "-cor",
                correction,
                "-out",
                output,
            ),
            cwd=root,
            environment=environment,
        )
        sentences = parse_m2(output)
    materialized = align_m2_by_position(pairs, sentences)
    records: list[AnnotationRecord] = []
    for pair, sentence in zip(pairs, materialized, strict=True):
        record = AnnotationRecord(
            dataset_id=spec.dataset_id,
            item_id=pair.item_id,
            system="ger",
            span_unit=spec.evaluation_span_unit,
            edits=normalized_m2_edits(
                pair,
                sentence,
                span_unit=spec.evaluation_span_unit,
                label_mapper=map_gerrant_de,
                tool_separator=" ",
            ),
        )
        record.validate_for_pair(pair, spec)
        records.append(record)
    return tuple(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="de")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=None)
    parser.add_argument("--gerrant-root", type=Path, default=None)
    parser.add_argument(
        "--paper-compat",
        action="store_true",
        help="reproduce the historical source-only join used by the paper",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="required raw-fresh sidecar when --paper-compat is selected",
    )
    args = parser.parse_args(argv)
    if args.paper_compat:
        if args.raw_output is None:
            parser.error("--paper-compat requires --raw-output so fresh output is retained")
        if args.output.resolve() == args.raw_output.resolve():
            parser.error("--output and --raw-output must be different files")
    elif args.raw_output is not None:
        parser.error("--raw-output is only used with --paper-compat")

    spec = dataset_spec(args.dataset)
    pairs = read_pairs(args.pairs, spec)
    raw_records = run_gerrant(
        pairs,
        spec,
        python=args.python or os.environ.get("GERRANT_PYTHON"),
        gerrant_root=args.gerrant_root or os.environ.get("GERRANT_ROOT"),
    )
    if args.paper_compat:
        write_annotations(args.raw_output, raw_records)
        records = apply_gerrant_paper_compat(pairs, raw_records)
    else:
        records = raw_records
    write_annotations(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
