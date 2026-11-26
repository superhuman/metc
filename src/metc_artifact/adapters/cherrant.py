"""MuCGEC ChERRANT gold-construction adapter in an isolated LTP environment."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from metc_artifact.schema import DatasetSpec, dataset_spec

from .contract import AnnotationRecord, PairRecord, read_pairs, write_annotations
from .errors import AdapterInputError, AdapterOutputError, AdapterUnavailableError
from .m2 import align_m2_by_position, normalized_m2_edits, parse_m2
from .mappings import map_cherrant_gold
from .process import require_directory, require_executable, require_file, run_checked


EXPECTED_COMMIT = "39221934ee4ccf07801879136a75dd7e11319977"
EXPECTED_LTP_REVISION = "dea87bf0da3c5d054bbd426a78d7fa9efd347799"
EXPECTED_PATCHES = {
    "lazy_ltp_import": True,
    "ltp_4_pipeline_api": True,
    "ltp_small": True,
}


def validate_patch_manifest(path: Path | str | None) -> None:
    manifest_path = require_file(path, "ChERRANT paper-patch manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdapterUnavailableError("ChERRANT patch manifest is invalid JSON") from error
    if payload.get("upstream_commit") != EXPECTED_COMMIT:
        raise AdapterUnavailableError("ChERRANT patch manifest has the wrong commit")
    if payload.get("ltp_model_revision") != EXPECTED_LTP_REVISION:
        raise AdapterUnavailableError("ChERRANT LTP model revision is not pinned")
    if payload.get("patches") != EXPECTED_PATCHES:
        raise AdapterUnavailableError("ChERRANT paper patches are incomplete")


def run_cherrant_reference(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    python: str | Path,
    entrypoint: str | Path,
    cherrant_root: str | Path,
    patch_manifest: str | Path,
    huggingface_cache: str | Path,
    environment: Mapping[str, str] | None = None,
) -> tuple[AnnotationRecord, ...]:
    if spec.dataset_id != "zh_mucgec":
        raise AdapterOutputError(f"ChERRANT reference does not apply to {spec.dataset_id}")
    executable = require_executable(python, "ChERRANT isolated Python executable")
    script = require_file(entrypoint, "patched ChERRANT parallel_to_m2.py")
    root = require_directory(cherrant_root, "pinned ChERRANT source root")
    cache = require_directory(huggingface_cache, "Hugging Face cache with pinned LTP/small")
    validate_patch_manifest(patch_manifest)
    for pair in pairs:
        pair.validate_for(spec)
        if any(character in pair.source + pair.correction for character in ("\t", "\n", "\r")):
            raise AdapterInputError(
                f"{pair.item_id}: ChERRANT parallel input cannot contain tabs or newlines"
            )
    if not pairs:
        return ()
    with tempfile.TemporaryDirectory(prefix="cherrant-adapter-") as temporary:
        working = Path(temporary)
        parallel = working / "pairs.para"
        output = working / "cherrant.m2"
        parallel.write_text(
            "".join(
                f"{index}\t{pair.source}\t{pair.correction}\n"
                for index, pair in enumerate(pairs)
            ),
            encoding="utf-8",
        )
        child_environment = dict(environment or {})
        child_environment.update({"HF_HOME": str(cache), "HF_HUB_OFFLINE": "1"})
        run_checked(
            (
                executable,
                script,
                "--file",
                parallel,
                "--output",
                output,
                "--granularity",
                "word",
            ),
            cwd=root,
            environment=child_environment,
        )
        sentences = align_m2_by_position(pairs, parse_m2(output))
    records: list[AnnotationRecord] = []
    for pair, sentence in zip(pairs, sentences, strict=True):
        record = AnnotationRecord(
            dataset_id=spec.dataset_id,
            item_id=pair.item_id,
            system="gold",
            span_unit=spec.evaluation_span_unit,
            edits=normalized_m2_edits(
                pair,
                sentence,
                span_unit=spec.evaluation_span_unit,
                label_mapper=map_cherrant_gold,
                tool_separator="",
            ),
        )
        record.validate_for_pair(pair, spec)
        records.append(record)
    return tuple(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="zh")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=None)
    parser.add_argument("--entrypoint", type=Path, default=None)
    parser.add_argument("--cherrant-root", type=Path, default=None)
    parser.add_argument("--patch-manifest", type=Path, default=None)
    parser.add_argument("--huggingface-cache", type=Path, default=None)
    args = parser.parse_args(argv)
    spec = dataset_spec(args.dataset)
    records = run_cherrant_reference(
        read_pairs(args.pairs, spec),
        spec,
        python=args.python or os.environ.get("CHERRANT_PYTHON"),
        entrypoint=args.entrypoint or os.environ.get("CHERRANT_ENTRYPOINT"),
        cherrant_root=args.cherrant_root or os.environ.get("CHERRANT_ROOT"),
        patch_manifest=args.patch_manifest or os.environ.get("CHERRANT_PATCH_MANIFEST"),
        huggingface_cache=args.huggingface_cache
        or os.environ.get("CHERRANT_HF_CACHE"),
    )
    write_annotations(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
