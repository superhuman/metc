"""JP-ERRANT adapter for all paper languages with transformation attestation."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from metc_artifact.schema import DatasetSpec, dataset_spec

from .contract import AnnotationRecord, PairRecord, read_pairs, write_annotations
from .errors import AdapterOutputError, AdapterUnavailableError
from .m2 import M2Sentence, align_m2_by_position, normalized_m2_edits, parse_m2, source_key
from .mappings import map_jp
from .process import require_directory, require_executable, require_file, run_checked


EXPECTED_COMMIT = "839ac8641b15272c2fcb6bc74015e88dda4a8842"
EXPECTED_TRANSFORMATIONS = {
    "parallel_legacy_true": True,
    "zh_explicit_gsdltp_models": True,
    "zh_empty_shape_table_fallback": True,
}
DEFAULT_BULK_WORKER = Path(__file__).with_name("_jp_errant_worker.py")
DEFAULT_SHARDED_DRIVER = Path(__file__).with_name("_jp_errant_sharded_driver.py")


@dataclass(frozen=True)
class JPErrantViews:
    raw: tuple[AnnotationRecord, ...]
    paper_compat: tuple[AnnotationRecord, ...]


def validate_transform_manifest(path: Path | str | None) -> None:
    manifest_path = require_file(path, "JP-ERRANT transformation manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AdapterUnavailableError(
            "JP-ERRANT transformation manifest is invalid JSON"
        ) from error
    if payload.get("upstream_commit") != EXPECTED_COMMIT:
        raise AdapterUnavailableError(
            "JP-ERRANT transformation manifest has the wrong commit"
        )
    if payload.get("transformations") != EXPECTED_TRANSFORMATIONS:
        raise AdapterUnavailableError("JP-ERRANT transformations are incomplete")


def _language_code(spec: DatasetSpec) -> str:
    return "en" if spec.experiment_code == "en1" else spec.experiment_code


def _normalize_sentences(
    pairs: Sequence[PairRecord],
    sentences: Sequence[M2Sentence],
    spec: DatasetSpec,
) -> tuple[AnnotationRecord, ...]:
    separator = "" if spec.dataset_id == "zh_mucgec" else " "
    records: list[AnnotationRecord] = []
    for pair, sentence in zip(pairs, sentences, strict=True):
        record = AnnotationRecord(
            dataset_id=spec.dataset_id,
            item_id=pair.item_id,
            system="jp",
            span_unit=spec.evaluation_span_unit,
            edits=normalized_m2_edits(
                pair,
                sentence,
                span_unit=spec.evaluation_span_unit,
                label_mapper=lambda label, dataset=spec.dataset_id: map_jp(dataset, label),
                tool_separator=separator,
                insertion_inside_token_uses_containing_boundary=(
                    spec.dataset_id == "uk_uagec"
                ),
            ),
        )
        record.validate_for_pair(pair, spec)
        records.append(record)
    return tuple(records)


def _paper_compat_sentences(
    pairs: Sequence[PairRecord],
    raw_sentences: Sequence[M2Sentence],
    spec: DatasetSpec,
) -> tuple[M2Sentence, ...]:
    """Reproduce the historical DE/CS source-first, fuzzy-fallback rebuild."""

    if spec.dataset_id not in {"de_merlin", "cs_merlin"}:
        return tuple(raw_sentences)

    aligned = align_m2_by_position(
        pairs,
        raw_sentences,
        allow_parser_normalization=True,
    )

    def numeric_id(pair: PairRecord) -> int:
        value = json.loads(pair.item_id)
        if isinstance(value, bool) or not isinstance(value, int):
            raise AdapterOutputError(
                "JP-ERRANT paper compatibility requires numeric MERLIN item IDs"
            )
        return value

    ordered = sorted(zip(pairs, aligned, strict=True), key=lambda item: numeric_id(item[0]))
    by_source: dict[str, M2Sentence] = {}
    source_keys: list[str] = []
    raw_by_id: dict[str, M2Sentence] = {}
    for pair, sentence in ordered:
        key = source_key(" ".join(sentence.tokens))
        by_source.setdefault(key, sentence)
        source_keys.append(key)
        raw_by_id[pair.item_id] = sentence

    output: list[M2Sentence] = []
    for pair in pairs:
        key = source_key(pair.source)
        sentence = by_source.get(key)
        if sentence is None:
            matches = difflib.get_close_matches(key, source_keys, n=1, cutoff=0.9)
            sentence = by_source[matches[0]] if matches else raw_by_id[pair.item_id]
        output.append(sentence)
    return tuple(output)


def _paper_compat_records(
    pairs: Sequence[PairRecord],
    raw_sentences: Sequence[M2Sentence],
    spec: DatasetSpec,
) -> tuple[AnnotationRecord, ...]:
    records = list(
        _normalize_sentences(
            pairs,
            _paper_compat_sentences(pairs, raw_sentences, spec),
            spec,
        )
    )
    if spec.dataset_id not in {"ko_kolla", "uk_uagec"}:
        return tuple(records)

    for index, (pair, record) in enumerate(zip(pairs, records, strict=True)):
        if spec.dataset_id == "ko_kolla" and pair.item_id == "823":
            # Fresh Stanza output correctly targets token 3.  The paper data
            # retains one older Korean boundary at [4, 5).
            expected_source = ("그래서", "Ph", ".D.를", "원하요", ".")
            expected_correction = ("그래서", "Ph", ".D.를", "원해요", ".")
            if (
                pair.source_tokens != expected_source
                or pair.correction_tokens != expected_correction
            ):
                raise AdapterOutputError(
                    "Korean JP-ERRANT legacy projection item 823 no longer matches "
                    "the frozen benchmark text"
                )
            matching = [
                edit_index
                for edit_index, edit in enumerate(record.edits)
                if (edit.start, edit.end, edit.replacement) == (3, 4, "원해요")
            ]
            if len(matching) != 1:
                raise AdapterOutputError(
                    "Korean JP-ERRANT legacy projection expected one [3, 4) edit "
                    "for item 823"
                )
            edits = list(record.edits)
            edit_index = matching[0]
            edits[edit_index] = replace(edits[edit_index], start=4, end=5)
            records[index] = replace(record, edits=tuple(edits))
            continue

        if spec.dataset_id != "uk_uagec" or pair.item_id != "1112":
            continue
        source_tokens = pair.source_tokens or ()
        correction_tokens = pair.correction_tokens or ()
        if source_tokens[75:78] != ("зайвим", "роботі", "розуму") or correction_tokens[
            75:78
        ] != ("зайвим,", "роботі", "розуму"):
            raise AdapterOutputError(
                "Ukrainian JP-ERRANT legacy projection item 1112 no longer matches "
                "the frozen benchmark text"
            )
        matching = [
            edit_index
            for edit_index, edit in enumerate(record.edits)
            if (
                edit.start,
                edit.end,
                edit.replacement,
                edit.native_label,
            )
            == (76, 76, ",", "M:PUNCT")
        ]
        if len(matching) != 1:
            raise AdapterOutputError(
                "Ukrainian JP-ERRANT legacy projection expected one [76, 76) "
                "punctuation edit for item 1112"
            )
        edits = list(record.edits)
        edit_index = matching[0]
        edits[edit_index] = replace(edits[edit_index], start=77, end=77)
        records[index] = replace(record, edits=tuple(edits))
    for pair, record in zip(pairs, records, strict=True):
        record.validate_for_pair(pair, spec)
    return tuple(records)


def _run_jp_errant_sentences(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    command_prefix: Sequence[str | Path],
    patch_manifest: str | Path,
    stanza_resources: str | Path,
    zh_gsdltp_dir: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[M2Sentence, ...]:
    """Run the external producer and return only its position-aligned raw M2."""

    if not command_prefix:
        raise AdapterUnavailableError("missing JP-ERRANT command")
    executable = require_executable(command_prefix[0], "JP-ERRANT executable")
    prefix = (executable, *(str(value) for value in command_prefix[1:]))
    validate_transform_manifest(patch_manifest)
    stanza_root = require_directory(stanza_resources, "JP-ERRANT Stanza resources")
    gsdltp = None
    if spec.dataset_id == "zh_mucgec":
        gsdltp = require_directory(zh_gsdltp_dir, "JP-ERRANT Chinese GSDLTP models")
    for pair in pairs:
        pair.validate_for(spec)
    if not pairs:
        return ()
    with tempfile.TemporaryDirectory(prefix="jp-errant-adapter-") as temporary:
        working = Path(temporary)
        source = working / "source.txt"
        correction = working / "correction.txt"
        output = working / "jp-errant.m2"
        source.write_text("".join(pair.source + "\n" for pair in pairs), encoding="utf-8")
        correction.write_text(
            "".join(pair.correction + "\n" for pair in pairs), encoding="utf-8"
        )
        child_environment = dict(environment or {})
        child_environment["STANZA_RESOURCES_DIR"] = str(stanza_root)
        if gsdltp is not None:
            child_environment["ZH_GSDLTP_DIR"] = str(gsdltp)
        run_checked(
            (
                *prefix,
                "-orig",
                source,
                "-cor",
                correction,
                "-out",
                output,
                "-lang",
                _language_code(spec),
            ),
            cwd=working,
            environment=child_environment,
        )
        return align_m2_by_position(
            pairs,
            parse_m2(output),
            allow_parser_normalization=True,
        )


def run_jp_errant_views(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    command_prefix: Sequence[str | Path],
    patch_manifest: str | Path,
    stanza_resources: str | Path,
    zh_gsdltp_dir: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> JPErrantViews:
    """Return both raw fresh annotations and the explicit paper-compatible view."""
    raw_sentences = _run_jp_errant_sentences(
        pairs,
        spec,
        command_prefix=command_prefix,
        patch_manifest=patch_manifest,
        stanza_resources=stanza_resources,
        zh_gsdltp_dir=zh_gsdltp_dir,
        environment=environment,
    )
    return JPErrantViews(
        raw=_normalize_sentences(pairs, raw_sentences, spec),
        paper_compat=_paper_compat_records(pairs, raw_sentences, spec),
    )


def run_jp_errant(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    command_prefix: Sequence[str | Path],
    patch_manifest: str | Path,
    stanza_resources: str | Path,
    zh_gsdltp_dir: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[AnnotationRecord, ...]:
    """Return raw fresh JP-ERRANT annotations without historical projection."""
    raw_sentences = _run_jp_errant_sentences(
        pairs,
        spec,
        command_prefix=command_prefix,
        patch_manifest=patch_manifest,
        stanza_resources=stanza_resources,
        zh_gsdltp_dir=zh_gsdltp_dir,
        environment=environment,
    )
    return _normalize_sentences(pairs, raw_sentences, spec)


def run_jp_errant_bulk(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    python: str | Path,
    patch_manifest: str | Path,
    stanza_resources: str | Path,
    zh_gsdltp_dir: str | Path | None = None,
    worker: str | Path = DEFAULT_BULK_WORKER,
    environment: Mapping[str, str] | None = None,
    allow_nonpaper: bool = False,
) -> tuple[AnnotationRecord, ...]:
    """Run diagnostic Stanza bulk mode, which is not paper-exact on German."""

    if not allow_nonpaper:
        raise AdapterOutputError(
            "Stanza bulk_process changes JP-ERRANT output on German; pass "
            "allow_nonpaper=True only for diagnostic use"
        )
    worker_path = require_file(worker, "JP-ERRANT batched worker")
    return run_jp_errant(
        pairs,
        spec,
        command_prefix=(python, worker_path, "--mode", "bulk"),
        patch_manifest=patch_manifest,
        stanza_resources=stanza_resources,
        zh_gsdltp_dir=zh_gsdltp_dir,
        environment=environment,
    )


def run_jp_errant_bulk_views(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    python: str | Path,
    patch_manifest: str | Path,
    stanza_resources: str | Path,
    zh_gsdltp_dir: str | Path | None = None,
    worker: str | Path = DEFAULT_BULK_WORKER,
    environment: Mapping[str, str] | None = None,
    allow_nonpaper: bool = False,
) -> JPErrantViews:
    if not allow_nonpaper:
        raise AdapterOutputError(
            "Stanza bulk_process changes JP-ERRANT output on German; pass "
            "allow_nonpaper=True only for diagnostic use"
        )
    worker_path = require_file(worker, "JP-ERRANT batched worker")
    return run_jp_errant_views(
        pairs,
        spec,
        command_prefix=(python, worker_path, "--mode", "bulk"),
        patch_manifest=patch_manifest,
        stanza_resources=stanza_resources,
        zh_gsdltp_dir=zh_gsdltp_dir,
        environment=environment,
    )


def run_jp_errant_sharded_views(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    python: str | Path,
    patch_manifest: str | Path,
    stanza_resources: str | Path,
    zh_gsdltp_dir: str | Path | None = None,
    workers: int = 2,
    worker: str | Path = DEFAULT_BULK_WORKER,
    driver: str | Path = DEFAULT_SHARDED_DRIVER,
    environment: Mapping[str, str] | None = None,
) -> JPErrantViews:
    """Shard exact one-document-at-a-time inference across child processes."""

    if workers < 1:
        raise AdapterOutputError("JP-ERRANT shard count must be positive")
    worker_path = require_file(worker, "JP-ERRANT sequential worker")
    driver_path = require_file(driver, "JP-ERRANT sharded driver")
    return run_jp_errant_views(
        pairs,
        spec,
        command_prefix=(
            python,
            driver_path,
            "--worker",
            worker_path,
            "--workers",
            str(workers),
        ),
        patch_manifest=patch_manifest,
        stanza_resources=stanza_resources,
        zh_gsdltp_dir=zh_gsdltp_dir,
        environment=environment,
    )


def run_jp_errant_sharded(
    pairs: Sequence[PairRecord],
    spec: DatasetSpec,
    *,
    python: str | Path,
    patch_manifest: str | Path,
    stanza_resources: str | Path,
    zh_gsdltp_dir: str | Path | None = None,
    workers: int = 2,
    worker: str | Path = DEFAULT_BULK_WORKER,
    driver: str | Path = DEFAULT_SHARDED_DRIVER,
    environment: Mapping[str, str] | None = None,
) -> tuple[AnnotationRecord, ...]:
    if workers < 1:
        raise AdapterOutputError("JP-ERRANT shard count must be positive")
    worker_path = require_file(worker, "JP-ERRANT sequential worker")
    driver_path = require_file(driver, "JP-ERRANT sharded driver")
    return run_jp_errant(
        pairs,
        spec,
        command_prefix=(
            python,
            driver_path,
            "--worker",
            worker_path,
            "--workers",
            str(workers),
        ),
        patch_manifest=patch_manifest,
        stanza_resources=stanza_resources,
        zh_gsdltp_dir=zh_gsdltp_dir,
        environment=environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=None)
    parser.add_argument("--worker", type=Path, default=DEFAULT_BULK_WORKER)
    parser.add_argument("--driver", type=Path, default=DEFAULT_SHARDED_DRIVER)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--bulk-nonpaper",
        action="store_true",
        help="diagnostic only; Stanza bulk mode is not exact on German",
    )
    parser.add_argument("--executable", default=None)
    parser.add_argument(
        "--transform-manifest",
        "--patch-manifest",
        dest="patch_manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--stanza-resources", type=Path, default=None)
    parser.add_argument("--zh-gsdltp-dir", type=Path, default=None)
    parser.add_argument("--paper-compat", action="store_true")
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="required raw-fresh sidecar when --paper-compat is selected",
    )
    args = parser.parse_args(argv)
    if args.paper_compat:
        if args.raw_output is None:
            parser.error("--paper-compat requires --raw-output so fresh output is retained")
        if args.bulk_nonpaper:
            parser.error("--bulk-nonpaper cannot be combined with --paper-compat")
        if args.output.resolve() == args.raw_output.resolve():
            parser.error("--output and --raw-output must be different files")
    elif args.raw_output is not None:
        parser.error("--raw-output is only used with --paper-compat")

    spec = dataset_spec(args.dataset)
    pairs = read_pairs(args.pairs, spec)
    common = {
        "patch_manifest": args.patch_manifest
        or os.environ.get("JP_ERRANT_TRANSFORM_MANIFEST")
        or os.environ.get("JP_ERRANT_PATCH_MANIFEST"),
        "stanza_resources": args.stanza_resources
        or os.environ.get("JP_ERRANT_STANZA_RESOURCES"),
        "zh_gsdltp_dir": args.zh_gsdltp_dir or os.environ.get("ZH_GSDLTP_DIR"),
    }
    python = args.python or os.environ.get("JP_ERRANT_PYTHON")
    executable = args.executable or os.environ.get("JP_ERRANT_PARALLEL")
    if python and executable:
        parser.error("choose either --python or --executable, not both")
    if args.bulk_nonpaper and not python:
        parser.error("--bulk-nonpaper requires --python")

    if python:
        if args.bulk_nonpaper:
            records = run_jp_errant_bulk(
                pairs,
                spec,
                python=python,
                worker=args.worker,
                allow_nonpaper=True,
                **common,
            )
        elif args.paper_compat:
            views = run_jp_errant_sharded_views(
                pairs,
                spec,
                python=python,
                worker=args.worker,
                driver=args.driver,
                workers=args.workers,
                **common,
            )
            write_annotations(args.raw_output, views.raw)
            records = views.paper_compat
        else:
            records = run_jp_errant_sharded(
                pairs,
                spec,
                python=python,
                worker=args.worker,
                driver=args.driver,
                workers=args.workers,
                **common,
            )
    else:
        command_prefix = (executable,) if executable else ()
        if args.paper_compat:
            views = run_jp_errant_views(pairs, spec, command_prefix=command_prefix, **common)
            write_annotations(args.raw_output, views.raw)
            records = views.paper_compat
        else:
            records = run_jp_errant(pairs, spec, command_prefix=command_prefix, **common)
    write_annotations(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
