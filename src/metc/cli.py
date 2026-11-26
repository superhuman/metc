"""Command-line interface for offline METC inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

from .model import DEFAULT_MAX_SEQUENCE_LENGTH, INFERENCE_PROFILES, TokenClassifier
from .pipeline import METC
from .tokenizer_preparation import prepare_tokenizer


DEFAULT_BATCH_SIZE = 8


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="local checkpoint directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--inference-profile",
        choices=INFERENCE_PROFILES,
        default="portable",
        help=(
            "portable runs on available hardware; paper requires the evaluated "
            "CUDA bfloat16 + FlashAttention path"
        ),
    )
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metc", description="Offline multilingual error type classification"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify", help="classify source/target edits")
    _add_model_arguments(classify)
    classify.add_argument("--source")
    classify.add_argument("--target")
    classify.add_argument("--input", help="JSONL input path, or - for stdin")
    classify.add_argument("--output", default="-", help="JSONL output path")
    classify.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"maximum JSONL rows processed at once (default: {DEFAULT_BATCH_SIZE})",
    )
    classify.add_argument("--tokenized", action="store_true")
    classify.add_argument("--cleanup-semantic", action="store_true")

    label = subparsers.add_parser("label", help="classify pre-marked text")
    _add_model_arguments(label)
    label.add_argument("--annotated", required=True)

    inspect = subparsers.add_parser("inspect-checkpoint", help="validate and describe a checkpoint")
    _add_model_arguments(inspect)

    prepare = subparsers.add_parser(
        "prepare-tokenizer",
        help="prepare the exact tokenizer from the pinned upstream mmBERT snapshot",
    )
    prepare.add_argument("--model", required=True, help="extracted checkpoint directory")
    prepare.add_argument(
        "--source-dir",
        help=(
            "optional directory containing the four pinned upstream files; "
            "omit to download them from Hugging Face"
        ),
    )
    return parser


def _open_input(path: str) -> tuple[TextIO, bool]:
    if path == "-":
        return sys.stdin, False
    return Path(path).open("r", encoding="utf-8"), True


def _paths_refer_to_same_file(input_path: str, output_path: str) -> bool:
    """Return whether two non-stdio paths identify the same file."""

    if input_path == "-" or output_path == "-":
        return False
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    try:
        if source.samefile(destination):
            return True
    except OSError:
        # One path may not exist yet. Resolving still catches lexical aliases
        # such as ``data/../pairs.jsonl`` before the output is opened.
        pass
    return source.resolve(strict=False) == destination.resolve(strict=False)


def _iter_jsonl(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    handle, should_close = _open_input(path)
    try:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            if not isinstance(row.get("source"), str) or not isinstance(row.get("target"), str):
                raise ValueError(f"line {line_number} requires string source and target fields")
            for flag in ("tokenized", "cleanup_semantic"):
                if flag in row and not isinstance(row[flag], bool):
                    raise ValueError(f"line {line_number} field {flag} must be a JSON boolean")
            yield line_number, row
    finally:
        if should_close:
            handle.close()


def _write_jsonl(rows: Iterable[dict[str, Any]], path: str) -> None:
    handle = sys.stdout if path == "-" else Path(path).open("w", encoding="utf-8")
    try:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        if handle is not sys.stdout:
            handle.close()


def _chunks(
    rows: Iterable[tuple[int, dict[str, Any]]], batch_size: int
) -> Iterator[list[tuple[int, dict[str, Any]]]]:
    chunk: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _classify_jsonl_rows(
    pipeline: METC,
    rows: Iterable[tuple[int, dict[str, Any]]],
    *,
    batch_size: int,
    default_tokenized: bool,
    default_cleanup_semantic: bool,
) -> Iterator[dict[str, Any]]:
    for chunk in _chunks(rows, batch_size):
        grouped: dict[tuple[bool, bool], list[tuple[int, int, tuple[str, str]]]] = defaultdict(list)
        for output_index, (line_number, row) in enumerate(chunk):
            tokenized = row.get("tokenized", default_tokenized)
            cleanup = row.get("cleanup_semantic", default_cleanup_semantic)
            grouped[(tokenized, cleanup)].append(
                (output_index, line_number, (row["source"], row["target"]))
            )

        outputs: list[dict[str, Any] | None] = [None] * len(chunk)
        for (tokenized, cleanup), indexed_pairs in grouped.items():
            input_lines = [line_number for _, line_number, _ in indexed_pairs]
            try:
                results = pipeline.classify_batch(
                    [pair for _, _, pair in indexed_pairs],
                    tokenized=tokenized,
                    cleanup_semantic=cleanup,
                )
            except Exception as error:
                lines = ", ".join(str(line_number) for line_number in input_lines)
                label = "line" if len(input_lines) == 1 else "lines"
                raise RuntimeError(
                    f"classification failed for input {label} {lines}: {error}"
                ) from error
            if len(results) != len(indexed_pairs):
                lines = ", ".join(str(line_number) for line_number in input_lines)
                raise RuntimeError(
                    "classification returned an unexpected number of results for "
                    f"input lines {lines}"
                )
            for (output_index, _, _), result in zip(indexed_pairs, results, strict=True):
                outputs[output_index] = result.to_dict()

        for output in outputs:
            if output is None:  # pragma: no cover - guarded by the result-count check above
                raise RuntimeError("classification did not produce a result for an input row")
            yield output


def _make_pipeline(args: argparse.Namespace) -> METC:
    return METC(
        args.model,
        max_sequence_length=args.max_sequence_length,
        device=args.device,
        inference_profile=args.inference_profile,
    )


def _classify(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    direct = args.source is not None or args.target is not None
    if bool(args.input) == direct:
        parser.error("provide either --input or both --source and --target")
    if direct and (args.source is None or args.target is None):
        parser.error("--source and --target must be supplied together")

    if direct:
        pipeline = _make_pipeline(args)
        result = pipeline.classify(
            args.source,
            args.target,
            tokenized=args.tokenized,
            cleanup_semantic=args.cleanup_semantic,
        )
        _write_jsonl([result.to_dict()], args.output)
        return 0

    if _paths_refer_to_same_file(args.input, args.output):
        parser.error("--input and --output must refer to different files")

    pipeline = _make_pipeline(args)
    _write_jsonl(
        _classify_jsonl_rows(
            pipeline,
            _iter_jsonl(args.input),
            batch_size=args.batch_size,
            default_tokenized=args.tokenized,
            default_cleanup_semantic=args.cleanup_semantic,
        ),
        args.output,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "classify":
        return _classify(args, parser)
    if args.command == "prepare-tokenizer":
        payload = prepare_tokenizer(args.model, source_dir=args.source_dir)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    classifier = TokenClassifier(
        args.model,
        max_sequence_length=args.max_sequence_length,
        device=args.device,
        inference_profile=args.inference_profile,
    )
    if args.command == "label":
        payload = {
            "predictions": [
                prediction.to_dict() for prediction in classifier.predict_marked(args.annotated)
            ]
        }
    else:
        payload = classifier.inspect_checkpoint()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
